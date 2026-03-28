/*************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net_telemetry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>

/* Global telemetry state */
int rcclTelemetryEnabled = 0;
RcclTelemetryConfig rcclTelemetryCfg;
RcclDeviceStats rcclTelemetryDevs[RCCL_TELEMETRY_MAX_DEVS];
int rcclTelemetryNumDevs = 0;

/* Internal state */
static int rcclTelemetryInitialized = 0;
static char rcclTelemetryStartTime[64];
static char rcclTelemetryProcessName[256];

/* ------------------------------------------------------------------ */
/* Multi-hardware counter descriptor table                            */
/* ------------------------------------------------------------------ */

enum RcclHwcSource {
  HWC_NONE = 0,
  HWC_IB_SYSFS,
  HWC_ETHTOOL,
  HWC_DEBUGFS
};

typedef struct {
  enum RcclHwcSource source;
  const char*        key;
} RcclHwcDriverMapping;

/* Driver column indices */
enum {
  RCCL_DRV_IONIC = 0,
  RCCL_DRV_BNXT_RE,
  RCCL_DRV_MLX5,
  RCCL_DRV_FALLBACK,
  RCCL_DRV_COUNT
};

typedef struct {
  const char*         json_name;
  int                 fw_dependent;
  RcclHwcDriverMapping drivers[RCCL_DRV_COUNT];
} RcclHwCounterDesc;

/*
 * Scalar hw counter descriptor table.
 * Order must match RCCL_HWC_* enum in net_telemetry.h.
 * Driver column order: IONIC, BNXT_RE, MLX5, FALLBACK
 */
static const RcclHwCounterDesc rcclHwcTable[RCCL_HWC_COUNT] = {
  /* ============================================================ */
  /* Shared / cross-driver counters                               */
  /* ============================================================ */
  /* [RCCL_HWC_RX_CNP_PKTS] */
  { "rx_cnp_pkts", 0, {
    { HWC_IB_SYSFS, "rx_rdma_cnp_pkts" },   /* ionic */
    { HWC_IB_SYSFS, "np_cnp_sent" },        /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_IB_SYSFS, "rx_cnp_pkts" },        /* fallback */
  }},
  /* [RCCL_HWC_TX_CNP_PKTS] */
  { "tx_cnp_pkts", 0, {
    { HWC_IB_SYSFS, "tx_rdma_cnp_pkts" },   /* ionic */
    { HWC_IB_SYSFS, "rp_cnp_handled" },     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_IB_SYSFS, "tx_cnp_pkts" },        /* fallback */
  }},
  /* [RCCL_HWC_RX_ROCE_DISCARDS] */
  { "rx_roce_discards", 0, {
    { HWC_IB_SYSFS, "rx_rdma_mtu_discard_pkts" }, /* ionic */
    { HWC_IB_SYSFS, "rx_roce_discards" },  /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_IB_SYSFS, "rx_roce_discards" },  /* fallback */
  }},
  /* [RCCL_HWC_PFC_RX_FRAMES_TOTAL] */
  { "pfc_rx_frames_total", 0, {
    { HWC_ETHTOOL, "frames_rx_pripause" },  /* ionic */
    { HWC_ETHTOOL, "pfc_pri3_rx_transitions" }, /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_PFC_TX_FRAMES_TOTAL] */
  { "pfc_tx_frames_total", 0, {
    { HWC_ETHTOOL, "frames_tx_pripause" },  /* ionic */
    { HWC_ETHTOOL, "pfc_pri3_tx_transitions" }, /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_HW_RX_DROPPED] */
  { "hw_rx_dropped", 0, {
    { HWC_ETHTOOL, "hw_rx_dropped" },       /* ionic */
    { HWC_ETHTOOL, "rx_stat_discard" },     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_HW_TX_DROPPED] */
  { "hw_tx_dropped", 0, {
    { HWC_ETHTOOL, "hw_tx_dropped" },       /* ionic */
    { HWC_IB_SYSFS, "tx_roce_discards" },   /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RX_ERRORS] */
  { "rx_errors", 0, {
    { HWC_ETHTOOL, "hw_rx_over_errors" },   /* ionic */
    { HWC_IB_SYSFS, "rx_roce_errors" },     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_TO_RETRANSMITS] */
  { "to_retransmits", 0, {
    { HWC_IB_SYSFS, "tx_rdma_ack_timeout" }, /* ionic */
    { HWC_IB_SYSFS, "roce_adp_retrans" },   /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_MAX_RETRY_EXCEEDED] */
  { "max_retry_exceeded", 0, {
    { HWC_IB_SYSFS, "req_tx_retry_excd_err" }, /* ionic */
    { HWC_IB_SYSFS, "max_retry_exceeded" }, /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_OOS_DROP_COUNT] */
  { "oos_drop_count", 0, {
    { HWC_IB_SYSFS, "resp_rx_outouf_seq" }, /* ionic */
    { HWC_IB_SYSFS, "out_of_sequence" },    /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_SEQ_ERR_NAKS_RCVD] */
  { "seq_err_naks_rcvd", 0, {
    { HWC_IB_SYSFS, "req_rx_pkt_seq_err" }, /* ionic */
    { HWC_IB_SYSFS, "packet_seq_err" },     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* ============================================================ */
  /* RDMA traffic counters                                        */
  /* ============================================================ */
  /* [RCCL_HWC_TX_RDMA_RETX_PKTS] */
  { "tx_rdma_retx_pkts", 0, {
    { HWC_IB_SYSFS, "tx_rdma_retx_pkts" },  /* ionic */
    { HWC_IB_SYSFS, "roce_adp_retrans" },   /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_TX_RDMA_RETX_BYTES] */
  { "tx_rdma_retx_bytes", 0, {
    { HWC_IB_SYSFS, "tx_rdma_retx_bytes" }, /* ionic */
    { HWC_NONE, NULL },                     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_TX_RDMA_ACK_TIMEOUT] */
  { "tx_rdma_ack_timeout", 0, {
    { HWC_IB_SYSFS, "tx_rdma_ack_timeout" }, /* ionic */
    { HWC_IB_SYSFS, "roce_adp_retrans_to" }, /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RX_RDMA_ECN_PKTS] */
  { "rx_rdma_ecn_pkts", 0, {
    { HWC_IB_SYSFS, "rx_rdma_ecn_pkts" },   /* ionic */
    { HWC_IB_SYSFS, "np_ecn_marked_roce_packets" }, /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RX_RDMA_MTU_DISCARD_PKTS] */
  { "rx_rdma_mtu_discard_pkts", 0, {
    { HWC_IB_SYSFS, "rx_rdma_mtu_discard_pkts" }, /* ionic */
    { HWC_NONE, NULL },                     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* ============================================================ */
  /* Requester error counters (RX path)                           */
  /* ============================================================ */
  /* [RCCL_HWC_REQ_RX_PKT_SEQ_ERR] */
  { "req_rx_pkt_seq_err", 0, {
    { HWC_IB_SYSFS, "req_rx_pkt_seq_err" }, /* ionic */
    { HWC_IB_SYSFS, "packet_seq_err" },     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_REQ_RX_RNR_RETRY_ERR] */
  { "req_rx_rnr_retry_err", 0, {
    { HWC_IB_SYSFS, "req_rx_rnr_retry_err" }, /* ionic */
    { HWC_IB_SYSFS, "rnr_nak_retry_err" },  /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_REQ_RX_RMT_ACC_ERR] */
  { "req_rx_rmt_acc_err", 0, {
    { HWC_IB_SYSFS, "req_rx_rmt_acc_err" }, /* ionic */
    { HWC_IB_SYSFS, "req_remote_access_errors" }, /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_REQ_RX_CQE_ERR] */
  { "req_rx_cqe_err", 0, {
    { HWC_IB_SYSFS, "req_rx_cqe_err" },     /* ionic */
    { HWC_IB_SYSFS, "req_cqe_error" },      /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_REQ_RX_DUP_RESPONSE] */
  { "req_rx_dup_response", 0, {
    { HWC_IB_SYSFS, "req_rx_dup_response" }, /* ionic */
    { HWC_IB_SYSFS, "bad_resp_err" },       /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* ============================================================ */
  /* Requester error counters (TX path)                           */
  /* ============================================================ */
  /* [RCCL_HWC_REQ_TX_RETRY_EXCD_ERR] */
  { "req_tx_retry_excd_err", 0, {
    { HWC_IB_SYSFS, "req_tx_retry_excd_err" }, /* ionic */
    { HWC_IB_SYSFS, "max_retry_exceeded" }, /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_REQ_TX_LOC_OPER_ERR] */
  { "req_tx_loc_oper_err", 0, {
    { HWC_IB_SYSFS, "req_tx_loc_oper_err" }, /* ionic */
    { HWC_IB_SYSFS, "local_qp_op_err" },    /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* ============================================================ */
  /* Responder error counters (RX path)                           */
  /* ============================================================ */
  /* [RCCL_HWC_RESP_RX_DUP_REQUEST] */
  { "resp_rx_dup_request", 0, {
    { HWC_IB_SYSFS, "resp_rx_dup_request" }, /* ionic */
    { HWC_IB_SYSFS, "duplicate_request" },   /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RESP_RX_OUTOF_BUF] */
  { "resp_rx_outof_buf", 0, {
    { HWC_IB_SYSFS, "resp_rx_outof_buf" },  /* ionic */
    { HWC_IB_SYSFS, "out_of_buffer" },      /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RESP_RX_OUTOUF_SEQ] */
  { "resp_rx_outouf_seq", 0, {
    { HWC_IB_SYSFS, "resp_rx_outouf_seq" }, /* ionic */
    { HWC_IB_SYSFS, "out_of_sequence" },    /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RESP_RX_CQE_ERR] */
  { "resp_rx_cqe_err", 0, {
    { HWC_IB_SYSFS, "resp_rx_cqe_err" },    /* ionic */
    { HWC_IB_SYSFS, "resp_cqe_error" },     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* ============================================================ */
  /* Responder error counters (TX path)                           */
  /* ============================================================ */
  /* [RCCL_HWC_RESP_TX_RNR_RETRY_ERR] */
  { "resp_tx_rnr_retry_err", 0, {
    { HWC_IB_SYSFS, "resp_tx_rnr_retry_err" }, /* ionic */
    { HWC_NONE, NULL },                     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* ============================================================ */
  /* RDMA traffic — unicast/multicast                             */
  /* ============================================================ */
  /* [RCCL_HWC_TX_RDMA_UCAST_BYTES] */
  { "tx_rdma_ucast_bytes", 0, {
    { HWC_IB_SYSFS, "tx_rdma_ucast_bytes" }, /* ionic */
    { HWC_IB_SYSFS, "tx_bytes" },            /* bnxt_re (total, no ucast/mcast split) */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_TX_RDMA_UCAST_PKTS] */
  { "tx_rdma_ucast_pkts", 0, {
    { HWC_IB_SYSFS, "tx_rdma_ucast_pkts" }, /* ionic */
    { HWC_IB_SYSFS, "tx_pkts" },            /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_TX_RDMA_MCAST_BYTES] */
  { "tx_rdma_mcast_bytes", 0, {
    { HWC_IB_SYSFS, "tx_rdma_mcast_bytes" }, /* ionic */
    { HWC_NONE, NULL },                     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_TX_RDMA_MCAST_PKTS] */
  { "tx_rdma_mcast_pkts", 0, {
    { HWC_IB_SYSFS, "tx_rdma_mcast_pkts" }, /* ionic */
    { HWC_NONE, NULL },                     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RX_RDMA_UCAST_BYTES] */
  { "rx_rdma_ucast_bytes", 0, {
    { HWC_IB_SYSFS, "rx_rdma_ucast_bytes" }, /* ionic */
    { HWC_IB_SYSFS, "rx_bytes" },            /* bnxt_re (total, no ucast/mcast split) */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RX_RDMA_UCAST_PKTS] */
  { "rx_rdma_ucast_pkts", 0, {
    { HWC_IB_SYSFS, "rx_rdma_ucast_pkts" }, /* ionic */
    { HWC_IB_SYSFS, "rx_pkts" },            /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RX_RDMA_MCAST_BYTES] */
  { "rx_rdma_mcast_bytes", 0, {
    { HWC_IB_SYSFS, "rx_rdma_mcast_bytes" }, /* ionic */
    { HWC_NONE, NULL },                     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RX_RDMA_MCAST_PKTS] */
  { "rx_rdma_mcast_pkts", 0, {
    { HWC_IB_SYSFS, "rx_rdma_mcast_pkts" }, /* ionic */
    { HWC_NONE, NULL },                     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* ============================================================ */
  /* CCL/CTS traffic (ionic FW-dependent, no bnxt_re equivalent)  */
  /* ============================================================ */
  /* [RCCL_HWC_TX_RDMA_CCL_CTS_BYTES] */
  { "tx_rdma_ccl_cts_bytes", 1, {
    { HWC_IB_SYSFS, "tx_rdma_ccl_cts_bytes" }, /* ionic */
    { HWC_NONE, NULL }, { HWC_NONE, NULL }, { HWC_NONE, NULL },
  }},
  /* [RCCL_HWC_TX_RDMA_CCL_CTS_PKTS] */
  { "tx_rdma_ccl_cts_pkts", 1, {
    { HWC_IB_SYSFS, "tx_rdma_ccl_cts_pkts" }, /* ionic */
    { HWC_NONE, NULL }, { HWC_NONE, NULL }, { HWC_NONE, NULL },
  }},
  /* [RCCL_HWC_TX_RDMA_CCL_CTS_RETX_BYTES] */
  { "tx_rdma_ccl_cts_retx_bytes", 1, {
    { HWC_IB_SYSFS, "tx_rdma_ccl_cts_retx_bytes" }, /* ionic */
    { HWC_NONE, NULL }, { HWC_NONE, NULL }, { HWC_NONE, NULL },
  }},
  /* [RCCL_HWC_TX_RDMA_CCL_CTS_RETX_PKTS] */
  { "tx_rdma_ccl_cts_retx_pkts", 1, {
    { HWC_IB_SYSFS, "tx_rdma_ccl_cts_retx_pkts" }, /* ionic */
    { HWC_NONE, NULL }, { HWC_NONE, NULL }, { HWC_NONE, NULL },
  }},
  /* [RCCL_HWC_TX_RDMA_CCL_CTS_ACK_TIMEOUT] */
  { "tx_rdma_ccl_cts_ack_timeout", 1, {
    { HWC_IB_SYSFS, "tx_rdma_ccl_cts_ack_timeout" }, /* ionic */
    { HWC_NONE, NULL }, { HWC_NONE, NULL }, { HWC_NONE, NULL },
  }},
  /* [RCCL_HWC_RX_RDMA_CCL_CTS_BYTES] */
  { "rx_rdma_ccl_cts_bytes", 1, {
    { HWC_IB_SYSFS, "rx_rdma_ccl_cts_bytes" }, /* ionic */
    { HWC_NONE, NULL }, { HWC_NONE, NULL }, { HWC_NONE, NULL },
  }},
  /* [RCCL_HWC_RX_RDMA_CCL_CTS_PKTS] */
  { "rx_rdma_ccl_cts_pkts", 1, {
    { HWC_IB_SYSFS, "rx_rdma_ccl_cts_pkts" }, /* ionic */
    { HWC_NONE, NULL }, { HWC_NONE, NULL }, { HWC_NONE, NULL },
  }},
  /* ============================================================ */
  /* Requester errors — additional RX                             */
  /* ============================================================ */
  /* [RCCL_HWC_REQ_RX_RMT_REQ_ERR] */
  { "req_rx_rmt_req_err", 0, {
    { HWC_IB_SYSFS, "req_rx_rmt_req_err" }, /* ionic */
    { HWC_IB_SYSFS, "req_remote_invalid_request" }, /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_REQ_RX_OPER_ERR] */
  { "req_rx_oper_err", 0, {
    { HWC_IB_SYSFS, "req_rx_oper_err" },    /* ionic */
    { HWC_NONE, NULL },                     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_REQ_RX_IMPL_NAK_SEQ_ERR] */
  { "req_rx_impl_nak_seq_err", 0, {
    { HWC_IB_SYSFS, "req_rx_impl_nak_seq_err" }, /* ionic */
    { HWC_IB_SYSFS, "implied_nak_seq_err" }, /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_REQ_RX_CQE_FLUSH] */
  { "req_rx_cqe_flush", 0, {
    { HWC_IB_SYSFS, "req_rx_cqe_flush" },   /* ionic */
    { HWC_IB_SYSFS, "req_cqe_flush_error" }, /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_REQ_RX_INVAL_PKTS] */
  { "req_rx_inval_pkts", 0, {
    { HWC_IB_SYSFS, "req_rx_inval_pkts" },  /* ionic */
    { HWC_NONE, NULL },                     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* ============================================================ */
  /* Requester errors — additional TX                             */
  /* ============================================================ */
  /* [RCCL_HWC_REQ_TX_LOC_ACC_ERR] */
  { "req_tx_loc_acc_err", 0, {
    { HWC_IB_SYSFS, "req_tx_loc_acc_err" }, /* ionic */
    { HWC_IB_SYSFS, "local_protection_err" }, /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_REQ_TX_MEM_MGMT_ERR] */
  { "req_tx_mem_mgmt_err", 0, {
    { HWC_IB_SYSFS, "req_tx_mem_mgmt_err" }, /* ionic */
    { HWC_IB_SYSFS, "mem_mgmt_op_err" },    /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_REQ_TX_LOC_SGL_INV_ERR] (FW-dependent) */
  { "req_tx_loc_sgl_inv_err", 1, {
    { HWC_IB_SYSFS, "req_tx_loc_sgl_inv_err" }, /* ionic */
    { HWC_NONE, NULL },                     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* ============================================================ */
  /* Responder errors — additional RX                             */
  /* ============================================================ */
  /* [RCCL_HWC_RESP_RX_CQE_FLUSH] */
  { "resp_rx_cqe_flush", 0, {
    { HWC_IB_SYSFS, "resp_rx_cqe_flush" },  /* ionic */
    { HWC_IB_SYSFS, "resp_cqe_flush_error" }, /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RESP_RX_LOC_LEN_ERR] */
  { "resp_rx_loc_len_err", 0, {
    { HWC_IB_SYSFS, "resp_rx_loc_len_err" }, /* ionic */
    { HWC_IB_SYSFS, "resp_local_length_error" }, /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RESP_RX_INVAL_REQUEST] */
  { "resp_rx_inval_request", 0, {
    { HWC_IB_SYSFS, "resp_rx_inval_request" }, /* ionic */
    { HWC_NONE, NULL },                     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RESP_RX_LOC_OPER_ERR] */
  { "resp_rx_loc_oper_err", 0, {
    { HWC_IB_SYSFS, "resp_rx_loc_oper_err" }, /* ionic */
    { HWC_NONE, NULL },                     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RESP_RX_OUTOF_ATOMIC] */
  { "resp_rx_outof_atomic", 0, {
    { HWC_IB_SYSFS, "resp_rx_outof_atomic" }, /* ionic */
    { HWC_NONE, NULL },                     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RESP_RX_CCL_CTS_OUTOUF_SEQ] (FW-dependent) */
  { "resp_rx_ccl_cts_outouf_seq", 1, {
    { HWC_IB_SYSFS, "resp_rx_ccl_cts_outouf_seq" }, /* ionic */
    { HWC_NONE, NULL },                     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RESP_RX_S0_TABLE_ERR] (FW-dependent) */
  { "resp_rx_s0_table_err", 1, {
    { HWC_IB_SYSFS, "resp_rx_s0_table_err" }, /* ionic */
    { HWC_NONE, NULL },                     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* ============================================================ */
  /* Responder errors — additional TX                             */
  /* ============================================================ */
  /* [RCCL_HWC_RESP_TX_PKT_SEQ_ERR] */
  { "resp_tx_pkt_seq_err", 0, {
    { HWC_IB_SYSFS, "resp_tx_pkt_seq_err" }, /* ionic */
    { HWC_NONE, NULL },                     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RESP_TX_RMT_INVAL_REQ_ERR] */
  { "resp_tx_rmt_inval_req_err", 0, {
    { HWC_IB_SYSFS, "resp_tx_rmt_inval_req_err" }, /* ionic */
    { HWC_IB_SYSFS, "res_rem_inv_err" },    /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RESP_TX_RMT_ACC_ERR] */
  { "resp_tx_rmt_acc_err", 0, {
    { HWC_IB_SYSFS, "resp_tx_rmt_acc_err" }, /* ionic */
    { HWC_IB_SYSFS, "resp_remote_access_errors" }, /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RESP_TX_RMT_OPER_ERR] */
  { "resp_tx_rmt_oper_err", 0, {
    { HWC_IB_SYSFS, "resp_tx_rmt_oper_err" }, /* ionic */
    { HWC_IB_SYSFS, "remote_op_err" },      /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
  /* [RCCL_HWC_RESP_TX_LOC_SGL_INV_ERR] (FW-dependent) */
  { "resp_tx_loc_sgl_inv_err", 1, {
    { HWC_IB_SYSFS, "resp_tx_loc_sgl_inv_err" }, /* ionic */
    { HWC_NONE, NULL },                     /* bnxt_re */
    { HWC_NONE, NULL },                     /* mlx5 */
    { HWC_NONE, NULL },                     /* fallback */
  }},
};

/* ------------------------------------------------------------------ */
/* Per-priority PFC key patterns per driver                           */
/* ------------------------------------------------------------------ */

typedef struct {
  const char* rx_frames_fmt;
  const char* tx_frames_fmt;
  const char* rx_pause_us_fmt;
  const char* tx_pause_us_fmt;
} RcclPfcDriverPatterns;

static const RcclPfcDriverPatterns rcclPfcPatterns[RCCL_DRV_COUNT] = {
  /* ionic */    { "frames_rx_pri_%d",         "frames_tx_pri_%d",
                   "rx_pripause_%d_1us_count",  "tx_pripause_%d_1us_count" },
  /* bnxt_re */  { "rx_pfc_ena_frames_pri%d",  "tx_pfc_ena_frames_pri%d",
                   NULL, NULL },
  /* mlx5 */     { NULL, NULL, NULL, NULL },
  /* fallback */ { NULL, NULL, NULL, NULL },
};

/* ------------------------------------------------------------------ */
/* Per-driver ethtool key names for delta byte/packet counters         */
/* ------------------------------------------------------------------ */

typedef struct {
  const char* tx_bytes;
  const char* rx_bytes;
  const char* tx_packets;
  const char* rx_packets;
} RcclDeltaDriverPatterns;

static const RcclDeltaDriverPatterns rcclDeltaPatterns[RCCL_DRV_COUNT] = {
  /* ionic */    { "octets_tx_ok", "octets_rx_ok", "frames_tx_ok", "frames_rx_ok" },
  /* bnxt_re */  { "tx_bytes", "rx_bytes", "tx_total_frames", "rx_total_frames" },
  /* mlx5 */     { "tx_bytes", "rx_bytes", "tx_packets",      "rx_packets" },
  /* fallback */ { "tx_bytes", "rx_bytes", "tx_packets",      "rx_packets" },
};

/* ------------------------------------------------------------------ */
/* Forward declarations                                               */
/* ------------------------------------------------------------------ */

static void rcclTelemetryParseConfig(const char* config_path);
static void rcclTelemetryCollectHwCounters(RcclDeviceStats* dev);
static int64_t rcclTelemetryReadSysfsCounter(const char* path);
static void rcclTelemetryGetDriverName(const char* roce_device, char* driver_name, size_t size);
static int rcclTelemetryResolveDriverColumn(const char* driver_name);
static int rcclTelemetryIsCounterEnabled(const char* counter_name);
static void rcclTelemetryGetTimestamp(char* buf, size_t size);
static void rcclTelemetryWriteJson(FILE* fp);
static void rcclTelemetrySnapshotInit(RcclDeviceStats* dev);

/* ------------------------------------------------------------------ */
/* Unified batched ethtool reader                                     */
/* ------------------------------------------------------------------ */

#define RCCL_ETHTOOL_MAX_WANTED 80

typedef struct {
  char     key[64];
  int64_t* target;
} RcclEthtoolWantedEx;

static void rcclTelemetryCollectEthtoolBatch(const char* eth_device,
                                              RcclEthtoolWantedEx* wanted,
                                              int num_wanted) {
  if (eth_device[0] == '\0' || num_wanted == 0) {
    return;
  }

  char cmd[256];
  snprintf(cmd, sizeof(cmd), "ethtool -S %s 2>/dev/null", eth_device);

  FILE* fp = popen(cmd, "r");
  if (fp == NULL) {
    return;
  }

  int found = 0;
  char line[256];
  while (fgets(line, sizeof(line), fp) != NULL && found < num_wanted) {
    for (int i = 0; i < num_wanted; i++) {
      if (*(wanted[i].target) >= 0) continue;

      char* p = strstr(line, wanted[i].key);
      if (p == NULL) continue;

      /* Verify word boundary: char before key must be whitespace or start of line */
      if (p > line) {
        char prev = *(p - 1);
        if (prev != ' ' && prev != '\t') continue;
      }
      /* Verify word boundary: char after key must not be alphanumeric or underscore */
      size_t klen = strlen(wanted[i].key);
      char after = *(p + klen);
      if (after != ':' && after != ' ' && after != '\t' &&
          after != '\0' && after != '\n') continue;

      p += klen;
      while (*p == ' ' || *p == ':') p++;
      if (*p != '\0' && *p != '\n') {
        *(wanted[i].target) = strtoll(p, NULL, 10);
        found++;
      }
    }
  }

  pclose(fp);
}

/* ------------------------------------------------------------------ */
/* Init / Flush / Register                                            */
/* ------------------------------------------------------------------ */

void rcclTelemetryInit(void) {
  if (__atomic_exchange_n(&rcclTelemetryInitialized, 1, __ATOMIC_SEQ_CST)) {
    return;
  }

  const char* enable_env = getenv("RCCL_TELEMETRY_ENABLE");
  if (enable_env == NULL || strcmp(enable_env, "1") != 0) {
    rcclTelemetryEnabled = 0;
    return;
  }

  rcclTelemetryEnabled = 1;

  strncpy(rcclTelemetryCfg.output_dir, "/tmp", sizeof(rcclTelemetryCfg.output_dir) - 1);
  rcclTelemetryCfg.output_dir[sizeof(rcclTelemetryCfg.output_dir) - 1] = '\0';
  rcclTelemetryCfg.histogram_max_buckets = 5;
  rcclTelemetryCfg.histogram_bucket_interval_ns = 30000;
  rcclTelemetryCfg.hw_counter_list[0] = '\0';

  const char* config_path = getenv("RCCL_TELEMETRY_CONFIG");
  if (config_path != NULL && config_path[0] != '\0') {
    rcclTelemetryParseConfig(config_path);
  }

  memset(rcclTelemetryDevs, 0, sizeof(rcclTelemetryDevs));
  rcclTelemetryNumDevs = 0;

  for (int i = 0; i < RCCL_TELEMETRY_MAX_DEVS; i++) {
    for (int c = 0; c < RCCL_HWC_COUNT; c++)
      rcclTelemetryDevs[i].hw_counters[c] = -1;
    for (int p = 0; p < 8; p++) {
      rcclTelemetryDevs[i].pfc_rx_frames[p] = -1;
      rcclTelemetryDevs[i].pfc_tx_frames[p] = -1;
      rcclTelemetryDevs[i].pfc_rx_pause_us[p] = -1;
      rcclTelemetryDevs[i].pfc_tx_pause_us[p] = -1;
    }
    rcclTelemetryDevs[i].snap_init_tx_bytes = -1;
    rcclTelemetryDevs[i].snap_init_rx_bytes = -1;
    rcclTelemetryDevs[i].snap_init_tx_packets = -1;
    rcclTelemetryDevs[i].snap_init_rx_packets = -1;
    rcclTelemetryDevs[i].delta_tx_bytes = -1;
    rcclTelemetryDevs[i].delta_rx_bytes = -1;
    rcclTelemetryDevs[i].delta_tx_packets = -1;
    rcclTelemetryDevs[i].delta_rx_packets = -1;
  }

  rcclTelemetryGetTimestamp(rcclTelemetryStartTime, sizeof(rcclTelemetryStartTime));

  rcclTelemetryProcessName[0] = '\0';
  char proc_path[64];
  snprintf(proc_path, sizeof(proc_path), "/proc/%d/comm", (int)getpid());
  FILE* fp = fopen(proc_path, "r");
  if (fp != NULL) {
    if (fgets(rcclTelemetryProcessName, sizeof(rcclTelemetryProcessName), fp) != NULL) {
      size_t len = strlen(rcclTelemetryProcessName);
      if (len > 0 && rcclTelemetryProcessName[len - 1] == '\n')
        rcclTelemetryProcessName[len - 1] = '\0';
    }
    fclose(fp);
  }

  atexit(rcclTelemetryFlush);
}

void rcclTelemetryFlush(void) {
  if (!rcclTelemetryEnabled) {
    return;
  }

  static int flushed = 0;
  if (__atomic_exchange_n(&flushed, 1, __ATOMIC_SEQ_CST)) {
    return;
  }

  for (int i = 0; i < rcclTelemetryNumDevs; i++) {
    rcclTelemetryCollectHwCounters(&rcclTelemetryDevs[i]);
  }

  char hostname[256];
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    strncpy(hostname, "unknown", sizeof(hostname) - 1);
    hostname[sizeof(hostname) - 1] = '\0';
  }

  char filepath[1024];
  snprintf(filepath, sizeof(filepath), "%s/rccl_telemetry_%s_%d.json",
           rcclTelemetryCfg.output_dir, hostname, (int)getpid());

  FILE* fp = fopen(filepath, "w");
  if (fp == NULL) {
    return;
  }

  rcclTelemetryWriteJson(fp);
  fclose(fp);
}

int rcclTelemetryRegisterDevice(int device_id, const char* roce_device,
                                 const char* eth_device, const char* transport) {
  if (!rcclTelemetryEnabled) {
    return -1;
  }

  if (rcclTelemetryNumDevs >= RCCL_TELEMETRY_MAX_DEVS) {
    return -1;
  }

  int idx = __atomic_fetch_add(&rcclTelemetryNumDevs, 1, __ATOMIC_SEQ_CST);
  if (idx >= RCCL_TELEMETRY_MAX_DEVS) {
    __atomic_fetch_sub(&rcclTelemetryNumDevs, 1, __ATOMIC_SEQ_CST);
    return -1;
  }

  RcclDeviceStats* dev = &rcclTelemetryDevs[idx];
  dev->device_id = device_id;

  if (roce_device != NULL) {
    strncpy(dev->roce_device, roce_device, sizeof(dev->roce_device) - 1);
    dev->roce_device[sizeof(dev->roce_device) - 1] = '\0';
  }

  if (eth_device != NULL) {
    strncpy(dev->eth_device, eth_device, sizeof(dev->eth_device) - 1);
    dev->eth_device[sizeof(dev->eth_device) - 1] = '\0';
  }

  if (transport != NULL) {
    strncpy(dev->transport, transport, sizeof(dev->transport) - 1);
    dev->transport[sizeof(dev->transport) - 1] = '\0';
  }

  rcclTelemetrySnapshotInit(dev);

  return idx;
}

void rcclTelemetryGetEthDevice(const char* roce_device, char* eth_device, size_t eth_device_size) {
  eth_device[0] = '\0';

  if (roce_device == NULL || roce_device[0] == '\0') {
    return;
  }

  char path[512];
  snprintf(path, sizeof(path), "/sys/class/infiniband/%s/device/net", roce_device);

  DIR* dir = opendir(path);
  if (dir == NULL) {
    return;
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != NULL) {
    if (entry->d_name[0] != '.') {
      strncpy(eth_device, entry->d_name, eth_device_size - 1);
      eth_device[eth_device_size - 1] = '\0';
      break;
    }
  }
  closedir(dir);
}

/* ------------------------------------------------------------------ */
/* Init snapshot for delta counters                                    */
/* ------------------------------------------------------------------ */

static void rcclTelemetrySnapshotInit(RcclDeviceStats* dev) {
  if (dev->eth_device[0] == '\0') {
    return;
  }

  dev->snap_init_tx_bytes = -1;
  dev->snap_init_rx_bytes = -1;
  dev->snap_init_tx_packets = -1;
  dev->snap_init_rx_packets = -1;

  char driver_name[64] = {0};
  rcclTelemetryGetDriverName(dev->roce_device, driver_name, sizeof(driver_name));
  int drv_col = rcclTelemetryResolveDriverColumn(driver_name);
  dev->drv_col = drv_col;
  const RcclDeltaDriverPatterns* dp = &rcclDeltaPatterns[drv_col];

  RcclEthtoolWantedEx wanted[4];
  int n = 0;

  snprintf(wanted[n].key, sizeof(wanted[n].key), "%s", dp->tx_bytes);
  wanted[n].target = &dev->snap_init_tx_bytes; n++;
  snprintf(wanted[n].key, sizeof(wanted[n].key), "%s", dp->rx_bytes);
  wanted[n].target = &dev->snap_init_rx_bytes; n++;
  snprintf(wanted[n].key, sizeof(wanted[n].key), "%s", dp->tx_packets);
  wanted[n].target = &dev->snap_init_tx_packets; n++;
  snprintf(wanted[n].key, sizeof(wanted[n].key), "%s", dp->rx_packets);
  wanted[n].target = &dev->snap_init_rx_packets; n++;

  rcclTelemetryCollectEthtoolBatch(dev->eth_device, wanted, n);
}

/* ------------------------------------------------------------------ */
/* Config parsing                                                     */
/* ------------------------------------------------------------------ */

static void rcclTelemetryParseConfig(const char* config_path) {
  FILE* fp = fopen(config_path, "r");
  if (fp == NULL) {
    return;
  }

  char line[1024];
  while (fgets(line, sizeof(line), fp) != NULL) {
    char* p;

    p = strstr(line, "\"output_dir\"");
    if (p != NULL) {
      p = strchr(p, ':');
      if (p != NULL) {
        p = strchr(p, '"');
        if (p != NULL) {
          p++;
          char* end = strchr(p, '"');
          if (end != NULL) {
            size_t len = (size_t)(end - p);
            if (len >= sizeof(rcclTelemetryCfg.output_dir))
              len = sizeof(rcclTelemetryCfg.output_dir) - 1;
            strncpy(rcclTelemetryCfg.output_dir, p, len);
            rcclTelemetryCfg.output_dir[len] = '\0';
          }
        }
      }
    }

    p = strstr(line, "\"histogram_max_buckets\"");
    if (p != NULL) {
      p = strchr(p, ':');
      if (p != NULL) {
        int val = atoi(p + 1);
        if (val > 0 && val <= RCCL_TELEMETRY_HISTOGRAM_SIZE)
          rcclTelemetryCfg.histogram_max_buckets = val;
      }
    }

    p = strstr(line, "\"histogram_bucket_interval_ns\"");
    if (p != NULL) {
      p = strchr(p, ':');
      if (p != NULL) {
        int64_t val = strtoll(p + 1, NULL, 10);
        if (val > 0)
          rcclTelemetryCfg.histogram_bucket_interval_ns = val;
      }
    }

    p = strstr(line, "\"hw_counter_list\"");
    if (p != NULL) {
      p = strchr(p, ':');
      if (p != NULL) {
        p = strchr(p, '"');
        if (p != NULL) {
          p++;
          char* end = strchr(p, '"');
          if (end != NULL) {
            size_t len = (size_t)(end - p);
            if (len >= sizeof(rcclTelemetryCfg.hw_counter_list))
              len = sizeof(rcclTelemetryCfg.hw_counter_list) - 1;
            strncpy(rcclTelemetryCfg.hw_counter_list, p, len);
            rcclTelemetryCfg.hw_counter_list[len] = '\0';
          }
        }
      }
    }
  }

  fclose(fp);
}

/* ------------------------------------------------------------------ */
/* Timestamp helper                                                   */
/* ------------------------------------------------------------------ */

static void rcclTelemetryGetTimestamp(char* buf, size_t size) {
  time_t now = time(NULL);
  struct tm* tm_info = localtime(&now);
  strftime(buf, size, "%Y-%m-%d %H:%M:%S", tm_info);
}

/* ------------------------------------------------------------------ */
/* Counter-reading primitives                                         */
/* ------------------------------------------------------------------ */

static int64_t rcclTelemetryReadSysfsCounter(const char* path) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) return -1;

  char buf[64];
  ssize_t n = read(fd, buf, sizeof(buf) - 1);
  close(fd);

  if (n <= 0) return -1;

  buf[n] = '\0';
  return strtoll(buf, NULL, 10);
}

static int rcclTelemetryIsCounterEnabled(const char* counter_name) {
  if (rcclTelemetryCfg.hw_counter_list[0] == '\0') return 1;

  const char* list = rcclTelemetryCfg.hw_counter_list;
  size_t name_len = strlen(counter_name);

  while (*list != '\0') {
    while (*list == ' ' || *list == ',') list++;
    if (strncmp(list, counter_name, name_len) == 0) {
      char next = list[name_len];
      if (next == '\0' || next == ',' || next == ' ') return 1;
    }
    while (*list != '\0' && *list != ',') list++;
  }
  return 0;
}

static void rcclTelemetryGetDriverName(const char* roce_device, char* driver_name, size_t size) {
  driver_name[0] = '\0';

  char link_path[512];
  snprintf(link_path, sizeof(link_path), "/sys/class/infiniband/%s/device/driver", roce_device);

  char resolved[512];
  ssize_t len = readlink(link_path, resolved, sizeof(resolved) - 1);
  if (len < 0) return;
  resolved[len] = '\0';

  char* last_slash = strrchr(resolved, '/');
  if (last_slash != NULL) {
    strncpy(driver_name, last_slash + 1, size - 1);
    driver_name[size - 1] = '\0';
  }
}

static int64_t rcclTelemetryReadHwCounter(const char* roce_device, const char* counter_name) {
  char path[512];
  int64_t val;

  snprintf(path, sizeof(path), "/sys/class/infiniband/%s/hw_counters/%s",
           roce_device, counter_name);
  val = rcclTelemetryReadSysfsCounter(path);
  if (val >= 0) return val;

  for (int port = 1; port <= 2; port++) {
    snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/%d/hw_counters/%s",
             roce_device, port, counter_name);
    val = rcclTelemetryReadSysfsCounter(path);
    if (val >= 0) return val;
  }

  return -1;
}

/* ------------------------------------------------------------------ */
/* Batched debugfs reader                                             */
/* ------------------------------------------------------------------ */

typedef struct {
  const char* key;
  int         counter_idx;
} RcclDebugfsWanted;

static void rcclTelemetryCollectDebugfs(RcclDeviceStats* dev,
                                         const char* driver_name,
                                         const RcclDebugfsWanted* wanted,
                                         int num_wanted) {
  if (driver_name[0] == '\0' || num_wanted == 0) return;

  char path[512];
  snprintf(path, sizeof(path), "/sys/kernel/debug/%s/%s/info",
           driver_name, dev->roce_device);

  FILE* fp = fopen(path, "r");
  if (fp == NULL) return;

  int found = 0;
  char line[256];
  while (fgets(line, sizeof(line), fp) != NULL && found < num_wanted) {
    for (int i = 0; i < num_wanted; i++) {
      if (dev->hw_counters[wanted[i].counter_idx] >= 0) continue;

      const char* key = wanted[i].key;
      char* p = strstr(line, key);
      if (p != NULL) {
        p += strlen(key);
        while (*p == ' ' || *p == ':' || *p == '=') p++;
        if (*p != '\0') {
          dev->hw_counters[wanted[i].counter_idx] = strtoll(p, NULL, 10);
          found++;
        }
      }
    }
  }

  fclose(fp);
}

/* ------------------------------------------------------------------ */
/* Main hw-counter collection                                         */
/* ------------------------------------------------------------------ */

static int rcclTelemetryResolveDriverColumn(const char* driver_name) {
  if (strcmp(driver_name, "ionic") == 0)      return RCCL_DRV_IONIC;
  if (strcmp(driver_name, "bnxt_re") == 0 ||
      strcmp(driver_name, "bnxt_en") == 0)    return RCCL_DRV_BNXT_RE;
  if (strcmp(driver_name, "mlx5_core") == 0)  return RCCL_DRV_MLX5;
  return RCCL_DRV_FALLBACK;
}

static void rcclTelemetryCollectHwCounters(RcclDeviceStats* dev) {
  if (dev->roce_device[0] == '\0') return;

  char driver_name[64];
  rcclTelemetryGetDriverName(dev->roce_device, driver_name, sizeof(driver_name));

  int drv_col = rcclTelemetryResolveDriverColumn(driver_name);

  /* 1. Collect IB sysfs hw_counters (individual reads) */
  for (int c = 0; c < RCCL_HWC_COUNT; c++) {
    const RcclHwcDriverMapping* m = &rcclHwcTable[c].drivers[drv_col];
    if (m->source == HWC_IB_SYSFS && m->key != NULL) {
      if (rcclTelemetryIsCounterEnabled(rcclHwcTable[c].json_name))
        dev->hw_counters[c] = rcclTelemetryReadHwCounter(dev->roce_device, m->key);
    }
  }

  /* 2. Build unified ethtool wanted list: scalar + PFC per-priority + deltas */
  RcclEthtoolWantedEx ew[RCCL_ETHTOOL_MAX_WANTED];
  int ew_n = 0;

  /* 2a. Scalar hw_counters that use ETHTOOL */
  for (int c = 0; c < RCCL_HWC_COUNT; c++) {
    const RcclHwcDriverMapping* m = &rcclHwcTable[c].drivers[drv_col];
    if (m->source == HWC_ETHTOOL && m->key != NULL) {
      if (rcclTelemetryIsCounterEnabled(rcclHwcTable[c].json_name) &&
          ew_n < RCCL_ETHTOOL_MAX_WANTED) {
        strncpy(ew[ew_n].key, m->key, 63);
        ew[ew_n].key[63] = '\0';
        ew[ew_n].target = &dev->hw_counters[c];
        ew_n++;
      }
    }
  }

  /* 2b. PFC per-priority counters */
  const RcclPfcDriverPatterns* pfc = &rcclPfcPatterns[drv_col];
  for (int pri = 0; pri < 8 && ew_n < RCCL_ETHTOOL_MAX_WANTED - 4; pri++) {
    if (pfc->rx_frames_fmt && ew_n < RCCL_ETHTOOL_MAX_WANTED) {
      snprintf(ew[ew_n].key, 64, pfc->rx_frames_fmt, pri);
      ew[ew_n].target = &dev->pfc_rx_frames[pri];
      ew_n++;
    }
    if (pfc->tx_frames_fmt && ew_n < RCCL_ETHTOOL_MAX_WANTED) {
      snprintf(ew[ew_n].key, 64, pfc->tx_frames_fmt, pri);
      ew[ew_n].target = &dev->pfc_tx_frames[pri];
      ew_n++;
    }
    if (pfc->rx_pause_us_fmt && ew_n < RCCL_ETHTOOL_MAX_WANTED) {
      snprintf(ew[ew_n].key, 64, pfc->rx_pause_us_fmt, pri);
      ew[ew_n].target = &dev->pfc_rx_pause_us[pri];
      ew_n++;
    }
    if (pfc->tx_pause_us_fmt && ew_n < RCCL_ETHTOOL_MAX_WANTED) {
      snprintf(ew[ew_n].key, 64, pfc->tx_pause_us_fmt, pri);
      ew[ew_n].target = &dev->pfc_tx_pause_us[pri];
      ew_n++;
    }
  }

  /* 2c. Delta snapshot current-values (stored in local vars, deltas computed after) */
  int64_t cur_tx_bytes = -1, cur_rx_bytes = -1;
  int64_t cur_tx_packets = -1, cur_rx_packets = -1;
  const RcclDeltaDriverPatterns* dp = &rcclDeltaPatterns[drv_col];

  if (ew_n + 4 <= RCCL_ETHTOOL_MAX_WANTED) {
    snprintf(ew[ew_n].key, 64, "%s", dp->tx_bytes);   ew[ew_n].target = &cur_tx_bytes;   ew_n++;
    snprintf(ew[ew_n].key, 64, "%s", dp->rx_bytes);   ew[ew_n].target = &cur_rx_bytes;   ew_n++;
    snprintf(ew[ew_n].key, 64, "%s", dp->tx_packets); ew[ew_n].target = &cur_tx_packets; ew_n++;
    snprintf(ew[ew_n].key, 64, "%s", dp->rx_packets); ew[ew_n].target = &cur_rx_packets; ew_n++;
  }

  /* Single ethtool pass for all wanted entries */
  rcclTelemetryCollectEthtoolBatch(dev->eth_device, ew, ew_n);

  /* Compute deltas: snap_init < 0 means snapshot was never taken -> delta = -1 */
  dev->delta_tx_bytes   = (dev->snap_init_tx_bytes   >= 0 && cur_tx_bytes   >= 0)
                          ? cur_tx_bytes   - dev->snap_init_tx_bytes   : -1;
  dev->delta_rx_bytes   = (dev->snap_init_rx_bytes   >= 0 && cur_rx_bytes   >= 0)
                          ? cur_rx_bytes   - dev->snap_init_rx_bytes   : -1;
  dev->delta_tx_packets = (dev->snap_init_tx_packets >= 0 && cur_tx_packets >= 0)
                          ? cur_tx_packets - dev->snap_init_tx_packets : -1;
  dev->delta_rx_packets = (dev->snap_init_rx_packets >= 0 && cur_rx_packets >= 0)
                          ? cur_rx_packets - dev->snap_init_rx_packets : -1;

  /* 3. Collect debugfs counters (single file read) */
  RcclDebugfsWanted debugfs_list[RCCL_HWC_COUNT];
  int debugfs_count = 0;
  for (int c = 0; c < RCCL_HWC_COUNT; c++) {
    const RcclHwcDriverMapping* m = &rcclHwcTable[c].drivers[drv_col];
    if (m->source == HWC_DEBUGFS && m->key != NULL) {
      if (rcclTelemetryIsCounterEnabled(rcclHwcTable[c].json_name)) {
        debugfs_list[debugfs_count].key = m->key;
        debugfs_list[debugfs_count].counter_idx = c;
        debugfs_count++;
      }
    }
  }
  rcclTelemetryCollectDebugfs(dev, driver_name, debugfs_list, debugfs_count);
}

/* ------------------------------------------------------------------ */
/* JSON writer                                                        */
/* ------------------------------------------------------------------ */

static void rcclTelemetryWriteJsonArray8(FILE* fp, const char* name, const int64_t arr[8], int trailing_comma) {
  fprintf(fp, "        \"%s\": [%ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld]%s\n",
          name,
          (long)arr[0], (long)arr[1], (long)arr[2], (long)arr[3],
          (long)arr[4], (long)arr[5], (long)arr[6], (long)arr[7],
          trailing_comma ? "," : "");
}

static void rcclTelemetryWriteJson(FILE* fp) {
  char end_time[64];
  rcclTelemetryGetTimestamp(end_time, sizeof(end_time));

  char hostname[256];
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    strncpy(hostname, "unknown", sizeof(hostname) - 1);
    hostname[sizeof(hostname) - 1] = '\0';
  }

  fprintf(fp, "{\n");
  fprintf(fp, "  \"version\": \"1.0\",\n");
  fprintf(fp, "  \"host_name\": \"%s\",\n", hostname);
  fprintf(fp, "  \"process_name\": \"%s\",\n", rcclTelemetryProcessName);
  fprintf(fp, "  \"process_id\": \"%d\",\n", (int)getpid());
  fprintf(fp, "  \"start_time\": \"%s\",\n", rcclTelemetryStartTime);
  fprintf(fp, "  \"end_time\": \"%s\",\n", end_time);

  const char* transport = "IB-CAST";
  if (rcclTelemetryNumDevs > 0 && rcclTelemetryDevs[0].transport[0] != '\0')
    transport = rcclTelemetryDevs[0].transport;
  fprintf(fp, "  \"transport\": \"%s\",\n", transport);

  fprintf(fp, "  \"devices\": [\n");

  int devsPrinted = 0;
  for (int d = 0; d < rcclTelemetryNumDevs; d++) {
    RcclDeviceStats* dev = &rcclTelemetryDevs[d];

    int activeChannels = 0;
    for (int c = 0; c < dev->num_channels; c++) {
      RcclChannelStats* ch = &dev->channels[c];
      if (ch->num_qps > 0 || ch->num_wqe_sent || ch->num_wqe_rcvd || ch->num_wqe_completed)
        activeChannels++;
    }
    if (dev->tx_bytes == 0 && dev->rx_bytes == 0 && dev->num_cq_errors == 0 && activeChannels == 0)
      continue;

    if (devsPrinted > 0) fprintf(fp, ",\n");
    fprintf(fp, "    {\n");
    fprintf(fp, "      \"device_id\": %d,\n", dev->device_id);
    fprintf(fp, "      \"roce_device\": \"%s\",\n", dev->roce_device);
    fprintf(fp, "      \"eth_device\": \"%s\",\n", dev->eth_device);
    fprintf(fp, "      \"tx_bytes\": %lu,\n", (unsigned long)dev->tx_bytes);
    fprintf(fp, "      \"rx_bytes\": %lu,\n", (unsigned long)dev->rx_bytes);
    fprintf(fp, "      \"num_cq_errors\": %lu,\n", (unsigned long)dev->num_cq_errors);
    fprintf(fp, "      \"num_channels\": %d,\n", dev->num_channels);
    fprintf(fp, "      \"active_channels\": %d,\n", activeChannels);

    fprintf(fp, "      \"channels\": [\n");
    int chPrinted = 0;
    for (int c = 0; c < dev->num_channels; c++) {
      RcclChannelStats* ch = &dev->channels[c];

      if (ch->num_qps == 0 && ch->num_data_qp == 0 && ch->num_cts_qp == 0)
        continue;

      if (chPrinted > 0) fprintf(fp, ",\n");
      fprintf(fp, "        {\n");
      fprintf(fp, "          \"id\": %d,\n", ch->id);
      fprintf(fp, "          \"num_wqe_sent\": %lu,\n", (unsigned long)ch->num_wqe_sent);
      fprintf(fp, "          \"num_wqe_rcvd\": %lu,\n", (unsigned long)ch->num_wqe_rcvd);
      fprintf(fp, "          \"num_wqe_completed\": %lu,\n", (unsigned long)ch->num_wqe_completed);
      fprintf(fp, "          \"num_cts_sent\": %lu,\n", (unsigned long)ch->num_cts_sent);
      fprintf(fp, "          \"num_data_qp\": %d,\n", ch->num_data_qp);
      fprintf(fp, "          \"num_cts_qp\": %d,\n", ch->num_cts_qp);

      fprintf(fp, "          \"queue_pairs\": [\n");
      for (int q = 0; q < ch->num_qps; q++) {
        RcclQpStats* qp = &ch->qp[q];

        fprintf(fp, "            {\n");
        fprintf(fp, "              \"id\": %d,\n", qp->id);
        fprintf(fp, "              \"num_wqe_sent\": %lu,\n", (unsigned long)qp->num_wqe_sent);
        fprintf(fp, "              \"num_wqe_rcvd\": %lu,\n", (unsigned long)qp->num_wqe_rcvd);
        fprintf(fp, "              \"num_wqe_completed\": %lu,\n", (unsigned long)qp->num_wqe_completed);
        fprintf(fp, "              \"num_slot_miss\": %lu,\n", (unsigned long)qp->num_slot_miss);
        fprintf(fp, "              \"wqe_completion_ns_min\": %ld,\n", (long)qp->wqe_completion_ns_min);
        fprintf(fp, "              \"wqe_completion_ns_max\": %ld,\n", (long)qp->wqe_completion_ns_max);

        fprintf(fp, "              \"wqe_completion_histogram\": [\n");
        int max_buckets = rcclTelemetryCfg.histogram_max_buckets;
        if (max_buckets > RCCL_TELEMETRY_HISTOGRAM_SIZE)
          max_buckets = RCCL_TELEMETRY_HISTOGRAM_SIZE;
        for (int b = 0; b < max_buckets; b++) {
          int64_t latency_ns = (int64_t)(b + 1) * rcclTelemetryCfg.histogram_bucket_interval_ns;
          fprintf(fp, "                {\"latency_ns\": %ld, \"num_wqe\": %lu}%s\n",
                  (long)latency_ns, (unsigned long)qp->wqe_completion_histogram[b],
                  (b < max_buckets - 1) ? "," : "");
        }
        fprintf(fp, "              ]\n");

        fprintf(fp, "            }%s\n", (q < ch->num_qps - 1) ? "," : "");
      }
      fprintf(fp, "          ]\n");

      fprintf(fp, "        }");
      chPrinted++;
    }
    if (chPrinted > 0) fprintf(fp, "\n");
    fprintf(fp, "      ],\n");

    /* Hardware counters */
    fprintf(fp, "      \"hw_counters\": {\n");

    /* Scalar counters from descriptor table — omit counters unmapped for this driver */
    int drv = dev->drv_col;
    for (int c = 0; c < RCCL_HWC_COUNT; c++) {
      if (rcclHwcTable[c].drivers[drv].source == HWC_NONE && dev->hw_counters[c] < 0)
        continue;
      fprintf(fp, "        \"%s\": %ld,\n",
              rcclHwcTable[c].json_name, (long)dev->hw_counters[c]);
    }

    /* Per-priority PFC arrays — omit when driver has no pattern for them */
    const RcclPfcDriverPatterns* pfc = &rcclPfcPatterns[drv];
    if (pfc->rx_frames_fmt)
      rcclTelemetryWriteJsonArray8(fp, "pfc_rx_frames",   dev->pfc_rx_frames,   1);
    if (pfc->tx_frames_fmt)
      rcclTelemetryWriteJsonArray8(fp, "pfc_tx_frames",   dev->pfc_tx_frames,   1);
    if (pfc->rx_pause_us_fmt)
      rcclTelemetryWriteJsonArray8(fp, "pfc_rx_pause_us", dev->pfc_rx_pause_us, 1);
    if (pfc->tx_pause_us_fmt)
      rcclTelemetryWriteJsonArray8(fp, "pfc_tx_pause_us", dev->pfc_tx_pause_us, 1);

    /* Delta counters */
    fprintf(fp, "        \"delta_tx_bytes\": %ld,\n",   (long)dev->delta_tx_bytes);
    fprintf(fp, "        \"delta_rx_bytes\": %ld,\n",   (long)dev->delta_rx_bytes);
    fprintf(fp, "        \"delta_tx_packets\": %ld,\n", (long)dev->delta_tx_packets);
    fprintf(fp, "        \"delta_rx_packets\": %ld\n",  (long)dev->delta_rx_packets);

    fprintf(fp, "      }\n");

    fprintf(fp, "    }");
    devsPrinted++;
  }
  if (devsPrinted > 0) fprintf(fp, "\n");

  fprintf(fp, "  ]\n");
  fprintf(fp, "}\n");
}

/* ---------- Stats query API (called by ANP plugin) ---------- */

static const char* rcclStatNames[RCCL_TELEMETRY_NUM_STATS] = {
  "num_wqe_sent",
  "num_wqe_rcvd",
  "num_wqe_completed",
  "num_slot_miss",
  "num_cts_sent",
  "tx_bytes",
  "rx_bytes",
  "num_cq_errors"
};

int rcclTelemetryGetStatNames(int* nstats, const char** names[]) {
  if (!nstats || !names) return -1;
  *nstats = RCCL_TELEMETRY_NUM_STATS;
  *names  = rcclStatNames;
  return 0;
}

int rcclTelemetryGetStats(int dev, int* nstats, uint64_t stats[]) {
  if (!nstats || !stats) return -1;
  *nstats = RCCL_TELEMETRY_NUM_STATS;
  memset(stats, 0, RCCL_TELEMETRY_NUM_STATS * sizeof(uint64_t));

  if (!rcclTelemetryEnabled || dev < 0 || dev >= rcclTelemetryNumDevs)
    return 0;

  RcclDeviceStats* d = &rcclTelemetryDevs[dev];

  uint64_t wqe_sent = 0, wqe_rcvd = 0, wqe_completed = 0;
  uint64_t slot_miss = 0, cts_sent = 0;

  for (int c = 0; c < RCCL_TELEMETRY_MAX_CHANNELS; c++) {
    RcclChannelStats* ch = &d->channels[c];
    if (ch->num_qps == 0 && ch->num_data_qp == 0 && ch->num_cts_qp == 0)
      continue;
    cts_sent += __atomic_load_n(&ch->num_cts_sent, __ATOMIC_RELAXED);
    for (int q = 0; q < ch->num_qps; q++) {
      wqe_sent      += __atomic_load_n(&ch->qp[q].num_wqe_sent,      __ATOMIC_RELAXED);
      wqe_rcvd      += __atomic_load_n(&ch->qp[q].num_wqe_rcvd,      __ATOMIC_RELAXED);
      wqe_completed += __atomic_load_n(&ch->qp[q].num_wqe_completed, __ATOMIC_RELAXED);
      slot_miss     += __atomic_load_n(&ch->qp[q].num_slot_miss,     __ATOMIC_RELAXED);
    }
  }

  stats[0] = wqe_sent;
  stats[1] = wqe_rcvd;
  stats[2] = wqe_completed;
  stats[3] = slot_miss;
  stats[4] = cts_sent;
  stats[5] = __atomic_load_n(&d->tx_bytes,      __ATOMIC_RELAXED);
  stats[6] = __atomic_load_n(&d->rx_bytes,      __ATOMIC_RELAXED);
  stats[7] = __atomic_load_n(&d->num_cq_errors, __ATOMIC_RELAXED);

  return 0;
}
