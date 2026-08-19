/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_NET_IB_CAST_HOSTLOGIC_H_
#define RCCL_NET_IB_CAST_HOSTLOGIC_H_

#include <cstdint>
#include <sys/socket.h>  /* sa_family_t */

#include "nccl.h"     /* ncclResult_t */
#include "ibvwrap.h"  /* union ibv_gid */

// NCCL_IB_PLANE_MAX_INDEX must be < 15 as we use int16_t for plane IDs
// Typically 12 user-defined planes + 1 plane for undefined plane IDs
#define NCCL_IB_PLANE_MAX_INDEX 14
#define NCCL_IB_PLANE_VIRT_BIT (0x1 << NCCL_IB_PLANE_MAX_INDEX)
static_assert(NCCL_IB_PLANE_MAX_INDEX < 15,
              "NCCL_IB_PLANE_MAX_INDEX must be < 15: plane IDs are stored in int16_t (bit 15 is the sign bit)");

// Pure CPU helpers for the net-ib-cast transport, split into their own
// translation unit so they carry no HIP/RDMA runtime dependencies. They only
// manipulate plane-id bookkeeping and raw GID bytes, which lets the host-only
// micro-tests link and exercise them GPU-free in well under a second.
ncclResult_t IbCastGetPlaneIndex(int devPlane, int16_t* count, int16_t* planes, int16_t* idx);

sa_family_t getGidAddrFamily(union ibv_gid* gid);
bool configuredGid(union ibv_gid* gid);
bool linkLocalGid(union ibv_gid* gid);
bool validGid(union ibv_gid* gid);
bool gidSameSubnet(union ibv_gid* local, union ibv_gid* remote, int prefixLen);
bool subnetMatchesAny(union ibv_gid* localGid, union ibv_gid* remoteGids, int nRemoteGids, int prefixLen);

#endif  // RCCL_NET_IB_CAST_HOSTLOGIC_H_
