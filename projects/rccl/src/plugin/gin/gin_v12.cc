/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_gin.h"
#include "proxy.h"
#include "os.h"

static ncclGin_v12_t* ncclGin_v12;
static ncclGin_t ncclGin;

static ncclResult_t ncclGin_connect(void* ctx, void* handles[], int nranks, int rank,
  void* listenComm, void** collComm) {
  return ncclGin_v12->connect(ctx, handles, nranks, rank, 1, 0, listenComm, collComm);
}

static ncclResult_t ncclGin_createContext(void* collComm, ncclGinConfig_t* config,
    void** ginCtx, ncclNetDeviceHandle_t** devHandle) {
  if (ncclGin_v12->createContext == NULL) {
    if (config->nContexts > 1) {
      WARN("GIN plugin v12 does not support multiple contexts");
      return ncclInvalidUsage;
    }
    *ginCtx = collComm;
    return ncclSuccess;
  } else {
    NCCLCHECK(ncclGin_v12->createContext(collComm, config->nSignals, config->nCounters, config->nContexts, ginCtx, devHandle));
  }
  return ncclSuccess;
}

static ncclResult_t ncclGin_destroyContext(void* ginCtx) {
  if (ncclGin_v12->destroyContext) {
    NCCLCHECK(ncclGin_v12->destroyContext(ginCtx));
  }
  return ncclSuccess;
}

// iflush was introduced to ensure data visibility after a get operation. ncclGin_v12 does not
// support get so this function is a no-op.
static ncclResult_t ncclGin_iflush(void* ginCtx, int context, void* mhandle, uint32_t rank, void** request) {
  *request = NULL;
  return ncclSuccess;
}

static ncclResult_t ncclGin_iput(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
    uint64_t dstOff, void* dstMhandle, uint32_t rank, void** request) {
  if (context != 0) {
    WARN("GIN plugin v12 does not support multiple contexts");
    return ncclInvalidUsage;
  }
  return ncclGin_v12->iput(ginCtx, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, 0, request);
}

static ncclResult_t ncclGin_iputSignal(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle,
    size_t size, uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff, void *signalMhandle,
    uint64_t signalValue, uint32_t signalOp, void** request) {
  if (context != 0) {
    WARN("GIN plugin v12 does not support multiple connections");
    return ncclInvalidUsage;
  }
  return ncclGin_v12->iputSignal(ginCtx, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, signalOff, signalMhandle, signalValue, signalOp, 0, request);
}

static ncclResult_t ncclGin_v12_fillProperties(ncclGin_v12_t* backing, int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v11_t props_v11;
  NCCLCHECK(backing->getProperties(dev, &props_v11));
  props->name = props_v11.name;
  props->pciPath = props_v11.pciPath;
  props->guid = props_v11.guid;
  props->ptrSupport = props_v11.ptrSupport;
  props->regIsGlobal = props_v11.regIsGlobal;
  props->forceFlush = props_v11.forceFlush;
  props->speed = props_v11.speed;
  props->port = props_v11.port;
  props->latency = props_v11.latency;
  props->maxComms = props_v11.maxComms;
  props->maxRecvs = props_v11.maxRecvs;
  props->netDeviceType = props_v11.netDeviceType;
  props->netDeviceVersion = props_v11.netDeviceVersion;
  props->vProps.ndevs = props_v11.vProps.ndevs;
  for (int i = 0; i < props_v11.vProps.ndevs; i++)
    props->vProps.devs[i] = props_v11.vProps.devs[i];
  props->maxP2pBytes = props_v11.maxP2pBytes;
  props->maxCollBytes = props_v11.maxCollBytes;
  props->maxMultiRequestSize = props_v11.maxMultiRequestSize;
#ifdef RCCL_DISABLE_2_30_CODE
  // NCCL 2.30 not yet merged completely
  props->railId = NCCL_NET_ID_UNDEF;
  props->planeId = NCCL_NET_ID_UNDEF;
#endif
  return ncclSuccess;
}

static ncclResult_t ncclGin_getProperties(int dev, ncclNetProperties_t* props) {
  return ncclGin_v12_fillProperties(ncclGin_v12, dev, props);
}

ncclGin_t* getNcclGin_v12(void* lib) {
  ncclGin_v12 = (ncclGin_v12_t*)ncclOsDlsym(lib, "ncclGinPlugin_v12");
  if (ncclGin_v12) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded gin plugin %s (v12)", ncclGin_v12->name);
    ncclGin.name = ncclGin_v12->name;
    ncclGin.init = ncclGin_v12->init;
    ncclGin.devices = ncclGin_v12->devices;
    ncclGin.getProperties = ncclGin_getProperties;
    ncclGin.listen = ncclGin_v12->listen;
    ncclGin.connect = ncclGin_connect;
    ncclGin.createContext = ncclGin_createContext;
    ncclGin.regMrSym = ncclGin_v12->regMrSym;
    ncclGin.regMrSymDmaBuf = ncclGin_v12->regMrSymDmaBuf;
    ncclGin.deregMrSym = ncclGin_v12->deregMrSym;
    ncclGin.destroyContext = ncclGin_destroyContext;
    ncclGin.closeColl = ncclGin_v12->closeColl;
    ncclGin.closeListen = ncclGin_v12->closeListen;
    ncclGin.iput = ncclGin_iput;
    ncclGin.iputSignal = ncclGin_iputSignal;
    ncclGin.iget = NULL;
    ncclGin.iflush = ncclGin_iflush;
    ncclGin.test = ncclGin_v12->test;
    ncclGin.ginProgress = ncclGin_v12->ginProgress;
    ncclGin.queryLastError = ncclGin_v12->queryLastError;
    ncclGin.finalize = ncclGin_v12->finalize;
    return &ncclGin;
  }
  return nullptr;
}

// Internal (built-in) GIN plugins are registered by passing a backing
// ncclGin_v12_t* directly, NOT via dlsym. Several built-ins (ncclGinIb,
// ncclGinIbProxy, ...) coexist, so the v12->v13 adapter cannot route through
// the single file-scope `ncclGin_v12` pointer used by the external dlsym path:
// that pointer is only set by getNcclGin_v12() and is NULL for built-ins,
// and one global cannot serve multiple backings at once. Instead, give each
// built-in slot its own backing pointer plus a set of slot-bound trampolines
// that close over that backing.
static const int ncclGin_built_in_count = 4;
static ncclGin_t ncclGin_built_in[ncclGin_built_in_count];
static ncclGin_v12_t* ncclGin_built_in_backing[ncclGin_built_in_count];
static int ncclGin_built_in_idx = 0;

// Trampolines: one set per slot, each forwarding to its slot's backing v12 plugin.
#define NCCL_GIN_V12_INTERNAL_TRAMPOLINES(SLOT)                                                   \
  static ncclResult_t ncclGin_v12_int##SLOT##_connect(void* ctx, void* handles[], int nranks,    \
      int rank, void* listenComm, void** collComm) {                                             \
    return ncclGin_built_in_backing[SLOT]->connect(ctx, handles, nranks, rank, 1, 0,             \
        listenComm, collComm);                                                                   \
  }                                                                                               \
  static ncclResult_t ncclGin_v12_int##SLOT##_createContext(void* collComm,                      \
      ncclGinConfig_t* config, void** ginCtx, ncclNetDeviceHandle_t** devHandle) {               \
    ncclGin_v12_t* b = ncclGin_built_in_backing[SLOT];                                            \
    if (b->createContext == NULL) {                                                               \
      if (config->nContexts > 1) {                                                                \
        WARN("GIN plugin v12 does not support multiple contexts");                                \
        return ncclInvalidUsage;                                                                  \
      }                                                                                            \
      *ginCtx = collComm;                                                                          \
      return ncclSuccess;                                                                          \
    }                                                                                              \
    NCCLCHECK(b->createContext(collComm, config->nSignals, config->nCounters,                    \
        config->nContexts, ginCtx, devHandle));                                                   \
    return ncclSuccess;                                                                            \
  }                                                                                               \
  static ncclResult_t ncclGin_v12_int##SLOT##_destroyContext(void* ginCtx) {                      \
    ncclGin_v12_t* b = ncclGin_built_in_backing[SLOT];                                            \
    if (b->destroyContext) NCCLCHECK(b->destroyContext(ginCtx));                                  \
    return ncclSuccess;                                                                            \
  }                                                                                               \
  static ncclResult_t ncclGin_v12_int##SLOT##_iput(void* ginCtx, int context, uint64_t srcOff,   \
      void* srcMhandle, size_t size, uint64_t dstOff, void* dstMhandle, uint32_t rank,           \
      void** request) {                                                                           \
    if (context != 0) {                                                                            \
      WARN("GIN plugin v12 does not support multiple contexts");                                  \
      return ncclInvalidUsage;                                                                     \
    }                                                                                              \
    return ncclGin_built_in_backing[SLOT]->iput(ginCtx, srcOff, srcMhandle, size, dstOff,        \
        dstMhandle, rank, 0, request);                                                            \
  }                                                                                               \
  static ncclResult_t ncclGin_v12_int##SLOT##_iputSignal(void* ginCtx, int context,              \
      uint64_t srcOff, void* srcMhandle, size_t size, uint64_t dstOff, void* dstMhandle,         \
      uint32_t rank, uint64_t signalOff, void* signalMhandle, uint64_t signalValue,              \
      uint32_t signalOp, void** request) {                                                        \
    if (context != 0) {                                                                            \
      WARN("GIN plugin v12 does not support multiple connections");                               \
      return ncclInvalidUsage;                                                                     \
    }                                                                                              \
    return ncclGin_built_in_backing[SLOT]->iputSignal(ginCtx, srcOff, srcMhandle, size, dstOff,  \
        dstMhandle, rank, signalOff, signalMhandle, signalValue, signalOp, 0, request);          \
  }                                                                                               \
  static ncclResult_t ncclGin_v12_int##SLOT##_getProperties(int dev, ncclNetProperties_t* p) {   \
    return ncclGin_v12_fillProperties(ncclGin_built_in_backing[SLOT], dev, p);                    \
  }

NCCL_GIN_V12_INTERNAL_TRAMPOLINES(0)
NCCL_GIN_V12_INTERNAL_TRAMPOLINES(1)
NCCL_GIN_V12_INTERNAL_TRAMPOLINES(2)
NCCL_GIN_V12_INTERNAL_TRAMPOLINES(3)

// iflush is a stateless no-op (v12 has no get), so it needs no per-slot backing.
#define NCCL_GIN_V12_INTERNAL_SLOT(DST, SLOT)                          \
  do {                                                                 \
    (DST)->connect        = ncclGin_v12_int##SLOT##_connect;          \
    (DST)->createContext  = ncclGin_v12_int##SLOT##_createContext;    \
    (DST)->destroyContext = ncclGin_v12_int##SLOT##_destroyContext;   \
    (DST)->iput           = ncclGin_v12_int##SLOT##_iput;             \
    (DST)->iputSignal     = ncclGin_v12_int##SLOT##_iputSignal;       \
    (DST)->getProperties  = ncclGin_v12_int##SLOT##_getProperties;    \
  } while (0)

ncclGin_t* getNcclGin_v12_internal(ncclGin_v12_t* backing) {
  if ((ncclGin_built_in_idx >= ncclGin_built_in_count) || !backing) {
    return nullptr;
  }

  int slot = ncclGin_built_in_idx++;
  ncclGin_built_in_backing[slot] = backing;

  ncclGin_t* __ncclGin = &ncclGin_built_in[slot];
  // Pass-through fields (no v12->v13 signature change): copy straight from backing.
  __ncclGin->name = backing->name;
  __ncclGin->init = backing->init;
  __ncclGin->devices = backing->devices;
  __ncclGin->listen = backing->listen;
  __ncclGin->regMrSym = backing->regMrSym;
  __ncclGin->regMrSymDmaBuf = backing->regMrSymDmaBuf;
  __ncclGin->deregMrSym = backing->deregMrSym;
  __ncclGin->closeColl = backing->closeColl;
  __ncclGin->closeListen = backing->closeListen;
  __ncclGin->iget = NULL;
  __ncclGin->iflush = ncclGin_iflush;
  __ncclGin->test = backing->test;
  __ncclGin->ginProgress = backing->ginProgress;
  __ncclGin->queryLastError = backing->queryLastError;
  __ncclGin->finalize = backing->finalize;
  // Adapted fields: slot-bound trampolines that forward to this slot's backing.
  switch (slot) {
    case 0: NCCL_GIN_V12_INTERNAL_SLOT(__ncclGin, 0); break;
    case 1: NCCL_GIN_V12_INTERNAL_SLOT(__ncclGin, 1); break;
    case 2: NCCL_GIN_V12_INTERNAL_SLOT(__ncclGin, 2); break;
    case 3: NCCL_GIN_V12_INTERNAL_SLOT(__ncclGin, 3); break;
    default: return nullptr;
  }
  return __ncclGin;
}
