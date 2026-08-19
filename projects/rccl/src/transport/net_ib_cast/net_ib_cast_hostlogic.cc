/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net_ib_cast_hostlogic.h"

#include <cstring>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "debug.h"                /* WARN */
#include "plugin/nccl_net.h"      /* NCCL_NET_MAX_DEVS_PER_NIC */
#include "plugin/net/net_v12.h"   /* NCCL_NET_ID_UNDEF */
#include "net_ib_cast_inspect.h"  /* extern "C" test wrapper prototypes */

ncclResult_t IbCastGetPlaneIndex(int devPlane, int16_t* count, int16_t* planes, int16_t* idx) {
  int16_t p = 0;
  while (p < *count && planes[p] != devPlane) p++;
  if (p == *count) {
    if (p == (NCCL_IB_PLANE_MAX_INDEX - 1)) {
      WARN("NCCL cannot use more than %d plane IDs.", NCCL_IB_PLANE_MAX_INDEX);
      return ncclInvalidUsage;
    }
    if (devPlane != NCCL_NET_ID_UNDEF && (devPlane & NCCL_IB_PLANE_VIRT_BIT)) {
      WARN("NCCL cannot use a plane ID that is %d.", devPlane);
      return ncclInvalidUsage;
    }
    planes[(*count)++] = devPlane;
  }
  *idx = p;
  return ncclSuccess;
}

sa_family_t getGidAddrFamily(union ibv_gid* gid) {
  const struct in6_addr* a = (struct in6_addr*)gid->raw;
  bool isIpV4Mapped = ((a->s6_addr32[0] | a->s6_addr32[1]) | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL;
  bool isIpV4MappedMulticast =
    (a->s6_addr32[0] == htonl(0xff0e0000) && ((a->s6_addr32[1] | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL));
  return (isIpV4Mapped || isIpV4MappedMulticast) ? AF_INET : AF_INET6;
}

bool configuredGid(union ibv_gid* gid) {
  const struct in6_addr* a = (struct in6_addr*)gid->raw;
  int trailer = (a->s6_addr32[1] | a->s6_addr32[2] | a->s6_addr32[3]);
  if (((a->s6_addr32[0] | trailer) == 0UL) || ((a->s6_addr32[0] == htonl(0xfe800000)) && (trailer == 0UL))) {
    return false;
  }
  return true;
}

bool linkLocalGid(union ibv_gid* gid) {
  const struct in6_addr* a = (struct in6_addr*)gid->raw;
  if (a->s6_addr32[0] == htonl(0xfe800000) && a->s6_addr32[1] == 0UL) {
    return true;
  }
  return false;
}

bool validGid(union ibv_gid* gid) {
  return (configuredGid(gid) && !linkLocalGid(gid));
}

// Check if two RoCE GIDs are on the same subnet.
// For IPv4-mapped GIDs (::ffff:a.b.c.d), uses the given prefix length (1..32).
// For native IPv6 GIDs, compares the 64-bit subnet prefix.
bool gidSameSubnet(union ibv_gid* local, union ibv_gid* remote, int prefixLen) {
  sa_family_t localFam = getGidAddrFamily(local);
  sa_family_t remoteFam = getGidAddrFamily(remote);
  if (localFam != remoteFam) return false;
  if (localFam == AF_INET) {
    // IPv4-mapped: compare using configured prefix length.
    // IPv4 address is in bytes 12-15 of the raw GID.
    uint32_t localIp, remoteIp;
    memcpy(&localIp, local->raw + 12, 4);
    memcpy(&remoteIp, remote->raw + 12, 4);
    uint32_t mask = htonl(~((1U << (32 - prefixLen)) - 1));
    return (localIp & mask) == (remoteIp & mask);
  } else {
    // IPv6: compare subnet prefix (first 64 bits)
    return local->global.subnet_prefix == remote->global.subnet_prefix;
  }
}

// check if a local GID matches ANY of the remote GIDs.
bool subnetMatchesAny(union ibv_gid* localGid, union ibv_gid* remoteGids, int nRemoteGids, int prefixLen) {
  for (int r = 0; r < nRemoteGids; r++) {
    if (validGid(&remoteGids[r]) && gidSameSubnet(localGid, &remoteGids[r], prefixLen)) return true;
  }
  return false;
}

extern "C" ncclResult_t ncclIbCastTestGetPlaneIndex(int devPlane, int16_t* count, int16_t* planes, int16_t* idx) {
  if (!count || !planes || !idx) return ncclInvalidArgument;
  return IbCastGetPlaneIndex(devPlane, count, planes, idx);
}

extern "C" int ncclIbCastTestGidSameSubnet(const uint8_t localGid[16], const uint8_t remoteGid[16], int prefixLen) {
  union ibv_gid l, r;
  memcpy(l.raw, localGid, 16);
  memcpy(r.raw, remoteGid, 16);
  return gidSameSubnet(&l, &r, prefixLen) ? 1 : 0;
}

extern "C" int ncclIbCastTestSubnetMatchesAny(const uint8_t localGid[16], const uint8_t* remoteGids, int nRemote,
                                              int prefixLen) {
  union ibv_gid l;
  memcpy(l.raw, localGid, 16);
  union ibv_gid r[NCCL_NET_MAX_DEVS_PER_NIC];
  if (nRemote < 0 || nRemote > NCCL_NET_MAX_DEVS_PER_NIC) return 0;
  for (int i = 0; i < nRemote; i++) memcpy(r[i].raw, remoteGids + (size_t)i * 16, 16);
  return subnetMatchesAny(&l, r, nRemote, prefixLen) ? 1 : 0;
}
