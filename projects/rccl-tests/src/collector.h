/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*************************************************************************
 * Network Counter Collector
 *
 * Self-contained library for collecting Thor2 / AINIC NIC counters
 * before/after an operation and printing a summary table.  No dependency
 * on NCCL, RCCL, or any GPU runtime -- can be integrated into any
 * application.
 *
 * Environment variables:
 *   RCCL_TESTS_NET_COUNTER_ENABLE=1   – enable collection
 *   RCCL_TESTS_NIC_COUNTER_LIST=a,b   – comma-separated counter subset
 *                                        (default: all counters for detected NIC)
 *   NCCL_IB_HCA=ib0,ib1,...           – IB device list (primary)
 *   RCCL_TESTS_NET_COUNTER_NIC_PREFIX – NIC prefix filter for auto-discovery
 *
 * Counter sources (looked up automatically per NIC type):
 *   COUNTER_SRC_ETHTOOL  – ethtool -S <ethernet device>
 *   COUNTER_SRC_IB_HW    – /sys/class/infiniband/<dev>/ports/<port>/hw_counters/
 *                           (port parsed from NCCL_IB_HCA suffix, default 1;
 *                            falls back to device-level hw_counters/ if needed)
 *   COUNTER_SRC_DEBUGFS  – /sys/kernel/debug/bnxt_re/<dev>/info (root only)
 *
 * Each counter lists alternative (source, key) pairs tried in order; the
 * first one that can be read wins. A counter none of them can read is
 * printed as "N/A N/A" (count, rate) so column positions are preserved.
 *************************************************************************/
#ifndef __COLLECTOR_H__
#define __COLLECTOR_H__

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ---- types --------------------------------------------------------------

enum CounterSource {
  COUNTER_SRC_ETHTOOL,
  COUNTER_SRC_IB_HW,
  COUNTER_SRC_DEBUGFS
};

struct CounterSourceKey {
  CounterSource source;
  // Exact key, or a "%d" pattern summed over priorities 0..7. With
  // CounterDescriptor::is_prefix, every ethtool key starting with it is summed.
  std::string key;
};

struct CounterDescriptor {
  std::string name;                      // display name (table column header)
  bool is_prefix;                        // ethtool keys are prefixes, summed
  std::vector<CounterSourceKey> sources; // tried in order, first readable wins
};

typedef enum { NIC_UNKNOWN, NIC_BNXT_RE, NIC_IONIC } NicType;

struct NetworkCounterSnapshot {
  std::map<std::string, uint64_t> counters;  // display name -> value; absent = unreadable
  char    nic_name[256];
  char    ib_device[256];
  int64_t timestamp_us;
  NicType nic_type;
};

struct NetworkCounterContext {
  std::vector<NetworkCounterSnapshot> snapshots_before;
  std::vector<std::string> nic_names;   // ethernet NIC names (for ethtool)
  std::vector<std::string> ib_names;    // IB device names (for hw_counters / debugfs)
  std::vector<CounterDescriptor> selected_counters;
  int  base_rank;
  int  nranks;
  int  nGpus;
  bool enabled;
};

// ---- public API ---------------------------------------------------------

// Detect NIC type from IB device driver symlink
NicType NetCounterDetectNicType(const std::string& ib_device);

// Scan snapshots and return the dominant NIC type; sets mixed=true if types differ
NicType DetectNicTypeFromSnapshots(
    const std::vector<NetworkCounterSnapshot>& snaps, bool& mixed);

// Human-readable NIC type string
const char* NicTypeStr(NicType t);

// Check RCCL_TESTS_NET_COUNTER_ENABLE=1
bool NetCounterIsEnabled();

// Return counter list for the given NIC type
std::vector<CounterDescriptor> NetCounterGetCounterList(NicType nic_type);

// Parse NCCL_IB_HCA into IB device names (empty if unset)
std::vector<std::string> NetCounterParseIbHcaList();

// Device lookup helpers
std::string NetCounterFindIbDeviceForNic(const std::string& nic);
std::string NetCounterFindNicForIbDevice(const std::string& ib_device);

// Discover ethernet NICs on the local node (filtered by NIC prefix env vars)
void NetCounterGetNetworkInterfaces(std::vector<std::string>& interfaces);

// Snapshot selected counters for one device (NIC + IB name)
NetworkCounterSnapshot NetCounterCollectSnapshot(
    const std::string& nic,
    const std::string& ib_dev,
    const std::vector<CounterDescriptor>& selected);

// Convenience: auto-resolve IB device from NIC name
NetworkCounterSnapshot NetCounterCollectSnapshot(
    const std::string& nic,
    const std::vector<CounterDescriptor>& selected);

// Compute delta for a single counter between two snapshots.
// Returns false if the counter could not be read in either snapshot.
bool NetCounterComputeDelta(
    const NetworkCounterSnapshot& before,
    const NetworkCounterSnapshot& after,
    const CounterDescriptor& desc,
    uint64_t& delta);

// Print one table per node to stdout
void NetCounterPrintTable(
    const std::vector<std::string>& nic_names,
    const std::vector<NetworkCounterSnapshot>& before,
    const std::vector<NetworkCounterSnapshot>& after,
    int rank,
    const std::vector<CounterDescriptor>& selected);

#endif // __COLLECTOR_H__
