/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*************************************************************************
 * Network Counter Collector – implementation
 *
 * See collector.h for the public API and environment variable reference.
 *************************************************************************/
#include "collector.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <set>
#include <unistd.h>

// =====================================================================
// Per-NIC-type counter tables
// =====================================================================
//
// Each NIC type has its own table with actual hardware key names.
// name      : display name (table column header; post-processing keys on it)
// is_prefix : true → ethtool keys are prefixes, all matching keys are summed
// src       : (source, key) alternatives tried in order, NULL key terminates.
//             A key containing "%d" is summed over priorities 0..7.

#define COUNTER_MAX_SOURCES 4

struct CounterTableSource {
  CounterSource source;
  const char*   key;
};

struct CounterTableEntry {
  const char*        name;
  bool               is_prefix;
  CounterTableSource src[COUNTER_MAX_SOURCES];
};

#define IB(k)  {COUNTER_SRC_IB_HW,   (k)}
#define ETH(k) {COUNTER_SRC_ETHTOOL, (k)}
#define DBG(k) {COUNTER_SRC_DEBUGFS, (k)}

// ----- IONIC (AINIC) counter table -----
static const CounterTableEntry ionic_table[] = {
  // ethtool – PFC frame counters
  {"frames_rx_pripause",       true,  {ETH("frames_rx_pripause")}},
  {"frames_tx_pripause",       true,  {ETH("frames_tx_pripause")}},
  // IB hw_counters – CNP
  {"rx_rdma_cnp_pkts",         false, {IB("rx_rdma_cnp_pkts")}},
  {"tx_rdma_cnp_pkts",         false, {IB("tx_rdma_cnp_pkts")}},
  // IB hw_counters – errors & retransmits
  {"rx_rdma_mtu_discard_pkts", false, {IB("rx_rdma_mtu_discard_pkts")}},
  {"resp_rx_outof_buf",        false, {IB("resp_rx_outof_buf")}},
  {"tx_rdma_ack_timeout",      false, {IB("tx_rdma_ack_timeout")}},
  {"req_tx_retry_excd_err",    false, {IB("req_tx_retry_excd_err")}},
  {"resp_rx_outouf_seq",       false, {IB("resp_rx_outouf_seq")}},
  {"req_rx_pkt_seq_err",       false, {IB("req_rx_pkt_seq_err")}},
  // IB hw_counters – RDMA throughput
  {"tx_rdma_ucast_bytes",      false, {IB("tx_rdma_ucast_bytes")}},
  {"rx_rdma_ucast_bytes",      false, {IB("rx_rdma_ucast_bytes")}},
  {"tx_rdma_ucast_pkts",       false, {IB("tx_rdma_ucast_pkts")}},
  {"rx_rdma_ucast_pkts",       false, {IB("rx_rdma_ucast_pkts")}},
};
static const int ionic_table_size = (int)(sizeof(ionic_table) / sizeof(ionic_table[0]));

// ----- BNXT_RE (Thor2) counter table -----
// bnxt_re sysfs names differ between the upstream kernel driver and the
// Broadcom out-of-tree driver (e.g. to_retransmits vs local_ack_timeout_err,
// seq_err_naks_rcvd vs packet_seq_err), so both are listed. debugfs needs root
// and is only the last resort.
static const CounterTableEntry bnxt_re_table[] = {
  // ethtool – PFC frame counters, summed over priorities
  {"rx_pfc_ena_frames_pri",   true,  {ETH("rx_pfc_ena_frames_pri")}},
  {"tx_pfc_ena_frames_pri",   true,  {ETH("tx_pfc_ena_frames_pri")}},
  // ethtool – PFC transition counters for the RoCE priority
  {"pfc_pri3_rx_transitions", false, {ETH("pfc_pri3_rx_transitions")}},
  {"pfc_pri3_tx_transitions", false, {ETH("pfc_pri3_tx_transitions")}},
  // IB hw_counters – CNP
  {"rx_cnp_pkts",             false, {IB("rx_cnp_pkts"), IB("rp_cnp_handled")}},
  {"tx_cnp_pkts",             false, {IB("tx_cnp_pkts"), IB("np_cnp_sent")}},
  {"rx_roce_discards",        false, {IB("rx_roce_discards"), IB("rx_discards")}},
  // retransmit / drop counters: sysfs first, then ethtool, debugfs last
  {"rx_stat_discards",        false, {ETH("rx_stat_discard"), DBG("rx_stat_discards")}},
  {"to_retransmits",          false, {IB("to_retransmits"), IB("local_ack_timeout_err"),
                                      DBG("to_retransmits")}},
  {"max_retry_exceeded",      false, {IB("max_retry_exceeded"), DBG("max_retry_exceeded")}},
  {"oos_drop_count",          false, {IB("res_oos_drop_count"), IB("oos_drop_count"),
                                      IB("out_of_sequence"), DBG("res_oos_drop_count")}},
  {"seq_err_naks_rcvd",       false, {IB("seq_err_naks_rcvd"), IB("packet_seq_err"),
                                      DBG("seq_err_naks_rcvd")}},
  // ethtool – PFC transitions summed over all priorities
  {"pfc_rx_transitions",      false, {ETH("pfc_pri%d_rx_transitions")}},
  {"pfc_tx_transitions",      false, {ETH("pfc_pri%d_tx_transitions")}},
};
static const int bnxt_re_table_size = (int)(sizeof(bnxt_re_table) / sizeof(bnxt_re_table[0]));

#undef IB
#undef ETH
#undef DBG

static void GetTable(NicType nic_type,
                     const CounterTableEntry*& table, int& size) {
  switch (nic_type) {
    case NIC_IONIC:
      table = ionic_table;
      size = ionic_table_size;
      break;
    case NIC_BNXT_RE:
      table = bnxt_re_table;
      size = bnxt_re_table_size;
      break;
    case NIC_UNKNOWN:
      fprintf(stderr,
              "# Warning: NIC type unknown, defaulting to bnxt_re counter set\n");
      table = bnxt_re_table;
      size = bnxt_re_table_size;
      break;
  }
}

// =====================================================================
// NIC prefix helpers (for fallback interface discovery)
// =====================================================================

static const char* NicPrefix() {
  static const char* prefix = NULL;
  static bool checked = false;
  if (!checked) {
    prefix = getenv("RCCL_TESTS_NET_COUNTER_NIC_PREFIX");
    checked = true;
  }
  return prefix;
}

// =====================================================================
// Public helpers
// =====================================================================

static const std::set<std::string>& RequestedCounterFilter() {
  static std::set<std::string> filter;
  static bool checked = false;
  if (!checked) {
    const char* env = getenv("RCCL_TESTS_NIC_COUNTER_LIST");
    if (env && strlen(env) > 0) {
      std::string list(env);
      size_t pos = 0;
      while (pos < list.size()) {
        size_t comma = list.find(',', pos);
        if (comma == std::string::npos) { comma = list.size(); }
        std::string token = list.substr(pos, comma - pos);
        while (!token.empty() && (token.front() == ' ' || token.front() == '\t')) {
          token.erase(token.begin());
        }
        while (!token.empty() && (token.back() == ' ' || token.back() == '\t')) {
          token.pop_back();
        }
        if (!token.empty()) { filter.insert(token); }
        pos = comma + 1;
      }
    }
    checked = true;
  }
  return filter;
}

static bool CounterMatchesRequestedFilter(const CounterTableEntry& entry) {
  const std::set<std::string>& filter = RequestedCounterFilter();
  if (filter.empty()) { return true; }
  if (filter.count(entry.name) > 0) { return true; }
  for (int s = 0; s < COUNTER_MAX_SOURCES && entry.src[s].key; s++) {
    if (filter.count(entry.src[s].key) > 0) { return true; }
  }
  return false;
}

bool NetCounterIsEnabled() {
  static int enabled = -1;
  if (enabled == -1) {
    const char* env = getenv("RCCL_TESTS_NET_COUNTER_ENABLE");
    enabled = (env && atoi(env) > 0) ? 1 : 0;
  }
  return enabled == 1;
}

std::vector<CounterDescriptor> NetCounterGetCounterList(NicType nic_type) {
  std::vector<CounterDescriptor> result;
  const CounterTableEntry* table;
  int size;
  GetTable(nic_type, table, size);
  for (int i = 0; i < size; i++) {
    if (!CounterMatchesRequestedFilter(table[i])) continue;
    CounterDescriptor d;
    d.name = table[i].name;
    d.is_prefix = table[i].is_prefix;
    for (int s = 0; s < COUNTER_MAX_SOURCES && table[i].src[s].key; s++) {
      d.sources.push_back({table[i].src[s].source, table[i].src[s].key});
    }
    result.push_back(d);
  }
  return result;
}

std::vector<std::string> NetCounterParseIbHcaList() {
  std::vector<std::string> result;

  const char* env = getenv("NCCL_IB_HCA");
  if (!env || strlen(env) == 0) { return result; }

  std::string list(env);

  size_t pos = 0;
  while (pos < list.size()) {
    size_t comma = list.find(',', pos);
    if (comma == std::string::npos) { comma = list.size(); }

    std::string token = list.substr(pos, comma - pos);

    while (!token.empty() && (token.front() == ' ' || token.front() == '\t')) {
      token.erase(token.begin());
    }

    while (!token.empty() && (token.back() == ' ' || token.back() == '\t')) {
      token.pop_back();
    }

    if (!token.empty()) { result.push_back(token); }

    pos = comma + 1;
  }
  return result;
}

// =====================================================================
// IB device name parsing
// =====================================================================

// Split "rdma0:2" -> base="rdma0", port=2.  Default port is 1.
static void ParseIbDevice(const std::string& ib_device,
                           std::string& base, int& port) {
  size_t colon = ib_device.find(':');
  if (colon == std::string::npos) {
    base = ib_device;
    port = 1;
  } else {
    base = ib_device.substr(0, colon);
    port = atoi(ib_device.c_str() + colon + 1);
    if (port <= 0) port = 1;
  }
}

// =====================================================================
// Device lookup
// =====================================================================

static std::string RunShellOneLiner(const char* cmd) {
  FILE* fp = popen(cmd, "r");
  if (!fp) { return ""; }
  char buf[256] = {0};
  if (fgets(buf, sizeof(buf), fp)) {
    size_t len = strlen(buf);
    if (len > 0 && buf[len-1] == '\n') {
      buf[len-1] = '\0';
    }
  }
  pclose(fp);
  return std::string(buf);
}

std::string NetCounterFindIbDeviceForNic(const std::string& nic) {
  char cmd[512] = {0};
  snprintf(cmd, sizeof(cmd),
           "ls /sys/class/net/%s/device/infiniband 2>/dev/null | head -1",
           nic.c_str());
  return RunShellOneLiner(cmd);
}

std::string NetCounterFindNicForIbDevice(const std::string& ib_device) {
  std::string ib_base;
  int port;
  ParseIbDevice(ib_device, ib_base, port);
  (void)port;
  char cmd[512] = {0};
  snprintf(cmd, sizeof(cmd),
           "ls /sys/class/infiniband/%s/device/net 2>/dev/null | head -1",
           ib_base.c_str());
  return RunShellOneLiner(cmd);
}

void NetCounterGetNetworkInterfaces(std::vector<std::string>& interfaces) {
  FILE* fp = popen("ls /sys/class/net 2>/dev/null", "r");
  if (fp == NULL) { return; }

  const char* prefix = NicPrefix();

  char path[256] = {0};
  while (fgets(path, sizeof(path), fp) != NULL) {
    if (path[strlen(path)-1] == '\n') {
      path[strlen(path)-1] = '\0';
    }
    if (prefix) {
      if (strncmp(path, prefix, strlen(prefix)) == 0) {
        interfaces.push_back(std::string(path));
      }
    } else {
      if (strcmp(path, "lo") != 0 &&
          strncmp(path, "veth", 4) != 0 &&
          strncmp(path, "fenic", 5) != 0) {
        interfaces.push_back(std::string(path));
      }
    }
  }
  pclose(fp);
}

// =====================================================================
// NIC type detection
// =====================================================================

NicType NetCounterDetectNicType(const std::string& ib_device) {
  if (ib_device.empty()) return NIC_UNKNOWN;

  std::string ib_base;
  int port;
  ParseIbDevice(ib_device, ib_base, port);
  (void)port;

  if (ib_base.compare(0, 7, "bnxt_re") == 0) return NIC_BNXT_RE;

  char link[4096] = {0};
  std::string driver_path =
      "/sys/class/infiniband/" + ib_base + "/device/driver";
  ssize_t len = readlink(driver_path.c_str(), link, sizeof(link) - 1);
  if (len <= 0) return NIC_UNKNOWN;
  link[len] = '\0';

  const char* drv = strrchr(link, '/');
  drv = drv ? drv + 1 : link;

  if (strcmp(drv, "bnxt_re") == 0 || strcmp(drv, "bnxt_en") == 0)
    return NIC_BNXT_RE;
  if (strcmp(drv, "ionic") == 0)
    return NIC_IONIC;
  return NIC_UNKNOWN;
}

const char* NicTypeStr(NicType t) {
  switch (t) {
    case NIC_BNXT_RE: return "BNXT_RE";
    case NIC_IONIC:   return "IONIC";
    default:          return "UNKNOWN";
  }
}

NicType DetectNicTypeFromSnapshots(
    const std::vector<NetworkCounterSnapshot>& snaps, bool& mixed) {
  NicType nic_type = NIC_UNKNOWN;
  mixed = false;
  for (size_t i = 0; i < snaps.size(); i++) {
    if (snaps[i].nic_type == NIC_UNKNOWN) continue;
    if (nic_type == NIC_UNKNOWN) {
      nic_type = snaps[i].nic_type;
    } else if (snaps[i].nic_type != nic_type) {
      mixed = true;
      break;
    }
  }
  return nic_type;
}

// =====================================================================
// Per-source collection
// =====================================================================

typedef std::map<std::string, uint64_t> RawCounterMap;

static void ReadEthtoolCounters(const std::string& nic, RawCounterMap& out) {
  if (nic.empty()) { return; }

  char command[512] = {0};
  snprintf(command, sizeof(command), "ethtool -S %s 2>/dev/null", nic.c_str());

  FILE* fp = popen(command, "r");
  if (!fp) { return; }

  char line[256] = {0};
  while (fgets(line, sizeof(line), fp)) {
    char key[256] = {0};
    unsigned long long tmp = 0;
    if (sscanf(line, "%255[^:]: %llu", key, &tmp) == 2) {
      char* start = key;
      while (*start == ' ' || *start == '\t') { start++; }
      out[std::string(start)] = (uint64_t)tmp;
    }
  }
  pclose(fp);
}

static void ReadDebugfsCounters(const std::string& ib_base, RawCounterMap& out) {
  if (ib_base.empty()) { return; }

  std::string info_path = "/sys/kernel/debug/bnxt_re/" + ib_base + "/info";
  FILE* fp = fopen(info_path.c_str(), "r");
  if (!fp) { return; }

  char line[512] = {0};
  while (fgets(line, sizeof(line), fp)) {
    unsigned long long tmp = 0;
    char key[256] = {0};
    if (sscanf(line, " %255[^:=] %*[:=] %llu", key, &tmp) == 2 ||
        sscanf(line, " %255s %llu", key, &tmp) == 2) {
      char* start = key;
      while (*start == ' ' || *start == '\t') { start++; }
      char* end = start + strlen(start) - 1;
      while (end > start && (*end == ' ' || *end == '\t')) { *end-- = '\0'; }
      out[std::string(start)] = (uint64_t)tmp;
    }
  }
  fclose(fp);
}

static bool ReadIbHwCounter(const std::string& base_port,
                            const std::string& base_dev,
                            const std::string& key, uint64_t& value) {
  FILE* fp = fopen((base_port + key).c_str(), "r");
  if (!fp) fp = fopen((base_dev + key).c_str(), "r");
  if (!fp) { return false; }
  unsigned long long tmp = 0;
  bool ok = fscanf(fp, "%llu", &tmp) == 1;
  fclose(fp);
  if (ok) { value = (uint64_t)tmp; }
  return ok;
}

// Looks up an exact key, a "%d" pattern (summed over priorities 0..7) or,
// with is_prefix, every key starting with 'key' (summed).
static bool LookupRawCounter(const RawCounterMap& raw, const std::string& key,
                             bool is_prefix, uint64_t& value) {
  bool found = false;
  uint64_t sum = 0;
  if (key.find("%d") != std::string::npos) {
    for (int prio = 0; prio < 8; prio++) {
      char name[256];
      snprintf(name, sizeof(name), key.c_str(), prio);
      auto it = raw.find(name);
      if (it != raw.end()) { sum += it->second; found = true; }
    }
  } else if (is_prefix) {
    for (auto it = raw.lower_bound(key);
         it != raw.end() && it->first.compare(0, key.size(), key) == 0; ++it) {
      sum += it->second;
      found = true;
    }
  } else {
    auto it = raw.find(key);
    if (it != raw.end()) { sum = it->second; found = true; }
  }
  if (found) { value = sum; }
  return found;
}

static bool NeedsSource(const std::vector<CounterDescriptor>& selected,
                        CounterSource source) {
  for (const auto& d : selected) {
    for (const auto& s : d.sources) {
      if (s.source == source) { return true; }
    }
  }
  return false;
}

// =====================================================================
// Snapshot collection
// =====================================================================

NetworkCounterSnapshot NetCounterCollectSnapshot(
    const std::string& nic,
    const std::string& ib_dev,
    const std::vector<CounterDescriptor>& selected) {
  NetworkCounterSnapshot snap;

  memset(snap.nic_name, 0, sizeof(snap.nic_name));
  memset(snap.ib_device, 0, sizeof(snap.ib_device));

  if (!nic.empty()) {
    strncpy(snap.nic_name, nic.c_str(), sizeof(snap.nic_name) - 1);
  }

  if (!ib_dev.empty()) {
    strncpy(snap.ib_device, ib_dev.c_str(), sizeof(snap.ib_device) - 1);
  }

  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  snap.timestamp_us = (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
  snap.nic_type = NetCounterDetectNicType(ib_dev);

  std::string ib_base;
  int ib_port = 1;
  if (!ib_dev.empty()) { ParseIbDevice(ib_dev, ib_base, ib_port); }
  char port_str[16];
  snprintf(port_str, sizeof(port_str), "%d", ib_port);
  std::string base_port =
      "/sys/class/infiniband/" + ib_base + "/ports/" + port_str + "/hw_counters/";
  std::string base_dev = "/sys/class/infiniband/" + ib_base + "/hw_counters/";

  RawCounterMap ethtool, debugfs;
  if (NeedsSource(selected, COUNTER_SRC_ETHTOOL)) { ReadEthtoolCounters(nic, ethtool); }
  bool debugfs_read = false;

  for (const auto& d : selected) {
    for (const auto& s : d.sources) {
      uint64_t value = 0;
      bool found = false;
      switch (s.source) {
        case COUNTER_SRC_IB_HW:
          if (ib_base.empty()) { break; }
          if (s.key.find("%d") != std::string::npos) {
            uint64_t sum = 0;
            for (int prio = 0; prio < 8; prio++) {
              char name[256];
              snprintf(name, sizeof(name), s.key.c_str(), prio);
              uint64_t v = 0;
              if (ReadIbHwCounter(base_port, base_dev, name, v)) { sum += v; found = true; }
            }
            value = sum;
          } else {
            found = ReadIbHwCounter(base_port, base_dev, s.key, value);
          }
          break;
        case COUNTER_SRC_ETHTOOL:
          found = LookupRawCounter(ethtool, s.key, d.is_prefix, value);
          break;
        case COUNTER_SRC_DEBUGFS:
          if (!debugfs_read) {
            ReadDebugfsCounters(ib_base, debugfs);
            debugfs_read = true;
          }
          found = LookupRawCounter(debugfs, s.key, false, value);
          break;
      }
      if (found) {
        snap.counters[d.name] = value;
        break;
      }
    }
  }

  return snap;
}

NetworkCounterSnapshot NetCounterCollectSnapshot(
    const std::string& nic,
    const std::vector<CounterDescriptor>& selected) {
  return NetCounterCollectSnapshot(nic, NetCounterFindIbDeviceForNic(nic),
                                   selected);
}

// =====================================================================
// Delta computation
// =====================================================================

bool NetCounterComputeDelta(
    const NetworkCounterSnapshot& before,
    const NetworkCounterSnapshot& after,
    const CounterDescriptor& desc,
    uint64_t& delta) {
  delta = 0;
  auto b = before.counters.find(desc.name);
  auto a = after.counters.find(desc.name);
  if (b == before.counters.end() || a == after.counters.end()) { return false; }
  delta = a->second - b->second;
  return true;
}

// =====================================================================
// Table printer
// =====================================================================

void NetCounterPrintTable(
    const std::vector<std::string>& nic_names,
    const std::vector<NetworkCounterSnapshot>& before,
    const std::vector<NetworkCounterSnapshot>& after,
    int rank,
    const std::vector<CounterDescriptor>& selected) {
  size_t num_nics     = nic_names.size();
  size_t num_counters = selected.size();

  char hostname[256] = {0};

  if (gethostname(hostname, sizeof(hostname)) != 0) {
    strncpy(hostname, "unknown", sizeof(hostname));
  }

  hostname[sizeof(hostname)-1] = '\0';

  if (num_counters == 0 || num_nics == 0) {
    printf("NET_COUNTER_TABLE: node=%s rank=%d status=NO_DATA\n",
           hostname, rank);
    fflush(stdout);
    return;
  }

  bool mixed = false;
  NicType nic_type = DetectNicTypeFromSnapshots(before, mixed);

  // Pre-compute deltas, rates, durations
  std::vector<std::vector<uint64_t>> deltas(
      num_nics, std::vector<uint64_t>(num_counters, 0));

  std::vector<std::vector<double>> rates(
      num_nics, std::vector<double>(num_counters, 0.0));

  std::vector<std::vector<bool>> avail(
      num_nics, std::vector<bool>(num_counters, false));

  std::vector<double> dur_sec(num_nics);

  for (size_t n = 0; n < num_nics; n++) {
    dur_sec[n] = (after[n].timestamp_us - before[n].timestamp_us) / 1e6;
    for (size_t c = 0; c < num_counters; c++) {
      avail[n][c] = NetCounterComputeDelta(before[n], after[n], selected[c], deltas[n][c]);
      rates[n][c] = (dur_sec[n] > 0.0)
                        ? (double)deltas[n][c] / dur_sec[n]
                        : 0.0;
    }
  }

  // Display labels: prefer "IB(NIC)" when both are known
  std::vector<std::string> labels(num_nics);

  for (size_t n = 0; n < num_nics; n++) {
    if (before[n].ib_device[0] != '\0' && before[n].nic_name[0] != '\0') {
      labels[n] = std::string(before[n].ib_device)
                  + "(" + before[n].nic_name + ")";
    } else if (before[n].ib_device[0] != '\0') {
      labels[n] = before[n].ib_device;
    } else {
      labels[n] = before[n].nic_name;
    }
  }

  // Column widths
  int nic_w = 6; // strlen("Device")
  for (const auto& lbl : labels) {
    nic_w = std::max(nic_w, (int)lbl.size());
  }

  nic_w += 2;

  std::vector<int> cnt_w(num_counters, 5);
  std::vector<int> rt_w(num_counters, 6);

  for (size_t c = 0; c < num_counters; c++) {
    for (size_t n = 0; n < num_nics; n++) {
      char buf[32];
      snprintf(buf, sizeof(buf), "%llu", (unsigned long long)deltas[n][c]);
      cnt_w[c] = std::max(cnt_w[c], (int)strlen(buf));
      snprintf(buf, sizeof(buf), "%.2f", rates[n][c]);
      rt_w[c] = std::max(rt_w[c], (int)strlen(buf));
    }
    int name_len = (int)selected[c].name.size();
    int pair_w   = cnt_w[c] + 2 + rt_w[c];
    if (name_len > pair_w) {
      rt_w[c] += (name_len - pair_w);
    }
  }

  // Separator
  std::string sep(nic_w, '-');
  for (size_t c = 0; c < num_counters; c++) {
    sep += "--";
    sep += std::string(cnt_w[c], '-');
    sep += "--";
    sep += std::string(rt_w[c], '-');
  }

  // Print
  printf("\nNET_COUNTER_TABLE: node=%s  rank=%d  duration_sec=%.3f  nic_type=%s\n",
         hostname, rank, dur_sec[0], mixed ? "MIXED" : NicTypeStr(nic_type));
  printf("%s\n", sep.c_str());

  printf("%-*s", nic_w, "Device");
  for (size_t c = 0; c < num_counters; c++) {
    int pair_w = cnt_w[c] + 2 + rt_w[c];
    printf("  %-*s", pair_w, selected[c].name.c_str());
  }
  printf("\n");

  printf("%-*s", nic_w, "");
  for (size_t c = 0; c < num_counters; c++) {
    printf("  %*s  %*s", cnt_w[c], "count", rt_w[c], "rate/s");
  }
  printf("\n");

  printf("%s\n", sep.c_str());

  for (size_t n = 0; n < num_nics; n++) {
    printf("%-*s", nic_w, labels[n].c_str());
    for (size_t c = 0; c < num_counters; c++) {
      if (avail[n][c]) {
        printf("  %*llu  %*.2f", cnt_w[c], (unsigned long long)deltas[n][c], rt_w[c], rates[n][c]);
      } else {
        // Two tokens so whitespace-split post-processing keeps column pairs aligned.
        printf("  %*s  %*s", cnt_w[c], "N/A", rt_w[c], "N/A");
      }
    }
    printf("\n");
  }

  printf("%s\n\n", sep.c_str());
  fflush(stdout);
}
