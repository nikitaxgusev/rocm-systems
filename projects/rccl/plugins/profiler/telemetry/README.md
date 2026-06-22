# RCCL Telemetry Profiler Plugin

A profiler plugin that attaches **per-collective network telemetry** to each RCCL
collective and emits a Perfetto/Chrome trace. It complements the whole-run network
telemetry engine (`src/transport/net_telemetry.*`) by attributing network activity
to the individual collective that caused it.

## What it records

For every collective (`ncclProfileColl`) the plugin emits a trace event whose
`args` carry, in addition to the usual func/algo/proto/count:

| field | meaning |
|---|---|
| `tel_tx_bytes`  | bytes transmitted by this collective's send WQEs (exact) |
| `tel_wqe_sent`  | number of send work requests |
| `tel_wqe_rcvd`  | number of receive work requests |

Attribution is **exact under pipelining**: each per-QP IB WQE event is mapped to
its owning collective through the profiler parent chain
`NetPlugin -> proxyStep -> proxyOp -> collective` (direction from `proxyOp->isSend`).
We do **not** subtract global device counters, which overcount ~4x when
collectives overlap. Validated on 32 ranks: sum of per-collective `tel_tx_bytes`
equals the whole-run telemetry flush `tx_bytes` exactly (ratio 1.0).

> Receive **byte** counts are not available per-collective: the net profiler event
> fires at recv-post time, before the RDMA receive size is known. We report the
> receive WQE count (`tel_wqe_rcvd`) but not rx bytes.

## Build

Configure RCCL with the plugin option (this also enables the per-QP net profiler
events the plugin depends on, `NCCL_ENABLE_NET_PROFILING`):

```bash
cmake -S . -B build -DBUILD_TELEMETRY_PROFILER=ON
cmake --build build --target rccl -j
```

Standalone plugin build (uses its own Makefile):

```bash
cd plugins/profiler/telemetry
make           # -> librccl-profiler-telemetry.so
```

## Run

```bash
export RCCL_TELEMETRY_ENABLE=1            # network telemetry engine
export NCCL_NET=IB-CAST                   # telemetry instruments the IB-CAST transport
export NCCL_PROFILER_PLUGIN=/path/to/librccl-profiler-telemetry.so
export NCCL_PROFILE_EVENT_MASK=218        # Coll+ProxyOp+ProxyStep+KernelCh+NetPlugin
export NCCL_PROFILE_DUMP_FILE=/path/out/trace
./all_reduce_perf -b 8M -e 256M -f 2 -g 1
# -> /path/out/trace_<commHash>_<rank>.json   (one per rank)
```

`NCCL_PROFILE_EVENT_MASK` **must** include ProxyOp/ProxyStep/KernelCh/NetPlugin
(218), not just Coll (2) — otherwise the Coll event closes before any network
traffic and the telemetry is empty.

## Post-processing (`tools/telemetry_trace.py`)

Filter, summarize and merge the per-rank traces into a single Perfetto timeline.
The tool scans only COLL lines, so it handles multi-GB traces quickly.

```bash
# Summary table (filterable)
tools/telemetry_trace.py summary 'out/trace_*.json'
tools/telemetry_trace.py summary 'out/trace_*.json' --op AllReduce --min-bytes 16777216

# Phase 2 — filter to a JSON list of matching collectives
tools/telemetry_trace.py filter 'out/trace_*.json' -o picked.json \
    --op AllReduce --min-bytes 16777216 --proto LL128

# Phase 3 — merge all ranks into one Perfetto trace (track per rank) +
#           per-SeqNum straggler report
tools/telemetry_trace.py merge 'out/trace_*.json' -o merged.json \
    --straggler stragglers.json
# open merged.json in https://ui.perfetto.dev
```

Filters (combine with AND): `--op`, `--proto`, `--algo`, `--min-bytes`,
`--max-bytes`, `--min-tx`.

The straggler report ranks collectives by the spread (max-min) of per-rank
duration for the same `SeqNum`, highlighting load imbalance (e.g. one slow rank
holding up a collective).
