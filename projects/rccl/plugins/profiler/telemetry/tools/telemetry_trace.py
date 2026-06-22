#!/usr/bin/env python3
"""
RCCL telemetry trace post-processor (Phases 2 & 3).

Consumes the per-rank Chrome/Perfetto JSON traces emitted by the telemetry
profiler plugin (NCCL_PROFILE_DUMP_FILE -> trace_<commHash>_<rank>.json) and:

  Phase 2 (filter): select COLL operations by op type / message size / tx bytes,
                    dropping everything that doesn't match.
  Phase 3 (merge) : combine N per-rank traces into ONE Perfetto/Chrome trace
                    with a track per (pid,rank), per-collective duration slices
                    carrying the telemetry args, plus a per-SeqNum straggler
                    report (min/max/spread of duration across ranks).

The merged JSON loads directly in https://ui.perfetto.dev (Open trace file).

Usage:
  telemetry_trace.py merge   <trace_glob> -o merged.json [filters] [--straggler stragglers.json]
  telemetry_trace.py filter  <trace_glob> -o filtered.json [filters]
  telemetry_trace.py summary <trace_glob> [filters]

Filters (all optional, combine with AND):
  --op AllReduce,AllGather       only these collective names
  --min-bytes N                  COLL message size (Count*dtype) >= N
  --max-bytes N                  COLL message size <= N
  --min-tx N                     tel_tx_bytes >= N
  --proto LL128,SIMPLE           only these protocols
  --algo TREE,RING               only these algorithms
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict

DTYPE_BYTES = {
    "ncclInt8":1,"ncclChar":1,"ncclUint8":1,"ncclInt32":4,"ncclInt":4,
    "ncclUint32":4,"ncclInt64":8,"ncclUint64":8,"ncclFloat16":2,"ncclHalf":2,
    "ncclFloat32":4,"ncclFloat":4,"ncclFloat64":8,"ncclDouble":8,
    "ncclBfloat16":2,"ncclFp8E4M3":1,"ncclFp8E5M2":1,
}

# These traces can be hundreds of MB (one event per WQE under NCCL_ENABLE_NET_
# PROFILING). COLL events are a tiny fraction, so we scan line-by-line and only
# JSON-parse the lines whose cat is COLL — orders of magnitude faster than
# loading the whole array.
_COLL_LINE = re.compile(r'"cat":\s*"COLL"')

def collectives(path):
    """Stream a plugin trace file; pair COLL begin/end by id with telemetry."""
    begins = {}
    out = []
    with open(path) as f:
        for line in f:
            if not _COLL_LINE.search(line):
                continue
            s = line.strip().rstrip(",")
            try:
                e = json.loads(s)
            except json.JSONDecodeError:
                continue
            eid = e.get("id")
            if e.get("ph") == "b":
                begins[eid] = e
            elif e.get("ph") == "e" and eid in begins:
                b = begins.pop(eid)
                a = b.get("args", {})
                cnt = a.get("Count", 0)
                dt = a.get("Datatype", "")
                msg_bytes = cnt * DTYPE_BYTES.get(dt, 1)
                out.append({
                    "name": b.get("name"),
                    "pid": b.get("pid"),
                    "rank": a.get("Rank"),
                    "seq": a.get("SeqNum"),
                    "commHash": a.get("CommHash"),
                    "count": cnt,
                    "datatype": dt,
                    "algo": a.get("Algorithm"),
                    "proto": a.get("Protocol"),
                    "nChannels": a.get("nChannels"),
                    "msg_bytes": msg_bytes,
                    "tel_tx_bytes": a.get("tel_tx_bytes", 0),
                    "tel_wqe_sent": a.get("tel_wqe_sent", 0),
                    "tel_wqe_rcvd": a.get("tel_wqe_rcvd", 0),
                    "start_us": b.get("ts", 0.0),
                    "stop_us": e.get("ts", 0.0),
                    "dur_us": e.get("ts", 0.0) - b.get("ts", 0.0),
                })
    return out

def make_filter(args):
    ops = set(args.op.split(",")) if args.op else None
    protos = set(args.proto.split(",")) if args.proto else None
    algos = set(args.algo.split(",")) if args.algo else None
    def keep(c):
        if ops is not None and c["name"] not in ops: return False
        if protos is not None and c["proto"] not in protos: return False
        if algos is not None and c["algo"] not in algos: return False
        if args.min_bytes is not None and c["msg_bytes"] < args.min_bytes: return False
        if args.max_bytes is not None and c["msg_bytes"] > args.max_bytes: return False
        if args.min_tx is not None and c["tel_tx_bytes"] < args.min_tx: return False
        return True
    return keep

def gather(trace_glob, keep):
    files = sorted(glob.glob(trace_glob))
    if not files:
        sys.exit(f"no trace files match: {trace_glob}")
    all_colls = []
    for fp in files:
        for c in collectives(fp):
            c["file"] = os.path.basename(fp)
            if keep(c):
                all_colls.append(c)
    return files, all_colls

def cmd_summary(args):
    files, colls = gather(args.trace_glob, make_filter(args))
    print(f"files: {len(files)}  matched collectives: {len(colls)}")
    by_op = defaultdict(lambda: [0,0,0])  # count, tx, dur
    for c in colls:
        b = by_op[(c["name"], c["msg_bytes"])]
        b[0]+=1; b[1]+=c["tel_tx_bytes"]; b[2]+=c["dur_us"]
    print(f"{'op':>12} {'msg_bytes':>12} {'n':>6} {'sum_tx':>16} {'avg_dur_us':>12}")
    for (op,mb),(n,tx,dur) in sorted(by_op.items()):
        print(f"{op:>12} {mb:>12,} {n:>6} {tx:>16,} {dur/n:>12.1f}")

def cmd_filter(args):
    files, colls = gather(args.trace_glob, make_filter(args))
    with open(args.out, "w") as f:
        json.dump(colls, f, indent=2)
    print(f"wrote {len(colls)} collectives -> {args.out}")

def cmd_merge(args):
    files, colls = gather(args.trace_glob, make_filter(args))
    # Assign one Perfetto track (pid/tid) per (commHash,rank). Perfetto groups
    # async/duration events by pid; we use a synthetic pid per rank for clean lanes.
    rank_key = lambda c: (c["commHash"], c["rank"])
    ranks = sorted({rank_key(c) for c in colls})
    rank_pid = {rk: i for i, rk in enumerate(ranks)}
    # normalize timeline to start at 0
    t0 = min((c["start_us"] for c in colls), default=0.0)
    out = []
    # process/track names
    for rk, pid in rank_pid.items():
        out.append({"name":"process_name","ph":"M","pid":pid,"tid":0,
                    "args":{"name":f"comm{rk[0]&0xffff:x}_rank{rk[1]}"}})
    # duration slices (ph X) with telemetry args
    for c in colls:
        pid = rank_pid[rank_key(c)]
        out.append({
            "name": f"{c['name']} {c['msg_bytes']//1024}KiB",
            "cat": "COLL", "ph": "X", "pid": pid, "tid": 0,
            "ts": c["start_us"] - t0, "dur": max(c["dur_us"], 0.0),
            "args": {
                "SeqNum": c["seq"], "Rank": c["rank"], "Algorithm": c["algo"],
                "Protocol": c["proto"], "Count": c["count"], "Datatype": c["datatype"],
                "msg_bytes": c["msg_bytes"], "tel_tx_bytes": c["tel_tx_bytes"],
                "tel_wqe_sent": c["tel_wqe_sent"], "tel_wqe_rcvd": c["tel_wqe_rcvd"],
                "dur_us": round(c["dur_us"],2),
            },
        })
    with open(args.out, "w") as f:
        json.dump({"traceEvents": out, "displayTimeUnit":"ns"}, f)
    print(f"merged {len(colls)} collectives from {len(files)} ranks -> {args.out}")
    print(f"open in https://ui.perfetto.dev (Open trace file)")

    # Straggler analysis: per (commHash, SeqNum) spread of duration across ranks.
    if args.straggler:
        groups = defaultdict(list)
        for c in colls:
            groups[(c["commHash"], c["seq"])].append(c)
        report = []
        for (ch, seq), members in sorted(groups.items()):
            durs = sorted(m["dur_us"] for m in members)
            if not durs: continue
            dmin, dmax = durs[0], durs[-1]
            davg = sum(durs)/len(durs)
            slow = max(members, key=lambda m: m["dur_us"])
            report.append({
                "commHash": ch, "seqNum": seq, "name": slow["name"],
                "msg_bytes": slow["msg_bytes"], "ranks": len(members),
                "dur_min_us": round(dmin,2), "dur_max_us": round(dmax,2),
                "dur_avg_us": round(davg,2),
                "spread_us": round(dmax-dmin,2),
                "spread_pct": round(100.0*(dmax-dmin)/davg,1) if davg else 0,
                "slowest_rank": slow["rank"],
            })
        report.sort(key=lambda r: r["spread_us"], reverse=True)
        with open(args.straggler, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nstraggler report -> {args.straggler}  (top 5 by spread):")
        print(f"{'seq':>6} {'op':>12} {'msg_bytes':>12} {'min_us':>10} {'max_us':>10} {'spread%':>8} {'slow_rank':>9}")
        for r in report[:5]:
            print(f"{r['seqNum']:>6} {r['name']:>12} {r['msg_bytes']:>12,} "
                  f"{r['dur_min_us']:>10.1f} {r['dur_max_us']:>10.1f} {r['spread_pct']:>8.1f} {r['slowest_rank']:>9}")

def main():
    p = argparse.ArgumentParser(description="RCCL telemetry trace post-processor")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("summary","filter","merge"):
        sp = sub.add_parser(name)
        sp.add_argument("trace_glob")
        sp.add_argument("--op")
        sp.add_argument("--proto")
        sp.add_argument("--algo")
        sp.add_argument("--min-bytes", type=int)
        sp.add_argument("--max-bytes", type=int)
        sp.add_argument("--min-tx", type=int)
        if name in ("filter","merge"):
            sp.add_argument("-o","--out", required=True)
        if name == "merge":
            sp.add_argument("--straggler")
    args = p.parse_args()
    {"summary":cmd_summary,"filter":cmd_filter,"merge":cmd_merge}[args.cmd](args)

if __name__ == "__main__":
    main()
