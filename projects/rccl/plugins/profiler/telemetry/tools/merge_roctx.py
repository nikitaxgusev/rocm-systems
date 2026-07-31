#!/usr/bin/env python3
"""EXACT GPU<->network correlation via roctx.
All three of {ncclAllReduce API, GPU kernels, per-collective roctx range} are
timestamped by rocprofiler-sdk on ONE clock -> no cross-clock estimation.
We only borrow tel_tx_bytes from the plugin trace, joined by collective order."""
import json, sys, glob
from collections import deque

rj   = sys.argv[1]     # rankN_results.json (rocprofiler)
telg = sys.argv[2]     # tel_N_*.json (plugin)
pid  = int(sys.argv[3])
out  = sys.argv[4]

d = json.load(open(rj))["rocprofiler-sdk-tool"][0]
ks = {s["kernel_id"]: s["kernel_name"] for s in d["kernel_symbols"]}
br = d["buffer_records"]

def short(n):
    for t in ("ncclDevKernel","copyBuffer","fillBuffer","prepareInput","prepareExpected","verify"):
        if t in n: return "AllReduce GPU kernel" if t=="ncclDevKernel" else t
    return n.split("(")[0][:40] or "kernel"

kernels = sorted((r["start_timestamp"], r["end_timestamp"], short(ks.get(r["dispatch_info"]["kernel_id"],"?")))
                 for r in br["kernel_dispatch"])

rccl_ops = [x for x in d["strings"]["buffer_records"] if x.get("kind")=="RCCL_API"][0]["operations"]
ar_op = rccl_ops.index("ncclAllReduce")
api = sorted((r["start_timestamp"], r["end_timestamp"])
             for r in br["rccl_api"] if r["operation"]==ar_op)

# roctx ranges = our per-collective markers, in rocprofiler clock
markers = sorted((r["start_timestamp"], r["end_timestamp"]) for r in br.get("marker_api",[]))

# plugin per-collective tel_tx, in seq (=chronological) order
ev = json.load(open(sorted(glob.glob(telg))[0]))
q = deque(); colls = []
for e in ev:
    if e.get("cat")!="COLL": continue
    if e["ph"]=="b": q.append(e)
    elif e["ph"]=="e" and q:
        b=q.popleft(); colls.append(b.get("args",{}))
# colls already in emission (seq) order; markers sorted by time == seq order
n = min(len(markers), len(colls))

allts = [k[0] for k in kernels] + [a[0] for a in api] + [m[0] for m in markers]
base = min(allts)
NS = 1000.0  # ns -> us

out_ev = [
 {"name":"process_name","ph":"M","pid":pid,"tid":0,"args":{"name":f"rank{pid}: API -> GPU -> network (rocprofiler clock, EXACT)"}},
 {"name":"thread_name","ph":"M","pid":pid,"tid":0,"args":{"name":"1) RCCL host API (ncclAllReduce)"}},
 {"name":"thread_name","ph":"M","pid":pid,"tid":1,"args":{"name":"2) GPU kernels (rocprofiler-sdk)"}},
 {"name":"thread_name","ph":"M","pid":pid,"tid":2,"args":{"name":"3) Collective + NIC tx (roctx + telemetry)"}},
]
for s,e in api:
    out_ev.append({"name":"ncclAllReduce","cat":"API","ph":"X","pid":pid,"tid":0,"ts":(s-base)/NS,"dur":max(1.0,(e-s)/NS)})
for s,e,name in kernels:
    out_ev.append({"name":name,"cat":"GPU","ph":"X","pid":pid,"tid":1,"ts":(s-base)/NS,"dur":max(1.0,(e-s)/NS)})
cum=0.0
for i in range(n):
    s,e = markers[i]; a = colls[i]
    mb = a.get("tel_tx_bytes",0)/1e6; msz=a.get("Count",0)*4
    out_ev.append({"name":f"AllReduce#{a.get('SeqNum',i)} {msz//1024//1024}MB tx={mb:.0f}MB",
                   "cat":"COLL","ph":"X","pid":pid,"tid":2,"ts":(s-base)/NS,"dur":max(1.0,(e-s)/NS),
                   "args":{"SeqNum":a.get("SeqNum"),"tel_tx_bytes":a.get("tel_tx_bytes",0),
                           "tel_wqe_sent":a.get("tel_wqe_sent",0),"Algorithm":a.get("Algorithm"),"msg_MB":msz/1e6}})
    out_ev.append({"name":"NIC_tx_MB","ph":"C","pid":pid,"tid":2,"ts":(s-base)/NS,"args":{"tx_this_coll_MB":round(mb,1)}})
    cum+=mb
    out_ev.append({"name":"NIC_tx_cumulative_MB","ph":"C","pid":pid,"tid":2,"ts":(s-base)/NS,"args":{"cum_MB":round(cum,1)}})

json.dump({"traceEvents":out_ev,"displayTimeUnit":"ns"}, open(out,"w"))
print("wrote",out)
print(f"api={len(api)} kernels={len(kernels)} markers={len(markers)} colls={len(colls)} paired={n}")
# report kernel-vs-marker overlap as an alignment sanity check (same clock => should overlap)
nk=[k for k in kernels if k[2]=="AllReduce GPU kernel"]
if nk and markers:
    ov=sum(1 for i in range(min(len(nk),len(markers)))
           if not (nk[i][1]<markers[i][0] or nk[i][0]>markers[i][1]))
    print(f"nccl kernels overlapping their marker: {ov}/{min(len(nk),len(markers))} (same-clock exactness check)")
