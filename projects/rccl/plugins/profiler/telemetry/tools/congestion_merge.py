#!/usr/bin/env python3
"""Build a congestion timeline (Perfetto JSON + PNG) from RCCL telemetry
hw_samples. Aggregates per host (shared CLOCK_MONOTONIC domain) across all
ranks/NICs into rate-vs-time tracks: NIC tx throughput plus RoCE congestion
counters (ECN marks, CNP sent/handled, packet seq err, out-of-sequence,
out-of-buffer)."""
import json, glob, os, sys
from collections import defaultdict

CONG = ['np_ecn_marked_roce_packets', 'np_cnp_sent', 'rp_cnp_handled',
        'out_of_buffer', 'packet_seq_err', 'out_of_sequence',
        'local_ack_timeout_err', 'rnr_nak_retry_err']
BIN_US = 20000  # 20 ms aggregation bins

def load(dirp):
    per_host = defaultdict(list)  # host -> list of (file, samples)
    for f in glob.glob(os.path.join(dirp, 'rccl_telemetry_*.json')):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        s = d.get('hw_samples', [])
        if s:
            per_host[d.get('host_name', os.path.basename(f))].append((f, s))
    return per_host

def series_rates(samples):
    """Per (file,device) consecutive-sample rates keyed by mid-timestamp."""
    byd = defaultdict(list)
    for x in samples:
        byd[(id(samples), x['roce_device'])].append(x)
    out = []  # (ts_us, tx_Bps, {counter: per_s})
    for _, xs in byd.items():
        xs.sort(key=lambda x: x['ts_us'])
        for a, b in zip(xs, xs[1:]):
            dt = (b['ts_us'] - a['ts_us']) / 1e6
            if dt <= 0:
                continue
            tx = (b['tx_bytes'] - a['tx_bytes']) / dt
            rates = {}
            for n in CONG:
                if a.get(n, -1) >= 0 and b.get(n, -1) >= 0:
                    rates[n] = (b[n] - a[n]) / dt
            out.append(((a['ts_us'] + b['ts_us']) // 2, tx, rates))
    return out

def aggregate(host_files):
    allr = []
    for f, s in host_files:
        allr += series_rates(s)
    if not allr:
        return None
    t0 = min(r[0] for r in allr)
    bins = defaultdict(lambda: {'tx': 0.0, **{n: 0.0 for n in CONG}})
    for ts, tx, rates in allr:
        b = ((ts - t0) // BIN_US) * BIN_US
        bins[b]['tx'] += tx
        for n, v in rates.items():
            bins[b][n] += v
    return t0, dict(sorted(bins.items()))

def main(dirp):
    per_host = load(dirp)
    if not per_host:
        print('no hw_samples found in', dirp); return
    # pick busiest host by total tx delta
    def host_tx(hf):
        tot = 0
        for _, s in hf:
            byd = defaultdict(list)
            for x in s: byd[x['roce_device']].append(x)
            for _, xs in byd.items():
                xs.sort(key=lambda x: x['ts_us']); tot += xs[-1]['tx_bytes'] - xs[0]['tx_bytes']
        return tot
    host = max(per_host, key=lambda h: host_tx(per_host[h]))
    agg = aggregate(per_host[host])
    if not agg:
        print('no rates'); return
    t0, bins = agg

    # Perfetto counter events (ts in microseconds)
    ev = [{'ph': 'M', 'name': 'process_name', 'pid': 1, 'tid': 0,
           'args': {'name': f'congestion @ {host}'}}]
    def C(name, ts_us, val):
        ev.append({'ph': 'C', 'name': name, 'ts': ts_us, 'pid': 1, 'tid': 0,
                   'args': {name: val}})
    for b, v in bins.items():
        ts = t0 + b
        C('NIC_tx_GBps', ts, round(v['tx'] / 1e9, 3))
        C('ECN_marked_per_s', ts, round(v['np_ecn_marked_roce_packets'], 1))
        C('CNP_sent_per_s', ts, round(v['np_cnp_sent'], 1))
        C('CNP_handled_per_s', ts, round(v['rp_cnp_handled'], 1))
        C('packet_seq_err_per_s', ts, round(v['packet_seq_err'], 1))
        C('out_of_sequence_per_s', ts, round(v['out_of_sequence'], 1))
        C('out_of_buffer_per_s', ts, round(v['out_of_buffer'], 1))
    out_json = os.path.join(dirp, 'congestion_timeline.json')
    json.dump(ev, open(out_json, 'w'))
    print('wrote', out_json, 'host', host, 'bins', len(bins))

    # PNG
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        xs = [(t0 + b - t0) / 1e6 for b in bins]  # seconds from start
        tx = [v['tx'] / 1e9 for v in bins.values()]
        ecn = [v['np_ecn_marked_roce_packets'] for v in bins.values()]
        cnp = [v['np_cnp_sent'] for v in bins.values()]
        seq = [v['packet_seq_err'] for v in bins.values()]
        oos = [v['out_of_sequence'] for v in bins.values()]
        fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
        ax[0].plot(xs, tx, color='#4C78A8'); ax[0].set_ylabel('NIC tx, GB/s')
        ax[0].set_title(f'Node throughput vs congestion — {host} (aggregated over ranks/NICs, 20ms bins)')
        ax[0].grid(alpha=0.3)
        ax[1].plot(xs, ecn, label='ECN marked/s', color='#E45756')
        ax[1].plot(xs, cnp, label='CNP sent/s', color='#F58518')
        ax[1].plot(xs, seq, label='packet_seq_err/s', color='#72B7B2')
        ax[1].plot(xs, oos, label='out_of_sequence/s', color='#54A24B')
        ax[1].set_ylabel('events/s'); ax[1].set_xlabel('time, s'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        peak = max(max(ecn, default=0), max(cnp, default=0), max(seq, default=0))
        if peak == 0:
            ax[1].text(0.5, 0.5, 'no congestion events this run (uncongested fabric)',
                       transform=ax[1].transAxes, ha='center', va='center', color='gray', fontsize=12)
        out_png = os.path.join(dirp, 'congestion_timeline.png')
        plt.tight_layout(); plt.savefig(out_png, dpi=130)
        print('wrote', out_png, 'peak congestion/s:', peak)
    except Exception as e:
        print('png skipped:', e)

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
