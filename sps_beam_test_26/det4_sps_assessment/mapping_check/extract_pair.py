"""Generic extractor for the runs where FEU 1 sits in combined_hits and FEU 3
(det4) sits in a separate per-FEU decode.

  python extract_pair.py run_56 meshscan_m60V
"""
import csv
import glob
import os
import sys

import numpy as np
import uproot

RUN, CFG = sys.argv[1], sys.argv[2]
URW_DIR = (f"/local/home/banco/P2_data/TB_July2026_H4/runs/{RUN}/{CFG}/"
           f"combined_hits_root/")
OUT = f"/tmp/eff_scan_out/det4_{RUN}_{CFG}_v3.npz"

urw_files = sorted(glob.glob(URW_DIR + "*combined_hits.root"))
PAIRS = []
for f in urw_files:
    idx = os.path.basename(f).split("_")[-3]           # '000', '001', ...
    d4 = f"/tmp/eff_scan_out/hits/{CFG}_{idx}_03_hits.root"
    if not os.path.exists(d4):
        raise SystemExit(f"missing det4 decode {d4}")
    PAIRS.append((f, d4))
print("pairs:", [(os.path.basename(a), os.path.basename(b)) for a, b in PAIRS])

urw_view = np.full(512, "", dtype="<U1")
urw_pos = np.full(512, np.nan)
urw_det = np.full(512, "", dtype="<U1")
with open("mapping_urwell.csv") as f:
    for row in csv.DictReader(f):
        c = int(row["channel"])
        urw_view[c] = row["view"]; urw_pos[c] = float(row["position_mm"])
        urw_det[c] = "f" if row["detector"] == "EIC_uRWELL_front" else "b"


def clusters(ev, pos, amp, n_ev, gap=3.0):
    lead = np.full(n_ev, np.nan); ncl = np.zeros(n_ev, np.int16)
    if not len(ev):
        return lead, ncl
    o = np.lexsort((pos, ev)); ev, pos, amp = ev[o], pos[o], amp[o]
    new = np.empty(len(ev), bool); new[0] = True
    new[1:] = (ev[1:] != ev[:-1]) | (np.diff(pos) > gap)
    cid = np.cumsum(new) - 1; nc = cid[-1] + 1
    cq = np.bincount(cid, weights=amp, minlength=nc)
    cp = np.bincount(cid, weights=pos * amp, minlength=nc) / np.maximum(cq, 1e-9)
    cev = np.zeros(nc, np.int64); cev[cid] = ev
    np.add.at(ncl, cev, 1)
    o2 = np.argsort(cq, kind="stable"); lead[cev[o2]] = cp[o2]
    return lead, ncl


acc = {}
def put(k, v):
    acc.setdefault(k, []).append(v)

off = 0
for urwf, d4f in PAIRS:
    a = uproot.open(urwf + ":hits").arrays(
        ["eventId", "channel", "amplitude", "trigger_timestamp_ns"], library="np")
    b = uproot.open(d4f + ":hits").arrays(
        ["eventId", "channel", "amplitude", "significance", "time"], library="np")
    ev_uniq = np.union1d(a["eventId"], b["eventId"])
    n_ev = len(ev_uniq)
    ia = np.searchsorted(ev_uniq, a["eventId"])
    ib = np.searchsorted(ev_uniq, b["eventId"])
    ts = np.zeros(n_ev, np.int64)
    ts[ia] = a["trigger_timestamp_ns"]
    ch, amp = a["channel"], np.abs(a["amplitude"])
    for det in "fb":
        for v in "xy":
            k = (urw_det[ch] == det) & (urw_view[ch] == v)
            p, n = clusters(ia[k], urw_pos[ch[k]], amp[k], n_ev)
            put(f"{det}{v}_p", p); put(f"{det}{v}_n", n)
    put("ev_ts", ts)
    put("ev_id", ev_uniq.astype(np.int64))
    put("h_ev", ib + off)
    put("h_ch", b["channel"].astype(np.int16))
    put("h_amp", np.abs(b["amplitude"]).astype(np.float32))
    put("h_sig", b["significance"].astype(np.float32))
    put("h_time", b["time"].astype(np.float32))
    off += n_ev
    print("  ", os.path.basename(urwf), n_ev, "events,", len(b["eventId"]), "det4 hits")

np.savez_compressed(OUT, **{k: np.concatenate(v) for k, v in acc.items()})
print("wrote", OUT)
