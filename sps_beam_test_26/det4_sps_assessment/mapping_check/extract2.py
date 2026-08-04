"""Re-extract with per-view cluster multiplicity, so pile-up can be cut offline."""
import csv
import os
import sys

import numpy as np
import uproot

RUN = sys.argv[1] if len(sys.argv) > 1 else "run_53"
CFG = "cfg_gain4.5_peaktime200_deflt"
OUT = f"/tmp/eff_scan_out/det4_{RUN}_v2.npz"
DIR = f"/local/home/banco/P2_data/TB_July2026_H4/runs/{RUN}/{CFG}/combined_hits_root/"
FILES = sorted(f for f in os.listdir(DIR) if f.endswith("combined_hits.root"))

urw_view = np.full(512, "", dtype="<U1")
urw_pos = np.full(512, np.nan)
urw_det = np.full(512, "", dtype="<U1")
with open("mapping_urwell.csv") as f:
    for row in csv.DictReader(f):
        ch = int(row["channel"])
        urw_view[ch] = row["view"]
        urw_pos[ch] = float(row["position_mm"])
        urw_det[ch] = "f" if row["detector"] == "EIC_uRWELL_front" else "b"


def clusters(ev, pos, amp, n_ev, gap=3.0):
    """Returns (leading position, leading charge, total charge, n_clusters)."""
    lead = np.full(n_ev, np.nan)
    qlead = np.zeros(n_ev)
    qtot = np.zeros(n_ev)
    ncl = np.zeros(n_ev, np.int16)
    if len(ev) == 0:
        return lead, qlead, qtot, ncl
    o = np.lexsort((pos, ev))
    ev, pos, amp = ev[o], pos[o], amp[o]
    new = np.empty(len(ev), bool)
    new[0] = True
    new[1:] = (ev[1:] != ev[:-1]) | (np.diff(pos) > gap)
    cid = np.cumsum(new) - 1
    nc = cid[-1] + 1
    cq = np.bincount(cid, weights=amp, minlength=nc)
    cnum = np.bincount(cid, weights=pos * amp, minlength=nc) / np.maximum(cq, 1e-9)
    cev = np.zeros(nc, np.int64)
    cev[cid] = ev
    np.add.at(ncl, cev, 1)
    np.add.at(qtot, cev, cq)
    order = np.argsort(cq, kind="stable")       # ascending -> last write is max
    lead[cev[order]] = cnum[order]
    qlead[cev[order]] = cq[order]
    return lead, qlead, qtot, ncl


acc = {k: [] for k in ("ev_id", "ev_file")}
for det in "fb":
    for v in "xy":
        for s in ("p", "q", "qt", "n"):
            acc[f"{det}{v}_{s}"] = []
acc.update({k: [] for k in ("h_ev", "h_ch", "h_amp", "h_sig", "h_time")})
off = 0
for ifile, fn in enumerate(FILES):
    a = uproot.open(DIR + fn + ":hits").arrays(
        ["feu", "eventId", "channel", "amplitude", "significance", "time"],
        library="np")
    feu, eid, ch = a["feu"], a["eventId"], a["channel"]
    amp, sig, tm = np.abs(a["amplitude"]), a["significance"], a["time"]
    ev_uniq, ev_idx = np.unique(eid, return_inverse=True)
    n_ev = len(ev_uniq)
    m_urw = feu == 1
    for det in "fb":
        for v in "xy":
            k = m_urw & (urw_det[ch] == det) & (urw_view[ch] == v)
            p, q, qt, n = clusters(ev_idx[k], urw_pos[ch[k]], amp[k], n_ev)
            acc[f"{det}{v}_p"].append(p); acc[f"{det}{v}_q"].append(q)
            acc[f"{det}{v}_qt"].append(qt); acc[f"{det}{v}_n"].append(n)
    m4 = feu == 3
    acc["h_ev"].append(ev_idx[m4] + off)
    acc["h_ch"].append(ch[m4].astype(np.int16))
    acc["h_amp"].append(amp[m4].astype(np.float32))
    acc["h_sig"].append(sig[m4].astype(np.float32))
    acc["h_time"].append(tm[m4].astype(np.float32))
    acc["ev_id"].append(ev_uniq.astype(np.int64))
    acc["ev_file"].append(np.full(n_ev, ifile, np.int8))
    off += n_ev
    print("done", fn, n_ev)

np.savez_compressed(OUT, **{k: np.concatenate(v) for k, v in acc.items()})
print("wrote", OUT)
