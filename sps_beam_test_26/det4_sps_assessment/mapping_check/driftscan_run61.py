#!/usr/bin/env python3
"""det4 drift-voltage scan at H4, run 61, 2026-08-02.

Resist held fixed at 750 V; drift stepped 700 -> 70 V in 10 points of 5 min
each (det4_drift_scan.log on banco), aborted at point 10 (70 V) when the
drift channel tripped. Mount: 15.465 deg (not the 25.64 deg config
placeholder -- that value is stale, see run notes). Gas: Ar/CF4/iso 88/10/2
(also stale in run_config.json, which still says Ar/Iso 95/5).

The scan spans three DAQ subruns (meshscan_m10V/m20V/m30V) whose own boundaries
don't line up with the drift dwell windows, so this rebuilds one continuous
absolute-time axis across all three before slicing into the 10 voltage points,
rather than reusing effmap.py's single-subrun --windows mechanism.
"""
import csv
import datetime as dt
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
                   "det4_sps_assessment")
from det4_sps_map import POSITION_MM, VIEW  # noqa: E402

BASE = "/home/dylan/x17/sps_run53_det4_check/rot15_ArCF4iso_88-10-2__run61_1214-1400/"
PAIRED = "/home/dylan/x17/sps_run53_det4_check/paired_npz/"
Z = 1120.0
LABEL = "run 61 drift scan, resist 750 V fixed, 15.465 deg, Ar/CF4/iso 88/10/2"

# subrun start wall time (from dream_daq.log "Subrun started"), date 2026-08-02
SUBRUNS = {
    "m10V": (PAIRED + "pair_m10V.npz", dt.datetime(2026, 8, 2, 12, 45, 22)),
    "m20V": (PAIRED + "pair_m20V.npz", dt.datetime(2026, 8, 2, 13, 15, 52)),
    "m30V": (PAIRED + "pair_m30V.npz", dt.datetime(2026, 8, 2, 13, 46, 19)),
}

# drift-scan dwell windows, from det4_drift_scan.log (resist fixed at 750 V)
POINTS = [
    (700.0, "13:13:06", "13:18:06"),
    (630.0, "13:18:11", "13:23:11"),
    (560.0, "13:23:16", "13:28:16"),
    (490.0, "13:28:21", "13:33:21"),
    (420.0, "13:33:26", "13:38:26"),
    (350.0, "13:38:31", "13:43:31"),
    (280.0, "13:43:36", "13:48:36"),
    (210.0, "13:48:41", "13:53:41"),
    (140.0, "13:53:46", "13:58:46"),
    (70.0, "13:58:51", "14:00:46"),
]


def parse_t(s):
    h, m, sec = s.split(":")
    return dt.datetime(2026, 8, 2, int(h), int(m), int(sec))


# ------------------------------------------------------------ load + concat
keys_ev = ("fx_p", "fx_n", "fy_p", "fy_n", "bx_p", "bx_n", "by_p", "by_n",
           "ev_ts", "ev_id")
acc = {k: [] for k in keys_ev}
acc_hit = {k: [] for k in ("h_ev", "h_ch", "h_amp", "h_sig", "h_time")}
wall = []

off = 0
for name, (path, t0) in SUBRUNS.items():
    D = np.load(path)
    n_ev = len(D["fx_p"])
    for k in keys_ev:
        acc[k].append(D[k])
    for k in acc_hit:
        v = D[k]
        if k == "h_ev":
            v = v + off
        acc_hit[k].append(v)
    t = np.array([t0 + dt.timedelta(seconds=float(s) / 1e9) for s in D["ev_ts"]])
    wall.append(t)
    print(f"{name}: {n_ev} events, wall time {t.min()} - {t.max()}")
    off += n_ev

D = {k: np.concatenate(v) for k, v in acc.items()}
H = {k: np.concatenate(v) for k, v in acc_hit.items()}
wall = np.concatenate(wall)
n_ev = len(D["fx_p"])
print(f"\ntotal: {n_ev} events, {len(H['h_ev'])} det4 hits")

# ---------------------------------------------------------- uRWELL reference
clean = np.ones(n_ev, bool)
for k in ("fx", "fy", "bx", "by"):
    clean &= (D[k + "_n"] == 1) & np.isfinite(D[k + "_p"])
tx = D["fx_p"] + (D["bx_p"] - D["fx_p"]) * Z / 1370.0
ty = D["fy_p"] + (D["by_p"] - D["fy_p"]) * Z / 1370.0
clean &= np.isfinite(tx + ty)
print(f"clean single-cluster uRWELL tracks: {clean.sum()}")

# ------------------------------------------------------------ det4 response
h_ev, h_ch, h_amp, h_t = H["h_ev"], H["h_ch"], H["h_amp"], H["h_time"]

hb, eb = np.histogram(h_t, bins=np.arange(-1000, 6000, 100.0))
cb = 0.5 * (eb[1:] + eb[:-1])
pk = np.argmax(hb)
thr = 0.25 * hb[pk]
lo_i = pk - np.argmax(hb[pk::-1] < thr) + 1
hi_i = pk + np.argmax(hb[pk:] < thr)
g0, g1 = cb[lo_i] - 50, cb[hi_i] + 50
print(f"det4 drift gate: {g0:.0f} - {g1:.0f} ns (auto, from the hit-time spectrum)")
GATE = (h_t > g0) & (h_t < g1)
sel = GATE & clean[h_ev]

lead, ncl, nstrip = {}, {}, {}
for v in ("x", "y"):
    k = sel & (VIEW[h_ch] == v)
    ev, pos, amp = h_ev[k], POSITION_MM[h_ch[k]], h_amp[k]
    o = np.lexsort((pos, ev))
    ev, pos, amp = ev[o], pos[o], amp[o]
    new = np.empty(len(ev), bool); new[0] = True
    new[1:] = (ev[1:] != ev[:-1]) | (np.diff(pos) > 3.0)
    cid = np.cumsum(new) - 1
    nc = cid[-1] + 1 if len(cid) else 0
    cq = np.bincount(cid, weights=amp, minlength=nc)
    cp = np.bincount(cid, weights=pos * amp, minlength=nc) / np.maximum(cq, 1e-9)
    cn = np.bincount(cid, minlength=nc)
    cev = np.zeros(nc, np.int64); cev[cid] = ev
    L = np.full(n_ev, np.nan); N = np.zeros(n_ev, np.int32); S = np.zeros(n_ev, np.int32)
    np.add.at(N, cev, 1)
    o2 = np.argsort(cq, kind="stable")
    L[cev[o2]] = cp[o2]; S[cev[o2]] = cn[o2]
    lead[v], ncl[v], nstrip[v] = L, N, S

fired = np.zeros(n_ev, bool)
fired[h_ev[GATE & clean[h_ev]]] = True
reco = np.isfinite(lead["x"] + lead["y"])
disch = (ncl["x"] + ncl["y"] >= 6) | (nstrip["x"] > 40) | (nstrip["y"] > 40)

# ------------------------------------------------------------ alignment fit
fit = clean & reco & ~disch
U = np.column_stack([tx[fit], ty[fit], np.ones(fit.sum())])
V = np.column_stack([lead["x"][fit], lead["y"][fit]])
A, *_ = np.linalg.lstsq(U, V, rcond=None)
for _ in range(4):
    r = np.hypot(*(V - U @ A).T)
    g = r < np.percentile(r, 80)
    A, *_ = np.linalg.lstsq(U[g], V[g], rcond=None)
M = A[:2].T
print(f"alignment (global fit, all voltages): rot {np.degrees(np.arctan2(M[1,0], M[0,0])):+.2f} deg, "
      f"det(A) {np.linalg.det(M):+.4f}, "
      f"median |res| {np.median(np.hypot(*(V - U @ A).T)):.2f} mm")

pred = np.column_stack([tx, ty, np.ones(n_ev)]) @ A
pX, pY = pred[:, 0], pred[:, 1]
resid = np.hypot(lead["x"] - pX, lead["y"] - pY)
within5 = clean & reco & (resid < 5.0)

# --------------------------------------------------------- per-voltage table
rows = []
print(f"\n{'drift V':>8} {'window':>17} {'n clean':>9} {'fired':>8} {'reco':>8} "
      f"{'within5':>9} {'discharge':>10}")
for v, t0s, t1s in POINTS:
    t0, t1 = parse_t(t0s), parse_t(t1s)
    m = clean & (wall >= t0) & (wall < t1)
    n = m.sum()
    if n == 0:
        print(f"{v:8.0f} {t0s+'-'+t1s:>17} {'--- no data ---':>9}")
        continue
    f_any = np.mean(fired[m])
    f_reco = np.mean(reco[m])
    f_w5 = np.mean(within5[m])
    f_disch = np.mean(disch[m])
    n5 = np.sum(within5[m])
    se = np.sqrt(f_w5 * (1 - f_w5) / n) if n else np.nan
    print(f"{v:8.0f} {t0s+'-'+t1s:>17} {n:9d} {f_any:8.1%} {f_reco:8.1%} "
          f"{f_w5:8.1%}+-{se:5.1%} {f_disch:9.1%}")
    rows.append(dict(drift_v=v, resist_v=750.0, n_clean=int(n),
                      fired=f_any, reco=f_reco, within5mm=f_w5, within5mm_err=se,
                      discharge=f_disch))

import json
with open(BASE + "driftscan_run61_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nwrote", BASE + "driftscan_run61_results.csv")

# ---------------------------------------------------------------- the figure
dv = np.array([r["drift_v"] for r in rows])
w5 = np.array([r["within5mm"] for r in rows])
w5e = np.array([r["within5mm_err"] for r in rows])
fa = np.array([r["fired"] for r in rows])
dc = np.array([r["discharge"] for r in rows])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                gridspec_kw=dict(height_ratios=[2, 1]))
ax1.errorbar(dv, w5, yerr=w5e, marker="o", color="#2a78d6", label="within 5 mm of uRWELL track")
ax1.plot(dv, fa, marker="s", ms=4, color="0.6", ls="--", label="fired at all")
ax1.set_ylabel("det4 efficiency")
ax1.set_ylim(0, 1)
ax1.legend()
ax1.set_title(f"det4 (mx17_E) drift-voltage scan — {LABEL}\n"
              f"resist fixed 750 V — point 10 (70 V) aborted by a trip, partial dwell")
ax2.plot(dv, dc, marker="o", color="#e34948")
ax2.set_ylabel("discharge-flagged")
ax2.set_xlabel("drift voltage [V]")
ax2.set_xlim(720, 50)
fig.tight_layout()
fig.savefig(BASE + "driftscan_run61.png", dpi=130)
print("wrote", BASE + "driftscan_run61.png")
