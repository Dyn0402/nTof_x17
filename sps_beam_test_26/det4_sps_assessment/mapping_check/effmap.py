#!/usr/bin/env python3
"""det4 efficiency map at H4, referenced to the uRWELL telescope.

Every clean uRWELL track is extrapolated to det4 and mapped into detector-local
mm through the alignment; det4's own response at that point is then classified
the way the June bench work classifies it (`DET4_SPS_ASSESSMENT.md` §3), so the
beam numbers and the cosmic numbers mean the same thing.

  python effmap.py [run_56_meshscan_m60V_v3] [--hv 610,615,620]
"""
import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, "/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
                   "det4_sps_assessment")
from det4_sps_map import POSITION_MM, VIEW           # noqa: E402

BASE = ("/media/dylan/data/x17/sps_run53_det4_check/flat_ArCO2iso_95-3-2__run53-56/"
        "mapping_check/")
OUTD = BASE

ap = argparse.ArgumentParser()
ap.add_argument("tag", nargs="?", default="run_56_meshscan_m60V_v3")
ap.add_argument("--z", type=float, default=1120.0, help="det4 z along the beam")
ap.add_argument("--windows", default="610:15:40:09-15:41:26,615:15:41:27-15:42:59,"
                                     "620:15:43:00-15:45:18",
                help="V:hh:mm:ss-hh:mm:ss, comma separated; '' = whole run")
ap.add_argument("--t0", default="15:34:56.066", help="sub-run 0 start wall time")
ap.add_argument("--gate", default="", help="det4 drift gate lo,hi in ns; empty = auto")
ap.add_argument("--ampcut", type=float, default=0.0)
ap.add_argument("--out", default="", help="suffix for output filenames")
ap.add_argument("--band", default="149,161", help="live band for the Y-uniformity profile")
ap.add_argument("--grid", type=float, default=3.0, help="map cell [mm]")
ap.add_argument("--label", default="run 56, resist 610-620 V, flat mount")
args = ap.parse_args()

D = np.load(BASE + f"det4_{args.tag}.npz")
n_ev = len(D["fx_p"])

# ---------------------------------------------------------------- HV windows
def secs(s):
    h, m, rest = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)

in_window = np.ones(n_ev, bool)
hv_label = args.label
if args.windows:
    t0 = secs(args.t0)
    t = t0 + D["ev_ts"] / 1e9
    in_window = np.zeros(n_ev, bool)
    for w in args.windows.split(","):
        v, span = w.split(":", 1)
        a, b = span.split("-")
        in_window |= (t >= secs(a)) & (t < secs(b))
    print(f"events inside the quoted HV windows: {in_window.sum()} of {n_ev}")

# ---------------------------------------------------------- uRWELL reference
clean = in_window.copy()
for k in ("fx", "fy", "bx", "by"):
    clean &= (D[k + "_n"] == 1) & np.isfinite(D[k + "_p"])
tx = D["fx_p"] + (D["bx_p"] - D["fx_p"]) * args.z / 1370.0
ty = D["fy_p"] + (D["by_p"] - D["fy_p"]) * args.z / 1370.0
clean &= np.isfinite(tx + ty)
print(f"clean single-cluster uRWELL tracks: {clean.sum()}")

# ------------------------------------------------------------ det4 response
h_ev, h_ch, h_amp, h_t = D["h_ev"], D["h_ch"], D["h_amp"], D["h_time"]

# The drift gate is per RUN, not a constant: run 53 runs 600-1850 ns and run 56
# 600-3650, because the sampling/peaking configuration differs between them.
# Read it off the hit-time spectrum rather than hard-coding either.
if args.gate:
    g0, g1 = (float(v) for v in args.gate.split(","))
else:
    hb, eb = np.histogram(h_t, bins=np.arange(-1000, 6000, 100.0))
    cb = 0.5 * (eb[1:] + eb[:-1])
    pk = np.argmax(hb)
    thr = 0.25 * hb[pk]
    lo_i = pk - np.argmax(hb[pk::-1] < thr) + 1
    hi_i = pk + np.argmax(hb[pk:] < thr)
    g0, g1 = cb[lo_i] - 50, cb[hi_i] + 50
print(f"det4 drift gate: {g0:.0f} - {g1:.0f} ns"
      f"{' (given)' if args.gate else ' (auto, from the hit-time spectrum)'}")
GATE = (h_t > g0) & (h_t < g1)
sel = GATE & (h_amp > args.ampcut) & clean[h_ev]

lead, ncl, qtot, nstrip = {}, {}, {}, {}
for v in ("x", "y"):
    k = sel & (VIEW[h_ch] == v)
    ev, pos, amp = h_ev[k], POSITION_MM[h_ch[k]], h_amp[k]
    o = np.lexsort((pos, ev))
    ev, pos, amp = ev[o], pos[o], amp[o]
    new = np.empty(len(ev), bool); new[0] = True
    new[1:] = (ev[1:] != ev[:-1]) | (np.diff(pos) > 3.0)   # cluster in mm
    cid = np.cumsum(new) - 1
    nc = cid[-1] + 1
    cq = np.bincount(cid, weights=amp, minlength=nc)
    cp = np.bincount(cid, weights=pos * amp, minlength=nc) / np.maximum(cq, 1e-9)
    cn = np.bincount(cid, minlength=nc)
    cev = np.zeros(nc, np.int64); cev[cid] = ev
    L = np.full(n_ev, np.nan); N = np.zeros(n_ev, np.int32)
    Q = np.zeros(n_ev); S = np.zeros(n_ev, np.int32)
    np.add.at(N, cev, 1)
    np.add.at(Q, cev, cq)
    o2 = np.argsort(cq, kind="stable")
    L[cev[o2]] = cp[o2]; S[cev[o2]] = cn[o2]
    lead[v], ncl[v], qtot[v], nstrip[v] = L, N, Q, S

fired = np.zeros(n_ev, bool)                     # any in-time hit at all
fired[h_ev[GATE & clean[h_ev]]] = True
reco = np.isfinite(lead["x"] + lead["y"])
# a discharge lights up the chamber: many clusters, or a very wide one
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
print(f"alignment: rot {np.degrees(np.arctan2(M[1,0], M[0,0])):+.2f} deg, "
      f"det(A) {np.linalg.det(M):+.4f}, "
      f"median |res| {np.median(np.hypot(*(V - U @ A).T)):.2f} mm")

pred = np.column_stack([tx, ty, np.ones(n_ev)]) @ A
pX, pY = pred[:, 0], pred[:, 1]
resid = np.hypot(lead["x"] - pX, lead["y"] - pY)

within5 = clean & reco & (resid < 5.0)
print(f"\n{'':22} {'all tracks':>12} {'excl. discharges':>18}")
nd = clean & ~disch
for name, m in (("fired at all", fired), ("reconstructed", reco),
                ("within 5 mm", within5), ("discharge-flagged", disch)):
    print(f"  {name:20} {np.mean(m[clean]):12.1%} {np.mean(m[nd]):18.1%}")

# ------------------------------------------------------------------- the map
G = args.grid
lo_x, hi_x = np.percentile(pX[clean], [0.2, 99.8])
lo_y, hi_y = np.percentile(pY[clean], [0.2, 99.8])
ex = np.arange(np.floor(lo_x / G) * G, np.ceil(hi_x / G) * G + G, G)
ey = np.arange(np.floor(lo_y / G) * G, np.ceil(hi_y / G) * G + G, G)
den, _, _ = np.histogram2d(pX[clean], pY[clean], bins=[ex, ey])
num, _, _ = np.histogram2d(pX[within5], pY[within5], bins=[ex, ey])
MIN_N = 25
eff = np.divide(num, den, out=np.full_like(num, np.nan), where=den >= MIN_N)
print(f"\nmap: {G:.0f} mm cells, {np.isfinite(eff).sum()} cells with >= {MIN_N} tracks;"
      f" median {np.nanmedian(eff):.1%}, max {np.nanmax(eff):.1%}")

# the X profile is the one that carries the stripes
px_e = np.arange(np.floor(lo_x), np.ceil(hi_x) + 1, 1.0)
dX, _ = np.histogram(pX[clean], bins=px_e)
nX, _ = np.histogram(pX[within5], bins=px_e)
aX, _ = np.histogram(pX[clean & fired], bins=px_e)
prof = np.divide(nX, dX, out=np.full(len(dX), np.nan), where=dX >= 40)
prof_any = np.divide(aX, dX, out=np.full(len(dX), np.nan), where=dX >= 40)
pc = 0.5 * (px_e[1:] + px_e[:-1])

# Along-strip uniformity must be measured INSIDE one live band: marginalising
# over X mixes in the band structure via the beam's own X-Y correlation.
bl, bh = (float(v) for v in args.band.split(","))
inb = (pX > bl) & (pX < bh)
py_e = np.arange(np.floor(lo_y), np.ceil(hi_y) + 1, 4.0)
dY, _ = np.histogram(pY[clean & inb], bins=py_e)
nY, _ = np.histogram(pY[within5 & inb], bins=py_e)
profY = np.divide(nY, dY, out=np.full(len(dY), np.nan), where=dY >= 30)
pcy = 0.5 * (py_e[1:] + py_e[:-1])

J = np.load("/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
            "det4_sps_assessment/stripes_g_det4.npz")

# ---------------------------------------------------------------- the figure
cmap = LinearSegmentedColormap.from_list(
    "eff", ["#2b1b3d", "#3b5b92", "#3f9e9e", "#8ecf6a", "#f2e661"])
cmap.set_bad("#e8e8e8")
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], width_ratios=[1.5, 1],
                      hspace=0.32, wspace=0.24)

ax = fig.add_subplot(gs[0, 0])
im = ax.pcolormesh(ex, ey, eff.T, cmap=cmap, vmin=0, vmax=1, shading="flat")
for lo, hi in J["bands"]:
    if hi > ex[0] and lo < ex[-1]:
        ax.axvline(lo, color="w", lw=0.7, alpha=0.45)
        ax.axvline(hi, color="w", lw=0.7, alpha=0.45)
ax.set(xlabel="detector-local X [mm]  (the striped coordinate)",
       ylabel="detector-local Y [mm]",
       title=f"det4 efficiency, within 5 mm of the uRWELL track "
             f"({G:.0f} mm cells)\nwhite lines: June cosmic band edges")
ax.set_aspect("equal")
fig.colorbar(im, ax=ax, label="efficiency", fraction=0.035)

ax = fig.add_subplot(gs[0, 1])
ax.hist(resid[clean & reco], bins=np.arange(0, 15, 0.15), color="#3b5b92")
ax.axvline(5, color="tab:red", ls="--", label="5 mm")
ax.set(xlabel="|det4 - uRWELL track| [mm]", ylabel="tracks",
       title="residual, reconstructed tracks")
ax.legend()

ax = fig.add_subplot(gs[1, :])
ax.plot(pc, prof_any, color="0.65", lw=1.1, label="det4 fired at all")
ax.plot(pc, prof, color="#1f4e79", lw=1.6, label="reconstructed within 5 mm")
for lo, hi in J["bands"]:
    ax.axvspan(lo, hi, color="tab:green", alpha=0.16)
ax2 = ax.twinx()
ax2.semilogy(J["c"], np.maximum(J["med"], 1), color="tab:orange", lw=1.0,
             alpha=0.8, label="June cosmic median charge")
ax2.set_ylabel("June charge [ADC]", color="tab:orange")
ax.set(xlim=(pc[0], pc[-1]), ylim=(0, 1),
       xlabel="detector-local X [mm]", ylabel="efficiency",
       title="X profile against the June cosmic stripe map "
             "(green = June live bands)")
ax.legend(loc="upper left")
fig.suptitle(f"det4 (mx17_E) efficiency at H4 — {args.label}, "
             f"{clean.sum():,} clean uRWELL tracks", y=0.975, fontsize=13)
fig.savefig(OUTD + f"det4_efficiency_map{args.out}.png", dpi=115, bbox_inches="tight")
print("wrote", OUTD + f"det4_efficiency_map{args.out}.png")

np.savez(OUTD + f"det4_efficiency_map{args.out}.npz", ex=ex, ey=ey, eff=eff, den=den,
         num=num, prof_x=prof, prof_any_x=prof_any, prof_x_centres=pc,
         prof_y=profY, prof_y_centres=pcy, A=A, z=args.z, grid=G,
         n_clean=clean.sum(), label=args.label)

# ----------------------------------------------------- the numbers to quote
print("\nefficiency in the June bands that the beam covers:")
for lo, hi in J["bands"]:
    m = clean & (pX > lo) & (pX < hi)
    if m.sum() < 200:
        continue
    print(f"  X {lo:6.1f}-{hi:6.1f} mm  n={m.sum():7d}  "
          f"within 5 mm {np.mean(within5[m]):6.1%}  "
          f"fired {np.mean(fired[m]):6.1%}  "
          f"discharge {np.mean(disch[m]):5.1%}")
inband = np.zeros(n_ev, bool)
for lo, hi in J["bands"]:
    inband |= (pX > lo) & (pX < hi)
print(f"\n  inside June live bands : {np.mean(within5[clean & inband]):6.1%} "
      f"(n={(clean & inband).sum()})")
print(f"  between them           : {np.mean(within5[clean & ~inband]):6.1%} "
      f"(n={(clean & ~inband).sum()})")
print(f"  whole illuminated spot : {np.mean(within5[clean]):6.1%}")

# ---- bands as the BEAM sees them, at half the local peak -------------------
good = np.isfinite(prof)
if good.sum() > 20:
    thr = 0.5 * np.nanmax(prof)
    live = good & (prof > thr)
    edges_i = np.flatnonzero(np.diff(live.astype(int)) != 0) + 1
    segs = np.split(np.arange(len(prof)), edges_i)
    print(f"\nbands as the beam resolves them (>{thr:.0%}, 1 mm bins):")
    beam_bands = []
    for s in segs:
        if not live[s[0]] or len(s) < 3:
            continue
        lo, hi = pc[s[0]] - 0.5, pc[s[-1]] + 0.5
        beam_bands.append((lo, hi))
        jl = [f"{a:.0f}-{b:.0f}" for a, b in J["bands"] if b > lo and a < hi]
        print(f"  X {lo:6.1f} - {hi:6.1f} mm  ({hi-lo:4.1f} mm wide)  "
              f"mean eff {np.nanmean(prof[s]):5.1%}  peak {np.nanmax(prof[s]):5.1%}"
              f"   June band(s) here: {', '.join(jl) or 'none'}")
    print("\nbest contiguous windows (beam-measured, June's table for comparison):")
    for w, june in ((10, "205-213: 0.97"), (20, "373-391: 0.93"),
                    (30, "365-393: 0.91"), (40, "177-215: 0.88")):
        k = int(w)
        if k >= good.sum():
            continue
        ker = np.convolve(np.nan_to_num(prof), np.ones(k) / k, mode="valid")
        okw = np.convolve(good.astype(float), np.ones(k), mode="valid") == k
        if not okw.any():
            continue
        ker = np.where(okw, ker, -1)
        i = int(np.argmax(ker))
        print(f"  {w:2d} mm: X {pc[i]-0.5:6.1f}-{pc[i+k-1]+0.5:6.1f} mm, "
              f"mean eff {ker[i]:5.1%}    (June bench: {june})")
    np.save(OUTD + f"det4_beam_bands{args.out}.npy", np.array(beam_bands))
