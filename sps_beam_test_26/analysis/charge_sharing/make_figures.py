#!/usr/bin/env python3
"""Figures for the det4 charge-sharing measurement.  Run sharing.py first."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
R = json.loads((HERE / "results.json").read_text())
PITCH = R["meta"]["pitch_mm"]

XC, YC = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


# --- 1. how many strips, and why the raw answer is the threshold's ---------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
w = 0.38
for i, (v, col) in enumerate((("x", XC), ("y", YC))):
    h = np.array(R[f"mult_{v}"]["hist"], float)
    h = 100 * h / h.sum()
    ks = np.arange(len(h))
    a1.bar(ks + (i - 0.5) * w, h, width=w - 0.04, color=col,
           label=f"{v.upper()} view — mean {R[f'mult_{v}']['raw']:.2f}")
a1.set_xlim(0.4, 6.6)
a1.set_xticks(range(1, 7))
a1.set_xlabel("strips in the cluster (above the 5σ zero-suppression)")
a1.set_ylabel("% of head-on tracks")
a1.set_title("As read out", color=INK, fontsize=10.5, loc="left", pad=8)
a1.legend(loc="upper right", fontsize=9)

for v, col in (("x", XC), ("y", YC)):
    pts = R[f"mult_{v}"]["vs_charge"]
    x = [(p["lo"] + (p["hi"] or 3400)) / 2 for p in pts]
    y = [p["mean"] for p in pts]
    a2.plot(x, y, "o-", ms=6, lw=1.8, color=col, label=f"{v.upper()} view")
    a2.plot([R[f"mult_{v}"]["median_charge"]], [R[f"mult_{v}"]["raw"]], "*",
            ms=15, color=col, zorder=5)
a2.axvspan(*R["meta"]["q_window"], color=MUTED, alpha=0.12, lw=0)
a2.text(np.mean(R["meta"]["q_window"]), 1.35, "matched\nwindow", ha="center",
        fontsize=8.5, color=MUTED)
a2.set_xlabel("cluster charge [ADC]")
a2.set_ylabel("mean strips in the cluster")
a2.set_title("At the same charge, the two views agree", color=INK,
             fontsize=10.5, loc="left", pad=8)
a2.legend(loc="lower right", fontsize=9)
a2.text(0.03, 0.96, "★ = where each view actually sits", transform=a2.transAxes,
        va="top", fontsize=8.5, color=MUTED)
save(fig, "multiplicity.png")

# --- 2. the kernels -------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.4, 4.4))
for v, col in (("x", XC), ("y", YC)):
    K = R[f"kernel_{v}"]
    c = np.array(K["centre_mm"], float)
    f = np.array([np.nan if q is None else q for q in K["frac"]], float)
    e = np.array([np.nan if q is None else q for q in K["err"]], float)
    m = np.isfinite(f)
    W = R[f"width_{v}"]["matched"]["rms_deconv_mm"] * 1e3
    ax.fill_between(c[m], (f - e)[m], (f + e)[m], color=col, alpha=0.20, lw=0)
    ax.plot(c[m], f[m], lw=2.0, color=col,
            label=f"{v.upper()} view — kernel rms {W:.0f} µm")
for k in (-3, -2, -1, 1, 2, 3):
    ax.axvline(k * PITCH, color="#e6e5e1", lw=0.8, zorder=0)
ax.set_yscale("log")
ax.set_xlabel("strip position − track impact point [mm]   (gridlines = strip pitch)")
ax.set_ylabel("mean share of the cluster's charge (log)")
ax.set_title("The two kernels, at matched cluster charge",
             color=INK, fontsize=11, loc="left", pad=10)
ax.legend(loc="upper right", fontsize=9)
ax.text(0.02, 0.94, "absent strips counted as zero — otherwise the profile\ncannot fall off, and does not", transform=ax.transAxes, va="top", fontsize=8.5, color=MUTED)
save(fig, "kernel.png")

# --- 3. charge budget -----------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 3.8))
ks = np.arange(4)
w = 0.38
for i, (v, col) in enumerate((("x", XC), ("y", YC))):
    b = [R[f"budget_{v}"][f"within_{j}"] for j in ks]
    ax.bar(ks + (i - 0.5) * w, b, width=w - 0.04, color=col, label=f"{v.upper()} view")
    for j, val in zip(ks, b):
        ax.text(j + (i - 0.5) * w, val + 0.015, f"{val:.2f}", ha="center",
                fontsize=8.5, color=INK2)
ax.set_xticks(ks)
ax.set_xticklabels(["central\nstrip only", "± 1 strip", "± 2 strips", "± 3 strips"])
ax.set_ylim(0, 1.12)
ax.set_ylabel("fraction of the cluster's charge")
ax.set_title("Where the charge sits, relative to the track",
             color=INK, fontsize=11, loc="left", pad=10)
ax.legend(loc="upper left", fontsize=9)
save(fig, "budget.png")

# --- 4. why X needs the selection, and the controls ------------------------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 5.0),
                             gridspec_kw=dict(height_ratios=[1.0, 1.25], hspace=0.55))
for i, (v, col) in enumerate((("x", XC), ("y", YC))):
    for lo, hi, n in R[f"bands_{v}"]:
        a1.barh(1 - i, hi - lo, left=lo, height=0.42, color=col)
    a1.text(104, 1 - i, f"{v.upper()} view", ha="right", va="center",
            fontsize=9.5, color=INK)
a1.set_xlim(100, 305)
a1.set_ylim(-0.55, 1.55)
a1.set_yticks([])
a1.set_xlabel("det4-local position [mm]")
a1.spines["left"].set_visible(False)
a1.grid(axis="y", visible=False)
a1.set_title("Live strips: X amplifies in two usable bands, Y in one 95 mm run",
             color=INK, fontsize=10.5, loc="left", pad=8)

for v, col in (("x", XC), ("y", YC)):
    for blk in R[f"controls_{v}"]["inband"]:
        p = [q["pos"] for q in blk["points"]]
        m = [q["mult"] for q in blk["points"]]
        a2.plot(p, m, "o-", ms=5, lw=1.6, color=col)
a2.plot([], [], "o-", color=XC, label="X view (inside each band)")
a2.plot([], [], "o-", color=YC, label="Y view")
a2.set_xlim(100, 305)
a2.set_xlabel("det4-local position [mm]")
a2.set_ylabel("mean strips/cluster")
a2.set_title("Control: multiplicity drifts with position in BOTH views — gain "
             "non-uniformity,\nso these are local numbers, not chamber constants",
             color=INK, fontsize=10.5, loc="left", pad=8)
a2.legend(loc="upper right", fontsize=9)
save(fig, "selection.png")
