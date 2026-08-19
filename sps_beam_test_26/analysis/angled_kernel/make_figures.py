#!/usr/bin/env python3
"""Figures for the angled-mount kernel measurement.  Run measure.py first."""
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

# same categorical slots as the companion charge-sharing note, so a colour
# means the same thing across the two documents
XC, YC = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
GOOD, BAD = "#1a7f37", "#b0341d"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
})
ARMS = [a for a in ("flat700", "rot_d425", "rot_d325", "rot_d225") if a in R]
LAB = {"flat700": "flat\n233 V/cm", "rot_d425": "25.64°\n142 V/cm",
       "rot_d325": "25.64°\n108 V/cm", "rot_d225": "25.64°\n75 V/cm"}


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def sym(o, key, d=1):
    v = [o.get(f"{key}_+{d}"), o.get(f"{key}_-{d}")]
    v = [q for q in v if q is not None and np.isfinite(q)]
    return float(np.mean(v)) if v else np.nan


# --- 1. the kernel A/B: X view is normal incidence in every arm ------------
# Plotted against FIELD, not as categories, because the flat arm is the only
# one at 233 V/cm -- mount and drift field are confounded in this dataset and
# the only honest test is whether the flat point sits on the trend the three
# rotated points define.
xs = np.arange(len(ARMS))
fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.9))
rot = [a for a in ARMS if a != "flat700"]
Ef = R["flat700"]["E_Vcm"]
for k, (key, ttl, ylab) in enumerate((
        ("pk", "±1 peak / centre", "ratio"),
        ("area", "±1 area / centre", "ratio"),
        ("shift", "±1 peak delay", "ns"))):
    a = ax[k]
    get = (lambda o: np.mean([o["shift_p1_ns"], o["shift_m1_ns"]])) \
        if key == "shift" else (lambda o: sym(o, key))
    Er = np.array([R[m]["E_Vcm"] for m in rot])
    yr = np.array([get(R[m]["x"]) for m in rot])
    yf = get(R["flat700"]["x"])
    a.plot(Er, yr, "o", color=YC, ms=9, label="25.64° mount")
    cf = np.polyfit(Er, yr, 1)
    Eg = np.linspace(60, 250, 50)
    a.plot(Eg, np.polyval(cf, Eg), "--", color=YC, lw=1.4,
           label="their field trend, extrapolated")
    a.plot([Ef], [yf], "s", color=XC, ms=11, label="flat mount")
    pred = np.polyval(cf, Ef)
    dev = 100 * (yf - pred) / abs(pred)
    a.annotate("", xy=(Ef, yf), xytext=(Ef, pred),
               arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.1))
    a.text(Ef - 8, 0.5 * (yf + pred), f"{dev:+.0f} %", ha="right", va="center",
           fontsize=9.5, color=INK2)
    a.set_xlabel("drift field  [V/cm]")
    a.set_ylabel(ylab)
    a.set_title(ttl, fontsize=11)
    lo = min(list(yr) + [yf, pred]); hi = max(list(yr) + [yf, pred])
    pad = 0.45 * (hi - lo) + 1e-9
    a.set_ylim(lo - pad, hi + pad)
    if k == 0:
        a.legend(fontsize=8.5, loc="lower right")
fig.suptitle("X view — normal incidence in ALL four arms, including the three "
             "at a 25.64° mount.  Does the flat point sit on the rotated "
             "arms' trend?", fontsize=11, y=1.05)
save(fig, "kernel_ab.png")

# --- 2. the geometry lever: Y view, normal vs 25.64 deg --------------------
fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.9))
cols = [MUTED] + [YC] * (len(ARMS) - 1)
for k, (getter, ttl, ylab) in enumerate((
        (lambda o: sym(o, "pk", 2) / sym(o, "pk"), "±2 / ±1 peak", "ratio"),
        (lambda o: o["width_20pct"], "cluster extent above 20 %", "strips"),
        (lambda o: o["shift_asym_ns"], "±1 delay asymmetry  (+1 − −1)", "ns"))):
    a = ax[k]
    y = [getter(R[m]["y"]) for m in ARMS]
    a.bar(xs, y, color=cols, width=0.62)
    for i, q in enumerate(y):
        a.text(i, q + (0.02 * max(np.abs(y)) * np.sign(q or 1)), f"{q:.3f}"
               if abs(q) < 10 else f"{q:.0f}", ha="center",
               va="bottom" if q >= 0 else "top", fontsize=9, color=INK2)
    a.set_xticks(xs)
    a.set_xticklabels([LAB[m] for m in ARMS], fontsize=8.5)
    a.set_title(ttl, fontsize=11)
    a.set_ylabel(ylab)
    if k == 2:
        a.axhline(0, color=INK2, lw=1)
fig.suptitle("Y view — normal incidence in the flat arm (grey), the 25.64° "
             "ladder in the rotated arms (orange)", fontsize=11.5, y=1.04)
save(fig, "geometry.png")

# --- 3. the stacks themselves ---------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
for k, (arm, ttl) in enumerate((("flat700", "flat mount, 233 V/cm"),
                                ("rot_d425", "25.64° mount, 142 V/cm"))):
    a = ax[k]
    for v, col, ls in (("x", XC, "-"), ("y", YC, "-")):
        o = R[arm][v]
        t = np.array(o["t_rel_ns"])
        for d, alpha, lw in ((0, 1.0, 2.0), (1, 0.7, 1.5), (2, 0.45, 1.3)):
            s = o["stacks"].get(str(d))
            if s is None:
                continue
            a.plot(t, np.array(s) / np.max(o["stacks"]["0"]), ls, color=col,
                   alpha=alpha, lw=lw,
                   label=(f"{v.upper()} view" if d == 0 else None))
    a.set_title(f"{ttl}\n({R[arm]['y']['incidence']} in Y)", fontsize=10.5)
    a.set_xlabel("time relative to the leading strip's peak  [ns]")
    a.axhline(0, color=INK2, lw=0.8)
    if k == 0:
        a.set_ylabel("trim20 stack, leading-strip peak = 1")
        a.legend(fontsize=9, loc="upper right")
    a.text(0.03, 0.93, "solid → faint = centre, ±1, ±2", transform=a.transAxes,
           fontsize=8.5, color=MUTED)
save(fig, "stacks.png")

# --- 4. lateral width vs drift field --------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.0))
for m in ARMS:
    if "width_x" not in R[m]:
        continue
    w = R[m]["width_x"]
    s = np.array(w["sigma_mm"]); q = np.array(w["charge"])
    t = np.array(w["t_ns"])
    pk = int(np.nanargmax(q))
    sel = (np.arange(len(s)) >= pk - 2) & (np.arange(len(s)) <= pk + 18) \
        & np.isfinite(s)
    ax[0].plot(t[sel] - t[pk], s[sel], "-o", ms=3.4, lw=1.6,
               label=f"{R[m]['E_Vcm']:.0f} V/cm  ({R[m]['mount']})")
ax[0].set_xlabel("time after the leading strip's peak  [ns]")
ax[0].set_ylabel("lateral rms about the telescope point  [mm]")
ax[0].set_title("σ vs drift depth", fontsize=11)
ax[0].legend(fontsize=8.5)

Ea, sa = [], []
for m in ARMS:
    if "width_x" not in R[m]:
        continue
    w = R[m]["width_x"]
    s = np.array(w["sigma_mm"]); q = np.array(w["charge"])
    Ea.append(R[m]["E_Vcm"]); sa.append(s[int(np.nanargmax(q))])
Ea, sa = np.array(Ea), np.array(sa)
ax[1].plot(Ea, sa, "o", color=XC, ms=10, label="measured")
Eg = np.linspace(min(Ea) * 0.9, max(Ea) * 1.1, 100)
ref = sa[np.argmax(Ea)]
ax[1].plot(Eg, ref * np.sqrt(max(Ea) / Eg), "--", color=BAD, lw=1.8,
           label=r"if it were all diffusion  ($\propto E^{-1/2}$)")
ax[1].axhline(np.mean(sa), color=GOOD, lw=1.8, ls=":",
              label="field-independent (the film)")
ax[1].set_xlabel("drift field  [V/cm]")
ax[1].set_ylabel("lateral rms at the peak  [mm]")
ax[1].set_title("the diffusion test", fontsize=11)
ax[1].legend(fontsize=8.5)
save(fig, "width.png")
