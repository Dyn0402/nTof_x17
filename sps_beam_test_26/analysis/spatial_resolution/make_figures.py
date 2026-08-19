#!/usr/bin/env python3
"""Figures for the det4 spatial-resolution measurement.  Run resolution.py first."""
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
D = np.load(HERE / "residuals.npz")

# pitch is an ordinal magnitude -> one hue, light->dark (validated)
PITCH_C = {0.5: "#86b6ef", 1.0: "#2a78d6", 1.5: "#104281"}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"

# Mount angle: the configuration epoch (RUN_TIMELINE.md), cross-checked at 25.4 deg
# in DET4_URW_MAPPING_2026-08-01.md.  Quote the record, not resolution.py's looser
# singular-value estimate.
MOUNT = 25.64

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


# --- 1. residual per back-plane pitch zone -------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.2))
zx = {0.5: (0.0, 15.5), 1.0: (64.44, 127.44), 1.5: (16.88, 63.38)}
for row in sorted(R["zones"]["uRW-x"], key=lambda q: q["pitch"]):
    p = row["pitch"]
    lo, hi = zx[p]
    m = (D["bx"] >= lo) & (D["bx"] <= hi)
    r = D["rx"][m] * 1000
    r = r[np.abs(r) < 1200]
    h, e = np.histogram(r, bins=70 if len(r) > 5000 else 28,
                        range=(-1200, 1200), density=True)
    c = 0.5 * (e[1:] + e[:-1])
    ax.plot(c, h, lw=1.8, color=PITCH_C[p],
            label=f"back pitch {p} mm   σ = {row['sigma_res']*1e3:.0f} µm")
ax.set_xlabel("det4 − track, uRW-x [µm]")
ax.set_ylabel("normalised density")
ax.set_title("The residual widens with the REFERENCE pitch, not with det4",
             color=INK, fontsize=11, loc="left", pad=10)
ax.legend(loc="upper right", fontsize=9)
ax.text(0.02, 0.95, "run 53, flat mount\n44 198 clean tracks", transform=ax.transAxes,
        va="top", fontsize=8.5, color=MUTED)
save(fig, "zone_residuals.png")

# --- 2. the decomposition ------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
rows = sorted(R["zones"]["uRW-x"], key=lambda q: q["pitch"])
P = np.array([q["pitch"] for q in rows])
Y = np.array([(q["sigma_res"] * 1e3) ** 2 for q in rows])
YE = np.array([2 * q["sigma_res"] * q["err"] * 1e6 for q in rows])
f = R["fit"]
xs = np.linspace(0, 2.6, 100)
inter = (f["sigma_det4"] * 1e3) ** 2
W_BACK = (1120 / 1370.0) ** 2
slope = W_BACK * (f["f_back"] * 1e3) ** 2          # the fitted slope, not the chord
ax.plot(xs, inter + slope * xs, lw=1.6, color=MUTED, zorder=1,
        label=f"fit: back plane = {f['f_back']:.3f} × pitch")
for q, y, ye in zip(rows, Y, YE):
    ax.errorbar(q["pitch"] ** 2, y, yerr=ye, fmt="o", ms=9, lw=1.6,
                color=PITCH_C[q["pitch"]], zorder=3)
    ax.annotate(f"{q['pitch']} mm", (q["pitch"] ** 2, y), textcoords="offset points",
                xytext=(12, -14), fontsize=9, color=INK2)
ax.axhline(inter, ls="--", lw=1.4, color=ORANGE, zorder=2)
ax.annotate(f"intercept = det4 alone,  σ = {f['sigma_det4']*1e3:.0f} µm",
            (1.30, inter), textcoords="offset points", xytext=(0, 10),
            fontsize=9.5, color=ORANGE, fontweight="bold")
ax.set_xlabel("back-plane pitch²  [mm²]")
ax.set_ylabel("σ(residual)²  [µm²]")
ax.set_xlim(0, 2.6)
ax.set_ylim(0, max(Y) * 1.25)
ax.set_title("Extrapolating the reference away: pitch → 0 leaves det4",
             color=INK, fontsize=11, loc="left", pad=10)
ax.legend(loc="lower right", fontsize=9)
save(fig, "decomposition.png")

# --- 3. the boundary step, with the orthogonal control -------------------
fig, ax = plt.subplots(figsize=(7.4, 4.2))
for tag, col, lab in [("signal_x", BLUE, "uRW-x residual — this coordinate's pitch changes"),
                      ("control_x", ORANGE, "uRW-y residual — same det4 region, pitch fixed")]:
    pts = R["profile"][tag]
    x = [q["z"] for q in pts]
    y = [q["sigma"] * 1e3 for q in pts]
    ye = [q["err"] * 1e3 for q in pts]
    ax.errorbar(x, y, yerr=ye, fmt="o-", ms=6, lw=1.8, color=col, label=lab)
ax.axvline(63.9, ls="--", lw=1.4, color=MUTED)
ax.annotate("zone boundary\n1.5 mm → 1.0 mm", (63.9, ax.get_ylim()[1]),
            textcoords="offset points", xytext=(8, -30), fontsize=9, color=INK2)
ax.set_xlabel("back-plane x position [mm]")
ax.set_ylabel("σ(residual) [µm]")
ax.set_title("The width steps at the pitch boundary; the control does not",
             color=INK, fontsize=11, loc="left", pad=10)
ax.legend(loc="lower left", fontsize=9)
save(fig, "boundary_step.png")

# --- 4. z-scan -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.0))
z = [q["z"] for q in R["zscan"]]
s = [q["sigma"] * 1e3 for q in R["zscan"]]
ax.plot(z, s, "o-", ms=6, lw=1.8, color=BLUE)
ax.axvline(1120, ls="--", lw=1.4, color=ORANGE)
ax.annotate("det4 survey estimate\nz = 1120 mm", (1120, max(s)),
            textcoords="offset points", xytext=(-96, -14), fontsize=9, color=ORANGE)
ax.set_xlabel("assumed det4 z [mm]")
ax.set_ylabel("σ(residual) [µm]")
ax.set_title(f"Interpolation is exact for a straight track — the minimum locates det4\n"
             f"curvature gives a beam divergence of {R['divergence_rad']*1e3:.2f} mrad",
             color=INK, fontsize=10.5, loc="left", pad=10)
save(fig, "zscan.png")

# --- 5. the angle effect -------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.2))
m0 = (D["bx"] >= 64.44) & (D["bx"] <= 127.44)
mt = (D["bx_tilt"] >= 64.44) & (D["bx_tilt"] <= 127.44)
flat_sigma = [q for q in R["zones"]["uRW-x"] if q["pitch"] == 1.0][0]["sigma_res"]
for r, col, lab in [(D["rx"][m0] * 1e3, BLUE, f"flat (0°)  σ = {flat_sigma*1e3:.0f} µm"),
                    (D["rx_tilt"][mt] * 1e3, ORANGE,
                     f"tilted ({MOUNT:.1f}°)  σ = "
                     f"{R['tilt']['rows'][0]['sigma_res']*1e3:.0f} µm")]:
    r = r[np.abs(r) < 6000]
    h, e = np.histogram(r, bins=90, range=(-6000, 6000), density=True)
    ax.plot(0.5 * (e[1:] + e[:-1]), h, lw=1.8, color=col, label=lab)
ax.set_yscale("log")
ax.set_xlabel("det4 − track, uRW-x [µm]")
ax.set_ylabel("normalised density (log)")
ax.set_title(f"The same chamber at {MOUNT:.1f}° — the drift gap projects onto the strips",
             color=INK, fontsize=11, loc="left", pad=10)
ax.legend(loc="upper right", fontsize=9)
save(fig, "angle.png")

# --- 6. the bench comparison, drawn in VARIANCE so the split is honest ----
# Bars are sigma^2 (which is what actually adds); labels carry the sigma.
from matplotlib.patches import Patch

fig, ax = plt.subplots(figsize=(8.6, 4.0))
bars = [
    ("SPS beam, det4 at 0°\nreference measured", 176, 242, BLUE, ""),
    ("Cosmic bench, det3 at θ < 5°\nsplit by M3's core pointing", 606, 206, ORANGE,
     "needs det3 3.4× worse than the fleet's worst chamber"),
    ("Cosmic bench, det3 at θ < 5°\nsplit by the beam-measured chamber", 176, 615, ORANGE,
     "needs 1.1 mrad scattering over the 558 mm lever"),
]
ypos = [1.0, 0.5, 0.0]
for (lab, dut, ref, col, note), y in zip(bars, ypos):
    a, b = dut ** 2 / 1e3, ref ** 2 / 1e3
    ax.barh(y, a, height=0.28, color=col)
    ax.barh(y, b, height=0.28, left=a, color=col, alpha=0.30)
    if a > 70:
        ax.text(a / 2, y, f"{dut} µm", ha="center", va="center",
                fontsize=9.5, color="white", fontweight="bold")
    else:
        ax.text(a / 2, y + 0.235, f"{dut} µm", ha="center", va="bottom",
                fontsize=9, color=col, fontweight="bold")
    ax.text(a + b / 2, y, f"{ref} µm", ha="center", va="center",
            fontsize=9.5, color=INK2)
    ax.text(-14, y + (0.07 if note else 0), lab, ha="right", va="center",
            fontsize=9.5, color=INK)
    if note:
        ax.text(-14, y - 0.16, note, ha="right", va="center",
                fontsize=8.5, color=MUTED, style="italic")
ax.legend(handles=[Patch(facecolor=MUTED, label="detector"),
                   Patch(facecolor=MUTED, alpha=0.30, label="reference pointing")],
          loc="lower right", fontsize=9)
ax.set_xlim(0, 460)
ax.set_ylim(-0.4, 1.45)
ax.set_yticks([])
ax.set_xlabel("σ²  [10³ µm²]  — variance, because that is what adds")
ax.spines["left"].set_visible(False)
ax.grid(axis="y", visible=False)
ax.set_title("The bench residual splits two ways, and only one of them is physical",
             color=INK, fontsize=11, loc="left", pad=10)
save(fig, "budget.png")
