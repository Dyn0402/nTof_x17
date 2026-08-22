#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figures_deck.py -- the three-slide MPGD2026 sequence for the VMM efficiency
deficit on P2_OUT.

The talk arrives here having already shown DREAM working (efficiency, timing,
space) and the VMM matching it on timing and space.  These three slides are the
turn: the VMM is *less efficient*, that deficit is a place on the chamber, and
it is one discriminator level sitting inside the Landau.

  deck_1_deficit.png   the observation -- same chamber, same beam, same working
                       point, 96 % vs 85 %, and the loss is a corner
  deck_2_gainmap.png   the cause is upstream of both readouts -- a factor 3.9
                       gas-gain gradient that BOTH measure, pad for pad
  deck_3_threshold.png the mechanism -- the Landau slides across a fixed
                       discriminator line; the closer (a 16:9 slide_ridge)

Every slide is 13.33 x 7.5 in (16:9) with the same chrome: a bold headline, a
two-line grey sub-line, the figure, and a small source stamp.  Text is kept to
those three; everything else on the slide is a mark or a direct label.

Convention carried from the rest of the deck: DREAM = orange, VMM = blue.  The
two maps on a slide share ONE sequential ramp in a hue neither readout owns, so
they are read against each other and not against their own labels.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Ellipse

import figures as F
import figures_slide as S

HERE = os.path.dirname(os.path.abspath(__file__))
FIG, DATA = os.path.join(HERE, "figures"), os.path.join(HERE, "data")

VMM_C, DREAM_C = F.C1, F.C2
VMM_LBL, DREAM_LBL = "VMM3a", "DREAM"
DEAD_PAD = 635

SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#f2efe6", "#cfe0d8", "#8fc0b2", "#4d9683", "#1f6a58", "#0c3d33"])

SLIDE = (13.33, 7.5)
DPI = 160
XLIM, YLIM = (345, 463), (163, 278)


def chrome(fig, head, sub, stamp):
    """Headline, sub-line and source stamp -- identical on all three slides."""
    fig.text(0.010, 0.938, head, ha="left", va="bottom", fontsize=16,
             fontweight="bold", color=F.INK)
    fig.text(0.010, 0.868, sub, ha="left", va="bottom", fontsize=10.5,
             color=F.INK2, linespacing=1.45)
    fig.text(0.990, 0.012, stamp, ha="right", va="bottom", fontsize=8,
             color=F.MUTED)


def padmap(ax, x, y, c, norm, label, colour):
    sc = ax.scatter(x, y, c=c, s=250, cmap=SEQ, norm=norm,
                    edgecolor=F.SURFACE, linewidth=1.1, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel("x  [mm]", fontsize=9.5)
    ax.grid(True, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_title(label, loc="left", color=colour, fontsize=14,
                 fontweight="bold")
    return sc


def hbar(fig, sc, label, ticks, rect=(0.075, 0.088, 0.40, 0.020)):
    """One horizontal colorbar under the two maps -- they share a scale, so
    they share a legend."""
    cax = fig.add_axes(rect)
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.outline.set_edgecolor(F.MUTED)
    cb.ax.tick_params(labelsize=9, color=F.MUTED)
    cb.set_ticks(ticks)
    cb.set_label(label, fontsize=9.5, color=F.INK2, labelpad=4)
    return cb


CORNER = dict(xy=(370.0, 214.0), width=42.0, height=64.0)


def corner(ax):
    """The low-gain / low-efficiency corner, marked identically on slide 1 and
    slide 2 so the eye carries it from one to the other."""
    ax.add_patch(Ellipse(**CORNER, facecolor="none", edgecolor=F.INK,
                         lw=1.6, ls=(0, (5, 4)), zorder=7))


def gradient(g):
    """The plane fit both readouts share: direction, per-readout slope along it,
    and how much of the pad-to-pad variance it takes."""
    x, y = g["x_v"].to_numpy(), g["y_v"].to_numpy()
    A = np.column_stack([np.ones(len(g)), x, y])
    out = {}
    for key, col in (("d", "amp_med_d"), ("v", "amp_med_v")):
        r = (g[col] / g[col].median()).to_numpy()
        b = np.linalg.lstsq(A, r, rcond=None)[0]
        out[key] = dict(b=b, ang=np.arctan2(b[2], b[1]),
                        r2=1 - np.var(r - A @ b) / np.var(r), rel=r)
    ang = out["d"]["ang"]
    s = (x - x.mean()) * np.cos(ang) + (y - y.mean()) * np.sin(ang)
    for key in ("d", "v"):
        out[key]["slope10"] = np.polyfit(s, out[key]["rel"], 1)[0] * 10
    out["s"], out["ang"] = s, ang
    out["dang"] = np.degrees(abs(np.arctan2(
        np.sin(out["v"]["ang"] - ang), np.cos(out["v"]["ang"] - ang))))
    return out


def _maps_grid(fig):
    return fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.97],
                            left=0.048, right=0.985, top=0.815, bottom=0.185,
                            wspace=0.17)


# --------------------------------------------------------------------------- #
def slide_1(g, n):
    """The observation.  Two efficiency maps on one scale, then the pad-by-pad
    pairing that says the deficit is not a uniform scale factor."""
    x, y = g["x_v"].to_numpy(), g["y_v"].to_numpy()
    ev, ed = g["eff_v"].to_numpy(), g["eff_d"].to_numpy()

    fig = plt.figure(figsize=SLIDE)
    gs = _maps_grid(fig)
    norm = Normalize(0.45, 1.0)

    axd = fig.add_subplot(gs[0, 0])
    sc = padmap(axd, x, y, ed, norm, DREAM_LBL, DREAM_C)
    axd.set_ylabel("y  [mm]", fontsize=9.5)
    axv = fig.add_subplot(gs[0, 1])
    padmap(axv, x, y, ev, norm, VMM_LBL, VMM_C)
    hbar(fig, sc, "efficiency of the tracks that point at the pad",
         [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    corner(axv)
    axv.annotate("every pad the VMM loses\nsits in this corner",
                 xy=(362, 246), xytext=(349, 276), fontsize=11,
                 color=F.INK, fontweight="bold", va="top", ha="left",
                 arrowprops=dict(arrowstyle="-|>", color=F.INK, lw=1.5))

    # -- pad-by-pad pairing -------------------------------------------------- #
    ax = fig.add_subplot(gs[0, 2])
    order = np.argsort(ev)
    r = np.arange(len(order))
    ax.vlines(r, ev[order], ed[order], color=F.MUTED, lw=1.2, alpha=0.65,
              zorder=2)
    ax.scatter(r, ed[order], s=38, color=DREAM_C, zorder=4,
               edgecolor=F.SURFACE, linewidth=0.8)
    ax.scatter(r, ev[order], s=38, color=VMM_C, zorder=4,
               edgecolor=F.SURFACE, linewidth=0.8)
    ax.grid(True, axis="y", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xticks([])
    ax.set_xlim(-2.5, len(order) + 1.5)
    ax.set_ylim(0.27, 1.10)
    ax.set_xlabel("the 53 pads under the beam, sorted by VMM efficiency",
                  fontsize=9.5)
    ax.set_ylabel("efficiency", fontsize=9.5)
    ax.text(len(order), 1.005, DREAM_LBL, fontsize=12.5, color=DREAM_C,
            fontweight="bold", ha="right", va="bottom")
    ax.text(32, 0.880, VMM_LBL, fontsize=12.5, color=VMM_C,
            fontweight="bold", ha="center", va="top")
    ax.text(0, 1.085, "DREAM keeps every live pad above 92 %", fontsize=10.5,
            color=F.INK2, ha="left", va="top")
    ax.annotate("the VMM loses HALF the tracks\non the weakest live pad",
                xy=(1.4, ev[order][1]), xytext=(7, 0.400), fontsize=10.5,
                color=F.INK2, va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=F.INK2, lw=0.9))
    ax.scatter([0], [ev[order][0]], s=150, facecolor="none",
               edgecolor=F.INK, linewidth=1.3, zorder=5)
    ax.text(2.0, ev[order][0], "dead in both", fontsize=9, color=F.INK2,
            va="center")
    ax.set_title("Not a scale factor — a place",
                 loc="left", color=F.INK, fontsize=12.5, fontweight="bold")

    chrome(fig,
           "Same chamber, same beam, same working point — DREAM 95.6 %, "
           "VMM 85.3 %",
           "P2_OUT.  Efficiency of the uRWELL tracks that point at each pad, "
           "measured the same way on both readouts.\nMatched runs: VMM run_46 "
           "and DREAM eff_nominal_1, both at mesh 450 V / drift 750 V.",
           "P2 SPS July 2026 · 53 pads · 0.84 M (VMM) / 1.87 M (DREAM) tracks")
    fig.savefig(f"{FIG}/deck_1_deficit.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def slide_2(g, n):
    """The cause is upstream of the electronics: one gas-gain gradient, and the
    two readouts return the same map pad for pad."""
    x, y = g["x_v"].to_numpy(), g["y_v"].to_numpy()
    G = gradient(g)
    rd, rv = G["d"]["rel"], G["v"]["rel"]

    fig = plt.figure(figsize=SLIDE)
    gs = _maps_grid(fig)
    norm = Normalize(0.5, 2.0)

    axd = fig.add_subplot(gs[0, 0])
    sc = padmap(axd, x, y, rd, norm, DREAM_LBL, DREAM_C)
    axd.set_ylabel("y  [mm]", fontsize=9.5)
    axv = fig.add_subplot(gs[0, 1])
    padmap(axv, x, y, rv, norm, VMM_LBL, VMM_C)
    hbar(fig, sc, "pulse height  /  median of that readout's own pads",
         [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])

    ang = G["ang"]
    for ax, key in ((axd, "d"), (axv, "v")):
        ax.annotate("", xy=(x.mean() + 42 * np.cos(ang),
                            y.mean() + 42 * np.sin(ang)),
                    xytext=(x.mean() - 42 * np.cos(ang),
                            y.mean() - 42 * np.sin(ang)),
                    arrowprops=dict(arrowstyle="-|>", color=F.INK, lw=2.2),
                    zorder=6)
        ax.text(0.970, 0.045, f"{G[key]['slope10'] * 100:+.0f} % per 10 mm",
                transform=ax.transAxes, fontsize=11.5, color=F.INK,
                fontweight="bold", va="bottom", ha="right")
    for ax in (axd, axv):
        corner(ax)
    axv.text(349, 276, "the same corner", fontsize=11, color=F.INK,
             fontweight="bold", va="top", ha="left")

    # -- pad for pad --------------------------------------------------------- #
    ax = fig.add_subplot(gs[0, 2])
    ax.grid(True, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    lim = (0.40, 2.16)
    ax.plot(lim, lim, ls=(0, (5, 4)), lw=1.3, color=F.MUTED, zorder=2)
    ax.text(2.02, 2.09, "1:1", fontsize=9.5, color=F.MUTED, ha="right",
            va="top")
    ax.scatter(rd, rv, s=np.clip(g["n_track_v"] / 420, 20, 140),
               facecolor=VMM_C, alpha=0.55, edgecolor=F.SURFACE, linewidth=0.7,
               zorder=4)
    b = np.polyfit(rd, rv, 1)
    xs = np.linspace(*lim, 20)
    ax.plot(xs, b[0] * xs + b[1], lw=2.4, color=VMM_C, zorder=5)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("relative pad gain, DREAM", fontsize=9.5)
    ax.set_ylabel("relative pad gain, VMM", fontsize=9.5)
    ax.text(0.45, 2.09, f"53 pads   ·   r = {np.corrcoef(rd, rv)[0, 1]:+.2f}",
            fontsize=13, color=F.INK, fontweight="bold", va="top")
    ax.text(0.45, 1.94, "one map, measured twice", fontsize=10.5,
            color=F.INK2, va="top")
    ax.text(0.45, 1.76, f"…but the VMM's copy is FLATTER\n"
            f"(slope {b[0]:.2f}) — hold that thought",
            fontsize=10, color=VMM_C, fontweight="bold", va="top")
    ax.set_title("Pad for pad, the two readouts agree",
                 loc="left", color=F.INK, fontsize=12.5, fontweight="bold")

    chrome(fig,
           "The gain rolls off ×3.9 across the beam spot — and both readouts "
           "measure the same roll-off",
           f"One plane in (x, y) takes {G['d']['r2'] * 100:.0f} % (DREAM) / "
           f"{G['v']['r2'] * 100:.0f} % (VMM) of the pad-to-pad variance, "
           f"pointing the same way to {G['dang']:.0f}°.\nThe variation is the "
           "chamber's, not the electronics' — and the VMM's low-efficiency "
           "corner is its low-gain corner.",
           "P2 SPS July 2026 · P2_OUT · leading-pad pulse height on tracked "
           "events")
    fig.savefig(f"{FIG}/deck_2_gainmap.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def slide_3(g, H, bw, Sp, n):
    """The closer: the spectra themselves, banded by gain, against the one
    discriminator line -- and what each band actually records."""
    gr = S.groups(g, H, bw)
    T = n["T"]
    f2 = [f["eff_all"] for f in n["fix"] if f["factor"] == 2.0][0]

    fig = plt.figure(figsize=SLIDE)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.55, 1],
                          left=0.062, right=0.978, top=0.780, bottom=0.105,
                          wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1], sharey=ax)
    S.ridge(ax, gr, bw, T)
    S.effbars(axb, gr, n)

    ax.annotate("the VMM's discriminator — one level, all six chips",
                xy=(T, S.NGROUP + 0.02), xytext=(T * 1.30, S.NGROUP + 0.26),
                fontsize=11.5, color=VMM_C, fontweight="bold", va="center",
                arrowprops=dict(arrowstyle="-", color=VMM_C, lw=1.1))
    ax.text(T * 0.50, -0.32, "lost", fontsize=12.5, color=DREAM_C,
            fontweight="bold", ha="center", va="center")
    ax.text(T * 2.2, -0.32, "recorded", fontsize=12.5, color=F.INK2,
            ha="center", va="center")
    ax.annotate("", xy=(T * 0.99, -0.32), xytext=(T * 0.70, -0.32),
                arrowprops=dict(arrowstyle="<-", color=DREAM_C, lw=1.2))
    ax.text(30.5, S.NGROUP + 0.12, "pad gain", fontsize=10.5, color=F.INK2,
            ha="left", va="bottom", style="italic")
    ax.set_xlabel("pulse height on the pad, as DREAM records it  [ADC]",
                  fontsize=10.5)
    ax.set_title("53 pads sorted by gain into 8 bands · log axis, so a gain "
                 "factor is a pure sideways shift",
                 loc="left", color=F.INK2, fontsize=10.5)

    axb.text(0.405, S.NGROUP + 0.12, "of the tracks that point at the pad,\n"
             "the % each readout records",
             fontsize=10.5, color=F.INK2, ha="left", va="bottom")
    axb.text(0.50, -0.32, VMM_LBL, fontsize=12, color=VMM_C,
             fontweight="bold", ha="center", va="center")
    axb.text(0.73, -0.32, DREAM_LBL, fontsize=12, color=DREAM_C,
             fontweight="bold", ha="center", va="center")

    chrome(fig,
           "On the weak pads the Landau slides onto the VMM's threshold",
           f"One fitted level, {T:.0f} DREAM ADC, cut into DREAM's per-pad "
           f"spectra reproduces the VMM's efficiency pad by pad: "
           f"r = {n['r_pred']:+.2f} ({n['eff_pred_all'] * 100:.1f} % predicted, "
           f"{n['eff_obs_all'] * 100:.1f} % measured).\nIt sits at "
           f"{n['T_over_mpv']:.2f}× the peak of an average pad and "
           f"{n['T_over_mpv_weak']:.2f}× the peak of a weak one — it eats the "
           f"peak, not the tail.  ×2 signal-over-threshold returns "
           f"{f2 * 100:.0f} %.",
           "P2 SPS July 2026 · P2_OUT · sdt = 224 on all six chips")
    fig.savefig(f"{FIG}/deck_3_threshold.png", dpi=DPI)
    plt.close(fig)


def main():
    g, H, bw, Sp, n = S.load()
    slide_1(g, n)
    slide_2(g, n)
    slide_3(g, H, bw, Sp, n)
    print("wrote figures/deck_{1_deficit,2_gainmap,3_threshold}.png")


if __name__ == "__main__":
    main()
