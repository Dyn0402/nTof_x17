"""Figures for the DREAM-returned-to-CEA pedestal check.

    ../.venv/bin/python -m dream_return_cea.figures
"""

from __future__ import annotations

import os
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates               # noqa: E402
import matplotlib.pyplot as plt                 # noqa: E402
import numpy as np                              # noqa: E402

from . import pedestals as P                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

# Same chart system as ntof_pedestal_qa/figures.py.
RAW, CMN = "#eb6834", "#2a78d6"                 # before / after common-noise subtraction
INK, INK2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#f2f1ec"
CRITICAL, GOOD = "#d03b3b", "#0ca30c"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 130,
})


def date_str(rows):
    return f"{rows[0]['start']:%-d %B %Y}"


def _feu_bands(ax, rows, label=False, y=None):
    for i, r in enumerate(rows):
        a, b = i * P.NCH, (i + 1) * P.NCH
        if i % 2:
            ax.axvspan(a - 0.5, b - 0.5, color=BAND, lw=0, zorder=0)
        if label:
            ax.text((a + b) / 2, y, f"FEU {r['slot']} · ID {r['feu_id']}\n"
                    f"taken {r['start']:%H:%M} · {r['events']} ev",
                    ha="center", va="bottom", fontsize=7.5, color=INK2,
                    transform=ax.get_xaxis_transform())


def fig_overview(rows):
    """Every channel of every FEU on one axis: baseline, then noise before and
    after common-noise subtraction."""
    n = len(rows)
    x = np.arange(n * P.NCH)
    mean = np.concatenate([r["mean"] for r in rows])
    raw = np.concatenate([r["raw_sigma"] for r in rows])
    cmn = np.concatenate([r["cns_sigma"] for r in rows])

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(13, 6.6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 1.15],
                                                  hspace=0.08))
    _feu_bands(a0, rows, label=True, y=1.01)
    _feu_bands(a1, rows)
    a0.plot(x, mean, ".", ms=1.6, color=INK2, rasterized=True)
    a0.set_ylabel("pedestal (ADC)")
    a0.set_ylim(0, max(520, mean.max() * 1.1))

    a1.plot(x, raw, ".", ms=1.6, color=RAW, rasterized=True)
    a1.plot(x, cmn, ".", ms=1.6, color=CMN, rasterized=True)
    a1.set_ylabel("noise σ (ADC)")
    a1.set_ylim(0, max(5, raw.max() * 1.25))
    a1.text(x[-1] + 30, np.median(raw[-P.NCH:]), f"raw\n{np.median(raw):.2f}",
            color=INK2, va="center", fontsize=8)
    a1.text(x[-1] + 30, np.median(cmn[-P.NCH:]) - 0.15,
            f"common noise\nsubtracted\n{np.median(cmn):.2f}",
            color=INK2, va="center", fontsize=8)
    a1.plot([], [], "o", ms=5, color=RAW, label="raw σ")
    a1.plot([], [], "o", ms=5, color=CMN,
            label="σ after common-noise subtraction (per chip, per sample median)")
    a1.legend(loc="lower left", ncol=2, fontsize=8, handletextpad=0.2)

    a1.set_xlim(-10, n * P.NCH + 10)
    a1.set_xticks([i * P.NCH + P.NCH // 2 for i in range(n)])
    a1.set_xticklabels([f"feu{r['slot']}" for r in rows])
    a1.tick_params(axis="x", length=0)
    a1.set_xlabel(f"{n} FEUs × {P.NCH} channels  (vertical bands: one FEU, "
                  f"one 8-chip card; median of each panel at right)")
    fig.suptitle(f"MX17 DREAM DAQ back at CEA — pedestals on all {n} FEUs, "
                 f"{date_str(rows)}", x=0.06, ha="left", fontsize=13,
                 fontweight="bold", y=0.995)
    fig.text(0.06, 0.945,
             f"{n * P.NCH} channels, {sum(r['events'] for r in rows):,} "
             f"pedestal events, one FEU at a time  ·  no channel more than "
             f"2× or less than ½× its FEU's median noise",
             ha="left", fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.06, right=0.92, top=0.86, bottom=0.08)
    return _save(fig, "01_all_pedestals")


def fig_timeline(rows, failed):
    """When each FEU was taken, and what it read: median noise with the
    5–95 % channel range, raw and after subtraction."""
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 1.4],
                                                  hspace=0.1))
    for r in rows:
        y = r["slot"]
        a0.barh(y, r["end"] - r["start"], left=r["start"], height=0.6,
                color=CMN, edgecolor="white", lw=0)
        a0.text(r["end"] + timedelta(seconds=15), y,
                f"ID {r['feu_id']} · {r['events']} ev", va="center",
                fontsize=7.5, color=INK2)
    for f in failed:
        d, t = f["stamp"].split("_")
        when = rows[0]["start"].replace(hour=int(t[:2]), minute=int(t[3:5]))
        a0.plot(when + timedelta(seconds=30), f["slot"], "x", color=CRITICAL,
                ms=8, mew=2)
        a0.annotate(f"{when:%H:%M} first try: {f['reason']} — retried",
                    (when + timedelta(seconds=30), f["slot"]),
                    xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=7.5, color=INK2)
    a0.set_yticks([r["slot"] for r in rows])
    a0.set_yticklabels([f"feu{r['slot']}" for r in rows], fontsize=8)
    a0.invert_yaxis()
    a0.set_ylabel("FEU (cfg slot)")
    a0.grid(axis="y", visible=False)

    for key, col, lab, dx in (("raw_sigma", RAW, "raw", -6),
                              ("cns_sigma", CMN, "common noise subtracted", 6)):
        t = [r["start"] + (r["end"] - r["start"]) / 2 + timedelta(seconds=dx)
             for r in rows]
        med = np.array([np.median(r[key]) for r in rows])
        lo = np.array([np.percentile(r[key], 5) for r in rows])
        hi = np.array([np.percentile(r[key], 95) for r in rows])
        a1.errorbar(t, med, yerr=[med - lo, hi - med], fmt="o", ms=6,
                    color=col, ecolor=col, elinewidth=1.5, capsize=0,
                    mec="white", mew=1.5, label=f"{lab} σ (median, 5–95 % of channels)")
    a1.set_ylim(0, 4.5)
    a1.set_ylabel("noise σ (ADC)")
    a1.legend(loc="lower left", fontsize=8)
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    a1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 2)))
    a1.set_xlabel(f"time on {date_str(rows)} (local)")
    lo_t = rows[0]["start"] - timedelta(minutes=3)
    hi_t = max(r["end"] for r in rows) + timedelta(minutes=2)
    a1.set_xlim(lo_t, hi_t)
    fig.suptitle(f"One FEU at a time — acquisition log, {date_str(rows)}",
                 x=0.08, ha="left", fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.1)
    return _save(fig, "02_timeline")


def fig_per_feu(rows):
    """Per-FEU noise distributions, raw vs common-noise subtracted."""
    n = len(rows)
    nc = 3
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(11, 2.5 * nr + 0.6),
                             sharex=True, sharey=True)
    bins = np.linspace(1.5, 4.5, 61)
    for ax, r in zip(axes.flat, rows):
        ax.hist(r["raw_sigma"], bins, color=RAW, alpha=0.85, label="raw")
        ax.hist(r["cns_sigma"], bins, color=CMN, alpha=0.85,
                label="common noise subtracted")
        ax.set_title(f"feu{r['slot']} · FEU ID {r['feu_id']} · "
                     f"{r['start']:%H:%M}", fontsize=9, loc="left")
        ax.text(0.98, 0.95,
                f"raw {np.median(r['raw_sigma']):.2f}\n"
                f"sub {np.median(r['cns_sigma']):.2f}\n"
                f"CM {np.median(r['cm_rms']):.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                color=INK2, family="monospace")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper left", fontsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("noise σ (ADC)")
    for ax in axes[:, 0]:
        ax.set_ylabel("channels")
    fig.suptitle(f"Noise per FEU, 512 channels each — {date_str(rows)}",
                 x=0.06, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "03_per_feu_noise")


# Crate layout as photographed 5 Oct 2026 (mx17_daq_feus.jpg), left to right.
# Chamber letters are the tape above the cards; the orange card's red ID
# sticker is illegible in the photo, its ID is the one its data stream carries.
CRATE = [(1, "D"), (2, "D"), (3, "A"), (4, "A"), (5, "B"), (6, "B"),
         (7, "C"), (8, "C"), (9, "")]


def fig_crate(rows):
    """Front view of the crate with each FEU's ID, legible."""
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
    ids = {r["slot"]: r["feu_id"] for r in rows}
    W, H, gap = 1.0, 9.0, 0.08
    BEIGE, BEIGE_EDGE = "#e9e2c3", "#b9b08a"
    ORANGE, ORANGE_EDGE = "#ef8a1f", "#b8650f"
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.grid(False)
    for i, (slot, det) in enumerate(CRATE):
        x = i * (W + gap)
        orange = slot == 9
        ax.add_patch(FancyBboxPatch((x, 0), W, H, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=ORANGE if orange else BEIGE,
                                    ec=ORANGE_EDGE if orange else BEIGE_EDGE, lw=1.2))
        for k in range(4):                       # 8 connectors, two staggered columns
            y0 = 0.9 + k * 1.75
            ax.add_patch(Rectangle((x + 0.2, y0), 0.16, 1.35, fc="#1c1c1c", lw=0))
            ax.add_patch(Rectangle((x + 0.62, y0 + 0.25), 0.16, 1.35, fc="#1c1c1c", lw=0))
        ax.add_patch(Circle((x + W / 2, H - 0.75), 0.44,
                            fc="#e05a4f" if orange else "#7fcf86", ec="white", lw=1.5))
        ax.text(x + W / 2, H - 0.75, str(ids[slot]), ha="center", va="center",
                fontsize=13 if ids[slot] < 100 else 10.5, fontweight="bold", color=INK)
        ax.text(x + W / 2, H + 0.45, det, ha="center", va="bottom", fontsize=14,
                color="#c4302b", fontweight="bold")
        ax.text(x + W / 2, -0.35, f"feu{slot}", ha="center", va="top",
                fontsize=9, color=INK2)
    # TCM and the empty slots to its right
    x = len(CRATE) * (W + gap)
    ax.add_patch(Rectangle((x, 0), 1.3, H, fc="#2b2b2b", ec="#111", lw=1))
    ax.text(x + 0.65, H / 2, "TCM", ha="center", va="center", color="white",
            fontsize=11, fontweight="bold", rotation=90)
    x += 1.3 + gap
    ax.add_patch(Rectangle((x, 0), 3.2, H, fc="#eceef0", ec="#c3c7cc", lw=1))
    ax.text(x + 1.6, H / 2, "empty", ha="center", va="center", color=MUTED, fontsize=10)
    ax.text(-0.2, H + 0.45, "chamber", ha="right", va="bottom", fontsize=9, color=INK2)
    ax.text(-0.2, -0.35, "cfg slot", ha="right", va="top", fontsize=9, color=INK2)
    ax.set_xlim(-1.6, x + 3.4)
    ax.set_ylim(-1.2, H + 1.4)
    fig.suptitle("MX17 DREAM crate, front view — FEU IDs left to right",
                 x=0.03, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.03, 0.025, "ID stickers as on the cards (green); the orange card's red "
             "sticker is illegible in the photo, its ID is read from its data stream.",
             fontsize=8, color=INK2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.06)
    return _save(fig, "04_crate_feu_ids")


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name + ".png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    rows, ctx = P.load()
    for p in (fig_overview(rows), fig_timeline(rows, ctx["failed"]),
              fig_per_feu(rows), fig_crate(rows)):
        print(p)


if __name__ == "__main__":
    main()
