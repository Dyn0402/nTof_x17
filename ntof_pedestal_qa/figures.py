"""Figures for the DREAM pedestal history.

Every figure is built from `data/ped_stats.npz` through `pedestals.py`, so
re-running after a new extraction moves the pictures and the report text
together.

    ../.venv/bin/python -m ntof_pedestal_qa.figures
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates                       # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402
import numpy as np                                      # noqa: E402
from matplotlib.colors import LogNorm, ListedColormap   # noqa: E402

from . import pedestals as P                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

# Chamber identity, fixed order, never cycled.  Validated for CVD separation
# and contrast against the light chart surface (dataviz validate_palette).
DET_COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#8a5cd6"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CRITICAL, WARNING, GOOD = "#d03b3b", "#fab219", "#0ca30c"

ORDER = [3, 4, 5, 6, 7, 8, 1, 2]        # A x, A y, B x, B y, C x, C y, D x, D y

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 130,
})

# The three DREAM clock configurations the campaign ran in.  Boundaries are the
# midpoint between the last pedestal of one and the first of the next.
EPOCHS = [
    ("20 ns · RdClk/WrClk 4/2", datetime(2026, 6, 28), datetime(2026, 7, 7)),
    ("60 ns · RdClk/WrClk 6/6", datetime(2026, 7, 7), datetime(2026, 7, 23, 13, 26)),
    ("60 ns · RdClk/WrClk 4/6", datetime(2026, 7, 23, 13, 26), datetime(2026, 8, 11)),
]


def _epochs(ax, label=True):
    """Shade the clock epochs behind a time axis."""
    for i, (name, a, b) in enumerate(EPOCHS):
        if i % 2:
            ax.axvspan(a, b, color="#f2f1ec", zorder=0, lw=0)
        if label:
            ax.text(a + (b - a) / 2, 1.012, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7.5, color=MUTED)


def _timeaxis(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlim(datetime(2026, 6, 30), datetime(2026, 8, 11))


def _series(rows, feu, key):
    s = sorted((r for r in rows if r["feu"] == feu), key=lambda r: r["when"])
    return [r["when"] for r in s], [r[key] for r in s]


def _plain_log(ax):
    """Log ticks as 4, 6, 10 rather than 4x10^0."""
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(
        matplotlib.ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.5)))
    ax.tick_params(axis="y", which="minor", labelsize=6.5)


def _label_ends(ax, ends, dx=6):
    """Direct-label each series past the end of its curve, nudged apart.

    ends is [(y, text, color)]; identity must not be carried by color alone,
    and overlapping labels would defeat that.
    """
    ends = sorted(ends, key=lambda e: e[0])
    y0, y1 = ax.get_ylim()
    log = ax.get_yscale() == "log"
    pos = [np.log10(e[0]) if log else e[0] for e in ends]
    span = (np.log10(y1) - np.log10(y0)) if log else (y1 - y0)
    gap = span * 0.036
    for i in range(1, len(pos)):
        if pos[i] - pos[i - 1] < gap:
            pos[i] = pos[i - 1] + gap
    for (yv, text, color), p in zip(ends, pos):
        ax.annotate(text, (1.0, 10 ** p if log else p),
                    xycoords=("axes fraction", "data"),
                    xytext=(dx, 0), textcoords="offset points",
                    va="center", fontsize=8, color=color, fontweight="600",
                    annotation_clip=False)


# ---------------------------------------------------------------- 1. history
def noise_history(sets, rows, path):
    """Coherent and incoherent noise per chamber, over the campaign."""
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 7.0), sharex=True,
                             gridspec_kw=dict(hspace=0.26))
    for ax, key, title in (
            (axes[0], "med_cm", "Common mode — coherent swing of each DREAM chip"),
            (axes[1], "med_res", "Residual — per-channel noise once the common mode is removed")):
        ends = []
        for feu in ORDER:
            det, view = P.FEU_DET[feu]
            t, v = _series(rows, feu, key)
            ax.plot(t, v, color=DET_COLOR[det], lw=1.8,
                    ls="-" if view == "x" else (0, (4, 2)),
                    marker="o", ms=3.2, mew=0, zorder=3)
            ends.append((v[-1], f"{det}{view}", DET_COLOR[det]))
        ax.set_yscale("log")
        ax.set_ylabel("ADC counts (RMS)")
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=20)
        _epochs(ax, label=(ax is axes[0]))
        ax.set_xlim(datetime(2026, 6, 30), datetime(2026, 8, 11))
        _plain_log(ax)
        _label_ends(ax, ends)

    _timeaxis(axes[1])
    for ax in axes:
        ax.set_xlim(datetime(2026, 6, 30), datetime(2026, 8, 11))
        ax.axvline(datetime(2026, 7, 23, 13, 26), color=CRITICAL, lw=1.2,
                   ls=(0, (2, 2)), zorder=2)
    axes[0].axvspan(datetime(2026, 7, 21, 12, 4), datetime(2026, 7, 27, 13),
                    color="#2a78d6", alpha=0.10, lw=0, zorder=1)
    axes[0].annotate("A common-mode excursion", (datetime(2026, 7, 24), 0.06),
                     xycoords=("data", "axes fraction"), ha="center",
                     fontsize=7.5, color="#2a78d6")
    axes[1].annotate("readout clock 6→4", (datetime(2026, 7, 22, 20), 0.90),
                     xycoords=("data", "axes fraction"), ha="right",
                     fontsize=7.5, color=CRITICAL)

    fig.legend(handles=[plt.Line2D([], [], color=DET_COLOR[d], lw=2, label=f"chamber {d}")
                        for d in P.DETS]
                       + [plt.Line2D([], [], color=MUTED, lw=1.6, ls="-", label="x view"),
                          plt.Line2D([], [], color=MUTED, lw=1.6, ls=(0, (4, 2)), label="y view")],
               loc="lower center", ncol=6, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.13, top=0.94, left=0.075, right=0.955)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 2. heatmaps
def _chip_grid(sets, kind):
    """(64 chips, n_sets) of a per-chip quantity, rows in ORDER."""
    g = np.full((64, len(sets)), np.nan)
    for j, s in enumerate(sets):
        for k, feu in enumerate(ORDER):
            d = s.feus.get(feu)
            if d is None:
                continue
            for c in range(8):
                sl = slice(c * 64, (c + 1) * 64)
                g[k * 8 + c, j] = (d["cm_rms"][c] if kind == "cm"
                                   else np.median(d["cns_sigma"][sl]))
    return g


def _chip_axes(fig, ax, sets, title, cbar_label):
    """Shared chrome for the 64-chip x 56-acquisition maps."""
    ax.grid(False)
    for k, feu in enumerate(ORDER):
        det, view = P.FEU_DET[feu]
        if k:
            ax.axhline(k * 8 - 0.5, color="white", lw=1.6)
        ax.annotate(f"{det}{view}", (-0.052, k * 8 + 3.5),
                    xycoords=("axes fraction", "data"), ha="right", va="center",
                    fontsize=9.5, fontweight="700", color=DET_COLOR[det],
                    annotation_clip=False)
    # only the ends of each chip block are ticked; eight labels per block is
    # unreadable at this height and the block order is D0 at the top
    ax.set_yticks([k * 8 + c for k in range(8) for c in (0, 7)])
    ax.set_yticklabels(["D0", "D7"] * 8, fontsize=6)
    ax.set_ylabel("DREAM chip — 64 channels each, D0 to D7 per FEU", labelpad=34)

    ticks = [j for j, s in enumerate(sets) if j == 0 or
             sets[j].when.date() != sets[j - 1].when.date()]
    ax.set_xticks(ticks)
    ax.set_xticklabels([sets[j].when.strftime("%d %b") for j in ticks],
                       rotation=90, fontsize=6.5)
    ax.set_xlabel("pedestal acquisition — 56 in all, equally spaced, not to scale in time")
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)


def chip_heatmap(sets, kind, path, title, vmin, vmax):
    """Absolute level: where the noise was."""
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    g = _chip_grid(sets, kind)
    im = ax.imshow(g, aspect="auto", origin="upper", interpolation="nearest",
                   cmap="magma_r", norm=LogNorm(vmin=vmin, vmax=vmax))
    _chip_axes(fig, ax, sets, title, None)
    fig.colorbar(im, ax=ax, pad=0.015, label="ADC counts (RMS)")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def chip_change_heatmap(sets, kind, path, title):
    """Stability: each chip against its own campaign median.

    Diverging, because the question is polarity — did this chip get louder or
    quieter than it normally was — and a chip's own median is the only fair
    reference when the eight FEUs sit two decades apart.
    """
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    g = _chip_grid(sets, kind)
    ref = np.nanmedian(g, axis=1, keepdims=True)
    r = np.log2(g / ref)
    im = ax.imshow(r, aspect="auto", origin="upper", interpolation="nearest",
                   cmap="RdBu_r", vmin=-2, vmax=2)
    _chip_axes(fig, ax, sets, title, None)
    cb = fig.colorbar(im, ax=ax, pad=0.015,
                      label="relative to this chip's campaign median")
    cb.set_ticks([-2, -1, 0, 1, 2])
    cb.set_ticklabels(["¼×", "½×", "1×", "2×", "4×"])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 3. baseline
def baseline_drift(sets, rows, path):
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.1), sharex=True,
                             gridspec_kw=dict(hspace=0.26, height_ratios=[2, 1]))
    for feu in ORDER:
        det, view = P.FEU_DET[feu]
        t, v = _series(rows, feu, "med_mean")
        axes[0].plot(t, v, color=DET_COLOR[det], lw=1.6,
                     ls="-" if view == "x" else (0, (4, 2)), marker="o", ms=3, mew=0)
        t, v = _series(rows, feu, "spread_mean")
        axes[1].plot(t, v, color=DET_COLOR[det], lw=1.6,
                     ls="-" if view == "x" else (0, (4, 2)), marker="o", ms=3, mew=0)
    axes[0].set_ylabel("median baseline (ADC)")
    axes[0].set_title("Where the baseline sat, and how far it spread across the 512 channels",
                      loc="left", fontsize=10, pad=20)
    axes[1].set_ylabel("channel-to-channel\nspread (ADC, 68 %)")
    for ax in axes:
        _epochs(ax, label=(ax is axes[0]))
    _timeaxis(axes[1])
    fig.legend(handles=[plt.Line2D([], [], color=DET_COLOR[d], lw=2, label=d)
                        for d in P.DETS],
               loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.13, top=0.93, left=0.09, right=0.98)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 4. health
def channel_health(sets, rows, path):
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.0), sharex=True,
                             gridspec_kw=dict(hspace=0.3))
    for ax, key, title in (
            (axes[0], "n_dead", "Disconnected — raw noise collapsed, the strip is off the preamp input"),
            (axes[1], "n_noisy", "Loud — residual noise more than 3x the FEU median")):
        for feu in ORDER:
            det, view = P.FEU_DET[feu]
            t, v = _series(rows, feu, key)
            ax.plot(t, v, color=DET_COLOR[det], lw=1.6,
                    ls="-" if view == "x" else (0, (4, 2)), marker="o", ms=3, mew=0)
        ax.set_ylabel("channels of 512")
        ax.set_title(title, loc="left", fontsize=10, pad=20)
        _epochs(ax, label=(ax is axes[0]))
    axes[0].axhline(64, color=MUTED, lw=1, ls=(0, (2, 2)))
    axes[0].annotate("one connector = 64 channels", (0.012, 64),
                     xycoords=("axes fraction", "data"), va="bottom",
                     fontsize=7.5, color=MUTED)
    _timeaxis(axes[1])
    fig.legend(handles=[plt.Line2D([], [], color=DET_COLOR[d], lw=2, label=d)
                        for d in P.DETS],
               loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.13, top=0.93, left=0.085, right=0.98)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 5. A detail
def chamber_a_detail(sets, path):
    """The A excursion and the dead connector, channel by channel."""
    idx = {s.stamp: i for i, s in enumerate(sets)}
    picks = [("260720_16H12", "20 Jul — before"),
             ("260722_10H13", "22 Jul — excursion + connector 8 gone"),
             ("260727_11H23", "27 Jul 11:23 — still"),
             ("260727_14H11", "27 Jul 14:11 — after the access")]
    fig, axes = plt.subplots(4, 1, figsize=(9.6, 7.4), sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=0.22))
    for ax, (stamp, label) in zip(axes, picks):
        d = sets[idx[stamp]].feus[3]
        ax.plot(np.arange(512), d["raw_sigma"], color="#2a78d6", lw=1.0,
                label="raw")
        ax.plot(np.arange(512), d["cns_sigma"], color=INK2, lw=1.0,
                label="residual")
        ax.set_yscale("log")
        ax.set_ylim(1.5, 400)
        ax.set_title(label, loc="left", fontsize=9, pad=3)
        for c in range(1, 8):
            ax.axvline(c * 64 - 0.5, color=GRID, lw=0.8)
        ax.axvspan(447.5, 511.5, color=CRITICAL, alpha=0.07, lw=0)
    axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    axes[-1].set_xlabel("FEU 3 channel  (chamber A, x view) — connector 8 shaded")
    axes[-1].set_xlim(0, 511)
    fig.text(0.02, 0.5, "noise (ADC RMS)", rotation=90, va="center", fontsize=9,
             color=INK2)
    fig.subplots_adjust(left=0.085, right=0.98, top=0.96, bottom=0.07)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 6. threshold
def threshold_history(sets, rows, path):
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    for feu in ORDER:
        det, view = P.FEU_DET[feu]
        t, v = _series(rows, feu, "med_thr")
        ax.plot(t, [x - 256 for x in v], color=DET_COLOR[det], lw=1.6,
                ls="-" if view == "x" else (0, (4, 2)), marker="o", ms=3, mew=0)
    ax.set_ylabel("5σ threshold above\nbaseline (ADC)")
    ax.set_title("The zero-suppression threshold the DAQ actually loaded",
                 loc="left", fontsize=10, pad=16)
    _epochs(ax)
    _timeaxis(ax)
    ax.legend(handles=[plt.Line2D([], [], color=DET_COLOR[d], lw=2, label=d)
                       for d in P.DETS], loc="upper left", ncol=4, fontsize=8.5)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.13)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 7. usage
def usage_timeline(sets, path):
    """How long each pedestal stayed in force, and how stale it got."""
    usage = P.load_usage()
    live = [r for r in usage if r["start"] and r["ped_dt"]]
    age = [(r["start"] - r["ped_dt"]).total_seconds() / 3600 for r in live]
    t = [r["start"] for r in live]

    fig, axes = plt.subplots(2, 1, figsize=(9.6, 4.8), sharex=True,
                             gridspec_kw=dict(hspace=0.2, height_ratios=[1, 2]))
    for s in sets:
        axes[0].axvline(s.when, color="#2a78d6", lw=0.9, alpha=0.75)
    axes[0].set_yticks([])
    axes[0].set_ylabel("pedestal\nruns", rotation=0, ha="right", va="center")
    axes[0].set_title("When pedestals were taken, and how old the one in force was",
                      loc="left", fontsize=10, pad=8)

    axes[1].plot(t, age, color=INK2, lw=0.9)
    axes[1].fill_between(t, 0, age, color="#2a78d6", alpha=0.14, lw=0)
    axes[1].set_ylabel("age of the pedestal\nin force (hours)")
    axes[1].axhline(24, color=WARNING, lw=1, ls=(0, (3, 2)))
    axes[1].text(datetime(2026, 6, 30, 12), 26, "24 h", fontsize=7.5, color=WARNING)
    _timeaxis(axes[1])
    for ax in axes:
        _epochs(ax, label=False)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.9, bottom=0.12)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    sets = P.load()
    rows, _cm = P.series(sets)
    noise_history(sets, rows, os.path.join(FIG, "noise_history.png"))
    chip_heatmap(sets, "cm", os.path.join(FIG, "chip_common_mode.png"),
                 "Common-mode swing of every DREAM chip, every pedestal run",
                 vmin=1, vmax=250)
    chip_heatmap(sets, "res", os.path.join(FIG, "chip_residual.png"),
                 "Residual per-channel noise of every DREAM chip, every pedestal run",
                 vmin=2, vmax=30)
    chip_change_heatmap(sets, "cm", os.path.join(FIG, "chip_common_mode_rel.png"),
                        "Common mode against each chip's own campaign median — "
                        "the stability view")
    chip_change_heatmap(sets, "res", os.path.join(FIG, "chip_residual_rel.png"),
                        "Residual noise against each chip's own campaign median")
    baseline_drift(sets, rows, os.path.join(FIG, "baseline_drift.png"))
    channel_health(sets, rows, os.path.join(FIG, "channel_health.png"))
    chamber_a_detail(sets, os.path.join(FIG, "chamber_a_detail.png"))
    threshold_history(sets, rows, os.path.join(FIG, "threshold_history.png"))
    usage_timeline(sets, os.path.join(FIG, "usage_timeline.png"))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
