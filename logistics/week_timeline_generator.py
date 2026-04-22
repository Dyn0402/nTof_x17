#!/usr/bin/env python3
"""
week_timeline_generator.py
──────────────────────────
Day-column / 24-hour-row timeline generator.
Each day is a vertical column; y-axis runs 00:00 (top) → 24:00 (bottom).
Activities spanning multiple days are split across columns.
Up to 2 overlapping activities per column are shown side-by-side.

USAGE
  from week_timeline_generator import draw_week_timeline
"""

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors

# ══════════════════════════════════════════════════════════════
#  COLOUR PALETTE  (matches timeline_generator.py)
# ══════════════════════════════════════════════════════════════
BG       = "#0a0f1a"
SURFACE  = "#111827"
BORDER   = "#1e2d3d"
GRID_COL = "#1e2d40"
TEXT_COL = "#e2e8f0"
MUTED    = "#64748b"
ACCENT   = "#38bdf8"
WEEKEND  = "#161f2e"

COL_PAD   = 0.05   # horizontal margin inside each day column (data units)
SPLIT_GAP = 0.03   # gap between two side-by-side bands (data units)
MIN_LABEL_H = 0.5  # minimum segment height (hours) needed to draw a label

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _to_dt(d):
    """Normalise date or datetime to datetime (midnight if date)."""
    if isinstance(d, datetime):
        return d
    return datetime(d.year, d.month, d.day)


def _assign_global_tracks(bands):
    """
    Greedy interval packing: give every band a fixed _track index so it
    occupies the same sub-column on every day it appears.
    """
    sorted_b = sorted(bands, key=lambda b: _to_dt(b["start"]))
    track_ends = []
    for b in sorted_b:
        placed = False
        for i, end_dt in enumerate(track_ends):
            if _to_dt(b["start"]) >= end_dt:
                b["_track"] = i
                track_ends[i] = _to_dt(b["end"])
                placed = True
                break
        if not placed:
            b["_track"] = len(track_ends)
            track_ends.append(_to_dt(b["end"]))


def _split_to_days(band, t_start, t_end):
    """Split a band into per-day segments within [t_start, t_end)."""
    bs = max(_to_dt(band["start"]), t_start)
    be = min(_to_dt(band["end"]),   t_end)
    if bs >= be:
        return []

    segments = []
    day = t_start
    while day < t_end:
        day_end = day + timedelta(days=1)
        seg_s   = max(bs, day)
        seg_e   = min(be, day_end)
        if seg_s < seg_e:
            segments.append({
                "band":       band,
                "day_idx":    (day - t_start).days,
                "start_hour": (seg_s - day).total_seconds() / 3600,
                "end_hour":   (seg_e - day).total_seconds() / 3600,
            })
        day = day_end

    return segments


# ══════════════════════════════════════════════════════════════
#  MAIN DRAW FUNCTION
# ══════════════════════════════════════════════════════════════

def draw_week_timeline(title, subtitle, start_date, end_date,
                       bands, out_png, out_pdf=None,
                       day_width=1.55, fig_height=9.0):

    t_start = _to_dt(start_date).replace(hour=0, minute=0, second=0, microsecond=0)
    t_end   = _to_dt(end_date).replace(  hour=0, minute=0, second=0, microsecond=0) \
              + timedelta(days=1)
    n_days  = (t_end - t_start).days

    # ── Assign global tracks & split into per-day segments ────
    active = [b for b in bands if not b.get("suppress", False)]
    _assign_global_tracks(active)

    day_segs = [[] for _ in range(n_days)]
    for b in active:
        for seg in _split_to_days(b, t_start, t_end):
            day_segs[seg["day_idx"]].append(seg)

    # Per-day: map global tracks → local sub-columns (so lonely bands
    # on a given day still fill the full column width).
    for segs in day_segs:
        if not segs:
            continue
        tracks    = sorted({seg["band"]["_track"] for seg in segs})
        track_map = {t: i for i, t in enumerate(tracks)}
        n_sc      = len(tracks)
        for seg in segs:
            seg["sub_col"]    = track_map[seg["band"]["_track"]]
            seg["n_sub_cols"] = n_sc

    # ── Figure ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(n_days * day_width, fig_height))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)

    ax.set_xlim(-0.5, n_days - 0.5)
    ax.set_ylim(24, 0)   # midnight at top → midnight at bottom

    # Day labels on top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # ── Weekend shading ───────────────────────────────────────
    for d_idx in range(n_days):
        if (t_start + timedelta(days=d_idx)).weekday() >= 5:
            ax.axvspan(d_idx - 0.5, d_idx + 0.5,
                       facecolor=WEEKEND, alpha=1.0, zorder=0, lw=0)

    # ── Vertical day dividers ─────────────────────────────────
    for d_idx in range(n_days + 1):
        ax.axvline(d_idx - 0.5, color=GRID_COL, lw=0.8, zorder=1)

    # ── Horizontal hour gridlines ─────────────────────────────
    for h in range(25):
        lw  = 1.0  if h % 6 == 0 else 0.35
        col = BORDER if h % 6 == 0 else GRID_COL
        ax.axhline(h, color=col, lw=lw, zorder=1)

    # ── Draw bands ────────────────────────────────────────────
    for d_idx in range(n_days):
        for seg in day_segs[d_idx]:
            b      = seg["band"]
            sh     = seg["start_hour"]
            eh     = seg["end_hour"]
            n_sc   = seg["n_sub_cols"]
            sc     = seg["sub_col"]

            avail_w = 1.0 - 2 * COL_PAD
            sub_w   = (avail_w - (n_sc - 1) * SPLIT_GAP) / n_sc
            x0      = d_idx - 0.5 + COL_PAD + sc * (sub_w + SPLIT_GAP)
            x1      = x0 + sub_w
            height  = eh - sh

            rgb     = mcolors.to_rgb(b["color"])
            tc      = b.get("text_color", "#ffffff")
            pattern = b.get("pattern", "solid")
            opt     = b.get("optional", False)

            if opt or pattern == "dashed":
                ax.add_patch(FancyBboxPatch(
                    (x0, sh), x1 - x0, height,
                    boxstyle="round,pad=0.0",
                    facecolor=(*rgb, 0.22), edgecolor=(*rgb, 0.85),
                    linestyle="--", linewidth=1.4, zorder=3))

            elif pattern == "stripe":
                ax.add_patch(FancyBboxPatch(
                    (x0, sh), x1 - x0, height,
                    boxstyle="round,pad=0.0",
                    facecolor=(*rgb, 0.55), edgecolor=(*rgb, 0.9),
                    linewidth=0.9, zorder=3))
                ax.add_patch(FancyBboxPatch(
                    (x0, sh), x1 - x0, height,
                    boxstyle="round,pad=0.0",
                    facecolor="none",
                    edgecolor=(*mcolors.to_rgb(tc), 0.22),
                    hatch="////", linewidth=0, zorder=4))

            else:
                ax.add_patch(FancyBboxPatch(
                    (x0, sh), x1 - x0, height,
                    boxstyle="round,pad=0.0",
                    facecolor=(*rgb, 1.0), edgecolor=(*rgb, 1.0),
                    linewidth=0.6, zorder=3))

            if height >= MIN_LABEL_H:
                label_color = MUTED if opt else tc
                ax.text((x0 + x1) / 2, (sh + eh) / 2,
                        b["label"],
                        ha="center", va="center",
                        color=label_color,
                        fontsize=8, fontfamily="monospace",
                        fontweight="bold", zorder=5,
                        clip_on=True, multialignment="center",
                        path_effects=[pe.withStroke(linewidth=1.8,
                                                    foreground=(*rgb, 0.35))])

    # ── X-axis: day column labels (top) ───────────────────────
    DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ax.set_xticks(range(n_days))
    tick_labels = []
    for d in range(n_days):
        day_dt = t_start + timedelta(days=d)
        tick_labels.append(f"{DOW[day_dt.weekday()]}\n{day_dt.strftime('%b')} {day_dt.day}")
    ax.set_xticklabels(tick_labels, color=TEXT_COL,
                       fontsize=9, fontfamily="monospace")
    ax.tick_params(axis="x", which="both", length=0, pad=6)

    # ── Y-axis: hour labels ───────────────────────────────────
    ax.set_yticks(range(0, 25, 2))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 25, 2)],
                       color=MUTED, fontsize=8, fontfamily="monospace")
    ax.tick_params(axis="y", which="both", length=0)

    # ── Spine cleanup ─────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Title block ───────────────────────────────────────────
    # char_frac: fraction of figure width per monospace char at fontsize 17
    char_frac = 0.142 / fig.get_figwidth()
    parts = [p.strip() for p in title.split("·", 1)]
    fig.text(0.012, 0.985, parts[0] + (" ·" if len(parts) > 1 else ""),
             ha="left", va="top",
             color=TEXT_COL, fontsize=17, fontfamily="monospace",
             fontweight="bold")
    if len(parts) > 1:
        offset = 0.012 + (len(parts[0]) + 2) * char_frac
        fig.text(offset, 0.985, parts[1],
                 ha="left", va="top",
                 color=MUTED, fontsize=17, fontfamily="monospace")

    fig.text(0.012, 0.955,
             subtitle + f"   ·   Generated {datetime.now():%Y-%m-%d %H:%M}",
             ha="left", va="top",
             color=MUTED, fontsize=9.5, fontfamily="monospace")

    # ── Legend ────────────────────────────────────────────────
    seen = {}
    for b in active:
        lbl = b.get("legend_label")
        if lbl and lbl not in seen and _to_dt(b["end"]) > t_start and _to_dt(b["start"]) < t_end:
            seen[lbl] = b

    legend_items = []
    for lbl, b in seen.items():
        rgb     = mcolors.to_rgb(b["color"])
        pattern = b.get("pattern", "solid")
        opt     = b.get("optional", False)
        if opt or pattern == "dashed":
            patch = mpatches.Patch(facecolor=(*rgb, 0.22),
                                   edgecolor=(*rgb, 0.85),
                                   linestyle="--", linewidth=1.2, label=lbl)
        elif pattern == "stripe":
            patch = mpatches.Patch(facecolor=(*rgb, 0.55),
                                   edgecolor=(*rgb, 0.9),
                                   hatch="////", label=lbl)
        else:
            patch = mpatches.Patch(facecolor=b["color"], label=lbl)
        legend_items.append(patch)

    if legend_items:
        ax.legend(handles=legend_items,
                  loc="lower center",
                  bbox_to_anchor=(0.5, -0.08),
                  ncol=len(legend_items),
                  frameon=True, framealpha=0.9,
                  facecolor=SURFACE, edgecolor=BORDER,
                  labelcolor=MUTED,
                  prop={"family": "monospace", "size": 8.5})

    plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.92])

    fig.savefig(out_png, dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"✓ PNG  → {out_png}")

    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        print(f"✓ PDF  → {out_pdf}")

    plt.show()
    plt.close(fig)
