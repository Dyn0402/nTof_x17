#!/usr/bin/env python3
"""
timeline_generator.py
─────────────────────
Reusable Gantt-style timeline generator for experiment scheduling.
Outputs a high-resolution PNG + PDF for Google Slides / email.

KEY FEATURES
  • Two swimlanes: "Activity" and "Events"
  • Overlapping bands within a swimlane auto-stack into sub-rows
  • "wiggle_days" on a band renders a semi-transparent extension showing
    schedule slack / buffer between activities
  • Weekend shading (Sat/Sun darker background)
  • suppress=True hides a band without deleting it

USAGE
  Import draw_timeline and call it with your schedule config, e.g.:
      from timeline_generator import draw_timeline
"""

from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors

# ══════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ══════════════════════════════════════════════════════════════
BG       = "#0a0f1a"
SURFACE  = "#111827"
BORDER   = "#1e2d3d"
GRID_COL = "#1e2d40"
TEXT_COL = "#e2e8f0"
MUTED    = "#64748b"
ACCENT   = "#38bdf8"
WEEKEND  = "#161f2e"

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _to_dt(d):
    """Normalize date or datetime to datetime (midnight if date)."""
    if isinstance(d, datetime):
        return d
    return datetime(d.year, d.month, d.day)


def date_to_x(d, start):
    x = (_to_dt(d) - _to_dt(start)).total_seconds() / 86400
    # For datetime objects, shift by -0.5 so midnight aligns with the left
    # edge of the day column (columns are centered on integer x values).
    if isinstance(d, datetime):
        x -= 0.5
    return x


_PACK_GAP = timedelta(seconds=int(0.15 * 86400 * 0))  # slightly > 2 × FancyBboxPatch pad=0.07 days


def assign_sub_rows(bands):
    """Greedy interval packing to avoid visual overlaps."""
    bands.sort(key=lambda b: _to_dt(b["start"]))
    sub_row_ends = []
    for b in bands:
        placed = False
        for i, end in enumerate(sub_row_ends):
            if _to_dt(b["start"]) >= _to_dt(end) + _PACK_GAP:
                b["sub_row"] = i
                sub_row_ends[i] = b["end"]
                placed = True
                break
        if not placed:
            b["sub_row"] = len(sub_row_ends)
            sub_row_ends.append(b["end"])
    n = len(sub_row_ends)
    for b in bands:
        b["n_sub_rows"] = n
    return bands, n


def add_bar(ax, x0, x1, y_top, bar_h, color, tc,
            pattern, opt, label, clip_width_threshold=1.2):
    """Draw a single band rectangle + label."""
    width = x1 - x0
    rgb   = mcolors.to_rgb(color)

    if opt or pattern == "dashed":
        rect = FancyBboxPatch(
            (x0, y_top), width, bar_h,
            # boxstyle="round,pad=0.07",
            boxstyle="round,pad=0.0",
            facecolor=(*rgb, 0.22),
            edgecolor=(*rgb, 0.85),
            linestyle="--", linewidth=1.4, zorder=3)
        ax.add_patch(rect)

    elif pattern == "stripe":
        rect = FancyBboxPatch(
            (x0, y_top), width, bar_h,
            # boxstyle="round,pad=0.07",
            boxstyle="round,pad=0.0",
            facecolor=(*rgb, 0.55),
            edgecolor=(*rgb, 0.9),
            linewidth=0.9, zorder=3)
        ax.add_patch(rect)
        hrect = FancyBboxPatch(
            (x0, y_top), width, bar_h,
            # boxstyle="round,pad=0.07",
            boxstyle="round,pad=0.0",
            facecolor="none",
            edgecolor=(*mcolors.to_rgb(tc), 0.22),
            hatch="////", linewidth=0, zorder=4)
        ax.add_patch(hrect)

    else:
        rect = FancyBboxPatch(
            (x0, y_top), width, bar_h,
            # boxstyle="round,pad=0.07",
            boxstyle="round,pad=0.0",
            facecolor=(*rgb, 1.0),
            edgecolor=(*rgb, 1.0),
            linewidth=0.6, zorder=3)
        ax.add_patch(rect)

    if width >= clip_width_threshold:
        label_color = MUTED if opt else tc
        ax.text((x0 + x1) / 2, y_top + bar_h / 2,
                label,
                ha="center", va="center",
                color=label_color,
                fontsize=9, fontfamily="monospace",
                fontweight="bold", zorder=5,
                clip_on=True, multialignment="center",
                path_effects=[pe.withStroke(linewidth=1.8,
                                            foreground=(*rgb, 0.35))])


# ══════════════════════════════════════════════════════════════
#  MAIN DRAW FUNCTION
# ══════════════════════════════════════════════════════════════

def draw_timeline(title, subtitle, start_date, end_date,
                  bands, milestones, out_png, out_pdf=None):

    total_days = (_to_dt(end_date) - _to_dt(start_date)).days + 1

    # Normalise to date for grid calculations so integer day positions are preserved
    _start = _to_dt(start_date).date()
    _end   = _to_dt(end_date).date()

    # Filter suppressed
    active = [b for b in bands if not b.get("suppress", False)]
    active, n_sub_rows = assign_sub_rows(active)

    # ── Layout constants ──────────────────────────────────────
    BAR_H     = 0.55
    SUB_GAP   = 0.20
    MILE_H    = 1.65
    TOP_PAD   = 0.25

    activity_stack_h = n_sub_rows * (BAR_H + SUB_GAP) - SUB_GAP
    activity_y_base  = TOP_PAD                         # top of Activity row
    divider_y        = activity_y_base + activity_stack_h + 0.55
    events_y_base    = divider_y                       # top of Events row
    total_height     = events_y_base + MILE_H + 0.3

    # ── Figure ────────────────────────────────────────────────
    fig_w = max(18, total_days * 0.295)
    fig_h = max(5.5, total_height * 1.05 + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)

    # x: 0 = start_date.  y increases downward.
    # We shift x so labels have room on the left.
    LABEL_MARGIN = 2.5   # days-worth of space reserved left of day-0
    ax.set_xlim(-LABEL_MARGIN, total_days - 0.5)
    ax.set_ylim(total_height, -0.15)

    # ── Weekend shading ───────────────────────────────────────
    d = _start
    while d <= _end:
        if d.weekday() >= 5:
            x = date_to_x(d, _start)
            ax.axvspan(x - 0.5, x + 0.5,
                       facecolor=WEEKEND, alpha=1.0, zorder=0, lw=0)
        d += timedelta(days=1)

    # ── Day grid lines ────────────────────────────────────────
    d = _start
    while d <= _end:
        x = date_to_x(d, _start)
        ax.axvline(x - 0.5, color=GRID_COL, lw=0.8, alpha=0.9, zorder=1)
        d += timedelta(days=1)

    # ── Swimlane labels & divider ─────────────────────────────
    act_mid = activity_y_base + activity_stack_h / 2
    ax.text(-LABEL_MARGIN + 0.1, act_mid, "Activity",
            ha="left", va="center",
            color=MUTED, fontsize=10, fontfamily="monospace",
            fontweight="bold", transform=ax.transData)

    ax.axhline(divider_y, color=BORDER, lw=0.9, zorder=1)

    evt_mid = events_y_base + MILE_H / 2
    ax.text(-LABEL_MARGIN + 0.1, evt_mid, "Events",
            ha="left", va="center",
            color=MUTED, fontsize=10, fontfamily="monospace",
            fontweight="bold", transform=ax.transData)

    # Vertical separator between label zone and chart area
    ax.axvline(-0.5, color=BORDER, lw=0.9, zorder=2)

    # ── Draw bands ────────────────────────────────────────────
    for b in active:
        if _to_dt(b["start"]) > _to_dt(end_date) or _to_dt(b["end"]) < _to_dt(start_date):
            continue
        # Clamp while preserving the original type so date_to_x applies the
        # correct shift (datetime gets -0.5; date does not).
        s = b["start"] if _to_dt(b["start"]) >= _to_dt(start_date) else start_date
        e = b["end"]   if _to_dt(b["end"])   <= _to_dt(end_date)   else end_date

        timed = isinstance(b["start"], datetime) or isinstance(b["end"], datetime)
        pad   = 0.0 if timed else 0.42
        x0    = date_to_x(s, start_date) - pad
        x1    = date_to_x(e, start_date) + pad
        sub   = b["sub_row"]
        y_top = activity_y_base + sub * (BAR_H + SUB_GAP)

        add_bar(ax, x0, x1, y_top, BAR_H,
                b["color"], b.get("text_color", "#ffffff"),
                b.get("pattern", "solid"),
                b.get("optional", False),
                b["label"])


    # ── Milestones ────────────────────────────────────────────
    visible_ms = [m for m in milestones
                  if _start <= _to_dt(m["date"]).date() <= _end]
    visible_ms.sort(key=lambda m: _to_dt(m["date"]))

    DIAM_Y = events_y_base + 0.30

    stagger_levels = [0.0, 0.55, 1.10]
    level_last_x   = [-999.0] * len(stagger_levels)
    MIN_GAP        = 4.5

    for m in visible_ms:
        x   = date_to_x(m["date"], _start)
        rgb = mcolors.to_rgb(m["color"])

        ax.plot(x, DIAM_Y, marker="D", markersize=8,
                color=m["color"], zorder=6,
                markeredgecolor=(*rgb, 0.5), markeredgewidth=0.9)

        ax.plot([x, x], [divider_y, DIAM_Y - 0.12],
                color=(*rgb, 0.28), lw=0.8, zorder=2)

        chosen = 0
        for lvl in range(len(stagger_levels)):
            if x - level_last_x[lvl] >= MIN_GAP:
                chosen = lvl
                break
        level_last_x[chosen] = x

        label_y = DIAM_Y + 0.22 + stagger_levels[chosen]

        if stagger_levels[chosen] > 0:
            ax.plot([x, x], [DIAM_Y + 0.18, label_y - 0.05],
                    color=(*rgb, 0.28), lw=0.7, zorder=5)

        ax.text(x, label_y, m["label"],
                ha="center", va="top",
                color=TEXT_COL, fontsize=8.5, fontfamily="monospace",
                zorder=6, multialignment="center",
                path_effects=[pe.withStroke(linewidth=2.5,
                                            foreground=SURFACE)])

    # ── X-axis ticks (no labels — day labels live above) ─────
    ax.set_xticks([])
    ax.tick_params(axis="x", which="both", bottom=False)

    # ── Day-of-week labels (just above top spine) ────────────
    # ── Day-of-month numbers (just below bottom spine) ───────
    DOW   = "MTWTFSS"
    trans = ax.get_xaxis_transform()   # x: data coords, y: axes fraction
    d = _start
    while d <= _end:
        x          = date_to_x(d, _start)
        is_weekend = d.weekday() >= 5
        col = GRID_COL if is_weekend else MUTED
        ax.text(x, 1.01, DOW[d.weekday()],
                ha="center", va="bottom",
                color=col, fontsize=6, fontfamily="monospace",
                transform=trans, clip_on=False)
        ax.text(x, -0.01, str(d.day),
                ha="center", va="top",
                color=col, fontsize=6, fontfamily="monospace",
                transform=trans, clip_on=False)
        d += timedelta(days=1)

    # ── Month labels centered on visible days of each month ──
    m = _start.replace(day=1)
    while m <= _end:
        m_end = (m.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        vis_start = max(m, _start)
        vis_end   = min(m_end, _end)
        x_center  = (date_to_x(vis_start, _start) + date_to_x(vis_end, _start)) / 2
        ax.text(x_center, -0.05, m.strftime("%B"),
                ha="center", va="top",
                color=ACCENT, fontsize=11, fontfamily="monospace",
                fontweight="bold", transform=trans, clip_on=False)
        m = m_end + timedelta(days=1)

    # ── Spine / axis cleanup ──────────────────────────────────
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Title block ───────────────────────────────────────────
    parts = [p.strip() for p in title.split("·", 1)]
    fig.text(0.012, 0.985, parts[0] + (" ·" if len(parts) > 1 else ""),
             ha="left", va="top",
             color=TEXT_COL, fontsize=17, fontfamily="monospace",
             fontweight="bold")
    if len(parts) > 1:
        offset = 0.02 + len(parts[0]) * 0.0068 + 0.026
        fig.text(offset, 0.985, parts[1],
                 ha="left", va="top",
                 color=MUTED, fontsize=17, fontfamily="monospace")

    fig.text(0.012, 0.94,
             subtitle + f"   ·   Generated {datetime.now():%Y-%m-%d %H:%M}",
             ha="left", va="top",
             color=MUTED, fontsize=9.5, fontfamily="monospace")

    # ── Legend — built from active bands ────────────
    seen = {}
    for b in active:
        lbl = b.get("legend_label")
        if lbl and lbl not in seen:
            seen[lbl] = b

    legend_items = []
    for lbl, b in seen.items():
        rgb     = mcolors.to_rgb(b["color"])
        tc_rgb  = mcolors.to_rgb(b.get("text_color", "#ffffff"))
        opt     = b.get("optional", False)
        pattern = b.get("pattern", "solid")
        if opt or pattern == "dashed":
            patch = mpatches.Patch(facecolor=(*rgb, 0.22),
                                   edgecolor=(*rgb, 0.85),
                                   linestyle="--", linewidth=1.2, label=lbl)
        elif pattern == "stripe":
            patch = mpatches.Patch(facecolor=(*rgb, 0.55),
                                   edgecolor=(*rgb, 0.9),
                                   hatch="////",
                                   label=lbl)
        else:
            patch = mpatches.Patch(facecolor=b["color"], label=lbl)
        legend_items.append(patch)

    ax.legend(handles=legend_items,
              loc="lower center",
              bbox_to_anchor=(0.5, -0.18),
              ncol=len(legend_items),
              frameon=True, framealpha=0.9,
              facecolor=SURFACE, edgecolor=BORDER,
              labelcolor=MUTED,
              prop={"family": "monospace", "size": 8.5})

    plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.93])

    fig.savefig(out_png, dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"✓ PNG  → {out_png}")

    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        print(f"✓ PDF  → {out_pdf}")

    plt.show()
    plt.close(fig)
