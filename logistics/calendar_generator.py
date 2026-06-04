#!/usr/bin/env python3
"""
calendar_generator.py
─────────────────────
Month-grid calendar view for experiment scheduling.
Bands and milestones share the same schema as timeline_generator.

datetime end-dates follow the same exclusive-midnight convention as
the Gantt view, so the same schedule dict works for both generators.

Usage:
    from calendar_generator import draw_calendar
"""

import calendar as _cal
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict

# ── Colour palette (mirrors timeline_generator) ───────────────
BG       = "#0a0f1a"
SURFACE  = "#111827"
BORDER   = "#1e2d3d"
TEXT_COL = "#e2e8f0"
MUTED    = "#64748b"
ACCENT   = "#38bdf8"
WEEKEND  = "#161f2e"

DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ── Layout constants (data units) ────────────────────────────
CELL_W  = 1.8    # day-cell width
CELL_H  = 1.8    # day-cell height
DOW_H   = 0.42   # day-of-week header row height
TITLE_H = 0.52   # month title height
GAP_X   = 2.0    # horizontal gap between months

# Event bar sizing within a cell
BAR_H   = 0.30
BAR_GAP = 0.05
BAR_TOP = 0.30   # distance from cell top to first bar


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def _to_date(d):
    """datetime or date → date."""
    return d.date() if isinstance(d, datetime) else d


def _inclusive_end(d):
    """
    Convert a band end to an inclusive date for calendar display.
    datetime objects use an exclusive midnight convention (the same as the
    Gantt generator), so subtract one day.  Plain date objects are already
    inclusive.
    """
    if isinstance(d, datetime):
        return d.date() - timedelta(days=1)
    return d


def _month_segs(b_start, b_end, year, month):
    """
    Split the inclusive date range [b_start, b_end] into
    (col_start, col_end, row) week-row segments within the given month.
    Mon = col 0, Sun = col 6.
    """
    first = date(year, month, 1)
    last  = date(year, month, _cal.monthrange(year, month)[1])
    s = max(_to_date(b_start),    first)
    e = min(_inclusive_end(b_end), last)
    if s > e:
        return []
    fdow = first.weekday()
    segs, cur = [], s
    while cur <= e:
        col      = cur.weekday()
        row      = (cur.day + fdow - 1) // 7
        week_end = min(e, cur + timedelta(days=6 - col))
        segs.append({"col_s": col, "col_e": week_end.weekday(), "row": row})
        cur = week_end + timedelta(days=1)
    return segs


# ══════════════════════════════════════════════════════════════
#  MONTH DRAWING
# ══════════════════════════════════════════════════════════════

def _draw_month(ax, year, month, x0, bands, milestones):
    """Draw one month grid with event bars and milestone diamonds."""
    first = date(year, month, 1)
    last  = date(year, month, _cal.monthrange(year, month)[1])
    fdow  = first.weekday()
    mw    = 7 * CELL_W

    # ── Month title ───────────────────────────────────────────
    ax.text(x0 + mw / 2, -(DOW_H + TITLE_H / 2),
            first.strftime("%B %Y"),
            ha="center", va="center",
            color=ACCENT, fontsize=13, fontfamily="monospace",
            fontweight="bold")

    # ── DOW headers ───────────────────────────────────────────
    for col, lbl in enumerate(DOW_LABELS):
        is_we = col >= 5
        ax.text(x0 + col * CELL_W + CELL_W / 2, -DOW_H / 2,
                lbl,
                ha="center", va="center",
                color=MUTED if is_we else TEXT_COL,
                fontsize=8, fontfamily="monospace", fontweight="bold")

    # ── Cell backgrounds and day numbers ─────────────────────
    for day_num in range(1, last.day + 1):
        d   = date(year, month, day_num)
        col = d.weekday()
        row = (day_num + fdow - 1) // 7
        xl  = x0 + col * CELL_W
        yt  = row * CELL_H
        is_we = col >= 5

        if is_we:
            ax.add_patch(mpatches.Rectangle(
                (xl, yt), CELL_W, CELL_H,
                facecolor=WEEKEND, edgecolor="none", zorder=0))

        ax.add_patch(mpatches.Rectangle(
            (xl, yt), CELL_W, CELL_H,
            facecolor="none", edgecolor=BORDER, linewidth=0.6, zorder=1))

        ax.text(xl + 0.11, yt + 0.11, str(day_num),
                ha="left", va="top",
                color=MUTED if is_we else TEXT_COL,
                fontsize=7.5, fontfamily="monospace")

    # ── Build per-row segment list ────────────────────────────
    active = [b for b in bands if not b.get("suppress", False)]

    row_segs = defaultdict(list)   # row → list of seg dicts
    for bi, b in enumerate(active):
        for seg in _month_segs(b["start"], b["end"], year, month):
            seg["bi"] = bi
            row_segs[seg["row"]].append(seg)

    # Greedy slot assignment per row (compact: no empty gaps)
    for row_idx, segs in row_segs.items():
        segs.sort(key=lambda s: s["col_s"])
        slot_end_col = []
        for s in segs:
            placed = False
            for i, ec in enumerate(slot_end_col):
                if s["col_s"] > ec:
                    s["slot"] = i
                    slot_end_col[i] = s["col_e"]
                    placed = True
                    break
            if not placed:
                s["slot"] = len(slot_end_col)
                slot_end_col.append(s["col_e"])

    # ── Draw event bars ───────────────────────────────────────
    for segs in row_segs.values():
        for s in segs:
            b     = active[s["bi"]]
            slot  = s["slot"]
            row   = s["row"]
            xl    = x0 + s["col_s"] * CELL_W + 0.07
            xr    = x0 + (s["col_e"] + 1) * CELL_W - 0.07
            yt    = row * CELL_H + BAR_TOP + slot * (BAR_H + BAR_GAP)
            width = xr - xl

            # Skip if bar would bleed out of cell bottom
            if yt + BAR_H > (row + 1) * CELL_H - 0.06:
                continue

            rgb   = mcolors.to_rgb(b["color"])
            pat   = b.get("pattern", "solid")
            opt   = b.get("optional", False)
            alpha = 0.22 if opt else (0.55 if pat in ("stripe", "dashed") else 1.0)

            rect = FancyBboxPatch(
                (xl, yt), width, BAR_H,
                boxstyle="round,pad=0.0",
                facecolor=(*rgb, alpha),
                edgecolor=(*rgb, 0.85) if pat == "dashed" else "none",
                linestyle="--" if pat == "dashed" else "-",
                linewidth=0.8, zorder=3)
            ax.add_patch(rect)

            if pat == "stripe":
                htch = FancyBboxPatch(
                    (xl, yt), width, BAR_H,
                    boxstyle="round,pad=0.0",
                    facecolor="none",
                    edgecolor=(*mcolors.to_rgb(b.get("text_color", "#ffffff")), 0.20),
                    hatch="////", linewidth=0, zorder=4)
                ax.add_patch(htch)

            if width > 0.25:
                label = b["label"].replace("\n", " ")
                tc    = b.get("text_color", "#ffffff")
                ax.text((xl + xr) / 2, yt + BAR_H / 2,
                        label,
                        ha="center", va="center",
                        color=MUTED if opt else tc,
                        fontsize=5.5, fontfamily="monospace",
                        fontweight="bold", zorder=5, clip_on=True,
                        path_effects=[pe.withStroke(linewidth=1.0,
                                                    foreground=(*rgb, 0.30))])

    # ── Milestone diamonds (bottom-centre of day cell) ────────
    for m in milestones:
        d = _to_date(m["date"])
        if d.year != year or d.month != month:
            continue
        col = d.weekday()
        row = (d.day + fdow - 1) // 7
        xc  = x0 + col * CELL_W + CELL_W / 2
        yc  = (row + 1) * CELL_H - 0.20   # near bottom of cell
        rgb = mcolors.to_rgb(m["color"])
        ax.plot(xc, yc, marker="D", markersize=5,
                color=m["color"], zorder=7,
                markeredgecolor=(*rgb, 0.5), markeredgewidth=0.7)
        # Label sits above the diamond
        ax.text(xc, yc - 0.08,
                m["label"].split("★")[0].strip().split("\n")[0],
                ha="center", va="bottom",
                color=m["color"], fontsize=4.8, fontfamily="monospace",
                fontweight="bold", zorder=7,
                path_effects=[pe.withStroke(linewidth=1.5, foreground=SURFACE)])


# ══════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════

def draw_calendar(title, subtitle, bands, milestones, out_png, out_pdf=None):
    # Determine months to show from band + milestone dates
    active = [b for b in bands if not b.get("suppress", False)]
    all_dates = (
        [_to_date(b["start"])        for b in active] +
        [_inclusive_end(b["end"])    for b in active] +
        [_to_date(m["date"])         for m in milestones]
    )
    min_d = min(all_dates)
    max_d = max(all_dates)

    months = []
    cur = min_d.replace(day=1)
    while cur <= max_d:
        months.append((cur.year, cur.month))
        nxt = cur.month + 1
        cur = cur.replace(year=cur.year + (1 if nxt > 12 else 0),
                          month=nxt if nxt <= 12 else 1)

    n_m  = len(months)
    mw   = 7 * CELL_W
    max_wk = max(
        (_cal.monthrange(y, m)[1] + date(y, m, 1).weekday() - 1) // 7 + 1
        for y, m in months)

    total_w = n_m * mw + (n_m - 1) * GAP_X
    total_h = max_wk * CELL_H
    top_h   = DOW_H + TITLE_H    # header height above y=0

    fig_w = max(18, total_w * 0.88 + 2.0)
    fig_h = max(8,  (total_h + top_h) * 0.88 + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)

    # y-axis inverted: ylim = (bottom_y, top_y)
    ax.set_xlim(-0.4, total_w + 0.4)
    ax.set_ylim(total_h + 0.3, -(top_h + 0.3))
    ax.axis("off")

    for mi, (y, m) in enumerate(months):
        x0 = mi * (mw + GAP_X)
        _draw_month(ax, y, m, x0, bands, milestones)

        # Outer accent border around each month grid
        ax.add_patch(mpatches.Rectangle(
            (x0, 0), mw, max_wk * CELL_H,
            facecolor="none", edgecolor=ACCENT, linewidth=0.9, zorder=2))

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

    # ── Legend ────────────────────────────────────────────────
    seen = {}
    for b in active:
        lbl = b.get("legend_label")
        if lbl and lbl not in seen:
            seen[lbl] = b

    legend_items = []
    for lbl, b in seen.items():
        rgb = mcolors.to_rgb(b["color"])
        pat = b.get("pattern", "solid")
        opt = b.get("optional", False)
        if opt or pat == "dashed":
            patch = mpatches.Patch(facecolor=(*rgb, 0.22),
                                   edgecolor=(*rgb, 0.85),
                                   linestyle="--", linewidth=1.2, label=lbl)
        elif pat == "stripe":
            patch = mpatches.Patch(facecolor=(*rgb, 0.55),
                                   edgecolor=(*rgb, 0.9),
                                   hatch="////", label=lbl)
        else:
            patch = mpatches.Patch(facecolor=b["color"], label=lbl)
        legend_items.append(patch)

    ax.legend(handles=legend_items,
              loc="lower center",
              bbox_to_anchor=(0.5, -0.10),
              ncol=len(legend_items),
              frameon=True, framealpha=0.9,
              facecolor=SURFACE, edgecolor=BORDER,
              labelcolor=MUTED,
              prop={"family": "monospace", "size": 8.5})

    plt.tight_layout(rect=[0.0, 0.04, 1.0, 0.93])

    fig.savefig(out_png, dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"✓ PNG  → {out_png}")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        print(f"✓ PDF  → {out_pdf}")

    plt.show()
    plt.close(fig)
