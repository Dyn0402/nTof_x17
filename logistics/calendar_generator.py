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
BAR_H   = 0.50
BAR_GAP = 0.06
BAR_TOP = 0.32   # distance from cell top to first bar


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


def _wrap_label(label):
    """
    Split a label into two roughly length-balanced lines on a word
    boundary, so narrow bars can use a larger font without overflowing.
    Labels that already carry an explicit "\n" are left as-is.
    """
    if "\n" in label:
        return label
    words = label.split(" ")
    if len(words) < 2:
        return label
    best_i, best_diff = 1, None
    for i in range(1, len(words)):
        l1, l2 = " ".join(words[:i]), " ".join(words[i:])
        diff = abs(len(l1) - len(l2))
        if best_diff is None or diff < best_diff:
            best_i, best_diff = i, diff
    return " ".join(words[:best_i]) + "\n" + " ".join(words[best_i:])


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


def _grid_segs(b_start, b_end, grid_start, grid_end):
    """
    Like _month_segs, but for a Monday-aligned [grid_start, grid_end]
    date range that need not respect calendar-month boundaries — row 0
    is the week containing grid_start, so a month transition can fall
    inside a single row.
    """
    s = max(_to_date(b_start),    grid_start)
    e = min(_inclusive_end(b_end), grid_end)
    if s > e:
        return []
    segs, cur = [], s
    while cur <= e:
        col      = cur.weekday()
        row      = (cur - grid_start).days // 7
        week_end = min(e, cur + timedelta(days=6 - col))
        segs.append({"col_s": col, "col_e": week_end.weekday(), "row": row})
        cur = week_end + timedelta(days=1)
    return segs


def _format_range(d0, d1):
    """Human-readable heading for an inclusive [d0, d1] date range."""
    if d0.year != d1.year:
        return f"{d0:%B %Y} – {d1:%B %Y}"
    if d0.month != d1.month:
        return f"{d0:%B} – {d1:%B %Y}"
    return f"{d0:%B} {d0.day}–{d1.day}, {d1.year}"


def _collect_range(bands, milestones):
    """Active (non-suppressed) bands, plus every date relevant to sizing the view."""
    active = [b for b in bands if not b.get("suppress", False)]
    all_dates = (
        [_to_date(b["start"])        for b in active] +
        [_inclusive_end(b["end"])    for b in active] +
        [_to_date(m["date"])         for m in milestones]
    )
    return active, all_dates


def _draw_event_bars(ax, x0, active, row_segs):
    """
    Slot-stack overlapping segments per row (compact, no empty gaps) and
    render each band as a rounded bar with a label — narrow bars (<= 2
    days wide) get a balanced two-line wrap so a larger font still fits.
    `row_segs` maps row -> seg dicts with col_s/col_e/row/bi keys, as
    populated by the caller via _month_segs / _grid_segs.
    """
    for segs in row_segs.values():
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
                span = s["col_e"] - s["col_s"] + 1
                if span <= 2:
                    label, fsize = _wrap_label(b["label"]), 10.0
                else:
                    label, fsize = b["label"].replace("\n", " "), 10.5
                tc = b.get("text_color", "#ffffff")
                ax.text((xl + xr) / 2, yt + BAR_H / 2,
                        label,
                        ha="center", va="center",
                        color=MUTED if opt else tc,
                        fontsize=fsize, fontfamily="monospace",
                        fontweight="bold", zorder=5, clip_on=True,
                        linespacing=1.1,
                        path_effects=[pe.withStroke(linewidth=1.0,
                                                    foreground=(*rgb, 0.30))])


def _draw_milestones(ax, x0, milestones, locate):
    """
    Draw milestone diamonds + labels at the bottom-centre of their day cell.
    `locate(d)` maps a milestone date to its (col, row) in the grid, or
    returns None for dates outside the displayed range (skipped).
    """
    for m in milestones:
        d   = _to_date(m["date"])
        loc = locate(d)
        if loc is None:
            continue
        col, row = loc
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
                color=m["color"], fontsize=8, fontfamily="monospace",
                fontweight="bold", zorder=7,
                path_effects=[pe.withStroke(linewidth=1.5, foreground=SURFACE)])


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

    # ── Build per-row segment list, then draw bands + milestones ──
    active = [b for b in bands if not b.get("suppress", False)]

    row_segs = defaultdict(list)   # row → list of seg dicts
    for bi, b in enumerate(active):
        for seg in _month_segs(b["start"], b["end"], year, month):
            seg["bi"] = bi
            row_segs[seg["row"]].append(seg)

    _draw_event_bars(ax, x0, active, row_segs)

    def _locate(d):
        if d.year != year or d.month != month:
            return None
        return d.weekday(), (d.day + fdow - 1) // 7

    _draw_milestones(ax, x0, milestones, _locate)


# ══════════════════════════════════════════════════════════════
#  CONTIGUOUS WEEK-GRID DRAWING (no month split)
# ══════════════════════════════════════════════════════════════

def _draw_grid(ax, grid_start, grid_end, bands, milestones):
    """
    Draw one Monday→Sunday week grid spanning [grid_start, grid_end]
    (inclusive), ignoring calendar-month boundaries — a month transition
    can land inside a single row (e.g. "Jun 29 ... Jul 5"). The first day
    of each month is called out with its abbreviated name.
    """
    mw = 7 * CELL_W

    # ── DOW headers ───────────────────────────────────────────
    for col, lbl in enumerate(DOW_LABELS):
        is_we = col >= 5
        ax.text(col * CELL_W + CELL_W / 2, -DOW_H / 2,
                lbl,
                ha="center", va="center",
                color=MUTED if is_we else TEXT_COL,
                fontsize=8, fontfamily="monospace", fontweight="bold")

    # ── Cell backgrounds and day numbers ─────────────────────
    n_days = (grid_end - grid_start).days + 1
    for offset in range(n_days):
        d   = grid_start + timedelta(days=offset)
        col = offset % 7
        row = offset // 7
        xl  = col * CELL_W
        yt  = row * CELL_H
        is_we = col >= 5
        is_month_start = d.day == 1

        if is_we:
            ax.add_patch(mpatches.Rectangle(
                (xl, yt), CELL_W, CELL_H,
                facecolor=WEEKEND, edgecolor="none", zorder=0))

        ax.add_patch(mpatches.Rectangle(
            (xl, yt), CELL_W, CELL_H,
            facecolor="none", edgecolor=BORDER, linewidth=0.6, zorder=1))

        label = f"{d:%b} {d.day}" if is_month_start else str(d.day)
        ax.text(xl + 0.11, yt + 0.11, label,
                ha="left", va="top",
                color=ACCENT if is_month_start else (MUTED if is_we else TEXT_COL),
                fontsize=7.5, fontfamily="monospace",
                fontweight="bold" if is_month_start else "normal")

    # ── Build per-row segment list ────────────────────────────
    active = [b for b in bands if not b.get("suppress", False)]

    row_segs = defaultdict(list)   # row → list of seg dicts
    for bi, b in enumerate(active):
        for seg in _grid_segs(b["start"], b["end"], grid_start, grid_end):
            seg["bi"] = bi
            row_segs[seg["row"]].append(seg)

    _draw_event_bars(ax, 0, active, row_segs)

    def _locate(d):
        if d < grid_start or d > grid_end:
            return None
        offset = (d - grid_start).days
        return offset % 7, offset // 7

    _draw_milestones(ax, 0, milestones, _locate)


# ══════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════

def draw_calendar(title, subtitle, bands, milestones, out_png, out_pdf=None):
    # Determine months to show from band + milestone dates
    active, all_dates = _collect_range(bands, milestones)
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

    _finish_figure(fig, ax, title, subtitle, active, out_png, out_pdf)


def _finish_figure(fig, ax, title, subtitle, active, out_png, out_pdf):
    """Shared title/subtitle/legend rendering and PNG/PDF export for calendar views."""
    # ── Title block ───────────────────────────────────────────
    # The second segment is positioned from the *measured* render width of
    # the first (rather than an estimate from character count), so the
    # spacing stays correct regardless of the figure's width.
    parts = [p.strip() for p in title.split("·", 1)]
    head = fig.text(0.012, 0.985, parts[0] + (" ·" if len(parts) > 1 else ""),
                    ha="left", va="top",
                    color=TEXT_COL, fontsize=17, fontfamily="monospace",
                    fontweight="bold")
    if len(parts) > 1:
        fig.canvas.draw()
        bbox = head.get_window_extent(renderer=fig.canvas.get_renderer())
        x_fig = fig.transFigure.inverted().transform((bbox.x1, bbox.y1))[0]
        fig.text(x_fig + 0.012, 0.985, parts[1],
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


# ══════════════════════════════════════════════════════════════
#  PUBLIC API — contiguous week-grid (no month split)
# ══════════════════════════════════════════════════════════════

def draw_week_calendar(title, subtitle, bands, milestones, out_png, out_pdf=None,
                       start_date=None, end_date=None):
    """
    Single contiguous week-row grid that does not split at calendar-month
    boundaries — rows run Monday→Sunday across the full range, so a month
    transition can land inside one row (e.g. "Jun 29 ... Jul 5").

    start_date / end_date optionally pin the displayed range; otherwise it
    is derived from the band + milestone dates and expanded outward to full
    Monday→Sunday weeks.
    """
    active, all_dates = _collect_range(bands, milestones)
    min_d: date = _to_date(start_date) if start_date is not None else min(all_dates)
    max_d: date = _to_date(end_date)   if end_date   is not None else max(all_dates)

    grid_start = min_d - timedelta(days=min_d.weekday())
    grid_end   = max_d + timedelta(days=6 - max_d.weekday())
    n_weeks    = (grid_end - grid_start).days // 7 + 1

    mw      = 7 * CELL_W
    total_h = n_weeks * CELL_H
    top_h   = DOW_H + TITLE_H

    fig_w = max(10, mw * 0.88 + 2.0)
    fig_h = max(8,  (total_h + top_h) * 0.88 + 2.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)

    # y-axis inverted: ylim = (bottom_y, top_y)
    ax.set_xlim(-0.4, mw + 0.4)
    ax.set_ylim(total_h + 0.3, -(top_h + 0.3))
    ax.axis("off")

    # ── Range heading (replaces the per-month title) ─────────
    ax.text(mw / 2, -(DOW_H + TITLE_H / 2),
            _format_range(grid_start, grid_end),
            ha="center", va="center",
            color=ACCENT, fontsize=13, fontfamily="monospace",
            fontweight="bold")

    _draw_grid(ax, grid_start, grid_end, bands, milestones)

    # Outer accent border around the grid
    ax.add_patch(mpatches.Rectangle(
        (0, 0), mw, total_h,
        facecolor="none", edgecolor=ACCENT, linewidth=0.9, zorder=2))

    _finish_figure(fig, ax, title, subtitle, active, out_png, out_pdf)
