#!/usr/bin/env python3
"""
July_Prep.py
────────────
Timeline: New detector build and n_TOF X17 physics run.
Jun 8 → Jul 14, 2026.

Run:
    python July_Prep.py
"""

from datetime import datetime
from timeline_generator import draw_timeline

# ══════════════════════════════════════════════════════════════
#  SCHEDULE CONFIG
# ══════════════════════════════════════════════════════════════

TITLE    = "n_TOF X17 · July 2026 Physics Run"
SUBTITLE = "Detector Build & EAR2 X17 Campaign"

START_DATE = datetime(2026, 6, 8)
END_DATE   = datetime(2026, 7, 6)

MILESTONES = [
    {"date": datetime(2026, 6, 26), "label": "Leave for CERN",        "color": "#f59e0b"},
    {"date": datetime(2026, 6, 29), "label": "EAR2 Install",          "color": "#22d3ee"},
    {"date": datetime(2026, 7, 1),  "label": "X17 Physics Start ★",   "color": "#4ade80"},
]

# Fields:
#   label       displayed inside bar
#   start/end   inclusive date range
#   color       bar fill colour (hex)
#   text_color  label colour
#   pattern     "solid" | "stripe" | "dashed"
#   optional    True → dashed border, reduced opacity
#   suppress    True → hidden (easy toggle, set False to restore)
BANDS = [
    # ── Hardware ──────────────────────────────────────────────
    {
        "label":        "Serigraphy Det 1 - 4",
        "start":        datetime(2026, 6, 8),
        "end":          datetime(2026, 6, 10),
        "color":        "#7c3aed",
        "text_color":   "#ede9fe",
        "pattern":      "solid",
        "legend_label": "Serigraphy",
    },

    {
        "label":        "Bulk Det 1 + 2",
        "start":        datetime(2026, 6, 10),
        "end":          datetime(2026, 6, 12),
        "color":        "#8b5cf6",
        "text_color":   "#ede9fe",
        "pattern":      "solid",
        "legend_label": "Bulk",
    },

    {
        "label":        "Integrate Det 1",
        "start":        datetime(2026, 6, 12),
        "end":          datetime(2026, 6, 13),
        "color":        "#6d28d9",
        "text_color":   "#ede9fe",
        "pattern":      "solid",
        "legend_label": "Integrate",
    },

    {
        "label":        "Cosmics Det 1",
        "start":        datetime(2026, 6, 13),
        "end":          datetime(2026, 6, 15),  # end is exclusive for datetime; covers Jun 27–28
        "color":        "#0e7490",
        "text_color":   "#cffafe",
        "pattern":      "solid",
        "legend_label": "Detector test",
    },

    {
        "label":        "Bulk Det 3 + 4",
        "start":        datetime(2026, 6, 15),
        "end":          datetime(2026, 6, 18),
        "color":        "#8b5cf6",
        "text_color":   "#ede9fe",
        "pattern":      "solid",
        "legend_label": "Bulk",
    },

    {
        "label":        "Integrate Det 2",
        "start":        datetime(2026, 6, 15),
        "end":          datetime(2026, 6, 16),
        "color":        "#6d28d9",
        "text_color":   "#ede9fe",
        "pattern":      "solid",
        "legend_label": "Integrate",
    },

    {
        "label":        "Cosmics Det 2",
        "start":        datetime(2026, 6, 16),
        "end":          datetime(2026, 6, 17),  # end is exclusive for datetime; covers Jun 27–28
        "color":        "#0e7490",
        "text_color":   "#cffafe",
        "pattern":      "solid",
        "legend_label": "Detector test",
    },

    {
        "label":        "Integrate Det 3 + 4",
        "start":        datetime(2026, 6, 17),
        "end":          datetime(2026, 6, 20),
        "color":        "#6d28d9",
        "text_color":   "#ede9fe",
        "pattern":      "solid",
        "legend_label": "Integrate",
    },

    {
        "label":        "Cosmics Det 3",
        "start":        datetime(2026, 6, 18),
        "end":          datetime(2026, 6, 20),  # end is exclusive for datetime; covers Jun 27–28
        "color":        "#0e7490",
        "text_color":   "#cffafe",
        "pattern":      "solid",
        "legend_label": "Detector test",
    },

    {
        "label":        "Cosmics Det 4",
        "start":        datetime(2026, 6, 20),
        "end":          datetime(2026, 6, 22),  # end is exclusive for datetime; covers Jun 27–28
        "color":        "#0e7490",
        "text_color":   "#cffafe",
        "pattern":      "solid",
        "legend_label": "Detector test",
    },

    {
        "label":        "Remake any Detectors Necessary",
        "start":        datetime(2026, 6, 22),
        "end":          datetime(2026, 6, 26),
        "color":        "#a78bfa",
        "text_color":   "#2e1065",
        "pattern":      "solid",
        "legend_label": "Detector rework",
    },

    # ── Travel ────────────────────────────────────────────────
    {
        "label":        "Leave for CERN",
        "start":        datetime(2026, 6, 26),
        "end":          datetime(2026, 6, 27),  # end exclusive for datetime; fills Jun 26 column
        "color":        "#f59e0b",
        "text_color":   "#1c1000",
        "pattern":      "solid",
        "legend_label": "Travel",
    },

    # ── Testing ───────────────────────────────────────────────
    {
        "label":        "Cosmics at\nn_TOF",
        "start":        datetime(2026, 6, 27),
        "end":          datetime(2026, 6, 29),  # end is exclusive for datetime; covers Jun 27–28
        "color":        "#0e7490",
        "text_color":   "#cffafe",
        "pattern":      "solid",
        "legend_label": "Detector test",
    },

    # ── Accelerator ───────────────────────────────────────────
    {
        "label":        "PS Beam Stop",
        "start":        datetime(2026, 6, 29),
        "end":          datetime(2026, 7, 1),
        "color":        "#991b1b",
        "text_color":   "#fee2e2",
        "pattern":      "stripe",
        "legend_label": "Accelerator stop",
    },

    # ── n_TOF Campaign ────────────────────────────────────────
    {
        "label":        "X17 Physics Run",
        "start":        datetime(2026, 7, 1),
        "end":          datetime(2026, 7, 6),
        "color":        "#15803d",
        "text_color":   "#bbf7d0",
        "pattern":      "solid",
        "legend_label": "n_TOF campaign",
        "extends_right": True,
    },
]

OUTPUT_PNG = "July_Prep.png"
OUTPUT_PDF = "July_Prep.pdf"   # set to None to skip

# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    draw_timeline(
        title=TITLE,
        subtitle=SUBTITLE,
        start_date=START_DATE,
        end_date=END_DATE,
        bands=BANDS,
        milestones=MILESTONES,
        out_png=OUTPUT_PNG,
        out_pdf=OUTPUT_PDF,
    )
