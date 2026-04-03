#!/usr/bin/env python3
"""
May_Test.py
───────────
Timeline: EAR2 beam test at CERN.
May 7 (travel) → May 18, 2026.

Run:
    python May_Test.py
"""

from datetime import date, datetime
from timeline_generator import draw_timeline

# ══════════════════════════════════════════════════════════════
#  SCHEDULE CONFIG
# ══════════════════════════════════════════════════════════════

TITLE    = "n_TOF X17 · EAR2 Beam Test"
SUBTITLE = "CERN Campaign — May 2026"

START_DATE = date(2026, 5, 7)
END_DATE   = date(2026, 5, 18)

MILESTONES = [
    {"date": date(2026, 5, 7),  "label": "Travel day",               "color": "#94a3b8"},
    {"date": date(2026, 5, 8),  "label": "Arrive CERN\nSetup begins","color": "#22d3ee"},
]

BANDS = [
    # ── Travel & Logistics ────────────────────────────────────
    {
        "label":        "Travel",
        "start":        date(2026, 5, 7),
        "end":          date(2026, 5, 7),
        "color":        "#94a3b8",
        "text_color":   "#0f172a",
        "pattern":      "solid",
        "legend_label": "Travel",
    },

    # ── n_TOF Campaign ────────────────────────────────────────
    {
        "label":        "CERN setup",
        "start":        date(2026, 5, 8),
        "end":          date(2026, 5, 10),
        "color":        "#065f46",
        "text_color":   "#a7f3d0",
        "pattern":      "solid",
        "legend_label": "n_TOF campaign",
    },
    {
        "label":        "EAR2 Beam time",
        "start":        date(2026, 5, 11),
        "end":          date(2026, 5, 18),
        "color":        "#15803d",
        "text_color":   "#bbf7d0",
        "pattern":      "solid",
        "legend_label": "n_TOF campaign",
    },
    {
        "label":        "Cosmics at CERN\n(if ready)",
        "start":        date(2026, 5, 8),
        "end":          date(2026, 5, 10),
        "color":        "#0e7490",
        "text_color":   "#cffafe",
        "pattern":      "dashed",
        "optional":     True,
        "legend_label": "Conditional / TBD",
    },
    {
        "label":        "Setup\nFlush",
        "start":        datetime(2026, 5, 11, 8),
        "end":          datetime(2026, 5, 12, 12),
        "color":        "#15803d",
        "text_color":   "#bbf7d0",
        "pattern":      "solid",
        "legend_label": "n_TOF campaign",
    },
]

OUTPUT_PNG = "May_Test.png"
OUTPUT_PDF = "May_Test.pdf"   # set to None to skip

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
