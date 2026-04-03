#!/usr/bin/env python3
"""
May_Test_Prep.py
────────────────
Timeline: MXv2 detector preparation leading up to the EAR2 beam test.
Apr 3 → May 18, 2026.

Run:
    python May_Test_Prep.py
"""

from datetime import date
from timeline_generator import draw_timeline

# ══════════════════════════════════════════════════════════════
#  SCHEDULE CONFIG
# ══════════════════════════════════════════════════════════════

TITLE    = "n_TOF X17 · May 2026 Beam Test"
SUBTITLE = "MXv2 Detector Preparation & EAR2 Campaign"

START_DATE = date(2026, 4, 3)
END_DATE   = date(2026, 5, 18)

MILESTONES = [
    {"date": date(2026, 4, 3),  "label": "Today",                    "color": "#94a3b8"},
    {"date": date(2026, 4, 15), "label": "CEA Go-No-Go Meeting",     "color": "white"},
    {"date": date(2026, 4, 17), "label": "Test DAQ",                 "color": "#d97706"},
    {"date": date(2026, 4, 20), "label": "Screen arrives",           "color": "#7c3aed"},
    {"date": date(2026, 4, 28), "label": "Measure Capacitance",      "color": "#7c3aed"},
    {"date": date(2026, 5, 6),  "label": "Pack for CERN",            "color": "#f59e0b"},
    {"date": date(2026, 5, 7),  "label": "Travel day",               "color": "#94a3b8"},
    {"date": date(2026, 5, 8),  "label": "Arrive CERN\nSetup begins","color": "#22d3ee"},
    {"date": date(2026, 5, 11), "label": "Beam time\nstarts ★",      "color": "#4ade80"},
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
    # ── Personnel ─────────────────────────────────────────────
    {
        "label":        "Stephan away",
        "start":        date(2026, 4, 6),
        "end":          date(2026, 4, 10),
        "color":        "#475569",
        "text_color":   "#cbd5e1",
        "pattern":      "stripe",
        "legend_label": "Personnel absence",
    },
    {
        "label":        "Dylan away",
        "start":        date(2026, 4, 9),
        "end":          date(2026, 4, 16),
        "color":        "#334155",
        "text_color":   "#94a3b8",
        "pattern":      "stripe",
        "legend_label": "Personnel absence",
        "suppress":     False,      # flip to False to restore
    },

    # ── DAQ work ──────────────────────────────────────────────
    {
        "label":        "DAQ prep",
        "start":        date(2026, 4, 13),
        "end":          date(2026, 4, 16),
        "color":        "#b45309",
        "text_color":   "#fef3c7",
        "pattern":      "solid",
        "legend_label": "DAQ work",
    },
    {
        "label":        "DAQ testing",
        "start":        date(2026, 4, 17),
        "end":          date(2026, 4, 17),
        "color":        "#d97706",
        "text_color":   "#fef3c7",
        "pattern":      "solid",
        "legend_label": "DAQ work",
    },

    # ── Hardware ──────────────────────────────────────────────
    {
        "label":        "Build MXv2 Det. 1\nBuild new drift",
        "start":        date(2026, 4, 20),
        "end":          date(2026, 4, 28),
        "color":        "#7c3aed",
        "text_color":   "#ede9fe",
        "pattern":      "solid",
        "legend_label": "Detector build",
    },
    {
        "label":        "",
        "start":        date(2026, 4, 29),
        "end":          date(2026, 4, 29),
        "color":        "#7c3aed",
        "text_color":   "#ede9fe",
        "pattern":      "stripe",
        "legend_label": "Detector build",
    },

    # ── Testing ───────────────────────────────────────────────
    {
        "label":        "",
        "start":        date(2026, 4, 30),
        "end":          date(2026, 4, 30),
        "color":        "#0e7490",
        "text_color":   "#cffafe",
        "pattern":      "stripe",
        "legend_label": "Detector test",
    },
    {
        "label":        "Cosmic tests Det. 1",
        "start":        date(2026, 5, 1),
        "end":          date(2026, 5, 5),
        "color":        "#0e7490",
        "text_color":   "#cffafe",
        "pattern":      "solid",
        "legend_label": "Detector test",
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
]

OUTPUT_PNG = "May_Test_Prep.png"
OUTPUT_PDF = "May_Test_Prep.pdf"   # set to None to skip

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
