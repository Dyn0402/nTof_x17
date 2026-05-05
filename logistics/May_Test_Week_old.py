#!/usr/bin/env python3
"""
May_Test_Week.py
────────────────
Hour-by-hour week view of the EAR2 beam test campaign.
Each day is a column; y-axis = 00:00 (top) → 24:00 (bottom).

Run:
    python May_Test_Week.py
"""

from datetime import datetime
from May_Test import START_DATE, END_DATE
from week_timeline_generator import draw_week_timeline

TITLE    = "n_TOF X17 · EAR2 Beam Test"
SUBTITLE = "CERN Campaign — May 2026 · Hour-by-Hour Schedule"

OUTPUT_PNG = "May_Test_Week.png"
OUTPUT_PDF = "May_Test_Week.pdf"   # set to None to skip

DAY_WIDTH  = 1.55   # inches per day column
FIG_HEIGHT = 9.0

# ── Per-category visual styles ────────────────────────────────
# Edit here to restyle an entire category at once.
STYLES = {
    "Travel":        {"color": "#94a3b8", "text_color": "#0f172a", "pattern": "solid"},
    "Setup":         {"color": "#d97706", "text_color": "#fef3c7", "pattern": "solid"},
    "Flush":         {"color": "#0891b2", "text_color": "#e0f7ff", "pattern": "stripe"},
    "Testing":       {"color": "#7c3aed", "text_color": "#ede9fe", "pattern": "solid"},
    "Measurement":   {"color": "#2563eb", "text_color": "#bfdbfe", "pattern": "solid"},
    "Drift Switch":  {"color": "#dc2626", "text_color": "#fee2e2", "pattern": "solid"},
    "Sample Switch": {"color": "#db2777", "text_color": "#fce7f3", "pattern": "solid"},
    "Gas Switch":    {"color": "#16a34a", "text_color": "#f0fdf4", "pattern": "solid"},
    "Event":         {"color": "#475569", "text_color": "#cbd5e1", "pattern": "stripe"},
    "Cosmics (TBD)": {"color": "#0e7490", "text_color": "#cffafe", "pattern": "dashed"},
}

BANDS = [
    # ── Travel & Logistics ────────────────────────────────────
    {"label": "Travel\nCEA to CERN",           "legend_label": "Travel",
     "start": datetime(2026, 5, 8, 10),        "end": datetime(2026, 5, 8, 17)},

    # ── Rack room prep ────────────────────────────────────────
    {"label": "Setup\nRack Room\nFor Cosmics\n30mm Drift",  "legend_label": "Setup",
     "start": datetime(2026, 5, 8, 17, 30),    "end": datetime(2026, 5, 8, 20)},
    {"label": "Flush\nArgon CO2",              "legend_label": "Flush",
     "start": datetime(2026, 5, 8, 20, 30),    "end": datetime(2026, 5, 9, 0)},
    {"label": "Cosmics\nResist HV Scan\nRack Room\nArgon CO2", "legend_label": "Cosmics (TBD)", "optional": True,
     "start": datetime(2026, 5, 9, 0, 30),     "end": datetime(2026, 5, 9, 14)},
    {"label": "Flush\nArgon CF4",          "legend_label": "Flush",
     "start": datetime(2026, 5, 9, 14, 30),    "end": datetime(2026, 5, 9, 15, 30)},
    {"label": "Test\nScintillator\nMechanics",          "legend_label": "Setup",
     "start": datetime(2026, 5, 9, 15, 45),    "end": datetime(2026, 5, 9, 18, 45)},
    {"label": "Flush\nArgon CF4",          "legend_label": "Flush",
     "start": datetime(2026, 5, 9, 19, 0),    "end": datetime(2026, 5, 9, 20, 0)},
    {"label": "Cosmics\nResist HV Scan\nRack Room\nArgon CF4","legend_label": "Cosmics (TBD)", "optional": True,
     "start": datetime(2026, 5, 9, 20, 30),    "end": datetime(2026, 5, 10, 11)},
    {"label": "Flush\nHelium Ethane",          "legend_label": "Flush",
     "start": datetime(2026, 5, 10, 11, 30),   "end": datetime(2026, 5, 10, 18)},
    {"label": "Cosmics\nResist HV Scan\nRack Room\nHelium Ethane","legend_label": "Cosmics (TBD)", "optional": True,
     "start": datetime(2026, 5, 10, 18, 30),   "end": datetime(2026, 5, 11, 8)},

    # ── EAR2 setup & commissioning ────────────────────────────
    {"label": "Setup\nEAR2\n30mm Drift",       "legend_label": "Setup",
     "start": datetime(2026, 5, 11, 8, 30),    "end": datetime(2026, 5, 11, 14)},
    {"label": "Flush\nHelium Ethane",          "legend_label": "Flush",
     "start": datetime(2026, 5, 11, 14, 30),   "end": datetime(2026, 5, 11, 17)},
    {"label": "Mesh Switch Testing\nHelium Ethane\n30mm Drift\nCarbon Capsule", "legend_label": "Testing",
     "start": datetime(2026, 5, 11, 17, 30),   "end": datetime(2026, 5, 11, 21)},

    # ── Measurements ──────────────────────────────────────────
    {"label": "Resist HV Scan\nHelium Ethane\n30mm Drift\nCarbon Capsule", "legend_label": "Measurement",
     "start": datetime(2026, 5, 11, 21, 30),   "end": datetime(2026, 5, 12, 9, 30)},
    {"label": "Drift Switch\n30mm to 3mm",     "legend_label": "Drift Switch",
     "start": datetime(2026, 5, 12, 10, 0),    "end": datetime(2026, 5, 12, 12)},
    {"label": "Flush\nHelium Ethane",          "legend_label": "Flush",
     "start": datetime(2026, 5, 12, 12, 30),   "end": datetime(2026, 5, 12, 15)},
    {"label": "Resist HV Scan\nHelium Ethane\n3mm Drift\nB4C", "legend_label": "Measurement",
     "start": datetime(2026, 5, 12, 15, 30),   "end": datetime(2026, 5, 12, 20, 45)},
    {"label": "B4C to Carbon",                 "legend_label": "Sample Switch",
     "start": datetime(2026, 5, 12, 21, 0),    "end": datetime(2026, 5, 12, 21, 30)},
    {"label": "Resist HV Scan\nHelium Ethane\n3mm Drift\nCarbon Capsule", "legend_label": "Measurement",
     "start": datetime(2026, 5, 12, 21, 45),   "end": datetime(2026, 5, 13, 8, 0)},
    {"label": "Gas Switch\nHe/Ethane to Ar/CO2",     "legend_label": "Gas Switch",
     "start": datetime(2026, 5, 13, 8, 30),    "end": datetime(2026, 5, 13, 9, 30)},
    {"label": "Stephan Leaves",                "legend_label": "Event",
     "start": datetime(2026, 5, 13, 10, 0),    "end": datetime(2026, 5, 13, 11)},
    {"label": "Flush\nArgon CO2",              "legend_label": "Flush",
     "start": datetime(2026, 5, 13, 11, 30),    "end": datetime(2026, 5, 13, 16, 0)},
    {"label": "Resist HV Scan\nArgon CO2\n3mm Drift\nB4C", "legend_label": "Measurement",
     "start": datetime(2026, 5, 13, 16, 30),   "end": datetime(2026, 5, 13, 20, 45)},
    {"label": "B4C to Carbon",                 "legend_label": "Sample Switch",
     "start": datetime(2026, 5, 13, 21, 0),    "end": datetime(2026, 5, 13, 21, 30)},
    {"label": "Resist HV Scan\nArgon CO2\n3mm Drift\nCarbon Capsule", "legend_label": "Measurement",
     "start": datetime(2026, 5, 13, 21, 45),   "end": datetime(2026, 5, 14, 10, 0)},
    {"label": "Gas Switch\nAr/CO2 to Ar/CF4",     "legend_label": "Gas Switch",
     "start": datetime(2026, 5, 14, 10, 30),    "end": datetime(2026, 5, 14, 11, 30)},
    {"label": "Flush\nArgon CF4",              "legend_label": "Flush",
     "start": datetime(2026, 5, 14, 12, 0),    "end": datetime(2026, 5, 14, 16, 0)},
    {"label": "Resist HV Scan\nArgon CF4\n3mm Drift\nB4C", "legend_label": "Measurement",
     "start": datetime(2026, 5, 14, 16, 30),   "end": datetime(2026, 5, 14, 20, 45)},
    {"label": "B4C to Carbon",                 "legend_label": "Sample Switch",
     "start": datetime(2026, 5, 14, 21, 0),    "end": datetime(2026, 5, 14, 21, 30)},
    {"label": "Resist HV Scan\nArgon CF4\n3mm Drift\nCarbon Capsule", "legend_label": "Measurement",
     "start": datetime(2026, 5, 14, 21, 45),   "end": datetime(2026, 5, 15, 7, 30)},
    {"label": "Drift Switch\n3mm to 30mm",     "legend_label": "Drift Switch",
     "start": datetime(2026, 5, 15, 8, 0),    "end": datetime(2026, 5, 15, 9)},
    {"label": "Flush\nPure Argon",         "legend_label": "Flush",
     "start": datetime(2026, 5, 15, 9, 15),    "end": datetime(2026, 5, 15, 10, 30)},
    {"label": "Setup\nFlammable Gas Line\nEAR2", "legend_label": "Setup",
     "start": datetime(2026, 5, 15, 10, 45),   "end": datetime(2026, 5, 15, 14, 30)},
    {"label": "Flush\nNeon Isobutane",         "legend_label": "Flush",
     "start": datetime(2026, 5, 15, 15, 0),    "end": datetime(2026, 5, 15, 17, 0)},
    {"label": "Resist HV Scan\nNeon Isobutane\n30mm Drift\nEmpty Sample", "legend_label": "Measurement",
     "start": datetime(2026, 5, 15, 17, 30),   "end": datetime(2026, 5, 15, 20, 45)},
    {"label": "Empty to Carbon",                 "legend_label": "Sample Switch",
     "start": datetime(2026, 5, 15, 21, 0),    "end": datetime(2026, 5, 15, 21, 30)},
    {"label": "Resist HV Scan\nNeon Isobutane\n30mm Drift\nCarbon Capsule", "legend_label": "Measurement",
     "start": datetime(2026, 5, 15, 21, 45),   "end": datetime(2026, 5, 16, 10, 0)},
    {"label": "Mesh Switch Testing\nNeon Isobutane\n30mm Drift\nCarbon Capsule", "legend_label": "Testing",
     "start": datetime(2026, 5, 16, 10, 30),   "end": datetime(2026, 5, 16, 20, 45)},
    {"label": "Carbon to B4C",                 "legend_label": "Sample Switch",
     "start": datetime(2026, 5, 16, 21, 0),    "end": datetime(2026, 5, 16, 21, 30)},
    {"label": "Resist HV Scan\nNeon Isobutane\n30mm Drift\nB4C", "legend_label": "Measurement",
     "start": datetime(2026, 5, 16, 21, 45),   "end": datetime(2026, 5, 17, 9, 0)},
    {"label": "Flush\nPure Argon",         "legend_label": "Flush",
     "start": datetime(2026, 5, 17, 9, 15),    "end": datetime(2026, 5, 17, 10, 30)},
    {"label": "Drift Switch (?)\n30mm to 15mm",     "legend_label": "Drift Switch", "optional": True,
     "start": datetime(2026, 5, 17, 10, 45),    "end": datetime(2026, 5, 17, 13, 30)},
    {"label": "Flush\nPure Argon",         "legend_label": "Flush",
     "start": datetime(2026, 5, 17, 13, 45),    "end": datetime(2026, 5, 17, 15, 30)},
    {"label": "Flush (?)\nNeon Isobutane",         "legend_label": "Flush", "optional": True,
     "start": datetime(2026, 5, 17, 15,  45),    "end": datetime(2026, 5, 17, 18, 0)},
    {"label": "Resist HV Scan (?)\nNeon Isobutane\n15mm Drift\nCarbon Capsule", "legend_label": "Measurement", "optional": True,
     "start": datetime(2026, 5, 17, 18, 30),   "end": datetime(2026, 5, 18, 6, 0)},
    {"label": "Flush\nPure Argon",         "legend_label": "Flush",
     "start": datetime(2026, 5, 18, 6, 30),    "end": datetime(2026, 5, 18, 8, 15)},
    {"label": "Setup\nDismount\nEAR2", "legend_label": "Setup",
     "start": datetime(2026, 5, 18, 8, 30),   "end": datetime(2026, 5, 18, 13, 30)},
]

# Apply styles from STYLES dict
for _b in BANDS:
    _b.update(STYLES.get(_b["legend_label"], {}))


if __name__ == "__main__":
    draw_week_timeline(
        title=TITLE,
        subtitle=SUBTITLE,
        start_date=START_DATE,
        end_date=END_DATE,
        bands=BANDS,
        out_png=OUTPUT_PNG,
        out_pdf=OUTPUT_PDF,
        day_width=DAY_WIDTH,
        fig_height=FIG_HEIGHT,
    )
