#!/usr/bin/env python3
"""
May_Flammable_Test_Week.py
──────────────────────────
Hour-by-hour view of the flammable-gas (Neon Isobutane) portion of the
EAR2 beam test: Friday May 15 – Monday May 18.

Run:
    python May_Flammable_Test_Week.py
"""

from datetime import datetime
from May_Test_Week import BANDS, STYLES
from week_timeline_generator import draw_week_timeline

TITLE    = "n_TOF X17 · EAR2 Beam Test — Neon Isobutane Weekend"
SUBTITLE = "CERN Campaign — May 2026 · Flammable Gas Setup & Measurements"

START_DATE = datetime(2026, 5, 15)
END_DATE   = datetime(2026, 5, 18)

OUTPUT_PNG = "May_Flammable_Test_Week.png"
OUTPUT_PDF = "May_Flammable_Test_Week.pdf"   # set to None to skip

DAY_WIDTH  = 2.5    # inches per day column
FIG_HEIGHT = 9.0


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