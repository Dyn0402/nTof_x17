#!/usr/bin/env python3
"""
July_Prep_Cal.py
────────────────
Calendar (month-grid) view of the July 2026 n_TOF schedule.
Reuses bands and milestones from July_Prep.py.

Run:
    python July_Prep_Cal.py
"""

from July_Prep import BANDS, MILESTONES, TITLE, SUBTITLE
from calendar_generator import draw_calendar

OUTPUT_PNG = "July_Prep_Cal.png"
OUTPUT_PDF = "July_Prep_Cal.pdf"

if __name__ == "__main__":
    draw_calendar(
        title=TITLE,
        subtitle=SUBTITLE,
        bands=BANDS,
        milestones=MILESTONES,
        out_png=OUTPUT_PNG,
        out_pdf=OUTPUT_PDF,
    )
