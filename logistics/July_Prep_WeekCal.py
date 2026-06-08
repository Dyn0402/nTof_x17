#!/usr/bin/env python3
"""
July_Prep_WeekCal.py
────────────────────
Single contiguous week-grid view of the July 2026 n_TOF schedule —
unlike July_Prep_Cal.py, this does not split into separate month blocks,
so the final week (Jun 29 – Jul 5) shows both June and July days in one row.
Reuses bands and milestones from July_Prep.py.

Run:
    python July_Prep_WeekCal.py
"""

from July_Prep import BANDS, MILESTONES, TITLE, SUBTITLE
from calendar_generator import draw_week_calendar

OUTPUT_PNG = "July_Prep_WeekCal.png"
OUTPUT_PDF = "July_Prep_WeekCal.pdf"

if __name__ == "__main__":
    draw_week_calendar(
        title=TITLE,
        subtitle=SUBTITLE,
        bands=BANDS,
        milestones=MILESTONES,
        out_png=OUTPUT_PNG,
        out_pdf=OUTPUT_PDF,
    )
