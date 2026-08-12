#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_all.py -- remeasure everything, rebuild both write-ups.

    .venv/bin/python -m ntof_active_area.run_all
"""
from . import (figures_mm, figures_note, figures_scint, make_note, make_report,
               mm_edges, scint_acceptance)


def main():
    print('--- chamber edges ---')
    mm_edges._print(mm_edges.measure())
    print('\n--- scintillator acceptance ---')
    scint_acceptance._print(scint_acceptance.measure())
    print()
    figures_mm.main()
    figures_scint.main()
    figures_note.main()
    make_report.main()      # report.html, for the DAQ Analysis tab
    make_note.main()        # note_active_area.html, self-contained for the site


if __name__ == '__main__':
    main()
