#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
det_labels.py

Experiment detector LETTER names, assigned 2026-07-03 when the detectors were
installed at the experiment. Keyed by the mx17_<N> number used everywhere else in
the analysis (DET_NAME.split('_')[-1]). The letter order A..E is the intended
report/page order.

    A = det 3      B = det 2      C = det 6      D = det 7      E = det 4

Used by build_final_pdf.py (overview) and build_hv_scan_pdf.py (HV scans) to title
and order the per-detector pages by letter while all the underlying analysis keys /
DET_NAMEs stay unchanged.
"""

DET_LETTER = {'3': 'A', '2': 'B', '6': 'C', '7': 'D', '4': 'E'}
_LETTER_ORDER = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}


def det_letter(num) -> str:
    """mx17 number -> experiment letter ('3' -> 'A'); unknown -> the number itself."""
    return DET_LETTER.get(str(num), str(num))


def order_key(num) -> int:
    """Sort key giving the A,B,C,D,E page order; unknown detectors sort last."""
    return _LETTER_ORDER.get(det_letter(num), 99)
