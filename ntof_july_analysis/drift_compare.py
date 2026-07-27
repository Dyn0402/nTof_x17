#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_july_analysis/drift_compare.py

Overlay multiple drift-HV resist scans from the same run:
 - solo (faceted-by-detector) plots, one line per drift, reusing
   compare_scans.plot_grid / plot_midwindow (hits/event grid, mean-amplitude
   grid, mid-window turn-off blowup).
 - per-drift "all detectors in one panel" mid-window turn-off plot, so each
   drift's cross-detector shape can be read at a glance.

Edit RUN / DRIFTS and run:  .venv/bin/python ntof_july_analysis/drift_compare.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from compare_scans import build_series, plot_grid, plot_midwindow  # noqa: E402
from july_hv_scan import ANALYSIS_DIR, TIME_WINDOWS, AMP_THRESHOLD, _save  # noqa: E402

# ---------------------------------------------------------------------------
RUN = 'run_15'
DRIFTS = [800, 600]          # order = plot order; drift values in V
MID_WINDOW = (1150, 3500)    # the post-flash mid-window turn-off row (ns)
OUT_LABEL = RUN              # output subdir under July_HV_Scan/


def _auto_scale_factor(peak: float, ref: float) -> float:
    """Nearest power-of-10 that brings `peak` within ~3x of `ref`; 1 if already close."""
    if not (peak > 0 and ref > 0):
        return 1.0
    return 10.0 ** round(np.log10(ref / peak))


def main():
    specs = [{'label': f'{RUN} dr{d}', 'run': RUN, 'match': rf'^dr{d}_'} for d in DRIFTS]
    out_dir = os.path.join(ANALYSIS_DIR, 'July_HV_Scan', OUT_LABEL)
    print(f'Output -> {out_dir}\n')
    series, det_names, gas = build_series(specs)
    if not any(entry for _, entry in series):
        print('No processed data matched any series — nothing plotted.')
        return

    # solo (per-detector) plots, one line per drift
    plot_grid(series, det_names, gas, out_dir, 'hits_per_event')
    plot_grid(series, det_names, gas, out_dir, 'mean_amplitude')
    plot_midwindow(series, det_names, gas, out_dir)

    # per-drift: all detectors overlaid in one panel
    wi = TIME_WINDOWS.index(MID_WINDOW)
    cmap = plt.get_cmap('tab10')
    det_color = {det: cmap(i) for i, det in enumerate(det_names)}
    lo, hi = MID_WINDOW
    for (slabel, entry), d in zip(series, DRIFTS):
        y_by_det = {}
        for det in det_names:
            s = entry.get(det)
            if s is None or s['hv'].size == 0:
                continue
            y = s['hits_per_event'][:, wi]
            good = np.isfinite(s['hv']) & np.isfinite(y)
            if not good.any():
                continue
            y_by_det[det] = (s['hv'][good], y[good])
        if not y_by_det:
            continue

        # auto-detect detectors whose peak is far off the pack and rescale them
        # by the nearest power of 10 so all curves read on a shared y-axis.
        peaks = {det: y.max() for det, (_, y) in y_by_det.items()}
        ref = float(np.median(list(peaks.values())))
        factors = {det: _auto_scale_factor(peaks[det], ref) for det in y_by_det}

        fig, ax = plt.subplots(figsize=(6.5, 5))
        for det, (hv, y) in y_by_det.items():
            f = factors[det]
            yy = y * f
            color = det_color[det]
            lab = det if f == 1 else f'{det}  (×{f:g})' if f > 1 else f'{det}  (÷{1/f:g})'
            ax.plot(hv, yy, '-o', color=color, ms=4, lw=1.6, label=lab)
            if f != 1:
                i_pk = int(np.argmax(yy))
                txt = f'×{f:g}' if f > 1 else f'÷{1/f:g}'
                ax.annotate(txt, xy=(hv[i_pk], yy[i_pk]), xytext=(0, 16),
                            textcoords='offset points', ha='center', fontsize=9,
                            fontweight='bold', color=color,
                            arrowprops=dict(arrowstyle='-|>', color=color, lw=1.2))
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Resist HV [V]')
        ax.set_ylabel(f'Hits / event  ({lo/1000:g}-{hi/1000:g} us)'
                       + ('  (some series rescaled, see legend)'
                          if any(f != 1 for f in factors.values()) else ''))
        gas_s = f'  —  {gas}' if gas else ''
        ax.set_title(f'{slabel}  (drift {d} V){gas_s}\nall detectors, post-flash '
                     f'mid-window,  amp >= {AMP_THRESHOLD} ADC', fontsize=10)
        ax.legend(fontsize=9)
        fig.tight_layout()
        _save(fig, out_dir, f'midwindow_alldets_dr{d}.png')

    print(f'\nFigures written to: {out_dir}')


if __name__ == '__main__':
    main()
