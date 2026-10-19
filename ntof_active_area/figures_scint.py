#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figures_scint.py -- figures for the arm-A scintillator acceptance.

    .venv/bin/python -m ntof_active_area.figures_scint
"""
from __future__ import annotations

import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.special import erf

from .mm_edges import FIG, OUT
from .scint_acceptance import smeared_step


def _panel(ax, binned, curve_x, curve_y, title, xlabel, nominal=None):
    c, p, e, n = [np.asarray(a) for a in binned]
    ax.errorbar(c, p, yerr=e, fmt='o', ms=4, color='#1f77b4', lw=1)
    ax.plot(curve_x, curve_y, color='#d62728', lw=1.6)
    if nominal is not None:
        for v in nominal:
            ax.axvline(v, color='0.4', ls='--', lw=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 1)


def main():
    FIG.mkdir(exist_ok=True)
    r = json.loads((OUT / 'results_scint.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    b = r['plastic_lr_boundary']
    xs = np.linspace(-350, 350, 400)
    ys = b['pedestal'] + b['amplitude'] * 0.5 * (
        1 + erf((xs - b['u0_mm']) / (np.sqrt(2) * b['sigma_mm'])))
    _panel(axes[0, 0], b['binned'], xs, ys,
           f"plastic L/R split: boundary {b['u0_mm']:+.1f} ± {b['u0_err_mm']:.1f} mm "
           f"(geometry 0), blur σ = {b['sigma_mm']:.0f} mm",
           'chamber u extrapolated to the plastic plane [mm]', nominal=[0.0])
    axes[0, 0].set_ylabel('fraction tagged by the L bar')

    for ax, key, label, nom in (
            (axes[0, 1], 'wall_v', 'SiPM wall, along the beam', 250.0),
            (axes[1, 0], 'plastic_v', 'plastics, along the beam', 150.0),
            (axes[1, 1], 'plastic_u', 'plastic pair, tangential', 200.0)):
        f = r[key]
        xs = np.linspace(-600, 600, 500)
        ys = f['pedestal'] + f['amplitude'] * smeared_step(
            xs, 0.0, f['sigma_mm'], -f['half_mm'], f['half_mm'])
        verdict = 'constrained' if f['constrained'] else 'NOT constrained'
        _panel(ax, f['binned'], xs, ys,
               f"{label}: fit half {f['half_mm']:.0f} ± {f['half_err_mm']:.0f} mm "
               f"vs survey {nom:.0f} — {verdict}",
               'extrapolated position [mm]', nominal=[-nom, nom])
        ax.set_ylabel('fraction with an n_TOF tag')

    fig.suptitle('Arm-A scintillator acceptance seen from chamber A '
                 '(run_79, n_TOF 224572).  Grey dashed = surveyed edge.', y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / 'scint_acceptance.png', dpi=110, bbox_inches='tight')
    plt.close(fig)

    # wall segment ordering
    w = r['wall_segments']
    fig, ax = plt.subplots(figsize=(6.5, 5))
    pred = [s['predicted_centre_u_mm'] for s in w['segments']]
    obs = [s['u_mean'] for s in w['segments']]
    ax.plot(pred, obs, 'o-', color='#1f77b4')
    for s in w['segments']:
        ax.annotate(f"seg {s['seg']}", (s['predicted_centre_u_mm'], s['u_mean']),
                    textcoords='offset points', xytext=(6, 6), fontsize=9)
    lim = [min(pred) - 40, max(pred) + 40]
    ax.plot(lim, lim, '--', color='0.5', label='1:1 (no blur)')
    ax.set_xlabel('geometric segment centre, chamber u [mm]')
    ax.set_ylabel('mean u of the tracks it tagged [mm]')
    ax.set_title(f"wall segment ordering: r = {w['ordering_corr']:+.3f}, "
                 f"slope {w['slope_ratio']:.2f}\n(slope < 1 is accidental-tag "
                 f"dilution, not a wrong pitch)", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / 'wall_segments.png', dpi=110)
    plt.close(fig)
    print('figures ->', FIG)


if __name__ == '__main__':
    main()
