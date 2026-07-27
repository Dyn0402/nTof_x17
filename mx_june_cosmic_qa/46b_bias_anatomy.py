#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
46b_bias_anatomy.py — discriminate "v really is ~42" from "raw-hit scan bias".

Two measurements from the same matched-hit table as 46:

(1) per-cluster floated v* binned in |tan_ref|.  A genuine drift velocity is
    angle-independent; an additive spatial floor w (resistive charge spreading)
    biases the ladder like  v*(tan) ~ v_true + w/(|tan| * T_span), i.e. falls
    toward v_true at steep angles.

(2) cluster spatial extent (ptp of strip positions) vs |tan_ref|.  Pure
    geometry — NO drift times: slope = visible drift column z_vis [mm],
    intercept = spatial floor w.  Hypotheses: z_vis = 29 mm (gap-filling,
    v~42) vs z_vis ~ 23 mm (v=34 with invisible deep column).

Usage: ../.venv/bin/python 46b_bias_anatomy.py
"""
import os
import sys
import importlib

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'engineer_package'))
m46 = importlib.import_module('46_vdrift_ref_metric_scan')
from make_event_displays import load_hits                     # noqa: E402
from make_event_displays_3d import load_full_reference        # noqa: E402

OUT = m46.OUT
T_SPAN_NS = 689.0
V_GEOM, V_GAP = 34.0, 42.1

TAN_BINS = np.array([0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.36, 0.44])


def binned_v(pv):
    at = np.abs(pv['tan'].to_numpy())
    v = pv['v'].to_numpy()
    xc, med, lo, hi, n = [], [], [], [], []
    for a, b in zip(TAN_BINS[:-1], TAN_BINS[1:]):
        s = (at >= a) & (at < b)
        if s.sum() < 25:
            continue
        xc.append(np.median(at[s])); med.append(np.median(v[s]))
        lo.append(np.percentile(v[s], 16)); hi.append(np.percentile(v[s], 84))
        n.append(s.sum())
    return map(np.array, (xc, med, lo, hi, n))


def cluster_extent(df):
    rec = []
    for (eid, pl), g in df.groupby(['eid', 'plane']):
        rec.append(dict(tan=abs(g['tan'].iloc[0]),
                        ext=np.ptp(g['pos'].to_numpy()),
                        ext_core=np.ptp(g.loc[g['core'], 'pos'].to_numpy())
                        if g['core'].sum() >= 2 else np.nan))
    return pd.DataFrame(rec)


def main():
    results, best, ref, by_eid = load_full_reference()
    hits, det = load_hits()
    df = m46.build_hit_table(hits, ref, by_eid, best, res_cut=6.0)

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # ---- (1) per-cluster v* vs |tan| --------------------------------------
    styles = {('all strips', 'xt'): ('#333333', 'o-'),
              ('core strips', 'xt'): ('#c0392b', 's-')}
    for (tag, direction), (c, fmt) in styles.items():
        sub = df if tag == 'all strips' else df[df.core]
        pv = m46.per_cluster_v(sub, direction)
        xc, med, lo, hi, n = binned_v(pv)
        ax[0].errorbar(xc, med, yerr=[med - lo, hi - med], fmt=fmt, color=c,
                       lw=1.8, ms=6, capsize=3, label=f'raw {tag} (median ±68%)')
    tt = np.linspace(0.08, 0.45, 100)
    for w, ls in ((2.0, ':'), (3.5, '--')):
        ax[0].plot(tt, V_GEOM + w * 1000.0 / (tt * T_SPAN_NS), ls, color='#1a9850',
                   lw=1.6, label=f'bias model: 34 + {w:.1f}mm floor/(tanθ·T)')
    ax[0].axhline(V_GEOM, color='#1a9850', lw=1.8)
    ax[0].axhline(V_GAP, color='#888', lw=1.6, ls='-.')
    ax[0].text(0.42, V_GEOM + 0.5, 'v=34', color='#1a9850', fontsize=10)
    ax[0].text(0.42, V_GAP + 0.5, 'v=42', color='#888', fontsize=10)
    ax[0].set_xlabel('|tan θ_ref| (hit axis)')
    ax[0].set_ylabel('per-cluster v* = (dx/dt)/tan  [µm/ns]')
    ax[0].set_ylim(20, 90)
    ax[0].set_title('(1) Is the preferred velocity a constant?\n'
                    'true velocity = flat; spatial-floor artifact falls as 1/tanθ')
    ax[0].legend(fontsize=8.5)

    # ---- (2) time-free extent vs |tan| ------------------------------------
    ce = cluster_extent(df)
    xc, med, lo, hi = [], [], [], []
    for a, b in zip(TAN_BINS[:-1], TAN_BINS[1:]):
        s = (ce.tan >= a) & (ce.tan < b)
        if s.sum() < 25:
            continue
        xc.append(ce.tan[s].median()); med.append(ce.ext[s].median())
        lo.append(np.percentile(ce.ext[s], 16))
        hi.append(np.percentile(ce.ext[s], 84))
    xc, med, lo, hi = map(np.array, (xc, med, lo, hi))
    ax[1].errorbar(xc, med, yerr=[med - lo, hi - med], fmt='o-', color='#333',
                   lw=1.8, ms=6, capsize=3, label='cluster spatial extent (median ±68%)')
    m, b = np.polyfit(xc, med, 1)
    ax[1].plot(tt, m * tt + b, '-', color='#c0392b', lw=2,
               label=f'fit: z_vis = {m:.1f} mm (+{b:.1f} mm floor)')
    for zv, c, lab in ((29.0, '#888', 'gap-filling hypothesis: 29 mm'),
                       (23.4, '#1a9850', 'v=34 visible column: 23.4 mm')):
        ax[1].plot(tt, zv * tt + b, '--', color=c, lw=1.6, label=lab)
    ax[1].set_xlabel('|tan θ_ref| (hit axis)')
    ax[1].set_ylabel('cluster extent [mm]')
    ax[1].set_title('(2) Time-free column measurement\n'
                    'extent = z_vis·|tanθ| + floor   (uses NO drift times)')
    ax[1].legend(fontsize=9)

    fig.suptitle('Discriminating a real v≈42 from a raw-hit scan bias',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUT, 'bias_anatomy.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'-> {p}')
    print(f'extent slope (visible column): {m:.2f} mm, floor {b:.2f} mm')
    print(f'  gap-filling (29 mm) predicts slope 29; v=34 predicts ~23.4')
    print(f'  implied v from extent slope: {m * 1000.0 / T_SPAN_NS:.1f} um/ns')


if __name__ == '__main__':
    main()
