#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16_gap_from_pulse_width.py

Measure the FULL-GAP drift time directly from single-strip pulse widths,
independently of the strip-timestamp spans used elsewhere.

Idea: a near-vertical muon (θ_ref ≈ 0) deposits ionisation along the full
drift gap above (nearly) one strip, so that strip's waveform is ON for the
whole drift-time window: pulse width ≈ T_gap + shaping. An inclined track
(|θ_ref| > 15°) spreads the charge across many strips, each sampling a thin
z-slice: pulse width ≈ shaping only. The difference of the two distributions'
peaks/endpoints estimates T_gap without any strip-timestamp systematics.

This discriminates:
    gap = 19.4 mm physical  → T_gap(1000 V) ≈  690 ns
    gap = 30 mm, signal from far half lost → single-strip width STILL shows
        only the surviving column, but a 30 mm gap with v=28 µm/ns would give
        T_gap ≈ 1070 ns if the full column contributes.

Usage: ../.venv/bin/python 16_gap_from_pulse_width.py [sat_det3] [--veto=50]
Output: <out>/alignment_tpc_veto50/bias_study/pulse_width_gap.png
"""
import os
import sys
import pickle

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

from qa_config import config_from_argv, setup_paths
setup_paths()
import uproot
import cosmic_micro_tpc_analysis as cm
from M3RefTracking import M3RefTracking, get_xy_angles

CFG = config_from_argv()
VETO = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--veto=')), 50)
SAMPLE_NS = 60.0
VERT_DEG = 3.0
INCL_DEG = 15.0

tag = f'_veto{VETO}'
out_dir = CFG.out_dir(f'alignment_tpc{tag}', 'bias_study')
cache = os.path.join(CFG.out_dir('cache'), f'event_results{tag}.pkl')
align_json = os.path.join(CFG.OUT_BASE, f'alignment_tpc{tag}', 'alignment.json')

results = pickle.load(open(cache, 'rb'))
best = cm.load_alignment(align_json)
rays = M3RefTracking(CFG.m3_tracking_dir, chi2_cut=20.0)
xang, yang, anum = get_xy_angles(rays.ray_data)
cm.attach_reference_positions(results, rays, best, np.array(xang), anum)

# classify events by BOTH-plane reference angle (needs a matched ray, r<10mm)
theta = {}
for r in results:
    if not (r.has_x and r.has_y) or not np.isfinite(r.radial_residual_mm) \
            or r.radial_residual_mm > 10.0 or np.isnan(r.ref_tan_theta_x):
        continue
    ax = abs(np.degrees(np.arctan(r.ref_tan_theta_x)))
    ay = abs(np.degrees(np.arctan(r.ref_tan_theta_y)))
    theta[r.event_id] = (ax, ay)

vert = {e for e, (ax, ay) in theta.items() if ax < VERT_DEG and ay < VERT_DEG}
incl = {e for e, (ax, ay) in theta.items() if max(ax, ay) > INCL_DEG}
print(f'{len(theta):,} matched events: {len(vert):,} vertical (<{VERT_DEG}°), '
      f'{len(incl):,} inclined (>{INCL_DEG}°)')

# raw hits: per event & plane, take the max-amplitude strip's pulse width
fs = sorted(f for f in os.listdir(CFG.combined_hits_dir)
            if f.endswith('.root') and '_datrun_' in f)
df = uproot.concatenate(
    [f'{CFG.combined_hits_dir}{f}:hits' for f in fs],
    expressions=['eventId', 'feu', 'amplitude', 'left_sample', 'right_sample',
                 'time_over_threshold', 'sample'],
    library='pd')
df = df[df['feu'].isin(CFG.MX17_FEUS)].copy()
df['width_ns'] = (df['right_sample'] - df['left_sample']) * SAMPLE_NS
df['railed'] = df['right_sample'] >= 31

sel = df[df['eventId'].isin(vert | incl)]
idx = sel.groupby(['eventId', 'feu'])['amplitude'].idxmax()
lead = sel.loc[idx]
lead_v = lead[lead['eventId'].isin(vert)]
lead_i = lead[lead['eventId'].isin(incl)]
print(f'lead strips: {len(lead_v):,} vertical, {len(lead_i):,} inclined; '
      f"railed right edge: vert {100*lead_v['railed'].mean():.1f}%  "
      f"incl {100*lead_i['railed'].mean():.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bins = np.arange(0, 2000, 60)
for ax, col, title in [(axes[0], 'width_ns', 'max-amplitude strip: (right−left) sample width'),
                       (axes[1], 'time_over_threshold', 'max-amplitude strip: time over threshold')]:
    for d, c, lab in [(lead_v, 'crimson', f'vertical (|θ|<{VERT_DEG:.0f}°)'),
                      (lead_i, 'steelblue', f'inclined (|θ|>{INCL_DEG:.0f}°)')]:
        vals = d[col].to_numpy()
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=bins, histtype='step', lw=2, color=c, density=True,
                label=f'{lab}  med={np.median(vals):.0f} ns, p90={np.percentile(vals,90):.0f}')
    for gap_mm, ls in [(19.4, '--'), (30.0, ':')]:
        ax.axvline(gap_mm * 1000.0 / 28.1, color='k', ls=ls, lw=1.2,
                   label=f'T_gap if gap={gap_mm:g} mm (v=28.1)')
    ax.set_xlabel('width [ns]'); ax.set_ylabel('norm.')
    ax.set_title(title); ax.legend(fontsize=8)
fig.suptitle(f'{CFG.RUN} / {CFG.SUB_RUN} — single-strip pulse width: full-gap drift time', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(out_dir, 'pulse_width_gap.png'), dpi=150)
print(f'Written {out_dir}/pulse_width_gap.png')

for name, d in [('vertical', lead_v), ('inclined', lead_i)]:
    w = d['width_ns'].to_numpy()
    print(f'{name}: width med={np.median(w):.0f}  p75={np.percentile(w,75):.0f}  '
          f'p90={np.percentile(w,90):.0f}  p95={np.percentile(w,95):.0f} ns')
