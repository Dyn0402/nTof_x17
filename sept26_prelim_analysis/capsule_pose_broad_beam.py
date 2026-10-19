#!/usr/bin/env python3
"""capsule_pose_broad_beam.py -- the capsule_pose toy with a measured-width EAR2 beam
(Gaussian, 30 mm FWHM: PPACMon at 19.95 m, EPJA 2025 / arXiv 2505.00042) in place of
Lambda2D, which is the n_TOF RESOLUTION FUNCTION (lambda = moderation length vs E),
not a radial beam profile.  Exploratory, 2026-10-14; see HANDOFF_CAPSULE_POSE_FIT.md.

    python -m sept26_prelim_analysis.capsule_pose_broad_beam [out.csv]
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from sept26_prelim_analysis import capsule_pose as CP
SIG = 30.0 / 2.355
def sample_beam(n, rng, window, centre=(0.0, 0.0), width_scale=1.0):
    E = CP.sample_energy(n, rng, window)
    x = centre[0] + rng.normal(0, SIG * width_scale, n)
    z = centre[1] + rng.normal(0, SIG * width_scale, n)
    return x, z, np.hypot(x - centre[0], z - centre[1]), E
CP.sample_beam = sample_beam
rows = []
cases = [(xc, 0, 0) for xc in (0, -5, -10.3, -15)] + \
        [(-10.3, th, ph) for th in (2, 5, 10) for ph in (0, 180, 90)]
for xc, th, ph in cases:
    pose = dict(CP.NOMINAL, x_c=xc, theta=th, phi=ph)
    C, fr = CP.beam_captures(pose, n=150_000, window=CP.WINDOWS['late'])
    r = dict(x_c=xc, theta=th, phi=ph, **{k: fr[k] for k in ('gas', 'al_per_ray', 'al_nose', 'al_barrel')})
    for arm in 'AC':
        e = CP.estimators(CP.toy_tracks(arm, C, n=50_000))
        r[f'x_{arm}'] = e['x_cross']; r[f'y_{arm}'] = e['y_cross']; r[f'w68_{arm}'] = e['h_w68']
        sh = np.asarray(e['shear']); r[f'shear_{arm}'] = ' '.join(f'{v:+.1f}' for v in sh[:, 1])
    rows.append(r); print(pd.DataFrame([r]).round(3).to_string(header=len(rows) == 1, index=False), flush=True)
pd.DataFrame(rows).to_csv(sys.argv[1] if len(sys.argv) > 1 else CP.paths.out('capsule_pose') / 'broad_beam.csv', index=False)
