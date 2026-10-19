#!/usr/bin/env python3
"""
12_shear_lever.py — measure the empirical p0-scan shear lever (T2.4 v2).

The stage-2 (p0, w) scan in `wft.reco._global_start` is centred on the
amplitude-weighted strip centroid p_c of the window. For an inclined track the
fit's p0 (position at the reference plane) sits at

    p0_ref  =  p_c  -  w * u_eff

for some effective lever u_eff [ns]. Shear v1 assumed u_eff = half the drift
column (15000/v ~ 410 ns) and was REJECTED on the bench — slightly worse
everywhere (`KERNEL_ARMS_2026-08-12.md` §4). The suspected reason: the
amplitude-weighted centroid already sits near the mesh (bright strips = early,
compressed charge), so the true lever is much shorter. This script measures it:

    u_eff = median over inclined reference tracks of (p_c - p0_ref) / w_ref

per plane, in |tan_ref| bins (a genuine geometric lever must be
angle-independent; angle dependence = the centroid systematically shifting
with inclination, which a constant shear cannot fix).

p_c is computed from the cached candidate windows exactly as _global_start
does (amp = per-strip max of W clipped at 0). The candidate is the brightest
one in the window list — the same one the fit ranks first in ~all events.

    ../.venv/bin/python mx_june_wft/12_shear_lever.py sat_det3
Output: <OUT_BASE>/wft/kernel_arms/shear_lever.json (+ .png)
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

TAN_MIN = 0.08          # below this, w is too small to divide by
TAN_MAX = 0.60
BINS = [(0.08, 0.14), (0.14, 0.20), (0.20, 0.28), (0.28, 0.45)]


def centroid(P):
    """Amplitude-weighted strip centroid, exactly as _global_start."""
    W = np.asarray(P['W'], dtype=float)
    pos = np.asarray(P['pos'], dtype=float)
    amp = np.maximum(W.max(axis=1), 0.0)
    if amp.sum() <= 0:
        return np.nan
    return float((pos * amp).sum() / amp.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key', nargs='?', default='sat_det3')
    ap.add_argument('--cache', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from wft.calib import CalibrationBundle

    cfg = get_config(args.run_key)
    cache = args.cache or os.path.join(cfg.OUT_BASE, 'wft',
                                       'bench_cache_ftst.pkl')
    out_dir = args.out or cfg.out_dir('wft', 'kernel_arms')
    os.makedirs(out_dir, exist_ok=True)

    with open(cache, 'rb') as f:
        data = pickle.load(f)
    events, meta = data['events'], data['meta']
    v = CalibrationBundle.load(meta['bundle']).v_drift
    print(f'{len(events):,} events, v = {v} um/ns '
          f'(v1 assumed lever u_mid = {15000.0 / v:.0f} ns)')

    u = {'x': [], 'y': []}          # (abs tan_ref, u_eff ns)
    for eid, ev in events.items():
        t = ev['truth']
        if ev.get('spark'):
            continue
        for plane, ref_key in (('x', 'ref_x'), ('y', 'ref_y')):
            tan = t.get(f'tan_{plane}', np.nan)
            ref = t.get(ref_key, np.nan)
            cand = ev['wins'].get(plane)
            if not cand or not np.isfinite(tan) or not np.isfinite(ref):
                continue
            if not (TAN_MIN <= abs(tan) <= TAN_MAX):
                continue
            best = max(cand, key=lambda P: float(
                np.maximum(np.asarray(P['W']).max(axis=1), 0.0).sum()))
            p_c = centroid(best)
            if not np.isfinite(p_c) or abs(p_c - ref) > 8.0:
                continue            # wrong candidate (noise window)
            w_ref = tan * v / 1e3   # mm/ns, same convention as the fit's w
            u[plane].append((abs(tan), (p_c - ref) / w_ref))

    res = dict(v_drift=v, u_mid_v1=15000.0 / v, bins={}, overall={})
    for plane in ('x', 'y'):
        arr = np.array(u[plane])
        res['overall'][plane] = dict(
            n=int(len(arr)),
            median=float(np.median(arr[:, 1])),
            sigma=float(1.4826 * np.median(np.abs(
                arr[:, 1] - np.median(arr[:, 1])))))
        res['bins'][plane] = []
        for lo, hi in BINS:
            m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
            res['bins'][plane].append(dict(
                lo=lo, hi=hi, n=int(m.sum()),
                median=float(np.median(arr[m, 1])) if m.sum() else np.nan))
        o = res['overall'][plane]
        line = '  '.join(f"[{b['lo']:.2f},{b['hi']:.2f}) "
                         f"{b['median']:+7.1f} (n={b['n']})"
                         for b in res['bins'][plane])
        print(f"{plane}: u_eff = {o['median']:+.1f} ns "
              f"(robust sigma {o['sigma']:.0f}, n={o['n']})   {line}")

    with open(os.path.join(out_dir, 'shear_lever.json'), 'w') as f:
        json.dump(res, f, indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, plane in zip(axes, ('x', 'y')):
        arr = np.array(u[plane])
        ax.scatter(arr[:, 0], arr[:, 1], s=4, alpha=0.25, lw=0)
        ax.axhline(15000.0 / v, color='crimson', ls='--',
                   label=f'v1 assumed u_mid = {15000.0 / v:.0f} ns')
        med = res['overall'][plane]['median']
        ax.axhline(med, color='k', label=f'measured median = {med:+.0f} ns')
        bx = [(b['lo'] + b['hi']) / 2 for b in res['bins'][plane]]
        by = [b['median'] for b in res['bins'][plane]]
        ax.plot(bx, by, 'o-', color='darkorange', label='bin medians')
        ax.set_xlabel('|tan θ_ref|')
        ax.set_title(f'{plane.upper()} plane')
        ax.set_ylim(-800, 800)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('u_eff = (p_c − p0_ref) / w_ref   [ns]')
    fig.suptitle('T2.4 v2 — empirical scan-shear lever (amp-weighted centroid '
                 'vs reference)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'shear_lever.png'), dpi=130)
    print(f'wrote {out_dir}/shear_lever.json + .png')


if __name__ == '__main__':
    main()
