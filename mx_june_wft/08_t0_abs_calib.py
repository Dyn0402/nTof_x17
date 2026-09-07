#!/usr/bin/env python3
"""
08_t0_abs_calib.py — calibrate the absolute-t0 prediction per ftst class (T1.1).

For each plane of each bench-cache event, fit t0 with (p0, w) pinned to the M3
reference (the ref-pinned path, as measure_dt_xy) and take the per-ftst-class
median. The trigger is the muon, so within a class the true t0 is one number;
the ref-pinned per-event scatter is large (the reference's own transverse error
enters as dt0 = dp0/w, plus per-event model-time mismatch) but unbiased over
many events, so the median converges as n^-1/2 (n ~ 1000/class here).

Output: t0_abs.json {plane: {ftst: t0_pred_ns}} for run_bench --t0-abs and for
the calibration bundle's t0_abs field, plus per-class spreads and a figure.

    ../.venv/bin/python mx_june_wft/08_t0_abs_calib.py sat_det3
"""
import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]


def robust_sigma(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) else np.nan


_EVENTS = None
_CAL = None


def _init(bundle_path, cache_path):
    global _EVENTS, _CAL
    from wft.calib import CalibrationBundle
    from wft import model as wm
    _CAL = CalibrationBundle.load(bundle_path)
    wm.use_calibration(_CAL)
    with open(cache_path, 'rb') as f:
        _EVENTS = pickle.load(f)['events']


def _one(eid):
    from wft import model as wm
    ev = _EVENTS[eid]
    t = ev['truth']
    out = {'eid': eid}
    for plane in ('x', 'y'):
        ref_p, ref_tan = t[f'ref_{plane}'], t[f'tan_{plane}']
        ftst = (ev.get('ftst') or {}).get(plane)
        if ftst is None or not (np.isfinite(ref_p) and np.isfinite(ref_tan)):
            continue
        # the candidate window that contains the reference position
        P = None
        for cand in ev['wins'].get(plane) or []:
            pos = np.asarray(cand['pos'])
            if pos.min() - 2.0 <= ref_p <= pos.max() + 2.0:
                P = cand
                break
        if P is None:
            continue
        if np.asarray(P['W']).shape[1] != wm.NSAMP:
            wm.set_nsamp(np.asarray(P['W']).shape[1])
        try:
            g = wm.init_guess(P, plane, ref_tan, ref_p, _CAL.v_drift)
            r = wm.fit_plane_raw(P, plane, *g, hyper=_CAL.hyper,
                                 fix_p0w=(ref_p, ref_tan * _CAL.v_drift * 1e-3))
        except Exception:
            continue
        if r is None or not np.isfinite(r['chi2']):
            continue
        out[plane] = (int(ftst), float(r['t0']),
                      float(r['chi2'] / max(r['dof'], 1)), float(ref_tan))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key', nargs='?', default='sat_det3')
    ap.add_argument('--cache', default=None)
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    cfg = get_config(args.run_key)
    cache = args.cache or os.path.join(cfg.OUT_BASE, 'wft', 'bench_cache_ftst.pkl')
    with open(cache, 'rb') as f:
        meta = pickle.load(f)['meta']
    bundle = args.bundle or os.path.join(cfg.OUT_BASE, 'wft', 'calib_bundle_lp2')
    out_dir = args.out or cfg.out_dir('wft', 't0_prior')
    os.makedirs(out_dir, exist_ok=True)

    with open(cache, 'rb') as f:
        eids = sorted(pickle.load(f)['events'].keys())
    print(f'[t0_abs] {len(eids):,} events, bundle {os.path.basename(bundle)}')
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init,
                             initargs=(bundle, cache)) as pool:
        for i, r in enumerate(pool.map(_one, eids, chunksize=16)):
            rows.append(r)
            if (i + 1) % 1000 == 0:
                print(f'  {i + 1:,}/{len(eids):,}', flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pred, spread, table = {}, {}, {}
    fig, axs = plt.subplots(2, 6, figsize=(20, 7), sharex='row')
    for pi, plane in enumerate(('x', 'y')):
        pred[plane], spread[plane] = {}, {}
        by_c = {}
        for r in rows:
            if plane in r:
                c, t0, c2, tan = r[plane]
                by_c.setdefault(c, []).append(t0)
        for c in sorted(by_c):
            ts = np.asarray(by_c[c])
            med = float(np.median(ts))
            sig = robust_sigma(ts)
            n = len(ts)
            pred[plane][c] = round(med, 2)
            spread[plane][c] = round(sig, 1)
            print(f'[t0_abs] {plane} ftst {c}: {med:7.1f} ns  '
                  f'(rsig {sig:5.1f}, n={n}, se {sig / np.sqrt(n):.1f})')
            ax = axs[pi, c]
            ax.hist(ts, bins=np.linspace(med - 250, med + 250, 100),
                    histtype='step', lw=1.5)
            ax.axvline(med, color='r', lw=1)
            ax.set_title(f'{plane} ftst {c}: {med:.1f} '
                         f'(rsig {sig:.0f}, n={n})', fontsize=8)
        cs = sorted(pred[plane])
        v = [pred[plane][c] for c in cs]
        print(f'[t0_abs] {plane} preds {v}  step diffs '
              f'{np.diff(v).round(1).tolist()} (expect ~ -10/step)')
    fig.suptitle(f'{args.run_key}: ref-pinned t0 by ftst class '
                 f'({os.path.basename(bundle)})')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 't0_abs_classes.png'), dpi=120)

    out = dict(pred=pred, spread=spread, run_key=args.run_key,
               bundle=bundle, cache=cache, n_events=len(rows),
               method='ref-pinned per-class median (08_t0_abs_calib.py)')
    with open(os.path.join(out_dir, 't0_abs.json'), 'w') as f:
        json.dump(out, f, indent=1)
    # the bare table run_bench --t0-abs wants
    with open(os.path.join(out_dir, 't0_abs_table.json'), 'w') as f:
        json.dump(pred, f, indent=1)
    print(f'[t0_abs] wrote {out_dir}/t0_abs.json and t0_abs_table.json')


if __name__ == '__main__':
    main()
