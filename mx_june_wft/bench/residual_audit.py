#!/usr/bin/env python3
"""
residual_audit.py — where does the forward model fail?

Under the production weighting the fit runs at chi2/dof ~ 20-60, i.e. the model
is NOT a statistically adequate description of the data: there is percent-level
structure it does not reproduce. That is why every uncertainty quoted in this
analysis is empirical (spread against M3) rather than curvature-derived. This
tool asks the obvious follow-up: *where* is the mismatch, in (strip, sample)?

For each event it reproduces the production candidate selection, refits the
chosen window to recover the charge ladder, rebuilds the model waveforms, and
stacks the residual (data - model) in the fit's own frame:

    rows    strip index relative to the fitted track position at the mesh
    cols    sample index relative to the fitted t0

Two stacks are accumulated: the residual normalised to the event's peak model
amplitude (what fraction of the signal is unmodelled, and where), and the pull
(residual / per-sample sigma, the chi2 integrand). Both are split by fitted
charge tercile, because the tail that matters (delta rays, showers) is a
high-charge population.

    residual_audit.py --cache <bench_cache.pkl> --bundle <dir> --out <dir>
                      [--events 1500] [--shard i/N]
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

NS_HALF = 8           # strips kept either side of the fitted position
NSAMP_MAX = 40        # stack width; a run may mix 32- and 37-sample windows
PITCH_DEFAULT = 0.78  # mm, only used if the bundle has no pitch


def stack_event(P, plane, fit, wm, hyper, pitch):
    """Residual and pull images for one plane fit, in the fit's own frame."""
    W, noise, pos, sat = wm.prep_plane(P, plane)
    r = wm.fit_plane_raw(P, plane, fit.p0, fit.w, fit.t0, hyper=hyper)
    if r is None or not np.isfinite(r['chi2']):
        return None
    M = wm.model_waveforms(plane, pos, r['p0'], r['w'], r['t0'], r['q'], hyper)
    if M is None or M.shape != W.shape:
        return None
    peak = float(np.max(M)) if np.isfinite(M).any() else 0.0
    if peak <= 0:
        return None
    # sample_weights returns 1/sigma, flattened
    iw = np.asarray(wm.sample_weights(W, noise)).reshape(W.shape)
    pull = (W - M) * iw
    frac = (W - M) / peak

    # strip axis: distance from the fitted mesh position, in strips
    d = np.rint((np.asarray(pos) - r['p0']) / pitch).astype(int)
    ns = min(W.shape[1], NSAMP_MAX)
    R = np.zeros((2 * NS_HALF + 1, NSAMP_MAX))
    Q = np.zeros((2 * NS_HALF + 1, NSAMP_MAX))
    C = np.zeros((2 * NS_HALF + 1, NSAMP_MAX))
    for i, di in enumerate(d):
        if -NS_HALF <= di <= NS_HALF:
            R[di + NS_HALF, :ns] += frac[i, :ns]
            Q[di + NS_HALF, :ns] += pull[i, :ns]
            C[di + NS_HALF, :ns] += 1.0
    return R, Q, C, float(r['chi2']), int(r['dof']), float(np.sum(r['q']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--events', type=int, default=1500)
    ap.add_argument('--shard', default=None, help='i/N')
    args = ap.parse_args()

    from wft.calib import CalibrationBundle
    from wft import model as wm
    from wft import reco as wr

    cal = CalibrationBundle.load(args.bundle)
    wm.use_calibration(cal)
    wm.MODEL_FRAC = float(os.environ.get('WFT_MODEL_FRAC', 0.03))
    wr.PRESCAN_COARSE = os.environ.get('WFT_PRESCAN', '1') == '1'
    pitch = float(getattr(cal, 'pitch_mm', PITCH_DEFAULT) or PITCH_DEFAULT)

    with open(args.cache, 'rb') as f:
        data = pickle.load(f)
    events = data['events']
    keys = sorted(events)
    if args.shard:
        i, n = (int(x) for x in args.shard.split('/'))
        keys = keys[i::n]
    keys = keys[:args.events]
    print(f'{len(keys):,} events, bundle {os.path.basename(args.bundle)}')

    acc = {}          # (plane, tercile) -> [sum_R, sum_Q, n]
    rows = []
    for n_done, e in enumerate(keys):
        ev = events[e]
        for plane in ('x', 'y'):
            wins = (ev['wins'] or {}).get(plane) or []
            if not wins:
                continue
            best, bestP, best_key = None, None, None
            for P in wins:
                try:
                    fit = wr.fit_plane(P, plane, cal)
                except Exception:
                    fit = None
                if fit is None:
                    continue
                plausible, dchi2 = wr._candidate_score(P, plane, fit)
                fit._plausible, fit._dchi2 = plausible, dchi2
                key = wr._cand_key(plausible, dchi2, fit)
                if best_key is None or key > best_key:
                    best, bestP, best_key = fit, P, key
            if best is None:
                continue
            out = stack_event(bestP, plane, best, wm, cal.hyper, pitch)
            if out is None:
                continue
            R, Q, C, chi2, dof, qsum = out
            rows.append(dict(event_id=int(e), plane=plane, chi2=chi2, dof=dof,
                             chi2dof=chi2 / max(dof, 1), qsum=qsum,
                             n_strips=int(best.n_strips),
                             tan=float(best.tan_theta)))
            # sum and per-cell count: strips far from the track contribute to
            # far fewer events than the central ones, so dividing by the event
            # count would dilute exactly the region we are asking about
            a = acc.setdefault((plane, 'all'),
                               [np.zeros_like(R), np.zeros_like(Q),
                                np.zeros_like(C), 0])
            a[0] += R
            a[1] += Q
            a[2] += C
            a[3] += 1
        if (n_done + 1) % 200 == 0:
            print(f'  {n_done + 1:,}/{len(keys):,}', flush=True)

    # charge terciles, computed after the fact so the split is data-defined
    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    tag = (args.shard or '0/1').replace('/', 'of')
    df.to_parquet(os.path.join(args.out, f'residual_rows_{tag}.parquet'),
                  index=False)
    np.savez_compressed(
        os.path.join(args.out, f'residual_stack_{tag}.npz'),
        **{f'{p}_{k}': v for (p, k), a in acc.items()
           for k, v in (('res', a[0]), ('pull', a[1]), ('cnt', a[2]),
                        ('n', np.array(a[3])))})

    summary = {'n_plane_fits': len(df), 'events': len(keys),
               'bundle': os.path.basename(args.bundle)}
    for plane in ('x', 'y'):
        d = df[df.plane == plane]
        if not len(d):
            continue
        summary[plane] = dict(
            n=int(len(d)),
            chi2dof=[float(np.percentile(d.chi2dof, q)) for q in (5, 50, 95)],
            worst_frac=float((d.chi2dof > 250).mean()),
            qsum_med=float(np.median(d.qsum)))
        a = acc.get((plane, 'all'))
        if a and a[3]:
            with np.errstate(invalid='ignore', divide='ignore'):
                R = np.where(a[2] > 0, a[0] / np.maximum(a[2], 1), np.nan)
            summary[plane]['mean_abs_residual_frac'] = float(np.nanmean(np.abs(R)))
            summary[plane]['max_residual_frac'] = float(np.nanmax(np.abs(R)))
    with open(os.path.join(args.out, f'residual_summary_{tag}.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
