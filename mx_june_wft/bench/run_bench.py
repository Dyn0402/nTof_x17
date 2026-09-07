#!/usr/bin/env python3
"""
run_bench.py — score a reconstruction variant on the benchmark cache.

Runs the per-event fit exactly as production's _worker_fit does, with a named
variant switching the behaviour, and scores against the cached M3 truth:

    within 5 mm / reco_far / no-reco (denominator: cached events in the box)
    core sigma |r| (rstd, r < 15), median |r|
    per-plane angle bias / robust sigma / s68 (slope_reliable only)
    runtime per plane-fit

    ../../.venv/bin/python mx_june_wft/bench/run_bench.py sat_det3 \
        --variant baseline --subset 1500 --jobs 5

Variants live in VARIANTS below; each is a dict of switches read by the
worker.
"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

VARIANTS = {
    # production as of 2026-07-29 (candidate seeder, per-plane selection)
    'baseline': {},
    # two-plane time-coincidence candidate selection (implemented, never enabled)
    'pair': {'pair_select': True},
    # candidate ranking on charge-column duration (q_u50 vs the run's full-gap
    # value), dchi2 as tie-break; det3 @1000V measured median q_u50 ~ 360 ns
    'dur': {'reco_globals': {'SCORE_MODE': 'dur', 'U50_EXPECT_NS': 360.0}},
    'durw': {'reco_globals': {'SCORE_MODE': 'durw', 'U50_EXPECT_NS': 360.0}},
    'dur_pair': {'pair_select': True,
                 'reco_globals': {'SCORE_MODE': 'dur', 'U50_EXPECT_NS': 360.0}},
    'durw_pair': {'pair_select': True,
                  'reco_globals': {'SCORE_MODE': 'durw', 'U50_EXPECT_NS': 360.0}},
    # fractional model-error term in the chi2 weights
    'mf3': {'model_globals': {'MODEL_FRAC': 0.03}},
    'mf5': {'model_globals': {'MODEL_FRAC': 0.05}},
    'mf10': {'model_globals': {'MODEL_FRAC': 0.10}},
    # Nelder-Mead convergence check: restart the local fit from its own result
    'nmx': {'patch': 'nm_restart'},
    # deeper charge basis (18 -> 22 x 60 ns): does the Y q_uend pile-up at the
    # basis end (1080 ns) distort the fit?
    'k22': {'k_bins': 22},
    # robust refit of high-chi2 planes (shower/delta blobs): per-strip sigma
    # inflated to the first fit's residual RMS, refit from a fresh global start
    'robust': {'reco_globals': {'ROBUST_REFIT': True}},
    'mf5_robust': {'reco_globals': {'ROBUST_REFIT': True},
                   'model_globals': {'MODEL_FRAC': 0.05}},
    # per-plane sharing delay on Y (the ~4% symmetric Y-angle compression at
    # |tan| > 0.14 suggests Y's delayed copies have a different timescale)
    'ktau080': {'hyper_patch': {'kTauY': 0.80}},
    'ktau120': {'hyper_patch': {'kTauY': 1.20}},
    'ktau140': {'hyper_patch': {'kTauY': 1.40}},
    # coarse-basis global pre-scan (speed): K=9 x 120 ns for the scan stage
    'fast': {'reco_globals': {'PRESCAN_COARSE': True}},
    'fast_robust': {'reco_globals': {'PRESCAN_COARSE': True,
                                     'ROBUST_REFIT': True}},
    # Y-compression diagnostics (0.33 deg symmetric at |tan|>0.14): kernel
    # amplitude, Y timescale, and left/right asymmetry
    'c1p10': {'hyper_patch': {'c1': 0.317}},        # det3 c1 * 1.10
    'kyp10': {'hyper_patch': {'kY': 1.513}},        # det3 kY * 1.10
    'ayp15': {'hyper_patch': {'aY': +0.15}},
    'aym15': {'hyper_patch': {'aY': -0.15}},
    # second t0 scan at the best (p0, w) in the global search
    'iter2': {'reco_globals': {'ITER_SCAN': True}},
    # §21.1: stage-2 global scan sheared to the mesh (p0 - w*u_mid); pair
    # with --t0-abs/--t0-sigma (T2.4 — the collapsed t0 scan pays for it)
    'p0shear': {'reco_globals': {'P0_SHEAR': True}},
    # half-lever arm: tests the valley-alignment reading of the p0shear
    # rejection (12_shear_lever.py measured the full lever ~390 ns = the
    # assumed u_mid, so the LEVER VALUE was not the problem)
    'p0shear200': {'reco_globals': {'P0_SHEAR': 200.0}},
    # the proposed production configuration (2026-07-29 bench outcome):
    # 3% fractional model-error weighting + coarse-basis global pre-scan;
    # pair with a w0-carrying calibration bundle via --bundle
    'prod': {'model_globals': {'MODEL_FRAC': 0.03},
             'reco_globals': {'PRESCAN_COARSE': True}},
    # RC-diffusion delay on the resistive-strip (Y) axis: the +-2 copy at
    # 4*tau (quadratic distance scaling) instead of the linear 2*tau
    'rc4': {'hyper_patch': {'tau2_fac_y': 4.0}},
    'rc4_prod': {'hyper_patch': {'tau2_fac_y': 4.0},
                 'model_globals': {'MODEL_FRAC': 0.03},
                 'reco_globals': {'PRESCAN_COARSE': True}},
    # --- readout-window ablation (see WINDOW_ABLATION_2026-07-30.md) --------
    # Emulate a shorter DAQ window by cropping the cached 32-sample windows.
    # `crop: (start, n)` keeps samples [start, start+n); start is measured, not
    # assumed -- `framing_compare.py` matches the bench frame to the beam frame
    # by the prompt onset sample, because the beam runs use a different DREAM
    # latency AND a different trigger G&D delay. A negative start prepends zero
    # (= baseline) samples. Build these with --crop START:N; the named ones
    # below are the run_79 scan, filled in once the framing is measured.
}


def crop_variant(start, n, k_bins=None, base='prod'):
    """A cropped-window variant on top of a base variant."""
    v = dict(VARIANTS[base])
    v['crop'] = (int(start), int(n))
    if k_bins:
        v['k_bins'] = int(k_bins)
    return v


def rstd(v, ns=3, it=5):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    for _ in range(it):
        m, s = np.median(v), np.std(v)
        k = np.abs(v - m) <= ns * s
        if k.all() or k.sum() < 10:
            break
        v = v[k]
    return float(np.std(v))


def robust_sigma(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if len(a) else np.nan


# ------------------------------------------------------------------- worker
_CAL = None
_VAR = {}


def _worker_init(bundle_path, variant):
    global _CAL, _VAR
    from wft.calib import CalibrationBundle
    from wft import model as wm
    from wft import reco as wr
    _CAL = CalibrationBundle.load(bundle_path)
    _VAR = dict(variant)
    for k, v in _VAR.get('hyper_patch', {}).items():
        _CAL.hyper[k] = v
    wm.use_calibration(_CAL)
    for k, v in _VAR.get('reco_globals', {}).items():
        setattr(wr, k, v)
    for k, v in _VAR.get('model_globals', {}).items():
        setattr(wm, k, v)
    if _VAR.get('k_bins'):
        wm.set_depth_bins(int(_VAR['k_bins']))
    if _VAR.get('patch') == 'nm_restart':
        orig = wm.fit_plane_raw

        def wrapped(P, plane, p0_init, w_init, t0_init, hyper=None,
                    fix_p0w=None):
            r = orig(P, plane, p0_init, w_init, t0_init, hyper=hyper,
                     fix_p0w=fix_p0w)
            if fix_p0w is not None or r is None or not np.isfinite(r['chi2']):
                return r
            r2 = orig(P, plane, r['p0'], r['w'], r['t0'], hyper=hyper)
            return r2 if r2['chi2'] < r['chi2'] - 0.01 else r

        wm.fit_plane_raw = wrapped


def _crop_windows(wins, start, n):
    """Keep samples [start, start+n) of every window; a negative start prepends
    zero (baseline) samples so the signal keeps its position in the frame."""
    out = {}
    for plane, cand in wins.items():
        new = []
        for P in (cand or []):
            W = np.asarray(P['W'])
            if start < 0:
                W = np.concatenate(
                    [np.zeros((W.shape[0], -start), dtype=W.dtype), W], axis=1)
                s = 0
            else:
                s = start
            Q = dict(P)
            Q['W'] = np.ascontiguousarray(W[:, s:s + n])
            new.append(Q)
        out[plane] = new
    return out


def _worker_fit(payload):
    from wft import reco as wr
    eid, wins, sinfo, n_hits, spark, fd, ftst = payload
    ftst = ftst or {}
    if _VAR.get('crop'):
        wins = _crop_windows(wins, *_VAR['crop'])
    t0 = time.time()
    fits, all_fits, n_planes = {}, {}, 0
    for plane in ('x', 'y'):
        cand = wins.get(plane)
        if not cand:
            fits[plane], all_fits[plane] = None, []
            continue
        seeds = sinfo.get(plane)

        class _S:
            pass

        sobjs = []
        for s in (seeds or []):
            o = _S()
            o.n_strips, o.n_dropped = s['n_strips'], s['n_dropped']
            sobjs.append(o)
        n_planes += len(cand)
        try:
            best, ranked = wr.fit_plane_candidates(
                cand, plane, _CAL, seeds=sobjs or None, return_all=True,
                t0_prior=wr.t0_prior_for(_CAL, plane, ftst.get(plane)))
            fits[plane], all_fits[plane] = best, ranked
        except Exception:
            fits[plane], all_fits[plane] = None, []
    if _VAR.get('pair_select'):
        try:
            fits = wr.select_pair(all_fits, fd, _CAL)
        except Exception:
            pass
    row = wr.row_from_fits(eid, fits, n_hits, spark)
    for plane in ('x', 'y'):
        f = ftst.get(plane)
        row[f'{plane}_ftst'] = int(f) if f is not None else -1
    row['_dt'] = time.time() - t0
    row['_n_plane_fits'] = n_planes
    return row


# -------------------------------------------------------------------- score
def score(rows, events, box, max_dropped=2):
    idx = {r['event_id']: r for r in rows}
    res = dict(n=0, near=0, far=0, no_reco=0, rs=[], dth={'x': [], 'y': []},
               vimp={'x': [], 'y': []}, tanref={'x': [], 'y': []})
    for eid, ev in events.items():
        t = ev['truth']
        if not (np.isfinite(t['ref_x']) and np.isfinite(t['ref_y'])):
            continue
        if box and not (box['x'][0] <= t['ref_x'] <= box['x'][1]
                        and box['y'][0] <= t['ref_y'] <= box['y'][1]):
            continue
        res['n'] += 1
        r = idx.get(eid)
        ok = r is not None and r['x_ok'] and r['y_ok']
        if ok and max_dropped is not None:
            ok = (r['x_n_dropped'] <= max_dropped
                  and r['y_n_dropped'] <= max_dropped)
        if not ok:
            res['no_reco'] += 1
        else:
            d = float(np.hypot(r['x_p0'] - t['ref_x'], r['y_p0'] - t['ref_y']))
            res['rs'].append(d)
            res['near' if d <= 5.0 else 'far'] += 1
        if r is not None:
            for p in ('x', 'y'):
                if r[f'{p}_ok'] and r[f'{p}_slope_reliable'] \
                        and np.isfinite(t[f'tan_{p}']):
                    dth = (np.degrees(np.arctan(r[f'{p}_tan_theta']))
                           - np.degrees(np.arctan(t[f'tan_{p}'])))
                    res['dth'][p].append(dth)
                    res['vimp'][p].append(r[f'{p}_w'] * 1e3 / t[f'tan_{p}'])
                    res['tanref'][p].append(t[f'tan_{p}'])
                # reference-selected sample: unbiased by construction (no
                # selection on the fitted slope)
                if r[f'{p}_ok'] and np.isfinite(t[f'tan_{p}']) \
                        and abs(t[f'tan_{p}']) >= 0.14:
                    dth = (np.degrees(np.arctan(r[f'{p}_tan_theta']))
                           - np.degrees(np.arctan(t[f'tan_{p}'])))
                    res.setdefault('dth14', {'x': [], 'y': []})[p].append(
                        dth if t[f'tan_{p}'] > 0 else -dth)
                # near-vertical, reference-selected: where the t0 prior /
                # joint fit are expected to act (T1.1, doc §22)
                if r[f'{p}_ok'] and np.isfinite(t[f'tan_{p}']) \
                        and abs(t[f'tan_{p}']) < 0.08:
                    dth = (np.degrees(np.arctan(r[f'{p}_tan_theta']))
                           - np.degrees(np.arctan(t[f'tan_{p}'])))
                    res.setdefault('dthnv', {'x': [], 'y': []})[p].append(dth)
    return res


ANGLE_BINS = [(0.08, 0.14), (0.14, 0.20), (0.20, 0.28), (0.28, 0.45)]


def summarize(res, rows, label):
    n = res['n']
    rs = np.array(res['rs'])
    out = dict(label=label, n=n,
               within5=100.0 * res['near'] / n,
               reco_far=100.0 * res['far'] / n,
               no_reco=100.0 * res['no_reco'] / n,
               core_sigma=rstd(rs[rs < 15]) if len(rs) else np.nan,
               median_r=float(np.median(rs)) if len(rs) else np.nan)
    for p in ('x', 'y'):
        dth = np.array(res['dth'][p])
        vimp = np.array(res['vimp'][p])
        at = np.abs(np.array(res['tanref'][p]))
        med = float(np.median(dth)) if len(dth) else np.nan
        medv = [float(np.nanmedian(vimp[(at >= lo) & (at < hi)]))
                if ((at >= lo) & (at < hi)).sum() else np.nan
                for lo, hi in ANGLE_BINS]
        out[f'bias_{p}'] = med
        out[f'sigma_{p}'] = robust_sigma(dth)
        out[f's68_{p}'] = (float(np.percentile(np.abs(dth - med), 68))
                           if len(dth) else np.nan)
        out[f'vspread_{p}'] = (float(np.nanmax(medv) - np.nanmin(medv))
                               if np.isfinite(medv).any() else np.nan)
        d14 = np.array(res.get('dth14', {}).get(p, []))
        out[f'comp14_{p}'] = float(np.median(d14)) if len(d14) else np.nan
        out[f'sig14_{p}'] = robust_sigma(d14)
        dnv = np.array(res.get('dthnv', {}).get(p, []))
        out[f'signv_{p}'] = robust_sigma(dnv)
        out[f'biasnv_{p}'] = float(np.median(dnv)) if len(dnv) else np.nan
        out[f'n_nv_{p}'] = int(len(dnv))
    dts = [r['_dt'] for r in rows if '_dt' in r]
    nf = sum(r.get('_n_plane_fits', 0) for r in rows)
    out['s_per_plane'] = float(np.sum(dts) / max(nf, 1))
    out['n_plane_fits'] = int(nf)
    return out


def fmt(s):
    return (f"{s['label']:22s} within5 {s['within5']:6.2f}%  far {s['reco_far']:5.2f}%  "
            f"noreco {s['no_reco']:5.2f}%  core {s['core_sigma']:.3f}  med {s['median_r']:.3f}  "
            f"| sX {s['sigma_x']:.2f} bX {s['bias_x']:+.2f}  sY {s['sigma_y']:.2f} "
            f"bY {s['bias_y']:+.2f}  vsp {s['vspread_x']:.1f}/{s['vspread_y']:.1f}  "
            f"cmp14 {s.get('comp14_x', float('nan')):+.2f}/"
            f"{s.get('comp14_y', float('nan')):+.2f} "
            f"s14 {s.get('sig14_x', float('nan')):.2f}/"
            f"{s.get('sig14_y', float('nan')):.2f}  "
            f"sNV {s.get('signv_x', float('nan')):.2f}/"
            f"{s.get('signv_y', float('nan')):.2f}  "
            f"| {s['s_per_plane']:.2f} s/fit ({s['n_plane_fits']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key')
    ap.add_argument('--cache', default=None)
    ap.add_argument('--bundle', default=None,
                    help='calibration bundle override (default: the one in '
                         'the cache meta)')
    ap.add_argument('--variant', default='baseline')
    ap.add_argument('--patch', default=None,
                    help='JSON dict merged into the bundle hyper on top of '
                         'the variant (e.g. \'{"kTauY": 2.4}\')')
    ap.add_argument('--tag', default=None,
                    help='label for the output json (default: variant name)')
    ap.add_argument('--crop', default=None,
                    help='readout-window ablation START:N -- keep samples '
                         '[START, START+N) of each cached window (see '
                         'framing_compare.py for the measured START)')
    ap.add_argument('--k-bins', type=int, default=None,
                    help='override the charge-basis depth bins (K)')
    ap.add_argument('--t0-abs', default=None,
                    help='JSON file {plane: {ftst: t0_pred_ns}} enabling the '
                         'absolute-t0 trigger prior (T1.1); requires a cache '
                         'with per-plane ftst (rebuild post-2026-08-11)')
    ap.add_argument('--t0-sigma', type=float, default=25.0,
                    help='prior width [ns] used with --t0-abs')
    ap.add_argument('--subset', type=int, default=None,
                    help='deterministic subset of N events')
    ap.add_argument('--subset-mod', default=None,
                    help='I:N -- keep every Nth event starting at I, applied '
                         'BEFORE --subset. Use 0:2 / 1:2 for a disjoint '
                         'scan / validation split (see ANGLE_SCAN)')
    ap.add_argument('--jobs', type=int, default=5)
    ap.add_argument('--out-rows', default=None,
                    help='also dump per-event rows (parquet)')
    ap.add_argument('--out-dir', default=None,
                    help='where to write bench_<tag>.json (default '
                         '<OUT_BASE>/wft/bench). Giving both --cache and '
                         '--out-dir avoids the qa_config run registry '
                         'entirely, which is what the condor worker needs')
    args = ap.parse_args()

    # The run registry is only needed to locate the cache and the output dir;
    # on a batch worker neither exists, so skip it when both are given.
    out_base = None
    if args.cache is None or args.out_dir is None:
        from qa_config import get_config, setup_paths
        setup_paths()
        out_base = get_config(args.run_key).OUT_BASE
    cache_path = args.cache or os.path.join(out_base, 'wft', 'bench_cache.pkl')
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    events, meta = data['events'], data['meta']
    if args.subset_mod:
        i, n = (int(x) for x in args.subset_mod.split(':'))
        keep = set(sorted(events)[i::n])
        events = {k: v for k, v in events.items() if k in keep}
        print(f'split {i}:{n} -> {len(events):,} events')
    if args.subset:
        keys = sorted(events)
        rng = np.random.RandomState(42)
        keep = set(rng.choice(keys, size=min(args.subset, len(keys)),
                              replace=False).tolist())
        events = {k: v for k, v in events.items() if k in keep}
    print(f'{len(events):,} events, variant={args.variant}')

    variant = dict(VARIANTS[args.variant])
    if args.crop:
        s, n = (int(x) for x in args.crop.split(':'))
        variant['crop'] = (s, n)
        print(f'window ablation: keeping samples [{s}, {s + n}) of 32')
    if args.k_bins:
        variant['k_bins'] = args.k_bins
    if args.patch:
        hp = dict(variant.get('hyper_patch', {}))
        hp.update(json.loads(args.patch))
        variant['hyper_patch'] = hp
    if args.t0_abs:
        with open(args.t0_abs) as f:
            t0a = {p: {int(k): float(v) for k, v in d.items()}
                   for p, d in json.load(f).items()}
        rg = dict(variant.get('reco_globals', {}))
        rg['T0_ABS'] = t0a
        rg['T0_PRIOR_SIGMA'] = float(args.t0_sigma)
        variant['reco_globals'] = rg
        print(f'absolute-t0 prior: sigma {args.t0_sigma} ns, table {args.t0_abs}')
    payloads = [(e, ev['wins'], ev['seeds'], ev['n_hits'], ev['spark'],
                 ev['ftst_diff'], ev.get('ftst'))
                for e, ev in sorted(events.items())]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=_worker_init,
                             initargs=(args.bundle or meta['bundle'],
                                       variant)) as pool:
        for i, row in enumerate(pool.map(_worker_fit, payloads, chunksize=8)):
            rows.append(row)
            if (i + 1) % 500 == 0:
                print(f'  {i + 1:,}/{len(payloads):,}  '
                      f'({time.time() - t0:.0f} s)', flush=True)
    wall = time.time() - t0

    res = score(rows, events, meta.get('box'))
    s = summarize(res, rows, args.tag or args.variant)
    s['wall_s'] = wall
    print(fmt(s))
    res_nc = score(rows, events, meta.get('box'), max_dropped=None)
    s_nc = summarize(res_nc, rows, args.variant + ' (nocut)')
    print(fmt(s_nc))
    s['nocut'] = {k: v for k, v in s_nc.items() if k != 'label'}
    s['config'] = dict(run_key=args.run_key, variant=args.variant,
                       patch=args.patch, crop=args.crop, k_bins=args.k_bins,
                       subset=args.subset, subset_mod=args.subset_mod,
                       bundle=os.path.basename((args.bundle or '').rstrip('/')))
    out_dir = args.out_dir or os.path.join(out_base, 'wft', 'bench')
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.tag or args.variant}_{len(events)}"
    if args.bundle:
        tag += '_' + os.path.basename(args.bundle.rstrip('/'))
    with open(os.path.join(out_dir, f'bench_{tag}.json'), 'w') as f:
        json.dump(s, f, indent=1)
    if args.out_rows:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(args.out_rows, index=False)
    print(f'wrote {out_dir}/bench_{tag}.json  (wall {wall:.0f} s)')


if __name__ == '__main__':
    main()
