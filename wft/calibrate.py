"""
Per-detector calibration: measure the impulse template, the sharing kernel and
the drift velocity for one detector under one set of run conditions.

This is the one part of the chain that is *allowed* to use the M3 reference:
calibration is a fit of the detector's response with the track geometry pinned
to the reference ("ref-pinned"), exactly as in the R&D study. Reconstruction
afterwards never sees the reference.

Stages (ports of the R&D scripts 01/11/13/12):

  1. cache      waveform windows along the reference corridor for N events
  2. template   per-plane impulse response from bright inclined strips
  3. hypers     8-parameter ref-pinned fit: c1, c2, kY, tau_s, sigma_s,
                sigma_p0, Dp, v          <- the expensive stage
  4. dt_xy      FEU t0 offset by ftst difference (for the joint fit)

The per-channel gain map (R&D script 12) is deliberately *not* refitted here:
the measured spread is 1.4-1.5 % and the ablation study found it changes the
per-event angle by less than the statistical noise. Bundles start with unit
gains; import a measured map with CalibrationBundle.from_legacy if one exists.

    python -m wft.calibrate <run_key> [--events 400] [--train 180] [--jobs 12]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize

from .calib import CalibrationBundle, HYPER_NAMES
from . import model as wm

SAT_FRAC = 0.98
TEMPLATE_GRID = np.arange(-360, 1400, 10.0)
TEMPLATE_TAN_MIN = 0.22          # only clearly inclined tracks carry a clean rise
TEMPLATE_MIN_AMP = 500.0
HYPER_X0 = np.array([0.306, 0.057, 1.0, 47.0, 87.0, 0.098, 0.0114, 36.65])
C1_MIN = 0.05                    # physical floor on the sharing kernel
HYPER_SCALE = np.array([0.05, 0.03, 0.15, 15.0, 20.0, 0.06, 0.005, 2.0])
FIT_NAMES = list(HYPER_NAMES) + ['v']


# --------------------------------------------------------------------- cache
def build_cache(cfg, n_events: int, pad_mm: float = 5.0, z_lo: float = -3.0,
                z_hi: float = 33.0, out_path: str | None = None,
                res_cut_mm: float = 10.0, veto: int = 50):
    """Waveform windows along the M3 reference corridor, for calibration only."""
    import uproot
    from . import io as wio
    from qa_config import M3_CHI2_CUT, M3_MIN_NCLUS
    import cosmic_micro_tpc_analysis as cm
    from M3RefTracking import M3RefTracking, get_xy_angles

    cache = os.path.join(cfg.out_dir('cache'), f'event_results_veto{veto}.pkl')
    align = os.path.join(cfg.OUT_BASE, f'alignment_tpc_veto{veto}', 'alignment.json')
    if not (os.path.exists(cache) and os.path.exists(align)):
        raise SystemExit(f'need the hits-chain alignment + event cache for the '
                         f'reference geometry:\n  {cache}\n  {align}\n'
                         f'(run 03_alignment_and_tpc.py <key> --veto={veto} first)')
    results = pickle.load(open(cache, 'rb'))
    best = cm.load_alignment(align)
    rays = M3RefTracking(cfg.m3_tracking_dir, chi2_cut=M3_CHI2_CUT,
                         min_nclus=M3_MIN_NCLUS)
    xang, _y, anum = get_xy_angles(rays.ray_data)
    xang = best.ref_x_sign * np.array(xang)
    cm.attach_reference_positions(results, rays, best, xang, anum)

    events = {}
    for r in results:
        if not (r.has_x and r.has_y):
            continue
        if not np.isfinite(r.radial_residual_mm) or r.radial_residual_mm > res_cut_mm:
            continue
        if np.isnan(r.ref_tan_theta_x) or np.isnan(r.ref_mesh_x_mm):
            continue
        tx, ty = cm._rotate_ref_tangents(r, best)
        events[int(r.event_id)] = dict(eid=int(r.event_id), tan_x=float(tx),
                                       tan_y=float(ty),
                                       ref_mesh_x=float(r.ref_mesh_x_mm),
                                       ref_mesh_y=float(r.ref_mesh_y_mm))
    keep = set(sorted(events)[:n_events * 3])       # over-request; some lack waveforms
    events = {k: v for k, v in events.items() if k in keep}
    print(f'[calib] {len(events):,} reference-matched candidate events')

    pos_maps = wio.strip_position_map(cfg)
    for plane, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
        pm = pos_maps[feu]
        for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu):
            rdr = wio.FeuReader(f)
            want = set(events) & set(rdr.event_ids.tolist())
            if not want:
                continue
            for eid, ftst, wfm in rdr.iter_events(want):
                ev = events[eid]
                p0 = ev[f'ref_mesh_{plane}']
                tn = ev[f'tan_{plane}']
                a, b = p0 + z_lo * tn, p0 + z_hi * tn
                lo, hi = min(a, b) - pad_mm, max(a, b) + pad_mm
                ch = np.where((pm >= lo) & (pm <= hi))[0]
                ch = ch[np.argsort(pm[ch])]
                if len(ch) < 4:
                    continue
                ev[plane] = dict(ch=ch.astype(np.int16), pos=pm[ch].astype(np.float32),
                                 W=wfm[ch].astype(np.float32),
                                 noise=np.maximum(rdr.noise[ch], 3.0).astype(np.float32))
                ev[f'ftst_{plane}'] = ftst
            if sum(1 for e in events.values() if plane in e) >= n_events:
                break
    events = {k: v for k, v in events.items() if 'x' in v and 'y' in v}
    events = {k: events[k] for k in sorted(events)[:n_events]}
    print(f'[calib] {len(events):,} events with both waveform windows')
    if out_path:
        with open(out_path, 'wb') as f:
            pickle.dump(events, f, protocol=4)
    return events


# ------------------------------------------------------------------ template
def measure_templates(events, sat_adc=3550.0, sample_ns=60.0):
    """Per-plane impulse response, normalised and baseline-referenced, from the
    brightest strips of clearly inclined tracks (their rise is not contaminated
    by a neighbour arriving at the same time)."""
    def t50(w):
        ipk = int(np.argmax(w))
        a = w[ipk]
        for k in range(1, ipk + 1):
            if w[k] >= 0.5 * a > w[k - 1]:
                return k - 1 + (0.5 * a - w[k - 1]) / (w[k] - w[k - 1])
        return np.nan

    out = {}
    for plane in ('x', 'y'):
        acc = []
        for ev in events.values():
            if plane not in ev or abs(ev[f'tan_{plane}']) < TEMPLATE_TAN_MIN:
                continue
            W = np.asarray(ev[plane]['W'], np.float32)
            ns = W.shape[1]
            amax = W.max(axis=1)
            for i in np.argsort(amax)[::-1][:2]:
                w = W[i]
                a = w.max()
                ipk = int(np.argmax(w))
                if a < TEMPLATE_MIN_AMP or a > sat_adc or ipk < 6 or ipk > ns - 12:
                    continue
                c = t50(w)
                if np.isfinite(c):
                    tt = (np.arange(ns) - c) * sample_ns
                    acc.append(np.interp(TEMPLATE_GRID, tt, w / a,
                                         left=np.nan, right=np.nan))
        if not acc:
            raise SystemExit(f'[calib] no template candidates on plane {plane} — '
                             f'is the detector gain high enough? '
                             f'(need |tan| > {TEMPLATE_TAN_MIN}, amp > {TEMPLATE_MIN_AMP})')
        t = np.nanmedian(np.array(acc), axis=0)
        t -= np.nanmedian(t[TEMPLATE_GRID < -250])
        out[plane] = np.nan_to_num(t)
        ipk = int(np.nanargmax(t))
        r10 = TEMPLATE_GRID[np.argmax(t >= 0.1)]
        r90 = TEMPLATE_GRID[np.argmax(t >= 0.9)]
        print(f'[calib] template {plane}: n={len(acc)} rise10-90={r90 - r10:.0f} ns '
              f'peak@{TEMPLATE_GRID[ipk]:.0f} ns undershoot={np.nanmin(t):.3f}')
    return TEMPLATE_GRID, out


# -------------------------------------------------------------------- hypers
_EV = None


def _init_hyper(cache_path, bundle_path):
    global _EV
    with open(cache_path, 'rb') as f:
        _EV = pickle.load(f)
    wm.use_calibration(CalibrationBundle.load(bundle_path))


def _event_chi2(payload):
    eid, hyper, v, warm = payload
    ev = _EV[eid]
    tot, t0s = 0.0, {}
    for plane in ('x', 'y'):
        if plane not in ev:
            continue
        P = ev[plane]
        if np.asarray(P['W']).shape[1] != wm.NSAMP:
            wm.set_nsamp(np.asarray(P['W']).shape[1])
        wline = ev[f'tan_{plane}'] * v * 1e-3
        p0l = ev[f'ref_mesh_{plane}']
        W, noise, pos, sat = wm.prep_plane(P, plane)
        wt0 = warm.get(plane)
        grid = (np.arange(150.0, 900.0, 30.0) if wt0 is None
                else np.arange(wt0 - 60.0, wt0 + 61.0, 15.0))
        chis = np.array([wm.chi2_plane(plane, W, noise, pos, sat, p0l, wline,
                                       float(t), hyper)[0] for t in grid])
        j = int(np.argmin(chis))
        if 0 < j < len(grid) - 1 and np.isfinite(chis[j - 1:j + 2]).all():
            a, b, c = chis[j - 1], chis[j], chis[j + 1]
            den = a - 2 * b + c
            frac = 0.5 * (a - c) / den if den > 0 else 0.0
            t0b = grid[j] + frac * (grid[1] - grid[0])
        else:
            t0b = grid[j]
        if np.isfinite(chis[j]):
            tot += float(chis[j])
            t0s[plane] = float(t0b)
    return eid, tot, t0s


def fit_hypers(cache_path, bundle_path, train_ids, jobs=12, maxiter=130,
               x0=None, v_fixed=None, fixed=None, extra_hyper=None):
    """8-parameter ref-pinned Nelder-Mead. This is the expensive stage:
    ~15 s per objective evaluation on 12 cores for 180 events.

    ``v_fixed`` pins the drift velocity instead of fitting it. Use it on the
    low-statistics chambers: v is a property of the gas and the field, not of
    the chamber, so a detector at 700 V can take the value the drift scan
    measured at 700 V (det6's free fit landed 1.3 % from it, which is the
    evidence this is safe). Pinning it also breaks the v <-> sharing
    degeneracy that wrecked det7's free fit.

    ``fixed`` pins any subset of the seven kernel/geometry hypers by name
    (e.g. ``{'c1': 0.28, 'tau_s': 60.0}``) and fits only the rest. Use it to
    impose an externally *measured* kernel — the H4 beam measured det4's
    sharing directly (c1, the +-1 delay) and showed it is gain- and
    drift-invariant, which is exactly the constraint the cosmic fit cannot
    produce on its own (the v <-> sharing degeneracy again).

    ``extra_hyper`` rides along in every chi2 evaluation without being fitted
    — the hypers outside HYPER_NAMES (``kTauY``, ``cX``, ``share_lp``), which
    the objective's dict(zip(HYPER_NAMES, ...)) would otherwise silently drop.
    """
    fixed = dict(fixed or {})
    unknown = set(fixed) - set(HYPER_NAMES)
    if unknown:
        raise SystemExit(f'[calib] unknown fixed hyper(s): {sorted(unknown)}')
    warm = {e: {} for e in train_ids}
    x0 = HYPER_X0 if x0 is None else np.asarray(x0, float)
    x0 = x0.copy()
    for k, val in fixed.items():
        x0[HYPER_NAMES.index(k)] = val
    free = [i for i in range(8)
            if not (i < 7 and HYPER_NAMES[i] in fixed)
            and not (i == 7 and v_fixed is not None)]
    neval = [0]

    def expand(xf):
        x = x0.copy()
        x[free] = xf
        return x

    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_hyper,
                             initargs=(cache_path, bundle_path)) as pool:
        def total_chi2(hv):
            hyper = dict(zip(HYPER_NAMES, hv[:7]))
            if extra_hyper:
                hyper.update(extra_hyper)
            v = v_fixed if v_fixed is not None else hv[7]
            c = 0.0
            for eid, tot, t0s in pool.map(
                    _event_chi2, [(e, hyper, v, warm[e]) for e in train_ids],
                    chunksize=6):
                c += tot
                warm[eid] = t0s
            neval[0] += 1
            return c

        t0 = time.time()
        c0 = total_chi2(x0)
        print(f'[calib] initial chi2 {c0:.4e} ({time.time() - t0:.0f} s/eval)',
              flush=True)

        def obj(xf):
            x = expand(np.asarray(xf))
            # c1 has a physical floor: these are resistive strips and the
            # sharing is a design property, measured at 0.2-0.5 across the
            # fleet. Without this bound the fit can run to c1 -> 0 and absorb
            # the missing sharing into a huge sigma_p0 and a wrong v -- exactly
            # what happened on det7 (c1 = 0.004, kY = 6.6, sigma_p0 = 0.52 mm,
            # v = 36.7 at a field where 26.4 is measured). The v <-> sharing
            # degeneracy is documented in WAVEFORM_FIRST_THREADING.md §17.2.
            if x[0] < C1_MIN:
                return 2 * c0
            if (x[:3] < 0).any() or x[3] < 0 or x[4] < 0 or x[5] < 0.03 or \
                    x[6] < 0 or not (5 < x[7] < 60):
                return 2 * c0
            if v_fixed is not None:
                x = np.array(list(x[:7]) + [v_fixed])
            c = total_chi2(x)
            print(f'[calib]   eval{neval[0]:3d} {np.round(x, 4)} {c:.5e}', flush=True)
            return c

        xf0 = x0[free]
        simplex = np.array([xf0] + [xf0 + np.eye(len(free))[j] * HYPER_SCALE[free][j]
                                    for j in range(len(free))])
        res = minimize(obj, xf0, method='Nelder-Mead',
                       options=dict(initial_simplex=simplex, xatol=1e-3,
                                    fatol=c0 * 1e-4, maxiter=maxiter))
    xfull = expand(res.x)
    out = {k: float(v) for k, v in zip(FIT_NAMES, xfull)}
    out['chi2'] = float(res.fun)
    out['chi2_init'] = float(c0)
    out['n_train'] = len(train_ids)
    if fixed:
        out['fixed'] = {k: float(v) for k, v in fixed.items()}
    return out


def measure_t0_abs(events, bundle_path, hyper, v, sample=None):
    """Absolute-t0 prediction per plane per ftst class, for the trigger prior
    (T1.1). Ref-pinned fits, like measure_dt_xy: with (p0, w) pinned to the
    reference, t0 is the only free parameter and is far more stable than the
    free fit's (whose chi2 surface has near-degenerate minima 60 ns apart).
    Returns ({plane: {ftst: median t0}}, {plane: {ftst: robust sigma}})."""
    wm.use_calibration(CalibrationBundle.load(bundle_path))
    vals = {'x': {}, 'y': {}}
    for eid in sorted(events)[:sample] if sample else sorted(events):
        ev = events[eid]
        for plane in ('x', 'y'):
            if f'ftst_{plane}' not in ev or plane not in ev:
                continue
            P = ev[plane]
            if np.asarray(P['W']).shape[1] != wm.NSAMP:
                wm.set_nsamp(np.asarray(P['W']).shape[1])
            g = wm.init_guess(P, plane, ev[f'tan_{plane}'],
                              ev[f'ref_mesh_{plane}'], v)
            r = wm.fit_plane_raw(P, plane, *g, hyper=hyper,
                                 fix_p0w=(ev[f'ref_mesh_{plane}'],
                                          ev[f'tan_{plane}'] * v * 1e-3))
            vals[plane].setdefault(int(ev[f'ftst_{plane}']), []).append(r['t0'])
    pred, spread = {}, {}
    for plane in ('x', 'y'):
        pred[plane], spread[plane] = {}, {}
        for c, ts in sorted(vals[plane].items()):
            if len(ts) < 5:
                continue
            ts = np.asarray(ts, float)
            med = float(np.median(ts))
            sig = float(1.4826 * np.median(np.abs(ts - med)))
            pred[plane][c] = med
            spread[plane][c] = sig
            print(f'[calib] t0_abs {plane} ftst {c}: {med:+.1f} ns '
                  f'(rsig {sig:.1f}, n={len(ts)})')
    return pred, spread


def measure_dt_xy(events, bundle_path, hyper, v, sample=200):
    """t0(x) - t0(y) by ftst difference, for the joint two-plane fit."""
    wm.use_calibration(CalibrationBundle.load(bundle_path))
    diffs = {}
    for eid in sorted(events)[:sample]:
        ev = events[eid]
        if 'ftst_x' not in ev or 'ftst_y' not in ev:
            continue
        t0 = {}
        for plane in ('x', 'y'):
            P = ev[plane]
            # the window length is per event (det4's 6-24 run mixes 32 and 37
            # samples) -- same guard as _event_chi2, or chi2_plane's mask breaks
            if np.asarray(P['W']).shape[1] != wm.NSAMP:
                wm.set_nsamp(np.asarray(P['W']).shape[1])
            g = wm.init_guess(P, plane, ev[f'tan_{plane}'], ev[f'ref_mesh_{plane}'], v)
            r = wm.fit_plane_raw(P, plane, *g, hyper=hyper,
                                 fix_p0w=(ev[f'ref_mesh_{plane}'],
                                          ev[f'tan_{plane}'] * v * 1e-3))
            t0[plane] = r['t0']
        diffs.setdefault(int(ev['ftst_x'] - ev['ftst_y']), []).append(
            t0['x'] - t0['y'])
    out = {}
    for k, vals in diffs.items():
        if len(vals) >= 5:
            out[k] = float(np.median(vals))
            print(f'[calib] ftst diff {k:+d}: t0x - t0y = {out[k]:+.1f} ns '
                  f'(n={len(vals)})')
    return out


# ---------------------------------------------------------------------- main
def calibrate(cfg, run_key, n_events=400, n_train=180, jobs=12, out=None,
              seed_bundle=None, maxiter=130, v_fixed=None, fix_hyper=None,
              share_mode='delay'):
    out = out or cfg.out_dir('wft', 'calib_bundle')
    work = cfg.out_dir('wft', 'calib_work')
    cache_path = os.path.join(work, 'calib_cache.pkl')

    if os.path.exists(cache_path):
        events = pickle.load(open(cache_path, 'rb'))
        print(f'[calib] reusing {len(events):,} cached events')
    else:
        events = build_cache(cfg, n_events, out_path=cache_path)

    grid, tmpl = measure_templates(events)

    # provisional bundle: measured template, seed kernel — the hyper fit refines it
    seed = (CalibrationBundle.load(seed_bundle) if seed_bundle else None)
    cal = CalibrationBundle(
        hyper=dict(zip(HYPER_NAMES, HYPER_X0[:7])) if seed is None else dict(seed.hyper),
        v_drift=float(HYPER_X0[7]) if seed is None else seed.v_drift,
        grid=grid, tmpl=tmpl,
        gain={'x': np.ones(512), 'y': np.ones(512)},
        share_mode=share_mode,
        detector=cfg.DET_NAME, run_key=run_key,
        conditions=dict(run=cfg.RUN, sub_run=cfg.SUB_RUN))
    prov_path = os.path.join(work, 'provisional_bundle')
    cal.save(prov_path, note='templates measured, hypers not yet fitted')

    train = sorted(events)[:n_train]
    print(f'[calib] fitting hypers on {len(train)} events, {jobs} jobs', flush=True)
    x0 = None if seed is None else np.array(
        [seed.hyper[k] for k in HYPER_NAMES] + [seed.v_drift])
    if v_fixed is not None and x0 is not None:
        x0[7] = v_fixed
    # Hypers the seed carries that are NOT fitted: they must ride along in every
    # chi2 evaluation AND survive into the output bundle. Dropping them is
    # silent and severe -- c2_over_c1 is what makes the +-2 copy exist at all,
    # so a refit that loses it writes a bundle whose stored c2 is 0.0 and whose
    # model draws NO +-2 copy, while every printed number looks fine. That is
    # what a v-refit seeded from calib_bundle_r06 did until 2026-08-21.
    extra = {k: float(v) for k, v in (seed.hyper if seed else {}).items()
             if k not in HYPER_NAMES}
    if extra:
        print(f'[calib] carrying unfitted seed hypers: {extra}', flush=True)
    hj = fit_hypers(cache_path, prov_path, train, jobs=jobs, maxiter=maxiter,
                    x0=x0, v_fixed=v_fixed, fixed=fix_hyper, extra_hyper=extra)
    if v_fixed is not None:
        hj['v'] = float(v_fixed)
    cal.hyper = {k: hj[k] for k in HYPER_NAMES}
    cal.hyper.update(extra)
    cal.v_drift = hj['v']
    cal.provenance.update(n_train=hj['n_train'], chi2=hj['chi2'],
                          chi2_init=hj['chi2_init'],
                          fitted='wft.calibrate', gain_map='unit (not fitted)')
    if fix_hyper:
        cal.provenance['fixed_hypers'] = {k: float(v)
                                          for k, v in fix_hyper.items()}
    cal.save(out, note='hypers fitted ref-pinned')

    cal.dt_xy = measure_dt_xy(events, out, cal.hyper, cal.v_drift)
    cal.save(out, note='hypers fitted ref-pinned; dt_xy measured')
    with open(os.path.join(work, 'hyper_fit.json'), 'w') as f:
        json.dump(hj, f, indent=1)
    print('[calib]', cal.summary())
    print('[calib] wrote', out)
    return cal


def main(argv=None):
    ap = argparse.ArgumentParser(prog='wft.calibrate')
    ap.add_argument('run_key')
    ap.add_argument('--events', type=int, default=400)
    ap.add_argument('--train', type=int, default=180)
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--maxiter', type=int, default=130)
    ap.add_argument('--fix-v', type=float, default=None,
                    help='pin the drift velocity (um/ns) instead of fitting it '
                         '-- use the drift-scan value for this field on the '
                         'low-statistics chambers')
    ap.add_argument('--seed-bundle', default=None,
                    help='start from another detector\'s kernel (e.g. det3, same batch)')
    ap.add_argument('--fix-hyper', default=None,
                    help='pin hypers to externally measured values, e.g. '
                         '"c1=0.28,c2=0.11,tau_s=60" (H4 beam kernel)')
    ap.add_argument('--share-mode', default='delay', choices=('delay', 'lp'),
                    help='sharing-kernel form: delay = legacy delayed copy; '
                         'lp = RC-dispersed copy (H4-measured structure, '
                         'tau_s becomes the RC constant)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args(argv)
    fix_hyper = None
    if args.fix_hyper:
        fix_hyper = {k: float(v) for k, v in
                     (kv.split('=') for kv in args.fix_hyper.split(','))}

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (repo, os.path.join(repo, 'mx_june_cosmic_qa'),
              os.path.join(repo, 'cosmic_bench_analysis')):
        if p not in sys.path:
            sys.path.insert(0, p)
    from qa_config import get_config, setup_paths
    setup_paths()
    cfg = get_config(args.run_key)
    calibrate(cfg, args.run_key, n_events=args.events, n_train=args.train,
              jobs=args.jobs, out=args.out, seed_bundle=args.seed_bundle,
              maxiter=args.maxiter, v_fixed=args.fix_v, fix_hyper=fix_hyper,
              share_mode=args.share_mode)


if __name__ == '__main__':
    main()
