#!/usr/bin/env python3
"""ladder.py -- FD (model-audit follow-up 10b): why the measured angle resolution does not reach the physics floor.

A realism ladder on the ws_floor toy.  Every event is generated electron by electron (ws_floor/floor_toy.py: Fischle
clusters + delta-ray ranges, Magboltz diffusion, Polya gain, X template, 60 ns sampling, noise, ADC ceiling) and then
one realistic effect after another is switched on IN THE GENERATION ONLY.  Every rung is fitted by the production
reconstruction itself -- ``wft.reco.fit_plane`` with the chamber's promoted calibration bundle (physical RC kernel,
pivot start, the per-event sigma0 choice, the bundle's t0 prior, censoring, dead mask) -- so whatever the rung adds is
something the production model does not describe.  Rung L0 is the floor: the same ideal events fitted by the toy's own
exact mean model, started at the truth (ws_floor's estimator).

Each rung's size is a MEASURED input (sources in REALISM below); nothing is tuned to the measured resolution.

Rungs (cumulative; ``--alone`` puts each one alone on top of L1 instead):
  L0 floor       ideal events, the toy's exact model, start at truth                  (ws_floor)
  L1 prod_fit    ideal events, the production fit (bundle kernel, blind start, sigma0 choice, t0 prior)
  L2 footprint   prompt X footprint: sigma0 0.25 (SPS head-on / per-event value) + pseudo-Voigt tails  (cloud_basics 23, F18)
  L3 xcopy       the one-sided delayed X copy onto the neighbour (cloud_basics 42/43)
  L4 gainmap     strip-to-strip gain non-uniformity, unknown to the fit (cloud_basics 35)
  L5 pulse       pulse-shape mismatch: the true pulse is peakier than the plane template (cloud_basics 37)
  L6 diffexcess  det3's measured extra transverse spread over Magboltz (cloud_basics 34/40); 0 elsewhere
  L7 dead_cm     dead strips the mask misses (cloud_basics 39, ws_dead_charge) + raw common mode removed by the
                 production plain 64-channel median, per-channel CM slope spread (rerun/cm_study)
  L8 vdrift      v changes during the run (gas flush; FOLLOWUPS 1 / ws_v_w0): the fit uses the run-mean v
then outside the toy, in quadrature (measured): the chamber bow (ws_widthtilt angle-map cross-validation) and the
M3 reference (<= 0.15 deg).

    ladder.py run --det det3 [--n 150] [--angles 3,9,16] [--jobs 8] [--alone]   -> results/ladder_<det>[_alone].json
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool

import numpy as np
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'rerun', 'model_audit', 'ws_floor'))
sys.path.insert(0, REPO)
import floor_toy as ft  # noqa: E402

AUDIT = os.environ.get('FD_AUDIT') or os.path.expanduser('~/x17/cosmic_bench/model_audit')
ft.AUDIT = AUDIT
OUT = os.path.join(HERE, 'results')
NBLK = 64            # strips per DREAM chip = per common-mode block
PITCH = ft.PITCH

# ------------------------------------------------------------------ measured realism inputs, per chamber (X view)
REALISM = {
    # sigma0_true: prompt X footprint core [mm]: SPS head-on RAW sigma0_X 0.24-0.29 (F18), per-event physical
    #   0.20-0.25 (cloud_basics 38/42) -> 0.25 on every chamber.
    # lor_eta / lor_gamma: pseudo-Voigt tail fraction and half-width [mm], head-on X profiles
    #   (cloud_basics/results/footprint_test.json, 'mixlor').
    # asym / asym_tau: one-sided delayed copy onto the lower neighbour, fraction of the strip's prompt pulse and delay
    #   [ns]: det3 0.13 / 114 ns (stacked -1 minus +1 pulse, cloud_basics 42), others scaled by their head-on aQ
    #   (det2 0.08, det4 0.08, det7 0.19, det6 -0.02 against det3 0.165); delays det4 105, det7 120 ns.
    # gain_sigma: per-strip ln-gain spread across X (median |d ln g| over 3 mm: det2/3 0.05-0.06, det6/7 0.06-0.11,
    #   det4 0.30; cloud_basics 35), sigma = median/0.95.
    # Dp_excess: extra transverse diffusion in quadrature [mm/sqrt(ns)]: det3 +235 um/sqrt(cm) over Magboltz at
    #   every field (cloud_basics 40) -> sqrt(v) * 0.0235 mm/sqrt(mm); 0 elsewhere (+-100 um/sqrt(cm) systematic only).
    # dead_extra: dead strips the bundle mask misses (cloud_basics 39: det3 39, det2 69/269, det7 ~20 in 67-250;
    #   ws_dead_charge: det6/det7 masks miss 15-18).  det4 over-masks: 0.
    # cm_rms / cm_slope_sigma: raw common mode per 64-ch block [ADC] and the per-channel slope spread
    #   (rerun/cm_study structure_det6_g: 135 ADC, slopes p25-p75 0.977-1.021); one value for the fleet.
    # v_rms: rms of v over the run around the run mean (FOLLOWUPS 1: det4 -10 %, det6 +12 %, det7 +4 % over the
    #   run, det2/3 ~2 %; uniform drift -> range / sqrt(12)).
    'det2': dict(lor_eta=0.11, lor_gamma=0.91, asym=0.063, asym_tau=114., gain_sigma=0.06, Dp_excess=0.0,
                 dead_extra=[69, 269], v_rms=0.02 / 12 ** .5),
    'det3': dict(lor_eta=0.10, lor_gamma=0.83, asym=0.13, asym_tau=114., gain_sigma=0.06, Dp_excess=0.0235,
                 dead_extra=[39], v_rms=0.02 / 12 ** .5),
    'det4': dict(lor_eta=0.27, lor_gamma=0.74, asym=0.063, asym_tau=105., gain_sigma=0.31, Dp_excess=0.0,
                 dead_extra=[], v_rms=0.10 / 12 ** .5),
    'det6': dict(lor_eta=0.22, lor_gamma=0.53, asym=0.0, asym_tau=114., gain_sigma=0.09, Dp_excess=0.0,
                 dead_extra='rand16', v_rms=0.12 / 12 ** .5),
    'det7': dict(lor_eta=0.36, lor_gamma=0.84, asym=0.15, asym_tau=120., gain_sigma=0.09, Dp_excess=0.0,
                 dead_extra='rand18', v_rms=0.04 / 12 ** .5),
}
COMMON = dict(sigma0_true=0.25, cm_rms=135.0, cm_slope_sigma=0.033, pulse_c=None)
# pulse-shape mismatch (cloud_basics 37, det3): residual data - model in units of the model peak, bright unsaturated
# peak strips: +0.061 at the peak, +0.055 one sample later, +0.022 two, -0.029 five.  Generated as the template with
# its falling edge compressed in time by (1 + c) about the peak; c calibrated in pulse_calib() to give +0.06 at the
# peak after the amplitude/time refit.
PULSE_TARGET = 0.06

# extra measured terms applied in quadrature outside the toy [deg]
# bow: ws_widthtilt angle-map cross-validation, X s68 improvement det2 -12, det3 -12, det4 -3, det6 -26, det7 -1 %
BOW_GAIN = dict(det2=0.12, det3=0.12, det4=0.03, det6=0.26, det7=0.01)
M3_DEG = 0.15

RUNGS = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8']
RUNG_NAME = dict(L0='floor (exact model)', L1='production fit', L2='footprint shape', L3='one-sided X copy',
                 L4='strip gain map', L5='pulse-shape mismatch', L6='extra diffusion (det3)',
                 L7='dead strips + common mode', L8='v drift over the run')
# unmeasured candidates (H*): sizes chosen to ask what it would take, not measured; run with --alone only
HYP = dict(tskew_ns=10.0, shape_sigma=0.05, s0_event=(0.20, 0.45))
RUNG_NAME.update(H1='per-channel time skew 10 ns', H2='per-channel pulse width +-5 %',
                 H3='per-event footprint 0.20-0.45 mm', H4='footprint 0.30 mm (FC) + tails')
RUNG_KEYS = dict(H4=('footprint', 's030'), H1=('tskew',), H2=('chshape',), H3=('s0event',), L2=('footprint',), L3=('xcopy',), L4=('gainmap',), L5=('pulse',), L6=('diffx',),
                 L7=('dead', 'cm'), L8=('vdrift',))


def flags_for(rung, alone=False):
    if rung in ('L0', 'L1'):
        return set()
    if alone:
        return set(RUNG_KEYS[rung])
    out = set()
    for r in RUNGS[2:RUNGS.index(rung) + 1]:
        out |= set(RUNG_KEYS[r])
    return out


# ------------------------------------------------------------------ pulse shape
PULSE_MEAS = np.array([0.061, 0.055, 0.022, -0.029])     # cloud_basics 37: peak, +1, +2, +5 samples
PULSE_SHARP_NS = None                                     # width of the sharpening kernel, set by pulse_calib


def distorted_template(g, h, c, s_ns=None):
    """the true pulse: the plane template with its falling edge compressed in time by (1 + c) about the peak.
    s_ns unused (kept for the call signature)."""
    ipk = int(np.argmax(h))
    tpk = g[ipk]
    t = np.where(g > tpk, tpk + (g - tpk) * (1 + c), g)
    return np.interp(t, g, h, left=0, right=h[-1])


def pulse_residual(g, h, c, s_ns=None):
    """an isolated deposit (one depth bin): refit the distorted pulse with the template (amplitude + time shift, 60 ns
    sampling, phase-averaged); residual in units of the model peak at the peak sample and 1, 2, 5 samples later.
    The production depth profile absorbs much of this on a long column (head-on: < 0.1 %), so sizing the distortion
    on the isolated pulse makes this rung an UPPER estimate of what cloud_basics 37's +6 % peak residual can do --
    part of that residual is the footprint (cloud_basics 38), which rung L2 carries."""
    hd = distorted_template(g, h, c)
    res = []
    for ph in np.linspace(0, 60, 6, endpoint=False):
        ts = np.arange(-300, 1500, 60.0) + ph
        y = np.interp(ts, g, hd, left=0, right=0)
        best = None
        for dt in np.arange(-30, 30.1, 1.0):
            m = np.interp(ts - dt, g, h, left=0, right=0)
            a = (m @ y) / (m @ m)
            r = y - a * m
            if best is None or r @ r < best[0]:
                best = (r @ r, r / (a * m.max()), int(np.argmax(m)))
        _, r, k = best
        res.append([r[k], r[min(k + 1, len(r) - 1)], r[min(k + 2, len(r) - 1)], r[min(k + 5, len(r) - 1)]])
    return np.mean(res, axis=0)


def pulse_calib(det):
    """c such that the isolated-pulse refit leaves +0.06 at the peak (cloud_basics 37); returns (c, None, pattern)."""
    g, h = ft.load_template(det)
    cs = np.linspace(0.0, 0.6, 61)
    r0 = np.array([pulse_residual(g, h, c)[0] for c in cs])
    c = float(np.interp(PULSE_MEAS[0], r0, cs))
    return c, None, pulse_residual(g, h, c).tolist()


# ------------------------------------------------------------------ generation
def gen_event(det, tan, flags, adc, tmpl_gen, rng, chamber_state):
    """one X plane: a 64-strip chip block with the track somewhere inside; returns the production window."""
    P = ft.params(det, 'full')
    R = REALISM[det]
    v_true = P['v']
    if 'vdrift' in flags:
        v_true = P['v'] * (1 + rng.normal(0, R['v_rms']))
    Pg = dict(P)
    Pg['v'] = v_true
    if 'diffx' in flags and R['Dp_excess'] > 0:
        Pg['Dp'] = float(np.hypot(P['Dp'], R['Dp_excess'] * np.sqrt(v_true)))
    # the bundle Dp is per sqrt(ns) at the bundle v: D_T = Dp / sqrt(v) is the gas property, Dp_true = D_T sqrt(v_true)
    if 'vdrift' in flags:
        Pg['Dp'] = Pg['Dp'] * np.sqrt(v_true / P['v'])
    g_t, h_t = tmpl_gen
    z, x = ft.electrons(tan, Pg, rng)
    xl, ul = ft.drift(z, x, Pg, rng)
    gain = rng.gamma(1 + ft.THETA, 1.0 / (1 + ft.THETA), len(z))
    t0 = 300.0 + rng.uniform(0, ft.DT_SAMP)
    # place the track in a random 64-strip block of the plane, fully inside it with 6 strips to spare
    xmax = Pg['G'] * tan
    span = int(np.ceil(abs(xmax) / PITCH)) + 12
    blk = int(rng.integers(0, 8))
    s_lo = int(rng.integers(0, max(NBLK - span, 1)))
    x_off = (s_lo + 6 + (0 if xmax >= 0 else span - 12)) * PITCH + rng.uniform(-PITCH / 2, PITCH / 2)
    xl = xl + x_off
    pos = np.arange(NBLK) * PITCH
    sig0 = COMMON['sigma0_true'] if 'footprint' in flags else P['s0']
    if 's030' in flags:
        sig0 = 0.30            # FC (cef45df): one sigma0 = 0.30 mm for both views, mostly resistive spread < 60 ns
    if 's0event' in flags:
        sig0 = rng.uniform(*HYP['s0_event'])
    F = ft.footprint(pos, xl, sig0)
    if 'footprint' in flags:
        eta, gam = R['lor_eta'], R['lor_gamma']
        a_hi = np.arctan((pos[:, None] + PITCH / 2 - xl[None, :]) / gam)
        a_lo = np.arctan((pos[:, None] - PITCH / 2 - xl[None, :]) / gam)
        F = (1 - eta) * F + eta * (a_hi - a_lo) / np.pi
    A = F * (gain * adc)[None, :]
    ts = np.arange(ft.NSAMP) * ft.DT_SAMP
    H = np.interp(ts[None, :] - (t0 + ul)[:, None], g_t, h_t, left=0, right=0)
    if 'tskew' in flags or 'chshape' in flags:
        W = np.empty((NBLK, ft.NSAMP))
        tt = ts[None, :] - (t0 + ul)[:, None]
        for s_ in range(NBLK):
            dt_s = chamber_state['tskew'][s_ + NBLK * blk] if 'tskew' in flags else 0.0
            sc_s = chamber_state['chshape'][s_ + NBLK * blk] if 'chshape' in flags else 1.0
            ipk = g_t[int(np.argmax(h_t))]
            W[s_] = A[s_] @ np.interp(ipk + (tt - dt_s - ipk) / sc_s, g_t, h_t, left=0, right=0)
    else:
        W = A @ H
    if 'xcopy' in flags and R['asym'] > 0:
        Hd = np.interp(ts[None, :] - (t0 + ul + R['asym_tau'])[:, None], g_t, h_t, left=0, right=0)
        Wd = A @ Hd
        W[:-1] += R['asym'] * Wd[1:]          # each strip's prompt pulse copied onto its lower neighbour, late
    gstrip = np.arange(NBLK) + NBLK * blk
    if 'gainmap' in flags:
        W *= chamber_state['gain'][gstrip][:, None]
    dead_true = np.isin(gstrip, chamber_state['dead_true']) if 'dead' in flags else np.zeros(NBLK, bool)
    W[dead_true] = 0.0
    noise = np.full(NBLK, P['noise'])
    W = W + rng.normal(0, 1, W.shape) * noise[:, None]
    if 'cm' in flags:
        cm = rng.normal(0, COMMON['cm_rms'], ft.NSAMP)
        slope = chamber_state['cm_slope'][gstrip]
        raw = W + slope[:, None] * cm[None, :]
        sat = raw >= ft.SAT
        raw = np.minimum(raw, ft.SAT)
        W = raw - np.median(raw, axis=0)[None, :]     # the production plain block median (wft.io CM_MASK = 0)
        W[sat] = ft.SAT
    else:
        sat = W >= ft.SAT
        W = np.minimum(W, ft.SAT)
    # production window: strips over 5 sigma (the hits), dead strips bridged, padded by 3 strips
    over = np.flatnonzero(W.max(axis=1) > 5 * noise)
    if not len(over):
        return None
    lo, hi = max(over.min() - 3, 0), min(over.max() + 3, NBLK - 1)
    sl = slice(lo, hi + 1)
    win = dict(W=W[sl], pos=pos[sl], noise=noise[sl], ch=gstrip[sl])
    truth = dict(tan=tan, t0=t0, p0=x_off, ne=len(z), v_true=v_true, sat_event=bool(sat[sl].any()),
                 sat_frac=float(sat[sl].mean()), n_strips=hi - lo + 1)
    return win, truth


# ------------------------------------------------------------------ workers
_CAL = None
_STATE = None


def _init(det):
    global _CAL, _STATE
    from wft.calib import CalibrationBundle
    from wft import model as wm
    _CAL = CalibrationBundle.load(os.path.join(AUDIT, det, 'calib_bundle'))
    wm.use_calibration(_CAL)
    rs = np.random.default_rng(777)
    R = REALISM[det]
    gain = np.exp(rs.normal(0, R['gain_sigma'], 512))
    gain /= np.exp(np.mean(np.log(gain)))
    dead = list(_CAL.dead.get('x', []))
    ex = R['dead_extra']
    if isinstance(ex, str):
        ex = list(rs.choice(np.setdiff1d(np.arange(512), dead), int(ex[4:]), replace=False))
    _STATE = dict(gain=gain, dead_true=np.array(sorted(set(dead) | set(int(e) for e in ex))),
                  cm_slope=1 + rs.normal(0, COMMON['cm_slope_sigma'], 512),
                  tskew=rs.normal(0, HYP['tskew_ns'], 512), chshape=1 + rs.normal(0, HYP['shape_sigma'], 512))


def one(args):
    det, rung, alone, ang, seed, n, pulse_c = args
    if _CAL is None:
        _init(det)
    from wft import reco
    flags = flags_for(rung, alone)
    rng = np.random.default_rng(seed)
    g, h = ft.load_template(det)
    tmpl_gen = (g, distorted_template(g, h, pulse_c)) if 'pulse' in flags else (g, h)
    adc = ft.adc_scale(ft.params(det, 'full'), np.random.default_rng(12345))
    tan = np.tan(np.radians(ang))
    P = ft.params(det, 'full')
    v_um = float(_CAL.v_drift)
    rows = []
    for i in range(n):
        ev = gen_event(det, tan, flags, adc, tmpl_gen, rng, _STATE)
        if ev is None:
            continue
        win, tr = ev
        sat = np.asarray(win['W']) >= ft.SAT
        if rung == 'L0':
            par, c2, _ = ft.fit_event(win['W'], sat, win['pos'], P, (g, h), dict(p0=tr['p0'], tan=tan, t0=tr['t0']))
            tfit, dof, chi2, s0u, t0f = par[1], int((~sat).sum()), c2, P['s0'], par[2]
        else:
            tp = reco.t0_prior_for(_CAL, 'x', 0)
            tp = (tr['t0'], tp[1]) if tp is not None else None
            f = reco.fit_plane(win, 'x', _CAL, t0_prior=tp)
            if f is None:
                rows.append((ang, np.nan, np.nan, 0, np.nan, np.nan, tr['sat_event'], tr['n_strips'], tr['ne']))
                continue
            tfit = f.w * 1e3 / v_um
            dof, chi2, s0u, t0f = f.dof, f.chi2, f._sigma_p0, f.t0
        rows.append((ang, np.degrees(np.arctan(tfit)) - ang, chi2 / max(dof, 1), dof, s0u, t0f - tr['t0'],
                     tr['sat_event'], tr['n_strips'], tr['ne']))
    return rows


def s68(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    a = a[np.abs(a) <= 5]
    return float(np.percentile(np.abs(a - np.median(a)), 68)) if len(a) > 5 else np.nan


def boot_err(a, nb=200):
    rs = np.random.default_rng(1)
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if len(a) < 10:
        return np.nan
    return float(np.std([s68(rs.choice(a, len(a))) for _ in range(nb)]))


def summarise(rows):
    rows = np.array(rows, float)
    out = {}
    for ang in sorted(set(rows[:, 0])):
        r = rows[rows[:, 0] == ang]
        d = r[:, 1]
        out[str(int(ang))] = dict(n=len(r), s68=s68(d), s68_err=boot_err(d), bias=float(np.nanmedian(d)),
                                  far5=float(np.mean(~(np.abs(d) <= 5))), chi2dof=float(np.nanmedian(r[:, 2])),
                                  chi2dof_p25=float(np.nanpercentile(r[:, 2], 25)),
                                  chi2dof_p75=float(np.nanpercentile(r[:, 2], 75)),
                                  wide_pick=float(np.nanmean(r[:, 4] > 0.3)), t0_s68=s68(r[:, 5] / 100) * 100,
                                  sat_event=float(np.mean(r[:, 6])), n_strips=float(np.median(r[:, 7])))
    d = rows[:, 1]
    out['all'] = dict(n=len(rows), s68=s68(d), s68_err=boot_err(d), chi2dof=float(np.nanmedian(rows[:, 2])),
                      sat_event=float(np.mean(rows[:, 6])), n_strips=float(np.median(rows[:, 7])))
    return out


def run(a):
    os.makedirs(OUT, exist_ok=True)
    angles = [float(x) for x in a.angles.split(',')]
    rungs = a.rungs.split(',') if a.rungs else RUNGS
    pulse_c, _, pulse_res = pulse_calib(a.det)
    res = dict(det=a.det, alone=a.alone, angles=angles, realism=REALISM[a.det], common=COMMON,
               pulse_c=pulse_c, pulse_residual=pulse_res, bow_gain=BOW_GAIN[a.det], m3_deg=M3_DEG, rungs={})
    path = os.path.join(OUT, f'ladder_{a.det}' + ('_alone' if a.alone else '') + '.json')
    if os.path.exists(path) and a.resume:
        old = json.load(open(path))
        res['rungs'] = old.get('rungs', {})
    chunk = 10
    with Pool(a.jobs, initializer=_init, initargs=(a.det,)) as pool:
        for rung in rungs:
            if rung in res['rungs'] and a.resume:
                continue
            if a.alone and rung in ('L0',):
                continue
            t = time.time()
            jobs = [(a.det, rung, a.alone, ang, 1000 * k + c + 7, chunk, pulse_c)
                    for k, ang in enumerate(angles) for c in range(a.n // chunk)]
            rows = [r for part in pool.map(one, jobs, chunksize=1) for r in part]
            np.save(path.replace('.json', f'_{rung}.npy'), np.array(rows, float))
            res['rungs'][rung] = summarise(rows)
            res['rungs'][rung]['seconds'] = time.time() - t
            json.dump(res, open(path, 'w'), indent=1)
            s = res['rungs'][rung]
            print(f"{a.det} {rung:3s} {RUNG_NAME[rung]:28s} " + ' '.join(
                f"{k}: {v['s68']:.2f}/{v['chi2dof']:.0f}" for k, v in s.items() if isinstance(v, dict)) +
                  f"  {time.time() - t:.0f} s", flush=True)


def job(a):
    """one chamber, rung(s) and angle(s) -> <out>/<det>_<rung>[_alone]_a<ang>.npy (condor: one of each; locally a
    comma list of rungs and angles runs on a pool of --jobs workers)"""
    pulse_c, _, _ = pulse_calib(a.det)
    os.makedirs(a.out, exist_ok=True)
    for rung in a.rungs.split(','):
        for ang in [float(x) for x in a.angles.split(',')]:
            jobs = [(a.det, rung, a.alone, ang, 1000 * int(ang) + c + 7, 10, pulse_c) for c in range(a.n // 10)]
            if a.jobs > 1:
                with Pool(a.jobs, initializer=_init, initargs=(a.det,)) as pool:
                    rows = [r for part in pool.map(one, jobs, chunksize=1) for r in part]
            else:
                _init(a.det)
                rows = [r for j in jobs for r in one(j)]
            np.save(os.path.join(a.out, f"{a.det}_{rung}{'_alone' if a.alone else ''}_a{int(ang)}.npy"),
                    np.array(rows, float))
            print(a.det, rung, ang, 'done', flush=True)


def collect(a):
    import glob
    for det in REALISM:
        for alone in (False, True):
            res = dict(det=det, alone=alone, realism=REALISM[det], common=COMMON, bow_gain=BOW_GAIN[det],
                       m3_deg=M3_DEG, rungs={})
            pc, _, pr = pulse_calib(det)
            res['pulse_c'], res['pulse_residual'] = pc, pr
            for rung in RUNGS + ['H1', 'H2', 'H3', 'H4']:
                fs = sorted(glob.glob(os.path.join(a.inp, f"{det}_{rung}{'_alone' if alone else ''}_a*.npy")))
                if not fs:
                    continue
                rows = np.concatenate([np.load(f) for f in fs])
                res['rungs'][rung] = summarise(rows)
            if res['rungs']:
                path = os.path.join(OUT, f"ladder_{det}{'_alone' if alone else ''}.json")
                json.dump(res, open(path, 'w'), indent=1)
                for rung, s in res['rungs'].items():
                    print(f"{det} {'alone' if alone else 'cum  '} {rung:3s} {RUNG_NAME[rung]:28s} " + ' '.join(
                        f"{k}: {v['s68']:.2f}/{v['chi2dof']:.0f}" for k, v in s.items() if isinstance(v, dict)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('run', 'pulse', 'job', 'collect'))
    ap.add_argument('--det', default='det3')
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--angles', default='3,9,16')
    ap.add_argument('--rungs', default='')
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--alone', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--out', default=os.path.join(OUT, 'ladder_raw'))
    ap.add_argument('--inp', default=os.path.join(OUT, 'ladder_raw'))
    a = ap.parse_args()
    if a.cmd == 'pulse':
        for d in REALISM:
            print(d, pulse_calib(d))
        return
    {'run': run, 'job': job, 'collect': collect}[a.cmd](a)


if __name__ == '__main__':
    main()
