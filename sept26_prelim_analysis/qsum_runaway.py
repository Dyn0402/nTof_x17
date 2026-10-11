#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qsum_runaway.py -- what the tracks with q_sum > 1e6 ADC actually are.

STATUS 2026-09-10 found that one gated track in four carries a fitted charge
that cannot be real (`q_total` to 1e34 on a 12-bit ADC) with an unremarkable
chi2, and left it there.  This module finds the mechanism, splits the tail into
the classes it really contains, and measures what each class does to the
GEOMETRY -- which is the question that matters, because nobody uses q_sum for
an opening angle but everybody uses p0, tan and t0.

THE MECHANISM.  `wft.model.chi2_plane` profiles the charge out with an
UNREGULARISED NNLS over a fixed 18 x 60 ns depth grid.  A depth bin whose
design-matrix column is ~0 inside the data window costs nothing to fill, so
NNLS can put any amount of charge there to absorb a residual.  There are two
ways a column becomes ~0:

  * time -- the bin's pulse peaks after the last sample.  The window is
    20 x 60 ns (last sample at 1140 ns) and the grid spans 1080 ns after t0,
    so for any t0 above ~0 the deepest bins are seen only through the first
    few ns of their leading edge (columns 1e-6 .. 1e-17).
  * space -- the fitted slope carries the deep bins' centres a few mm past
    the edge of the strip window, where only the erf tail reaches a strip.

`refit` measures both on real windows and compares the production fit against
a GUARDED fit that drops, inside every NNLS solve, any column whose weighted
norm is below `REL` of the largest -- the same fit with the unobservable
charge simply not offered.  The guard is a monkeypatch of
`wft.model.chi2_plane` in THIS process only: a study instrument, not a
proposed production change.

    python -m sept26_prelim_analysis.qsum_runaway census
    python -m sept26_prelim_analysis.qsum_runaway refit --arms A B C D --n 300
    python -m sept26_prelim_analysis.make_qsum_runaway_figures
    python -m sept26_prelim_analysis.make_qsum_runaway_report
"""
from __future__ import annotations

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import datetime as dt
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402

ARMS = ('A', 'B', 'C', 'D')
BIG_Q = 1e6            # the STATUS 2026-09-10 threshold, ADC
LATE_T0 = 300.0        # ns; above this the census shows the late class
REL = 1e-2             # guard: drop columns below this fraction of the largest
T0_BANDS = [-np.inf, -200, 200, 300, np.inf]
T0_LABELS = ['t0 < -200', '|t0| < 200', '200-300', 't0 > 300']
FLAT_TAN = 0.01

RUN, SUB, TAG = 'run_145', 'stat090_0000', '260805_14H06_000'


def out_dir() -> Path:
    return paths.out('qsum_runaway')


def reco_base() -> Path:
    return paths.out() / 'reco_fullpass' / RUN / SUB


# --------------------------------------------------------------------------- #
# census -- the whole campaign track table, no refitting
# --------------------------------------------------------------------------- #
CENSUS_COLS = ['run', 'arm', 'gated', 'condition', 'coinc_this_arm',
               'x_t0', 'y_t0', 'x_q_sum', 'y_q_sum', 'tan_raw_x', 'tan_raw_y',
               'chi2dof_x', 'chi2dof_y', 'x_n_strips', 'y_n_strips']


def census() -> None:
    import pyarrow.parquet as pq
    src = paths.require(paths.out() / 'stage3_fullpass' / 'tracks_campaign.parquet',
                        'campaign track table')
    od = out_dir()
    d = pq.read_table(src, columns=CENSUS_COLS,
                      filters=[('gated', '=', True)]).to_pandas()
    print(f'[census] {len(d):,} gated tracks')
    rows, hist = [], []
    tb = np.arange(-600, 1201, 20)
    for p in 'xy':
        t0, q, tan = d[f'{p}_t0'], d[f'{p}_q_sum'], d[f'tan_raw_{p}']
        big = q > BIG_Q
        band = pd.cut(t0, T0_BANDS, labels=T0_LABELS)
        g = pd.DataFrame(dict(arm=d.arm, band=band, big=big, coinc=d.coinc_this_arm == 1,
                              abstan=tan.abs(), flat=tan.abs() < FLAT_TAN,
                              chi2dof=d[f'chi2dof_{p}'], nstr=d[f'{p}_n_strips']))
        for (arm, b, bg), s in g.groupby(['arm', 'band', 'big'], observed=True):
            rows.append(dict(plane=p, arm=arm, band=b, big=bool(bg), n=len(s),
                             coinc=s.coinc.mean(), abstan_p50=s.abstan.median(),
                             flat=s.flat.mean(), chi2dof_p50=s.chi2dof.median(),
                             nstr_p50=s.nstr.median()))
        for bg in (False, True):
            h, _ = np.histogram(t0[big == bg], tb)
            hist.append(pd.DataFrame(dict(plane=p, big=bg, t0_lo=tb[:-1], n=h)))
    C = pd.DataFrame(rows)
    C.to_csv(od / 'census.csv', index=False)
    pd.concat(hist).to_csv(od / 'census_t0_hist.csv', index=False)

    late = lambda p: (d[f'{p}_t0'] > LATE_T0) & (d[f'{p}_q_sum'] > BIG_Q)  # noqa: E731
    anyb = (d.x_q_sum > BIG_Q) | (d.y_q_sum > BIG_Q)
    lb = late('x') | late('y')
    co = d.coinc_this_arm == 1
    per_arm = []
    for arm, s in d.groupby('arm'):
        m = d.arm == arm
        per_arm.append(dict(arm=arm, n=int(m.sum()),
                            big_x=float((s.x_q_sum > BIG_Q).mean()),
                            big_y=float((s.y_q_sum > BIG_Q).mean()),
                            big_any=float(anyb[m].mean()),
                            late_big=float(lb[m].mean()),
                            early_big=float((anyb & ~lb)[m].mean()),
                            coinc_n=int((m & co).sum()),
                            coinc_late_big=float(lb[m & co].mean()),
                            coinc_early_big=float((anyb & ~lb)[m & co].mean())))
    pd.DataFrame(per_arm).to_csv(od / 'census_per_arm.csv', index=False)
    meta = dict(src=str(src), n_gated=int(len(d)), big_q=BIG_Q, late_t0=LATE_T0,
                big_any=float(anyb.mean()), late_big=float(lb.mean()),
                early_big=float((anyb & ~lb).mean()),
                coinc_n=int(co.sum()), coinc_late_big=float(lb[co].mean()),
                coinc_early_big=float((anyb & ~lb)[co].mean()),
                by_condition={k: float(v) for k, v in
                              anyb.groupby(d.condition).mean().items()},
                generated=dt.datetime.now().isoformat(timespec='minutes'))
    (od / 'census.meta.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


# --------------------------------------------------------------------------- #
# refit -- production vs guarded, on real windows
# --------------------------------------------------------------------------- #
_CAL = None
_ORIG = None


def _guarded_chi2(plane, W, noise, pos, sat, p0, w, t0, hyper, censor=True,
                  snap_t0=True, t0_prior=None):
    """`wft.model.chi2_plane` with unobservable columns removed from the NNLS."""
    from scipy.optimize import nnls
    from wft import model as wm
    if snap_t0:
        t0 = round(t0 / wm.T0_STEP) * wm.T0_STEP
    M = wm.build_matrix(plane, pos, p0, w, t0, hyper)
    ok = ~sat.reshape(-1)
    if not ok.any():
        return np.inf, None
    Wt = np.repeat(1.0 / noise, wm.NSAMP)
    A = (M * Wt[:, None])[ok]
    y = (W / noise[:, None]).reshape(-1)[ok]
    cn = np.sqrt((A ** 2).sum(0))
    live = cn >= REL * cn.max()
    q = np.zeros(A.shape[1])
    try:
        q[live], rn = nnls(A[:, live], y, maxiter=50 * wm.K)
    except Exception:
        return np.inf, None
    chi = rn * rn
    if censor and sat.any():
        model = (M @ q).reshape(W.shape)
        pen = (np.maximum(0.0, W[sat] - model[sat])
               / np.repeat(noise, wm.NSAMP).reshape(W.shape)[sat])
        chi += float((pen ** 2).sum())
    if t0_prior is not None:
        chi += ((t0 - t0_prior[0]) / t0_prior[1]) ** 2
    return chi, q


def _init(bundle):
    global _CAL, _ORIG
    from wft import model as wm
    from wft.calib import CalibrationBundle
    _CAL = CalibrationBundle.load(bundle)
    wm.use_calibration(_CAL)
    _ORIG = wm.chi2_plane


def _anatomy(f, P, plane, solver=None):
    """Per-depth-bin view of one fit: q, column peak/norm, arrival, position.
    ``solver`` is the chi2 function whose NNLS gives q (production by default)."""
    from wft import model as wm
    W, noise, ps, sat = wm.prep_plane(P, plane)
    M = wm.build_matrix(plane, ps, f.p0, f.w, f.t0, _CAL.hyper)
    _c, q = (solver or _ORIG)(plane, W, noise, ps, sat, f.p0, f.w, f.t0, _CAL.hyper,
                              snap_t0=False)
    Wt = np.repeat(1.0 / noise, wm.NSAMP)
    ok = ~sat.reshape(-1)
    cn = np.sqrt(((M * Wt[:, None])[ok] ** 2).sum(0))
    peak = M.reshape(len(ps), wm.NSAMP, wm.K).max(axis=(0, 1))
    tm, _ = wm._templates(plane, _CAL.hyper['sigma_s'])
    t_peak = float(wm.TGRID[np.argmax(tm)])
    arr = f.t0 + wm.UK
    pk = f.p0 + f.w * wm.UK
    off = np.maximum(ps.min() - pk, pk - ps.max())
    return dict(q=q, cn=cn, peak=peak, arr=arr, pk=pk, off=off, t_peak=t_peak,
                t_last=float(wm.TS[-1]), W=W, noise=noise, pos=ps, sat=sat, M=M)


def _classify(a) -> str:
    q = a['q']
    if q is None or q.sum() <= BIG_Q:
        return 'normal'
    k = int(np.argmax(q))
    if a['arr'][k] + a['t_peak'] > a['t_last']:
        return 'time'
    if a['off'][k] > 0:
        return 'space'
    return 'other'


def _work(payload):
    from wft import model as wm, reco as wr
    eid, wins, _sd, _nh, _spark, _ftst = payload
    out, ex = [], []
    for plane in ('x', 'y'):
        for i, P in enumerate(wins.get(plane, [])):
            wm.chi2_plane = _ORIG
            try:
                f0 = wr.fit_plane(P, plane, _CAL)
            except Exception:
                f0 = None
            if f0 is None:
                continue
            a = _anatomy(f0, P, plane)
            live = a['cn'] >= REL * a['cn'].max()
            wm.chi2_plane = _guarded_chi2
            try:
                f1 = wr.fit_plane(P, plane, _CAL)
            except Exception:
                f1 = None
            finally:
                wm.chi2_plane = _ORIG
            if f1 is None:
                continue
            cls = _classify(a)
            k = int(np.argmax(a['q']))
            d = dict(event_id=int(eid), plane=plane, win=i, cls=cls,
                     q_obs=float(a['q'][live].sum()), n_unobs=int((~live).sum()),
                     q_unobs_share=float(a['q'][~live].sum() / max(a['q'].sum(), 1e-30)),
                     vis=float((a['q'] * a['peak']).sum()),
                     vis_unobs=float((a['q'] * a['peak'])[~live].sum()),
                     kmax=k, arr_kmax=float(a['arr'][k]), off_kmax=float(a['off'][k]),
                     peak_kmax=float(a['peak'][k]), n_strips=f0.n_strips,
                     wmax=float(a['W'].max()))
            for tag, f in (('prod', f0), ('grd', f1)):
                for c in ('p0', 'w', 't0', 'tan_theta', 'chi2', 'dof', 'q_sum',
                          'q_u50', 'q_uend', 'quality_ok'):
                    d[f'{c}_{tag}'] = getattr(f, c)
                d[f'plaus_{tag}'] = wr._candidate_score(P, plane, f)[0]
            out.append(d)
            if cls != 'normal' or len(ex) < 1:
                a1 = _anatomy(f1, P, plane, solver=_guarded_chi2)
                ex.append(dict(meta=d, W=a['W'], noise=a['noise'], pos=a['pos'],
                               sat=a['sat'], q_prod=a['q'], q_grd=a1['q'],
                               model_prod=(a['M'] @ a['q']).reshape(a['W'].shape),
                               model_grd=(a1['M'] @ np.nan_to_num(a1['q'])
                                          ).reshape(a['W'].shape),
                               arr=a['arr'], pk=a['pk'], cn=a['cn'], peak=a['peak'],
                               t_last=a['t_last'], t_peak=a['t_peak']))
    return out, ex


def refit(arms, n: int, jobs: int, seed: int) -> None:
    from ntof_tracking import wft_beam as wb
    from wft import io as wio
    from wft.calib import CalibrationBundle
    od = out_dir()
    res, examples = [], []
    for arm in arms:
        base = reco_base() / f'mx17_{arm}'
        bundle = str(paths.require(base / 'calib_bundle_prelim', f'arm {arm} bundle'))
        cal = CalibrationBundle.load(bundle)
        cfg = wb.beam_config(arm, run=RUN, sub_run=SUB)
        pos = wio.strip_position_map(cfg)
        C = pd.read_parquet(base / f'events_{TAG}.candidates.parquet', columns=['event_id'])
        eids = np.random.default_rng(seed).choice(C.event_id.unique(), n, replace=False)
        hits = wb.read_hits_tag(wb.hits_file_for_tag(cfg, TAG),
                                (cfg.MX17_FEU_X, cfg.MX17_FEU_Y))
        seeds = wb.seeds_from_hits_beam(hits, pos, cfg.MX17_FEU_X, cfg.MX17_FEU_Y,
                                        hot=cal.hot)
        pl = wb._windows_for_tag(cfg, TAG, pos, seeds, set(int(e) for e in eids), 3)
        n0 = len(res)
        with ProcessPoolExecutor(jobs, initializer=_init, initargs=(bundle,)) as ex:
            for o, e in ex.map(_work, pl, chunksize=2):
                for r in o:
                    r['arm'] = arm
                res.extend(o)
                for x in e:
                    x['meta']['arm'] = arm
                examples.extend(e)
        print(f'[refit] {arm}: {len(pl)} events, {len(res) - n0} plane fits', flush=True)
    R = pd.DataFrame(res)
    R.to_parquet(od / 'refit.parquet', index=False)
    # keep a handful per (arm, class) for the displays -- the full set is large
    keep = []
    for (arm, cls), grp in pd.DataFrame(
            [dict(i=i, arm=x['meta']['arm'], cls=x['meta']['cls'],
                  late=x['meta']['t0_prod'] > LATE_T0)
             for i, x in enumerate(examples)]).groupby(['arm', 'cls']):
        keep.extend(grp.i.head(6).tolist())
    with open(od / 'examples.pkl', 'wb') as f:
        pickle.dump([examples[i] for i in keep], f)
    meta = dict(run=RUN, sub_run=SUB, tag=TAG, arms=list(arms), n_per_arm=n,
                seed=seed, rel=REL, big_q=BIG_Q, late_t0=LATE_T0,
                n_fits=int(len(R)),
                generated=dt.datetime.now().isoformat(timespec='minutes'))
    (od / 'refit.meta.json').write_text(json.dumps(meta, indent=1))
    print(f'[refit] wrote {od}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('census')
    r = sp.add_parser('refit')
    r.add_argument('--arms', nargs='+', default=list(ARMS))
    r.add_argument('--n', type=int, default=300, help='events per arm')
    r.add_argument('--jobs', type=int, default=12)
    r.add_argument('--seed', type=int, default=2)
    a = ap.parse_args()
    if a.cmd == 'census':
        census()
    else:
        refit(a.arms, a.n, a.jobs, a.seed)
    return 0


if __name__ == '__main__':
    sys.exit(main())
