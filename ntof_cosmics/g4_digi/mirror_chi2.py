#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mirror_chi2.py -- are the steep-track mirror fits a missed basin or a true
degeneracy of the model?  HANDOFF_TRACKING §15 check 1, open item.

On steep synthetic muons (run_digi.py muons --tan-u 0 1.1, seed 7) 13-40 % of
x fits at |tan| >= 0.3 come out with the wrong sign: |raw| 1.2-4x |true|, half
the q_sum, earlier t0.  Doubling W_SCAN_HALF did not help.  Here each event is
re-digitised identically (run_digi.setup, same seed) and the production x fit
is repeated (same candidate rule, same t0 prior).  Then the chosen window is
refitted with the slope held on the TRUE side:

  scan    (p0, t0) at the true slope, then (p0, w) over the right-sign half
          only, |raw tan| 0.05-1.6, p0 re-centred per slope (shear)
  truth   the true (p0 at the mesh, w, t0)
  each start -> wm.fit_plane_raw (the production Nelder-Mead); the better
  right-sign end point is chi2_right.

dchi2 = chi2_right - chi2_reco.  > 0: the data prefer the mirror (the model
cannot tell; a search change cannot fix it).  < 0: the right basin is better
and the search misses it.  Right-sign reco fits are the control (dchi2 ~ 0).

    PYTHONPATH=. .venv/bin/python ntof_cosmics/g4_digi/mirror_chi2.py --arm A --bundle is2_A
    PYTHONPATH=. .venv/bin/python ntof_cosmics/g4_digi/mirror_chi2.py summary
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ntof_cosmics.g4_digi import digitise as DG  # noqa: E402
from ntof_cosmics.g4_digi import run_digi as RD  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / 'results' / 'repass_readiness'
RAW_LO, RAW_HI, RAW_STEP = 0.05, 1.6, 0.05
#: --two-sided: |raw tan| probed in the (p0, t0) stage, per sign
TS_PROBE = (0.15, 0.4, 0.8)
TWO_SIDED = False


def two_sided_fit(P, plane, cal, f, t0_prior=None):
    """Truth-free candidate fix: search each slope SIGN separately and keep the
    lowest chi2 of {the production fit, + side, - side}.

    Per sign: (p0, t0) at |raw| in TS_PROBE, then (p0, w) over that sign's half,
    |raw| RAW_LO-RAW_HI, at the best t0 (p0 re-centred per slope by the half-
    column shear), then wm.fit_plane_raw (production Nelder-Mead).  The
    production _global_start scans t0 only at w = 0 and w only at that t0, which
    a steep track's charge does not constrain: the measured failure is the
    mirror basin.  Returns dict(raw, chi2, t0, p0, side) or None."""
    from wft import model as wm
    v, kw, w0 = cal.v_drift, cal.kw.get(plane, 1.0), cal.w0.get(plane, 0.0)
    to_w = lambda raw: (raw * kw * v + w0) * 1e-3          # noqa: E731
    to_raw = lambda w: (w * 1e3 - w0) / (kw * v)           # noqa: E731
    hyper = cal.hyper
    Wm, noise, pos, sat = wm.prep_plane(P, plane)

    def chi(p0, w, t0):
        return wm.chi2_plane(plane, Wm, noise, pos, sat, p0, w, t0, hyper, t0_prior=t0_prior)[0]

    shear = 15000.0 / v
    amp = np.maximum(Wm.max(axis=1), 0.0)
    p_c = float((pos * amp).sum() / amp.sum())
    p0s = p_c + np.arange(-6.0, 6.0 + 1e-9, 0.5)
    t0s = (np.array([t0_prior[0]]) if t0_prior else
           np.arange(f.t0 - 240.0, f.t0 + 240.0 + 1e-9, 40.0))
    best = dict(raw=f.tan_theta, chi2=f.chi2, t0=f.t0, p0=f.p0, side='prod')
    for sgn in (1.0, -1.0):
        c1 = min((chi(p - to_w(sgn * r) * shear, to_w(sgn * r), t), t)
                 for t in t0s for p in p0s for r in TS_PROBE)
        t0b = c1[1]
        raws = sgn * np.arange(RAW_LO, RAW_HI + 1e-9, RAW_STEP)
        c2 = min((chi(p - to_w(r) * shear, to_w(r), t0b), p - to_w(r) * shear, to_w(r))
                 for p in p0s for r in raws)
        r = wm.fit_plane_raw(P, plane, c2[1], c2[2], t0b, hyper=hyper, t0_prior=t0_prior)
        if r is not None and np.isfinite(r['chi2']) and r['chi2'] < best['chi2']:
            best = dict(raw=float(to_raw(r['w'])), chi2=float(r['chi2']), t0=float(r['t0']),
                        p0=float(r['p0']), side='+' if sgn > 0 else '-')
    return best


def _x_windows(job):
    """Digitise one event and return (x windows, x seeds, ftst) as reco_event does."""
    from ntof_tracking import wft_beam as WB
    from wft import io as wio
    from wft import reco as wreco
    rng = np.random.default_rng(job['seed'])
    W = DG.digitise_event(job['steps'], job['ov'], job['t0'], job['adc_per_e'],
                          job['xy_split'], job['v_true'], rng, job.get('pos_offset', (0.0, 0.0)))
    st = DG._ST
    eid = job['eid']
    H = pd.concat([DG.emulate_hits(W[p], st['hf_noise'][p], st['feu'][p], eid) for p in 'xy'])
    pm = {st['feu'][p]: st['pos_maps'][p] for p in 'xy'}
    seeds = WB.seeds_from_hits_beam(H, pm, st['feu']['x'], st['feu']['y'],
                                    hot=wreco._CAL.hot, min_strips=st['min_strips'])
    if eid not in seeds:
        return None, None
    ws, us = [], []
    for s in seeds[eid]['x']:
        win = wio.extract_window(W['x'], st['noise']['x'], st['pos_maps']['x'], s.channels, 3)
        if win is not None:
            ws.append(dict(W=win.W, pos=win.pos, noise=win.noise, ch=win.ch))
            us.append(s)
    return ws, us


def mirror_one(job):
    from wft import model as wm
    from wft import reco as wreco
    from ntof_tracking.run145_target_imaging import STRIP_MAP_HALF
    tr = job['truth']
    out = dict(arm=tr['arm'], i=tr['i'], tan_u=tr['tan_u'], u_mesh=tr['u_mesh'], t0_true=job['t0'])
    try:
        ws, us = _x_windows(job)
        if not ws:
            return out
        cal = wreco._CAL
        prior = wreco.t0_prior_for(cal, 'x', job['ov']['ftst_x'])
        # the production candidate rule, keeping the window it picked
        best = None
        for P, s in zip(ws, us):
            f = wreco.fit_plane(P, 'x', cal, n_seed=getattr(s, 'n_strips', 0), n_dropped=getattr(s, 'n_dropped', 0), t0_prior=prior)
            if f is None:
                continue
            plaus, d = wreco._candidate_score(P, 'x', f)
            key = (1 if plaus else 0, d)
            if best is None or key > best[0]:
                best = (key, P, f)
        if best is None:
            return out
        _k, P, f = best
        v, kw, w0 = cal.v_drift, cal.kw.get('x', 1.0), cal.w0.get('x', 0.0)
        to_w = lambda raw: (raw * kw * v + w0) * 1e-3          # noqa: E731
        to_raw = lambda w: (w * 1e3 - w0) / (kw * v)           # noqa: E731
        out.update(n_cand=len(ws), raw_reco=f.tan_theta, chi2_reco=f.chi2, dof=f.dof, t0_reco=f.t0,
                   p0_reco=f.p0, q_reco=f.q_sum, n_strips=f.n_strips)

        if TWO_SIDED:
            out.update({f'ts_{k}': v_ for k, v_ in two_sided_fit(P, 'x', cal, f, prior).items()})
        hyper = cal.hyper
        Wm, noise, pos, sat = wm.prep_plane(P, 'x')

        def chi(p0, w, t0):
            return wm.chi2_plane('x', Wm, noise, pos, sat, p0, w, t0, hyper, t0_prior=prior)[0]

        sgn = np.sign(tr['tan_u'])
        shear = 15000.0 / v
        amp = np.maximum(Wm.max(axis=1), 0.0)
        p_c = float((pos * amp).sum() / amp.sum())
        p0s = p_c + np.arange(-6.0, 6.0 + 1e-9, 0.5)
        w_t = to_w(tr['tan_u'])
        t0s = (np.array([prior[0]]) if prior else
               np.arange(f.t0 - 200.0, f.t0 + 200.0 + 1e-9, 40.0))
        # (p0, t0) at the true slope, p0 shear-centred, then (p0, w) right side only
        c1 = min((chi(p - w_t * shear, w_t, t), p, t) for t in t0s for p in p0s)
        t0b = c1[2]
        raws = sgn * np.arange(RAW_LO, RAW_HI + 1e-9, RAW_STEP)
        c2 = min((chi(p - to_w(r) * shear, to_w(r), t0b), p - to_w(r) * shear, to_w(r))
                 for p in p0s for r in raws)
        starts = dict(scan=(c2[1], c2[2], t0b),
                      truth=(STRIP_MAP_HALF - tr['u_mesh'], w_t, job['t0']))
        out['chi2_truth_point'] = float(chi(*starts['truth']))
        out['chi2_scan_point'] = float(c2[0])
        ends = {}
        for name, (p0i, wi, t0i) in starts.items():
            r = wm.fit_plane_raw(P, 'x', p0i, wi, t0i, hyper=hyper, t0_prior=prior)
            if r is not None and np.isfinite(r['chi2']):
                raw = to_raw(r['w'])
                ends[name] = r
                out[f'chi2_{name}'] = float(r['chi2'])
                out[f'raw_{name}'] = float(raw)
                out[f'q_{name}'] = float(np.nansum(r['q'])) if r.get('q') is not None else np.nan
        right = [(r['chi2'], n) for n, r in ends.items() if np.sign(to_raw(r['w'])) == sgn]
        if right:
            c, n = min(right)
            out.update(chi2_right=float(c), right_from=n, raw_right=float(to_raw(ends[n]['w'])),
                       t0_right=float(ends[n]['t0']), p0_right=float(ends[n]['p0']))
    except Exception as err:                                                  # noqa: BLE001
        out['error'] = repr(err)[:200]
    return out


def run(argv) -> int:
    ap = RD.parser()
    ap.add_argument('--out-label', default=None)
    ap.add_argument('--two-sided', action='store_true', help='also run two_sided_fit (ts_* columns)')
    a = ap.parse_args(['muons'] + argv)
    global TWO_SIDED
    TWO_SIDED = a.two_sided
    a.tan_u = a.tan_u or [0.0, 1.1]
    jobs, bpath, spath, state = RD.setup(a)
    from wft import reco as wreco
    if a.tan_max is not None:
        wreco.TAN_MAX = a.tan_max
    print(f'TAN_MAX {wreco.TAN_MAX} (raw), {len(jobs)} events', flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs, initializer=DG.worker_init,
                             initargs=(str(bpath), state, a.min_strips, spath and str(spath))) as pool:
        for i, r in enumerate(pool.map(mirror_one, jobs, chunksize=2)):
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(f'  {i + 1}/{len(jobs)}', flush=True)
    R = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    lab = a.out_label or f'mirror_chi2_{a.arm}'
    R.to_parquet(OUT / f'{lab}.parquet', index=False)
    print(f'-> {OUT / lab}.parquet ({len(R)} rows)')
    return 0


def summary() -> pd.DataFrame:
    rows = []
    for f in sorted(OUT.glob('mirror_chi2_[AC].parquet')):
        R = pd.read_parquet(f)
        R = R[np.isfinite(R.get('raw_reco', np.nan))]
        steep = R.tan_u.abs() >= 0.3
        mirror = np.sign(R.raw_reco) != np.sign(R.tan_u)
        d = R.chi2_right - R.chi2_reco
        for name, m in (('mirror', steep & mirror), ('right', steep & ~mirror)):
            dm = d[m]
            ok = dm.notna()
            rows.append(dict(arm=R.arm.iat[0], fits=name, n=int(m.sum()), refit_right=int(ok.sum()),
                             dchi2_med=float(dm[ok].median()) if ok.any() else np.nan,
                             frac_mirror_better=float((dm[ok] > 1.0).mean()) if ok.any() else np.nan,
                             frac_right_better=float((dm[ok] < -1.0).mean()) if ok.any() else np.nan,
                             frac_tie=float((dm[ok].abs() <= 1.0).mean()) if ok.any() else np.nan,
                             rel_dchi2_med=float((dm[ok] / R.dof[m][ok]).median()) if ok.any() else np.nan,
                             raw_right_over_true=float((R.raw_right[m][ok] / R.tan_u[m][ok]).median())
                             if ok.any() else np.nan))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'mirror_chi2_summary.csv', index=False)
    with pd.option_context('display.width', 200):
        print(T.round(3).to_string(index=False))
    return T


TS_EDGES = (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.1)


def _mad(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) > 10 else np.nan


def summary_two_sided() -> pd.DataFrame:
    """Production vs two_sided_fit per arm and |true tan| bin (mirror_ts_*)."""
    from ntof_tracking.run145_target_imaging import STRIP_MAP_HALF
    rows = []
    for f in sorted(OUT.glob('mirror_ts_[AC].parquet')):
        R = pd.read_parquet(f)
        R = R[np.isfinite(R.raw_reco) & np.isfinite(R.ts_raw)]
        t = R.tan_u.to_numpy(float)
        p0_true = STRIP_MAP_HALF - R.u_mesh.to_numpy(float)
        for lo, hi in zip(TS_EDGES[:-1], TS_EDGES[1:]):
            b = (np.abs(t) >= lo) & (np.abs(t) < hi)
            row = dict(arm=R.arm.iat[0], lo=lo, hi=hi, n=int(b.sum()),
                       flipped=float((np.sign(R.ts_raw[b]) != np.sign(R.raw_reco[b])).mean()),
                       dchi2_med=float((R.ts_chi2[b] - R.chi2_reco[b]).median()))
            for tag, raw, t0, p0 in (('prod', R.raw_reco, R.t0_reco, R.p0_reco),
                                     ('ts', R.ts_raw, R.ts_t0, R.ts_p0)):
                raw, t0, p0 = raw.to_numpy(float)[b], t0.to_numpy(float)[b], p0.to_numpy(float)[b]
                tb = t[b]
                right = np.sign(raw) == np.sign(tb)
                row[f'{tag}_wrong'] = float((~right).mean()) if lo >= 0.1 else np.nan
                row[f'{tag}_ratio'] = float(np.median(raw[right] / tb[right])) if lo >= 0.1 else np.nan
                row[f'{tag}_res'] = _mad(raw - tb)
                row[f'{tag}_dt0'] = float(np.median(t0 - R.t0_true.to_numpy(float)[b]))
                row[f'{tag}_dp0_along'] = float(np.median((p0 - p0_true[b]) * np.sign(tb)))
                row[f'{tag}_p0_res'] = _mad(p0 - p0_true[b])
            rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'mirror_two_sided_summary.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(T.round(3).to_string(index=False))
    return T


if __name__ == '__main__':
    if sys.argv[1:2] == ['summary']:
        summary()
        summary_two_sided()
        raise SystemExit(0)
    raise SystemExit(run(sys.argv[1:]))
