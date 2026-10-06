#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
insitu_calib.py -- calibrate and test the waveform fit IN SITU at n_TOF, with
the line through chambers A and C playing the role the M3 telescope played on
the bench.  HANDOFF_TRACKING_2026-10-06.md §7-§8.

WHY.  The n_TOF bundles are bench transfers (template + sharing kernel + w0/kw)
with v swapped for the 42.6 um/ns Magboltz prior (`wft_beam.make_bundle`).  On
the bench, v was fitted TOGETHER with the kernel -- the two trade off along a
valley that keeps angles right only as a pair (`ANALYSIS_STATE_2026-07-31`
S7/S8) -- so replacing one without the other breaks the angle scale.  PLAN_08
§6 (the in-situ calibration) was never done; k_arm has been patching it with a
single factor.  This package does §6 with an external truth.

THE TRUTH.  A clean A-C through-goer (one gated track per arm, the two lines
within 60 mm) is one straight line through both mesh-plane crossings, 469 mm
apart.  Its slope in each chamber's local frame is the reference tan; the
chamber's own fitted p0 is the reference mesh position (positions do not depend
on v or k).  The local sign convention is the reconstruction's own: chosen so
that the fitted tan correlates positively with the truth, per arm and plane.

STEPS (each writes under --work, a scratch dir off /media):

  truth    clean A-C events of the fetched sub-runs -> truth.parquet
  cache    waveform windows along each truth corridor, per arm, in the format
           `wft.calibrate.fit_hypers` reads -> cache_<arm>.pkl
  profile  chi2 vs w for every plane, p0 and t0 profiled, under a bundle:
           does the model's minimum sit at the truth?
  hyper    ref-pinned hyper fit (wft.calibrate.fit_hypers) on the train half
  reco     the PRODUCTION driver (wft_beam.reconstruct_subrun) on an allowlist
           of the truth events, under any bundle -> reco_<label>_<arm>.parquet
  score    response, offset, head-on and resolution of a reco against truth

Waveforms: `WFT_BEAM_BASE=<work>/beam/` laid out as <run>/<sub>/decoded_root
and combined_hits_root, with <run>/run_config.json (see fetch in the handoff).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import cosmic_tracks as CT  # noqa: E402

RUN = 'run_149'
K_REF = 'run_150'          # any borrowed k: positions and raw tans do not depend on it
SEP_MAX = 60.0
ARMS_AC = ('A', 'C')
FEUS = {'A': (3, 4), 'C': (7, 8)}
PAD_MM = 5.0
Z_LO, Z_HI = -3.0, 33.0
#: production bundles, one copy per sub-run (identical across sub-runs)
PROD_BUNDLE = CT.OUT / 'reco' / RUN / 'cosbounce_cos_0000' / 'mx17_{arm}' / 'calib_bundle_prelim'


def _guard(p) -> Path:
    p = Path(p)
    if str(p.resolve()).startswith('/media/'):
        sys.exit(f'FATAL: {p} is on /media')
    return p


# --------------------------------------------------------------------------- #
def truth(work: Path, subs: list[str]) -> pd.DataFrame:
    rows = []
    d = CT.OUT / f'k_{K_REF}'
    for sub in subs:
        P = pd.read_parquet(d / f'pairs_{RUN}_{sub}.parquet')
        t = pd.read_parquet(CT.tracks_path(RUN, sub, K_REF))
        ng = t[t.gated].groupby(['event_id', 'arm']).size().unstack(fill_value=0)
        c = P[(P.pair == 'AC') & (P.sep_mm < SEP_MAX)]
        c = c[c.event_id.map(lambda e: ng.loc[e, 'A'] == 1 and ng.loc[e, 'C'] == 1)]
        ti = t.set_index(['event_id', 'arm', 'track_id'])
        a = ti.loc[list(zip(c.event_id, c.arm1, c.track1))].reset_index()
        b = ti.loc[list(zip(c.event_id, c.arm2, c.track2))].reset_index()
        Jz = b.p0_z.to_numpy() - a.p0_z.to_numpy()
        jx = (b.p0_x.to_numpy() - a.p0_x.to_numpy()) / Jz
        jy = (b.p0_y.to_numpy() - a.p0_y.to_numpy()) / Jz
        for arm, tr in (('A', a), ('C', b)):
            rows.append(pd.DataFrame(dict(
                subrun=sub, tag=tr.tag.to_numpy(), event_id=tr.event_id.to_numpy(), arm=arm,
                sep_mm=c.sep_mm.to_numpy(), jx=jx, jy=jy,
                ref_mesh_x=tr.x_p0.to_numpy(), ref_mesh_y=tr.y_p0.to_numpy(),
                prod_tan_x=tr.x_tan_theta.to_numpy(), prod_tan_y=tr.y_tan_theta.to_numpy(),
                prod_w_x=tr.x_w.to_numpy(), prod_w_y=tr.y_w.to_numpy(),
                prod_t0_x=tr.x_t0.to_numpy(), prod_t0_y=tr.y_t0.to_numpy())))
    T = pd.concat(rows, ignore_index=True)
    # the local sign of each (arm, plane): the reconstruction's own convention
    sign = {}
    for arm in ARMS_AC:
        for ax in 'xy':
            g = T[(T.arm == arm) & (T[f'j{ax}'].abs() > 0.1)]
            r = np.corrcoef(g[f'prod_tan_{ax}'], g[f'j{ax}'])[0, 1]
            if abs(r) < 0.5:
                raise RuntimeError(f'{arm}{ax}: |corr(tan, j)| = {abs(r):.2f}, no clear sign')
            sign[f'{arm}{ax}'] = float(np.sign(r))
    for ax in 'xy':
        T[f'tan_{ax}'] = T[f'j{ax}'] * T.arm.map(lambda a: sign[f'{a}{ax}'])
    # deterministic train/test split by event, shared by both arms
    import zlib
    T['train'] = (T.event_id * 2654435761 + T.subrun.map(lambda x: zlib.crc32(x.encode()))) % 3 == 0
    T.attrs['sign'] = sign
    T.to_parquet(work / 'truth.parquet', index=False)
    (work / 'truth_sign.json').write_text(json.dumps(sign))
    print(f'{T.event_id.nunique():,} events x 2 arms; train {int(T.train.sum() / 2)}; sign {sign}')
    return T


# --------------------------------------------------------------------------- #
def _cfg(arm, sub):
    from ntof_tracking import wft_beam as WB
    return WB.beam_config(arm, RUN, sub)


def cache(work: Path, arm: str) -> dict:
    from wft import io as wio
    T = pd.read_parquet(work / 'truth.parquet')
    T = T[T.arm == arm]
    events = {}
    for sub, g in T.groupby('subrun'):
        cfg = _cfg(arm, sub)
        pos_maps = wio.strip_position_map(cfg)
        ev = {int(r.event_id): dict(eid=int(r.event_id), subrun=sub, tag=r.tag,
                                    tan_x=float(r.tan_x), tan_y=float(r.tan_y),
                                    ref_mesh_x=float(r.ref_mesh_x),
                                    ref_mesh_y=float(r.ref_mesh_y), train=bool(r.train))
              for r in g.itertuples()}
        for plane, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
            pm = pos_maps[feu]
            for f in wio.subrun_files(cfg.BASE_PATH, RUN, sub, feu):
                rdr = wio.FeuReader(f)
                want = set(ev) & set(rdr.event_ids.tolist())
                for eid, ftst, wfm in rdr.iter_events(want):
                    e = ev[eid]
                    p0, tn = e[f'ref_mesh_{plane}'], e[f'tan_{plane}']
                    a, b = p0 + Z_LO * tn, p0 + Z_HI * tn
                    lo, hi = min(a, b) - PAD_MM, max(a, b) + PAD_MM
                    ch = np.where((pm >= lo) & (pm <= hi))[0]
                    ch = ch[np.argsort(pm[ch])]
                    if len(ch) < 4:
                        continue
                    e[plane] = dict(ch=ch.astype(np.int16), pos=pm[ch].astype(np.float32),
                                    W=wfm[ch].astype(np.float32),
                                    noise=np.maximum(rdr.noise[ch], 3.0).astype(np.float32))
                    e[f'ftst_{plane}'] = ftst
        # sub-run-unique key: event_id repeats across sub-runs
        for eid, e in ev.items():
            if 'x' in e and 'y' in e:
                events[f'{sub}:{eid}'] = e
        print(f'  {arm} {sub}: {sum(1 for e in ev.values() if "x" in e and "y" in e)} events')
    with open(work / f'cache_{arm}.pkl', 'wb') as f:
        pickle.dump(events, f, protocol=4)
    print(f'{arm}: {len(events)} events cached')
    return events


# --------------------------------------------------------------------------- #
W_PROFILE = np.arange(-0.024, 0.02401, 0.0005)       # mm/ns


def _profile_one(payload):
    from wft import model as wm
    key, ev, bundle = payload
    out = []
    for plane in ('x', 'y'):
        P = ev[plane]
        W = np.asarray(P['W'])
        if W.shape[1] != wm.NSAMP:
            wm.set_nsamp(W.shape[1])
        Wp, noise, pos, sat = wm.prep_plane(P, plane)
        p0r = ev[f'ref_mesh_{plane}']
        prof = []
        for w in W_PROFILE:
            best = np.inf
            for dp in (-0.4, 0.0, 0.4):
                for t0 in np.arange(-200.0, 401.0, 20.0):
                    c = wm.chi2_plane(plane, Wp, noise, pos, sat, p0r + dp, w, t0,
                                      wm.HYPER)[0]
                    best = min(best, c)
            prof.append(best)
        out.append(dict(key=key, plane=plane, tan_true=ev[f'tan_{plane}'],
                        chi2=np.asarray(prof, float), dof=int((~sat).sum())))
    return out


def profile(work: Path, arm: str, bundle: str, label: str, n: int, jobs: int):
    from concurrent.futures import ProcessPoolExecutor
    from wft import reco as wreco
    with open(work / f'cache_{arm}.pkl', 'rb') as f:
        E = pickle.load(f)
    keys = sorted(E, key=lambda k: abs(E[k]['tan_x']))   # head-on first
    # an even spread in |tan_x|: every k-th after sorting
    if n and n < len(keys):
        keys = keys[::max(1, len(keys) // n)][:n]
    rows = []
    with ProcessPoolExecutor(max_workers=jobs, initializer=wreco._worker_init,
                             initargs=(bundle,)) as pool:
        for r in pool.map(_profile_one, [(k, E[k], bundle) for k in keys], chunksize=2):
            rows.extend(r)
    D = pd.DataFrame(rows)
    D['w_grid'] = [W_PROFILE] * len(D)
    D.to_pickle(work / f'profile_{label}_{arm}.pkl')
    return D


# --------------------------------------------------------------------------- #
# The ref-pinned hyper fit, n_TOF framing.  `wft.calibrate.fit_hypers` cannot
# be used as is: its per-event t0 search is hard-wired to 150-900 ns (the
# bench framing, prompt at sample ~7), and these cosmics have t0 ~ -40 ns.
# Same model, same chi2; w pinned to tan_true * v, p0 and t0 profiled per
# event (p0 is not independently known here: the truth line passes through the
# chambers' own p0, so it is a nuisance parameter, not a pin).
FIT_HYPERS = ('c1', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')
_HEV = None


def _hinit(cache_path, bundle_path, keys):
    global _HEV
    from wft import model as wm
    from wft.calib import CalibrationBundle
    with open(cache_path, 'rb') as f:
        E = pickle.load(f)
    _HEV = {k: E[k] for k in keys}
    wm.use_calibration(CalibrationBundle.load(bundle_path))


def _hchi(payload):
    from wft import model as wm
    key, hyper, v, warm = payload
    ev = _HEV[key]
    tot, best = 0.0, {}
    for plane in ('x', 'y'):
        P = ev[plane]
        W = np.asarray(P['W'])
        if W.shape[1] != wm.NSAMP:
            wm.set_nsamp(W.shape[1])
        Wp, noise, pos, sat = wm.prep_plane(P, plane)
        w = ev[f'tan_{plane}'] * v * 1e-3
        p0r = ev[f'ref_mesh_{plane}']
        st = warm.get(plane)
        if st is None:
            t0s, p0s = np.arange(-250.0, 451.0, 30.0), p0r + np.arange(-0.6, 0.61, 0.3)
        else:
            t0s = st[1] + np.arange(-30.0, 31.0, 10.0)
            p0s = st[0] + np.arange(-0.2, 0.21, 0.1)
        b = (np.inf, p0r, 0.0)
        for p0 in p0s:
            for t0 in t0s:
                c = wm.chi2_plane(plane, Wp, noise, pos, sat, p0, w, t0, hyper)[0]
                if c < b[0]:
                    b = (c, float(p0), float(t0))
        if np.isfinite(b[0]):
            tot += b[0]
            best[plane] = (b[1], b[2])
    return key, tot, best


def hyper_fit(work: Path, arm: str, bundle: str, label: str, jobs: int,
              maxiter: int = 250, ratio: float = 0.6, fit_v: bool = True):
    from concurrent.futures import ProcessPoolExecutor
    from scipy.optimize import minimize
    from wft.calib import CalibrationBundle
    cache_path = work / f'cache_{arm}.pkl'
    with open(cache_path, 'rb') as f:
        E = pickle.load(f)
    keys = sorted(k for k, e in E.items() if e['train'])
    seed = CalibrationBundle.load(bundle)
    h0 = dict(seed.hyper)
    h0['c2_over_c1'] = ratio
    v0 = 36.6 if arm == 'A' else 30.0
    x0 = np.array([h0[k] for k in FIT_HYPERS] + [v0])
    scale = np.array([0.03, 0.6, 40.0, 20.0, 0.1, 0.005, 3.0])
    warm = {k: {} for k in keys}
    log = []
    with ProcessPoolExecutor(max_workers=jobs, initializer=_hinit,
                             initargs=(str(cache_path), bundle, keys)) as pool:
        def total(x):
            hyper = dict(h0)
            hyper.update(dict(zip(FIT_HYPERS, x[:6])))
            v = x[6] if fit_v else v0
            c = 0.0
            for key, tot, best in pool.map(_hchi, [(k, hyper, v, warm[k]) for k in keys],
                                           chunksize=4):
                c += tot
                warm[key] = best
            return c
        c0 = total(x0)
        print(f'[hyper {arm}] {len(keys)} train events, initial chi2 {c0:.5e}', flush=True)

        def obj(x):
            if x[0] < 0.01 or x[1] < 0 or x[2] < 5 or x[3] < 0 or x[4] < 0.02 \
                    or x[5] < 0 or not (10 < x[6] < 60):
                return 2 * c0
            c = total(x)
            log.append(list(map(float, x)) + [c])
            if len(log) % 10 == 0:
                print(f'[hyper {arm}] eval {len(log)} {np.round(x, 4)} {c:.5e}', flush=True)
            return c
        sim = np.array([x0] + [x0 + np.eye(7)[j] * scale[j] for j in range(7)])
        res = minimize(obj, x0, method='Nelder-Mead',
                       options=dict(initial_simplex=sim, xatol=1e-3, fatol=c0 * 2e-5,
                                    maxiter=maxiter))
    x = res.x
    cal = CalibrationBundle.load(bundle)
    cal.hyper = dict(h0)
    cal.hyper.update(dict(zip(FIT_HYPERS, map(float, x[:6]))))
    cal.v_drift = float(x[6])
    # w0/kw are re-measured post hoc from free fits; start from the identity
    cal.w0, cal.kw = {}, {}
    cal.provenance = dict(cal.provenance)
    cal.provenance.update(
        fitted='ntof_cosmics/insitu_calib.py hyper (ref-pinned to the A-C cosmic line)',
        insitu_train=len(keys), chi2=float(res.fun), chi2_init=float(c0),
        seeded_from_prod=bundle, c2_over_c1=ratio,
        status='IN-SITU CANDIDATE -- run_149 cosmics; not yet validated on beam')
    out = _guard(work / 'bundles' / f'{label}_{arm}')
    cal.save(str(out), note=f'in-situ ref-pinned fit, {label}')
    (work / f'hyper_{label}_{arm}.json').write_text(json.dumps(dict(
        x=list(map(float, x)), names=list(FIT_HYPERS) + ['v'], chi2=float(res.fun),
        chi2_init=float(c0), nit=int(res.nit), log=log), indent=1))
    print(f'[hyper {arm}] done: {dict(zip(list(FIT_HYPERS) + ["v"], np.round(x, 4)))} '
          f'chi2 {c0:.4e} -> {res.fun:.4e}')
    return str(out)


def set_w0kw(work: Path, arm: str, label: str, bundle: str) -> dict:
    """Post-hoc angle constants from the TRAIN half's free fits: tan = (w*1e3 -
    w0)/(kw*v), the bench's set_w0 step with the cosmic line as reference."""
    from wft.calib import CalibrationBundle
    T = pd.read_parquet(work / 'truth.parquet')
    T = T[(T.arm == arm) & T.train]
    R = pd.read_parquet(work / f'reco_{label}_{arm}.parquet')
    M = T.merge(R, on=['subrun', 'event_id'])
    cal = CalibrationBundle.load(bundle)
    w0, kw = {}, {}
    for ax in 'xy':
        t, w = M[f'tan_{ax}'].to_numpy(), M[f'{ax}_w'].to_numpy() * 1e3
        m = np.isfinite(w) & (np.abs(t) > 0.1) & (np.abs(t) < 0.5)
        b, a = np.polyfit(t[m], w[m], 1)
        w0[ax], kw[ax] = float(a), float(b / cal.v_drift)
    cal.w0, cal.kw = w0, kw
    cal.save(bundle, note=f'w0/kw from {label} train free fits')
    print(f'[w0kw {arm}] w0 {w0} kw {kw}')
    return dict(w0=w0, kw=kw)


# --------------------------------------------------------------------------- #
def reco(work: Path, arm: str, bundle: str, label: str, jobs: int, split: str = 'all'):
    """The production driver on the truth events, under `bundle`."""
    from ntof_tracking import wft_beam as WB
    T = pd.read_parquet(work / 'truth.parquet')
    T = T[T.arm == arm]
    if split != 'all':
        T = T[T.train == (split == 'train')]
    outs = []
    for sub, g in T.groupby('subrun'):
        cfg = _cfg(arm, sub)
        allow = {tag: set(int(e) for e in gg.event_id) for tag, gg in g.groupby('tag')}
        cfg.file_tags = sorted(allow)
        out = str(_guard(work / 'reco' / label / arm / f'{sub}.parquet'))
        WB.reconstruct_subrun(cfg, bundle, out, jobs=jobs, allow_events=allow,
                              verbose=False)
        d = pd.read_parquet(out)
        d.insert(0, 'subrun', sub)
        outs.append(d)
    R = pd.concat(outs, ignore_index=True)
    R.to_parquet(work / f'reco_{label}_{arm}.parquet', index=False)
    print(f'{label} {arm}: {len(R)} events reconstructed')
    return R


def score(work: Path, arm: str, label: str, split: str = 'test') -> dict:
    """Against truth: response (w vs true tan), offset, head-on, resolution.
    Default on the TEST split: the train third fixed the hypers and w0/kw."""
    T = pd.read_parquet(work / 'truth.parquet')
    T = T[T.arm == arm]
    if split != 'all':
        T = T[T.train == (split == 'train')]
    R = pd.read_parquet(work / f'reco_{label}_{arm}.parquet')
    M = T.merge(R, on=['subrun', 'event_id'], how='inner')
    out = dict(label=label, arm=arm, n=int(len(M)))
    for ax in 'xy':
        t = M[f'tan_{ax}'].to_numpy()
        w = M[f'{ax}_w'].to_numpy() * 1e3
        ok = np.isfinite(w)
        t, w = t[ok], w[ok]
        core = (np.abs(t) > 0.12) & (np.abs(t) < 0.45)
        # w = a*sign(t) + b*t : b is the geometric v, a the outward |w| offset
        A = np.c_[np.sign(t[core]), t[core]]
        a, b = np.linalg.lstsq(A, w[core], rcond=None)[0]
        b_lin = np.polyfit(t[core], w[core], 1)[0]
        tan = M[f'{ax}_tan_theta'].to_numpy()[ok]
        res = tan - t
        mad = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else np.nan  # noqa: E731
        bins = [(0, 0.04), (0.04, 0.08), (0.08, 0.12), (0.12, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 0.6)]
        # resolution with the in-situ scale: (w - a sgn w)/b against truth
        tc = (w - a * np.sign(w)) / b
        out[ax] = dict(v_geom=float(b), v_geom_linear=float(b_lin), w_offset=float(a),
                       median_ratio_core=float(np.median(tan[core] / t[core])),
                       bins=[dict(lo=lo, hi=hi, n=int(m.sum()),
                                  sigma_tan_as_reco=mad(res[m]),
                                  sigma_tan_insitu=mad((tc - t)[m]),
                                  tail_insitu=float((np.abs(tc - t)[m] > 0.15).mean()) if m.any() else np.nan,
                                  sign_ok=float((np.sign(w[m]) == np.sign(t[m])).mean()) if m.any() else np.nan)
                             for lo, hi in bins
                             for m in [(np.abs(t) >= lo) & (np.abs(t) < hi)]])
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('step', choices=('truth', 'cache', 'profile', 'reco', 'score', 'hyper', 'w0kw'))
    ap.add_argument('--maxiter', type=int, default=250)
    ap.add_argument('--ratio', type=float, default=0.6)
    ap.add_argument('--work', required=True)
    ap.add_argument('--arm', default='A')
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--label', default='prod')
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--split', default='all', choices=('all', 'train', 'test'))
    ap.add_argument('--subs', default=None, help='file with one sub-run per line')
    a = ap.parse_args()
    work = _guard(a.work)
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('WFT_BEAM_BASE', str(work / 'beam') + '/')
    bundle = a.bundle or str(PROD_BUNDLE).format(arm=a.arm)
    if a.step == 'truth':
        subs = Path(a.subs).read_text().split()
        truth(work, subs)
    elif a.step == 'cache':
        cache(work, a.arm)
    elif a.step == 'profile':
        profile(work, a.arm, bundle, a.label, a.n, a.jobs)
    elif a.step == 'reco':
        reco(work, a.arm, bundle, a.label, a.jobs, a.split)
    elif a.step == 'hyper':
        hyper_fit(work, a.arm, bundle, a.label, a.jobs, a.maxiter, a.ratio)
    elif a.step == 'w0kw':
        set_w0kw(work, a.arm, a.label, bundle)
    elif a.step == 'score':
        print(json.dumps(score(work, a.arm, a.label, a.split if a.split != 'all' else 'test'), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
