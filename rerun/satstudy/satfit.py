#!/usr/bin/env python3
"""satfit.py -- how the forward fit treats saturated strips, and what a different treatment does (Dylan 10-26: "show
me what we currently do here and I'll try to give ways to fix the saturation treatment").

    satfit.py extract            waveforms of every det3 golden-row event with >= 3 saturated hits in a view, plus a
                                 random unsaturated control set -> ~/x17/cosmic_bench/satstudy/wf_det3.pkl
    satfit.py fit [--arms ...]   refit those events' production windows under each arm -> fits_det3.parquet

Arms (all on the production phys bundle of the golden row, t0 prior as production; only the censor mask changes):
  prod      production: samples >= SAT (3550 ADC, pedestal + CM subtracted) are censored -- out of the NNLS, plus a
            one-sided penalty if the model falls below the data there. Every other sample of a saturated strip
            (rising edge, falling edge, undershoot) is fitted with the linear template.
  low       censor at 3300 instead of 3550: catches plateau samples the common-mode subtraction pulled below 3550
  tail      prod + censor every sample of a saturated strip after its first clipped sample (the recovery: falling edge
            and undershoot), keep its rising edge
  strip     censor every sample of a saturated strip (keep only the unsaturated strips)
  both      two-sided: censor >= 3300 (top rail, as `low`) AND, on a strip that clipped, every sample <= -200 (the
            BOTTOM rail: 58 % of clipped det3 strips swing to ADC 0 after the plateau, max - min = 4100 ADC, the full
            12-bit range; railstats.py)
Run with WFT_PIVOT_START=1 (the golden rows' setting: with it the refit reproduces the table's p0 / w to < 0.05 mm /
1e-4 mm/ns; without it ~10 % of windows move by 0.1-0.5 mm). The window is rebuilt from the full-plane waveforms (strips with SNR > 6 in the cluster under the production fit,
padded by 3 strips as wft.io.extract_window), so `prod` is a re-fit, not a copy: its agreement with the table is
checked per event (fits_det3.parquet: prod_vs_table).
"""
import argparse, os, pickle, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis'),
                os.path.join(REPO, 'rerun', 'outliers')]
from olib import load  # noqa: E402

OUT = os.path.expanduser('~/x17/cosmic_bench/satstudy')
BUNDLE = os.path.join(OUT, 'bundle_det3_phys')   # EOS Analysis_rr1/<golden row>/mx17_3/wft/w12/it1/phys/bundle
KEY, DET = 'sat_det3', 'det3'
NSAT_MIN, N_CONTROL, THR, PAD = 3, 400, 6.0, 3
ARMS = ('prod', 'low', 'tail', 'strip', 'both')
BOTTOM = -200.0


def selection():
    d = load(DET)
    sat = (d.hx_nsat.fillna(0) >= NSAT_MIN) | (d.hy_nsat.fillna(0) >= NSAT_MIN)
    unsat = (d.hx_nsat.fillna(0) == 0) & (d.hy_nsat.fillna(0) == 0)
    ctrl = d[unsat].sample(N_CONTROL, random_state=1).event_id
    return d, set(d.event_id[sat]), set(ctrl)


def extract():
    from qa_config import get_config
    from wft import io as wio
    os.makedirs(OUT, exist_ok=True)
    _, sat, ctrl = selection(); want = sat | ctrl
    cfg = get_config(KEY); have = {}
    for v, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
        for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu):
            r = wio.FeuReader(f)
            if not np.isin(r.event_ids, list(want)).any():
                continue
            for eid, ft, W in r.iter_events(want):
                e = have.setdefault(int(eid), {})
                e[v] = W.astype(np.float16); e[f'noise_{v}'] = r.noise.astype(np.float16)
    pickle.dump(dict(wf=have, sat=sorted(sat), ctrl=sorted(ctrl)), open(os.path.join(OUT, 'wf_det3.pkl'), 'wb'))
    print(f'{len(have)} events ({len(sat)} saturated, {len(ctrl)} controls)')


def window(W, noise, pos, p0, tan):
    """The production-like window: the SNR > THR cluster that holds the fitted segment, padded by PAD strips."""
    ok = np.isfinite(pos); order = np.argsort(np.where(ok, pos, np.inf))[:ok.sum()]
    P = pos[order]; snr = W[order].max(1) / np.maximum(noise[order], 3.0)
    on = np.flatnonzero(snr > THR)
    if not len(on):
        return None
    pitch = np.median(np.diff(P))
    cls = np.split(on, np.flatnonzero(np.diff(P[on]) > 2.5 * pitch) + 1)
    lo, hi = sorted((p0, p0 + 30 * tan))
    best = min(cls, key=lambda c: 0.0 if (P[c].max() >= lo and P[c].min() <= hi) else
               min(abs(P[c].min() - hi), abs(P[c].max() - lo)) + 1e-3 * (-W[order][c].max()))
    a, b = max(0, best.min() - PAD), min(len(order) - 1, best.max() + PAD)
    ch = order[a:b + 1]
    return dict(ch=ch, pos=pos[ch], W=W[ch].astype(np.float64), noise=np.maximum(noise[ch], 3.0).astype(np.float64))


_CAL = None


def _init():
    global _CAL
    from wft.calib import CalibrationBundle
    from wft import model as wm
    _CAL = CalibrationBundle.load(BUNDLE); wm.use_calibration(_CAL); wm.set_nsamp(32)


def censor_mask(Wraw, arm, SAT):
    """The `sat` mask of wft.model.prep_plane under each arm (raw = before the gain division, as prep_plane)."""
    clip = Wraw >= SAT
    if arm == 'prod':
        return clip
    if arm == 'low':
        return Wraw >= 3300.0
    if arm == 'both':
        return (Wraw >= 3300.0) | ((Wraw <= BOTTOM) & clip.any(1)[:, None])
    m = clip.copy()
    for i in np.flatnonzero(clip.any(1)):
        if arm == 'tail':
            m[i, np.argmax(clip[i]):] = True
        elif arm == 'strip':
            m[i, :] = True
    return m


def fit_one(args):
    eid, v, P, ftst, arms = args
    from wft import model as wm, reco
    out = []
    pr = reco.t0_prior_for(_CAL, v, ftst) if ftst is not None and np.isfinite(ftst) else None
    real_prep = wm.prep_plane
    for arm in arms:
        def prep(PP, plane, _arm=arm):
            W, noise, pos, sat = real_prep(PP, plane)
            m = censor_mask(np.asarray(PP['W'], float), _arm, wm.SAT)
            dead = sat & ~(np.asarray(PP['W'], float) >= wm.SAT)
            return W, noise, pos, m | dead
        wm.prep_plane = prep
        try:
            f = reco.fit_plane(P, v, _CAL, t0_prior=pr)
        except Exception:
            f = None
        finally:
            wm.prep_plane = real_prep
        nc = int(censor_mask(P['W'], arm, wm.SAT).sum())
        out.append(dict(event_id=eid, view=v, arm=arm, ok=f is not None, n_cens=nc,
                        p0=f.p0 if f else np.nan, w=f.w if f else np.nan, t0=f.t0 if f else np.nan,
                        chi2=f.chi2 if f else np.nan, dof=f.dof if f else np.nan, q_sum=f.q_sum if f else np.nan,
                        n_strips=len(P['ch']), n_satstrip=int((P['W'] >= wm.SAT).any(1).sum())))
    return out


N_SAT_VIEWS, N_CTRL_VIEWS = 350, 150     # a random subset: ~8 s per saturated fit and arm on 7 cores


def fit(arms, nproc):
    from multiprocessing import Pool
    S = pickle.load(open(os.path.join(OUT, 'wf_det3.pkl'), 'rb')); wf = S['wf']; sat = set(S['sat'])
    d = load(DET).set_index('event_id')
    from wft.io import strip_position_map
    from qa_config import get_config
    cfg = get_config(KEY); pm = strip_position_map(cfg); pmv = {'x': pm[cfg.MX17_FEU_X], 'y': pm[cfg.MX17_FEU_Y]}
    jobs = []
    for eid, ev in wf.items():
        r = d.loc[eid]
        for v in 'xy':
            if v not in ev or not r[f'{v}_ok']:
                continue
            P = window(ev[v].astype(np.float32), ev[f'noise_{v}'].astype(np.float32), pmv[v], r[f'{v}_p0'],
                       r[f'{v}_w'] * 1e3 / 35.8)
            if P is not None:
                # an unsaturated control has the same mask under prod / tail / strip: only `low` can differ
                ar = arms if eid in sat else [x for x in arms if x in ('prod', 'low', 'both')]
                jobs.append((int(eid), v, P, r.get(f'{v}_ftst', np.nan), ar))
    # the view itself must be saturated (>= NSAT_MIN saturated hits), not just its event; then a fixed random subset
    rng = np.random.default_rng(7)
    js = [j for j in jobs if j[0] in sat and d.loc[j[0], f'h{j[1]}_nsat'] >= NSAT_MIN]
    jc = [j for j in jobs if j[0] not in sat]
    js = [js[i] for i in sorted(rng.choice(len(js), min(N_SAT_VIEWS, len(js)), replace=False))]
    jc = [jc[i] for i in sorted(rng.choice(len(jc), min(N_CTRL_VIEWS, len(jc)), replace=False))]
    jobs = js + jc
    print(len(jobs), 'event-views')
    rows = []
    with Pool(nproc, initializer=_init) as pool:
        for i, res in enumerate(pool.imap_unordered(fit_one, jobs, chunksize=4)):
            rows += res
            if i % 25 == 0:
                print(i, flush=True)
    F = pd.DataFrame(rows)
    F['group'] = np.where(F.event_id.isin(S['sat']), 'sat', 'ctrl')
    out = os.path.join(OUT, 'fits_det3.parquet')
    if os.path.exists(out):      # a later run adds arms
        old = pd.read_parquet(out); F = pd.concat([old[~old.arm.isin(F.arm.unique())], F], ignore_index=True)
    F.to_parquet(out); print('wrote', len(F))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['extract', 'fit'])
    ap.add_argument('--arms', nargs='+', default=list(ARMS)); ap.add_argument('--nproc', type=int, default=7)
    a = ap.parse_args()
    if a.cmd == 'extract':
        extract()
    else:
        fit(a.arms, a.nproc)


if __name__ == '__main__':
    main()
