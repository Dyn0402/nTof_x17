#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
angle_response.py -- the angle response of chambers A and C, measured on
through-going cosmics with NO capsule assumption, and tested on beam tracks.
HANDOFF_TRACKING_2026-10-06.md §2 steps 1-2, as revised in its §7.

THE REFERENCE.  A straight particle crossing A (z = +234.6) and C (z = -234.6)
lies on the line through the two chambers' mesh-plane crossings, ``p0``.  Its
slope ``j = dx/dz`` (or dy/dz) has a 469 mm lever arm, so a 0.5 mm position
error is 0.002 in j: j is truth for this purpose.  Positions do not depend on
k (`build_tracks.local_and_global`), so j is identical under every borrowed k.

THE MEASUREMENT IS IN RAW UNITS.  The track slope a table carries is
``s = k_borrowed * tan_raw``, so ``s / j = k_borrowed / k_true`` -- a ratio
above 1 means the tan reads too LARGE.  Everything below is quoted on
``raw = s / k_borrowed``, which is the same number under either borrowed k
(checked: ``check_k_independence``), and as ``k_eff = |j| / raw``, the k a
track at that angle needs.  A single multiplicative k is the hypothesis being
tested, not assumed.

THE RESPONSE, R.  In bins of |j|: the median of the folded raw tan,
``raw * sign(j)``.  Its inverse, ``R_inv`` (monotone linear interpolation of
the bin medians), maps a raw tan to a corrected tan and is what the beam test
applies.  Folding pools the two signs; ``response`` also reports them apart.

THE BEAM TEST (step 5's first half).  The pointing-coincident beam sample of
`k_arm` (gated, the track's arm fired, x charge in the 25-75 % window, lever
30-130 mm), built from the campaign track table: expected tan = (x_local -
foot)/234.6, x view only (the capsule is 80 mm long in y).  Two tests:

  * the folded beam response against the cosmic one, bin by bin;
  * closure: apply the cosmic R_inv to the beam tans and re-run `k_arm`'s band
    (gradient) and track (median ratio) estimators.  If the cosmic response is
    the beam's response, both come out at 1 and agree with each other.

Nothing is written outside ``results/tracking/pooled/``.

    python ntof_cosmics/angle_response.py            # build sample + all tables
    python ntof_cosmics/angle_response.py --no-beam  # cosmics only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import cosmic_tracks as CT  # noqa: E402

RUN = 'run_149'
K_FROM = ('run_147', 'run_150')
#: the borrowed k the per-track tables quote by default (k-independent below)
K_REF = 'run_150'
OUT = CT.OUT / 'pooled'
CAMPAIGN = Path('/home/dylan/x17/sept26_prelim/stage3_fullpass/tracks_campaign.parquet')
BEAM_RUNS = ('run_145', 'run_147', 'run_150', 'run_152')

#: |j| bins for the response.  Below 0.1 the single-track angle is not
#: measured (see RES_EDGES) and the ratio is undefined anyway.
R_EDGES = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60])
#: |j| bins for the resolution, down to normal incidence
RES_EDGES = np.array([0.0, 0.04, 0.08, 0.12, 0.20, 0.30, 0.45, 0.60])
#: the primary selection's lateral-miss cap [mm].  Loose on purpose: the 20 mm
#: CLEAN_SEP_MM cut selects on the very agreement being measured.  20 mm and
#: no cap are the variants.
SEP_PRIMARY = 60.0
TAIL = 0.15
N_BOOT = 300
RNG = np.random.default_rng(20261006)

TRACK_COLS = ['x_local', 'y_local', 'tan_raw_x', 'tan_raw_y', 'x_n_strips',
              'y_n_strips', 'chi2dof_x', 'chi2dof_y', 'x_q_uend', 'y_q_uend',
              'x_t0', 'y_t0', 'q_total', 'x_tan_err', 'y_tan_err', 'k_arm']


def mad(v):
    v = np.asarray(v, float)
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


# --------------------------------------------------------------------------- #
# The cosmic sample
# --------------------------------------------------------------------------- #
def ac_sample(k_from: str) -> pd.DataFrame:
    """Every A-C pair of gated, calibrated tracks in one trigger (no sep cut),
    with the joined-line slopes, both tracks' raw tans and the gated-track
    multiplicity of every arm in that trigger."""
    d = CT.OUT / f'k_{k_from}'
    rows = []
    for pf in sorted(d.glob(f'pairs_{RUN}_*.parquet')):
        sub = pf.name[len(f'pairs_{RUN}_'):-len('.parquet')]
        P = pd.read_parquet(pf)
        c = P[P.pair == 'AC']
        if c.empty:
            continue
        t = pd.read_parquet(CT.tracks_path(RUN, sub, k_from))
        ng = t[t.gated].groupby(['event_id', 'arm']).size().unstack(fill_value=0)
        ng = ng.reindex(columns=list(CT.ARMS), fill_value=0)
        ti = t.set_index(['event_id', 'arm', 'track_id'])
        a = ti.loc[list(zip(c.event_id, c.arm1, c.track1))].reset_index()
        b = ti.loc[list(zip(c.event_id, c.arm2, c.track2))].reset_index()
        o = pd.DataFrame(dict(subrun=sub, event_id=c.event_id.to_numpy(),
                              sep_mm=c.sep_mm.to_numpy(), open_deg=c.open_deg.to_numpy()))
        Jz = b.p0_z.to_numpy() - a.p0_z.to_numpy()
        o['jx'] = (b.p0_x.to_numpy() - a.p0_x.to_numpy()) / Jz
        o['jy'] = (b.p0_y.to_numpy() - a.p0_y.to_numpy()) / Jz
        for lab, tr in (('A', a), ('C', b)):
            for ax in 'xy':
                # global slope over the borrowed k: the raw tan in the GLOBAL
                # sign convention (C's local y runs opposite to global y)
                o[f'{lab}_raw_{ax}'] = (tr[f'd_{ax}'] / tr['d_z']).to_numpy() / tr.k_arm.to_numpy()
            for cc in TRACK_COLS:
                o[f'{lab}_{cc}'] = tr[cc].to_numpy()
        for arm in CT.ARMS:
            o[f'n_{arm}'] = ng.reindex(c.event_id)[arm].fillna(0).astype(int).to_numpy()
        rows.append(o)
    return pd.concat(rows, ignore_index=True)


def selections(O: pd.DataFrame) -> dict:
    one = (O.n_A == 1) & (O.n_C == 1)
    return {'primary': one & (O.sep_mm < SEP_PRIMARY),
            'sep20': one & (O.sep_mm < CT.CLEAN_SEP_MM),
            'nosep': one}


def check_k_independence(samples: dict) -> dict:
    """raw must be the same number under every borrowed k (same events)."""
    a, b = (samples[k].set_index(['subrun', 'event_id', 'A_x_local', 'C_x_local'])
            for k in K_FROM)
    common = a.index.intersection(b.index)
    out = {}
    for arm in 'AC':
        for ax in 'xy':
            col = f'{arm}_raw_{ax}'
            d = (a.loc[common, col] - b.loc[common, col]).abs()
            out[f'{arm}{ax}'] = float(d.max())
    out['n_common'] = int(len(common))
    return out


# --------------------------------------------------------------------------- #
# Response and resolution
# --------------------------------------------------------------------------- #
def _boot_median(vals, groups, n=N_BOOT):
    keys = np.unique(groups)
    idx = {k: np.flatnonzero(groups == k) for k in keys}
    v = []
    for _ in range(n):
        pick = RNG.choice(keys, len(keys))
        ii = np.concatenate([idx[k] for k in pick])
        v.append(np.median(vals[ii]))
    return float(np.std(v))


def response(O: pd.DataFrame, sel: pd.Series, label: str) -> pd.DataFrame:
    rows = []
    g = O[sel]
    for arm in 'AC':
        for ax in 'xy':
            j = g[f'j{ax}'].to_numpy()
            raw = g[f'{arm}_raw_{ax}'].to_numpy()
            subs = g.subrun.to_numpy()
            aj = np.abs(j)
            fr = raw * np.sign(j)
            for lo, hi in zip(R_EDGES[:-1], R_EDGES[1:]):
                m = (aj >= lo) & (aj < hi)
                if m.sum() < 20:
                    continue
                ratio = fr[m] / aj[m]
                r = dict(selection=label, arm=arm, axis=ax, lo=lo, hi=hi, n=int(m.sum()),
                         j_med=float(np.median(aj[m])), raw_med=float(np.median(fr[m])),
                         raw_med_err=_boot_median(fr[m], subs[m]),
                         ratio_med=float(np.median(ratio)),
                         ratio_med_err=_boot_median(ratio, subs[m]))
                r['k_eff'] = 1.0 / r['ratio_med']
                r['k_eff_err'] = r['ratio_med_err'] / r['ratio_med'] ** 2
                for sgn, nm in ((1, 'pos'), (-1, 'neg')):
                    mm = m & (np.sign(j) == sgn)
                    r[f'ratio_med_{nm}'] = float(np.median(fr[mm] / aj[mm])) if mm.sum() > 10 else np.nan
                rows.append(r)
    return pd.DataFrame(rows)


def gradient_k(R: pd.DataFrame) -> pd.DataFrame:
    """The local derivative k_d = d|j| / d raw between neighbouring bins -- the
    scale a DIFFERENCE of two tans (a pair's divergence) needs."""
    rows = []
    for (sel, arm, ax), g in R.groupby(['selection', 'arm', 'axis']):
        g = g.sort_values('j_med')
        jm, rm = g.j_med.to_numpy(), g.raw_med.to_numpy()
        for i in range(len(g) - 1):
            dr = rm[i + 1] - rm[i]
            rows.append(dict(selection=sel, arm=arm, axis=ax,
                             j_mid=0.5 * (jm[i] + jm[i + 1]),
                             k_grad=(jm[i + 1] - jm[i]) / dr if dr > 0 else np.nan))
    return pd.DataFrame(rows)


def _huber(A, y, c=1.5, it=40):
    w = np.ones(len(y))
    for _ in range(it):
        sw = np.sqrt(w)
        p = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
        r = y - A @ p
        s = 1.4826 * np.median(np.abs(r))
        w = np.minimum(1.0, c * s / np.maximum(np.abs(r), 1e-12))
    return p, s


def models(O: pd.DataFrame, sel: pd.Series) -> pd.DataFrame:
    """Two response models on |j| in [0.1, 0.6), robust (Huber) fits of the
    folded raw tan: multiplicative raw = |j|/k, and gradient + offset
    raw = |j|/k_d + c.  Also the trend of the per-track ratio with |j|, with a
    sub-run bootstrap error: zero slope is "a single k describes it"."""
    rows = []
    g = O[sel]
    for arm in 'AC':
        for ax in 'xy':
            j = g[f'j{ax}'].to_numpy()
            aj, fr = np.abs(j), g[f'{arm}_raw_{ax}'].to_numpy() * np.sign(j)
            m = (aj >= R_EDGES[0]) & (aj < R_EDGES[-1])
            aj, fr, subs = aj[m], fr[m], g.subrun.to_numpy()[m]
            p1, s1 = _huber(aj[:, None], fr)
            p2, s2 = _huber(np.c_[aj, np.ones_like(aj)], fr)

            def trend(ii):
                p, _ = _huber(np.c_[aj[ii], np.ones(len(ii))], fr[ii] / aj[ii], it=15)
                return p[0]
            keys = np.unique(subs)
            idx = {k: np.flatnonzero(subs == k) for k in keys}
            tb = [trend(np.concatenate([idx[k] for k in RNG.choice(keys, len(keys))]))
                  for _ in range(100)]
            t0 = trend(np.arange(len(aj)))
            rows.append(dict(arm=arm, axis=ax, n=int(m.sum()),
                             k_mult=1 / p1[0], sigma_mult=s1,
                             k_grad=1 / p2[0], offset=p2[1], sigma_grad=s2,
                             ratio_trend_per_unit_tan=t0, ratio_trend_err=float(np.std(tb)),
                             trend_signif=abs(t0) / float(np.std(tb))))
    return pd.DataFrame(rows)


def r_inverse(R: pd.DataFrame, arm: str, ax: str, selection='primary'):
    """raw -> corrected tan, from the folded response bin medians.  Below the
    first bin: the first bin's ratio (through the origin); above the last: the
    last bin's ratio.  Sign preserved."""
    g = R[(R.selection == selection) & (R.arm == arm) & (R.axis == ax)].sort_values('raw_med')
    rx, jy = g.raw_med.to_numpy(), g.j_med.to_numpy()
    if np.any(np.diff(rx) <= 0):
        raise ValueError(f'{arm}{ax}: response not monotone, cannot invert')
    k_lo, k_hi = jy[0] / rx[0], jy[-1] / rx[-1]

    def f(raw):
        raw = np.asarray(raw, float)
        a = np.abs(raw)
        out = np.interp(a, rx, jy)
        out = np.where(a < rx[0], a * k_lo, out)
        out = np.where(a > rx[-1], a * k_hi, out)
        return np.sign(raw) * out
    return f


def resolution(O: pd.DataFrame, sel: pd.Series, R: pd.DataFrame) -> pd.DataFrame:
    """Per-track angle error against the joined line, in TRUE-tan units: the
    raw tan through the cosmic R_inv, minus j, folded.  So the core bins are
    unbiased by construction and the number is the scatter a corrected track
    carries.  Near normal (|j| < 0.1) R_inv is an extrapolation -- the point of
    those bins is the scatter, which is far larger than any bias it could
    leave."""
    rows = []
    g = O[sel]
    for arm in 'AC':
        for ax in 'xy':
            f = r_inverse(R, arm, ax)
            j = g[f'j{ax}'].to_numpy()
            raw = g[f'{arm}_raw_{ax}'].to_numpy()
            res = (f(raw) - j) * np.where(j >= 0, 1.0, -1.0)
            pull_err = g[f'{arm}_{ax}_tan_err'].to_numpy() * g[f'{arm}_k_arm'].to_numpy()
            for lo, hi in zip(RES_EDGES[:-1], RES_EDGES[1:]):
                m = (np.abs(j) >= lo) & (np.abs(j) < hi)
                if m.sum() < 20:
                    continue
                rows.append(dict(arm=arm, axis=ax, lo=lo, hi=hi, n=int(m.sum()),
                                 bias=float(np.median(res[m])), sigma=mad(res[m]),
                                 tail_frac=float((np.abs(res[m]) > TAIL).mean()),
                                 tan_err_quoted=float(np.median(pull_err[m])),
                                 pull_mad=mad(res[m] / pull_err[m])))
    return pd.DataFrame(rows)


def drift_span(O: pd.DataFrame, sel: pd.Series) -> pd.DataFrame:
    """Every through-goer crosses the whole 30 mm gap, so the drift-time extent
    of its charge profile is a drift-velocity handle independent of any angle.
    INDICATIVE ONLY: q_uend is the last 60 ns depth bin above 5 % of the
    profile peak (diffusion and shaping push it late), and it rails at 1080 ns."""
    rows = []
    g = O[sel]
    for arm in 'AC':
        for ax in 'xy':
            u = g[f'{arm}_{ax}_q_uend'].to_numpy(float)
            t0 = g[f'{arm}_{ax}_t0'].to_numpy(float)
            ok = np.isfinite(u) & (u < 1079)
            span = u[ok] - t0[ok]
            rows.append(dict(arm=arm, axis=ax, n=int(ok.sum()),
                             frac_railed=float(1 - ok.mean()),
                             span_med_ns=float(np.median(span)),
                             span_q25=float(np.quantile(span, .25)),
                             span_q75=float(np.quantile(span, .75)),
                             v_um_ns=30.0e3 / float(np.median(span)),
                             v_prior_over_v=42.6 / (30.0e3 / float(np.median(span)))))
    return pd.DataFrame(rows)


def slope_flag(O: pd.DataFrame, sel: pd.Series) -> pd.DataFrame:
    """`wft`'s ``slope_reliable`` (|raw tan| >= TAN_MIN_SLOPE = 0.08) against
    the true slope.  It is set on the RECONSTRUCTED tan, so a near-normal track
    whose fit lands away from zero is called reliable."""
    from wft.reco import TAN_MIN_SLOPE
    rows = []
    g = O[sel]
    for arm in 'AC':
        for ax in 'xy':
            j = g[f'j{ax}'].abs().to_numpy()
            rel = g[f'{arm}_tan_raw_{ax}'].abs().to_numpy() >= TAN_MIN_SLOPE
            near = j < 0.06
            rows.append(dict(arm=arm, axis=ax, tan_min_slope=TAN_MIN_SLOPE,
                             n_true_near=int(near.sum()),
                             frac_true_near_flagged_reliable=float(rel[near].mean()),
                             frac_reliable_true_near=float(near[rel].mean()),
                             n_unreliable=int((~rel).sum()),
                             unreliable_true_median=float(np.median(j[~rel])),
                             frac_unreliable_true_below_008=float((j[~rel] < 0.08).mean())))
    return pd.DataFrame(rows)


def multiplicity(O: pd.DataFrame) -> dict:
    """On clean single-muon crossings (best pair per trigger, sep < 10 mm): how
    often a chamber reports a second gated track.  A production-chain false
    split on a straight MIP, with the joined line as truth."""
    best = O.sort_values('sep_mm').drop_duplicates(['subrun', 'event_id'])
    cl = best[best.sep_mm < 10]
    return dict(n_clean=int(len(cl)),
                **{f'frac_{a}_ge2': float((cl[f'n_{a}'] >= 2).mean()) for a in 'AC'},
                **{f'n_{a}_ge2': int((cl[f'n_{a}'] >= 2).sum()) for a in 'AC'})


# --------------------------------------------------------------------------- #
# The beam test
# --------------------------------------------------------------------------- #
def beam_sample(runs=BEAM_RUNS) -> pd.DataFrame:
    """`k_arm.coincident_tracks`, rebuilt from the campaign track table.  The
    local `coincident_tracks` path finds ~10 % of the sample condor did (the
    slim export here is not the one k_arm ran on); this one reproduces k_arm's
    per-sub-run band/track values to ~3 % (run_150 A stat090_0000: 1.260/1.219
    against 1.263/1.230)."""
    import pyarrow.dataset as ds
    from ntof_tracking import run145_target_imaging as TI
    from sept26_prelim_analysis import k_arm as K
    d = ds.dataset(str(CAMPAIGN))
    cols = ['run', 'subrun', 'arm', 'gated', 'coinc_this_arm', 'x_local',
            'tan_raw_x', 'x_q_sum', 'k_arm']
    t = d.to_table(columns=cols, filter=ds.field('run').isin(list(runs))
                   & ds.field('arm').isin(['A', 'C'])).to_pandas()
    t = t[t.gated & t.coinc_this_arm.astype(bool) & (t.x_q_sum > 0)
          & np.isfinite(t.tan_raw_x)]
    out = []
    for (run, arm, sub), g in t.groupby(['run', 'arm', 'subrun']):
        lo, hi = np.percentile(g.x_q_sum, K.CHARGE_WINDOW)
        g = g[(g.x_q_sum >= lo) & (g.x_q_sum <= hi)].copy()
        g['lev'] = g.x_local - TI.PINWHEEL[arm]
        g = g[(g.lev.abs() > K.LEVER_WINDOW_MM[0]) & (g.lev.abs() < K.LEVER_WINDOW_MM[1])
              & (g.tan_raw_x.abs() > 1e-3)]
        g['t_exp'] = g.lev / K.D_PERP_MM
        out.append(g)
    return pd.concat(out, ignore_index=True)


def beam_response(B: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, arm), g in B.groupby(['run', 'arm']):
        te, raw = g.t_exp.to_numpy(), g.tan_raw_x.to_numpy()
        a, fr = np.abs(te), raw * np.sign(te)
        for lo, hi in zip(R_EDGES[:-1], R_EDGES[1:]):
            m = (a >= lo) & (a < hi)
            if m.sum() < 50:
                continue
            rows.append(dict(run=run, arm=arm, axis='x', lo=lo, hi=hi, n=int(m.sum()),
                             j_med=float(np.median(a[m])), raw_med=float(np.median(fr[m])),
                             ratio_med=float(np.median(fr[m] / a[m])),
                             ratio_med_err=_boot_median(fr[m] / a[m], g.subrun.to_numpy()[m], 100)))
    return pd.DataFrame(rows)


def beam_closure(B: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    """k_arm's band and track estimators on the beam sample, per run and sub-run,
    on the raw tans and on the cosmic-corrected tans.  Closure = both at 1."""
    from sept26_prelim_analysis import k_arm as K
    rows = []
    for arm in 'AC':
        f = r_inverse(R, arm, 'x')
        for (run, sub), g in B[B.arm == arm].groupby(['run', 'subrun']):
            for lab, tx in (('raw', g.tan_raw_x.to_numpy()),
                            ('cosmic_corrected', f(g.tan_raw_x.to_numpy()))):
                S = dict(xl=g.x_local.to_numpy(), tx=tx, foot_x=g.x_local.to_numpy()[0] - g.lev.to_numpy()[0])
                rows.append(dict(arm=arm, run=run, subrun=sub, tans=lab, n=int(len(g)),
                                 band=K.band_k(S), track=K.track_k(S),
                                 k_arm_applied=float(g.k_arm.iloc[0])))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--no-beam', action='store_true')
    a = ap.parse_args()
    dest = CT._guard(OUT)
    dest.mkdir(parents=True, exist_ok=True)

    samples = {k: ac_sample(k) for k in K_FROM}
    for k, O in samples.items():
        O.to_parquet(dest / f'ac_sample_{RUN}_k{k}.parquet', index=False)
    summary = dict(run=RUN, k_from=list(K_FROM), k_ref=K_REF,
                   k_independence_max_abs_raw_diff=check_k_independence(samples))
    O = samples[K_REF]
    sels = selections(O)
    summary['n_selected'] = {k: int(v.sum()) for k, v in sels.items()}
    summary['n_ac_pairs'] = int(len(O))

    R = pd.concat([response(O, s, lab) for lab, s in sels.items()], ignore_index=True)
    R.to_csv(dest / 'response.csv', index=False)
    gradient_k(R).to_csv(dest / 'response_gradient.csv', index=False)
    M = models(O, sels['primary'])
    M.to_csv(dest / 'response_models.csv', index=False)
    Rs = resolution(O, sels['primary'], R)
    Rs.to_csv(dest / 'resolution.csv', index=False)
    drift_span(O, sels['primary']).to_csv(dest / 'drift_span.csv', index=False)
    slope_flag(O, sels['primary']).to_csv(dest / 'slope_flag.csv', index=False)
    summary['multiplicity'] = multiplicity(O)

    # per-k ratio -> k_true, the handoff's pooled medians made k-independent
    kt = []
    for k, Ok in samples.items():
        s = selections(Ok)['sep20']
        for arm in 'AC':
            for ax in 'xy':
                g = Ok[s]
                j = g[f'j{ax}']
                m = j.abs() > 0.1
                kb = float(g[f'{arm}_k_arm'].iloc[0])
                ratio = float(np.median(g[f'{arm}_raw_{ax}'][m] * kb / j[m]))
                kt.append(dict(k_from=k, arm=arm, axis=ax, k_borrowed=kb,
                               median_ratio=ratio, k_true=kb / ratio))
    pd.DataFrame(kt).to_csv(dest / 'k_true_by_borrowed_k.csv', index=False)

    if not a.no_beam:
        B = beam_sample()
        summary['beam_n'] = B.groupby(['run', 'arm']).size().rename('n').reset_index().to_dict('records')
        tq = {}
        for (run, arm), g in B.groupby(['run', 'arm']):
            tq[f'{run}_{arm}'] = [float(x) for x in g.t_exp.abs().quantile([.16, .5, .84])]
        summary['beam_abs_texp_quantiles_16_50_84'] = tq
        beam_response(B).to_csv(dest / 'beam_response.csv', index=False)
        C = beam_closure(B, R)
        C.to_csv(dest / 'beam_closure.csv', index=False)
        summary['beam_closure'] = (C.groupby(['arm', 'run', 'tans'])[['band', 'track']]
                                   .median().reset_index().to_dict('records'))
    (dest / 'angle_response.json').write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    print(M.to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
