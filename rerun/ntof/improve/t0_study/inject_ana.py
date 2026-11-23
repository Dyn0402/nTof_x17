#!/usr/bin/env python3
"""inject_ana.py -- response of the reconstruction to the injected tracks of inject.py.

    python -m rerun.ntof.improve.t0_study.inject_ana [--label ''] [--glob 'run_145_*']

The injected track in a host is the gated reconstructed track (x and y candidates joined by track_id) whose two
positions p0 lie within MATCH_MM of the donor's pass-1 positions.  In a pair host the host's own track is matched the
same way to the host's pass-1 positions.  Writes inject/matched[_label].parquet (one row per planned host) and prints:
  - acceptance (a gated track matched) per k, and any extra gated track (a fake or a split);
  - t0 - pin, its change from the donor's own value, and the rail fraction, against the true offset delta;
  - tc = (t0 - pin) + u50 (mean of the views) and its change from the donor's tc;
  - the tan change per view;
  - pairs: both tracks found, |dtc| of the reconstructed pair, and the change of the opening angle.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

WD = os.environ.get('WORK_DATA', '/media/ucla/work')
D = f'{WD}/ntof_pass1/ana/t0_study/inject'
MATCH_MM = 15.0
RAIL = 175.0


def tracks(cand):
    """Gated tracks: one row per (event, track_id) with x_ and y_ columns."""
    c = cand[cand.track_gated.fillna(False).astype(bool) & (cand.track_id >= 0)]
    cols = ['p0', 't0', 'q_u50', 'tan_theta', 'chi2', 'dof', 'split_child']
    x = c[c.plane == 'x'][['acq_tag', 'event_id', 'track_id'] + cols].add_prefix('x_')
    y = c[c.plane == 'y'][['acq_tag', 'event_id', 'track_id'] + cols].add_prefix('y_')
    t = x.merge(y, left_on=['x_acq_tag', 'x_event_id', 'x_track_id'], right_on=['y_acq_tag', 'y_event_id', 'y_track_id'])
    return t.rename(columns=dict(x_acq_tag='tag', x_event_id='event_id', x_track_id='track_id')).drop(
        columns=['y_acq_tag', 'y_event_id', 'y_track_id'])


def angle(tx1, ty1, tx2, ty2):
    a = np.c_[tx1, ty1, np.ones_like(tx1)]
    b = np.c_[tx2, ty2, np.ones_like(tx2)]
    c = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def match(plan, T):
    g = {k: v for k, v in T.groupby(['tag', 'event_id'])}
    rows = []
    for r in plan.itertuples(index=False):
        t = g.get((r.host_tag, int(r.host_eid)))
        out = dict(n_gated=0 if t is None else len(t))
        used = set()
        for who, px, py in (('ri', getattr(r, 'd_x_p0', np.nan), getattr(r, 'd_y_p0', np.nan)),
                            ('rh', getattr(r, 'h_x_p0', np.nan), getattr(r, 'h_y_p0', np.nan))):
            if t is None or not np.isfinite(px):
                continue
            dd = np.hypot(t.x_p0 - px, t.y_p0 - py).to_numpy()
            dd[list(used)] = np.inf
            j = int(np.argmin(dd))
            if dd[j] < MATCH_MM:
                used.add(j)
                out.update({f'{who}_{c}': t.iloc[j][c] for c in t.columns if c[:2] in ('x_', 'y_')})
        out['n_extra'] = out['n_gated'] - len(used)
        # pairs: each view on its own (the x/y combination of two tracks can swap; dtc and the dot product do not care)
        if r.kind == 'pair' and t is not None and len(t) == 2:
            for p in 'xy':
                pi, ph = getattr(r, f'd_{p}_p0'), getattr(r, f'h_{p}_p0')
                v = t[f'{p}_p0'].to_numpy()
                c0 = abs(v[0] - pi) + abs(v[1] - ph)
                c1 = abs(v[1] - pi) + abs(v[0] - ph)
                ii, hh = (0, 1) if c0 <= c1 else (1, 0)
                if abs(v[ii] - pi) < MATCH_MM and abs(v[hh] - ph) < MATCH_MM:
                    for c in ('t0', 'q_u50', 'tan_theta'):
                        out[f'pi_{p}_{c}'] = t.iloc[ii][f'{p}_{c}']
                        out[f'ph_{p}_{c}'] = t.iloc[hh][f'{p}_{c}']
        rows.append(out)
    return pd.concat([plan.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def q(v, p):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return np.quantile(v, p) if len(v) else np.nan


def report(M, arm):
    E = M[M.kind == 'empty'].copy()
    C = M[M.kind == 'control']
    if not len(E):
        return report_pairs(M, arm)
    print(f'\n===== {arm}: {E.donor.nunique()} donors x {E.k.nunique()} shifts; control hosts {len(C)} make a gated '
          f'track in {np.mean(C.n_gated > 0):.3f}')
    for p in 'xy':
        E[f'{p}_dt0'] = E[f'ri_{p}_t0'] - E[f'host_pin_{p}']
        E[f'{p}_ddt0'] = E[f'{p}_dt0'] - E[f'd_{p}_dt0']
    E['tc'] = 0.5 * (E.x_dt0 + E.ri_x_q_u50 + E.y_dt0 + E.ri_y_q_u50)
    E['tc_d'] = 0.5 * (E.d_x_dt0 + E.d_x_q_u50 + E.d_y_dt0 + E.d_y_q_u50)
    E['delta'] = 0.5 * (E.delta_x + E.delta_y)
    E['dtc'] = E.tc - E.tc_d
    E['rail'] = (E.x_dt0.abs() >= RAIL) | (E.y_dt0.abs() >= RAIL)
    E['found'] = E.ri_x_p0.notna()
    rows = []
    for k, g in E.groupby('k'):
        f = g[g.found]
        rows.append(dict(k=k, delta=g.delta.median(), acc=g.found.mean(), extra=np.mean(g.n_extra > 0),
                         x_ddt0=f.x_ddt0.median(), y_ddt0=f.y_ddt0.median(), rail=f.rail.mean(),
                         dtc=f.dtc.median(), dtc_16=q(f.dtc, .16), dtc_84=q(f.dtc, .84),
                         slope_tc=np.nan, dtanx=(f.ri_x_tan_theta - f.d_x_tan_theta).median(),
                         dtany=(f.ri_y_tan_theta - f.d_y_tan_theta).median(),
                         rtanx=np.median(np.abs(f.ri_x_tan_theta) / np.abs(f.d_x_tan_theta).clip(0.02)),
                         sdtanx=(f.ri_x_tan_theta - f.d_x_tan_theta).std(),
                         dang=np.median(angle(f.ri_x_tan_theta.to_numpy(), f.ri_y_tan_theta.to_numpy(),
                                              f.d_x_tan_theta.to_numpy(), f.d_y_tan_theta.to_numpy())) if len(f) else np.nan))
    R = pd.DataFrame(rows)
    print('acceptance, t0 and tc response, tan change vs k (delta = true arrival offset, ns; d* = injected - donor):')
    print(R.drop(columns='slope_tc').round(3).to_string(index=False))
    P = report_pairs(M, arm)
    return E, P


def report_pairs(M, arm):
    P = M[M.kind == 'pair'].copy()
    if len(P):
        for w in ('pi', 'ph'):
            for p in 'xy':
                pin = P[f'host_pin_{p}']
                P[f'{w}_{p}_dt0'] = P[f'{w}_{p}_t0'] - pin
            P[f'{w}_tc'] = 0.5 * (P[f'{w}_x_dt0'] + P[f'{w}_x_q_u50'] + P[f'{w}_y_dt0'] + P[f'{w}_y_q_u50'])
        P['both'] = P.pi_x_t0.notna() & P.pi_y_t0.notna()
        P['sep'] = np.hypot(P.d_x_p0 - P.h_x_p0, P.d_y_p0 - P.h_y_p0)
        P['dtc'] = (P.pi_tc - P.ph_tc).abs()
        P['open_true'] = angle(P.d_x_tan_theta.to_numpy(), P.d_y_tan_theta.to_numpy(),
                               P.h_x_tan_theta.to_numpy(), P.h_y_tan_theta.to_numpy())
        P['open_reco'] = angle(P.pi_x_tan_theta.to_numpy(), P.pi_y_tan_theta.to_numpy(),
                               P.ph_x_tan_theta.to_numpy(), P.ph_y_tan_theta.to_numpy())
        rows = []
        for k, g in P.groupby('k'):
            b = g[g.both & (g.sep > 50)]
            far = g[g.sep > 50]
            rows.append(dict(k=k, n=len(g), two_gated=np.mean(g.n_gated == 2), both=g.both.mean(), both_sep50=far.both.mean(),
                             gt300=np.mean(b.dtc > 300), dtc_med=b.dtc.median(),
                             dopen=(b.open_reco - b.open_true).median(),
                             dopen_68=q(np.abs(b.open_reco - b.open_true), .68)))
        print(f'{arm} pairs (an in-time track + the injected one) vs k; both_sep50 = both found when > 50 mm apart:')
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        for lab, g in (('|k| <= 1', P[P.k.abs() <= 1]), ('all k', P)):
            far = g[g.sep > 50]
            n = len(far)
            b = far.both.mean()
            print(f'  {lab}: > 50 mm apart, both found {b:.3f} +- {np.sqrt(b * (1 - b) / max(n, 1)):.3f} (n {n}); '
                  f'two gated tracks {np.mean(far.n_gated == 2):.3f}')
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='')
    ap.add_argument('--glob', default='run_145_*')
    a = ap.parse_args()
    allM = []
    for pf in sorted(glob.glob(f'{D}/{a.glob}{a.label}.plan.parquet')):
        stem = pf[:-len('.plan.parquet')]
        if not os.path.exists(stem + '.candidates.parquet'):
            continue
        plan = pd.read_parquet(pf)
        T = tracks(pd.read_parquet(stem + '.candidates.parquet'))
        M = match(plan, T).assign(file=os.path.basename(stem), arm=stem[-1 - len(a.label)])
        allM.append(M)
    M = pd.concat(allM, ignore_index=True)
    M.to_parquet(f'{D}/matched{a.label}.parquet', index=False)
    for arm, g in M.groupby('arm'):
        report(g, arm)


if __name__ == '__main__':
    main()
