#!/usr/bin/env python3
"""dbeam.py -- why D's column-duration box fit fails on beam (run_145) and works on cosmics (run_149).

Inputs: centroids.py per-trigger main clusters (strip sums S(t), 20 x 60 ns) for A, C, D; the beam trigger attribution
from the scintillator stack (/media/dylan/data/x17/scint_stack/tracks/stack_run_145.parquet: hw_<arm> = that arm's
wall-sum AND plastic coincidence fired, i.e. the arm's own trigger; t_since_flash_ns), and for cosmics the n_TOF clock
match (ntof_cosmics/clock_match.py: the triggering arm).

Per arm and view, the stacked median of the normalised S is fitted with colsum.GridFit (the pass-1 X template, t0 grid
-300..900 ns, T 300..1500 ns), exactly as the v-in-situ measurement did, for:
  * all clean single clusters (>= 5 strips, one cluster, not saturated) -- the selection that failed for D on beam;
  * the arm's OWN trigger (hw_<arm>) -- the particle that made the trigger crossed this arm's wall and plastic;
  * another arm's trigger only (the cluster in this arm is then not the triggering particle).
Plus, per selection, the per-event box start t0 (one fit per event) and the fraction of events whose box start lies
within +-150 ns of the prompt time.

    dbeam.py           # stack fits + json under d_b/dbeam/
    dbeam.py dreco     # D's own v1_D beam fits by trigger and time since the flash (after reco_D.sh)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun', 'model_audit', 'followups', 'fa_gas_v_tilt')]
import colsum  # noqa: E402

W = '/media/ucla/work/ntof_pass1'
CEN = f'{W}/d_b/centroids'
OUT = f'{W}/d_b/dbeam'
STACK = '/media/dylan/data/x17/scint_stack/tracks/stack_run_145.parquet'
GAP = {'A': 28.08, 'B': 31.12, 'C': 28.0, 'D': 29.44}
colsum.T0_GRID = np.arange(-300.0, 900.0 + 1e-9, 5.0)
colsum.T_GRID = np.arange(300.0, 1500.0 + 1e-9, 5.0)
BEAM_SUBS = ['stat090_0000', 'stat090_0001', 'stat090_0002']
COS_SUBS = [f'cosbounce_cos_{i:04d}' for i in range(12)]


def gridfit(arm):
    t = np.load(f'{W}/colsum/tmpl/tmpl_{arm}.npz')
    return colsum.GridFit(colsum.Kern(t['grid'], t['tmpl_x'] / t['tmpl_x'].max()), 20)


def load(run, subs, arm):
    rows = []
    for s in subs:
        f = f'{CEN}/r{run}_{s}.npz'
        if not os.path.exists(f):
            continue
        z = np.load(f)
        d = dict(subrun=np.full(len(z[f'{arm}_eid']), s), event_id=z[f'{arm}_eid'])
        for p in 'xy':
            for c in ('has', 'nsig', 'ncl', 'sat', 'cen', 'q', 'smax', 'ftst'):
                d[f'{p}_{c}'] = z[f'{arm}_{p}_{c}']
            d[f'{p}_S'] = list(z[f'{arm}_{p}_S'])
        rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True)


def triggers_beam():
    s = pd.read_parquet(STACK, columns=['subrun', 'event_id', 'hw_A', 'hw_B', 'hw_C', 'hw_D', 't_since_flash_ns'])
    s['subrun'] = s.subrun.astype(str)
    return s.drop_duplicates(['subrun', 'event_id'])


def triggers_cos():
    from ntof_cosmics import cosmic_wall_scale as CWS
    out = []
    for s in COS_SUBS:
        t = CWS.matched_triggers(s)
        if len(t):
            out.append(pd.DataFrame(dict(subrun=s, event_id=t.event_id.to_numpy(), trig_arm=t.ntof_arm.to_numpy())))
    t = pd.concat(out, ignore_index=True)
    for a in 'ABCD':
        t[f'hw_{a}'] = t.trig_arm == a
    return t


def stack_fit(g, S, rng, nboot=30):
    S = np.stack(S).astype(float)
    S /= np.maximum(S.max(1, keepdims=True), 1)
    med = np.median(S, 0)
    r = g.fit(med)
    boots = [g.fit(np.median(S[rng.integers(0, len(S), len(S))], 0)) for _ in range(nboot)]
    return med, r, float(np.std([b[0] for b in boots])), float(np.std([b[1] for b in boots]))


def per_event_t0(g, S, nmax=1500, rng=None):
    S = np.stack(S).astype(float)
    if len(S) > nmax:
        S = S[rng.choice(len(S), nmax, replace=False)]
    r = np.array([g.fit(s)[:2] for s in S])
    return r[:, 0], r[:, 1]


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(1)
    TB, TC = triggers_beam(), triggers_cos()
    rows, stacks = [], {}
    for arm in 'ACD':
        g = gridfit(arm)
        for run, subs, trig in (('145', BEAM_SUBS, TB), ('149', COS_SUBS, TC)):
            d = load(run, subs, arm)
            d = d.merge(trig, on=['subrun', 'event_id'], how='left')
            for p in 'xy':
                clean = d[f'{p}_has'] & (d[f'{p}_nsig'] >= 5) & (d[f'{p}_ncl'] == 1) & ~d[f'{p}_sat']
                own = d[f'hw_{arm}'].astype('boolean').fillna(False).astype(bool)
                known = d['hw_A'].notna()
                other = known & ~own
                busy = d[f'{p}_has'] & (d[f'{p}_ncl'] > 1)
                for lab, m in (('all clean', clean), ('clean, own trigger', clean & own),
                               ('clean, other arm trigger', clean & other)):
                    if m.sum() < 30:
                        continue
                    med, r, e0, eT = stack_fit(g, d.loc[m, f'{p}_S'], rng)
                    t0e, Te = per_event_t0(g, d.loc[m, f'{p}_S'], rng=rng)
                    stacks[f'{run}_{arm}{p}_{lab}'] = med.tolist()
                    rows.append(dict(run=f'run_{run}', arm=arm, view=p, selection=lab, n=int(m.sum()),
                                     stack_t0=r[0], stack_t0_err=e0, stack_T=r[1], stack_T_err=eT,
                                     v=GAP[arm] * 1e3 / r[1], v_err=GAP[arm] * 1e3 / r[1] ** 2 * eT,
                                     t0_at_grid_edge=bool(r[0] <= colsum.T0_GRID[0] + 1e-6),
                                     T_at_grid_edge=bool(r[1] >= colsum.T_GRID[-1] - 5),
                                     evt_t0_p16=float(np.nanpercentile(t0e, 16)), evt_t0_p50=float(np.nanmedian(t0e)),
                                     evt_t0_p84=float(np.nanpercentile(t0e, 84)),
                                     frac_late=float(np.mean(t0e > 200)), frac_end_past_window=float(np.mean(t0e + r[1] > 1200)),
                                     frac_prompt=float(np.mean(np.abs(t0e - 0) < 150)),
                                     frac_t0_edge=float(np.mean(t0e <= colsum.T0_GRID[0] + 1e-6))))
                # activity: triggers with a cluster, and with more than one cluster, per trigger kind
                n_trig = len(d)
                rows.append(dict(run=f'run_{run}', arm=arm, view=p, selection='ACTIVITY',
                                 n=int(d[f'{p}_has'].sum()), frac_busy=float(busy.sum() / max(d[f'{p}_has'].sum(), 1)),
                                 frac_clean=float(clean.sum() / max(d[f'{p}_has'].sum(), 1)),
                                 own_frac_of_clean=float((clean & own).sum() / max((clean & known).sum(), 1)),
                                 nsig_p50=float(np.median(d.loc[d[f'{p}_has'], f'{p}_nsig']))))
    R = pd.DataFrame(rows)
    R.to_csv(f'{OUT}/stack_fits.csv', index=False)
    json.dump(stacks, open(f'{OUT}/stacks.json', 'w'))
    with pd.option_context('display.width', 260, 'display.max_columns', 30, 'display.max_rows', 200):
        print(R.round(3).to_string(index=False))


if __name__ == '__main__' and len(sys.argv) == 1:
    main()


def dreco(path=f'{W}/d_b/reco/r145_s0_D_v1.parquet', ref=f'{W}/tanmax/r145_A_tm0.6.parquet'):
    """D's own beam fits (v1_D, run_145 stat090_0000 _000, first 400 seeded events) by trigger and time since the
    flash, with A (tanmax study, v0 bundle, same acquisition) as the reference."""
    TB = triggers_beam()
    TB = TB[TB.subrun == 'stat090_0000']
    rows = []
    for arm, p in (('D', path), ('A', ref)):
        if not os.path.exists(p):
            continue
        e = pd.read_parquet(p).merge(TB, on='event_id', how='inner')
        e['own'] = e[f'hw_{arm}'].astype(bool)
        ms = e.t_since_flash_ns / 1e6
        for lab, m in (('all', np.ones(len(e), bool)), ('own trigger', e.own), ('other trigger', ~e.own),
                       ('< 5 ms', ms < 5), ('5-10 ms', (ms >= 5) & (ms < 10)), ('> 10 ms', ms >= 10),
                       ('own, > 10 ms', e.own & (ms >= 10))):
            g = e[m]
            if not len(g):
                continue
            r = dict(arm=arm, sample=lab, n_events=len(g), frac_track=float((g.n_tracks > 0).mean()))
            for v in 'xy':
                ok = g[f'{v}_ok'].astype(bool)
                gg = g[ok]
                r[f'{v}_ok'] = float(ok.mean())
                r[f'{v}_quality_ok'] = float(g[f'{v}_quality_ok'].astype(bool).mean())
                r[f'{v}_t0_p50'] = float(gg[f'{v}_t0'].median())
                r[f'{v}_t0_w68'] = float(0.5 * np.diff(np.percentile(gg[f'{v}_t0'], [16, 84]))[0]) if len(gg) else np.nan
                r[f'{v}_chi2dof_p50'] = float((gg[f'{v}_chi2'] / gg[f'{v}_dof'].clip(lower=1)).median())
                r[f'{v}_abs_tan_p50'] = float(gg[f'{v}_tan_theta'].abs().median())
                r[f'{v}_isochronous'] = float(gg[f'{v}_isochronous'].astype(bool).mean())
            rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(f'{OUT}/../dbeam/d_beam_reco.csv', index=False)
    with pd.option_context('display.width', 260, 'display.max_columns', 40):
        print(R.round(3).to_string(index=False))


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'dreco':
    dreco()
