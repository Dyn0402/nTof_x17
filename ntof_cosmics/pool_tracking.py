#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pool_tracking.py -- pool the per-sub-run cosmic pair tables of run_149 and redo
the three numbers of HANDOFF_TRACKING_2026-10-02 §5 on the whole sample.

Pooling is by sample, never by merging directories: `event_id` repeats across
sub-runs, so every row is keyed (subrun, event_id).  Errors are a bootstrap
over SUB-RUNS (the unit that varies with HV, gas and rate), not over pairs.

Writes `results/tracking/pooled/` (summary JSON + CSVs).  Nothing leaves the
package, and nothing is written under /media.

    python ntof_cosmics/pool_tracking.py [--run run_149] [--k-from run_147,run_150]
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

N_BOOT = 500
RNG = np.random.default_rng(20261006)


def subruns(run, k_from):
    d = CT.OUT / f'k_{k_from}'
    return sorted(p.name[len(f'summary_{run}_'):-5] for p in d.glob(f'summary_{run}_*.json'))


def slope_samples(t, P):
    """Per clean opposing pair: (pair, arm, axis, track slope, joined-line slope).
    Same selection and slopes as `cosmic_tracks.slope_check`, but the raw
    samples, so the pooled median and its error can be taken across sub-runs."""
    ti = t.set_index(['event_id', 'arm', 'track_id'])
    rows = []
    for pair, normal, inplane in (('AC', 'z', 'xy'), ('BD', 'x', 'zy')):
        c = P[(P.pair == pair) & (P.sep_mm < CT.CLEAN_SEP_MM)]
        if c.empty:
            continue
        a = ti.loc[list(zip(c.event_id, c.arm1, c.track1))]
        b = ti.loc[list(zip(c.event_id, c.arm2, c.track2))]
        J = {ax: b[f'p0_{ax}'].to_numpy() - a[f'p0_{ax}'].to_numpy() for ax in 'xyz'}
        for arm, tr in ((pair[0], a), (pair[1], b)):
            for ax in inplane:
                rows.append(pd.DataFrame(dict(
                    arm=arm, axis=ax,
                    s=tr[f'd_{ax}'].to_numpy() / tr[f'd_{normal}'].to_numpy(),
                    j=J[ax] / J[normal])))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def ratio_stats(df):
    m = df[np.abs(df.j) > 0.1]
    if m.empty:
        return np.nan, np.nan, np.nan, 0
    return (float(np.median(m.s / m.j)), float(np.sum(m.s * m.j) / np.sum(m.j ** 2)),
            float(np.corrcoef(m.s, m.j)[0, 1]) if len(m) > 2 else np.nan, len(m))


def boot(groups, fn):
    """groups: list of per-subrun DataFrames; fn(concat) -> float."""
    keep = [g for g in groups if len(g)]
    if len(keep) < 3:
        return np.nan
    v = []
    for _ in range(N_BOOT):
        v.append(fn(pd.concat([keep[i] for i in RNG.integers(0, len(keep), len(keep))])))
    return float(np.nanstd(v))


def pool(run, k_from):
    subs = subruns(run, k_from)
    d = CT.OUT / f'k_{k_from}'
    tot = dict(n_triggers=0, n_ge1=0, n_ge2=0)
    pairs, samples, per_sub = [], [], []
    for s in subs:
        sm = json.loads((d / f'summary_{run}_{s}.json').read_text())
        tot['n_triggers'] += sm['n_triggers']
        tot['n_ge1'] += sm['n_trig_ge1_track']
        tot['n_ge2'] += sm['n_trig_ge2_arms']
        P = pd.read_parquet(d / f'pairs_{run}_{s}.parquet')
        P.insert(0, 'subrun', s)
        pairs.append(P)
        t = pd.read_parquet(CT.tracks_path(run, s, k_from))
        ss = slope_samples(t, P)
        if len(ss):
            ss.insert(0, 'subrun', s)
            samples.append(ss)
        per_sub.append(dict(subrun=s, n_triggers=sm['n_triggers'],
                            n_ge1=sm['n_trig_ge1_track'], n_ge2=sm['n_trig_ge2_arms'],
                            n_clean=sm['n_opposing_clean']))
    P = pd.concat(pairs, ignore_index=True)
    S = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()

    # one pair per (subrun, trigger): the most collinear
    best = P.sort_values('open_deg', ascending=False).drop_duplicates(['subrun', 'event_id'])
    opp = best[best.topo == 'opposing']
    clean = opp[opp.sep_mm < CT.CLEAN_SEP_MM]

    def f170(b):
        return float((b.open_deg > CT.BACK_TO_BACK_DEG).mean()) if len(b) else np.nan

    bysub = lambda b: [g for _, g in b.groupby('subrun')]  # noqa: E731
    out = dict(
        run=run, k_from=k_from, n_subruns=len(subs), **tot,
        frac_ge1=tot['n_ge1'] / tot['n_triggers'], frac_ge2=tot['n_ge2'] / tot['n_triggers'],
        trig_by_pair=best.pair.value_counts().to_dict(),
        n_opposing=int(len(opp)), n_opposing_clean=int(len(clean)),
        frac_opposing_above_170=f170(opp), frac_opposing_above_170_err=boot(bysub(opp), f170),
        frac_clean_above_170=f170(clean), frac_clean_above_170_err=boot(bysub(clean), f170),
        clean_open_quantiles={str(q): float(clean.open_deg.quantile(q))
                              for q in (0.05, 0.16, 0.5, 0.84, 0.95)} if len(clean) else {},
    )
    rows = []
    if len(S):
        for (arm, ax), g in S.groupby(['arm', 'axis']):
            med, lsq, corr, n = ratio_stats(g)
            err = boot(bysub(g), lambda x: ratio_stats(x)[0])
            rows.append(dict(arm=arm, axis=ax, n=n, median_ratio=med, median_err=err,
                             lsq_ratio=lsq, corr=corr))
    out['slope_ratio'] = rows
    return out, pd.DataFrame(per_sub), P, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='run_149')
    ap.add_argument('--k-from', default='run_147,run_150')
    a = ap.parse_args()
    dest = CT._guard(CT.OUT / 'pooled')
    dest.mkdir(parents=True, exist_ok=True)
    res = {}
    for k in a.k_from.split(','):
        out, per_sub, P, S = pool(a.run, k)
        res[k] = out
        per_sub.to_csv(dest / f'per_subrun_{a.run}_k{k}.csv', index=False)
        P.to_parquet(dest / f'pairs_{a.run}_k{k}.parquet', index=False)
        S.to_parquet(dest / f'slope_samples_{a.run}_k{k}.parquet', index=False)
        print(f'--- k from {k}')
        print(json.dumps(out, indent=1, default=str))
    (dest / f'pooled_{a.run}.json').write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
