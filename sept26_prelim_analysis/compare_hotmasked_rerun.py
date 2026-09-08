#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_hotmasked_rerun.py -- frozen D/run_145 vs a hot-wildcard re-run,
event by event.

Written after the first hot-wildcard condor re-run (2026-09-08) turned out to
make D WORSE, not better -- see HANDOFF_HOT_WILDCARD_TUNING.md. This is the
comparison that found that, pulled out of the one-off shell it was written in
so the next session does not have to re-derive it from scratch.

    python -m sept26_prelim_analysis.compare_hotmasked_rerun \\
        --new /home/dylan/x17/wft_beam145_hotmasked/analysis/run_145/stat090_0000/mx17_D/events_prelim.parquet

**The aggregate comparison this file was born doing is misleading on D, and
``--strata`` is the fix.** D's 42 hot x channels carry 55.6 % of the plane's
hits, and a third of the frozen events' largest x cluster is made ENTIRELY of
them. Those noise columns fit *better* than real tracks -- median chi2/dof
1.27 against 25 for an uncontaminated window, 99.6 % quality_ok -- because a
smooth coherent-noise deposit is easy for the forward model to explain. So
every aggregate number is flattered by them, and a mask that removes them
reads as a catastrophic regression when it is doing its job.

``--strata`` splits both tables by the hot content of the event's baseline
cluster, which is what makes the two comparable: compare like with like, and
the question becomes "did the tracks that were always real get better or
worse", which is answerable. Pass ``--strata`` the parquet written by
``sept26_prelim_analysis/hot_seed_strata.py``.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

FROZEN_DEFAULT = ('/media/dylan/data/x17/sept26_prelim/fullpass/run_145/'
                  'stat090_0000/mx17_D/events_prelim.parquet')


def summarize(df: pd.DataFrame, label: str) -> dict:
    both_ok = df.x_ok & df.y_ok
    m = both_ok
    out = dict(
        label=label, n_rows=len(df),
        x_ok=float(df.x_ok.mean()), y_ok=float(df.y_ok.mean()),
        both_ok=float(both_ok.mean()), n_both_ok=int(both_ok.sum()),
        quality_ok=float((df.x_quality_ok & df.y_quality_ok & both_ok).mean()),
        median_x_n_strips=float(df.loc[m, 'x_n_strips'].median()) if m.any() else np.nan,
        median_y_n_strips=float(df.loc[m, 'y_n_strips'].median()) if m.any() else np.nan,
        median_x_chi2dof=float((df.loc[m, 'x_chi2']
                               / df.loc[m, 'x_dof'].clip(lower=1)).median())
        if m.any() else np.nan)
    if 'x_n_flagged_strips' in df.columns:
        out['frac_x_touches_flagged'] = float((df.x_n_flagged_strips > 0).mean())
        out['frac_y_touches_flagged'] = float((df.y_n_flagged_strips > 0).mean())
        flagged = (df.x_n_flagged_strips > 0) | (df.y_n_flagged_strips > 0)
        for tag, mask in (('flagged', flagged), ('clean', ~flagged)):
            sub = df[mask]
            bo = sub.x_ok & sub.y_ok
            out[f'{tag}_both_ok'] = float(bo.mean()) if len(sub) else np.nan
            out[f'{tag}_n'] = int(len(sub))
    print(f'--- {label} (n_rows={len(df)}) ---')
    print(f'  x_ok {out["x_ok"]:.4f}  y_ok {out["y_ok"]:.4f}  '
         f'both_ok {out["both_ok"]:.4f} (n={out["n_both_ok"]})')
    print(f'  quality_ok (both) {out["quality_ok"]:.4f}')
    print(f'  median x_n_strips {out["median_x_n_strips"]:.1f}  '
         f'y_n_strips {out["median_y_n_strips"]:.1f}')
    print(f'  median x_chi2/dof {out["median_x_chi2dof"]:.2f}')
    if 'frac_x_touches_flagged' in out:
        print(f'  windows touching a flagged strip: x {out["frac_x_touches_flagged"]:.1%}  '
             f'y {out["frac_y_touches_flagged"]:.1%}')
        print(f'  both_ok | flagged window: {out["flagged_both_ok"]:.4f} (n={out["flagged_n"]})'
             f'   both_ok | clean window: {out["clean_both_ok"]:.4f} (n={out["clean_n"]})')
    print()
    return out


def compare_common(frozen: pd.DataFrame, new: pd.DataFrame) -> None:
    common = sorted(set(frozen.event_id) & set(new.event_id))
    f = frozen.set_index('event_id').loc[common]
    n = new.set_index('event_id').loc[common]
    bf, bn = f.x_ok & f.y_ok, n.x_ok & n.y_ok
    gained, lost = (bn & ~bf), (bf & ~bn)
    print(f'common event_ids: {len(common)} (frozen {len(frozen)}, new {len(new)})')
    print(f'  frozen both_ok {bf.mean():.4f} ({int(bf.sum())})   '
         f'new both_ok {bn.mean():.4f} ({int(bn.sum())})')
    print(f'  gained a good fit: {int(gained.sum())}   '
         f'lost a good fit: {int(lost.sum())}')


STRATA_ORDER = ['all-hot', 'mostly-hot', 'clean', 'no cluster']


def compare_by_stratum(frozen: pd.DataFrame, new: pd.DataFrame,
                       strata: pd.DataFrame, plane: str = 'x') -> pd.DataFrame:
    """The comparison that is actually readable on D: like against like.

    Restricted to the events BOTH tables could have seen (the new table's
    events plus the frozen ones it dropped), so a stratum's 'kept' column is a
    real retention and not an artefact of one table being a subset.
    """
    col = f'{plane}_stratum'
    s = strata[['event_id', col]].drop_duplicates('event_id')
    f = frozen.merge(s, on='event_id', how='inner')
    n = new.merge(s, on='event_id', how='inner')
    f['chi2dof'] = f[f'{plane}_chi2'] / f[f'{plane}_dof'].clip(lower=1)
    n['chi2dof'] = n[f'{plane}_chi2'] / n[f'{plane}_dof'].clip(lower=1)
    ok = f'{plane}_ok'

    rows = []
    for st in STRATA_ORDER:
        fs, ns = f[f[col] == st], n[n[col] == st]
        if not len(fs) and not len(ns):
            continue
        rows.append(dict(
            stratum=st, n_frozen=len(fs), n_new=len(ns),
            kept=len(ns) / len(fs) if len(fs) else np.nan,
            frozen_ok=fs[ok].mean() if len(fs) else np.nan,
            new_ok=ns[ok].mean() if len(ns) else np.nan,
            frozen_chi2dof=fs.loc[fs[ok], 'chi2dof'].median() if fs[ok].any() else np.nan,
            new_chi2dof=ns.loc[ns[ok], 'chi2dof'].median() if ns[ok].any() else np.nan,
            frozen_nstrips=fs.loc[fs[ok], f'{plane}_n_strips'].median() if fs[ok].any() else np.nan,
            new_nstrips=ns.loc[ns[ok], f'{plane}_n_strips'].median() if ns[ok].any() else np.nan,
        ))
    out = pd.DataFrame(rows)
    print(f'--- by stratum of the BASELINE {plane} cluster '
          f'(hot content before any mask) ---')
    print(out.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--frozen', default=FROZEN_DEFAULT)
    ap.add_argument('--new', required=True,
                    help='the hot-masked re-run\'s events_prelim.parquet')
    ap.add_argument('--strata', default=None,
                    help='hot_seed_strata.py parquet. Without it you get the '
                         'aggregate numbers, which on D are a mixture and will '
                         'mislead you -- see the module docstring.')
    ap.add_argument('--plane', default='x', choices=('x', 'y'))
    a = ap.parse_args()

    f = pd.read_parquet(a.frozen)
    n = pd.read_parquet(a.new)
    if a.strata:
        common = set(f.event_id) | set(n.event_id)
        f, n = f[f.event_id.isin(common)], n[n.event_id.isin(common)]
        st = pd.read_parquet(a.strata)
        f = f[f.event_id.isin(st.event_id)]
        n = n[n.event_id.isin(st.event_id)]
        compare_by_stratum(f, n, st, a.plane)
    summarize(f, 'FROZEN (no hot wildcards)')
    summarize(n, 'HOT-MASKED')
    compare_common(f, n)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
