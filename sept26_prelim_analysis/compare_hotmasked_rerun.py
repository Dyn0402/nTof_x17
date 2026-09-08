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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--frozen', default=FROZEN_DEFAULT)
    ap.add_argument('--new', required=True,
                    help='the hot-masked re-run\'s events_prelim.parquet')
    a = ap.parse_args()

    f = pd.read_parquet(a.frozen)
    n = pd.read_parquet(a.new)
    summarize(f, 'FROZEN (no hot wildcards)')
    summarize(n, 'HOT-MASKED')
    compare_common(f, n)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
