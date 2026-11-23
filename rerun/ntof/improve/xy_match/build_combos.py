#!/usr/bin/env python3
"""build_combos.py -- every x-y candidate combination of the n_TOF pass-1 reconstruction, with its matching features.

The reconstruction fits the x and y views of a chamber separately (wft.reco._worker_fit) and writes every fitted
candidate to the stage-2 candidates table (up to 5 per view, 6 after a split).  Only then does select_tracks decide
which x goes with which y.  So the x-y matching can be studied, and redone, from the tables alone.

For each sub-run and arm (A, C, D) this writes one row per x-y combination of an event:
  features   tc   column-time difference x - y [ns] (wft.reco.col_dt: t0 + u50, ftst phase-corrected)
             dt0  t0 difference, phase-corrected [ns];  du50, du90, duend  [ns]
             lq   ln(q_x / q_y)
  per view   rank, q_sum, n_strips, tan, p0, chi2/dof, plausible, dchi2, rescued, n_sat/flagged/band strips
  pass 1     pair1: the pass-1 reconstruction paired this x with this y (same track_id >= 0); gated1: that track's gate
  context    t_ms (time since the flash; NaN beam-off / unmatched), wall / plastic fired on this arm, ncx, ncy

and a second table of MIXED combinations: the x of one single-candidate event with the y of the next single-candidate
event of the same tag and arm.  Two particles from different triggers have nothing in common but the trigger timing,
which is what two in-time particles of one capture also share.  So the mixed pairs are the "wrong pairing" sample.

  build_combos.py [--subruns run_145/stat090_0000,...] [--out DIR] [--jobs N]
"""
import argparse, csv, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NT = HERE.parent.parent
REPO = NT.parent.parent
sys.path[:0] = [str(NT), str(NT / 'stage3'), str(REPO)]
import ntof_ledger as L  # noqa: E402
from ntof_stage3 import canonical  # noqa: E402

WD = Path(os.environ.get('WORK_DATA', '/media/ucla/work'))
P = WD / 'ntof_pass1'
ROOTS = {'stage2': P / 'stage2', 'side': P / 'side', 'tests': P / 'tests', 'rerun_dbeam_L1': P / 'rerun_dbeam_L1'}
CTX = P / 'context_r2'
ARMS = ('A', 'C', 'D')
FTST_NS = 10.0
DEFAULT_SUBRUNS = ('run_145/stat090_0000,run_145/stat090_0001,run_145/stat090_0002,run_104/stat090_0003,'
                   'run_139/stat090_0003')
VIEW_COLS = ['rank', 'q_sum', 'n_strips', 'tan_theta', 'p0', 'w', 't0', 'q_u50', 'q_u90', 'q_uend', 'chi2', 'dof',
             'plausible', 'dchi2', 'rescued', 'n_sat_strips', 'n_flagged_strips', 'n_band_strips', 'split_child',
             'ftst', 'track_id', 'track_gated', 'quality_ok']


def context(run, sub):
    tt = pd.read_parquet(CTX / 'trigtime' / f'trigtime_run_{run}_{sub}.parquet')
    tt['t_ms'] = np.where(tt['matched'] == 1, tt['t_pred_ns'] / 1e6, np.nan)
    s1 = pd.read_parquet(CTX / 'stage1' / f'candidates_run_{run}_{sub}.parquet')
    s1 = s1.drop_duplicates('eventId')
    keep = ['eventId'] + [f'{k}_{a}' for k in ('wall', 'plastic', 'coinc') for a in ARMS]
    c = tt[['eventId', 't_ms', 'is_flash']].merge(s1[keep], on='eventId', how='left')
    return c.rename(columns={'eventId': 'event_id'})


def features(X, Y):
    """x-minus-y features for aligned frames X, Y (columns VIEW_COLS)."""
    ux, uy = X['q_u50'].fillna(0).to_numpy(), Y['q_u50'].fillna(0).to_numpy()
    ph = FTST_NS * (X['ftst'].to_numpy() - Y['ftst'].to_numpy())
    ph = np.where((X['ftst'].to_numpy() >= 0) & (Y['ftst'].to_numpy() >= 0), ph, 0.0)
    return dict(tc=(X['t0'].to_numpy() + ux) - (Y['t0'].to_numpy() + uy) + ph,
                dt0=X['t0'].to_numpy() - Y['t0'].to_numpy() + ph,
                du50=X['q_u50'].to_numpy() - Y['q_u50'].to_numpy(),
                du90=X['q_u90'].to_numpy() - Y['q_u90'].to_numpy(),
                duend=X['q_uend'].to_numpy() - Y['q_uend'].to_numpy(),
                lq=np.log(np.maximum(X['q_sum'].to_numpy(), 1.0) / np.maximum(Y['q_sum'].to_numpy(), 1.0)))


def one(run, sub, rows, canon):
    ctx = context(run, sub)
    out, mixed = [], []
    for r in rows:
        for arm in ARMS:
            d, stem = canonical(r, arm, canon, {k: str(v) for k, v in ROOTS.items()})
            f = Path(d) / f'{stem}.candidates.parquet'
            if not f.exists():
                continue
            c = pd.read_parquet(f, columns=['event_id', 'plane'] + VIEW_COLS + ['split_replaced'])
            c = c[(c['rank'] >= 0) & ~c['split_replaced']].drop(columns='split_replaced')
            x, y = c[c.plane == 'x'].drop(columns='plane'), c[c.plane == 'y'].drop(columns='plane')
            m = x.merge(y, on='event_id', suffixes=('_x', '_y'))
            if not len(m):
                continue
            X = m[[f'{k}_x' for k in VIEW_COLS]].set_axis(VIEW_COLS, axis=1)
            Y = m[[f'{k}_y' for k in VIEW_COLS]].set_axis(VIEW_COLS, axis=1)
            F = pd.DataFrame(features(X, Y))
            ncx = x.groupby('event_id').size().rename('ncx')
            ncy = y.groupby('event_id').size().rename('ncy')
            t = pd.concat([m[['event_id']].reset_index(drop=True), F,
                           X.add_prefix('x_').reset_index(drop=True), Y.add_prefix('y_').reset_index(drop=True)],
                          axis=1)
            t = t.join(ncx, on='event_id').join(ncy, on='event_id')
            t['pair1'] = (t['x_track_id'] >= 0) & (t['x_track_id'] == t['y_track_id'])
            t['gated1'] = t['pair1'] & t['x_track_gated']
            t['run'], t['subrun'], t['tag'], t['arm'] = f'run_{run}', sub, r['tag'], arm
            t = t.merge(ctx, on='event_id', how='left')
            t['wall'] = t[f'wall_{arm}'].fillna(False).astype(bool)
            t['plastic'] = t[f'plastic_{arm}'].fillna(False).astype(bool)
            t['coinc'] = t[f'coinc_{arm}']
            t = t.drop(columns=[f'{k}_{a}' for k in ('wall', 'plastic', 'coinc') for a in ARMS])
            out.append(t)
            # mixed: single-candidate events, x of event i with y of event i+1 (same tag, arm)
            s = t[(t.ncx == 1) & (t.ncy == 1)].sort_values('event_id').reset_index(drop=True)
            if len(s) > 2:
                Xs = s[[f'x_{k}' for k in VIEW_COLS]].set_axis(VIEW_COLS, axis=1).iloc[:-1].reset_index(drop=True)
                Ys = s[[f'y_{k}' for k in VIEW_COLS]].set_axis(VIEW_COLS, axis=1).iloc[1:].reset_index(drop=True)
                mx = pd.DataFrame(features(Xs, Ys))
                mx['t_ms'] = s['t_ms'].iloc[:-1].to_numpy()
                mx['t_ms_y'] = s['t_ms'].iloc[1:].to_numpy()
                for k in ('q_sum', 'n_strips', 'tan_theta', 'p0', 'rank'):
                    mx[f'x_{k}'], mx[f'y_{k}'] = Xs[k].to_numpy(), Ys[k].to_numpy()
                mx['wall'] = s['wall'].iloc[:-1].to_numpy() & s['wall'].iloc[1:].to_numpy()
                mx['run'], mx['subrun'], mx['tag'], mx['arm'] = f'run_{run}', sub, r['tag'], arm
                mixed.append(mx)
    T = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    M = pd.concat(mixed, ignore_index=True) if mixed else pd.DataFrame()
    return run, sub, T, M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subruns', default=DEFAULT_SUBRUNS)
    ap.add_argument('--out', default=str(P / 'ana' / 'xy_match'))
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    canon = L.load_canon(str(NT / 'ledger_canon.json'))
    reg = [r for r in csv.DictReader(open(NT / 'ntof_registry.csv')) if r['excluded'] != 'True']
    if a.subruns == 'production':
        want = {(r['run'], r['subrun']) for r in reg if r['rclass'] == 'production'}
    else:
        want = {tuple(s.replace('run_', '').split('/')) for s in a.subruns.split(',')}
    groups = {}
    for r in reg:
        if (r['run'], r['subrun']) in want:
            groups.setdefault((r['run'], r['subrun']), []).append(r)
    out = Path(a.out)
    (out / 'combos').mkdir(parents=True, exist_ok=True)
    (out / 'mixed').mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        fs = [ex.submit(one, run, sub, rows, canon) for (run, sub), rows in sorted(groups.items())]
        for f in as_completed(fs):
            run, sub, T, M = f.result()
            T.to_parquet(out / 'combos' / f'combos_run_{run}_{sub}.parquet', index=False)
            M.to_parquet(out / 'mixed' / f'mixed_run_{run}_{sub}.parquet', index=False)
            print(f'run_{run} {sub}: {len(T)} combos, {int(T.pair1.sum()) if len(T) else 0} pass-1 pairs, '
                  f'{len(M)} mixed', flush=True)


if __name__ == '__main__':
    main()
