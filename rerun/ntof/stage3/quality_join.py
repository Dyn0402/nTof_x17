#!/usr/bin/env python3
"""quality_join.py -- track-quality flags of the n_TOF analysis view, folded into `gated` (tracking audit 2026-11-22).

The last step of the analysis view, after censor_join.py.  It adds to every table:

    amp_norm         the track's strip amplitude, min over the two views of q_sum / n_strips, divided by the median of
                     that same quantity for the run and chamber (late >= 30 ms, wall fired, < 50 % censored, gated
                     tracks: `quality_refs.json`, step `refs`).  1 = a typical ionising track; it follows the gain.
    t0_railed        either view's fitted t0 sits at the reconstruction's hard box, |t0 - pin| >= T0_RAIL_NS (the pin
                     is the bundle's per-ftst t0_abs).  The fit hit its limit: an early arrival reads 14-33 % too steep
                     (reco review 6).  Late A/C: 1.2-1.7 % of tracks; they point at 0.40-0.47 within 15 mm against
                     0.59-0.62 for the others.  A FLAG ONLY, not in gated: it would also remove 0.9 % (A), 1.3 % (C)
                     and 4.3 % (D) of the late pointing, wall-confirmed tracks (run_145).  Decision pending.
    d_faint          chamber D only: amp_norm < D_AMP_MIN or the x window >= D_CENS_X censored.  D's hot-strip fakes
                     are wide and faint (finding 15).  Tuned on late D: keeps 98.7 % of the reference tracks (>= 30 ms,
                     wall, pointing), the 20-30 ms pointing purity rises 0.51 -> 0.75 of the late value.
    low_charge       the A/C post-flash ringing fits (finding 14).  The rule is not chosen yet (the waveform ringing
                     study, rerun/ntof/ringing/); --low-charge none (default) leaves it all False.
    dca_source_mm    closest approach of the track line to the vertical line through the measured source
                     (pass1_tracks.SOURCE_XZ_MM); dca_axis_mm stays, about the nominal axis (finding 22).
    gated_precut     `gated` as censor_join left it
    gated            gated_precut & ~d_faint & ~low_charge

Re-running restarts from gated_precut, so it is idempotent.  B rows get the flags False (no angle, its own gate).

    quality_join.py refs  [--view $WORK_DATA/ntof_pass1/stage3_vt]        -> <view>/quality_refs.json
    quality_join.py apply [--view ...] [--jobs 12] [--low-charge none]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NT = HERE.parent
REPO = NT.parents[1]
sys.path[:0] = [str(HERE), str(NT), str(REPO)]

T0_RAIL_NS = 175.0          # the box is pin +- 180 ns (wft.reco); 5 ns inside it
D_AMP_MIN = 0.25
D_CENS_X = 0.8
REF_MIN_MS = 30.0
REF_CENS_MAX = 0.5
LOW_CHARGE_RULES = ('none',)
FLAGS = ('amp_norm', 't0_railed', 'd_faint', 'low_charge', 'dca_source_mm', 'gated_precut')


def _amp(t):
    with np.errstate(divide='ignore', invalid='ignore'):
        ax = t['x_q_sum'].to_numpy(float) / t['x_n_strips'].to_numpy(float)
        ay = t['y_q_sum'].to_numpy(float) / t['y_n_strips'].to_numpy(float)
    return ax, ay


def refs(view: Path) -> dict:
    """Per (run, arm): the median strip amplitude of each view on late, wall-confirmed, mostly uncensored tracks."""
    import pandas as pd
    import pyarrow.parquet as pq
    acc = {}
    for f in sorted(view.glob('tracks_run_*.parquet')):
        names = pq.read_schema(f).names
        g = 'gated_precut' if 'gated_precut' in names else 'gated'
        cols = ['run', 'arm', g, 't_since_flash_ns', 'x_q_sum', 'y_q_sum', 'x_n_strips', 'y_n_strips',
                'x_cens_frac', 'y_cens_frac'] + [f'wall_{a}' for a in 'ACD' if f'wall_{a}' in names]
        t = pd.read_parquet(f, columns=cols)
        t = t[t[g].fillna(False).astype(bool) & (t['t_since_flash_ns'] >= REF_MIN_MS * 1e6)]
        for arm in ('A', 'C', 'D'):
            a = t[t['arm'] == arm]
            if not len(a) or f'wall_{arm}' not in a:
                continue
            a = a[a[f'wall_{arm}'].astype('boolean').fillna(False).astype(bool)
                  & (np.maximum(a['x_cens_frac'].fillna(0), a['y_cens_frac'].fillna(0)) < REF_CENS_MAX)]
            ax, ay = _amp(a)
            if not len(a):
                continue
            acc.setdefault((str(a['run'].iloc[0]), arm), []).append(np.minimum(ax, ay))
    out = {'rule': f'median of min over views of q_sum / n_strips; gated, t >= {REF_MIN_MS} ms, wall fired, '
                   f'censored < {REF_CENS_MAX}', 'runs': {}, 'arm': {}}
    pooled = {}
    for (run, arm), vs in sorted(acc.items()):
        v = np.concatenate(vs)
        out['runs'].setdefault(run, {})[arm] = dict(amp=float(np.nanmedian(v)), n=int(len(v)))
        pooled.setdefault(arm, []).append(v)
    for arm, vs in pooled.items():
        out['arm'][arm] = dict(amp=float(np.nanmedian(np.concatenate(vs))))
    (view / 'quality_refs.json').write_text(json.dumps(out, indent=1))
    return out


def _pins(rows, canon):
    """{(tag, arm): (t0_abs dict per plane)} from each row's bundle (as ntof_stage3.bundle_for picks it)."""
    import ntof_stage3 as S3
    from wft.calib import CalibrationBundle
    pins, cache = {}, {}
    for r in rows:
        for arm in ('A', 'C', 'D'):
            name = S3.bundle_for(r, arm, canon)
            if name not in cache:
                cal = CalibrationBundle.load(str(S3.BUNDLES / name))
                cache[name] = {p: {int(k): float(v) for k, v in (cal.t0_abs or {}).get(p, {}).items()} for p in 'xy'}
            pins[(r['tag'], arm)] = cache[name]
    return pins


def _source_xz():
    from sept26_prelim_analysis.pass1_tracks import SOURCE_XZ_MM
    return SOURCE_XZ_MM


def dca_source(t):
    from sept26_prelim_analysis.pass1_tracks import SOURCE_XZ_MM
    P = t[['p0_x', 'p0_y', 'p0_z']].to_numpy(float)
    D = t[['d_x', 'd_y', 'd_z']].to_numpy(float)
    S = np.array([SOURCE_XZ_MM[0], 0.0, SOURCE_XZ_MM[1]])
    n = np.c_[-D[:, 2], np.zeros(len(D)), D[:, 0]]           # D x y_hat
    nn = np.linalg.norm(n, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.abs(((S - P) * n).sum(axis=1)) / np.where(nn > 0, nn, np.nan)


def one(path, rows, canon, R, low_rule):
    import pandas as pd
    path = Path(path)
    t = pd.read_parquet(path)
    if 'gated_precut' in t.columns:
        t['gated'] = t['gated_precut']
        t = t.drop(columns=[c for c in FLAGS if c in t.columns and c != 'gated_precut'])
    else:
        t['gated_precut'] = t['gated']
    run = str(t['run'].iloc[0])
    arm = t['arm'].to_numpy()
    ax, ay = _amp(t)
    amp = np.full(len(t), np.nan)
    for a in ('A', 'C', 'D'):
        m = arm == a
        ref = R['runs'].get(run, {}).get(a) or R['arm'].get(a)
        if m.any() and ref:
            amp[m] = np.minimum(ax[m], ay[m]) / ref['amp']
    t['amp_norm'] = amp.astype('float32')
    pins = _pins(rows, canon)
    rail = np.zeros(len(t), bool)
    for p in 'xy':
        t0 = t[f'{p}_t0'].to_numpy(float)
        ft = t[f'{p}_ftst'].fillna(-1).astype(int).to_numpy()
        pin = np.array([pins.get((tg, a), {}).get(p, {}).get(int(f), np.nan) if a != 'B' else np.nan
                        for tg, a, f in zip(t['tag'], arm, ft)])
        with np.errstate(invalid='ignore'):
            rail |= np.abs(t0 - pin) >= T0_RAIL_NS
    t['t0_railed'] = rail
    with np.errstate(invalid='ignore'):
        t['d_faint'] = (arm == 'D') & ((amp < D_AMP_MIN) | (t['x_cens_frac'].fillna(0).to_numpy(float) >= D_CENS_X))
    if low_rule == 'none':
        t['low_charge'] = False
    else:
        raise ValueError(f'unknown --low-charge rule {low_rule!r}')
    t['dca_source_mm'] = dca_source(t).astype('float32')
    cut = t['d_faint'].to_numpy() | t['low_charge'].to_numpy()
    b = arm == 'B'
    cut[b] = False
    t.loc[b, ['t0_railed', 'd_faint', 'low_charge']] = False
    t['gated'] = t['gated_precut'].fillna(False).astype(bool).to_numpy() & ~cut
    tmp = path.with_suffix('.part')
    t.to_parquet(tmp, index=False, compression='snappy')
    os.replace(tmp, path)
    mj = path.with_suffix('.meta.json')
    meta = json.load(open(mj)) if mj.exists() else {}
    by = t.groupby('arm').agg(precut=('gated_precut', 'sum'), gated=('gated', 'sum'), t0_railed=('t0_railed', 'sum'),
                              d_faint=('d_faint', 'sum'), low_charge=('low_charge', 'sum'))
    meta['quality'] = dict(rule=f'gated = gated_precut & ~d_faint (D: amp_norm < {D_AMP_MIN} or x censored >= '
                                f'{D_CENS_X}) & ~low_charge ({low_rule}); t0_railed (|t0 - pin| >= {T0_RAIL_NS} ns) '
                                'is a flag only',
                           source_xz_mm=list(_source_xz()),
                           by_arm={a: {k: int(v) for k, v in r.items()} for a, r in by.iterrows()},
                           built_by='rerun/ntof/stage3/quality_join.py')
    json.dump(meta, open(mj, 'w'), indent=1)
    return path.name, ' '.join(f'{a} {int(r.precut)}->{int(r.gated)}' for a, r in by.iterrows())


def main():
    import ntof_ledger as L
    wd = os.environ.get('WORK_DATA', '/media/ucla/work')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('refs', 'apply'))
    ap.add_argument('--view', default=f'{wd}/ntof_pass1/stage3_vt')
    ap.add_argument('--canon', default=str(NT / 'ledger_canon.json'))
    ap.add_argument('--only', default=None)
    ap.add_argument('--jobs', type=int, default=12)
    ap.add_argument('--low-charge', default='none', choices=LOW_CHARGE_RULES)
    a = ap.parse_args()
    view = Path(a.view)
    if a.step == 'refs':
        R = refs(view)
        print(f"{len(R['runs'])} runs; per arm {R['arm']}")
        return 0
    R = json.loads((view / 'quality_refs.json').read_text())
    canon = L.load_canon(a.canon)
    reg = [r for r in csv.DictReader(open(NT / 'ntof_registry.csv')) if r['excluded'] != 'True']
    groups = {}
    for r in reg:
        groups.setdefault((r['run'], r['subrun']), []).append(r)
    fs = sorted(view.glob(a.only or 'tracks_run_*.parquet'))
    with ProcessPoolExecutor(a.jobs) as ex:
        jobs = {}
        for f in fs:
            run, sub = f.stem.split('_')[2], '_'.join(f.stem.split('_')[3:])
            jobs[ex.submit(one, f, groups[(run, sub)], canon, R, a.low_charge)] = f
        for i, j in enumerate(as_completed(jobs)):
            name, msg = j.result()
            print(f'[{i + 1}/{len(fs)}] {name}: {msg}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
