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
    low_charge       the A/C post-flash ringing fits (finding 14).  --low-charge none (default) leaves it all False.
                     q600: min(x_q_sum, y_q_sum) < 600 ADC (usable on pass 1).  r28 (Dylan's rule of 2026-11-23,
                     roadmap 2.8): the track fails the waveform test (min over the views of n_sig5 < 1; n_sig5 = strips
                     of the fit window above 5 sigma, wft.reco.window_sig, written from the re-audit code on and
                     meaningful with the ringing correction on), or it is before 8 ms and min q < 600 ADC.  Tracks
                     with no time since the flash take the late rule.  r28 refuses a table without n_sig5.
    d_hole           chamber D, 0 <= x_local <= 60 mm: D's efficiency there is 9-13 % of the neighbouring regions for
                     the whole campaign (dead strips of FEU 1 connector 4; re-audit 2026-11-24, W2b).  A FLAG ONLY:
                     the tracks are real, but an efficiency or position-shape analysis of D must exclude the region
                     or use an efficiency that changes at run_133 (five channels hotter after the 4 August pedestal).
    b_box            chamber B, x_local -52..-44 mm and y_local -200..-160 mm, runs up to run_132: a noisy channel
                     group (B-x channels 258, 259 and B-y channels 13-63) makes fakes there (17.8 % of B's late gated
                     tracks through run_106).  A flag; --b-box-cut also removes them from `gated`.
    dtc_ns           the x-y column-time difference the reco's coincidence test uses (wft.reco.col_dt: t0 + u50 per view,
                     ftst phase-corrected).
    xy_time_ok       |dtc_ns| <= the n_TOF p98 of clean late single tracks (TC_TOL_NTOF: A 151, C 296, D 535 ns).  The
                     reco used the bench tolerance (197, 456, 726 ns), which passes 70-89 % of unrelated late x-y
                     pairs (reco review 4).  A flag only.
    xy_balance_ok    A, C: |ln(qx/qy) - run median| <= 4 sigma(t), sigma(t) the x-y charge-balance width of pointing
                     tracks at that time after the flash (0.40 at 1.5 ms to 0.07 (A) / 0.11 (C) late; the ringing
                     widens it).  Mis-paired views of two particles fail it.  A flag only (costs 2-3 % of good late
                     tracks; purity at 12-16 ms 0.82 -> 0.85).  D: True.
    dca_source_mm    closest approach of the track line to the vertical line through the measured source
                     (pass1_tracks.SOURCE_XZ_MM); dca_axis_mm stays, about the nominal axis (finding 22).
    gated_precut     `gated` as censor_join left it
    gated            gated_precut & ~d_faint & ~low_charge (& ~b_box with --b-box-cut)

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
LOW_CHARGE_RULES = ('none', 'q600', 'r28')
LOW_Q_ADC = 600.0          # finding 14: keeps 99.7 % (A) / 98.7 % (C) of the reference tracks
LOW_Q_BEFORE_MS = 8.0      # r28: the charge cut applies before this time since the flash
D_HOLE_X = (0.0, 60.0)     # x_local [mm]
B_BOX = dict(x=(-52.0, -44.0), y=(-200.0, -160.0), last_run=132)
FLAGS = ('amp_norm', 't0_railed', 'd_faint', 'low_charge', 'd_hole', 'b_box', 'dca_source_mm', 'dtc_ns', 'xy_time_ok',
         'xy_balance_ok', 'gated_precut')
#: n_TOF x-y column-time tolerance [ns]: p98 of |dtc| on late (>= 30 ms), wall-confirmed, single, pointing tracks of
#: runs 139-154 (2026-11-22).  The bench values in the bundles: A 197, C 456, D 726.
TC_TOL_NTOF = {'A': 151.0, 'C': 296.0, 'D': 535.0}
#: x-y charge-balance width sigma(t) of pointing tracks vs ToF [ms] (low-charge study 2026-11-22, run-median subtracted)
BAL_T_MS = (1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 13.0, 15.0, 17.0, 19.0, 22.5, 27.5, 35.0, 57.5)
BAL_SIG = {'A': (0.404, 0.243, 0.158, 0.149, 0.150, 0.126, 0.113, 0.098, 0.090, 0.082, 0.080, 0.076, 0.072, 0.072,
                 0.072, 0.072, 0.071, 0.071, 0.071),
           'C': (0.408, 0.239, 0.182, 0.174, 0.175, 0.172, 0.157, 0.144, 0.133, 0.131, 0.121, 0.117, 0.117, 0.114,
                 0.114, 0.113, 0.113, 0.113, 0.113)}
BAL_NSIG = 4.0


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
            with np.errstate(divide='ignore', invalid='ignore'):
                lb = np.log(a['x_q_sum'].to_numpy(float) / a['y_q_sum'].to_numpy(float))
            acc.setdefault((str(a['run'].iloc[0]), arm), []).append((np.minimum(ax, ay), lb))
    out = {'rule': f'median of min over views of q_sum / n_strips; gated, t >= {REF_MIN_MS} ms, wall fired, '
                   f'censored < {REF_CENS_MAX}', 'runs': {}, 'arm': {}}
    pooled = {}
    for (run, arm), vs in sorted(acc.items()):
        v = np.concatenate([x for x, _ in vs])
        lb = np.concatenate([y for _, y in vs])
        out['runs'].setdefault(run, {})[arm] = dict(amp=float(np.nanmedian(v)), lnbal=float(np.nanmedian(lb)),
                                                    n=int(len(v)))
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


def one(path, rows, canon, R, low_rule, b_box_cut=False):
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
    ac = (arm == 'A') | (arm == 'C')
    with np.errstate(invalid='ignore'):
        qlow = np.fmin(t['x_q_sum'].to_numpy(float), t['y_q_sum'].to_numpy(float)) < LOW_Q_ADC
    if low_rule == 'none':
        t['low_charge'] = False
    elif low_rule == 'q600':
        t['low_charge'] = ac & qlow
    elif low_rule == 'r28':
        if not {'x_n_sig5', 'y_n_sig5'} <= set(t.columns):
            raise ValueError(f'{path.name}: --low-charge r28 needs x_n_sig5 / y_n_sig5 (a reconstruction from the '
                             're-audit code on); use q600 on pass 1')
        ns = np.fmin(t['x_n_sig5'].to_numpy(float), t['y_n_sig5'].to_numpy(float))
        if ac.any() and not (ns[ac] >= 0).any():
            raise ValueError(f'{path.name}: n_sig5 is unset (-1) on every A/C track')
        t_ms = t['t_since_flash_ns'].to_numpy(float) / 1e6
        with np.errstate(invalid='ignore'):
            early = t_ms < LOW_Q_BEFORE_MS              # NaN (no flash time) -> False: the late rule
        t['low_charge'] = ac & (~(ns >= 1) | (early & qlow))
    else:
        raise ValueError(f'unknown --low-charge rule {low_rule!r}')
    xl, yl = t['x_local'].to_numpy(float), t['y_local'].to_numpy(float)
    with np.errstate(invalid='ignore'):
        t['d_hole'] = (arm == 'D') & (xl >= D_HOLE_X[0]) & (xl <= D_HOLE_X[1])
        run_no = int(''.join(ch for ch in run if ch.isdigit()) or 0)
        t['b_box'] = ((arm == 'B') & (run_no <= B_BOX['last_run']) & (xl >= B_BOX['x'][0]) & (xl <= B_BOX['x'][1])
                      & (yl >= B_BOX['y'][0]) & (yl <= B_BOX['y'][1]))
    t['dca_source_mm'] = dca_source(t).astype('float32')
    dtc = ((t['x_t0'] + t['x_q_u50'].fillna(0)) - (t['y_t0'] + t['y_q_u50'].fillna(0))
           + 10.0 * (t['x_ftst'] - t['y_ftst'])).to_numpy(float)
    t['dtc_ns'] = dtc.astype('float32')
    tol = np.array([TC_TOL_NTOF.get(a, np.inf) for a in arm])
    t['xy_time_ok'] = np.abs(dtc) <= tol
    with np.errstate(divide='ignore', invalid='ignore'):
        lb = np.log(t['x_q_sum'].to_numpy(float) / t['y_q_sum'].to_numpy(float))
    ms = t['t_since_flash_ns'].to_numpy(float) / 1e6
    bal = np.ones(len(t), bool)
    for a in ('A', 'C'):
        m = arm == a
        ref = R['runs'].get(run, {}).get(a, {}).get('lnbal')
        if not m.any() or ref is None:
            continue
        sig = np.interp(np.where(np.isfinite(ms[m]), ms[m], 75.0), BAL_T_MS, BAL_SIG[a])
        bal[m] = np.abs(lb[m] - ref) <= BAL_NSIG * sig
    t['xy_balance_ok'] = bal
    cut = t['d_faint'].to_numpy() | t['low_charge'].to_numpy()
    b = arm == 'B'
    cut[b] = False
    t.loc[b, ['t0_railed', 'd_faint', 'low_charge']] = False
    if b_box_cut:
        cut = cut | t['b_box'].to_numpy()
    t['gated'] = t['gated_precut'].fillna(False).astype(bool).to_numpy() & ~cut
    tmp = path.with_suffix('.part')
    t.to_parquet(tmp, index=False, compression='snappy')
    os.replace(tmp, path)
    mj = path.with_suffix('.meta.json')
    meta = json.load(open(mj)) if mj.exists() else {}
    by = t.groupby('arm').agg(precut=('gated_precut', 'sum'), gated=('gated', 'sum'), t0_railed=('t0_railed', 'sum'),
                              d_faint=('d_faint', 'sum'), low_charge=('low_charge', 'sum'),
                              d_hole=('d_hole', 'sum'), b_box=('b_box', 'sum'))
    meta['quality'] = dict(rule=f'gated = gated_precut & ~d_faint (D: amp_norm < {D_AMP_MIN} or x censored >= '
                                f'{D_CENS_X}) & ~low_charge ({low_rule})' + (' & ~b_box' if b_box_cut else '')
                                + f'; t0_railed (|t0 - pin| >= {T0_RAIL_NS} ns), d_hole and '
                                + ('b_box are' if not b_box_cut else 'd_hole are') + ' flags only',
                           low_charge_rule=low_rule, b_box_cut=bool(b_box_cut),
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
    ap.add_argument('--b-box-cut', action='store_true', help='remove the chamber-B noisy box (b_box) from gated')
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
            jobs[ex.submit(one, f, groups[(run, sub)], canon, R, a.low_charge, a.b_box_cut)] = f
        for i, j in enumerate(as_completed(jobs)):
            name, msg = j.result()
            print(f'[{i + 1}/{len(fs)}] {name}: {msg}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
