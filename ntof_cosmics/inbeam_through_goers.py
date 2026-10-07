#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inbeam_through_goers.py -- cosmic muons crossing A and C DURING beam runs: does
the beam environment (gain, noise, space charge) change A's angle scale?
HANDOFF_TRACKING_2026-10-06.md §10e.

Same particle (muon), same truth (the A-C line through the two chambers' track
points, as `insitu_calib.truth`), same estimator, beam-on against beam-off
(run_149).  Selection: exactly one gated track in A and in C; the two lines
within SEP of each other; the joined line > 60 mm from the beam axis (no
capsule pairs); beam runs at >= 20 ms after the flash (0-10 ms is flash-
correlated junk).  Scale = median(true tan / raw tan), 0.1 < |raw| < 0.6.

    python ntof_cosmics/inbeam_through_goers.py            # build pairs + the beam-off/beam-on table
    python ntof_cosmics/inbeam_through_goers.py pooled     # per period, and pooled gas-normalised (§13)

`pooled`: each beam muon's raw tan is put on run REF_RUN's gas by the per-run
k_arm band ratio (k_run / k_ref), then all 20-80 ms pairs are pooled and the
scale is compared with run_149 at the same sep cut: what the in-situ bundle
(calibrated on run_149) reads on beam-period muons.  Arms A and C, x and y.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ntof_cosmics.cosmic_tracks import _axis_dca  # noqa: E402
from sept26_prelim_analysis.source_imaging import _dca_two_lines  # noqa: E402

CAMPAIGN = Path('/media/dylan/data/x17/sept26_prelim/stage3_fullpass/tracks_campaign.parquet')
COSMIC = HERE / 'results' / 'tracking' / 'k_run_147'
OUT = Path('/media/dylan/data/x17/ntof_cosmics/inbeam_through_goers')
COLS = ['run', 'subrun', 'event_id', 'arm', 'track_id', 'gated', 'p0_x', 'p0_y', 'p0_z',
        'd_x', 'd_y', 'd_z', 'tan_raw_x', 'tan_raw_y', 'q_per_len', 't_since_flash_ns']
JDCA_MIN = 60.0
KCAL = Path('/media/dylan/data/x17/sept26_prelim/kcal')
REF_RUN = 145
PERIODS = {'79-98': (79, 98), '100-124': (100, 124), '128-147': (128, 147), '150-162': (150, 162)}


def ac_pairs(t: pd.DataFrame) -> pd.DataFrame:
    t = t[t.gated.astype(bool) & t.arm.isin(['A', 'C'])]
    k = [c for c in ('run', 'subrun', 'event_id') if c in t]
    n = t.groupby(k + ['arm']).size().unstack(fill_value=0)
    ok = n[(n.get('A', 0) == 1) & (n.get('C', 0) == 1)].index
    t = t.set_index(k)
    t = t[t.index.isin(ok)].reset_index()
    a = t[t.arm == 'A'].sort_values(k).reset_index(drop=True)
    b = t[t.arm == 'C'].sort_values(k).reset_index(drop=True)
    P1, D1 = a[['p0_x', 'p0_y', 'p0_z']].to_numpy(float), a[['d_x', 'd_y', 'd_z']].to_numpy(float)
    P2, D2 = b[['p0_x', 'p0_y', 'p0_z']].to_numpy(float), b[['d_x', 'd_y', 'd_z']].to_numpy(float)
    _V, sep = _dca_two_lines(P1, D1, P2, D2)
    J = P2 - P1
    J /= np.linalg.norm(J, axis=1)[:, None]
    out = a[k + ['tan_raw_x', 'tan_raw_y', 'q_per_len']].copy()
    out['q_C'] = b.q_per_len.to_numpy()
    out['tan_raw_x_C'] = b.tan_raw_x.to_numpy()   # C's own raw tans, for C's scale (sign fixed in scale())
    out['tan_raw_y_C'] = b.tan_raw_y.to_numpy()
    out['sep'] = sep
    out['jdca'] = _axis_dca(P1, J)[0]
    out['jx'] = (P2[:, 0] - P1[:, 0]) / (P2[:, 2] - P1[:, 2])   # A's local x sign is +1 (corr 0.89)
    out['jy'] = (P2[:, 1] - P1[:, 1]) / (P2[:, 2] - P1[:, 2])   # sign fixed per sample in scale()
    out['vert_deg'] = np.degrees(np.arccos(np.abs(J[:, 1])))
    out['ms'] = a.t_since_flash_ns.to_numpy() / 1e6 if 't_since_flash_ns' in a else np.nan
    return out


def scale(d: pd.DataFrame, view: str = 'x') -> dict:
    d = d[(d[f'tan_raw_{view}'].abs() > 0.1) & (d[f'tan_raw_{view}'].abs() < 0.6)]
    j, r = d[f'j{view}'].to_numpy(), d[f'tan_raw_{view}'].to_numpy()
    j = j * np.sign(np.corrcoef(j, r)[0, 1])   # local-vs-global sign (x: +1)
    rng = np.random.default_rng(0)
    bs = [np.median((j / r)[rng.integers(0, len(d), len(d))]) for _ in range(200)]
    return dict(n=len(d), median_ratio=float(np.median(j / r)), err=float(np.std(bs)),
                regression=float(np.sum(j * j) / np.sum(j * r)),
                q_A=float(d.q_per_len.median()))


def k_band() -> dict:
    """{run number: {arm: k_arm band}} from the campaign k_arm JSONs."""
    import json
    import re
    k = {}
    for f in KCAL.glob('k_arm_run_*.json'):
        if not re.search(r'run_\d+\.json$', f.name):
            continue
        d = json.loads(f.read_text())
        k[int(d['run'][4:])] = {a: d['arms'].get(a, {}).get('per_estimator', {}).get('band', np.nan)
                                for a in 'ABCD'}
    return k


def _ratio(j, r, nboot=300):
    m = (np.abs(r) > 0.1) & (np.abs(r) < 0.6) & np.isfinite(r) & np.isfinite(j)
    j, r = j[m], r[m]
    j = j * np.sign(np.corrcoef(j, r)[0, 1])
    x = j / r
    rng = np.random.default_rng(0)
    return float(np.median(x)), float(np.std([np.median(x[rng.integers(0, len(x), len(x))])
                                              for _ in range(nboot)])), int(len(x))


def pooled() -> int:
    B = pd.read_parquet(OUT / 'pairs_beam.parquet')
    C = pd.read_parquet(OUT / 'pairs_run149.parquet')
    B['rn'] = B.run.astype(str).str.extract(r'(\d+)')[0].astype(float)
    k = k_band()
    rows = []
    for sp in (10, 6):
        cb = (C.sep < sp) & (C.jdca > JDCA_MIN)
        bb = (B.sep < sp) & (B.jdca > JDCA_MIN) & (B.ms >= 20) & (B.ms < 80)
        for arm in 'AC':
            kr = B.rn.map(lambda r: k.get(int(r), {}).get(arm, np.nan) if np.isfinite(r) else np.nan)
            for view in 'xy':
                col = f'tan_raw_{view}' + ('' if arm == 'A' else '_C')
                ref = _ratio(C[f'j{view}'][cb].to_numpy(), C[col][cb].to_numpy())
                row = dict(sep_max=sp, arm=arm, view=view, run149=ref[0], run149_err=ref[1])
                for nm, (lo, hi) in PERIODS.items():
                    m = bb & (B.rn >= lo) & (B.rn <= hi)
                    v, e, n = _ratio(B[f'j{view}'][m].to_numpy(), B[col][m].to_numpy())
                    row.update({f'p{nm}': v, f'p{nm}_err': e, f'p{nm}_n': n,
                                f'k{nm}': float(np.nanmedian([k[r][arm] for r in k if lo <= r <= hi]))})
                m = bb & np.isfinite(kr)
                v, e, n = _ratio(B[f'j{view}'][m].to_numpy(),
                                 (B[col] * kr / k[REF_RUN][arm])[m].to_numpy())
                row.update(pooled=v, pooled_err=e, pooled_n=n,
                           norm=v / ref[0], norm_err=float(np.hypot(e / ref[0], v * ref[1] / ref[0] ** 2)))
                rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'pooled_norm.csv', index=False)
    with pd.option_context('display.width', 250):
        print(R[['sep_max', 'arm', 'view', 'run149', 'pooled', 'pooled_n', 'norm', 'norm_err']
                + [f'p{p}' for p in PERIODS]].round(3).to_string(index=False))
    print(f'-> {OUT / "pooled_norm.csv"}  (norm = what is2 reads on beam muons at run_{REF_RUN} gas)')
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == 'pooled':
        return pooled()
    OUT.mkdir(parents=True, exist_ok=True)
    B = ac_pairs(pq.read_table(CAMPAIGN, columns=COLS, filters=[('arm', 'in', ['A', 'C']),
                                                               ('gated', '==', True)]).to_pandas())
    C = ac_pairs(pd.concat(pd.read_parquet(f, columns=[c for c in COLS if c not in ('run', 't_since_flash_ns')])
                           for f in glob.glob(str(COSMIC / 'tracks_run_149_*.parquet'))))
    B.to_parquet(OUT / 'pairs_beam.parquet', index=False)
    C.to_parquet(OUT / 'pairs_run149.parquet', index=False)
    rows = []
    for view in 'xy':
        for sp in (60, 20, 10):
            rows.append(dict(view=view, sample='run_149 (beam off)', ms='-', sep_max=sp,
                             **scale(C[(C.sep < sp) & (C.jdca > JDCA_MIN)], view)))
            for lo, hi in ((10, 20), (20, 80), (40, 80)):
                rows.append(dict(view=view, sample='beam runs', ms=f'{lo}-{hi}', sep_max=sp,
                                 **scale(B[(B.sep < sp) & (B.jdca > JDCA_MIN) & (B.ms >= lo) & (B.ms < hi)], view)))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'scales.csv', index=False)
    print(R.round(3).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
