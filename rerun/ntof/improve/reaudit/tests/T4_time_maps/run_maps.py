#!/usr/bin/env python3
"""T4 driver part 1: maps (25 mm), corrections, quadratic fits, x vs y, systematics variants, stability splits.
Writes tmaps.npz, results_maps.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tmaps as TM  # noqa: E402

D = pd.read_parquet(HERE / 'cache.parquet', filters=[('arm', 'in', sys.argv[1:] or ['A', 'C', 'D', 'B'])])
ARMS = sys.argv[1:] or ['A', 'C', 'D', 'B']
npz = dict(edges25=TM.E25, edges50=TM.E50, edges100=TM.E100)
R = {}
for arm in ARMS:
    d = TM.prep(D, arm)
    rn = d.run.str.replace('run_', '').astype(int).to_numpy()
    R[arm] = dict(n_tracks=int(len(d)), n_runs=int(len(np.unique(rn))))
    print(arm, len(d), flush=True)
    M = {}
    for s in ('x', 'y', 'xy'):
        o = TM.build(d, arm, s)
        M[s] = o
        npz[f'{arm}_{s}_T'] = np.exp(o['lnmap'])
        npz[f'{arm}_{s}_err'] = o['err']            # error on ln T (= relative error on T)
        npz[f'{arm}_{s}_n'] = o['n']
        R[arm][s] = dict(quad=TM.quad_fit(o['lnmap'], o['err'], TM.E25), quad_all=TM.quad_fit(o['lnmap'], o['err'], TM.E25, 999.), T_med_ns=float(np.exp(o['med'])),
                         per_track_sigma=o['sig'],
                         prof={k: {str(i): float(v) for i, v in p.items()} for k, p in o['prof'].items()})
        print(' ', s, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in R[arm][s]['quad'].items()
                       if k not in ('p', 'pe')}, flush=True)
    R[arm]['x_vs_y'] = TM.diff_stats(M['x']['lnmap'], M['x']['err'], M['y']['lnmap'], M['y']['err'])
    print('  x vs y', R[arm]['x_vs_y'], flush=True)
    # variants (systematics) on the xy series
    var = {'raw': ('table',), 'no_t0': ('tanx', 'tany', 'q', 'table'), 'no_tan': ('t0', 'q', 'table')}
    for name, which in var.items():
        o = TM.build(d, arm, 'xy', which=which)
        npz[f'{arm}_xy_T_{name}'] = np.exp(o['lnmap'])
        R[arm][f'var_{name}'] = TM.diff_stats(M['xy']['lnmap'], M['xy']['err'], o['lnmap'], o['err'])
        R[arm][f'var_{name}']['quad'] = TM.quad_fit(o['lnmap'], o['err'], TM.E25)
    ue = np.fmax(d.x_q_uend.to_numpy(float), d.y_q_uend.to_numpy(float))
    subs = {'not_railed': ue < 1169, 'railed_only': ue >= 1169}
    if arm == 'A':
        subs['no_run79'] = rn != 79
    # t0 map and railed-fraction map (what could fake a T map)
    cell, n = TM.cell_codes(d, TM.E25)
    t0 = 0.5 * (d.x_t0 + d.y_t0).to_numpy(float)
    npz[f'{arm}_t0_map'] = TM.gmed(cell[cell >= 0], t0[cell >= 0], n * n).reshape(n, n)
    npz[f'{arm}_railed_frac'] = (np.bincount(cell[cell >= 0], weights=(ue >= 1169)[cell >= 0], minlength=n * n)
                                 / np.maximum(np.bincount(cell[cell >= 0], minlength=n * n), 1)).reshape(n, n)
    R[arm]['railed_frac'] = float(np.mean(ue >= 1169))
    # stability splits
    order = np.sort(np.unique(rn))
    cum = np.cumsum([np.sum(rn == r) for r in order]) / len(rn)
    first = order[cum <= 0.5]
    idx = {r: i for i, r in enumerate(order)}
    split = {'halves': np.isin(rn, first), 'run_index_parity': np.array([idx[r] % 2 == 0 for r in rn]),
             'run_number_parity': rn % 2 == 0}
    R[arm]['first_half_runs'] = [int(first[0]), int(first[-1])]
    for name, m in list(subs.items()) + [(k + '_a', v) for k, v in split.items()]:
        if m.sum() < 1000 or (~m).sum() < 1000:
            continue
        o1 = TM.build(d[m], arm, 'xy')
        npz[f'{arm}_xy_T_{name}'] = np.exp(o1['lnmap'])
        npz[f'{arm}_xy_err_{name}'] = o1['err']
        if name.endswith('_a'):
            o2 = TM.build(d[~m], arm, 'xy')
            npz[f'{arm}_xy_T_{name[:-2]}_b'] = np.exp(o2['lnmap'])
            npz[f'{arm}_xy_err_{name[:-2]}_b'] = o2['err']
            st = TM.diff_stats(o1['lnmap'], o1['err'], o2['lnmap'], o2['err'], 100, o1['n'], o2['n'])
            st['frac_a'] = float(m.mean())
            st['quad_a'] = TM.quad_fit(o1['lnmap'], o1['err'], TM.E25)
            st['quad_b'] = TM.quad_fit(o2['lnmap'], o2['err'], TM.E25)
            R[arm][f'stab_{name[:-2]}'] = st
            print('  stab', name[:-2], {k: v for k, v in st.items() if not k.startswith('quad')}, flush=True)
        else:
            st = TM.diff_stats(M['xy']['lnmap'], M['xy']['err'], o1['lnmap'], o1['err'], 100, M['xy']['n'], o1['n'])
            st['frac'] = float(m.mean())
            st['quad'] = TM.quad_fit(o1['lnmap'], o1['err'], TM.E25)
            R[arm][f'sub_{name}'] = st
            print('  sub', name, {k: v for k, v in st.items() if k != 'quad'}, flush=True)
    # 50 mm and 100 mm maps for the angle-map comparison (direct, not rebinned medians)
    for tag, edges in (('50', TM.E50), ('100', TM.E100)):
        for s in ('x', 'y', 'xy'):
            o = TM.build(d, arm, s, edges=edges)
            npz[f'{arm}_{s}_T{tag}'] = np.exp(o['lnmap'])
            npz[f'{arm}_{s}_err{tag}'] = o['err']
            npz[f'{arm}_{s}_n{tag}'] = o['n']
    # per-table normalisation: the fitted table offsets against ln v_t (should be ~0 after the v_t division)
    if arm != 'B':
        o = M['xy']
        tab = d.run.astype(str) + '/' + d.subrun.astype(str)
        codes = pd.factorize(tab)[0]
        vt = pd.Series(np.log(d.v_t.to_numpy(float))).groupby(codes).median()
        off = pd.Series(o['prof']['table'])
        k = off.index.intersection(vt.index)
        R[arm]['table_offset_vs_lnvt'] = dict(
            slope=float(np.polyfit(vt[k] - vt[k].median(), off[k], 1)[0]),
            offset_rms_pct=float(100 * off.std()), lnvt_rms_pct=float(100 * vt.std()))
        # without the v_t division: slope of the raw per-table T against v_t
        lt = np.log(TM.series(d, 'xy'))
        rawm = pd.Series(lt).groupby(codes).median()
        R[arm]['raw_table_T_vs_lnvt_slope'] = float(np.polyfit(vt[k] - vt[k].median(), rawm[k] - rawm[k].median(), 1)[0])
        print('  table', R[arm]['table_offset_vs_lnvt'], 'raw slope', R[arm]['raw_table_T_vs_lnvt_slope'], flush=True)

tag = '_'.join(ARMS)
np.savez(HERE / f'tmaps_{tag}.npz', **npz)
json.dump(R, open(HERE / f'results_maps_{tag}.json', 'w'), indent=1, default=float)
