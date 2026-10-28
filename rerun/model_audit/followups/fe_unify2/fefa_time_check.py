#!/usr/bin/env python3
"""fefa_time_check.py -- is fefa's gain over fe a static v(x, y) map, or the gas changing over the run aliased into space?

FA's v(x, y) map (vgas_<det>.json) was fitted on time-normalised data (v(t) divided out), but fefa applies it WITHOUT
v(t) (the event record has no DREAM timestamp).  The map only rescales the angle after the fit (FA: bundle v +-10 % ->
bit-identical w), so fe and fefa differ only in the angle conversion.  If the map is a real static map, it flattens the
cell-to-cell angle scale equally early and late in the run, and the fefa - fe s68 gain is the same in each time slice.
If the gas pattern moves (a flushing front), the fe cell pattern changes between slices and the map helps one slice
and not another.

Per golden row: the gate sample (8000 M3 muons spanning the run; event_id is trigger order) split into 3 equal-count
time slices.  Per slice and view: s68 of theta_fit - theta_ref for fe and fefa (paired bootstrap of the difference);
the angle scale per 2 x 2 cell (200 mm; robust slope tan_fit / tan_ref, |tan_ref| > 0.1) for fe and fefa, their
spread over cells, and the correlation of fe's cell pattern between slices and with FA's map.
-> results/fefa_time.json / .md
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.expanduser('~/x17/cosmic_bench/model_audit')
T = os.path.join(M, 'fe_unify2', 'gate_tree')
FA = os.path.join(HERE, '..', 'fa_gas_v_tilt', 'results')
ROWS = {'det2': 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2',
        'det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'det4': 'mx17_det4_day_6-24-26/long_run/mx17_4', 'det6': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6',
        'det7': 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7'}
NSL, NB, EDGES = 3, 300, np.array([0.0, 200.0, 400.0])
rng = np.random.default_rng(17)


def s68(r):
    r = r[np.isfinite(r)]
    return float(np.percentile(np.abs(r - np.median(r)), 68.27)) if len(r) > 20 else np.nan


def slope(tr, tf):
    """Robust scale tan_fit / tan_ref: least squares through 0 after a 3-MAD clip of the residual (two passes)."""
    k = (np.abs(tr) > 0.1) & np.isfinite(tf) & np.isfinite(tr)
    tr, tf = tr[k], tf[k]
    if len(tr) < 25:
        return np.nan, np.nan
    s = np.median(tf / tr)
    for _ in range(2):
        r = tf - s * tr
        m = np.abs(r - np.median(r)) < 3 * 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        s = np.sum(tr[m] * tf[m]) / np.sum(tr[m] ** 2)
    e = np.std(tf[m] - s * tr[m]) / np.sqrt(np.sum(tr[m] ** 2))
    return float(s), float(e)


def load(det, arm):
    d = os.path.join(T, ROWS[det], 'wft', arm)
    z = np.load(os.path.join(d, 'angles', 'per_event.npz'))
    e = pd.read_parquet(os.path.join(d, 'events.parquet'), columns=['event_id', 'x_p0', 'y_p0']).drop_duplicates('event_id').set_index('event_id')
    out = {}
    for v in 'xy':
        a = pd.DataFrame({'tr': z[f'{v}_tan_ref'], 'tf': z[f'{v}_tan_fit']}, index=z[f'{v}_eid'])
        a = a[~a.index.duplicated()]
        out[v] = a.join(e, how='inner')
    return out


def cell_of(df):
    ix = np.clip(np.digitize(df.x_p0, EDGES) - 1, 0, 1)
    iy = np.clip(np.digitize(df.y_p0, EDGES) - 1, 0, 1)
    return ix * 2 + iy


def fa_cells(det):
    """FA's v map (4 x 4, 100 mm) averaged onto the 2 x 2 cells (both views share it): vXY[ix][iy]."""
    vm = json.load(open(os.path.join(FA, f'vgas_{det}.json')))['vmap']
    V = np.asarray(vm['vXY'], float)
    return np.array([np.nanmean(V[2 * i:2 * i + 2, 2 * j:2 * j + 2]) for i in range(2) for j in range(2)])


def main():
    res, L = {}, []
    L += ['## fefa vs fe by time slice (gate sample in 3 equal-count thirds of the run, by event_id)', '',
          '| chamber | view | slice | n | s68 fe | s68 fefa | Δ fefa−fe (°) | cell-scale spread fe → fefa (%) |',
          '|---|---|---|---|---|---|---|---|']
    L2 = ['', '## Is fe\'s residual cell pattern (what the v map should remove) stable in time?', '',
          '| chamber | view | corr(slice 1, slice 3) of fe cell scale | rms(slice 3 − slice 1) / stat | corr(fe cell scale, FA vXY) all |',
          '|---|---|---|---|---|']
    for det in ROWS:
        try:
            A, B = load(det, 'fe'), load(det, 'fefa')
        except FileNotFoundError:
            continue
        fav = fa_cells(det)
        for v in 'xy':
            a, b = A[v], B[v]
            com = a.index.intersection(b.index)
            a, b = a.loc[com].sort_index(), b.loc[com].sort_index()
            sl = np.array_split(np.arange(len(a)), NSL)
            ra = np.degrees(np.arctan(a.tf.values) - np.arctan(a.tr.values))
            rb = np.degrees(np.arctan(b.tf.values) - np.arctan(b.tr.values))
            cells = cell_of(a)
            pat = {}
            for k, idx in enumerate(sl + [np.arange(len(a))]):
                lab = f'{k + 1}' if k < NSL else 'all'
                d0, d1 = s68(ra[idx]), s68(rb[idx])
                bs = []
                for _ in range(NB):
                    j = rng.choice(idx, len(idx))
                    bs.append(s68(rb[j]) - s68(ra[j]))
                de = float(np.nanstd(bs))
                sc = {}
                for arm, df in (('fe', a), ('fefa', b)):
                    s = [slope(df.tr.values[idx][cells[idx] == c], df.tf.values[idx][cells[idx] == c]) for c in range(4)]
                    sc[arm] = np.array([x[0] for x in s]); sc[arm + '_err'] = np.array([x[1] for x in s])
                spr = {arm: float(100 * np.nanstd(sc[arm] / np.nanmean(sc[arm]))) for arm in ('fe', 'fefa')}
                pat[lab] = sc
                res[f'{det}|{v}|{lab}'] = dict(n=int(len(idx)), s68_fe=d0, s68_fefa=d1, d=d1 - d0, d_err=de,
                                                cell_fe=sc['fe'].tolist(), cell_fefa=sc['fefa'].tolist(),
                                                cell_fe_err=sc['fe_err'].tolist(), spread_fe=spr['fe'], spread_fefa=spr['fefa'])
                L.append(f'| {det} | {v} | {lab} | {len(idx)} | {d0:.3f} | {d1:.3f} | {d1 - d0:+.3f} ± {de:.3f} | '
                         f'{spr["fe"]:.1f} → {spr["fefa"]:.1f} |')
            p1, p3 = pat['1']['fe'], pat[str(NSL)]['fe']
            e13 = np.hypot(pat['1']['fe_err'], pat[str(NSL)]['fe_err'])
            c13 = float(np.corrcoef(p1, p3)[0, 1])
            rr = float(np.sqrt(np.nanmean(((p3 / np.nanmean(p3) - p1 / np.nanmean(p1)) / e13) ** 2)))
            cfa = float(np.corrcoef(pat['all']['fe'], fav)[0, 1])
            res[f'{det}|{v}|pattern'] = dict(corr13=c13, rms_pull13=rr, corr_fa=cfa, fa_cells=fav.tolist())
            L2.append(f'| {det} | {v} | {c13:+.2f} | {rr:.2f} | {cfa:+.2f} |')
    L2 += ['', 'Cell scale = robust slope tan_fit/tan_ref in 2 × 2 cells of 200 mm (track mesh position p0). rms pull ≈ 1 = '
           'slice-1 and slice-3 patterns equal within statistics (after removing each slice\'s mean, i.e. v(t)).']
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, 'results', 'fefa_time.json'), 'w'), indent=1)
    open(os.path.join(HERE, 'results', 'fefa_time.md'), 'w').write('\n'.join(L + L2) + '\n')
    print('\n'.join(L + L2))


if __name__ == '__main__':
    main()
