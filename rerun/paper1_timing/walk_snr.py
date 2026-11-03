#!/usr/bin/env python3
"""walk_snr.py --rows DIR --events DIR [--out results/walk_snr.json] -- Paper I §7, Dylan 2026-11-03: "(b) is mostly
saying that the timing gain comes from the increase in signal amplitude, right? Is that something a different timing
algorithm could correct for?"  Amplitude walk (a shift of the mean time with the signal size: correctable) or a
signal-to-noise jitter (a width that falls with the signal size: not correctable by any amplitude correction)?

Chamber A only, from timing_row.py's per-trigger times (edge25 column start, X and Y, and the early amplitude = strip sum
at the edge's cut sample, X + Y):
  1. Fixed HV: Runs 1+2 (490 V mesh, 1000 V drift; M3-matched single tracks, matched_rows.py's sample) binned in the
     per-track early amplitude (Landau fluctuations at one gain), against the same bins on the pooled 6-27 HV-scan rows
     (amplitude changed by the gain) and the pooled drift-scan rows (amplitude changed by the drift velocity).
  2. Walk: median event time and median t_X - t_Y per amplitude bin (Runs 1+2); the widths after subtracting that
     median curve (what any amplitude-walk correction can do at best).
  3. Drift scan (490 V): the matched widths of rows_matched.csv in ns and in drift distance (x the timing v of the row,
     rerun/paper1_gas/results/gas.json scan points, in30 head-on stack), with the common part
     sqrt(sigma_evt^2 - sigma_xy^2 / 2).
Per-row times carry each row's own median removed (rows differ in delay).  s68 = half the 16-84 % range.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
GOLD = {'sat_det3': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3',
        'c26__mx17_det3_p2_det1_overnight_6-27-26__long_run_p2_det1_sanity_check__mx17_3':
            'mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check/mx17_3'}
EDGES = np.geomspace(300, 20000, 16)
MIN_N = 80
GAP_CM = 2.819        # pg_gas det3 gap (28.19 mm): E of the gas_vE figure


def s68(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 30:
        return np.nan
    q = np.percentile(a, [16, 84])
    return float((q[1] - q[0]) / 2)


def load(rows, key, events=None):
    z = np.load(os.path.join(rows, key + '.npz')); tags = list(z['acq_tags'])
    T = pd.DataFrame(dict(acq_tag=[tags[i] for i in z['acq']], event_id=z['eid'], tx=z['tx'].astype(float),
                          ty=z['ty'].astype(float), amp=z['amp'].astype(float)))
    if events is not None:
        E = pd.read_parquet(events, columns=['acq_tag', 'event_id', 'x_ok', 'y_ok', 'n_tracks'])
        T = T.merge(E, on=['acq_tag', 'event_id'])
        T = T[T.x_ok & T.y_ok & (T.n_tracks == 1)]
    m = np.nanmedian((T.tx + T.ty) / 2)
    return T.assign(tx=T.tx - m, ty=T.ty - m)


def curve(T, rng=None, nboot=100):
    a = T.amp.to_numpy(); D = (T.tx - T.ty).to_numpy(); M = ((T.tx + T.ty) / 2).to_numpy()
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = (a >= lo) & (a < hi)
        if s.sum() < MIN_N:
            continue
        r = dict(amp=float(np.median(a[s])), n=int(s.sum()), sigma_xy=s68(D[s]) / np.sqrt(2), sigma_evt=s68(M[s]),
                 med_evt=float(np.median(M[s])), med_xy=float(np.median(D[s] - np.median(D))))
        if rng is not None:
            idx = np.flatnonzero(s)
            bs = [rng.choice(idx, len(idx)) for _ in range(nboot)]
            r['sigma_xy_err'] = float(np.std([s68(D[i]) / np.sqrt(2) for i in bs]))
            r['sigma_evt_err'] = float(np.std([s68(M[i]) for i in bs]))
            r['med_evt_err'] = float(np.std([np.median(M[i]) for i in bs]))
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', required=True, help='timing_row.py outputs (<key>.npz), EOS x17rerun/paper1_timing/rows')
    ap.add_argument('--events', required=True, help='Analysis_pass1 copy: <run>/<subrun>/mx17_3/wft/events.parquet')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'walk_snr.json'))
    a = ap.parse_args()
    rng = np.random.default_rng(3)
    G = pd.concat([load(a.rows, k, os.path.join(a.events, rel, 'wft', 'events.parquet')) for k, rel in GOLD.items()])
    hv, dr = [], []
    for f in sorted(glob.glob(os.path.join(a.rows, 'c26__mx17_det3_saturday_scan_6-27-26__*__mx17_3.npz'))):
        sub = os.path.basename(f).split('__')[2]
        T = load(a.rows, os.path.basename(f)[:-4])
        (hv if sub.startswith('hv_scan') else dr if sub.startswith('drift_scan') else []).append(T)
    res = dict(fixed_hv=curve(G, rng), hv_scan_pooled=curve(pd.concat(hv)), drift_scan_pooled=curve(pd.concat(dr)),
               n_fixed_hv=int(len(G)))
    # walk: subtract the binned median curve (in log amplitude) -- the best any amplitude-walk correction can do
    c = res['fixed_hv']
    la = np.log(G.amp.to_numpy()); xc = np.log([r['amp'] for r in c])
    M = ((G.tx + G.ty) / 2).to_numpy(); D = (G.tx - G.ty).to_numpy() - np.median(G.tx - G.ty)
    Mc = M - np.interp(la, xc, [r['med_evt'] for r in c]); Dc = D - np.interp(la, xc, [r['med_xy'] for r in c])
    q = np.percentile(G.amp, [0, 25, 50, 75, 100])
    res['walk'] = dict(sigma_evt=s68(M), sigma_evt_walk_corrected=s68(Mc), sigma_xy=s68(D) / np.sqrt(2),
                       sigma_xy_walk_corrected=s68(Dc) / np.sqrt(2),
                       med_evt_span_ns=float(max(r['med_evt'] for r in c) - min(r['med_evt'] for r in c)),
                       med_xy_span_ns=float(max(r['med_xy'] for r in c) - min(r['med_xy'] for r in c)),
                       quartiles=[dict(lo=float(q[i]), hi=float(q[i + 1]),
                                       sigma_xy=s68(D[(G.amp >= q[i]) & (G.amp <= q[i + 1])]) / np.sqrt(2),
                                       sigma_evt=s68(M[(G.amp >= q[i]) & (G.amp <= q[i + 1])]),
                                       med_evt=float(np.median(M[(G.amp >= q[i]) & (G.amp <= q[i + 1])])))
                                  for i in range(4)])
    x = np.log([r['amp'] for r in c]); y = np.log([r['sigma_xy'] for r in c]); w = np.sqrt([r['n'] for r in c])
    res['fixed_hv_exponent'] = float(np.polyfit(x, y, 1, w=w)[0])
    # at equal amplitude: fixed-HV curve against the HV-scan curve (log-interpolated)
    h = res['hv_scan_pooled']
    hx, hy = np.log([r['amp'] for r in h]), np.log([r['sigma_xy'] for r in h])
    res['fixed_over_hvscan'] = [dict(amp=r['amp'], ratio=float(r['sigma_xy'] / np.exp(np.interp(np.log(r['amp']), hx, hy))))
                                for r in c]
    # drift scan, matched sample (rows_matched.csv), in ns and in drift distance
    m = pd.read_csv(os.path.join(HERE, 'results', 'rows_matched.csv'))
    gas = json.load(open(os.path.join(REPO, 'rerun', 'paper1_gas', 'results', 'gas.json')))
    vsc = {int(p['V']): (p['v'], p['v_err']) for p in gas['scan']['points']}
    vsc[1000] = (35.8, 0.5)          # Run 1's timing v (physics_inputs.json), the 1000 V row of the same night
    d = m[(m.det == 'mx17_3') & (m.run == 'mx17_det3_saturday_scan_6-27-26') & (m.resist_V == 490) & (m.n >= 150)
          & (m.subrun.str.startswith('drift_scan') | (m.key == 'sat_det3'))]
    ds = []
    for r in d.sort_values('drift_V').itertuples():
        v = vsc.get(int(r.drift_V), (np.nan, np.nan))[0]
        ds.append(dict(subrun=r.subrun, drift_V=int(r.drift_V), E_Vcm=r.drift_V / GAP_CM, n=int(r.n), v=v,
                       sigma_xy=r.sigma_xy, sigma_xy_err=r.sigma_xy_err, sigma_evt=r.sigma_evt, sigma_evt_err=r.sigma_evt_err,
                       sigma_common=r.sigma_common, sigma_common_err=r.sigma_common_err,
                       xy_mm=r.sigma_xy * v / 1e3, evt_mm=r.sigma_evt * v / 1e3, common_mm=r.sigma_common * v / 1e3,
                       T_x=r.T_x, T_y=r.T_y))
    res['drift_scan'] = ds
    res['note'] = ('300 V row: the column overruns the readout window (T truncated) and no timing v was measured, so no mm; '
                   '100 V row: no matched clean trigger. 1000 V = Run 1 (long_run), v = 35.8 (its timing v).')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('walk', 'fixed_hv_exponent')}, indent=1))
    for r in res['fixed_hv']:
        print(f"fixed HV amp {r['amp']:7.0f} n {r['n']:5d} sxy {r['sigma_xy']:5.1f}+-{r['sigma_xy_err']:.1f} "
              f"evt {r['sigma_evt']:5.1f} med_evt {r['med_evt']:+6.1f}+-{r['med_evt_err']:.1f}")
    for r in res['fixed_over_hvscan']:
        print(f"  ratio to HV scan at amp {r['amp']:.0f}: {r['ratio']:.2f}")
    for r in ds:
        print(f"drift {r['drift_V']:5d} V {r['E_Vcm']:5.0f} V/cm n {r['n']} xy {r['sigma_xy']:.1f} evt {r['sigma_evt']:.1f} "
              f"common {r['sigma_common']:.1f} ns | v {r['v']:.1f}: xy {r['xy_mm']:.2f} evt {r['evt_mm']:.2f} common {r['common_mm']:.2f} mm")


if __name__ == '__main__':
    main()
