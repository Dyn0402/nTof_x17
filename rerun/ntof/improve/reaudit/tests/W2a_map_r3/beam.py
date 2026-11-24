#!/usr/bin/env python3
"""beam.py -- independent beam validation (horizontal view) of angle maps on the late capsule tracks.

Sample: T5 late_tracks.parquet via pm.load (>= 25 ms, gated before the map, not censored, one track per chamber and
event, not low_charge / t0_railed / split child, pre-access runs out, D needs its wall): A 1.42 M, C 1.12 M, D 1.02 M.
Maps: none; r2 (the shipped file); r3 maps from build_maps.py.  A map whose fit saw pointing cells (name *_split) is
applied to each track with the version trained on the OTHER run group (run_groups.json), so no track is seen twice.

Metrics (per arm), with r = corrected horizontal global slope - expected capsule slope e (pm.arm_arrays, source
x -10.9 / z -4.5 mm):
  core_tan      1.4826 MAD of r about its windowed centre, T5 region (reach/near-normal/dead-band masks, |y| <= 125)
  cellrms       std over T5-valid 25 mm cells of the per-cell residual centre; cellrms_detr after removing const + e
  mm metrics    on tracks with |slope| >= 0.1 (angle_map.beam): miss = pos + disp - x_meas at the source plane;
                w68_mm (null-subtracted core, track_quality.pointing), f15 = fraction |miss| < 15 mm, exc15 = f15 - null;
                x_cross = capsule_tof.crossing (source coordinate from this map); focus_s = scale of the displacement
                that minimises the median perpendicular miss about x_cross (angle_map.beam's estimator)
    beam.py r3_lines.json [r3_joint_g0.json,r3_joint_g1.json] -> beam_<tag>.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
T5 = HERE.parent / 'T5_pointing_map'
sys.path.insert(0, str(T5)); sys.path.insert(0, str(HERE))
import pm  # noqa: E402
import hmap  # noqa: E402
import apply_r3 as AP  # noqa: E402
from rerun.ntof.improve.track_quality import pointing  # noqa: E402
from sept26_prelim_analysis.capsule_tof import crossing  # noqa: E402
from rerun.ntof.stage3.map_view import interp  # noqa: E402

R2 = json.load(open(pm.MAP_FILE))['maps']
SRC = {'A': pm.XS, 'C': pm.XS, 'D': pm.ZS}


def corrected(arm, A, which, grp):
    sh, sv = A['sh'], A['sv']
    if which == 'none':
        return sh
    if which == 'r2':
        a, s = interp(R2[f'{arm}_x'], A['x'], A['y'])
        return (sh - a) / (1 + s)
    if isinstance(which, tuple):                  # split pair (map trained with group-0 pointing, with group-1)
        M0, M1 = which
        out = np.empty(len(sh))
        for k, M in ((1, M0), (0, M1)):          # group-1 tracks get the g0-trained map and vice versa
            m = grp == k
            out[m] = AP.correct_global(M, arm, A['x'][m], A['y'][m], sh[m], sv[m])[0]
        return out
    return AP.correct_global(which, arm, A['x'], A['y'], sh, sv)[0]


def metrics(arm, g, A, shc, nomv, hx, rng):
    r = shc - A['e']
    tk = hmap.track_ok(A, hx)
    reg = tk & (np.abs(A['y']) <= 125)
    c0, _, n0, core = pm.centre(r[reg])
    X, Y = np.where(tk, A['x'], np.nan), np.where(tk, A['y'], np.nan)
    m = pm.cell_map(X, Y, r, pm.E25, nmin=300)
    v = nomv & np.isfinite(m['c'])
    cc, ee = m['c'][v], nomv_e[arm][v]
    Xd = np.vstack([np.ones(v.sum()), ee]).T
    det = cc - Xd @ np.linalg.lstsq(Xd, cc, rcond=None)[0]
    # mm at the source
    h, n = pm.FR[arm]
    pos = g[f'p0_{h}'].to_numpy(float); pn = g[f'p0_{n}'].to_numpy(float)
    disp = -pn * shc
    s = np.abs(shc) >= 0.1
    pos, disp, sl = pos[s], disp[s], shc[s]
    miss = pos + disp - SRC[arm]
    sh_ = rng.permutation(len(disp))
    P = pointing(miss, pos + disp[sh_] - SRC[arm])
    xc, xce, _ = crossing(pos, disp)
    S = np.round(np.arange(0.85, 1.1501, 0.005), 3)
    med = [np.median(np.abs((pos + s_ * disp - xc) / np.hypot(s_ * sl, 1.0))) for s_ in S]
    return dict(arm=arm, n=int(len(r)), core_tan=core, centre=c0, n_core=n0, cellrms=float(np.std(cc)),
                cellrms_detr=float(np.std(det)), n_cells=int(v.sum()), w68_mm=P.get('w68'), f15=P.get('f1'),
                exc15=P.get('exc1'), exc15_err=P.get('exc1_err'), x_cross=xc, x_cross_err=xce,
                focus_s=float(S[int(np.argmin(med))]), focus_med=float(np.min(med)), n_mm=int(s.sum()))


nomv_e = {}


def main():
    specs = {'none': 'none', 'r2': 'r2'}
    for a in sys.argv[1:]:
        if ',' in a:
            f0, f1 = a.split(',')
            specs[Path(f0).stem.replace('_g0', '') + '_split'] = (AP.load(HERE / f0), AP.load(HERE / f1))
        else:
            specs[Path(a).stem] = AP.load(HERE / a)
    grp_of = {int(k): v for k, v in json.load(open(HERE / 'run_groups.json')).items()}
    d = pm.load()
    rng = np.random.default_rng(7)
    rows = []
    for arm in 'ACD':
        g = d[d.arm == arm]
        A = pm.arm_arrays(g, arm)
        A['e_normal'] = hmap.normal_halfwidth(A['sh'], A['y']) + 0.06
        hx = hmap.holes(A['x'], A['y'])
        nom = np.load(HERE / f'pc_{arm}_all.npz')
        nomv, nomv_e[arm] = nom['valid'], nom['e']
        grp = g.rn.map(grp_of).to_numpy()
        for lab, which in specs.items():
            shc = corrected(arm, A, which, grp)
            row = metrics(arm, g, A, shc, nomv, hx, rng)
            # per run group as well (the split maps: each group with the map that did not see it)
            for k in (0, 1):
                mk = grp == k
                Ak = {kk: (vv[mk] if isinstance(vv, np.ndarray) and len(vv) == len(mk) else vv) for kk, vv in A.items()}
                rk = metrics(arm, g[mk], Ak, shc[mk], nomv, hx, rng)
                row[f'core_tan_g{k}'], row[f'cellrms_detr_g{k}'] = rk['core_tan'], rk['cellrms_detr']
                row[f'w68_mm_g{k}'], row[f'x_cross_g{k}'] = rk['w68_mm'], rk['x_cross']
            row['map'] = lab
            rows.append(row)
            print(pd.Series(row).to_dict(), flush=True)
    T = pd.DataFrame(rows)
    tag = '_'.join(k for k in specs if k not in ('none', 'r2')) or 'base'
    T.to_csv(HERE / f'beam_{tag}.csv', index=False)
    pd.set_option('display.width', 250)
    print(T[['arm', 'map', 'core_tan', 'cellrms', 'cellrms_detr', 'w68_mm', 'f15', 'exc15', 'x_cross', 'focus_s']]
          .round(4).to_string())


if __name__ == '__main__':
    main()
