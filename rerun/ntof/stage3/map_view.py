#!/usr/bin/env python3
"""map_view.py -- the in-situ angle map of chambers A and C applied to the analysis view (Dylan 2026-11-20).

Input: the v(t) analysis view (vt_view.py, `stage3_vt/`).  Output: the same tables with the map applied under the plain
names (default `stage3_vtmap/`; after the 11-20 switch this IS `stage3_vt/`, the analysis view):

    <col>         = map-corrected, for every angle-derived column (build_tracks.ANGLE_DERIVED), tan_raw_x/y,
                    in_acceptance and gated (true |tan| < 0.5 recomputed on the corrected angles)
    <col>_nomap   = the v(t) value without the map (the input view's <col>)
    <col>_frozen  = unchanged (frozen bundle v)

The map (`angle_map/angle_map.json`: A, C fitted by rerun/ntof/improve/angle_map.py on the A-C cosmic lines of every
beam-off sub-run, D by bd_map.py on the B-D lines after the censored-window cut; study and validation:
rerun/ntof/improve/README.md) corrects each view's GLOBAL slope s (d_p/d_z on A, C; d_p/d_x on D, FRAME) as
s' = (s - a(x, y)) / (1 + s_(x, y)), with a and s_ bilinear in the track's (x_local, y_local) over 50 mm cells.  In the
view's own tan t = g s, g = +-1 per arm and view (checked per table), so t' = (t - g a) / (1 + s_).  The corrected tans
go through build_tracks' own geometry with k = 1 (local_and_global, pointing, predictions), exactly as vt_join does,
so every derived column sits on one calibration.  drift_len_mm (v only) is unchanged; path_len_mm and q_per_len
follow the new secant.  B rows pass through unchanged (no angle; their `_nomap` twins equal the plain columns).

    map_view.py [--src $WORK_DATA/ntof_pass1/stage3_vt] [--out $WORK_DATA/ntof_pass1/stage3_vtmap] [--jobs 14]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
MAP_FILE = HERE / 'angle_map' / 'angle_map.json'
VIEW = 'pass1_vt_map'
TAN_MAX = 0.5
MAP_ARMS = ('A', 'C', 'D')
#: the global coordinate each view's tan measures, and the chamber normal: the map is fitted on global slopes
#: d_<coord>/d_<normal> (A, C against the A-C cosmic line; D against the B-D line)
FRAME = {'A': {'x': 'x', 'y': 'y', 'n': 'z'}, 'C': {'x': 'x', 'y': 'y', 'n': 'z'}, 'D': {'x': 'z', 'y': 'y', 'n': 'x'}}


def interp(mp, x, y):
    c = np.asarray(mp['centres'])
    A, S = np.asarray(mp['a']), np.asarray(mp['s'])
    xx = np.clip(np.asarray(x, float), c[0], c[-1])
    yy = np.clip(np.asarray(y, float), c[0], c[-1])
    bad = ~(np.isfinite(xx) & np.isfinite(yy))
    xx, yy = np.where(bad, 0.0, xx), np.where(bad, 0.0, yy)
    i = np.clip(np.searchsorted(c, xx) - 1, 0, len(c) - 2)
    k = np.clip(np.searchsorted(c, yy) - 1, 0, len(c) - 2)
    tx = (xx - c[i]) / (c[i + 1] - c[i])
    ty = (yy - c[k]) / (c[k + 1] - c[k])

    def bl(Z):
        return ((1 - tx) * (1 - ty) * Z[i, k] + tx * (1 - ty) * Z[i + 1, k]
                + (1 - tx) * ty * Z[i, k + 1] + tx * ty * Z[i + 1, k + 1])
    return np.where(bad, mp['a0'], bl(A)), np.where(bad, mp['s0'], bl(S))


def slope_sign(t, d_p, d_z):
    """g with d_p/d_z = g * t (+-1), from the rows themselves; refuses anything else."""
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (d_p / d_z) / t
    r = r[np.isfinite(r) & (np.abs(t) > 0.02)]
    if len(r) == 0:
        return None
    g = float(np.median(r))
    if abs(abs(g) - 1) > 1e-3 or np.mean(np.abs(r - g) < 1e-3) < 0.99:
        raise AssertionError(f'global slope is not +-tan: median ratio {g}')
    return float(np.sign(g))


def apply(t, maps, trs):
    import pandas as pd
    from sept26_prelim_analysis import build_tracks as BT
    from ntof_tracking.reco import geometry as G
    cols = list(BT.ANGLE_DERIVED) + ['tan_raw_x', 'tan_raw_y', 'in_acceptance', 'gated']
    out = t.copy()
    for c in cols:
        out[c + '_nomap'] = t[c]
    info = {}
    for arm in MAP_ARMS:
        m = (t['arm'] == arm).to_numpy()
        if not m.any():
            continue
        a = t.loc[m]
        new = {}
        for p in 'xy':
            tt = a[f'tan{p}'].to_numpy(float)
            fr = FRAME[arm]
            g = slope_sign(tt, a[f'd_{fr[p]}'].to_numpy(float), a[f'd_{fr["n"]}'].to_numpy(float))
            if g is None:
                new[p] = tt
                continue
            A_, S_ = interp(maps[f'{arm}_{p}'], a['x_local'].to_numpy(float), a['y_local'].to_numpy(float))
            new[p] = (tt - g * A_) / (1.0 + S_)
            ok = np.abs(tt) < TAN_MAX
            info[f'{arm}_{p}'] = dict(g=g, median_abs_shift_in_acc=float(np.nanmedian(np.abs(new[p] - tt)[ok]))
                                      if ok.any() else None)
        gg = a[['x_p0', 'y_p0', 'x_q_uend', 'y_q_uend', 'x_q_sum', 'y_q_sum']].copy()
        gg['x_tan_theta'] = new['x']
        gg['y_tan_theta'] = new['y'] / BT.TAN_SIGN_Y
        gg = BT.local_and_global(gg, trs[G.DET_NAME[arm]], k=1.0)
        if not np.allclose(gg['tanx'], new['x'], equal_nan=True) or not np.allclose(gg['tany'], new['y'], equal_nan=True):
            raise AssertionError('local_and_global did not reproduce the corrected tans')
        gg['drift_len_mm'] = a['drift_len_mm'].to_numpy(float)
        sec = np.sqrt(1.0 + gg['tanx'] ** 2 + gg['tany'] ** 2)
        gg['path_len_mm'] = gg['drift_len_mm'] * sec
        with np.errstate(divide='ignore', invalid='ignore'):
            qpl = a['q_total'].to_numpy(float) / gg['path_len_mm'].replace(0, np.nan).to_numpy(float)
        gg['q_per_len'] = np.where(a['drift_railed'].to_numpy(bool), np.nan, qpl)
        gg = BT.pointing(gg)
        gg = BT.predictions(gg)
        for c in BT.ANGLE_DERIVED:
            v = gg[c].to_numpy()
            if out[c].dtype == bool or str(out[c].dtype) == 'boolean':
                out.loc[m, c] = pd.array(v, dtype='boolean')
            else:
                out.loc[m, c] = v
        out.loc[m, 'tan_raw_x'] = gg['tanx'].to_numpy()
        out.loc[m, 'tan_raw_y'] = gg['tany'].to_numpy()
        acc = (np.abs(gg['tanx'].to_numpy(float)) < TAN_MAX) & (np.abs(gg['tany'].to_numpy(float)) < TAN_MAX)
        out.loc[m, 'in_acceptance'] = acc
        out.loc[m, 'gated'] = a['gated_reco'].fillna(False).to_numpy(bool) & acc
    return out, info


def one(src, out, maps, map_md5, force):
    import pandas as pd
    from ntof_tracking.reco import geometry as G
    from sept26_prelim_analysis import paths
    src, out = Path(src), Path(out)
    dst = out / src.name
    if dst.exists() and not force:
        return src.name, 'exists'
    t = pd.read_parquet(src)
    run = str(t['run'].iloc[0]).replace('run_', '')
    cfg = json.loads((Path(str(paths.root('runs'))) / f'run_{run}' / 'run_config.json').read_text())
    v, info = apply(t, maps, G.detector_transforms(cfg))
    tmp = dst.with_suffix('.part')
    v.to_parquet(tmp, index=False, compression='snappy')
    os.replace(tmp, dst)
    mj = src.with_suffix('.meta.json')
    meta = json.load(open(mj)) if mj.exists() else {}
    meta['map'] = dict(name=VIEW, source=str(src), map_file=str(MAP_FILE), map_md5=map_md5,
                       rule="s' = (s - a)/(1 + s_) on the global slope per (x_local, y_local); A, C, D; <col>_nomap = "
                            'the v(t) view value; gated/in_acceptance recomputed (true |tan| < 0.5)',
                       n_gated_by_arm={k: int(x) for k, x in v.groupby('arm')['gated'].sum().items()},
                       n_gated_nomap_by_arm={k: int(x) for k, x in v.groupby('arm')['gated_nomap'].sum().items()},
                       per_view=info, built_by='rerun/ntof/stage3/map_view.py')
    json.dump(meta, open(out / mj.name, 'w'), indent=1)
    return src.name, f"gated A {meta['map']['n_gated_nomap_by_arm'].get('A')}->{meta['map']['n_gated_by_arm'].get('A')}"


def main():
    wd = os.environ.get('WORK_DATA', '/media/ucla/work')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=f'{wd}/ntof_pass1/stage3_vt')
    ap.add_argument('--out', default=f'{wd}/ntof_pass1/stage3_vtmap')
    ap.add_argument('--map', default=str(MAP_FILE))
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--only', default=None, help='glob on file names (test)')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    if Path(a.src).resolve() == Path(a.out).resolve():
        sys.exit('--src and --out are the same directory')
    maps = json.load(open(a.map))['maps']
    md5 = hashlib.md5(open(a.map, 'rb').read()).hexdigest()
    os.makedirs(a.out, exist_ok=True)
    fs = sorted(Path(a.src).glob(a.only or 'tracks_run_*.parquet'))
    with ProcessPoolExecutor(a.jobs) as ex:
        jobs = [ex.submit(one, f, a.out, maps, md5, a.force) for f in fs]
        for i, j in enumerate(as_completed(jobs)):
            name, msg = j.result()
            print(f'[{i + 1}/{len(fs)}] {name}: {msg}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
