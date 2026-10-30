#!/usr/bin/env python3
"""tan_check.py --pass-dir D: the inclined-view tan of the rows without v, rescaled by the bench v they were fitted with
(33.0 um/ns, the pass det4 bundle; sps_reco keeps the bundle v when the registry has none) over the column-duration v
(v_table.py), against the 25.64 deg mount and against the CF4 rows that carry a v.  tan = w / v, so the rescale is
tan x 33.0 / v_box (the time-domain fit itself does not change).  Read-only on the pass tables."""
import argparse, csv, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V_BENCH = 33.0
T_BEAM_BACK_OV07 = 91170.0


def med_tan(path, view, key):
    t = pd.read_parquet(path, columns=[f'{view}_tan_theta', f'{view}_ok', f'{view}_quality_ok', 't_wall'])
    m = t[f'{view}_ok'].astype(bool) & t[f'{view}_quality_ok'].astype(bool)
    if key == 'run60_flush__ov07':
        m &= t.t_wall >= T_BEAM_BACK_OV07
    return float(np.median(t[f'{view}_tan_theta'][m])), int(m.sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pass-dir', required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'tan_check.json'))
    a = ap.parse_args()
    V = {r['key']: r for r in csv.DictReader(open(os.path.join(HERE, 'results', 'v_no_v_rows.csv')))}
    reg = {r['key']: r for r in csv.DictReader(open(os.path.join(HERE, '..', 'sps_registry.csv')))}
    out = {}
    for key in sorted(reg):
        r = reg[key]
        if r['mount'] != '25.64 deg':
            continue
        p = os.path.join(a.pass_dir, key, 'events.parquet')
        tx, nx = med_tan(p, 'x', key); ty, ny = med_tan(p, 'y', key)
        v_fit = float(r['v_span_um_ns']) if r['v_span_um_ns'] else V_BENCH
        vb = V.get(key, {}).get('v_box') or ''
        o = dict(v_fit=v_fit, tan_x=tx, tan_y=ty, n=ny, v_box=float(vb) if vb else None)
        o['tan_y_rescaled'] = ty * v_fit / o['v_box'] if o['v_box'] else (ty if r['v_span_um_ns'] else None)
        out[key] = o
        print(f"{key:28s} v_fit {v_fit:5.1f}  tan_y {ty:+.3f}  v_box {vb!s:6s} -> tan_y {o['tan_y_rescaled'] if o['tan_y_rescaled'] is not None else float('nan'):+.3f}"
              f"   (mount tan {np.tan(np.radians(25.64)):.3f})")
    json.dump(out, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
