#!/usr/bin/env python3
"""tan_known.py --pass-dir D [--vbox-dir V ...] -- the known-angle check of Paper II (R7) on the final SPS products.

For every rotated SPS row (mount 25.64 or 15.465 deg in rerun/sps_registry.csv): the median tan of each view over the
pass's good fits (x/y_ok and quality_ok; run60 ov07 from the beam's return, t_wall >= 91170 s), then the inclined
view's tan rescaled to each available drift velocity.  tan = w / v and the fit used v_fit (the registry's v_span, or
the bench 33.0 um/ns where the row has none), so tan(v) = tan_fit * v_fit / v.  Velocities:
  registry   the row's own v_span (the retired estimate below ~220 V/cm, rerun/sps_followup README)
  column     the column-duration v = 28.8 mm / T (vbox.py), where the column is 'defined' (v_table.py rule)
Only |tan| is defined at SPS (COORDINATES.md 4b).  The stage angles: 25.4 deg measured (yaw, DET4_URW_MAPPING; the
config said 25.64), 15.465 deg from the config (not measured).  Writes results/known_angle.json.
"""
import argparse, csv, glob, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
V_BENCH = 33.0
STAGE = {'25.64 deg': 25.4, '15.465 deg': 15.465}
T_BEAM_BACK_OV07 = 91170.0


def med_tan(path, key):
    t = pd.read_parquet(path, columns=['x_tan_theta', 'y_tan_theta', 'x_ok', 'y_ok', 'x_quality_ok', 'y_quality_ok', 't_wall'])
    o = {}
    for v in 'xy':
        m = t[f'{v}_ok'].astype(bool) & t[f'{v}_quality_ok'].astype(bool)
        if key == 'run60_flush__ov07':
            m &= t.t_wall >= T_BEAM_BACK_OV07
        o[f'tan_{v}'] = float(np.median(t[f'{v}_tan_theta'][m])); o[f'n_{v}'] = int(m.sum())
    return o


def vbox_rows(dirs):
    """{key: v_col} where the column duration is defined in both views (v_table.py: box and flat box within 8 %)."""
    V = {}
    for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_followup', 'results', 'v_no_v_rows.csv'))):
        if r['verdict'] == 'defined' and r['v_box']:
            V[r['key']] = float(r['v_box'])
    for d in dirs:
        for p in glob.glob(os.path.join(d, '*', 'vbox.json')) + glob.glob(os.path.join(d, 'vbox.json')):
            for k, r in json.load(open(p)).items():
                tb = [r[v]['T_box'] for v in 'xy' if r.get(v)]
                tf = [r[v]['T_flat'] for v in 'xy' if r.get(v)]
                if len(tb) == 2 and all(abs(a - b) / a <= 0.08 for a, b in zip(tb, tf)):
                    V[k] = 28.8e3 / np.mean(tb)
    return V


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pass-dir', required=True); ap.add_argument('--vbox-dir', nargs='*', default=[])
    a = ap.parse_args()
    reg = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'sps_registry.csv')))}
    Vc = vbox_rows(a.vbox_dir)
    rows = {}
    for key, r in sorted(reg.items()):
        if r['mount'] not in STAGE:
            continue
        p = os.path.join(a.pass_dir, key, 'events.parquet')
        if not os.path.exists(p):
            continue
        o = med_tan(p, key)
        v_fit = float(r['v_span_um_ns']) if r['v_span_um_ns'] else V_BENCH
        o.update(mount=r['mount'], stage_deg=STAGE[r['mount']], field_Vcm=float(r['field_Vcm']), gas=r['gas'],
                 v_fit=v_fit, v_reg=float(r['v_span_um_ns']) if r['v_span_um_ns'] else None, v_col=Vc.get(key))
        if o['v_reg']:
            o['tan_y_registry'] = o['tan_y']
        else:
            o['tan_y_bench'] = o['tan_y']
        if o['v_col']:
            o['tan_y_column'] = o['tan_y'] * v_fit / o['v_col']
        rows[key] = o
        print(f"{key:28s} {r['mount']:10s} E {o['field_Vcm']:6.1f}  tan_x {o['tan_x']:+.3f}  tan_y {o['tan_y']:+.3f} (v {v_fit:4.1f})"
              f"  v_col {o['v_col'] if o['v_col'] else float('nan'):5.2f} -> {o.get('tan_y_column', float('nan')):+.3f}")
    head = {}
    for src in ('column', 'registry'):
        for st in (25.4, 15.465):
            r_ = [abs(o[f'tan_y_{src}']) / np.tan(np.radians(st)) for o in rows.values()
                  if o.get(f'tan_y_{src}') is not None and o['stage_deg'] == st and o['field_Vcm'] >= 190]
            if r_:
                head[f'{src} {st} deg, E >= 190 V/cm'] = dict(n=len(r_), mean_ratio=round(float(np.mean(r_)), 3),
                                                             rms=round(float(np.std(r_)), 3))
    json.dump(dict(rows=rows, headline=head), open(os.path.join(HERE, 'results', 'known_angle.json'), 'w'), indent=1)
    print(json.dumps(head, indent=1))


if __name__ == '__main__':
    main()
