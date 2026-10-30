#!/usr/bin/env python3
"""v_table.py [--out results/v_no_v_rows.csv]: a verdict per SPS row without v (pass-1 review P3), from results/vbox.json.

Where the column-duration route defines T (ws_spsv REPORT.md: the controls reproduce it to <= 4 %), a row gets
v = 28.8 mm / T, T the mean of the two views' stacked-box T_box and the half view difference its systematic.  T is
called defined when, in BOTH views, the box with gradient and the flat box agree (|T_box - T_flat| <= TOL x T_box: the
column end is an edge, not a fade) and the box ends inside the window (T_box <= T_MAX on 64 samples).  Otherwise:
  overrun      - 32-sample window (1.92 us) or a field < 170 V/cm: the column does not fit; positions only.
  fade         - the charge falls to zero along the column (gradient railed at -1 or box/flat disagree): T undefined.
"""
import argparse, csv, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, '..', 'sps_registry.csv')
GAP_UM = 28800.0
TOL, T_MAX = 0.08, 2900.0
CONTROLS = ('run63_flat__flat700', 'run62_rot25_ladder__d700', 'run71_raw__raw700')


def verdict(row, r):
    if int(row['n_samples']) < 64 or float(row['field_Vcm']) < 170:
        return 'overrun', None, None
    if 'x' not in r or 'y' not in r:
        return 'no result', None, None
    for v in 'xy':
        q = r[v]
        if q['grad'] <= -0.95 or abs(q['T_box'] - q['T_flat']) > TOL * q['T_box'] or q['T_box'] > T_MAX:
            return 'fade', None, None
    T = 0.5 * (r['x']['T_box'] + r['y']['T_box'])
    v = GAP_UM / T
    return 'defined', v, v * abs(r['x']['T_box'] - r['y']['T_box']) / (2 * T)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(HERE, 'results', 'v_no_v_rows.csv'))
    a = ap.parse_args()
    B = json.load(open(os.path.join(HERE, 'results', 'vbox.json')))
    reg = {r['key']: r for r in csv.DictReader(open(REG))}
    rows = []
    for key, row in reg.items():
        if row['v_span_um_ns'] and key not in CONTROLS:
            continue
        r = B.get(key, {})
        vd, v, ve = verdict(row, r)
        o = dict(key=key, control=key in CONTROLS, gas=row['gas'], mount=row['mount'], field_Vcm=row['field_Vcm'],
                 n_samples=row['n_samples'], registry_v=row['v_span_um_ns'], verdict=vd,
                 v_box=round(v, 2) if v else '', v_sys=round(ve, 2) if ve else '')
        for vv in 'xy':
            q = r.get(vv, {})
            o.update({f'T_box_{vv}': q.get('T_box', ''), f'T_flat_{vv}': q.get('T_flat', ''),
                      f'grad_{vv}': round(q['grad'], 2) if 'grad' in q else ''})
        rows.append(o)
        print(f"{key:28s} {vd:9s} v {o['v_box']!s:6s} +- {o['v_sys']!s:5s}  X {o['T_box_x']}/{o['T_flat_x']} g{o['grad_x']}  "
              f"Y {o['T_box_y']}/{o['T_flat_y']} g{o['grad_y']}")
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
