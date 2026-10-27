#!/usr/bin/env python3
"""dp_arms.py --q DIR --stack zsstack_fixed.json --res results.json --out dp_gap.json

The model lever on F16: the inclined-view slope |w_y|/tan(25.64 deg) and the implied gap |w| T_box / tan on three
fixed caches, with the bench det4 bundle's diffusion Dp = 0.020 mm/sqrt(ns) (results.json, z arm) against SPS-gas
values (qfit.py --set Dp=..., ws_sps measured Dp_X 0.010-0.011, Dp_Y ~ Magboltz), zeros fitted as measured.
"""
import argparse, json, os
import numpy as np

TAN = float(np.tan(np.radians(25.64)))
E = {'run63_rot25_operating_00_d675': 234.5, 'run63_rot25_operating_00_d575': 199.8, 'run62_rot25_ladder_driftscan_gap150V_d700': 243.2}
ARMS = (('dp0.006', 'Dp 0.006'), ('dp0.010', 'Dp 0.010'), ('dp010s', 'Dp 0.010, σ0_y 0.32, no σ0 profile'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--q', required=True); ap.add_argument('--stack', required=True)
    ap.add_argument('--res', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); ST = json.load(open(a.stack)); R = json.load(open(a.res)); out = {}
    print('| cache | E | arm | |w_y|/tan [µm/ns] | χ²/dof y | gap = |w|·T_box/tan [mm] | with T_flat |')
    print('|' + '---|' * 7)
    for k, e in E.items():
        T, Tf = ST[k]['y']['T_box'], ST[k]['y']['T_flat']
        b = R['z_' + k]['y']; ws = [('bench model (Dp 0.020)', abs(b['w']), b['chi2dof'])]
        for arm, lab in ARMS:
            P = np.load(os.path.join(a.q, f'{arm}_{k}.npz'))['per']; m = P[:, 1] == 1
            ws.append((lab, abs(float(np.median(P[m, 3]))), float(np.median(P[m, 4] / np.maximum(P[m, 5], 1)))))
        for lab, w, c in ws:
            g, gf = w * T / 1e3 / TAN, w * Tf / 1e3 / TAN
            out.setdefault(lab, []).append([e, g])
            print(f'| {k.replace("_operating_00", "").replace("_ladder_driftscan_gap150V", "")} | {e} | {lab} | {w / TAN:.2f} | {c:.1f} | {g:.1f} | {gf:.1f} |')
    json.dump(out, open(a.out, 'w'))


if __name__ == '__main__':
    main()
