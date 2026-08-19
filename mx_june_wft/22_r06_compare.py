#!/usr/bin/env python3
"""
22_r06_compare.py -- print the frozen-vs-r06 gate table for one or more keys.

Reads each key's wft/angles/angular_resolution.json (frozen arm) and
wft/angles_r06/angular_resolution.json (the c2-slaved arm) and prints the
metrics the t0-prior gate was decided on, plus implied-v flatness, which is
the trustworthy judge of whether a chain is geometrically honest (WFT 35).

    ../.venv/bin/python mx_june_wft/22_r06_compare.py sat_det3 g_det7_long
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]


def main():
    from qa_config import get_config, setup_paths
    setup_paths()
    rows = []
    for key in sys.argv[1:]:
        W = os.path.join(get_config(key).OUT_BASE, 'wft')
        arms = {}
        for tag, d in (('frozen', 'angles'), ('r06', 'angles_r06')):
            p = os.path.join(W, d, 'angular_resolution.json')
            arms[tag] = json.load(open(p)) if os.path.exists(p) else None
        rows.append((key, arms))

    hdr = (f"{'key':14s} {'plane':5s} {'arm':7s} {'n':>6s} {'bias':>7s} "
           f"{'s68':>7s} {'sigma':>7s} {'vspread':>8s} {'relfrac':>8s}")
    print(hdr)
    print('-' * len(hdr))
    for key, arms in rows:
        for plane in ('x', 'y'):
            got = {}
            for tag in ('frozen', 'r06'):
                a = arms[tag]
                if a is None:
                    print(f'{key:14s} {plane:5s} {tag:7s}  (missing)')
                    continue
                p = a['planes'][plane]
                got[tag] = p
                print(f"{key:14s} {plane:5s} {tag:7s} {p['n']:6d} "
                      f"{p['bias_deg']:+7.3f} {p['s68_deg']:7.3f} "
                      f"{p['sigma_deg']:7.3f} {p['implied_v_spread']:8.3f} "
                      f"{p['frac_slope_reliable']:8.3f}")
            if len(got) == 2:
                f, r = got['frozen'], got['r06']
                print(f"{'':14s} {'':5s} {'delta':7s} {r['n']-f['n']:+6d} "
                      f"{r['bias_deg']-f['bias_deg']:+7.3f} "
                      f"{r['s68_deg']-f['s68_deg']:+7.3f} "
                      f"{r['sigma_deg']-f['sigma_deg']:+7.3f} "
                      f"{r['implied_v_spread']-f['implied_v_spread']:+8.3f} "
                      f"{r['frac_slope_reliable']-f['frac_slope_reliable']:+8.3f}")
        print()
    print('bias/s68/sigma in deg; vspread = implied-v spread across |tan_ref| '
          'bins (lower = flatter = more honest geometry).')
    print('NOTE: s68 has a ~1 deg physics floor (diffusion + granularity) -- '
          'nothing below it reads as an improvement.')


if __name__ == '__main__':
    main()
