#!/usr/bin/env python3
"""cap_table.py DIR [DIR ...] -- the cap scan (r5) and the main arms against base, per chamber and view, on all / sat /
unsat planes: Delta s68 +- paired bootstrap, Delta far, chi2/dof, charge ratio.  DIRs hold <det>_<arm>.npz (the first
with <det>_base.npz).  -> results/arms/cap_scan.json + printed markdown."""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sat_arms import compare_one, SELS, DETS  # noqa: E402

ARMS = ['rail', 'railub', 'cap2000', 'cap1500', 'cap1200', 'cap1000', 'cap800', 'cap1200ub']


def main(dirs):
    out = {}
    for det in DETS:
        f = [f'{d}/{det}_base.npz' for d in dirs if os.path.exists(f'{d}/{det}_base.npz')]
        if not f:
            continue
        base = np.load(f[0])['per']
        print(f'\n**{det}** (Δs68 [°] ± paired bootstrap · Δfar pp) — X all | X sat | Y all | Y sat | χ²/dof all X/Y | q X/Y\n')
        for arm in ARMS:
            g = [f'{d}/{det}_{arm}.npz' for d in dirs if os.path.exists(f'{d}/{det}_{arm}.npz')]
            if not g:
                continue
            per = np.load(g[0])['per']; cells = []; rec = {}
            for vi, v in enumerate('xy'):
                for sn in ('all', 'sat', 'unsat'):
                    q = compare_one(base, per, vi, SELS[sn], nboot=300); rec[f'{v}_{sn}'] = q
                    if sn != 'unsat':
                        cells.append(f"{q['ds68']:+.3f} ± {q['ds68_err']:.3f} · {q['far'] - q['far_base']:+.1f}")
            out[f'{det}_{arm}'] = rec
            print(f"| {arm} | " + ' | '.join(cells) + f" | {rec['x_all']['chi2']:.0f}/{rec['y_all']['chi2']:.0f} "
                  f"({rec['x_all']['chi2_base']:.0f}/{rec['y_all']['chi2_base']:.0f}) | {rec['x_all']['q_ratio']:.3f}/{rec['y_all']['q_ratio']:.3f} |")
    json.dump(out, open(f'{HERE}/results/arms/cap_scan.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
