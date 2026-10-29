#!/usr/bin/env python3
"""big_det3.py -- a second, larger det3 held-out sample for the saturation arms that matter (rail, floor bound, cap).

The cache: rerun/w12/build_cache_wft.py sat_det3 --align <Analysis_rr1/.../w12/it0/alignment/alignment.json> --n 8000
(reproduces the rr1 it1 cache: same events, W identical, ref_mesh within 1.7 um) -> 6318 events; events 1500-6318
(4818) are disjoint from the rr1 cache (whose 180-1500 the other rounds use) and were not in the base bundle's
calibration cache.  Rails: rails.py det3 --cache ... (bit-exact).  Four chunks per arm on condor.

    big_det3.py build OUT CODE           DAG into OUT (cache + rails go to EOS satur/big/)
    big_det3.py compare DIR              DIR holds det3big_<arm>_c<k>.npz -> results/arms/big_det3.json + printed table
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sat_arms import ARMS, EXTRA, SH, SUB, compare_one, SELS  # noqa: E402

EOS = '/eos/user/d/dneff/x17rerun'
BIG_ARMS = ['base', 'rail', 'railub', 'cap2000', 'cap1500', 'cap1200', 'cap1200ub']
CHUNKS = [(1500, 1205), (2705, 1205), (3910, 1205), (5115, 1203)]


def build(out, code):
    os.makedirs(f'{out}/log', exist_ok=True); L = []
    for arm in BIG_ARMS:
        for k, (s, n) in enumerate(CHUNKS):
            nm = f'SATBIG_{arm}_c{k}'
            args = (f'--cache {EOS}/satur/big/det3_8000.pkl --bundle {EOS}/t15/bundles/det3/base '
                    f'--rails {EOS}/satur/big/rails_det3_big.pkl --out {EOS}/satur/big/out/det3big_{arm}_c{k}.npz '
                    f'--start {s} --n {n} --jobs 8 --k 24 --t0-fix' + EXTRA.get(arm, ''))
            L += [f'JOB {nm} sat.sub', f'VARS {nm} nodename="{nm}" code="{code}" diag="{ARMS[arm] or "none"}" args="{args}"',
                  f'RETRY {nm} 1']
    open(f'{out}/sat_big.dag', 'w').write('\n'.join(L) + '\n')
    open(f'{out}/sat.sub', 'w').write(SUB); open(f'{out}/run_sat.sh', 'w').write(SH)
    print(f'{out}/sat_big.dag: {len(L) // 3} nodes')


def load(d, arm):
    fs = sorted(glob.glob(f'{d}/det3big_{arm}_c*.npz'))
    return np.concatenate([np.load(f)['per'] for f in fs]) if len(fs) == len(CHUNKS) else None


def compare(d):
    base = load(d, 'base'); out = {}
    print('| arm | X all | X sat | Y all | Y sat | far X / Y (base) | χ²/dof X / Y (base) | q X / Y |')
    print('|---|---|---|---|---|---|---|---|')
    for arm in BIG_ARMS:
        per = load(d, arm)
        if per is None:
            continue
        rec = {f'{v}_{s}': compare_one(base, per, vi, SELS[s], nboot=400) for vi, v in enumerate('xy') for s in ('all', 'sat', 'unsat')}
        out[arm] = rec
        if arm == 'base':
            print(f"| base | s68 {rec['x_all']['s68']:.3f} (n {rec['x_all']['n']}) | {rec['x_sat']['s68']:.3f} (n {rec['x_sat']['n']}) | "
                  f"{rec['y_all']['s68']:.3f} | {rec['y_sat']['s68']:.3f} (n {rec['y_sat']['n']}) | {rec['x_all']['far']:.1f} / {rec['y_all']['far']:.1f} | "
                  f"{rec['x_all']['chi2']:.0f} / {rec['y_all']['chi2']:.0f} | |")
            continue
        c = [f"{rec[k]['ds68']:+.3f} ± {rec[k]['ds68_err']:.3f}" for k in ('x_all', 'x_sat', 'y_all', 'y_sat')]
        print(f"| {arm} | " + ' | '.join(c) + f" | {rec['x_all']['far']:.1f} / {rec['y_all']['far']:.1f} ({rec['x_all']['far_base']:.1f} / "
              f"{rec['y_all']['far_base']:.1f}) | {rec['x_all']['chi2']:.0f} / {rec['y_all']['chi2']:.0f} | "
              f"{rec['x_all']['q_ratio']:.3f} / {rec['y_all']['q_ratio']:.3f} |")
    json.dump(out, open(f'{HERE}/results/arms/big_det3.json', 'w'), indent=1)


if __name__ == '__main__':
    {'build': lambda: build(sys.argv[2], sys.argv[3]), 'compare': lambda: compare(sys.argv[2])}[sys.argv[1]]()
