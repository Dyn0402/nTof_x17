#!/usr/bin/env python3
"""ga_qgain_compare.py -- paired comparison of the condor held-out arms (ga_eval.sh: heldout_arm.py on the ws_t0
cache events [0, 1500), 6 chunks of 250): base (promoted bundle, unit gain) vs qgain (ga_chargegain.py's per-channel
gain from the head-on charge profile of DISJOINT finalize events).  Same statistics as ga_analyse.compare.

    ga_qgain_compare.py [--inp ~/x17/cosmic_bench/model_audit/ga/eval]  -> results/qgain_compare.json/.md
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ga_analyse import compare  # noqa: E402


def rows(inp, det, arm):
    fs = sorted(glob.glob(os.path.join(inp, f'{det}_{arm}_*.npz')))
    return np.concatenate([np.load(f)['per'][:, :12] for f in fs]) if fs else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--inp', default=os.path.expanduser('~/x17/cosmic_bench/model_audit/ga/eval'))
    a = ap.parse_args()
    R, md = {}, ['| det | view | n | s68 all base → qgain (Δ ± σ) | head-on Δ | inclined Δ | far Δ pp | χ²/dof med base → qgain |',
                 '|---|---|---|---|---|---|---|---|']
    for det in ('det4', 'det3'):
        b, q = rows(a.inp, det, 'base'), rows(a.inp, det, 'qgain')
        if b is None or q is None:
            continue
        R[det] = {p: compare(b, q, vi) for vi, p in enumerate('xy')}
        for p in 'xy':
            e = R[det][p]
            md.append(f"| {det} | {p} | {e['n']} | {e['all']['s68'][0]:.3f} → {e['all']['s68'][1]:.3f} ({e['all']['s68'][2]:+.3f} ± "
                      f"{e['all']['s68'][3]:.3f}) | {e['headon']['s68'][2]:+.3f} ± {e['headon']['s68'][3]:.3f} | {e['inclined']['s68'][2]:+.3f} ± "
                      f"{e['inclined']['s68'][3]:.3f} | {e['all']['far'][2]:+.2f} ± {e['all']['far'][3]:.2f} | {e['all']['chi2dof'][0]:.1f} → "
                      f"{e['all']['chi2dof'][1]:.1f} |")
    json.dump(R, open(os.path.join(HERE, 'results', 'qgain_compare.json'), 'w'), indent=1)
    open(os.path.join(HERE, 'results', 'qgain_compare.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
