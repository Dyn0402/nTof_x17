#!/usr/bin/env python3
"""make_boot_jobs.py <dag dir> --code <code dir> -- the residual bootstrap DAG (BOOTSTRAP_2026-11-01.md).

EXTRACT_<det>: the pass replay of the row's clean events (boot_extract.sub);
TOY_<det>_<view>_<angle>_<group>: toy truths at one angle point with a group of residual arms;
DATA_<det>_<view>_<group>: data truths.  Every node RETRY 3 (EOS transients).
"""
import argparse
import os

E = '/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/floor_boot'
ROWS = dict(det2='o22_long_det2', det3='sat_det3', det4='g_det4', det7='g_det7_long')
TOY_GROUPS = dict(a='T,N,D', b='Dcm,Dmean,Dtshape', c='Danti,Dtshuf,Dsshuf')
DATA_GROUPS = dict(a='dD,dN,dDtshuf,dDcm', b='dDanti,dDtshape,dDmean')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dag')
    ap.add_argument('--code', required=True)
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--no-extract', action='store_true', help='the data pickles exist already')
    a = ap.parse_args()
    L = []
    for det, key in ROWS.items():
        pkl = f'{E}/data/{key}.pkl'
        if not a.no_extract:
            L += [f'JOB EXTRACT_{det} boot_extract.sub',
                  f'VARS EXTRACT_{det} code="{a.code}" key="{key}" ids="{E}/ids/{key}.json" '
                  f'out="root://eosexperiment.cern.ch/{pkl}"', f'RETRY EXTRACT_{det} 3']
        kids = []
        for view in ('x', 'y'):
            for ang in (3, 9, 16):
                for g, arms in TOY_GROUPS.items():
                    n = f'TOY_{det}_{view}_{ang}_{g}'
                    kids.append(n)
                    L += [f'JOB {n} boot.sub',
                          f'VARS {n} code="{a.code}" view="{view}" name="{n}" '
                          f'args="toy --det {det} --angle {ang} --arms {arms} --data {pkl} --n {a.n}"',
                          f'RETRY {n} 3']
            for g, arms in DATA_GROUPS.items():
                n = f'DATA_{det}_{view}_{g}'
                kids.append(n)
                L += [f'JOB {n} boot.sub',
                      f'VARS {n} code="{a.code}" view="{view}" name="{n}" '
                      f'args="data --det {det} --arms {arms} --data {pkl} --bundle {E}/bundles/{det}/calib_bundle '
                      f'--n {a.n}"', f'RETRY {n} 3']
        if not a.no_extract:
            L.append(f'PARENT EXTRACT_{det} CHILD ' + ' '.join(kids))
    os.makedirs(os.path.join(a.dag, 'log'), exist_ok=True)
    open(os.path.join(a.dag, 'boot.dag'), 'w').write('\n'.join(L) + '\n')
    print(sum(1 for x in L if x.startswith('JOB ')), 'nodes')


if __name__ == '__main__':
    main()
