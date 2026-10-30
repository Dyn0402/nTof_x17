#!/usr/bin/env python3
"""make_dag.py <code_dir> [--root R] -- the det4 Y ablation DAG (DET4Y_ABLATION_2026-10-30.md), run from ~/x17rerun/det4y_ablation.

g_det4, the G1 events (wft.cli reco --matched-only --limit 8000), one code archive for every arm.  Row dir
<R>/mx17_det4_day_6-24-26/long_run/mx17_4/wft/ holds, staged by hand before submitting:
    bundle_a, bundle_b, bundle_d      (make_bundles.py)
    bow_block_pass.json               (Analysis_pass1 .../wft/calib/c2/bow_block.json: the bow the pass bundle carries)
    align_pass/alignment.json         (Analysis_pass1 .../wft/calib/r3/alignment: the alignment G1 scored the pass on)
    align_prod/alignment.json         (Analysis_fe .../wft/prod_alignment: the alignment G1 scored the audit on)
Per arm X: RECO_X -> ANG_X_pass, ANG_X_prod (03_angles on each fixed alignment), BUD_X (effbudget, pass alignment).
Arms c and e: W0KW (set_w0kw --bow-json --table on arm a's, resp. b's, reconstruction and angles) -> bundle_c / bundle_e.
"""
import argparse
import os

R = '/eos/user/d/dneff/x17rerun/det4y_ablation/Analysis'
KEY = 'g_det4'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('code')
    ap.add_argument('--root', default=R)
    ap.add_argument('--out', default='ablation.dag')
    a = ap.parse_args()
    L = []
    pre = f'{KEY} --eos-analysis {a.root}'

    def node(name, args, cpus, mem, parents=()):
        L.append(f'JOB {name} stage1.sub')
        L.append(f'VARS {name} code="{a.code}" args="{pre} {args}" cpus="{cpus}" mem="{mem}" node="{name}"')
        L.append(f'RETRY {name} 3')
        for p in parents:
            L.append(f'PARENT {p} CHILD {name}')

    w = '{out}/wft'

    def arm(x, parents=()):
        t = f'{w}/arm_{x}/events.parquet'
        node(f'RECO_{x}', f'--module wft.cli -- reco {{key}} --bundle {w}/bundle_{x} --out {t} --matched-only '
                          f'--limit 8000 --jobs 16', 16, 24000, parents)
        for al in ('pass', 'prod'):
            node(f'ANG_{x}_{al}', f'--parts run_config,m3,hits --script mx_june_wft/03_angles.py -- --table {t} '
                                  f'--alignment {w}/align_{al}/alignment.json --out {w}/arm_{x}/angles_{al}',
                 1, 8000, [f'RECO_{x}'])
        node(f'BUD_{x}', f'--parts run_config,m3,hits --script rerun/effbudget/budget.py -- det4 --table {t} '
                         f'--alignment {w}/align_pass/alignment.json --out {w}/arm_{x}/budget', 1, 8000, [f'RECO_{x}'])

    for x in 'abd':
        arm(x)
    for x, src in (('c', 'a'), ('e', 'b')):
        node(f'W0KW_{x}', f'--parts run_config --script rerun/w12/set_w0kw.py -- --bundle {w}/bundle_{src} '
                          f'--angles {w}/arm_{src}/angles_pass --table {w}/arm_{src}/events.parquet '
                          f'--bow-json {w}/bow_block_pass.json --out {w}/bundle_{x}', 1, 4000, [f'ANG_{src}_pass'])
        arm(x, [f'W0KW_{x}'])
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print(f'wrote {a.out}: {sum(1 for l in L if l.startswith("JOB"))} nodes')


if __name__ == '__main__':
    main()
