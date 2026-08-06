#!/usr/bin/env python3
"""
make_package.py — stage everything the condor gap-fit needs into one directory.

Builds, under <stage> (default /home/dylan/x17/cosmic_bench/condor_wft):

    payload.tar.gz          wft/ + mx_june_wft/bench/ (the fitting code)
    inputs/<key>.pkl        the bench cache of each dataset
    inputs/<key>__<b>.tgz   each calibration bundle
    jobs.txt                one line per (dataset x bundle x shard)
    run_gap_fit.sh, gap_fit.sub, README.md
    out/, log/

then rsync <stage> to lxplus and `condor_submit gap_fit.sub` there.

    make_package.py [--stage DIR] [--shards 8] [--datasets sat_det3 ...]
                    [--cross]   also pair every dataset with every OTHER
                                dataset's bundle (the calibration-systematic
                                sweep this study needs)
"""
import argparse
import itertools
import os
import shutil
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

DEFAULT_STAGE = '/home/dylan/x17/cosmic_bench/condor_wft'

# dataset -> the bundle directory name inside <OUT_BASE>/wft/
DATASETS = {
    'sat_det3':      'calib_bundle_lp2',
    'g_det3_wknd':   'calib_bundle_lp',
    'o22_long_det2': 'calib_bundle_lp',
    'g_det2':        'calib_bundle_lp',
    'g_det4':        'calib_bundle_lp',
    'g_det6_long':   'calib_bundle_lp',
    'g_det7_long':   'calib_bundle_lp',
}


def tar_dir(src, dest, arcname):
    with tarfile.open(dest, 'w:gz') as t:
        t.add(src, arcname=arcname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default=DEFAULT_STAGE)
    ap.add_argument('--shards', type=int, default=8)
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--k-bins', type=int, default=None,
                    help='charge-basis depth for the fits; slow chambers need '
                         '22 (1320 ns) or their endpoint runs off the model. '
                         'Labels get a _k<N> suffix so the results never mix '
                         'with the default-basis ones.')
    ap.add_argument('--cross', action='store_true',
                    help='every dataset x every bundle (calibration sweep)')
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()

    stage = args.stage
    for d in ('inputs', 'out', 'log'):
        os.makedirs(os.path.join(stage, d), exist_ok=True)

    keys = args.datasets or list(DATASETS)
    have, bundles = {}, {}
    for key in keys:
        cfg = get_config(key)
        W = os.path.join(cfg.OUT_BASE, 'wft')
        cache = os.path.join(W, 'bench_cache.pkl')
        bdir = os.path.join(W, DATASETS.get(key, 'calib_bundle_lp'))
        if not os.path.exists(cache):
            print(f'-- {key}: no bench_cache.pkl yet, skipped')
            continue
        if not os.path.exists(os.path.join(bdir, 'bundle.json')):
            print(f'-- {key}: no bundle at {bdir}, skipped')
            continue
        dst = os.path.join(stage, 'inputs', f'{key}.pkl')
        if not os.path.exists(dst) or os.path.getmtime(cache) > os.path.getmtime(dst):
            shutil.copy2(cache, dst)
        btgz = os.path.join(stage, 'inputs', f'{key}__bundle.tgz')
        tar_dir(bdir, btgz, 'bundle')
        have[key] = f'{key}.pkl'
        bundles[key] = f'{key}__bundle.tgz'
        print(f'   {key}: cache {os.path.getsize(dst)/1e6:.0f} MB, bundle staged')

    if not have:
        raise SystemExit('nothing staged')

    # ---- payload: the code, no data
    payload = os.path.join(stage, 'payload.tar.gz')
    with tarfile.open(payload, 'w:gz') as t:
        t.add(os.path.join(REPO, 'wft'), arcname='payload/wft',
              filter=lambda ti: None if '__pycache__' in ti.name else ti)
        for f in ('gap_fit.py', 'gap_merge.py'):
            t.add(os.path.join(REPO, 'mx_june_wft', 'bench', f),
                  arcname=f'payload/mx_june_wft/bench/{f}')
    print(f'   payload.tar.gz {os.path.getsize(payload)/1e3:.0f} kB')

    # ---- job list
    pairs = [(d, d) for d in have]
    if args.cross:
        pairs = list(itertools.product(have, have))
    lines = []
    for data_key, bundle_key in pairs:
        label = (data_key if data_key == bundle_key
                 else f'{data_key}__with__{bundle_key}')
        if args.k_bins:
            label += f'_k{args.k_bins}'
        for i in range(args.shards):
            lines.append(f'{have[data_key]} {bundles[bundle_key]} {label} '
                         f'{i} {args.shards} none {args.k_bins or "none"}\n')
    with open(os.path.join(stage, 'jobs.txt'), 'w') as f:
        f.writelines(lines)
    print(f'   jobs.txt: {len(lines)} jobs '
          f'({len(pairs)} fits x {args.shards} shards)')

    for f in ('run_gap_fit.sh', 'gap_fit.sub', 'README.md'):
        src = os.path.join(HERE, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(stage, f))
    os.chmod(os.path.join(stage, 'run_gap_fit.sh'), 0o755)
    print(f'\nstaged in {stage}')
    print('next:\n'
          f'  rsync -av {stage}/ lxplus:~/wft_gap/\n'
          '  ssh lxplus "cd ~/wft_gap && condor_submit gap_fit.sub"')


if __name__ == '__main__':
    main()
