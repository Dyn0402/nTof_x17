#!/usr/bin/env python3
"""
make_bench_package.py — stage a benchmark scan (run_bench.py points) for condor.

The gap package (make_package.py) ships the *gap* fit; this ships the
*reconstruction* benchmark, i.e. `bench/run_bench.py` scored against the cached
M3 truth. One condor job = one scan point = one (bundle patch, window crop)
configuration evaluated on the same fixed event subset.

Why it exists: the 07-30 sensitivity scan (WINDOW_ABLATION §3) found that the
det3 production bundle is NOT at the angle-resolution optimum -- several +-25 %
perturbations improve sigma_theta_Y from 1.21 to 1.07 deg, far outside the
+-0.03 deg statistical error. The bundle was fitted by chi2, not by angle
resolution, so that is not a contradiction, but it means a targeted scan can
buy ~10 % of the Y angle resolution. A scan is ~50 fits of a few thousand
events each: hours serially, minutes on the grid.

    make_bench_package.py [--stage DIR] [--key sat_det3] [--bundle NAME]
                          [--events 2000] [--split 0:2]

then

    rsync -a <stage>/ lxplus:~/wft_bench/
    ssh lxplus 'cd ~/wft_bench && condor_submit bench_scan.sub'
"""
import argparse
import json
import os
import shutil
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]

DEFAULT_STAGE = '/home/dylan/x17/cosmic_bench/condor_bench'


def scan_points(hyper):
    """The det3 angle-optimality scan.

    Multiplicative factors on the production hyper, so the grid is bundle-
    relative and can be re-used on another detector unchanged. The 1-D arms
    cover each constant the sensitivity scan flagged; the 2-D corners cover the
    two known degeneracies (sigma_p0 <-> Dp both set transverse spread;
    tau_s <-> kTauY both set the Y copy timescale). The crop arm settles the
    'start 4 with n = 24-26' corner that WINDOW_ABLATION §2d could not reach:
    latency and window length were never moved together.
    """
    pts = [dict(tag='base', patch={}, crop=None)]

    def arm(name, factors):
        for f in factors:
            pts.append(dict(tag=f'{name}_{f:g}'.replace('.', 'p'),
                            patch={name: hyper[name] * f}, crop=None))

    # A calibration whose sigma_p0 railed at its lower guard (det6: 0.039 mm
    # against det3's 0.41) needs a much wider arm than +-60 % to be tested at
    # all -- and the angle resolution is the constant's most sensitive metric.
    arm('sigma_p0', ([0.6, 0.75, 0.9, 1.1, 1.25, 1.4, 1.6] if hyper['sigma_p0'] > 0.1
                     else [2, 4, 6, 8, 10, 12, 16]))
    arm('Dp',       [0.25, 0.5, 0.75, 1.5, 2.0, 3.0])
    arm('tau_s',    [0.7, 0.85, 1.15, 1.3])
    arm('kTauY',    [0.7, 0.85, 1.15, 1.3])
    arm('kY',       [0.7, 0.85, 1.2, 1.4])
    arm('c1',       [0.5, 2.0, 4.0])      # c1 sits ON its 0.05 floor: is that real?
    arm('sigma_s',  [0.5, 2.0])

    for fs in (0.8, 1.25):                # sigma_p0 x Dp corners
        for fd in (0.5, 2.0):
            pts.append(dict(tag=f'sp0{fs:g}_Dp{fd:g}'.replace('.', 'p'),
                            patch={'sigma_p0': hyper['sigma_p0'] * fs,
                                   'Dp': hyper['Dp'] * fd}, crop=None))
    for ft in (0.85, 1.15):               # tau_s x kTauY corners
        for fk in (0.85, 1.15):
            pts.append(dict(tag=f'tau{ft:g}_kty{fk:g}'.replace('.', 'p'),
                            patch={'tau_s': hyper['tau_s'] * ft,
                                   'kTauY': hyper['kTauY'] * fk}, crop=None))

    # the untested window corner: raise the DAQ latency AND the sample count
    for start, n in ((6, 20), (5, 20), (4, 20), (4, 22), (4, 24), (4, 26),
                     (3, 24), (3, 26), (5, 24)):
        pts.append(dict(tag=f'crop{start}_{n}', patch={},
                        crop=f'{start}:{n}'))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default=DEFAULT_STAGE)
    ap.add_argument('--key', default='sat_det3')
    ap.add_argument('--bundle', default='calib_bundle_lp2')
    ap.add_argument('--events', type=int, default=2000)
    ap.add_argument('--split', default='0:2',
                    help='I:N event split -- 0:2 scans, 1:2 validates')
    ap.add_argument('--only', nargs='*', default=None,
                    help='stage only these tags (validation pass)')
    ap.add_argument('--residual', type=int, default=0,
                    help='also emit N shards of the residual/goodness-of-fit '
                         'audit (bench/residual_audit.py)')
    args = ap.parse_args()

    from qa_config import get_config, setup_paths
    setup_paths()
    W = os.path.join(get_config(args.key).OUT_BASE, 'wft')
    cache, bdir = os.path.join(W, 'bench_cache.pkl'), os.path.join(W, args.bundle)
    for p in (cache, os.path.join(bdir, 'bundle.json')):
        if not os.path.exists(p):
            raise SystemExit(f'missing {p}')

    stage = args.stage
    for d in ('inputs', 'out', 'log'):
        os.makedirs(os.path.join(stage, d), exist_ok=True)

    dst = os.path.join(stage, 'inputs', f'{args.key}.pkl')
    if not os.path.exists(dst) or os.path.getmtime(cache) > os.path.getmtime(dst):
        shutil.copy2(cache, dst)
    btgz = os.path.join(stage, 'inputs', f'{args.key}__{args.bundle}.tgz')
    with tarfile.open(btgz, 'w:gz') as t:
        t.add(bdir, arcname='bundle')

    hyper = json.load(open(os.path.join(bdir, 'bundle.json')))['hyper']
    pts = scan_points(hyper)
    if args.only:
        pts = [p for p in pts if p['tag'] in set(args.only)]
        if not pts:
            raise SystemExit('no scan point matched --only')
    for p in pts:
        p['events'], p['split'] = args.events, args.split
        p['run_key'] = args.key
    scan = os.path.join(stage, 'inputs', 'scan_points.json')
    with open(scan, 'w') as f:
        json.dump(pts, f, indent=1)

    payload = os.path.join(stage, 'payload.tar.gz')
    with tarfile.open(payload, 'w:gz') as t:
        t.add(os.path.join(REPO, 'wft'), arcname='payload/wft',
              filter=lambda ti: None if '__pycache__' in ti.name else ti)
        for f in ('run_bench.py', 'residual_audit.py'):
            t.add(os.path.join(REPO, 'mx_june_wft', 'bench', f),
                  arcname=f'payload/mx_june_wft/bench/{f}')

    with open(os.path.join(stage, 'bench_jobs.txt'), 'w') as f:
        for i, p in enumerate(pts):
            f.write(f'{args.key}.pkl {args.key}__{args.bundle}.tgz '
                    f'scan_points.json {i} {p["tag"]}\n')

    # the goodness-of-fit audit: same inputs, sharded over events
    if args.residual:
        with open(os.path.join(stage, 'residual_jobs.txt'), 'w') as f:
            for i in range(args.residual):
                f.write(f'{args.key}.pkl {args.key}__{args.bundle}.tgz '
                        f'{i} {args.residual} {args.key}_{i}\n')
        print(f'  residual_jobs.txt: {args.residual} shards')

    for f in ('run_bench_job.sh', 'bench_scan.sub',
              'run_residual_job.sh', 'residual.sub'):
        src = os.path.join(HERE, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(stage, f))
            if f.endswith('.sh'):
                os.chmod(os.path.join(stage, f), 0o755)

    print(f'staged {len(pts)} scan points in {stage}\n'
          f'  cache  {os.path.getsize(dst)/1e6:.0f} MB\n'
          f'  events {args.events} of split {args.split}\n'
          f'next:\n  rsync -a {stage}/ lxplus:~/wft_bench/\n'
          '  ssh lxplus "cd ~/wft_bench && condor_submit bench_scan.sub"')


if __name__ == '__main__':
    main()
