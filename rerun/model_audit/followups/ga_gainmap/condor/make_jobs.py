#!/usr/bin/env python3
"""make_jobs.py <staging dir> [--arms A,B] [--dets ...] -> <dir>/jobs.txt, <dir>/ga_code.tgz  (GA toy on condor, ga.sub)"""
import argparse, os, tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
GA = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(GA, '..', '..', '..', '..'))
AUDIT = os.path.expanduser('~/x17/cosmic_bench/model_audit')
DETS = ('det2', 'det3', 'det4', 'det6', 'det7')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('st')
    ap.add_argument('--arms', default=None); ap.add_argument('--dets', default=','.join(DETS))
    ap.add_argument('--angles', default='0,3,9,16,25'); ap.add_argument('--n', type=int, default=150)
    a = ap.parse_args()
    os.makedirs(a.st, exist_ok=True)
    lines = []
    for det in a.dets.split(','):
        white = 'white31' if det == 'det4' else 'white06'
        arms = a.arms.split(',') if a.arms else ['none', white, 'meas', 'meas_strip', 'meas_large', 'meas_known']
        for arm in arms:
            for ang in a.angles.split(','):
                lines.append(f'--det {det} --arms {arm} --angles {ang} --n {a.n}')
    open(os.path.join(a.st, 'jobs.txt'), 'w').write('\n'.join(lines) + '\n')
    with tarfile.open(os.path.join(a.st, 'ga_code.tgz'), 'w:gz') as t:
        t.add(os.path.join(REPO, 'wft'), 'wft', filter=lambda ti: None if '__pycache__' in ti.name or
              '/archive' in ti.name or '/tests' in ti.name else ti)
        t.add(os.path.join(REPO, 'rerun', 'model_audit', 'ws_floor', 'floor_toy.py'), 'rerun/model_audit/ws_floor/floor_toy.py')
        t.add(os.path.join(REPO, 'rerun', 'model_audit', 'followups', 'fd_floor_realism', 'ladder.py'),
              'rerun/model_audit/followups/fd_floor_realism/ladder.py')
        t.add(os.path.join(GA, 'ga_toy.py'), 'rerun/model_audit/followups/ga_gainmap/ga_toy.py')
        for f in os.listdir(os.path.join(GA, 'results')):
            if f == 'chargemap.json' or (f.startswith('gainfit_') and f.endswith('.npz')):
                t.add(os.path.join(GA, 'results', f), f'rerun/model_audit/followups/ga_gainmap/results/{f}')
        for det in DETS:
            for f in ('bundle.json', 'arrays.npz'):
                t.add(os.path.join(AUDIT, det, 'calib_bundle', f), f'bundles/{det}/calib_bundle/{f}')
    print(len(lines), 'jobs')


if __name__ == '__main__':
    main()
