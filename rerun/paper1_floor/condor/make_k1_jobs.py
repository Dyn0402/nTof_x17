#!/usr/bin/env python3
"""make_k1_jobs.py -- the K1 test campaign (rerun/paper1_floor/K1_TEST_2026-11-01.md): DAG + code tarball.

    make_k1_jobs.py <staging dir> --repo <checkout> --bundles <dir with <det>/calib_bundle> [--n 600]

Per chamber, view and angle (3 / 9 / 16 deg):
  - L8 (the full toy, the baseline: re-run so its rows carry the residual metric and q_sum);
  - every K1 variant at three sizes: G / S / W / L at 0.03, 0.06, 0.10 (fractions) and T at 5, 10, 20 ns;
  - the K1-aware fit (MODEL_FRAC = 0.06) on L8 and on every variant at its middle size.
Every node RETRY 3 (EOS transients).
"""
import argparse
import os
import tarfile

DETS = ('det2', 'det3', 'det4', 'det6', 'det7')
AMPS = dict(G=(0.03, 0.06, 0.10), S=(0.03, 0.06, 0.10), W=(0.03, 0.06, 0.10), L=(0.03, 0.06, 0.10), T=(5, 10, 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('staging')
    ap.add_argument('--repo', required=True)
    ap.add_argument('--bundles', required=True)
    ap.add_argument('--n', type=int, default=600)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.staging, 'logs'), exist_ok=True)
    jobs = []
    for det in DETS:
        for view in ('x', 'y'):
            for ang in (3, 9, 16):
                base = f'--det {det} --angles {ang} --n {a.n} --jobs 1'
                jobs.append((view, 0, 0.06, f'--rungs L8 {base}'))
                jobs.append((view, 0, 0.06, f'--rungs L8w {base}'))
                for v, amps in AMPS.items():
                    for amp in amps:
                        jobs.append((view, amp, 0.06, f'--rungs K1{v} {base}'))
                    jobs.append((view, amps[1], 0.06, f'--rungs K1{v}w {base}'))
    with open(os.path.join(a.staging, 'k1.dag'), 'w') as f:
        for i, (view, amp, wt, args) in enumerate(jobs):
            f.write(f'JOB j{i} k1.sub\nVARS j{i} view="{view}" amp="{amp}" wt="{wt}" args="{args}"\nRETRY j{i} 3\n')
    R = a.repo
    with tarfile.open(os.path.join(a.staging, 'fd_code.tgz'), 'w:gz') as t:
        t.add(os.path.join(R, 'wft'), 'wft', filter=lambda ti: None if '__pycache__' in ti.name or
              '/archive' in ti.name or '/tests' in ti.name else ti)
        t.add(os.path.join(R, 'rerun', 'model_audit', 'ws_floor', 'floor_toy.py'), 'rerun/model_audit/ws_floor/floor_toy.py')
        for f in ('ladder.py', 'scan_a.py', 'pass1_basis.py'):
            t.add(os.path.join(R, 'rerun', 'model_audit', 'followups', 'fd_floor_realism', f),
                  f'rerun/model_audit/followups/fd_floor_realism/{f}')
        t.add(os.path.join(R, 'rerun', 'model_audit', 'followups', 'fd_floor_realism', 'results', 'toy_inputs_pass1.json'),
              'toy_inputs_pass1.json')
        for det in DETS:
            for f in ('bundle.json', 'arrays.npz'):
                t.add(os.path.join(a.bundles, det, 'calib_bundle', f), f'bundles/{det}/calib_bundle/{f}')
    print(len(jobs), 'jobs')


if __name__ == '__main__':
    main()
