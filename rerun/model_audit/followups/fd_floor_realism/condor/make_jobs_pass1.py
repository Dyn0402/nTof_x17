#!/usr/bin/env python3
"""make_jobs_pass1.py -- the FD toy campaign on the pass-1 basis (2026-10-31): job DAG, code tarball, bundles.

    make_jobs_pass1.py <staging dir> --code <code archive dir> --inputs toy_inputs_pass1.json
        -> <dir>/fd_code.tgz (wft + the toy + the five pass-1 calibration bundles + the inputs), <dir>/fd_pass1.dag

Jobs (334): the 46 scan_a runs and the 240 ladder runs of make_jobs.py, placement 'ig' (interior_good; the Paper I
sample), plus det7's 48 ladder runs again with placement 'full' (the sample whose dead-strip term needs det7 X ch < 256).
Every node RETRY 3 (EOS transients); a failed node is re-run by condor_submit_dag of the same .dag (rescue DAG).
"""
import argparse
import json
import os
import tarfile

DETS = ('det2', 'det3', 'det4', 'det6', 'det7')


def ladder_lines(dets):
    out = []
    for det in dets:
        for rung in ('L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8'):
            for ang in (3, 9, 16):
                out.append(f'ladder --det {det} --rungs {rung} --angles {ang} --n 150')
                if rung not in ('L0', 'L1'):
                    out.append(f'ladder --det {det} --rungs {rung} --angles {ang} --n 150 --alone')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('staging')
    ap.add_argument('--code', required=True)
    ap.add_argument('--inputs', required=True)
    a = ap.parse_args()
    st = a.staging
    os.makedirs(os.path.join(st, 'logs'), exist_ok=True)
    jobs = []
    for det in DETS:
        for b in (20, 30, 60, 90):
            for cfg in ('full', 'no_landau'):
                jobs.append(('ig', f'scan_a --det {det} --dtbin {b} --config {cfg} --n 100'))
    for p in (0.5, 1.2):
        for b in (30, 60):
            jobs.append(('ig', f'scan_a --det det3 --dtbin {b} --config full --pitch {p} --n 100'))
    for s0 in (0.10, 0.45):
        jobs.append(('ig', f'scan_a --det det3 --dtbin 60 --config full --s0 {s0} --n 100'))
    jobs += [('ig', l) for l in ladder_lines(DETS)]
    jobs += [('full', l) for l in ladder_lines(('det7',))]
    with open(os.path.join(st, 'fd_pass1.dag'), 'w') as f:
        for i, (place, args) in enumerate(jobs):
            f.write(f'JOB j{i} fd_pass1.sub\nVARS j{i} place="{place}" args="{args}"\nRETRY j{i} 3\n')
    inputs = json.load(open(a.inputs))
    with tarfile.open(os.path.join(st, 'fd_code.tgz'), 'w:gz') as t:
        t.add(os.path.join(a.code, 'wft'), 'wft', filter=lambda ti: None if '__pycache__' in ti.name or
              '/archive' in ti.name or '/tests' in ti.name else ti)
        t.add(os.path.join(a.code, 'rerun', 'model_audit', 'ws_floor', 'floor_toy.py'),
              'rerun/model_audit/ws_floor/floor_toy.py')
        for f in ('ladder.py', 'scan_a.py', 'pass1_basis.py'):
            t.add(os.path.join(a.code, 'rerun', 'model_audit', 'followups', 'fd_floor_realism', f),
                  f'rerun/model_audit/followups/fd_floor_realism/{f}')
        t.add(a.inputs, 'toy_inputs_pass1.json')
        for det, rec in inputs.items():
            for f in ('bundle.json', 'arrays.npz'):
                t.add(os.path.join(rec['wdir'], 'calib_bundle', f), f'bundles/{det}/calib_bundle/{f}')
    print(len(jobs), 'jobs')


if __name__ == '__main__':
    main()
