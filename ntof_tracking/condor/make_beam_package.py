#!/usr/bin/env python3
"""
make_beam_package.py — stage a beam wft reconstruction for lxplus condor.

Why this exists: the beam waveforms live on EOS at CERN (6 GB per sub-run) and
the laptop that owns this analysis was, on 2026-08-19, on a 0.3 MB/s link.
Bringing the data home is a 10 h download; running the reco where the data
already is and bringing back the ~1 MB/tag parquet is minutes.  Same driver
(`ntof_tracking.wft_beam`), one job per (arm, file tag).

Builds <dest>/:
    code.tar.gz      git archive of HEAD, only the dirs the beam reco imports
    bundles.tar.gz   bundles/mx17_<arm>/<bundle>/ — the BENCH bundle each arm
                     is seeded from
    jobs.txt         arm,tag,extra
    beam_reco.sub, run_beam_wrapper.sh, run_beam_job.py, log/

Then:
    rsync -av <dest>/ lxplus:~/wft_beam145/
    ssh lxplus 'cd ~/wft_beam145 && condor_submit beam_reco.sub'

    ../../.venv/bin/python ntof_tracking/condor/make_beam_package.py
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

CODE_PATHS = ['wft', 'ntof_tracking', 'common', 'mx17_m1_map.csv']

# run_145/stat090_0000's seven file tags (EOS listing, 2026-08-19).
TAGS_RUN145 = [f'260805_14H06_{i:03d}' for i in range(7)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default='/home/dylan/x17/wft_beam145')
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subrun', default='stat090_0000')
    ap.add_argument('--arms', default='A,B,D',
                    help='arms to run. Default A,B,D: the three seeded from a '
                         'bundle whose sharing kernel was inverted. C (det6) '
                         'was already physical and is NOT re-run.')
    ap.add_argument('--tags', default=','.join(TAGS_RUN145))
    ap.add_argument('--v-drift', type=float, default=42.6,
                    help='pinned for every arm, as the published run_145 '
                         'bundles were (V_DRIFT_MAGBOLTZ, not the per-arm '
                         'prior) — so the kernel is the only thing that moved.')
    ap.add_argument('--allow-dirty', action='store_true')
    a = ap.parse_args()

    from ntof_tracking.wft_beam import BEAM_DETS
    arms = a.arms.split(',')
    tags = a.tags.split(',')

    os.makedirs(os.path.join(a.dest, 'log'), exist_ok=True)

    # ---- code
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                            capture_output=True, text=True,
                            check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--'] + CODE_PATHS,
                           cwd=REPO, capture_output=True, text=True,
                           check=True).stdout.strip()
    if dirty and not a.allow_dirty:
        sys.exit('FATAL: uncommitted changes under the code paths — the job '
                 'runs `git archive`, so anything uncommitted would NOT ship '
                 'and the workers would silently run the old code:\n' + dirty)
    with open(os.path.join(a.dest, 'code.tar.gz'), 'wb') as f:
        subprocess.run(['git', 'archive', '--format=tar.gz', '--prefix=code/',
                        commit, '--'] + CODE_PATHS, cwd=REPO, stdout=f,
                       check=True)
    with open(os.path.join(a.dest, 'CODE_COMMIT.txt'), 'w') as f:
        f.write(commit + '\n')
    print(f'code.tar.gz at {commit[:9]}')

    # ---- bundles: whatever BEAM_DETS currently points each arm at
    bdir = os.path.join(a.dest, 'bundles')
    shutil.rmtree(bdir, ignore_errors=True)
    names = {}
    for arm in arms:
        src = BEAM_DETS[arm]['bundle']
        if not os.path.isdir(src):
            sys.exit(f'FATAL: arm {arm} bundle missing: {src}')
        b = json.load(open(os.path.join(src, 'bundle.json')))
        h = b['hyper']
        r = h.get('c2_over_c1')
        c2 = float(r) * h['c1'] if r is not None else h['c2']
        if c2 > h['c1']:
            sys.exit(f'FATAL: arm {arm} bundle {src} has c2 > c1 '
                     f'({c2:.4f} > {h["c1"]:.4f}) — an inverted sharing '
                     'ladder. The +-2 strip is reached only through the +-1; '
                     'this is the defect the r06 refit exists to fix.')
        if b.get('provenance', {}).get('w0_kw_stale'):
            sys.exit(f'FATAL: arm {arm} bundle {src} is stamped w0_kw_stale')
        names[arm] = os.path.basename(src)
        shutil.copytree(src, os.path.join(bdir, f'mx17_{arm}', names[arm]))
        print(f'  arm {arm}: {names[arm]}  c2/c1={c2 / h["c1"]:.3f}  '
              f'v_bench={b["v_drift"]:.2f}  (pinned to {a.v_drift} for the beam)')
    with tarfile.open(os.path.join(a.dest, 'bundles.tar.gz'), 'w:gz') as t:
        t.add(bdir, arcname='bundles')
    shutil.rmtree(bdir)

    # ---- jobs
    jobs = os.path.join(a.dest, 'jobs.txt')
    with open(jobs, 'w') as f:
        for arm in arms:
            for tag in tags:
                extra = (f'--run {a.run} --subrun {a.subrun} '
                         f'--bundle-name {names[arm]} --v-drift {a.v_drift}')
                f.write(f'{arm},{tag},{extra}\n')
    print(f'jobs.txt: {len(arms) * len(tags)} jobs '
          f'({len(arms)} arms x {len(tags)} tags)')

    for f in ('beam_reco.sub', 'run_beam_wrapper.sh', 'run_beam_job.py'):
        shutil.copy2(os.path.join(HERE, f), os.path.join(a.dest, f))
    os.chmod(os.path.join(a.dest, 'run_beam_wrapper.sh'), 0o755)
    print('package at', a.dest)
    print(f'  rsync -av {a.dest}/ lxplus:~/{os.path.basename(a.dest)}/')
    print(f'  ssh lxplus "cd ~/{os.path.basename(a.dest)} && '
          'condor_submit beam_reco.sub"')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
