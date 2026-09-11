#!/usr/bin/env python3
"""
make_stage1_package.py -- stage the stage-1 candidate filter for lxplus condor.

Why this exists: ``campaign_census.sh`` runs stage 1 on Dylan's desktop and
streams each sub-run down from EOS first.  Its own header puts that at 17-25 h
across 8 workers -- "a multi-night job, not an overnight one".  The data is
already at CERN and the filter is embarrassingly parallel per sub-run, so one
condor job per sub-run turns the same work into ~1.5 h wall.

Nothing in ``candidate_filter`` changes: ``paths.py`` already resolves every
root through an environment variable, so the worker points ``X17_RUNS`` at its
own staged copy and runs the unmodified CLI.

Builds <dest>/:
    code.tar.gz     the code the job imports, FROM THE WORKING TREE
    PROVENANCE.txt  git commit, uncommitted files, sha256 of code.tar.gz
    jobs.txt        run,subrun -- the frozen sample minus what is already done
    stage1.sub, run_stage1_wrapper.sh, log/

**The tarball is the working tree, not ``git archive HEAD``.**  The beam
package refuses to build dirty because a ``git archive`` would silently ship
the old code.  Here the tree carries uncommitted analysis modules that are part
of the run, and committing on Dylan's behalf is not this script's call, so it
ships what is actually on disk and records exactly what that was.  PROVENANCE
names the commit, every uncommitted file, and the tarball hash -- which is
reproducible provenance, and is checked into the product directory.

    ../../.venv/bin/python sept26_prelim_analysis/condor/make_stage1_package.py
    rsync -av <dest>/ lxplus:~/sept26_stage1/
    ssh lxplus 'cd ~/sept26_stage1 && condor_submit stage1.sub'
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

#: Everything candidate_filter imports, transitively. ntof_tracking.reco.io /
#: noise / segments, the sept26 package itself, and the strip maps under common.
CODE_PATHS = ['sept26_prelim_analysis', 'ntof_tracking', 'wft', 'common',
              'mx17_m1_map.csv', 'mx17_m4_map.csv']

EXCLUDES = ['__pycache__', '*.pyc', '.venv', '.git', '*.parquet', '*.root',
            'figures', '*.png', '*.pdf']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default='/home/dylan/x17/sept26_stage1')
    ap.add_argument('--sample',
                    default='/media/dylan/data/x17/sept26_prelim/stage0/sample.csv')
    ap.add_argument('--done-dir',
                    default='/media/dylan/data/x17/sept26_prelim/stage1',
                    help='sub-runs with a census CSV here are already done and '
                         'get no job -- this is what makes the pass resumable')
    ap.add_argument('--all', action='store_true',
                    help='job for every sub-run in the sample, done or not')
    a = ap.parse_args()

    os.makedirs(os.path.join(a.dest, 'log'), exist_ok=True)

    # ---- code, from the working tree
    tgz = os.path.join(a.dest, 'code.tar.gz')
    cmd = ['tar', 'czf', tgz, '--transform', 's,^,code/,']
    for e in EXCLUDES:
        cmd += ['--exclude', e]
    cmd += ['-C', REPO] + CODE_PATHS
    subprocess.run(cmd, check=True)
    h = hashlib.sha256(open(tgz, 'rb').read()).hexdigest()

    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                            capture_output=True, text=True,
                            check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--'] + CODE_PATHS,
                           cwd=REPO, capture_output=True, text=True,
                           check=True).stdout.strip()
    with open(os.path.join(a.dest, 'PROVENANCE.txt'), 'w') as f:
        f.write(f'stage-1 campaign census package\n'
                f'built            {__import__("datetime").datetime.now().isoformat()}\n'
                f'git commit       {commit}\n'
                f'code.tar.gz      sha256 {h}\n'
                f'source           WORKING TREE, not `git archive` -- the tree\n'
                f'                 carried uncommitted modules at build time.\n\n'
                f'uncommitted under the shipped code paths:\n'
                f'{dirty or "  (none -- tree is clean, tarball == commit)"}\n')
    print(f'code.tar.gz  {os.path.getsize(tgz) / 1e6:.1f} MB  sha256 {h[:12]}')
    print(f'  commit {commit[:9]}' + ('  +uncommitted' if dirty else '  (clean)'))

    # ---- job list
    import pandas as pd
    s = pd.read_csv(a.sample)
    k = s[s.in_sample].sort_values(['run', 'subrun'])
    rows, skipped = [], 0
    for _, r in k.iterrows():
        run, sub = f'run_{r["run"]}', r['subrun']
        census = os.path.join(a.done_dir, f'census_{run}_{sub}.csv')
        if not a.all and os.path.exists(census) and os.path.getsize(census):
            skipped += 1
            continue
        rows.append(f'{run},{sub}')
    with open(os.path.join(a.dest, 'jobs.txt'), 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print(f'jobs.txt     {len(rows)} jobs  ({skipped} already done, skipped)')

    for fn in ('stage1.sub', 'run_stage1_wrapper.sh'):
        shutil.copy(os.path.join(HERE, fn), os.path.join(a.dest, fn))
    os.chmod(os.path.join(a.dest, 'run_stage1_wrapper.sh'), 0o755)
    print(f'-> {a.dest}')


if __name__ == '__main__':
    raise SystemExit(main())
