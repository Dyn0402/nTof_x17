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

``--hot`` ships a hot-channel wildcard JSON (sept26_prelim_analysis/
apply_hot_wildcards.py --export-json) the same way ``--allow`` ships a stage-2
allowlist; each job merges only its own arm's entry into the seeded bundle's
``hot`` field (HANDOFF_D_NOISY_CHANNELS.md).

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
    ap.add_argument('--allow', default=None,
                    help='stage-2 allowlist JSON (sept26_prelim_analysis/'
                         'allowlist.py). Ships with the package, and the job '
                         'list is built FROM it: an (arm, tag) the selection '
                         'chose nothing in gets no job, rather than a job that '
                         'runs and writes an empty table.')
    ap.add_argument('--v-drift', type=float, default=42.6,
                    help='pinned for every arm, as the published run_145 '
                         'bundles were (V_DRIFT_MAGBOLTZ, not the per-arm '
                         'prior) — so the kernel is the only thing that moved.')
    ap.add_argument('--hot', default=None,
                    help='hot-channel wildcard JSON (sept26_prelim_analysis/'
                         'apply_hot_wildcards.py --export-json), '
                         '{arm: {plane: [channels]}}. Ships with the package; '
                         'each job applies only its own arm\'s entry '
                         '(HANDOFF_D_NOISY_CHANNELS.md).')
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

    # ---- allowlist
    allow_name, allow_doc = None, None
    if a.allow:
        allow_doc = json.load(open(a.allow))
        if allow_doc.get('run') != a.run or allow_doc.get('subrun') != a.subrun:
            sys.exit(f'FATAL: allowlist is for {allow_doc.get("run")}/'
                     f'{allow_doc.get("subrun")}, package is for {a.run}/'
                     f'{a.subrun}. Event ids are per sub-run; crossing them '
                     'would fit the wrong triggers and say nothing about it.')
        allow_name = os.path.basename(a.allow)
        shutil.copy2(a.allow, os.path.join(a.dest, allow_name))
        c = allow_doc['counts']
        print(f'allowlist {allow_name}: {c["n_arm_events"]:,} (arm, event) fits '
              f'of {c["n_arm_events_full_reco"]:,} '
              f'({c["n_arm_events"] / c["n_arm_events_full_reco"]:.2%} of a full '
              f'reco); prescale {allow_doc["policy"]["control_prescale"]}')

    # ---- hot-channel wildcards
    hot_name, hot_doc = None, None
    if a.hot:
        hot_doc = json.load(open(a.hot))
        hot_name = os.path.basename(a.hot)
        shutil.copy2(a.hot, os.path.join(a.dest, hot_name))
        for arm in arms:
            h = hot_doc.get(arm, {})
            nx, ny = len(h.get('x', [])), len(h.get('y', []))
            if nx or ny:
                print(f'hot wildcards {hot_name}: arm {arm} -> x={nx} y={ny}')

    # ---- jobs
    jobs = os.path.join(a.dest, 'jobs.txt')
    n_jobs, n_skip, n_ev = 0, 0, 0
    with open(jobs, 'w') as f:
        for arm in arms:
            for tag in tags:
                extra = (f'--run {a.run} --subrun {a.subrun} '
                         f'--bundle-name {names[arm]} --v-drift {a.v_drift}')
                if allow_doc is not None:
                    n = len(allow_doc['events'].get(arm, {}).get(tag, []))
                    if not n:
                        n_skip += 1
                        continue
                    n_ev += n
                    extra += f' --allow {allow_name}'
                if hot_name is not None:
                    extra += f' --hot {hot_name}'
                f.write(f'{arm},{tag},{extra}\n')
                n_jobs += 1
    print(f'jobs.txt: {n_jobs} jobs of {len(arms) * len(tags)} '
          f'({len(arms)} arms x {len(tags)} tags)'
          + (f'; {n_skip} skipped (nothing selected), {n_ev:,} events queued'
             if allow_doc is not None else ''))

    for f in ('beam_reco.sub', 'run_beam_wrapper.sh', 'run_beam_job.py'):
        shutil.copy2(os.path.join(HERE, f), os.path.join(a.dest, f))
    if allow_name:
        # Condor transfers only what the submit file names. Patch the default
        # rather than ask the submitter to remember `-a`: a job that runs
        # without its allowlist reconstructs the whole tag and looks fine.
        p = os.path.join(a.dest, 'beam_reco.sub')
        txt = open(p).read()
        old = '  allowfile             =\n'
        if old not in txt:
            sys.exit(f'FATAL: cannot find the allowfile default in {p}')
        open(p, 'w').write(txt.replace(old, f'  allowfile             = , {allow_name}\n'))
        print(f'beam_reco.sub: transfers {allow_name}')
    if hot_name:
        p = os.path.join(a.dest, 'beam_reco.sub')
        txt = open(p).read()
        old = '  hotfile               =\n'
        if old not in txt:
            sys.exit(f'FATAL: cannot find the hotfile default in {p}')
        open(p, 'w').write(txt.replace(old, f'  hotfile               = , {hot_name}\n'))
        print(f'beam_reco.sub: transfers {hot_name}')
    os.chmod(os.path.join(a.dest, 'run_beam_wrapper.sh'), 0o755)
    print('package at', a.dest)
    print(f'  rsync -av {a.dest}/ lxplus:~/{os.path.basename(a.dest)}/')
    print(f'  ssh lxplus "cd ~/{os.path.basename(a.dest)} && '
          'condor_submit beam_reco.sub"')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
