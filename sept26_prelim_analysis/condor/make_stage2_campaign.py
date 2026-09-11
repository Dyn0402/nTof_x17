#!/usr/bin/env python3
"""
make_stage2_campaign.py -- stage the CAMPAIGN stage-2 waveform reco for condor.

`ntof_tracking/condor/make_beam_package.py` builds a package for ONE sub-run.
The campaign is 293 of them, so this builds one package covering all of them:
`run_beam_job.py` already takes `--run`/`--subrun`, and `beam_reco.sub`'s
`$(extra)` was already written to absorb trailing arguments, so nothing in the
driver changes -- only the job list and how the allowlists are shipped.

Measured cost, from the run_145/stat090_0000 pass that already ran: 28 jobs,
median 2.3 min, max 5.7, 9.3 cpu-hours for ~47 k triggers.  Scaled to the
campaign's 25.6 M triggers that is ~8 000 short jobs and ~5 000 cpu-hours.

THE ONE CALIBRATION ASSUMPTION, STATED BECAUSE IT IS NOT FREE.  Every arm is
seeded from its run_145 bundle with `v_drift` pinned at 42.6 um/ns
(V_DRIFT_MAGBOLTZ), which is exactly how the published run_145 bundles were
built -- so `v` is a fixed constant here, not a per-run fit, and the bundle
carries no run_145-specific gas calibration into another run.  The real
per-run gas variation (S3: v falls down the A->B->C->D line, and the flush has
a 1.7 h lag) is absorbed DOWNSTREAM by `k_arm`, which measures the in-situ
angle scale per run.  That is what makes one bundle defensible across 36 runs
under CLAUDE.md's "per detector AND per run condition" rule -- the whole
in-sample set is one noise configuration and one gas mixture (Ar/Iso 90/10),
verified from `sample.csv`.
**The check that falsifies it: if `k` varies strongly run to run, the single
bundle is wrong and stage 2 has to be re-cut by condition.**  Nothing here
verifies that; it must be done on the campaign `k_arm` output.

    ../../.venv/bin/python sept26_prelim_analysis/condor/make_stage2_campaign.py
    rsync -av <dest>/ lxplus:~/sept26_stage2/
    ssh lxplus 'cd ~/sept26_stage2 && condor_submit stage2_campaign.sub'
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402

CODE_PATHS = ['sept26_prelim_analysis', 'ntof_tracking', 'wft', 'common',
              'mx17_m1_map.csv', 'mx17_m4_map.csv']
EXCLUDES = ['__pycache__', '*.pyc', '.venv', '.git', '*.parquet', '*.root',
            'figures', '*.png', '*.pdf']
V_DRIFT_PINNED = 42.6


def tags_from_stage1(stage1_dir, run, sub):
    """Every file tag of one sub-run, from its stage-1 candidate table.

    The candidate table has one row per DREAM trigger and carries `tag`, so
    its distinct tags ARE the sub-run's tags -- no EOS listing, no second
    source of truth, and nothing that can silently drop a tag the allowlist
    happened to empty.
    """
    import pandas as pd
    p = os.path.join(stage1_dir, f'candidates_{run}_{sub}.parquet')
    if not os.path.exists(p):
        return []
    return sorted(pd.read_parquet(p, columns=['tag']).tag.unique().tolist())


def main():
    ap = argparse.ArgumentParser()
    # Defaults are SPELLED, not resolved -- argparse builds them on every run,
    # --help included.
    ap.add_argument('--dest', default=str(paths.spell('x17', 'sept26_stage2')))
    ap.add_argument('--sample',
                    default=str(paths.spell('out', 'stage0', 'sample.csv')))
    ap.add_argument('--stage1',
                    default=str(paths.spell('out', 'stage1')))
    ap.add_argument('--stage2-out',
                    default=str(paths.spell('out', 'stage2')))
    ap.add_argument('--arms', default='A,B,C,D')
    ap.add_argument('--build-allowlists', action='store_true',
                    help='(re)build the per-sub-run allowlists from stage 1 '
                         'before packaging. Skips any sub-run with no census.')
    ap.add_argument('--allow-dirty', action='store_true')
    ap.add_argument('--allow-src', default=None,
                    help='directory holding allowlist_<run>_<sub>.json. '
                         'Default <out>/stage2. The calibration pass keeps its '
                         'own (higher SINGLE prescale) set elsewhere.')
    ap.add_argument('--subset', default=None,
                    help='JSON {run: [subruns]} limiting the pass to those '
                         'sub-runs -- how the calibration pass is scoped.')
    ap.add_argument('--full-pass', action='store_true',
                    help='NO allowlist: emit a job for every (run, sub-run, '
                         'arm, file tag) and fit every event the seeder finds. '
                         '~20x the compute of the allowlist pass. Tags come '
                         'from the stage-1 candidate tables, which cover every '
                         'trigger, so the job list is complete by construction '
                         'rather than limited to what the filter selected.')
    ap.add_argument('--done-list', default=None,
                    help='file of outnames already on EOS (one per line, with '
                         'or without .tar.gz). Those get no job, which is what '
                         'makes this safe to run in WAVES as stage 1 lands and '
                         'safe to re-run to pick up failures.')
    a = ap.parse_args()
    arms = a.arms.split(',')
    os.makedirs(os.path.join(a.dest, 'log'), exist_ok=True)

    import pandas as pd
    from sept26_prelim_analysis import allowlist as AL

    s = pd.read_csv(a.sample)
    want = [(f'run_{r["run"]}', r['subrun'])
            for _, r in s[s.in_sample].sort_values(['run', 'subrun']).iterrows()]
    if a.subset:
        sub = json.load(open(a.subset))
        keep = {(r, x) for r, xs in sub.items() for x in xs}
        want = [w for w in want if w in keep]
        print(f'subset             {len(want)} sub-run(s)')
    allow_src = a.allow_src or a.stage2_out

    # ---- allowlists
    if a.build_allowlists:
        built = skipped = 0
        for run, sub in want:
            cand = os.path.join(a.stage1, f'candidates_{run}_{sub}.parquet')
            if not os.path.exists(cand):
                skipped += 1
                continue
            try:
                AL.build(run, sub, stage1_dir=__import__('pathlib').Path(a.stage1))
                built += 1
            except Exception as exc:                      # noqa: BLE001
                print(f'  !! allowlist {run}/{sub}: {exc}')
                skipped += 1
        print(f'allowlists: built {built}, skipped {skipped} '
              f'(no stage-1 census yet)')

    # ---- package the allowlists that exist
    adir = os.path.join(a.dest, 'allow')
    shutil.rmtree(adir, ignore_errors=True)
    os.makedirs(adir)
    have = []
    for run, sub in want:
        src = os.path.join(allow_src, f'allowlist_{run}_{sub}.json')
        if os.path.exists(src):
            shutil.copy(src, os.path.join(adir, os.path.basename(src)))
            have.append((run, sub))
    subprocess.run(['tar', 'czf', os.path.join(a.dest, 'allowlists.tar.gz'),
                    '--transform', 's,^allow,allow,', '-C', a.dest, 'allow'],
                   check=True)
    print(f'allowlists.tar.gz  {len(have)}/{len(want)} sub-runs')
    if a.full_pass:
        # A full pass needs no allowlist, but it still needs the sub-run list.
        # Stage 1 is what says a sub-run exists and is readable, so the
        # candidate table -- not the allowlist -- is the gate here.
        have = [(r, x) for r, x in want
                if os.path.exists(os.path.join(a.stage1,
                                               f'candidates_{r}_{x}.parquet'))]
        print(f'FULL PASS          {len(have)}/{len(want)} sub-runs have a '
              f'stage-1 candidate table')
    if not have:
        sys.exit('FATAL: no allowlists -- run with --build-allowlists once '
                 'stage 1 has written censuses.')

    # ---- jobs: one per (run, subrun, arm, tag) the allowlist actually selected
    # Bundles FIRST: the job list needs each arm's bundle NAME.
    from ntof_tracking.wft_beam import BEAM_DETS
    bdir = os.path.join(a.dest, 'bundles')
    shutil.rmtree(bdir, ignore_errors=True)
    names = {}
    for arm in arms:
        src = BEAM_DETS[arm]['bundle']
        if not os.path.isdir(src):
            sys.exit(f'FATAL: arm {arm} bundle missing: {src}')
        b = json.load(open(os.path.join(src, 'bundle.json')))
        hy = b['hyper']
        r = hy.get('c2_over_c1')
        c2 = float(r) * hy['c1'] if r is not None else hy['c2']
        # CLAUDE.md: the +-2 strip is reached only through the +-1. Every
        # bundle shipped before 2026-08-21 had this backwards; refusing here
        # keeps a retired bundle from reaching 8 000 jobs.
        if c2 > hy['c1']:
            sys.exit(f'FATAL: arm {arm} bundle {src} has c2 > c1 '
                     f'({c2:.4f} > {hy["c1"]:.4f}) -- inverted sharing ladder.')
        if b.get('provenance', {}).get('w0_kw_stale'):
            sys.exit(f'FATAL: arm {arm} bundle {src} is stamped w0_kw_stale')
        shutil.copytree(src, os.path.join(bdir, f'mx17_{arm}',
                                          os.path.basename(src)))
        names[arm] = os.path.basename(src)
    subprocess.run(['tar', 'czf', os.path.join(a.dest, 'bundles.tar.gz'),
                    '-C', a.dest, 'bundles'], check=True)
    print(f'bundles.tar.gz     {names}')


    done = set()
    if a.done_list and os.path.exists(a.done_list):
        for line in open(a.done_list):
            t = line.strip()
            if t:
                done.add(t[:-7] if t.endswith('.tar.gz') else t)
        print(f'already on EOS  {len(done)} job output(s) -- skipped')

    rows = []
    skipped_done = 0
    for run, sub in have:
        if a.full_pass:
            all_tags = tags_from_stage1(a.stage1, run, sub)
            sel = {arm: {t: [1] for t in all_tags} for arm in arms}
        else:
            sel = AL.load(os.path.join(adir, f'allowlist_{run}_{sub}.json'))
        for arm in arms:
            for tag, events in sorted(sel.get(arm, {}).items()):
                if not events:
                    continue          # no job for a tag the selection emptied
                out = f'{run}_{sub}_beam_{arm}_{tag}'
                if out in done:
                    skipped_done += 1
                    continue
                # --bundle-name is REQUIRED, not cosmetic: run_beam_job.py
                # defaults it to calib_bundle_r06, which is correct for A, B
                # and D and WRONG for C (calib_bundle_lp). Leaving it out held
                # every arm-C job with a missing-bundle FileNotFoundError.
                allow = ('' if a.full_pass else
                         f'--allow allow/allowlist_{run}_{sub}.json ')
                extra = (f'--run {run} --subrun {sub} '
                         f'{allow}'
                         f'--bundle-name {names[arm]} '
                         f'--v-drift {V_DRIFT_PINNED}')
                # AFS caps directory entries and a flat 22k-file log dir
                # degraded the shared schedd once already, so stderr is
                # sharded by run -- ~36 directories of a few hundred.
                rows.append(f'{arm},{tag},{out},{run},{extra}'
                            if a.full_pass else f'{arm},{tag},{out},{extra}')
    with open(os.path.join(a.dest, 'jobs.txt'), 'w') as f:
        f.write('\n'.join(rows) + '\n')
    print(f'jobs.txt           {len(rows)} jobs'
          + (f'  ({skipped_done} already done)' if skipped_done else ''))
    if not rows:
        print('  nothing left to submit for the sub-runs that have a census')

    # ---- code + bundles
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
        f.write(f'campaign stage-2 reco package\n'
                f'git commit    {commit}\n'
                f'code.tar.gz   sha256 {h}\n'
                f'source        WORKING TREE, not `git archive`\n'
                f'bundles       {json.dumps(names)}\n'
                f'v_drift       {V_DRIFT_PINNED} um/ns, PINNED for every arm\n'
                f'sub-runs      {len(have)} of {len(want)}\n'
                f'jobs          {len(rows)}\n\n'
                f'uncommitted under the shipped code paths:\n'
                f'{dirty or "  (none)"}\n')

    shutil.copy(os.path.join(REPO, 'ntof_tracking/condor/run_beam_job.py'),
                os.path.join(a.dest, 'run_beam_job.py'))
    ship = (('stage2_fullpass.sub', 'run_stage2_fullpass_wrapper.sh')
            if a.full_pass else
            ('stage2_campaign.sub', 'run_stage2_wrapper.sh'))
    for fn in ship:
        shutil.copy(os.path.join(HERE, fn), os.path.join(a.dest, fn))
    os.chmod(os.path.join(a.dest, ship[1]), 0o755)
    if a.full_pass:
        for run, _ in have:
            os.makedirs(os.path.join(a.dest, 'log', run), exist_ok=True)
    print(f'-> {a.dest}')


if __name__ == '__main__':
    raise SystemExit(main())
