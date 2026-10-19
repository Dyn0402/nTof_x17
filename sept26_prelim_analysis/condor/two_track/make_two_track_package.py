#!/usr/bin/env python3
"""
make_two_track_package.py -- stage the two-track validation for lxplus condor.

The long runs of TWO_TRACK_LIMIT_RESUME.md (overlay bench of the fixed chain,
split-ab on real triggers), sharded by file tag -- and split-ab of the fixed
chain further by event id, since one tag of it is ~50 core-hours. All seven
tags of run_145/stat090_0000, not the one tag the laptop could afford.

Builds <dest>/:
    code.tar.gz          the WORKING TREE (the chain is uncommitted), + PROVENANCE.txt
    inputs.tar.gz        out/ subset every job reads: reco_fullpass products and
                         bundles of A and C, stage-3 tracks, x/y pairing JSONs.
                         Goes to EOS (TT_INPUTS), not through the schedd.
    jobs.txt             kind arm tag outname flavour extra...
    two_track.sub, run_two_track_wrapper.sh, run_two_track_job.py, log/

(--ladder: only the split-ab F rescan, into ~/x17/two_track_ladder_condor;
stage it at lxplus:~/two_track_ladder/, results share the EOS results dir.)

Then:
    xrdcp <dest>/inputs.tar.gz root://eosuser.cern.ch//eos/user/d/dneff/x17/two_track_limit/
    rsync -av --exclude inputs.tar.gz <dest>/ lxplus:~/two_track_limit/
    ssh lxplus 'cd ~/two_track_limit && condor_submit two_track.sub'
and, when done, merge_two_track.py.

    .venv/bin/python sept26_prelim_analysis/condor/two_track/make_two_track_package.py
"""
import argparse
import datetime
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
from sept26_prelim_analysis import paths  # noqa: E402

RUN, SUBRUN = 'run_145', 'stat090_0000'
CODE_PATHS = ['sept26_prelim_analysis', 'ntof_tracking', 'wft', 'common',
              'mx17_m1_map.csv', 'mx17_m4_map.csv']
EXCLUDES = ['__pycache__', '*.pyc', '.venv', '.git', '*.parquet', '*.root',
            'figures', '*.png', '*.pdf']

FIX = ('--worker-opt TWO_TRACK_SCALE=two --worker-opt TWO_TRACK_SEARCH=grid '
       '--worker-opt TWO_TRACK_RESID_Z=-inf --worker-opt TWO_TRACK_MAX_TRY=99 '
       '--worker-opt TWO_TRACK_SELECTED_ONLY=false')
BENCH = ('--pairing --local-mm 16 --local-mode rescue --two-track --two-track-t0 tied '
         '--two-track-resid-z 8 --overlay replace')
#: thresholds matched to production's false-split rate on real singles
#: (TWO_TRACK_FIT_LOG, 2026-09-30): (F, F_corroborated)
MATCHED_F = {'A': (1200, 480), 'C': (2400, 960)}
SPLITAB_SHARDS = 8
#: the split-ab F rescan (--ladder): every F replayed in one pass by
#: wft.reco.two_track_ladder; corroborated F at the matched 0.4 ratio. The
#: primary F stays the matched one, so the ladder's entry there must reproduce
#: split_ab_fixed_<arm>_7tags exactly.
LADDER_F = (300, 400, 600, 800, 1000, 1200, 1600, 2000, 2400, 3200, 4800)


def job_list(tags, have_local):
    """(kind, arm, tag, outname, flavour, extra) -- everything still outstanding."""
    rows = []
    for arm, (f, fc) in MATCHED_F.items():
        fixed = f'--two-track-f {f} --two-track-f-corrob {fc} {FIX}'
        benches = [(f'fixed_{arm}_replace', ''), (f'fixed_{arm}_replace_profc', '--pairing-tag profc')]
        for variant, extra in benches:
            if variant in have_local:
                continue
            for tag in tags:
                rows.append(('bench', arm, tag, f'bench_{variant}_{tag}', 'workday',
                             f'--variant {variant} {BENCH} {extra} {fixed}'.replace('  ', ' ')))
        for tag in tags:
            for i in range(SPLITAB_SHARDS):
                rows.append(('splitab', arm, tag,
                             f'splitab_fixed_{arm}_{tag}_s{i}of{SPLITAB_SHARDS}', 'tomorrow',
                             f'--pairing --variant fixed_{arm} --shard {i}/{SPLITAB_SHARDS} '
                             f'--worker-opt TWO_TRACK_F={f} --worker-opt TWO_TRACK_F_CORROB={fc} '
                             f'{FIX}'))
            rows.append(('splitab', arm, tag, f'splitab_current_{arm}_{tag}', 'workday',
                         f'--pairing --variant current_{arm} --worker-opt TWO_TRACK_F=300 '
                         '--worker-opt TWO_TRACK_F_CORROB=120'))
    return rows


def ladder_job_list(tags):
    """split-ab of the fixed chain, every tag and shard, with the F ladder on."""
    lad = ','.join(str(f) for f in LADDER_F)
    rows = []
    for arm, (f, fc) in MATCHED_F.items():
        for tag in tags:
            for i in range(SPLITAB_SHARDS):
                rows.append(('splitab', arm, tag,
                             f'splitab_ladder_{arm}_{tag}_s{i}of{SPLITAB_SHARDS}', 'tomorrow',
                             f'--pairing --variant ladder_{arm} --shard {i}/{SPLITAB_SHARDS} '
                             f'--worker-opt TWO_TRACK_F={f} --worker-opt TWO_TRACK_F_CORROB={fc} '
                             f'--worker-opt TWO_TRACK_F_LADDER={lad} {FIX}'))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default=None,
                    help='default ~/x17/two_track_condor (~/x17/two_track_ladder_condor with --ladder)')
    ap.add_argument('--ladder', action='store_true',
                    help='only the split-ab F rescan of the fixed chain (LADDER_F)')
    a = ap.parse_args()
    a.dest = a.dest or str(paths.spell('x17', 'two_track_ladder_condor' if a.ladder
                                       else 'two_track_condor'))
    os.makedirs(os.path.join(a.dest, 'log'), exist_ok=True)

    # ---- code, from the working tree
    tgz = os.path.join(a.dest, 'code.tar.gz')
    cmd = ['tar', 'czf', tgz, '--transform', 's,^,code/,']
    for e in EXCLUDES:
        cmd += ['--exclude', e]
    subprocess.run(cmd + ['-C', REPO] + CODE_PATHS, check=True)
    h = hashlib.sha256(open(tgz, 'rb').read()).hexdigest()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True,
                            text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--'] + CODE_PATHS, cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.strip()

    # ---- inputs: the out/ subset, laid out as sept26_prelim
    out = paths.spell('out')
    members = []
    for arm in MATCHED_F:
        rd = os.path.join('reco_fullpass', RUN, SUBRUN, f'mx17_{arm}')
        members.append(os.path.join(rd, 'calib_bundle_prelim'))
        members += [os.path.relpath(p, out) for p in
                    sorted(glob.glob(os.path.join(out, rd, 'events_*')))]
    members += [f'stage3_fullpass/tracks_{RUN}_{SUBRUN}.parquet',
                f'stage3_fullpass/tracks_{RUN}_{SUBRUN}.meta.json']
    members += [os.path.relpath(p, out) for p in
                sorted(glob.glob(os.path.join(out, 'intra_bench', 'xy_pairing_*.json')))]
    itgz = os.path.join(a.dest, 'inputs.tar.gz')
    with tarfile.open(itgz, 'w:gz') as t:
        for m in members:
            t.add(os.path.join(out, m), arcname=os.path.join('out', m))
    ih = hashlib.sha256(open(itgz, 'rb').read()).hexdigest()

    # ---- jobs
    rd = out / 'reco_fullpass' / RUN / SUBRUN / 'mx17_A'
    import re
    tags = sorted(t for t in (p.name[len('events_'):-len('.parquet')]
                              for p in rd.glob('events_*.parquet'))
                  if re.fullmatch(r'\d{6}_\d{2}H\d{2}_\d{3}', t))
    have = {p.name for p in (out / 'intra_bench').iterdir()
            if (p / 'build.meta.json').exists()}
    rows = ladder_job_list(tags) if a.ladder else job_list(tags, have)
    with open(os.path.join(a.dest, 'jobs.txt'), 'w') as f:
        for r in rows:
            f.write(' '.join(r) + '\n')
    for fn in ('two_track.sub', 'run_two_track_wrapper.sh', 'run_two_track_job.py'):
        shutil.copy2(os.path.join(HERE, fn), a.dest)

    with open(os.path.join(a.dest, 'PROVENANCE.txt'), 'w') as f:
        f.write(f'two-track validation package\n'
                f'built            {datetime.datetime.now().isoformat()}\n'
                f'git commit       {commit}\n'
                f'code.tar.gz      sha256 {h}\n'
                f'inputs.tar.gz    sha256 {ih}\n'
                f'source           WORKING TREE, not `git archive`\n'
                f'tags             {", ".join(tags)}\n'
                f'jobs             {len(rows)}\n\n'
                f'uncommitted under the shipped code paths:\n'
                f'{dirty or "  (none)"}\n')
    from collections import Counter
    kinds = Counter((r[0], r[1]) for r in rows)
    print(f'code.tar.gz   {os.path.getsize(tgz) / 1e6:.1f} MB  commit {commit[:9]}'
          + ('  +uncommitted' if dirty else ''))
    print(f'inputs.tar.gz {os.path.getsize(itgz) / 1e6:.1f} MB')
    print(f'tags {tags}')
    print(f'{len(rows)} jobs {dict(kinds)}; benches already done locally, skipped: '
          f'{sorted(v for v in have if v.startswith("fixed_"))}')
    print(f'package in {a.dest}')


if __name__ == '__main__':
    main()
