#!/usr/bin/env python3
"""ntof_reco.py -- P4 of NTOF_PASS_PLAN_2026-11-02: reconstruct one n_TOF (run, subrun, arm) on the pass-1 algorithm.

    ntof_reco.py --run 145 --subrun stat090_0000 --arm A --bundle DIR --out FILE.parquet
                 [--base /media/ucla/work/ntof_pass1/runs/] [--tags N | --tag T ...] [--limit-per-tag N] [--jobs 8]

The fit is wft as locked by pass 1 (LM, pivot start, column guard, two-track, dead censoring, quality flags); the
seeding, windowing and checkpointing are ntof_tracking/wft_beam.py's (hits-seeded, busy plane > 150, up to 5
candidates, isochronous veto), with min strips 3 (is2's measured choice, HANDOFF_TRACKING §9-10) passed explicitly.
Every per-row number comes from the registry (rerun/ntof/ntof_registry.csv) and the bundle (rerun/ntof/ntof_bundle.py);
the registry row, the bundle provenance and the code commit go into the sidecar under ``ntof_pass``.  Study switches
go through WFT_DIAG only (wft.diag), which records them.
"""
import argparse
import csv
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, REPO)

REGISTRY = os.path.join(HERE, 'ntof_registry.csv')
EOS_RUNS = '/eos/experiment/ntof/data/x17/july_beam/runs/'
MIN_STRIPS = 3
FEUS = dict(A=(3, 4), B=(5, 6), C=(7, 8), D=(1, 2))


def registry_rows(run, subrun):
    with open(REGISTRY) as f:
        return [r for r in csv.DictReader(f) if int(r['run']) == int(run) and r['subrun'] == subrun]


def code_commit():
    c = os.environ.get('RERUN_CODE_COMMIT')
    if c:
        return c
    try:
        return subprocess.check_output(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], text=True).strip() + (
            '+dirty' if subprocess.call(['git', '-C', REPO, 'diff', '--quiet']) else '')
    except Exception:
        return '?'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=int, required=True)
    ap.add_argument('--subrun', required=True)
    ap.add_argument('--arm', choices='ABCD', required=True)
    ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--base', default=EOS_RUNS, help='directory holding run_<n>/<subrun>/{decoded_root,combined_hits_root}')
    ap.add_argument('--tags', type=int, default=None, help='first N acquisitions only')
    ap.add_argument('--tag', nargs='*', default=None, help='these acquisitions only')
    ap.add_argument('--limit-per-tag', type=int, default=None)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--seed', choices=('hits', 'waveform'), default='hits', help='Q4: seeding source')
    ap.add_argument('--hits-dir', default='combined_hits_root', help='hits generation dir name under the sub-run')
    ap.add_argument('--events', default=None, help='text file of event ids: fit only these (studies; seeding still '
                                                   'runs over every tag, so the seeded-but-not-listed count is kept)')
    a = ap.parse_args()

    from ntof_tracking import wft_beam as wb
    from wft import diag
    rows = registry_rows(a.run, a.subrun)
    rows = [r for r in rows if r['excluded'] != 'True']
    if not rows:
        raise SystemExit(f'run_{a.run}/{a.subrun}: no non-excluded registry row')
    base = a.base if a.base.endswith('/') else a.base + '/'
    fx, fy = FEUS[a.arm]
    r0 = rows[0]
    cfg = wb.BeamConfig(KEY=f'run_{a.run}_{a.subrun}_mx17_{a.arm}', RUN=f'run_{a.run}', SUB_RUN=a.subrun,
                        DET_NAME=f'mx17_{a.arm}', MX17_FEU_X=fx, MX17_FEU_Y=fy, BASE_PATH=base,
                        OUT_BASE=os.path.dirname(os.path.abspath(a.out)), HITS_DIR=a.hits_dir,
                        conditions=dict(gas=r0['gas'], sample_ns=float(r0['sample_period_ns']),
                                        n_samples=int(float(r0['n_samples']))))
    tags = sorted(r['tag'] for r in rows)
    if a.tag:
        tags = [t for t in tags if t in set(a.tag)]
    elif a.tags:
        tags = tags[:a.tags]
    cfg.file_tags = tags
    extra = dict(plan='rerun/ntof/NTOF_PASS_PLAN_2026-11-02.md', code=code_commit(), min_strips=MIN_STRIPS,
                 registry={k: r0[k] for k in r0 if k not in ('tag', 't_local', 'n_dec', 'n_fdf', 'n_hits', 'dec_gb',
                                                              'fdf_gb', 'dec_min_bytes')},
                 tags=tags, base=base, diag=dict(diag.ACTIVE) if hasattr(diag, 'ACTIVE') else {})
    allow = None
    if a.events:
        ids = {int(x) for x in open(a.events).read().split()}
        allow = {t_: ids for t_ in tags}          # event ids run on across a sub-run's acquisitions: no collision
        extra['events_file'] = os.path.abspath(a.events)
    t = time.time()
    df = wb.reconstruct_subrun(cfg, a.bundle, a.out, jobs=a.jobs, limit_per_tag=a.limit_per_tag,
                               allow_events=allow, allow_meta=dict(source=a.events) if allow else None,
                               allow_bow=True, min_strips=MIN_STRIPS, extra_meta=extra, seed_source=a.seed)
    dt = time.time() - t
    print(f'[ntof] {len(df):,} events in {dt:.0f} s wall ({a.jobs} workers): '
          f'{dt * a.jobs / max(len(df), 1):.3f} cpu-s/event')


if __name__ == '__main__':
    main()
