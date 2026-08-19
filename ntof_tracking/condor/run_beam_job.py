#!/usr/bin/env python3
"""
run_beam_job.py — one condor job = one (arm, file tag) of a beam sub-run.

Fetches that tag's inputs from EOS into the scratch dir, seeds the prelim
bundle from the shipped bench bundle, runs `ntof_tracking.wft_beam`'s
reconstruction on that ONE tag, and leaves

    out/mx17_<arm>/events_<tag>.parquet (+ .candidates.parquet, .meta.json)
    out/mx17_<arm>/calib_bundle_prelim/          (the seeded bundle, once)

for the wrapper to tar.  Merging the tags back into one table is
`merge_beam_tags.py`, run locally.

Why per tag: the network to the laptop that owns this analysis is ~0.3 MB/s
(measured 2026-08-19), so 5 GB of decoded waveforms cannot come home.  The
data is already on EOS at CERN; only the ~1 MB/tag parquet travels.

    python3 run_beam_job.py <arm> <tag> [--run run_145] [--subrun stat090_0000]
                            [--bundle-name calib_bundle_r06] [--v-drift 42.6]
                            [--jobs 8]
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.environ.get('WFT_CODE', os.path.join(HERE, 'code'))
EOS_URL = os.environ.get('EOS_URL', 'root://eospublic.cern.ch/')
EOS_BASE = os.environ.get(
    'EOS_BEAM_BASE', '/eos/experiment/ntof/data/x17/july_beam/runs')


def sh(cmd, **kw):
    print('[job]', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def fetch(eos_path, dest):
    os.makedirs(os.path.dirname(dest.rstrip('/')), exist_ok=True)
    if os.path.isfile(eos_path):          # fuse mount, or a --local rerun
        shutil.copy2(eos_path, dest)
        return
    sh(['xrdcp', '-s', '-f', EOS_URL + eos_path, dest])


def stamp_provenance(bundle_dir, bench_origin):
    """Record what `git archive` cannot: the code commit (there is no .git on a
    worker, so wft stamps `[unknown]`) and the bench bundle this arm was really
    seeded from."""
    import json
    commit = None
    for c in (os.path.join(HERE, 'CODE_COMMIT.txt'),
              os.path.join(HERE, '..', 'CODE_COMMIT.txt')):
        if os.path.isfile(c):
            commit = open(c).read().strip()
            break
    p = os.path.join(bundle_dir, 'bundle.json')
    b = json.load(open(p))
    b.setdefault('provenance', {}).update(
        seeded_from_bench=bench_origin,
        code_commit=commit or 'unknown',
        ran_on='lxplus condor (ntof_tracking/condor/run_beam_job.py)')
    with open(p, 'w') as f:
        json.dump(b, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arm')
    ap.add_argument('tag', help="file tag, e.g. 260805_14H06_004")
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subrun', default='stat090_0000')
    ap.add_argument('--bundle-name', default='calib_bundle_r06')
    ap.add_argument('--bundles', default=os.path.join(HERE, 'bundles'))
    # 42.6 (V_DRIFT_MAGBOLTZ) is what the published run_145 bundles used for
    # EVERY arm -- not the per-arm V_DRIFT_PRIOR.  Pinning it here keeps the
    # kernel the only thing that differs from the superseded tables.
    ap.add_argument('--v-drift', type=float, default=42.6)
    ap.add_argument('--jobs', type=int, default=int(os.environ.get('RECO_JOBS', 8)))
    ap.add_argument('--limit-per-tag', type=int, default=None,
                    help='reconstruct only the first N seeded events -- for '
                         'smoke-testing the stack on a login node, never for '
                         'a production job')
    a = ap.parse_args()

    sys.path.insert(0, CODE)
    data = os.path.join(HERE, 'data')
    sub = f'{EOS_BASE}/{a.run}/{a.subrun}'
    loc = os.path.join(data, a.run, a.subrun)

    # BEFORE the import: BeamConfig.BASE_PATH takes its default from the module
    # global at CLASS-CREATION time, so assigning wft_beam.BEAM_BASE afterwards
    # is silently ignored and the job reads the laptop's paths (which do not
    # exist on a worker).  The env overrides are the only ones that take.
    os.environ['WFT_BEAM_BASE'] = data + '/'
    os.environ['WFT_BEAM_ANALYSIS'] = os.path.join(HERE, 'out') + '/'

    from ntof_tracking import wft_beam as wb          # noqa: E402
    feu_x, feu_y = wb.BEAM_DETS[a.arm]['feu_x'], wb.BEAM_DETS[a.arm]['feu_y']

    fetch(f'{EOS_BASE}/{a.run}/run_config.json',
          os.path.join(data, a.run, 'run_config.json'))
    fetch(f'{sub}/combined_hits_root/Mx17_{a.subrun}_datrun_{a.tag}'
          '_feu-combined_hits.root',
          os.path.join(loc, 'combined_hits_root',
                       f'Mx17_{a.subrun}_datrun_{a.tag}_feu-combined_hits.root'))
    for feu in (feu_x, feu_y):
        f = f'Mx17_{a.subrun}_datrun_{a.tag}_{feu:02d}.root'
        fetch(f'{sub}/decoded_root/{f}', os.path.join(loc, 'decoded_root', f))

    # the shipped bench bundle replaces the laptop path in the seed table --
    # but keep the laptop path, it is the only record of WHICH bench key and
    # sub-run this arm is seeded from (`bundles/mx17_A/calib_bundle_r06` is
    # not: three arms would look alike).
    bench_origin = wb.BEAM_DETS[a.arm]['bundle']
    wb.BEAM_DETS[a.arm] = dict(
        wb.BEAM_DETS[a.arm],
        bundle=os.path.join(a.bundles, f'mx17_{a.arm}', a.bundle_name))

    out_dir = os.path.join(HERE, 'out', f'mx17_{a.arm}')
    os.makedirs(out_dir, exist_ok=True)
    bundle = wb.make_bundle(a.arm, out=os.path.join(out_dir,
                                                    'calib_bundle_prelim'),
                            v_drift=a.v_drift, run=a.run, sub_run=a.subrun)
    stamp_provenance(bundle, bench_origin)

    # The bench absolute-time table must not ride along on a beam run (see
    # wft_beam.make_bundle). Assert it here too: this is a worker, nobody reads
    # its log unless something already went wrong.
    import json as _json
    _b = _json.load(open(os.path.join(bundle, 'bundle.json')))
    if _b.get('t0_prior_sigma') or _b.get('t0_abs'):
        sys.exit('FATAL: seeded bundle still carries a t0 prior '
                 f'(sigma={_b.get("t0_prior_sigma")}) — that is the BENCH '
                 'trigger, not this run\'s')

    cfg = wb.beam_config(a.arm, a.run, a.subrun)
    cfg.file_tags = [a.tag]
    wb.reconstruct_subrun(cfg, bundle,
                          os.path.join(out_dir, f'events_{a.tag}.parquet'),
                          jobs=a.jobs, limit_per_tag=a.limit_per_tag)
    shutil.rmtree(data, ignore_errors=True)          # don't tar 300 MB back
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
