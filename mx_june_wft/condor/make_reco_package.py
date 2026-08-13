#!/usr/bin/env python3
"""
make_reco_package.py — stage the MPGD26 full-June reco campaign for lxplus.

Builds <dest>/ (default /home/dylan/x17/cosmic_bench/condor_campaign/):
    code.tar.gz            git archive of the FREEZE COMMIT (recorded), only
                           the dirs the reco path imports
    bundles.tar.gz         bundles/<det>/<bundle>/ — every frozen bundle, plus
                           any *_t0p gate variants that exist
    campaign_manifest.csv  copied from this directory (build it first)
    jobs.txt               row,tag,extra — tier A (prod), tier B (prod,
                           --vrefit), tier C resist-scans (offcond), and the
                           t0-prior gate arms on the four non-det3 golden keys
    reco.sub, run_reco_job.py, run_reco_wrapper.sh, log/

Then:  rsync -av <dest>/ lxplus:~/wft_campaign/
       ssh lxplus 'cd ~/wft_campaign && condor_submit reco.sub'

    ../../.venv/bin/python mx_june_wft/condor/make_reco_package.py
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ANALYSIS = '/home/dylan/x17/cosmic_bench/Analysis'

CODE_PATHS = ['wft', 'mx_june_cosmic_qa', 'cosmic_bench_analysis', 'common',
              'mx17_m1_map.csv']

# Frozen bundle per detector (must match make_manifest.BUNDLE) and where the
# golden copy lives.
GOLDEN_WFT = {
    'mx17_2': f'{ANALYSIS}/mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2/wft',
    'mx17_3': (f'{ANALYSIS}/mx17_det3_saturday_scan_6-27-26/'
               'long_run_resist_490V_drift_1000V/mx17_3/wft'),
    'mx17_4': f'{ANALYSIS}/mx17_det4_day_6-24-26/long_run/mx17_4/wft',
    'mx17_6': f'{ANALYSIS}/mx17_det6_det7_overnight_6-26-26/long_run/mx17_6/wft',
    'mx17_7': f'{ANALYSIS}/mx17_det6_det7_overnight_6-26-26/long_run/mx17_7/wft',
}
BUNDLE = {'mx17_2': 'calib_bundle_lp', 'mx17_3': 'calib_bundle_lp2_t0p',
          'mx17_4': 'calib_bundle_lp', 'mx17_6': 'calib_bundle_lp',
          'mx17_7': 'calib_bundle_lp'}
GATE_BUNDLE = {d: f'{b}_t0p' for d, b in BUNDLE.items() if d != 'mx17_3'}
GOLDEN_KEYS = {'o22_long_det2': 'mx17_2', 'g_det4': 'mx17_4',
               'g_det6_long': 'mx17_6', 'g_det7_long': 'mx17_7'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest',
                    default='/home/dylan/x17/cosmic_bench/condor_campaign')
    ap.add_argument('--manifest', default=os.path.join(HERE,
                                                       'campaign_manifest.csv'))
    ap.add_argument('--matched-lists',
                    default=('/home/dylan/x17/cosmic_bench/condor_campaign/'
                             'rerun_20260813/matched_lists_all'),
                    help='directory of matched_row<NNN>.json built by '
                         'make_matched_lists.py. Every submitted row must have '
                         'one — the worker stack cannot be trusted to resolve '
                         'the recipe itself (v1 NClus, 184/214 rows).')
    ap.add_argument('--gate-arms', action='store_true',
                    help='also emit the golden keys\' t0p comparison arms. '
                         'The gate was decided 2026-08-12; only pass this to '
                         're-open it.')
    ap.add_argument('--t0p-dets', default=None,
                    help='comma-separated dets whose gate ADOPTED the t0 '
                         'prior (e.g. mx17_2,mx17_6): jobs_rest.txt runs '
                         'their rows with <bundle>_t0p. Only set this after '
                         'gate_eval.py — it is the gate decision, not a '
                         'default.')
    args = ap.parse_args()
    t0p_dets = set(filter(None, (args.t0p_dets or '').split(',')))

    os.makedirs(os.path.join(args.dest, 'log'), exist_ok=True)

    # ---- code at the freeze commit (HEAD of this checkout)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                            capture_output=True, text=True,
                            check=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', '--'] + CODE_PATHS,
                           cwd=REPO, capture_output=True, text=True,
                           check=True).stdout.strip()
    if dirty:
        sys.exit('FATAL: uncommitted changes under the code paths — commit '
                 'the freeze first:\n' + dirty)
    code_tar = os.path.join(args.dest, 'code.tar.gz')
    with open(code_tar, 'wb') as f:
        subprocess.run(['git', 'archive', '--format=tar.gz',
                        '--prefix=code/', commit, '--'] + CODE_PATHS,
                       cwd=REPO, stdout=f, check=True)
    with open(os.path.join(args.dest, 'FREEZE_COMMIT.txt'), 'w') as f:
        f.write(commit + '\n')
    print(f'code.tar.gz at {commit[:9]}')

    # ---- bundles
    bdir = os.path.join(args.dest, 'bundles')
    shutil.rmtree(bdir, ignore_errors=True)
    missing_gate = []
    for det, wft in GOLDEN_WFT.items():
        src = os.path.join(wft, BUNDLE[det])
        if not os.path.isdir(src):
            sys.exit(f'FATAL: frozen bundle missing: {src}')
        shutil.copytree(src, os.path.join(bdir, det, BUNDLE[det]))
        gate = GATE_BUNDLE.get(det)
        if gate:
            gsrc = os.path.join(wft, gate)
            if os.path.isdir(gsrc):
                shutil.copytree(gsrc, os.path.join(bdir, det, gate))
            else:
                missing_gate.append(f'{det}/{gate}')
    with tarfile.open(os.path.join(args.dest, 'bundles.tar.gz'), 'w:gz') as t:
        t.add(bdir, arcname='bundles')
    shutil.rmtree(bdir)
    if missing_gate:
        print('WARNING: t0p gate bundles not built yet (08_t0_abs_calib + '
              'make_bundle_variant):', ', '.join(missing_gate))
        print('         jobs.txt will still list the gate arms — build the '
              'bundles and re-run this script before submitting them.')

    # ---- manifest + jobs
    shutil.copy2(args.manifest, os.path.join(args.dest,
                                             'campaign_manifest.csv'))
    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    # Phase 1 (jobs_gate.txt): the five golden keys with their frozen bundle
    # (= the 7-31-configuration baseline arm) + the four non-det3 t0p arms.
    # Phase 2 (jobs_rest.txt): every other runnable row, with each detector's
    # gate-decided bundle (--t0p-dets after gate_eval.py).
    # ---- matched lists: one per submitted row, staged beside the scripts.
    # A row without a list is NOT submitted — running it would let the worker
    # fall back to its own M3 read, which is the bug this package exists to
    # avoid. Refusing is the whole point; do not "helpfully" skip the flag.
    mdest = os.path.join(args.dest, 'matched_lists')
    shutil.rmtree(mdest, ignore_errors=True)
    os.makedirs(mdest)
    unlistable = []

    def add(bucket, i, tag, extra):
        """Queue row i, or record why it cannot be run on a worker."""
        row = rows[i]
        name = f'matched_row{i:03d}.json'
        src = os.path.join(args.matched_lists, name)
        if not os.path.exists(src):
            unlistable.append((i, row['tier'], row['key'], 'no matched list'))
            return
        if '--vrefit' in extra and row['has_m3v2'] != '1':
            # calibrate.py's reference corridor reads M3 itself (rays geometry,
            # not just event ids), so a list cannot protect a v-refit. One row
            # (100) is affected; run it locally.
            unlistable.append((i, row['tier'], row['key'],
                               'v1 + --vrefit: calibrate reads M3 directly'))
            return
        shutil.copy2(src, os.path.join(mdest, name))
        bucket.append((i, tag, f'{extra} --matched-list {name}'.strip(), name))

    gate, rest = [], []
    for i, row in enumerate(rows):
        adopted = row['det'] in t0p_dets
        if row['key'] in GOLDEN_KEYS or row['key'] == 'sat_det3':
            # The golden keys' prod arm IS the production product for those
            # rows, so it has to run the gate-ADOPTED bundle like every other
            # row of that detector -- building it on the base bundle would
            # promote mx17_2/mx17_4 back onto the un-adopted configuration.
            add(gate, i, 'prod',
                f'--bundle-name {row["bundle"]}_t0p' if adopted else '')
            # The t0p comparison arm only exists to decide the gate. That
            # decision is made (gate_eval, 8-12); re-running it would just
            # re-litigate a settled question, so it is opt-in now.
            if args.gate_arms and row['key'] in GOLDEN_KEYS:
                add(gate, i, 't0p',
                    f'--bundle-name {GATE_BUNDLE[GOLDEN_KEYS[row["key"]]]}')
            continue
        extra = (f'--bundle-name {row["bundle"]}_t0p' if adopted else '')
        if row['tier'] == 'A':
            add(rest, i, 'prod', extra)
        elif row['tier'] == 'B':
            add(rest, i, 'prod', (extra + ' --vrefit').strip())
        elif row['tier'] == 'C' and row['reason'].startswith('resist'):
            add(rest, i, 'offcond', extra)
    for name, jobs in (('jobs_gate.txt', gate), ('jobs_rest.txt', rest)):
        with open(os.path.join(args.dest, name), 'w') as f:
            for r, tag, extra, mlist in jobs:
                # mlist BEFORE extra: see reco.sub's queue line -- extra holds
                # spaces and only the last queue variable absorbs them.
                f.write(f'{r}, {tag}, {mlist}, {extra}\n')
        print(f'{name}: {len(jobs)} jobs')
    print(f'matched_lists/: {len(os.listdir(mdest))} lists staged')
    if unlistable:
        print(f'NOT SUBMITTED — {len(unlistable)} rows cannot run on a worker:')
        for i, tier, key, why in unlistable:
            print(f'   row {i:3d} [{tier}] {key[:50]:50s} {why}')
    print(f'jobs_rest.txt t0p-adopted dets: {sorted(t0p_dets) or "none yet"}')
    # default jobs.txt -> the gate phase, so a bare condor_submit is phase 1
    shutil.copy2(os.path.join(args.dest, 'jobs_gate.txt'),
                 os.path.join(args.dest, 'jobs.txt'))

    # ---- scripts
    for s in ('reco.sub', 'run_reco_job.py', 'run_reco_wrapper.sh'):
        shutil.copy2(os.path.join(HERE, s), os.path.join(args.dest, s))
    os.chmod(os.path.join(args.dest, 'run_reco_wrapper.sh'), 0o755)
    print('package ready:', args.dest)
    print('next: rsync -av', args.dest + '/', 'lxplus:~/wft_campaign/')


if __name__ == '__main__':
    main()
