#!/usr/bin/env python3
"""
Reprocess the whole cosmic bench with the 2026-07-24 waveform analyzer
(mm_strip_reconstruction 33e132b: unified trigger, median baseline + guard gap,
full-span integral, CNS on with post-CNS pedestal RMS, Release build).

Runs decoded_root -> hits_root -> combined_hits_root IN PLACE, overwriting the
old-analyzer outputs. Decoding is NOT re-run: the decoded_root files are the input.

Usage (from mx_june_cosmic_qa/):
    ../.venv/bin/python reprocess_cosmic_bench.py --dry-run
    ../.venv/bin/python reprocess_cosmic_bench.py --jobs 6
    ../.venv/bin/python reprocess_cosmic_bench.py --only det6_det7
    ../.venv/bin/python reprocess_cosmic_bench.py --snapshot-only   # old hit counts

Pedestal resolution per (subrun, FEU), first match wins:
    1. pedthr root in this subrun whose run-name prefix matches the data files
    2. any unique pedthr root in this subrun
    3. unique pedthr root among sibling subruns of the same run (HV/drift scan
       points share the run's single pedestal set -- verified on lxplus for the
       6-27 saturday scan)
Anything unresolved is reported and skipped, never silently run without a
pedestal (no pedestal => the analyzer assumes zero-suppressed input, baseline
256 / RMS 1, which is wrong for these non-ZS runs).
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

BASE_DATA = '/media/dylan/data/x17/cosmic_bench'
# Pin the analyzer build. The shared repo checkout is a moving target (it advanced
# mid-session and a rebuild broke a running pass). Set REPROC_SOFT to a build of a
# fixed commit so every subrun is one generation. Falls back to the shared build.
BASE_SOFT = os.environ.get(
    'REPROC_SOFT',
    '/home/dylan/CLionProjects/mm_strip_reconstruction/cmake-build-release')
ANALYZE_EXE = f'{BASE_SOFT}/waveform_analysis/analyze_waveforms'
COMBINE_EXE = f'{BASE_SOFT}/feu_hit_combiner/combine_feus_hits'

# Reuse the OFFICIAL process_run.py recipe for the matched-filter gate width so our
# hits match what the DAQ pipeline produces: --tps (sample period) and per-FEU
# --mf = max(3, round(1.7 * Dream_peaking_ns / tps)). Importing is side-effect free
# (process_run guards main() under __main__).
PROCESS_RUN_DIR = os.environ.get(
    'PROCESS_RUN_DIR',
    '/home/dylan/CLionProjects/mm_strip_reconstruction/orchestrator')
sys.path.insert(0, PROCESS_RUN_DIR)
try:
    import process_run as _pr
    _HAVE_PR = True
except Exception as _e:  # fall back to the analyzer's own AUTO gate width
    _HAVE_PR = False
    _PR_IMPORT_ERR = str(_e)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reprocess_logs')

# Strict: <name>_datrun_<date>_<time>_<filenum>_<feu>.root. Deliberately excludes the
# legacy *_decoded_array.root files, which hold a different tree the analyzer can't read.
DAT_RE = re.compile(r'_datrun_.*_(\d{3})_(\d{2})\.root$')
PED_RE = re.compile(r'_pedthr_.*_(\d{3})_(\d{2})\.root$')
COMB_RE = re.compile(r'_(\d{3})_feu-combined')


def run_prefix(fname):
    """'MX17_long_run_datrun_260627_...' -> 'MX17_long_run'."""
    for tag in ('_datrun_', '_pedthr_'):
        if tag in fname:
            return fname.split(tag)[0]
    return ''


def find_pedestals(subrun_dir):
    """{feu: [paths]} for every pedthr root under a subrun (skipping output dirs)."""
    peds = defaultdict(list)
    for root, dirs, files in os.walk(subrun_dir):
        dirs[:] = [d for d in dirs if d not in ('hits_root', 'combined_hits_root',
                                                'm3_tracking_root', 'm3_tracking_root_v2')]
        for f in files:
            m = PED_RE.search(f)
            if m:
                peds[m.group(2)].append(os.path.join(root, f))
    return peds


def resolve_ped(feu, prefix, own_peds, sibling_peds):
    cands = own_peds.get(feu, [])
    if len(cands) > 1:
        exact = [p for p in cands if run_prefix(os.path.basename(p)) == prefix]
        if len(exact) == 1:
            return exact[0], 'own/prefix-match'
        return None, f'ambiguous ({len(cands)} pedestals in subrun, no prefix match)'
    if len(cands) == 1:
        return cands[0], 'own'
    cands = sibling_peds.get(feu, [])
    stamps = {os.path.basename(p).split('_pedthr_')[1] for p in cands}
    if len(stamps) == 1:
        return sorted(cands)[0], 'sibling subrun (shared run pedestal)'
    if not cands:
        return None, 'no pedestal found for this FEU'
    return None, f'ambiguous ({len(stamps)} pedestal timestamps among sibling subruns)'


def is_fresh_subrun(subrun_dir):
    """True if this subrun's combined hits are already the new generation.

    The a1cce79 (matched-filter) generation adds a per-hit `significance` branch.
    A subrun counts as fresh only if it has combined files and every one of them
    carries `significance`, so a half-done subrun is re-processed rather than
    skipped (and older-generation combined files are correctly seen as stale).
    """
    import uproot
    cdir = os.path.join(subrun_dir, 'combined_hits_root')
    combs = glob.glob(os.path.join(cdir, '*.root'))
    if not combs:
        return False
    for c in combs:
        try:
            with uproot.open(c) as fh:
                if 'significance' not in fh['hits'].keys():
                    return False
        except Exception:
            return False
    return True


def gate_params(run_dir, subrun_dir):
    """(tps_ns, peaking_dict) from the run's Dream cfg, via the official helpers.

    Returns (None, {}) when process_run isn't importable or no cfg is found, in
    which case the analyzer applies its own AUTO gate width.
    """
    if not _HAVE_PR:
        return None, {}
    try:
        tps = _pr.run_sample_period_ns(run_dir)
        cfg = _pr.find_dream_cfg(os.path.join(subrun_dir, 'raw_daq_data'), run_dir)
        peaking = _pr.parse_dream_peaking(cfg) if cfg else {}
        return tps, peaking
    except Exception:
        return None, {}


def analyze_flags(feu, tps, peaking):
    """The --tps/--mf tail process_run.py would pass for this FEU (matched exactly)."""
    extra = ''
    if tps:
        extra += f' --tps {tps:g}'
    if peaking and tps:
        peak_ns = peaking.get(str(int(feu)), peaking.get('*'))
        if peak_ns:
            mf = max(3, round(_pr.MF_WIDTH_OVER_PEAKING * peak_ns / tps))
            extra += f' --mf {mf}'
    return extra


def build_plan(only=None, exclude=(), skip_fresh=False):
    """List of subrun jobs plus the pedestal-resolution problems found."""
    jobs, problems = [], []
    for bench in sorted(os.listdir(BASE_DATA)):
        bench_dir = os.path.join(BASE_DATA, bench)
        if not os.path.isdir(bench_dir) or bench in ('Analysis', 'pedestals', '_m3check'):
            continue
        for run in sorted(os.listdir(bench_dir)):
            run_dir = os.path.join(bench_dir, run)
            if not os.path.isdir(run_dir):
                continue
            # pedestals of every subrun in this run, for the shared-pedestal fallback
            run_peds = defaultdict(list)
            subruns = [s for s in sorted(os.listdir(run_dir))
                       if os.path.isdir(os.path.join(run_dir, s))]
            sub_peds = {}
            for s in subruns:
                sub_peds[s] = find_pedestals(os.path.join(run_dir, s))
                for feu, paths in sub_peds[s].items():
                    run_peds[feu].extend(paths)

            for s in subruns:
                subrun_dir = os.path.join(run_dir, s)
                dec = os.path.join(subrun_dir, 'decoded_root')
                if not os.path.isdir(dec):
                    continue
                dats = sorted(f for f in os.listdir(dec) if DAT_RE.search(f))
                if not dats:
                    continue
                rel = os.path.relpath(subrun_dir, BASE_DATA)
                if only and only not in rel:
                    continue
                if any(x in rel for x in exclude):
                    continue
                if skip_fresh and is_fresh_subrun(subrun_dir):
                    continue
                prefix = run_prefix(dats[0])
                tps, peaking = gate_params(run_dir, subrun_dir)
                sibling = {feu: [p for p in paths
                                 if os.path.commonpath([p, subrun_dir]) != subrun_dir]
                           for feu, paths in run_peds.items()}
                files = []
                for f in dats:
                    fnum, feu = DAT_RE.search(f).groups()
                    ped, how = resolve_ped(feu, prefix, sub_peds[s], sibling)
                    if ped is None:
                        problems.append((rel, feu, how))
                        continue
                    files.append(dict(fnum=fnum, feu=feu, ped=ped, ped_src=how,
                                      flags=analyze_flags(feu, tps, peaking),
                                      src=os.path.join(dec, f),
                                      out=os.path.join(subrun_dir, 'hits_root',
                                                       f.replace('.root', '_hits.root'))))
                if files:
                    jobs.append(dict(rel=rel, dir=subrun_dir, files=files))
    return jobs, problems


def snapshot_old(jobs):
    """Entry count of every existing combined-hits file, before we overwrite it."""
    import uproot
    snap = {}
    for job in jobs:
        cdir = os.path.join(job['dir'], 'combined_hits_root')
        if not os.path.isdir(cdir):
            continue
        for f in sorted(os.listdir(cdir)):
            if not f.endswith('.root'):
                continue
            path = os.path.join(cdir, f)
            try:
                with uproot.open(path) as fh:
                    key = fh.keys(recursive=False)[0]
                    snap[os.path.relpath(path, BASE_DATA)] = int(fh[key].num_entries)
            except Exception as e:  # corrupt/empty legacy file -- record, don't crash
                snap[os.path.relpath(path, BASE_DATA)] = f'ERROR: {e}'
    return snap


def analyze_one(spec):
    os.makedirs(os.path.dirname(spec['out']), exist_ok=True)
    t0 = time.time()
    cmd = [ANALYZE_EXE, spec['src'], spec['out'], spec['ped']]
    cmd += spec.get('flags', '').split()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0 and os.path.exists(spec['out'])
    return dict(out=spec['out'], src=spec['src'], ok=ok, secs=time.time() - t0,
                rc=proc.returncode, tail=(proc.stdout + proc.stderr)[-2000:])


def combine_subrun(job, log):
    """Regenerate combined hits for every file number that has fresh per-FEU hits."""
    cdir = os.path.join(job['dir'], 'combined_hits_root')
    os.makedirs(cdir, exist_ok=True)
    by_fnum = defaultdict(dict)
    for f in job['files']:
        if os.path.exists(f['out']):
            by_fnum[f['fnum']][f['feu']] = f['out']
    made = []
    for fnum, feu_map in sorted(by_fnum.items()):
        first = os.path.basename(sorted(feu_map.values())[0])
        out = os.path.join(cdir, re.sub(r'(_\d{3}_)\d{2}', r'\1feu-combined', first, count=1))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            for feu, path in sorted(feu_map.items()):
                tmp.write(f'{path} {int(feu)}\n')
            listfile = tmp.name
        try:
            proc = subprocess.run([COMBINE_EXE, listfile, out], capture_output=True, text=True)
            if proc.returncode != 0:
                log(f'  [combine FAIL] {os.path.basename(out)} rc={proc.returncode}\n'
                    f'    {(proc.stdout + proc.stderr)[-500:]}')
            else:
                made.append(out)
        finally:
            os.unlink(listfile)
    return made, set(by_fnum)


def quarantine_orphans(job, fresh_fnums, log):
    """Old-generation combined files whose decoded input is gone -> _old_analyzer/.

    Leaving them next to new-generation files would silently mix analyzer
    generations in the same directory for downstream QA.
    """
    cdir = os.path.join(job['dir'], 'combined_hits_root')
    if not os.path.isdir(cdir):
        return []
    moved = []
    for f in sorted(os.listdir(cdir)):
        m = COMB_RE.search(f)
        if not m or m.group(1) in fresh_fnums:
            continue
        qdir = os.path.join(cdir, '_old_analyzer')
        os.makedirs(qdir, exist_ok=True)
        os.rename(os.path.join(cdir, f), os.path.join(qdir, f))
        moved.append(f)
    if moved:
        log(f'  [quarantine] {len(moved)} old-analyzer combined files -> _old_analyzer/')
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--only', default=None, help='substring filter on <bench>/<run>/<subrun>')
    ap.add_argument('--exclude', action='append', default=[],
                    help='substring to skip (repeatable), e.g. a run still downloading')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--snapshot-only', action='store_true')
    ap.add_argument('--skip-fresh', action='store_true',
                    help='skip subruns whose combined hits already carry trunc_right')
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logpath = os.path.join(LOG_DIR, f'reprocess_{stamp}.log')
    logfh = open(logpath, 'a', buffering=1)

    def log(msg):
        print(msg, flush=True)
        logfh.write(msg + '\n')

    jobs, problems = build_plan(args.only, tuple(args.exclude), args.skip_fresh)
    nfiles = sum(len(j['files']) for j in jobs)
    gb = sum(os.path.getsize(f['src']) for j in jobs for f in j['files']) / 1e9
    log(f'== reprocess_cosmic_bench {stamp} ==')
    log(f'analyzer: {ANALYZE_EXE}')
    log(f'gate recipe: process_run helpers {"IMPORTED" if _HAVE_PR else "UNAVAILABLE (AUTO gate)"}'
        + ('' if _HAVE_PR else f' -- {_PR_IMPORT_ERR}'))
    log(f'{len(jobs)} subruns, {nfiles} FEU files, {gb:.1f} GB decoded input')
    if problems:
        log(f'\n!! {len(problems)} unresolved pedestals (these FEU files are SKIPPED):')
        for rel, feu, why in problems:
            log(f'   {rel}  FEU {feu}: {why}')
    for j in jobs:
        srcs = defaultdict(int)
        for f in j['files']:
            srcs[f['ped_src']] += 1
        gates = sorted({f['flags'].strip() or 'AUTO' for f in j['files']})
        log(f'  {j["rel"]:<70} {len(j["files"]):>3} files  ped:{dict(srcs)}  gate:{gates}')

    if args.dry_run:
        log(f'\ndry run -- nothing executed. log: {logpath}')
        return

    snap_path = os.path.join(LOG_DIR, f'old_combined_counts_{stamp}.json')
    log('\nsnapshotting old combined-hits entry counts...')
    snap = snapshot_old(jobs)
    with open(snap_path, 'w') as fh:
        json.dump(snap, fh, indent=1, sort_keys=True)
    log(f'  {len(snap)} old combined files recorded -> {snap_path}')
    if args.snapshot_only:
        return

    specs = [f for j in jobs for f in j['files']]
    t0 = time.time()
    done = failed = 0
    log(f'\nanalyzing {len(specs)} files with {args.jobs} workers...')
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(analyze_one, s): s for s in specs}
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if not r['ok']:
                failed += 1
                log(f'  [FAIL {done}/{len(specs)}] {os.path.basename(r["src"])} '
                    f'rc={r["rc"]}\n    {r["tail"]}')
            elif done % 10 == 0 or done == len(specs):
                rate = done / (time.time() - t0)
                eta = (len(specs) - done) / rate / 60
                log(f'  [{done}/{len(specs)}] {rate*60:.0f} files/min, ETA {eta:.1f} min')
    log(f'analyze done in {(time.time()-t0)/60:.1f} min, {failed} failures')

    log('\ncombining...')
    total_comb, total_orph = 0, 0
    for j in jobs:
        made, fresh = combine_subrun(j, log)
        moved = quarantine_orphans(j, fresh, log)
        total_comb += len(made)
        total_orph += len(moved)
        log(f'  {j["rel"]:<70} {len(made)} combined')
    log(f'\n{total_comb} combined files written, {total_orph} old ones quarantined')
    log(f'total wall time {(time.time()-t0)/60:.1f} min')
    log(f'log: {logpath}')


if __name__ == '__main__':
    main()
