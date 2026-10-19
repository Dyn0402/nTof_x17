#!/usr/bin/env python3
"""Which DREAM beam sub-runs can be slimmed TODAY, and what blocks the rest?

A DREAM sub-run is processable only where its wall-clock window is covered by an
n_TOF run that has been reprocessed on the v12 UserInput. This joins three
listings (cached under `coverage_inputs/`, see `refresh_inputs()`):

  reprocessed n_TOF runs   `index` tree Date/Time of every file in
                           /eos/experiment/ntof/processing/official/done/
                           -- v12 content, audited 2026-08-08
                           (../SLIM_FEASIBILITY_2026-08-08.md section 6a)
  raw n_TOF runs           mtimes of */stream1/*_s1.raw* under
                           /eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement
                           -- what could still BE reprocessed
  DREAM beam sub-runs      the `datrun_YYMMDD_HHhMM` stamp in the first decoded
                           file name, plus the sub-run's duration

and classifies every second of DREAM beam time as one of

  READY        covered by a v12-reprocessed n_TOF run -- process it now
  RECOVERABLE  covered by an n_TOF run that exists in raw but is not reprocessed
  LOST         covered by an n_TOF run with no raw left on the EOS disk
  NO n_TOF     no n_TOF run was live

`LOST` means gone from the disk area we can see; n_TOF may hold a tape copy.
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'coverage_inputs'

LOCAL = timezone(timedelta(hours=2))     # the DAQ writes local = UTC+2
DATRUN = re.compile(r'datrun_(\d{6})_(\d{2})H(\d{2})')
READY, RECOV, LOST, NONE = 'READY', 'RECOVERABLE', 'LOST', 'NO n_TOF'


def refresh_inputs():
    """Commands that regenerate the three cached listings.

    ntof_index_times.txt  run first_bunch_epoch last_bunch_epoch n_bunches
        ssh lxplus 'source /cvmfs/sft.cern.ch/lcg/views/LCG_105/\\
                    x86_64-el9-gcc13-opt/setup.sh; python3 ntof_index_times.py'

    ntof_raw_times.txt    run first_mtime last_mtime n_files
        ssh lxplus 'D=/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement;
                    find $D -mindepth 3 -maxdepth 3 -name "*_s1.raw*" \\
                    -printf "%h %T@\\n" | awk ...'

    dream_eos_subruns.txt  run subrun n_files first_file_name
    dream_daq_subruns.txt  run subrun mtime_first mtime_last n_files name
        one `for d in <runs>/*/stat*/` loop on lxplus and on the DAQ.
    """
    raise SystemExit(refresh_inputs.__doc__)


def _spans(path: Path, skip_short=True, shift_s=0.0):
    """[(run, t_start, t_end)] from a `run lo hi n` listing, epochs in UTC.

    `shift_s` corrects a listing whose times are not UTC. The n_TOF `index`
    tree's Date/Time fields are LOCAL (the DAQ writes UTC+2) and
    `ntof_index_times.py` turns them into an epoch as if they were UTC, so that
    listing needs -7200 s. Confirmed against the raw mtimes, which are true
    UTC: over the 109 runs that have both, raw_start - index_start is a flat
    -7127 s (p10 -7185, p90 -7080), i.e. -7200 s plus the ~1 min it takes the
    first raw file to finish writing.
    """
    out = []
    for ln in path.read_text().splitlines():
        f = ln.split()
        if len(f) != 4 or f[1] == 'ERR':
            continue
        run, lo, hi, n = int(f[0]), int(f[1]), int(f[2]), int(f[3])
        if skip_short and (n < 2 or hi <= lo):
            continue
        out.append((run, float(lo) + shift_s, float(hi) + shift_s))
    return sorted(out)


# The index tree's Date/Time is LOCAL Geneva time (UTC+2, whole campaign in
# CEST); everything else lives on the UTC psTime/CSV base. Measured twice,
# independently, against different reference clocks:
#   * raw file mtimes (true UTC), 109 runs: raw_start - index_start flat at
#     -7127 s (p10 -7185, p90 -7080) = -7200 plus ~1 min of first-file write
#     lag (see _spans);
#   * PKUP psTime of run 224572: index_start - psTime_start = 7199.5 s
#     (pulse_ledger, 2026-08-13 — found when whole sub-runs classified
#     NTOF_NO_BUNCH while their products sat on disk).
# This is the ONE home of the constant: pulse_ledger imports it from here, so
# the ledger and the segment proposals cannot silently disagree about it.
INDEX_LOCAL_SHIFT_S = -7200.0
RAW_WRITE_LAG_S = 73.0            # median (raw_start - corrected index_start)


def dream_start(name: str):
    m = DATRUN.search(name)
    if not m:
        return None
    d, hh, mm = m.group(1), int(m.group(2)), int(m.group(3))
    return datetime(2000 + int(d[:2]), int(d[2:4]), int(d[4:6]), hh, mm,
                    tzinfo=LOCAL).timestamp()


def load_dream(eos: Path, daq: Path):
    subs = {}
    for ln in eos.read_text().splitlines():
        f = ln.split()
        if len(f) != 4:
            continue
        t = dream_start(f[3])
        if t is not None:
            subs[(f[0], f[1])] = [t, None, int(f[2])]
    for ln in daq.read_text().splitlines():
        f = ln.split()
        if len(f) != 6:
            continue
        t = dream_start(f[5])
        subs[(f[0], f[1])] = [t if t is not None else float(f[2]),
                              float(f[3]), int(f[4])]
    rates = sorted((v[1] - v[0]) / v[2] for v in subs.values()
                   if v[1] and v[2] > 0 and v[1] > v[0])
    per_file = rates[len(rates) // 2] if rates else 47.0
    for v in subs.values():
        if v[1] is None:
            v[1] = v[0] + per_file * v[2]

    # A file-count estimate overshoots -- the DAQ writes a fixed 60 min sub-run
    # (`dream_daq.log`: "Subrun started: stat090_0000 run_time=60.0min") and the
    # file rate varies with trigger rate, so 104 files x 47 s reads as 81 min for
    # an hour of data. Sub-runs never overlap: the log shows a ~13 s DAQ restart
    # between them. So clamp each end at the next sub-run's start, which is
    # measured rather than inferred.
    order = sorted(subs, key=lambda k: subs[k][0])
    for a, b in zip(order, order[1:]):
        if subs[b][0] > subs[a][0]:
            subs[a][1] = min(subs[a][1], subs[b][0])
    return subs, per_file


def _union(iv):
    """Merge [(lo,hi)] into a disjoint, sorted cover."""
    out = []
    for lo, hi in sorted(iv):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def _length(iv):
    return sum(hi - lo for lo, hi in iv)


def _subtract(a, b):
    """Interval cover `a` minus interval cover `b` (both disjoint, sorted)."""
    out = []
    for lo, hi in a:
        cur = lo
        for blo, bhi in b:
            if bhi <= cur or blo >= hi:
                continue
            if blo > cur:
                out.append((cur, min(blo, hi)))
            cur = max(cur, bhi)
            if cur >= hi:
                break
        if cur < hi:
            out.append((cur, hi))
    return out


def classify(t0, t1, v12, raw, v12set):
    """Seconds of [t0,t1) in each class, plus the n_TOF runs implicated.

    The classes are disjoint by construction: RECOVERABLE is the raw-only cover
    with the v12 cover subtracted, so overlapping n_TOF run windows (index and
    mtime windows do overlap at run boundaries) cannot be counted twice.
    """
    win = [(t0, t1)]
    cov_v12 = _union([(max(a, t0), min(b, t1)) for _, a, b in v12
                      if min(b, t1) > max(a, t0)])
    raw_only = [(max(a, t0), min(b, t1)) for r, a, b in raw
                if r not in v12set and min(b, t1) > max(a, t0)]
    cov_recov = _subtract(_union(raw_only), cov_v12)

    runs = {READY: {r for r, a, b in v12 if min(b, t1) > max(a, t0)},
            RECOV: {r for r, a, b in raw
                    if r not in v12set and min(b, t1) > max(a, t0)},
            LOST: set()}
    acc = {READY: _length(cov_v12), RECOV: _length(cov_recov),
           LOST: _length(_subtract(_subtract(win, cov_v12), cov_recov))}
    return acc, runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=str(DATA))
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--min-ready', type=float, default=0.999,
                    help='fraction of a run that must be v12-covered to be READY')
    args = ap.parse_args()
    d = Path(args.dir)

    v12 = _spans(d / 'ntof_index_times.txt',
                 shift_s=INDEX_LOCAL_SHIFT_S)
    raw = _spans(d / 'ntof_raw_times.txt', skip_short=False,
                 shift_s=-RAW_WRITE_LAG_S)
    subs, per_file = load_dream(d / 'dream_eos_subruns.txt',
                                d / 'dream_daq_subruns.txt')
    v12set = {r for r, _, _ in v12}
    rawset = {r for r, _, _ in raw}
    print(f'{len(v12)} n_TOF runs reprocessed on v12 '
          f'({min(v12set)}-{max(v12set)})')
    print(f'{len(raw)} n_TOF runs with raw stream1 still on the EOS disk')
    print(f'{len(subs)} DREAM beam sub-runs, {per_file:.0f} s per decoded file\n')

    by_run = {}
    for (run, sub), (t0, t1, n) in subs.items():
        acc, runs = classify(t0, t1, v12, raw, v12set)
        by_run.setdefault(run, []).append((sub, t0, t1, n, acc, runs))

    def keyrun(r):
        return int(r.split('_')[1])

    print(f'{"DREAM run":<9} {"subs":>4} {"start":<12} {"h":>5} {"files":>6} '
          f'{"READY":>7} {"RECOV":>7} {"other":>7}  needs')
    print('-' * 92)
    tot = {READY: 0.0, RECOV: 0.0, 'other': 0.0}
    files = {'ready': 0, 'partial': 0, 'none': 0}
    ready_runs, recov_need = [], set()
    for run in sorted(by_run, key=keyrun):
        v = by_run[run]
        t0, t1 = min(x[1] for x in v), max(x[2] for x in v)
        nf = sum(x[3] for x in v)
        span = sum(x[2] - x[1] for x in v)
        a = {k: sum(x[4][k] for x in v) for k in (READY, RECOV, LOST)}
        other = a[LOST]
        fr = a[READY] / max(span, 1)
        need = sorted({r for x in v for r in x[5][RECOV]})
        recov_need |= set(need)
        tot[READY] += a[READY]; tot[RECOV] += a[RECOV]; tot['other'] += other
        st = datetime.fromtimestamp(t0, LOCAL).strftime('%m-%d %H:%M')
        mark = '  <= SLIM NOW' if fr >= args.min_ready else ''
        if fr >= args.min_ready:
            ready_runs.append((run, nf)); files['ready'] += nf
        elif fr > 0.001:
            files['partial'] += nf
        else:
            files['none'] += nf
        ns = ','.join(str(r) for r in need[:6]) + ('…' if len(need) > 6 else '')
        print(f'{run:<9} {len(v):>4} {st:<12} {(t1-t0)/3600:>5.1f} {nf:>6} '
              f'{fr:>6.1%} {a[RECOV]/max(span,1):>7.1%} '
              f'{other/max(span,1):>7.1%}  {ns}{mark}')
        if args.verbose:
            for sub, s0, s1, n, acc, runs in sorted(v, key=lambda x: x[1]):
                sp = max(s1 - s0, 1)
                print(f'    {sub:<16} {datetime.fromtimestamp(s0, LOCAL):%m-%d %H:%M}'
                      f' {(s1-s0)/60:>6.1f} min {n:>4} f  '
                      f'ready {acc[READY]/sp:>6.1%}  '
                      f'v12 {",".join(str(r) for r in sorted(runs[READY])) or "-"}')

    total = sum(tot.values())
    print(f'\n{"":<20}{"beam-seconds":>14}{"share":>9}')
    for k, lbl in ((READY, 'READY (v12)'), (RECOV, 'RECOVERABLE (raw on disk)'),
                   ('other', 'LOST or no n_TOF')):
        print(f'{lbl:<28}{tot[k]/3600:>10.1f} h {tot[k]/max(total,1):>8.1%}')
    print(f'\nDREAM runs 100 % v12-covered: {len(ready_runs)} '
          f'({files["ready"]} decoded files)')

    # The n_TOF run boundaries fall INSIDE DREAM runs, so the whole-run view
    # above understates what is processable. The natural unit is the sub-run.
    flat = sorted(((x[1], run, x[0], x[2], x[3], x[4]) for run, v in by_run.items()
                   for x in v))
    rdy = [f for f in flat if f[5][READY] / max(f[3] - f[0], 1) >= args.min_ready]
    nf_r = sum(f[4] for f in rdy)
    nf_all = sum(f[4] for f in flat)
    print(f'\nBY SUB-RUN, which is the real unit:')
    print(f'  fully v12-covered  {len(rdy):>4} of {len(flat)} sub-runs, '
          f'{nf_r:>6} of {nf_all} decoded files ({nf_r/nf_all:.1%})')

    # Contiguous stretches of ready sub-runs, in wall-clock order.
    blocks, cur = [], []
    rset = {(f[1], f[2]) for f in rdy}
    for f in flat:
        if (f[1], f[2]) in rset:
            cur.append(f)
        elif cur:
            blocks.append(cur); cur = []
    if cur:
        blocks.append(cur)
    blocks.sort(key=lambda b: -sum(x[4] for x in b))
    print(f'\n  largest contiguous READY stretches:')
    print(f'  {"from":<12} {"to":<12} {"h":>5} {"subs":>5} {"files":>6}  runs')
    for b in blocks[:8]:
        t0, t1 = b[0][0], b[-1][3]
        runs = sorted({x[1] for x in b}, key=lambda r: int(r.split('_')[1]))
        print(f'  {datetime.fromtimestamp(t0, LOCAL):%m-%d %H:%M} '
              f'{datetime.fromtimestamp(t1, LOCAL):%m-%d %H:%M} '
              f'{(t1-t0)/3600:>5.1f} {len(b):>5} {sum(x[4] for x in b):>6}  '
              f'{",".join(runs)}')

    print(f'\nn_TOF runs to ask for, to close the RECOVERABLE part: '
          f'{len(recov_need)}')
    print('  ' + ', '.join(str(r) for r in sorted(recov_need)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
