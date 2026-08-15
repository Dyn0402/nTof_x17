#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
segments.py -- which (DREAM sub-run x n_TOF run) pairs exist, and which are ready.

Proposes segments from wall clock. It is only a PROPOSAL: the ground truth is
`bunch_join`, which matches each DREAM burst to a real n_TOF bunch through the
beam record, and the slim reports how many events actually joined. A segment
proposed here that joins ~0 events is a wall-clock miss and is skipped, loudly.

Reads the same cached listings as `../slim_study/coverage_map.py`; that module
holds the timezone correction and the interval arithmetic, and this one is a
thin front end so the two can never disagree.

USAGE
    python segments.py                 # every segment, grouped by n_TOF run
    python segments.py --ntof 224572   # just that run's
    python segments.py --ready         # only fully v12-covered segments
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'slim_study'))

import coverage_map as cm            # noqa: E402

DATA = HERE.parent / 'slim_study' / 'coverage_inputs'
MIN_OVERLAP_S = 60.0                 # ignore a few seconds of run-boundary touch


@dataclass
class Proposal:
    dream_run: str
    dream_subrun: str
    ntof_run: int
    overlap_s: float
    subrun_s: float
    n_files: int
    reprocessed: bool

    @property
    def fraction(self):
        return self.overlap_s / max(self.subrun_s, 1.0)


def load(data_dir: Path = DATA):
    v12 = cm._spans(data_dir / 'ntof_index_times.txt',
                    shift_s=cm.INDEX_LOCAL_SHIFT_S)
    raw = cm._spans(data_dir / 'ntof_raw_times.txt', skip_short=False,
                    shift_s=-cm.RAW_WRITE_LAG_S)
    subs, _ = cm.load_dream(data_dir / 'dream_eos_subruns.txt',
                            data_dir / 'dream_daq_subruns.txt')
    return v12, raw, subs


def propose(data_dir: Path = DATA, min_overlap_s: float = MIN_OVERLAP_S):
    """Every (sub-run, n_TOF run) pair that shares more than `min_overlap_s`."""
    v12, raw, subs = load(data_dir)
    v12set = {r for r, _, _ in v12}
    spans = {r: (a, b) for r, a, b in v12}
    for r, a, b in raw:
        spans.setdefault(r, (a, b))

    out = []
    for (run, sub), (t0, t1, n) in subs.items():
        for r, (a, b) in spans.items():
            ov = min(b, t1) - max(a, t0)
            if ov > min_overlap_s:
                out.append(Proposal(run, sub, r, ov, t1 - t0, n, r in v12set))
    out.sort(key=lambda p: (p.ntof_run, p.dream_run, p.dream_subrun))
    return out


def for_ntof_run(run: int, data_dir: Path = DATA, ready_only: bool = True,
                 min_overlap_s: float = MIN_OVERLAP_S):
    """Segments to slim when a job is handed one n_TOF run.

    `min_overlap_s` below the default admits the run-boundary slivers the
    campaign skips by design (a few bunches at the tail of a sub-run in the
    NEXT n_TOF run) -- for the deliberate mop-up of 2026-08-16, not routine.
    """
    return [p for p in propose(data_dir, min_overlap_s)
            if p.ntof_run == run and (p.reprocessed or not ready_only)]


def ready_ntof_runs(data_dir: Path = DATA):
    """n_TOF runs that are reprocessed AND overlap at least one beam sub-run."""
    return sorted({p.ntof_run for p in propose(data_dir) if p.reprocessed})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ntof', type=int, default=None)
    ap.add_argument('--ready', action='store_true',
                    help='only segments whose n_TOF run is reprocessed')
    ap.add_argument('--dir', default=str(DATA))
    args = ap.parse_args()

    props = propose(Path(args.dir))
    if args.ntof:
        props = [p for p in props if p.ntof_run == args.ntof]
    if args.ready:
        props = [p for p in props if p.reprocessed]

    runs = sorted({p.ntof_run for p in props})
    tot_files = 0
    for r in runs:
        g = [p for p in props if p.ntof_run == r]
        tag = 'v12' if g[0].reprocessed else 'NOT REPROCESSED'
        print(f'n_TOF {r}  [{tag}]  {len(g)} segment(s)')
        for p in g:
            t0 = datetime.fromtimestamp(
                cm.load_dream(Path(args.dir) / 'dream_eos_subruns.txt',
                              Path(args.dir) / 'dream_daq_subruns.txt'
                              )[0][(p.dream_run, p.dream_subrun)][0], cm.LOCAL)
            print(f'    {p.dream_run:<8} {p.dream_subrun:<16} '
                  f'{t0:%m-%d %H:%M}  overlap {p.overlap_s/60:6.1f} min '
                  f'({p.fraction:5.1%} of the sub-run)  {p.n_files:>4} files')
            tot_files += p.n_files
    print(f'\n{len(props)} segments over {len(runs)} n_TOF runs')
    if not args.ntof:
        rr = ready_ntof_runs(Path(args.dir))
        print(f'{len(rr)} n_TOF runs are reprocessed AND cover beam: '
              f'{rr[0]}..{rr[-1]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
