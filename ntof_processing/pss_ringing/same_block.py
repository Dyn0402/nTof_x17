#!/usr/bin/env python3
"""Do the plastics ring? Part 6: same recorded block, or a new threshold crossing?

Every follower hit is one of two very different things:

  * INSIDE the leader's own zero-suppressed block -- the PSA found a second hit
    in a continuously recorded stretch of trace. It may be a real secondary
    pulse or a fit artifact on the primary's tail; the raw samples are there
    either way.
  * In a NEW block -- the analog signal fell back below the zero-suppression
    threshold and then crossed it again. Whatever the PSA made of it, the
    front end really did see something come back.

Splitting the Delta-t spectrum this way says which mechanism carries the tail.

    python same_block.py <head_N.bin>... [--dets PSSB WALA]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import uproot

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh  # noqa: E402

REPROC = Path('/media/dylan/data/x17/ntof_reproc/v12_liqpileup')
PRE_SAMPLES = 259
T_PHYS = 20_000.0
BR = ['segment', 'BunchNumber', 'detn', 'tof', 'amp_0']
BANDS = [(18, 30), (30, 45), (45, 60), (60, 79), (79, 85), (85, 120),
         (120, 200), (200, 400), (400, 800), (800, 2000), (2000, 20000)]


def segment_of(path):
    return int(''.join(c for c in Path(path).stem if c.isdigit()))


def load_spans(raw, dets):
    """{(det, bunch, chan): sorted [(t0, t1)]} of physics-time blocks."""
    spans, bunch = {}, -1
    for _o, tag, _v, pay in iter_banks(raw):
        if tag == 'EVEH':
            bunch = int(parse_eveh(pay)['words'][1])
            continue
        if tag != 'ACQC':
            continue
        if pay[0:4].decode('ascii', 'replace') not in dets:
            continue
        det, chan, blks = parse_acqc(pay, with_samples=False)
        for start, n in blks:
            if start >= T_PHYS:
                t0 = start - PRE_SAMPLES
                spans.setdefault((det, bunch, chan), []).append((t0, t0 + n))
    for v in spans.values():
        v.sort()
    return spans


def block_index(spans, key, t):
    """Index of the block containing t, or -1."""
    v = spans.get(key)
    if not v:
        return -1
    lo = np.searchsorted([b[0] for b in v], t, side='right') - 1
    if lo >= 0 and v[lo][0] <= t < v[lo][1]:
        return lo
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw', nargs='+')
    ap.add_argument('--dets', nargs='+', default=['PSSA', 'PSSB', 'PSSC',
                                                  'PSSD', 'WALA', 'LIQA'])
    ap.add_argument('--quiet', type=float, default=5000.0)
    ap.add_argument('--lead-amp', type=float, default=3000.0)
    ap.add_argument('--max-dt', type=float, default=20000.0)
    ap.add_argument('-o', '--out', default='same_block.json')
    args = ap.parse_args()
    dets = set(args.dets)

    tally = {d: {b: [0, 0, 0] for b in BANDS} for d in dets}   # same, new, gap
    nlead = {d: 0 for d in dets}
    for raw in args.raw:
        seg = segment_of(raw)
        spans = load_spans(raw, dets)
        have = {(d, b) for d, b, _c in spans}
        with uproot.open(REPROC / f'run224572_{seg // 10 + 1:04d}.root') as fh:
            for det in sorted(dets):
                a = fh[det].arrays(BR, library='np')
                keep = (a['segment'] == seg) & (a['tof'] > T_PHYS)
                a = {k: v[keep] for k, v in a.items()}
                grp = a['BunchNumber'].astype(np.int64) * 100 + a['detn']
                t, amp = a['tof'].astype(np.float64), a['amp_0'].astype(np.float64)
                o = np.lexsort((t, grp))
                grp, t, amp = grp[o], t[o], amp[o]
                prev = np.full(t.size, np.inf)
                prev[1:] = np.where(grp[1:] == grp[:-1], t[1:] - t[:-1], np.inf)
                inchunk = np.array([(det, int(g // 100)) in have for g in grp])
                li = np.flatnonzero((amp > args.lead_amp) & (prev > args.quiet)
                                    & inchunk)
                nlead[det] += li.size
                for i in li:
                    key = (det, int(grp[i] // 100), int(grp[i] % 100))
                    b_lead = block_index(spans, key, t[i])
                    j = i + 1
                    while j < t.size and grp[j] == grp[i] \
                            and t[j] - t[i] <= args.max_dt:
                        dt = t[j] - t[i]
                        for band in BANDS:
                            if band[0] <= dt < band[1]:
                                b = block_index(spans, key, t[j])
                                k = 2 if b < 0 else (0 if b == b_lead else 1)
                                tally[det][band][k] += 1
                                break
                        j += 1

    res = {'bands': BANDS, 'n_leaders': nlead, 'dets': {}}
    for det in sorted(dets):
        n = max(nlead[det], 1)
        print(f'\n{det}: {nlead[det]:,} isolated leaders')
        print('   Delta-t band      per leader   same block    new block   '
              'not in any')
        res['dets'][det] = []
        for band in BANDS:
            s, nw, g = tally[det][band]
            tot = s + nw + g
            res['dets'][det].append(dict(lo=band[0], hi=band[1], n_leaders=n,
                                         same=s, new=nw, none=g))
            print(f'   {band[0]:6d}-{band[1]:6d}   {tot / n:10.4f} '
                  f'{s / n:12.4f} {nw / n:12.4f} {g / n:12.4f}')
    Path(args.out).write_text(json.dumps(res))
    print(f'\nwrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
