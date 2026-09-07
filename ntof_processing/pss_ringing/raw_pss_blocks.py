#!/usr/bin/env python3
"""Do the plastics ring? Part 2: what the raw stream1 trace does after a pulse.

Pulls every zero-suppressed block of a plastic channel out of a raw chunk,
locates the largest excursion in each, and reports the shape of what follows.
Run with --stack to build the amplitude-normalised MEDIAN trace after a large
isolated pulse -- ringing survives a median over thousands of pulses, noise and
uncorrelated pile-up do not.

    python raw_pss_blocks.py <head.bin> [--dets PSSA PSSB] [--stack out.npz]
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh  # noqa: E402

PRE = 259
RAIL_NEG = -32768
T_PHYS = 20_000          # samples; below this is the flash block / divert


def iter_blocks(raw, dets, max_bunches=None):
    """(det, chan, bunch, start_sample, samples) for the wanted detectors."""
    bunch = -1
    seen = set()
    for _o, tag, _v, pay in iter_banks(raw):
        if tag == 'EVEH':
            bunch = int(parse_eveh(pay)['words'][1])
            seen.add(bunch)
            if max_bunches and len(seen) > max_bunches:
                return
            continue
        if tag != 'ACQC':
            continue
        if pay[0:4].decode('ascii', 'replace') not in dets:
            continue
        det, chan, blks = parse_acqc(pay, with_samples=True)
        for start, s in blks:
            yield det, chan, bunch, start, s.view('<i2')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw')
    ap.add_argument('--dets', nargs='+', default=['PSSA', 'PSSB', 'PSSC', 'PSSD'])
    ap.add_argument('--max-bunches', type=int, default=None)
    ap.add_argument('--stack', help='write the aligned normalised stack here')
    ap.add_argument('--pre', type=int, default=120, help='samples kept before the peak')
    ap.add_argument('--post', type=int, default=2000, help='samples kept after')
    ap.add_argument('--min-amp', type=float, default=2000.0,
                    help='minimum peak excursion from baseline, ADC counts')
    ap.add_argument('--max-stack', type=int, default=20000)
    args = ap.parse_args()
    dets = set(args.dets)

    lens = defaultdict(list)
    polarity = Counter()
    stack, stack_meta, truncated = defaultdict(list), defaultdict(list), Counter()
    nblk = Counter()

    for det, chan, bunch, start, s in iter_blocks(args.raw, dets, args.max_bunches):
        if start < T_PHYS:
            continue                                   # flash block
        nblk[det] += 1
        lens[det].append(len(s))
        v = s.astype(np.float64)
        base = np.median(v[:min(200, len(v))])
        dev = v - base
        i = int(np.argmax(np.abs(dev)))
        pol = 1 if dev[i] > 0 else -1
        polarity[(det, pol)] += 1
        if abs(dev[i]) < args.min_amp:
            continue
        if not args.stack:
            continue
        # isolated: the peak must be the only large excursion in the block.
        # The search has to start beyond the pulse itself, whose width differs
        # by a factor ~10 between the plastics (~5-18 ns) and the walls (~72 ns),
        # so find where the pulse falls back through 20 % rather than assuming.
        if i < args.pre:
            truncated[f'{det}:head'] += 1
            continue
        tail = dev[i:] * pol
        below = np.flatnonzero(tail < 0.2 * abs(dev[i]))
        if below.size == 0:
            truncated[f'{det}:never-returns'] += 1
            continue
        seg = tail[below[0]:]
        if seg.size and seg.max() > 0.5 * abs(dev[i]):
            truncated[f'{det}:second-pulse'] += 1
            continue
        w = np.full(args.pre + args.post, np.nan)
        lo, hi = i - args.pre, min(len(dev), i + args.post)
        n = hi - (i - args.pre)
        w[:n] = dev[lo:hi] * pol / abs(dev[i])
        if hi < i + args.post:
            truncated[f'{det}:block-ends'] += 1
        if len(stack[det]) < args.max_stack:
            stack[det].append(w)
            stack_meta[det].append((abs(dev[i]), start + i - PRE, len(s), hi - i))

    for det in sorted(nblk):
        L = np.array(lens[det])
        pos = polarity[(det, 1)]
        neg = polarity[(det, -1)]
        print(f'{det}: {nblk[det]:,} physics blocks | length median {np.median(L):.0f} '
              f'p90 {np.percentile(L, 90):.0f} max {L.max():,} samples | '
              f'peak polarity +{pos:,} / -{neg:,}')
    if truncated:
        print('rejected/flagged:', dict(truncated))
    if args.stack:
        out = {}
        for det, rows in stack.items():
            a = np.array(rows)
            out[f'{det}_median'] = np.nanmedian(a, axis=0)
            out[f'{det}_n'] = np.sum(~np.isnan(a), axis=0)
            out[f'{det}_meta'] = np.array(stack_meta[det])
            print(f'{det}: stacked {a.shape[0]:,} pulses '
                  f'(median peak {np.median(np.array(stack_meta[det])[:, 0]):,.0f} ADC)')
        out['pre'] = args.pre
        np.savez_compressed(args.stack, **out)
        print(f'wrote {args.stack}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
