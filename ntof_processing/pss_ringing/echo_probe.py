#!/usr/bin/env python3
"""Do the plastics ring? Part 4: is the 81 ns echo on every pulse or a few?

Part 1 found a razor-sharp 2 ns-wide spike of PSA hits at Delta-t = 81-82 ns
behind every large plastic pulse, on all four channels, absent on the walls.
Part 2's amplitude-normalised MEDIAN trace shows nothing there. Those two are
only compatible if the echo is present on a minority of pulses -- or is present
on all of them but below the level a median resolves.

This keeps the individual traces so the question can be answered directly:
for each large isolated pulse, measure the trace at 78-86 ns above a local trend
interpolated from 60-75 and 95-115 ns, and look at the DISTRIBUTION of that
excess. Deterministic reflection -> a narrow distribution at a fixed fraction of
the pulse, on every pulse. Sporadic afterpulse -> a spike at zero with a tail.

    python echo_probe.py <head_N.bin> [--dets PSSA ...] -o echo.npz
"""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh  # noqa: E402

PRE_SAMPLES = 259
T_PHYS = 20_000


def iter_blocks(raw, dets):
    bunch = -1
    for _o, tag, _v, pay in iter_banks(raw):
        if tag == 'EVEH':
            bunch = int(parse_eveh(pay)['words'][1])
            continue
        if tag != 'ACQC':
            continue
        if pay[0:4].decode('ascii', 'replace') not in dets:
            continue
        det, chan, blks = parse_acqc(pay, with_samples=True)
        for start, s in blks:
            if start >= T_PHYS:
                yield det, chan, bunch, start, s.view('<i2')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw', nargs='+')
    ap.add_argument('--dets', nargs='+',
                    default=['PSSA', 'PSSB', 'PSSC', 'PSSD', 'WALA', 'LIQA'])
    ap.add_argument('--pre', type=int, default=40)
    ap.add_argument('--post', type=int, default=260)
    ap.add_argument('--min-amp', type=float, default=2000.0)
    ap.add_argument('--max-rows', type=int, default=60000)
    ap.add_argument('-o', '--out', default='echo.npz')
    args = ap.parse_args()
    dets = set(args.dets)

    rows, meta = defaultdict(list), defaultdict(list)
    for raw in args.raw:
        for det, chan, bunch, start, s in iter_blocks(raw, dets):
            if len(rows[det]) >= args.max_rows:
                continue
            v = s.astype(np.float64)
            base = np.median(v[:min(200, len(v))])
            dev = v - base
            i = int(np.argmax(np.abs(dev)))
            pk = dev[i]
            pol = 1.0 if pk > 0 else -1.0
            if abs(pk) < args.min_amp or i < args.pre:
                continue
            if i + args.post > len(v):
                continue                      # need the whole window, no NaNs
            w = dev[i - args.pre:i + args.post] * pol / abs(pk)
            # reject blocks with a second full-size pulse inside the window
            tail = w[args.pre:]
            below = np.flatnonzero(tail < 0.2)
            if below.size == 0 or tail[below[0]:].max() > 0.5:
                continue
            rows[det].append(w.astype(np.float32))
            meta[det].append((abs(pk), chan, bunch, start + i - PRE_SAMPLES))

    out = {'pre': args.pre}
    for det, r in rows.items():
        m = np.array(r)
        out[f'{det}_rows'] = m
        out[f'{det}_meta'] = np.array(meta[det], dtype=np.float64)
        print(f'{det}: {m.shape[0]:,} traces, median peak '
              f'{np.median(np.array(meta[det])[:, 0]):,.0f} ADC')
    np.savez_compressed(args.out, **out)
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
