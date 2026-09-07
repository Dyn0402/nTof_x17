#!/usr/bin/env python3
"""Do the plastics ring? Part 5: is the 81 ns hit ON anything in the trace?

The 81-82 ns spike is razor sharp, on all four plastics, absent on the walls,
carries a follower amplitude of ~120 ADC INDEPENDENT of the leader's size, and
leaves no bump in the amplitude-normalised mean trace. The remaining question is
binary: does the trace behind those particular hits differ from the trace behind
the leaders that get no such hit?

  same trace  -> the PSA is inventing the hit (fit artifact)
  bump at 81  -> a real secondary pulse the PSA is right to report

    python echo_conditional.py <head_N.bin> [--det PSSB] [--lo 79 --hi 85]
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import uproot

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh  # noqa: E402

REPROC = Path('/media/dylan/data/x17/ntof_reproc/v12_liqpileup')
PRE_SAMPLES = 259
T_PHYS = 20_000.0
BR = ['segment', 'BunchNumber', 'detn', 'tof', 'amp_0', 'amp', 'fwhm', 'chi2',
      'pileup1', 'pileup2', 'pulseshape']


def segment_of(path):
    return int(''.join(c for c in Path(path).stem if c.isdigit()))


def load_blocks(raw, det):
    blocks, bunch = {}, -1
    for _o, tag, _v, pay in iter_banks(raw):
        if tag == 'EVEH':
            bunch = int(parse_eveh(pay)['words'][1])
            continue
        if tag != 'ACQC' or pay[0:4].decode('ascii', 'replace') != det:
            continue
        _d, chan, blks = parse_acqc(pay, with_samples=True)
        for start, s in blks:
            if start >= T_PHYS:
                blocks.setdefault((bunch, chan), []).append(
                    (start - PRE_SAMPLES, s.view('<i2').astype(np.float64)))
    return blocks


def trace_at(blocks, bunch, chan, t, pre, post):
    for t0, v in blocks.get((bunch, chan), ()):
        if t0 <= t < t0 + len(v):
            i = int(round(t - t0))
            if i - pre < 0 or i + post > len(v):
                return None
            base = np.median(v[max(0, i - 250):max(1, i - 60)])
            return -(v[i - pre:i + post] - base)   # plastics go negative
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw', nargs='+')
    ap.add_argument('--det', default='PSSB')
    ap.add_argument('--lo', type=float, default=79.0)
    ap.add_argument('--hi', type=float, default=85.0)
    ap.add_argument('--guard', type=float, default=120.0,
                    help='ns; a control leader must have no follower this close')
    ap.add_argument('--quiet', type=float, default=5000.0)
    ap.add_argument('--lead-amp', type=float, default=3000.0)
    ap.add_argument('--pre', type=int, default=40)
    ap.add_argument('--post', type=int, default=200)
    ap.add_argument('-o', '--out', default='echo_conditional.npz')
    args = ap.parse_args()

    A, B, ampA, ampB = [], [], [], []
    for raw in args.raw:
        seg = segment_of(raw)
        with uproot.open(REPROC / f'run224572_{seg // 10 + 1:04d}.root') as fh:
            a = fh[args.det].arrays(BR, library='np')
        keep = (a['segment'] == seg) & (a['tof'] > T_PHYS)
        a = {k: v[keep] for k, v in a.items()}
        blocks = load_blocks(raw, args.det)
        have = {b for b, _c in blocks}

        grp = a['BunchNumber'].astype(np.int64) * 100 + a['detn']
        t, amp = a['tof'].astype(np.float64), a['amp_0'].astype(np.float64)
        o = np.lexsort((t, grp))
        grp, t, amp = grp[o], t[o], amp[o]

        prev = np.full(t.size, np.inf)
        prev[1:] = np.where(grp[1:] == grp[:-1], t[1:] - t[:-1], np.inf)
        nxt = np.full(t.size, np.inf)
        nxt[:-1] = np.where(grp[1:] == grp[:-1], t[1:] - t[:-1], np.inf)
        lead = ((amp > args.lead_amp) & (prev > args.quiet)
                & np.isin(grp // 100, list(have)))

        in_win = lead & (nxt >= args.lo) & (nxt <= args.hi)
        control = lead & (nxt > args.guard)
        for cls, sel, store, astore in (('A', in_win, A, ampA),
                                        ('B', control, B, ampB)):
            for i in np.flatnonzero(sel):
                w = trace_at(blocks, int(grp[i] // 100), int(grp[i] % 100),
                             t[i], args.pre, args.post)
                if w is None:
                    continue
                pk = w[args.pre - 4:args.pre + 8].max()
                if pk < args.lead_amp * 0.5:
                    continue
                store.append((w / pk).astype(np.float32))
                astore.append(pk)
        print(f'{Path(raw).name} seg {seg}: bunches {sorted(have)} | '
              f'with {args.lo:.0f}-{args.hi:.0f} ns follower {int(in_win.sum())}, '
              f'control {int(control.sum())}')

    A, B = np.array(A), np.array(B)
    print(f'\n{args.det}: class A (has the {args.lo:.0f}-{args.hi:.0f} ns hit) '
          f'{A.shape[0]:,} traces, median peak {np.median(ampA):,.0f} ADC | '
          f'class B (nothing within {args.guard:.0f} ns) {B.shape[0]:,}, '
          f'median peak {np.median(ampB):,.0f} ADC')
    if A.size == 0 or B.size == 0:
        return 1
    pre = args.pre
    print('\ndelay      A median    B median      A mean      B mean    A-B mean')
    for d in list(range(60, 100, 2)) + [110, 130, 160]:
        am, bm = np.median(A[:, pre + d]), np.median(B[:, pre + d])
        aa, bb = A[:, pre + d].mean(), B[:, pre + d].mean()
        print(f'{d:5d}   {am:+10.5f}  {bm:+10.5f}  {aa:+10.5f}  {bb:+10.5f}  '
              f'{aa - bb:+10.5f}')
    np.savez_compressed(args.out, A_median=np.median(A, axis=0),
                        B_median=np.median(B, axis=0), A_mean=A.mean(axis=0),
                        B_mean=B.mean(axis=0), A_n=A.shape[0], B_n=B.shape[0],
                        pre=pre, det=args.det)
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
