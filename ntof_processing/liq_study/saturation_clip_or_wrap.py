#!/usr/bin/env python3
"""Do saturated pulses CLIP at the rail, or do they WRAP past it? Both happen --
this measures which, where, and how often.

The two look alike in a plot, because the liquid baseline (~+31 200) is only
~1 550 counts below the POSITIVE rail. So a sample near +32 767 can be either:

  * a WRAP -- the true value went below -32 768 and the stored value is
    true + 65 536, which lands just under the positive rail; or
  * an ordinary positive OVERSHOOT that clipped on the near rail while the pulse
    was recovering.

They are told apart by their NEIGHBOURS, not by their value:

  * wrap      : a positive sample sitting immediately next to negative-rail
                samples, i.e. inside the deepest part of the pulse, with no
                intermediate values between them (a 65 000-count step);
  * overshoot : a positive sample whose neighbours are near baseline, i.e. after
                the pulse has come back up.

    python saturation_clip_or_wrap.py [<outdir>] <raw_head.bin> [...]

Prints a per-detector table and, if an outdir is given, draws one zoomed example
of each behaviour found.

RESULT (segments 8 and 20 of run 224572): there is no arithmetic wrap anywhere.
Saturated samples are always exactly at a rail code. What the "wrap-like" column
counts is a RAIL-TO-RAIL FLIP: in the deepest flash saturations on LIQA/LIQB the
output jumps from -32768 to exactly +32767 for a few samples and back, with no
intermediate values. A true wrap would store true+65536, i.e. arbitrary values
just below the positive rail; those do not occur.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh  # noqa: E402

RAIL_LO, RAIL_HI = -32768, 32767
DEEP_NEG = -30000        # "in the deepest part of the pulse"
HIGH_POS = 30000         # "at or near the positive rail"
APPROACH = 5000          # a genuine clip is walked into; a ZS fill is not
T_LATE = 1_000_000       # ns


def segment(path):
    digits = ''.join(c for c in Path(path).stem if c.isdigit() or c == '_')
    tail = digits.rsplit('_', 1)[-1]
    return int(tail) if tail else -1


def runs_at(v, rail):
    at = np.flatnonzero(v == rail)
    if at.size == 0:
        return []
    return [(int(g[0]), int(g[-1]))
            for g in np.split(at, np.flatnonzero(np.diff(at) != 1) + 1)]


def genuine(v, i0, i1, rail):
    """Approached from outside, so not a zero-suppression fill (0x8000)."""
    near = False
    if i0 > 0:
        near |= abs(int(v[i0 - 1]) - rail) < APPROACH
    if i1 + 1 < len(v):
        near |= abs(int(v[i1 + 1]) - rail) < APPROACH
    return near


def classify(v, i0, i1):
    """Look at one clipped run: does a wrapped sample sit inside or beside it?

    Returns (n_wrap, n_overshoot) for the neighbourhood of this run.
    """
    lo, hi = max(0, i0 - 6), min(len(v), i1 + 7)
    seg = v[lo:hi]
    pos = np.flatnonzero(seg >= HIGH_POS)
    n_wrap = 0
    for p in pos:
        left = seg[p - 1] if p > 0 else None
        right = seg[p + 1] if p + 1 < len(seg) else None
        if ((left is not None and left <= DEEP_NEG) or
                (right is not None and right <= DEEP_NEG)):
            n_wrap += 1
    # overshoot: positive-rail samples elsewhere in the block, away from the dip
    far = np.concatenate([v[:max(0, i0 - 20)], v[min(len(v), i1 + 20):]])
    n_over = int((far >= RAIL_HI).sum())
    return n_wrap, n_over


def main():
    args = sys.argv[1:]
    outdir = None
    if args and not args[0].endswith('.bin'):
        outdir = Path(args[0])
        outdir.mkdir(parents=True, exist_ok=True)
        args = args[1:]
    if not args:
        print(__doc__)
        return 1

    st = defaultdict(lambda: dict(clip=0, clip_late=0, wrapped=0, wrapped_late=0,
                                  overshoot_blocks=0, fill=0, maxpos=-1 << 30))
    shots = {}
    for path in args:
        seg, bunch, event = segment(path), -1, -1
        for _o, tag, _v, pay in iter_banks(path):
            if tag == 'EVEH':
                h = parse_eveh(pay)
                bunch, event = int(h['words'][1]), int(h['event'])
                continue
            if tag != 'ACQC':
                continue
            det, _chan, blks = parse_acqc(pay, with_samples=True)
            for start, s in blks:
                if len(s) < 40:
                    continue
                v = s.view('<i2').astype(np.int64)
                base = float(np.median(v[:40]))
                rail = RAIL_LO if base > 0 else RAIL_HI
                d = st[det]
                d['maxpos'] = max(d['maxpos'], int(v.max()))
                for i0, i1 in runs_at(v, rail):
                    if not genuine(v, i0, i1, rail):
                        d['fill'] += 1
                        continue
                    late = (start + i0) > T_LATE
                    d['clip'] += 1
                    d['clip_late'] += int(late)
                    n_wrap, n_over = classify(v, i0, i1)
                    if n_wrap:
                        d['wrapped'] += 1
                        d['wrapped_late'] += int(late)
                    if n_over:
                        d['overshoot_blocks'] += 1
                    key = ('wrap' if n_wrap else ('late' if late else 'flash'), det)
                    if key not in shots:
                        shots[key] = dict(det=det, v=v, i0=i0, i1=i1, base=base,
                                          start=start, seg=seg, bunch=bunch,
                                          event=event, n_wrap=n_wrap, late=late)

    print(f'\n{"det":6s} {"clipped runs":>12s} {"late":>6s} {"with a wrapped":>15s} '
          f'{"late":>6s} {"overshoot":>10s} {"fill runs":>10s} {"max sample":>11s}')
    for det in sorted(st):
        d = st[det]
        print(f'{det:6s} {d["clip"]:12d} {d["clip_late"]:6d} {d["wrapped"]:15d} '
              f'{d["wrapped_late"]:6d} {d["overshoot_blocks"]:10d} {d["fill"]:10d} '
              f'{d["maxpos"]:11d}')

    if outdir is None or not shots:
        return 0

    pick = ([k for k in shots if k[0] == 'wrap'][:2] +
            [k for k in shots if k[0] == 'late'][:2] +
            [k for k in shots if k[0] == 'flash'][:2])[:4]
    fig, axes = plt.subplots(1, len(pick), figsize=(4.6 * len(pick), 4.4))
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, pick):
        b = shots[key]
        v, i0, i1 = b['v'], b['i0'], b['i1']
        lo, up = max(0, i0 - 25), min(len(v), i1 + 25)
        t = np.arange(lo, up)
        ax.axhline(RAIL_LO, color='crimson', lw=1.2, label='rail -32768')
        ax.axhline(RAIL_HI, color='tab:purple', lw=1.2, label='rail +32767')
        ax.axhline(b['base'], color='0.4', lw=1.0, ls=':', label='baseline')
        ax.plot(t, v[lo:up], color='tab:blue', lw=1.4, marker='.', ms=3)
        at = np.arange(i0, i1 + 1)
        ax.plot(at, v[at], color='crimson', lw=2.5)
        pos = t[v[lo:up] >= HIGH_POS]
        if len(pos):
            ax.plot(pos, v[pos], 'o', mfc='none', mec='tab:purple', ms=7,
                    label='at/near +rail')
        ax.set_ylim(RAIL_LO - 3000, RAIL_HI + 3000)
        ax.set_xlabel('sample [ns]')
        ax.set_ylabel('sample value [ADC, signed]')
        kind = ('WRAP: positive sample inside the clip' if b['n_wrap'] else
                ('clip, physics time' if b['late'] else 'clip, flash'))
        ax.set_title(f'{b["det"]}  seg {b["seg"]} bunch {b["bunch"]}\n'
                     f't = {(b["start"] + i0) / 1e6:.3f} ms — {kind}', fontsize=8)
        ax.legend(fontsize=6, loc='center right')
    fig.suptitle('Clip or wrap? run 224572 stream1, samples as signed int16',
                 fontsize=11)
    fig.tight_layout()
    p = outdir / 'sat_clip_or_wrap.png'
    fig.savefig(p, dpi=130)
    print('wrote', p)
    return 0


if __name__ == '__main__':
    sys.exit(main())
