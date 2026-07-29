#!/usr/bin/env python3
"""Dump every genuine clipped liquid run in one raw chunk, at ns precision.

One line per clipped run:  det seg bunch trig t_ns nsamples region

`t_ns` is the sample index of the FIRST sample at the rail, which is what a
matching PSA hit time has to be compared against. Companion to
`verify_satuflag.py`, which does the matching.

    python dump_clips.py <raw_head.bin> <out.txt>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh  # noqa: E402

LIQ = ('LIQA', 'LIQB', 'LIQC', 'LIQD')
RAIL = -32768
APPROACH = 5000          # genuine clip is walked into; the 0x8000 ZS fill is not
PRESAMPLES = 259         # measured, see time_base_offset.py
T_LATE = 1_000_000


def segment(path):
    digits = ''.join(c for c in Path(path).stem if c.isdigit() or c == '_')
    tail = digits.rsplit('_', 1)[-1]
    return int(tail) if tail else -1


def main():
    src, dst = sys.argv[1], sys.argv[2]
    seg, bunch, event = segment(src), -1, -1
    out = []
    for _o, tag, _v, pay in iter_banks(src):
        if tag == 'EVEH':
            h = parse_eveh(pay)
            bunch, event = int(h['words'][1]), int(h['event'])
            continue
        if tag != 'ACQC':
            continue
        det, _chan, blks = parse_acqc(pay, with_samples=True)
        if det not in LIQ:
            continue
        for start, s in blks:
            if len(s) < 40:
                continue
            v = s.view('<i2').astype(np.int64)
            at = np.flatnonzero(v == RAIL)
            if at.size == 0:
                continue
            for g in np.split(at, np.flatnonzero(np.diff(at) != 1) + 1):
                i0, i1 = int(g[0]), int(g[-1])
                near = False
                if i0 > 0:
                    near |= abs(int(v[i0 - 1]) - RAIL) < APPROACH
                if i1 + 1 < len(v):
                    near |= abs(int(v[i1 + 1]) - RAIL) < APPROACH
                if not near:
                    continue
                # `start` is the zero-suppression trigger sample; the payload
                # begins PRESAMPLES earlier, so absolute time is start+j-PRESAMPLES.
                # Measured at 258.7 +- 0.6 ns over 220 pulses by time_base_offset.py.
                # The flash block starts at 0 and carries no pre-samples.
                t = start + i0 - (PRESAMPLES if start > 0 else 0)
                out.append(f'{det} {seg} {bunch} {event} {t} {i1 - i0 + 1} '
                           f'{"physics" if t > T_LATE else "flash"}')
    Path(dst).write_text('\n'.join(out) + ('\n' if out else ''))
    print(f'{src}: {len(out)} clipped liquid runs -> {dst}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
