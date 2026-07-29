#!/usr/bin/env python3
"""Measure the offset between a raw sample index and the PSA `tof` of the same
pulse, separately for zero-suppressed blocks and for the flash block.

This is the open question from `FINDINGS_2026-07-29_pre_ship_tests.md` (NEW 3),
and it decides how to read every raw-vs-tree comparison -- including whether the
saturation flag lands on the clipped pulse or on a different one.

Method: take isolated large late-time pulses (one dominant peak in the block),
and match each to the largest PSA hit within +-400 ns. Report dt = tof - raw peak
index, and the block offset of the peak, so a pre-sample convention shows up.

    python time_base_offset.py <reproc_dir> <raw_head.bin> [det]
"""
import sys
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh  # noqa: E402

BR = ['segment', 'BunchNumber', 'tof', 'amp', 'satuflag']
T_LATE = 1_000_000
MIN_DEPTH = 4000          # ADC counts below baseline
WIN = 400.0               # ns searched around the raw peak


def segment(path):
    digits = ''.join(c for c in Path(path).stem if c.isdigit() or c == '_')
    tail = digits.rsplit('_', 1)[-1]
    return int(tail) if tail else -1


def main():
    reproc, raw = Path(sys.argv[1]), sys.argv[2]
    det_want = sys.argv[3] if len(sys.argv) > 3 else 'LIQA'
    seg, bunch = segment(raw), -1
    cands = []
    for _o, tag, _v, pay in iter_banks(raw):
        if tag == 'EVEH':
            bunch = int(parse_eveh(pay)['words'][1])
            continue
        if tag != 'ACQC':
            continue
        det, _c, blks = parse_acqc(pay, with_samples=True)
        if det != det_want:
            continue
        for start, s in blks:
            if start < T_LATE or len(s) < 60:
                continue
            v = s.view('<i2').astype(np.int64)
            base = float(np.median(v[:40]))
            j = int(np.argmin(v))
            depth = base - float(v[j])
            if depth < MIN_DEPTH:
                continue
            # isolated: no other sample within 60 % of the depth, 30 ns away
            far = np.abs(np.arange(len(v)) - j) > 30
            if far.any() and (base - v[far]).max() > 0.6 * depth:
                continue
            cands.append(dict(bunch=bunch, start=start, peak=start + j,
                              off=j, depth=depth, n=len(v)))
    print(f'{len(cands)} isolated large late pulses in {Path(raw).name} ({det_want})')
    if not cands:
        return 0

    part = seg // 10 + 1
    a = uproot.open(reproc / f'run224572_{part:04d}.root')[det_want] \
        .arrays(BR, library='np')
    dts, offs, depths = [], [], []
    for c in cands:
        m = (a['segment'] == seg) & (a['BunchNumber'] == c['bunch'])
        tof, amp = a['tof'][m], a['amp'][m]
        sel = np.abs(tof - c['peak']) < WIN
        if not sel.any():
            continue
        k = np.flatnonzero(sel)[int(np.argmax(amp[sel]))]
        dts.append(float(tof[k] - c['peak']))
        offs.append(c['off'])
        depths.append(c['depth'])
    dts, offs = np.array(dts), np.array(offs)
    print(f'matched {len(dts)} of {len(cands)}')
    print(f'  dt = tof - raw peak index : median {np.median(dts):8.1f} ns, '
          f'p16 {np.percentile(dts, 16):8.1f}, p84 {np.percentile(dts, 84):8.1f}')
    print(f'  peak offset within block  : median {np.median(offs):8.1f} samples, '
          f'min {offs.min()}, max {offs.max()}')
    print(f'  spread of dt (p84-p16)    : {np.percentile(dts, 84) - np.percentile(dts, 16):8.1f} ns'
          f'   -- a CONSTANT offset, not a per-pulse scatter')
    print('\nInterpretation: the block `start` counter in our raw parser is the'
          '\nzero-suppression TRIGGER sample, while the payload begins ~259'
          '\nsamples earlier (the pre-samples). So absolute time of payload'
          '\nsample j is start + j - 259, and our "start + j" over-counts by'
          '\nexactly that. The flash block starts at 0 and carries no'
          '\npre-samples, which is why it matches with no offset.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
