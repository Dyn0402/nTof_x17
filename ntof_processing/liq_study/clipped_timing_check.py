#!/usr/bin/env python3
"""Does a clipped liquid pulse still get the right TIME from the PSA?

Motivation: `satuflag`-set hits have to be cut for amplitude (their `amp` is a
fit extrapolation, 66 k to 7.6 M against a 63 800 ceiling). But a physics-time
liquid clip is only 2-5 samples wide, and the tree data already hint that the
distortion is confined to the top of the pulse: amplitude-matched, saturated
hits are 35-85 % wider at HALF height but unchanged at TENTH height. If the
sub-clip waveform is intact, the arrival time should be too, and the hits are
recoverable as time-only hits.

This checks it directly on the raw traces. `tof` cannot be compared to a
constant-fraction crossing of the pulse's own amplitude -- for a clipped pulse
that amplitude is exactly what is unknown -- so instead both populations are
referenced to a crossing of a FIXED absolute depth on the rising edge, well
below any clip:

    dt = tof - t_cross(base - LEVEL)          LEVEL = 5 000 counts

Then compare dt for clipped pulses against dt for large UNCLIPPED pulses. Equal
dt means the PSA time reference is stable under clipping. The control is
amplitude-restricted so the two populations have comparable edge slopes; that
match is approximate, and with the handful of clips available this is a
consistency check, not a calibration.

    python clipped_timing_check.py <reproc_dir> <raw_head.bin> [det]

Time base: absolute sample time of payload sample j is `start + j - 259` for a
zero-suppressed block and `start + j` for the flash block (which carries no
pre-samples) -- see FINDINGS_2026-07-29_signed_decoding.md §4.
"""
import sys
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh  # noqa: E402

BR = ['segment', 'BunchNumber', 'tof', 'amp', 'satuflag', 'fwhm', 'fwtm']
RAIL_NEG = -32768
PRESAMPLES = 259
LEVEL = 5_000.0           # fixed depth for the timing reference [counts]
CTL_MIN = 40_000.0        # unclipped control: depth at least this
WIN = 400.0               # ns searched around the raw reference time


def segment(path):
    digits = ''.join(c for c in Path(path).stem if c.isdigit() or c == '_')
    tail = digits.rsplit('_', 1)[-1]
    return int(tail) if tail else -1


def cross_time(v, base, j_peak, level):
    """Absolute-index crossing of `base - level` on the rising (falling) edge,
    linearly interpolated. Returns None if the edge never reaches the level."""
    target = base - level
    seg = v[:j_peak + 1]
    below = np.flatnonzero(seg <= target)
    if below.size == 0:
        return None
    k = int(below[0])
    if k == 0:
        return 0.0
    y0, y1 = float(seg[k - 1]), float(seg[k])
    if y0 == y1:
        return float(k)
    return (k - 1) + (y0 - target) / (y0 - y1)


def main():
    reproc, raw = Path(sys.argv[1]), sys.argv[2]
    det = sys.argv[3] if len(sys.argv) > 3 else 'LIQA'
    seg, bunch = segment(raw), -1
    clipped, control = [], []

    for _o, tag, _v, pay in iter_banks(raw):
        if tag == 'EVEH':
            bunch = int(parse_eveh(pay)['words'][1])
            continue
        if tag != 'ACQC':
            continue
        d, _c, blks = parse_acqc(pay, with_samples=True)
        if d != det:
            continue
        for start, s in blks:
            if len(s) < 60:
                continue
            v = s.view('<i2').astype(np.int64)
            base = float(np.median(v[:40]))
            j = int(np.argmin(v))
            depth = base - float(v[j])
            at_rail = v == RAIL_NEG
            # a clip is APPROACHED; zero-suppression fill (also 0x8000) is not
            is_clip = bool(at_rail.any()) and depth > 20_000 and j > 5
            if not is_clip and depth < CTL_MIN:
                continue
            tc = cross_time(v, base, j, LEVEL)
            if tc is None:
                continue
            off = -PRESAMPLES if start > 0 else 0
            rec = dict(bunch=bunch, t_ref=start + tc + off, depth=depth,
                       n_rail=int(at_rail.sum()), flash=start == 0)
            (clipped if is_clip else control).append(rec)

    print(f'{Path(raw).name}  {det}: {len(clipped)} clipped, '
          f'{len(control)} unclipped controls (depth > {CTL_MIN:,.0f})')
    if not clipped and not control:
        return 0

    part = seg // 10 + 1
    with uproot.open(reproc / f'run224572_{part:04d}.root') as fh:
        a = fh[det].arrays(BR, library='np')

    def match(recs, label, want_flash=None):
        if want_flash is not None:
            recs = [r for r in recs if r['flash'] == want_flash]
            if not recs:
                print(f'  {label}: none')
                return None
        rows = []
        for c in recs:
            m = (a['segment'] == seg) & (a['BunchNumber'] == c['bunch'])
            tof, amp, sf = a['tof'][m], a['amp'][m], a['satuflag'][m]
            fw, ft = a['fwhm'][m], a['fwtm'][m]
            sel = np.abs(tof - c['t_ref']) < WIN
            if not sel.any():
                continue
            k = np.flatnonzero(sel)[int(np.argmax(amp[sel]))]
            rows.append((float(tof[k] - c['t_ref']), float(amp[k]), bool(sf[k]),
                         c['depth'], c['n_rail'], c['flash'],
                         float(fw[k]), float(ft[k])))
        if not rows:
            print(f'  {label}: matched 0')
            return None
        dt = np.array([r[0] for r in rows])
        amp = np.array([r[1] for r in rows])
        flag = np.array([r[2] for r in rows])
        print(f'  {label}: matched {len(rows)} of {len(recs)}   '
              f'dt = tof - t_cross({LEVEL:,.0f}) : median {np.median(dt):7.1f} ns'
              f'  p16 {np.percentile(dt, 16):7.1f}  p84 {np.percentile(dt, 84):7.1f}')
        print(f'      flagged {int(flag.sum())}/{len(rows)}   '
              f'amp p50 {np.median(amp):12.0f}   '
              f'fwhm p50 {np.median([r[6] for r in rows]):5.1f}   '
              f'fwtm p50 {np.median([r[7] for r in rows]):5.1f}')
        return dt

    # The split that matters: a flash-region clip sits on a recovering baseline
    # with neighbours inside its own window, so a mistiming there says nothing
    # about a clean physics-time clip.
    dt_c = match(clipped, 'clipped  ')
    match(clipped, 'clipped, PHYSICS', want_flash=False)
    match(clipped, 'clipped, flash  ', want_flash=True)
    dt_u = match(control, 'unclipped')
    if dt_c is not None and dt_u is not None:
        print(f'\n  shift of the clipped population: '
              f'{np.median(dt_c) - np.median(dt_u):+.1f} ns '
              f'(clipped n={dt_c.size}, control n={dt_u.size})')
        print('  A shift of order the sampling period means the PSA time survives')
        print('  clipping and these hits are recoverable as TIME-only hits.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
