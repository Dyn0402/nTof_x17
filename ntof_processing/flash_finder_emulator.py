#!/usr/bin/env python3
"""Emulate the PSA gamma-flash finder on raw waveforms, to pre-validate
G-FLASH UserInput parameters before spending a condor round-trip.

Implements G-FLASH OPTION=0 as documented in PSA_Guide_20240704.pdf:

  * work on polarity-corrected signal p = pol * (y - baseline) ("pulses are
    always treated as negative" internally; here p > 0 means "into the pulse")
  * the flash is the FIRST pulse crossing G-FLASH THRESHOLD (in ADC channels
    relative to the flash finder's own baseline), optionally not before the
    lower TIME LIMIT (`threshold/time_limit`)
  * a candidate is rejected if the contiguous chunk above baseline containing
    it is narrower than G-FLASH MIN WIDTH
  * the reported flash time is the constant-fraction (default 0.3) crossing on
    the leading edge of that pulse

This is an approximation of the real code (which uses a derivative-based pulse
recognition and its own baseline estimator), but it reproduces the observed
behaviour of the official files well enough to tell whether a parameter set
latches onto the intended waveform feature in 100 % of bunches.

Usage:
    python flash_finder_emulator.py <raw_head.bin> [<raw_head.bin> ...]
"""
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path.home()
                       / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh  # noqa: E402

# Signal polarity, measured from isolated late-time pulses of run 224572:
# walls positive, plastics and liquids negative, PKUP positive.
POLARITY = {'WAL': +1, 'PSS': -1, 'LIQ': -1, 'PKU': +1}

# (threshold, time_limit_ns, min_width_ns, constant_fraction)
CURRENT = {
    'WAL': (500.0, 0.0, 0.0, 0.3),
    'PSS': (50.0, 0.0, 0.0, 0.3),
    'LIQ': (500.0, 0.0, 100.0, 0.3),
    'PKU': (100.0, 0.0, 1.0, 0.3),
}
PROPOSED = {
    'WAL': (250.0, 11400.0, 0.0, 0.3),
    'PSS': (5000.0, 10000.0, 0.0, 0.3),
    'LIQ': (500.0, 10000.0, 100.0, 0.3),
    'PKU': (100.0, 0.0, 1.0, 0.3),
}


def find_flash(y, pol, threshold, time_limit=0.0, min_width=0.0, cf=0.3,
               base_n=2000):
    """Return the flash time in ns, or None if no candidate qualifies."""
    base = np.median(y[:base_n])
    p = pol * (y.astype(np.float64) - base)
    n = len(p)
    start = int(max(0, time_limit))
    if start >= n:
        return None
    idx = np.flatnonzero(p[start:] > threshold)
    if not len(idx):
        return None
    for i0 in idx + start:
        # contiguous chunk above baseline containing this crossing
        lo = i0
        while lo > 0 and p[lo - 1] > 0:
            lo -= 1
        hi = i0
        while hi < n - 1 and p[hi + 1] > 0:
            hi += 1
        if (hi - lo + 1) < min_width:
            continue
        amp = p[lo:hi + 1].max()
        lvl = cf * amp
        seg = p[lo:i0 + 1]
        j = np.flatnonzero(seg >= lvl)
        return float(lo + (j[0] if len(j) else 0))
    return None


def scan(paths, tmax=30000):
    """{(det, chan, bunch): samples} for the mandatory flash block."""
    out = {}
    cur = None
    for path in paths:
        for _o, tag, _v, pay in iter_banks(path):
            if tag == 'EVEH':
                cur = parse_eveh(pay)['words'][1]
            elif tag == 'ACQC' and cur is not None:
                det, chan, blks = parse_acqc(pay, with_samples=True)
                if det[:3] not in POLARITY:
                    continue
                for start, s in blks:
                    if start == 0 and len(s) >= tmax:
                        out[(det, chan, cur)] = s[:tmax].astype(np.float64)
                        break
    return out


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 1
    blocks = scan(paths)
    bunches = sorted({k[2] for k in blocks})
    print(f'{len(blocks)} flash blocks, {len(bunches)} bunches: {bunches}\n')

    for label, cfg in (('CURRENT', CURRENT), ('PROPOSED', PROPOSED)):
        res = defaultdict(list)
        for (det, chan, b), y in blocks.items():
            fam = det[:3]
            t = find_flash(y, POLARITY[fam], *cfg[fam])
            res[det].append(np.nan if t is None else t)
        print(f'== {label} ==')
        print(f'{"tree":6s} {"n":>4s} {"median":>8s} {"p2":>8s} {"p98":>8s} '
              f'{"spread":>7s} {"bad>50ns":>9s} {"notfound":>9s}')
        med_all = {}
        for det in sorted(res):
            a = np.array(res[det], dtype=float)
            good = a[~np.isnan(a)]
            if not len(good):
                print(f'{det:6s} {len(a):4d}   ALL NOT FOUND')
                continue
            m = np.median(good)
            med_all[det] = m
            bad = np.mean(np.abs(good - m) > 50) * 100
            print(f'{det:6s} {len(a):4d} {m:8.0f} {np.percentile(good,2):8.0f} '
                  f'{np.percentile(good,98):8.0f} '
                  f'{np.percentile(good,98)-np.percentile(good,2):7.0f} '
                  f'{bad:8.1f}% {np.mean(np.isnan(a))*100:8.1f}%')
        walls = [v for k, v in med_all.items() if k.startswith('WAL')]
        scint = [v for k, v in med_all.items()
                 if k.startswith('PSS') or k.startswith('LIQ')]
        if walls and scint:
            print(f'  cross-detector: median(WAL) - median(PSS+LIQ) = '
                  f'{np.median(walls) - np.median(scint):+.0f} ns')
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
