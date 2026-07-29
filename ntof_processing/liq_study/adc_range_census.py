#!/usr/bin/env python3
"""Census of ADC under-range WRAP-AROUND in the raw stream1 samples.

Found 2026-07-29 while checking the liquid photon-statistics floor (T6). The
largest liquid pulses were described in the report as "reaching the ~31 000 ADC
rail". They do reach a rail, but they do NOT clip flat there, and that
difference matters:

  * every detector here is NEGATIVE-going and sits on a baseline near ADC 31 000
    (liquids, plastics) or 34 100 (walls), so the largest measurable amplitude
    is the baseline itself;
  * a pulse bigger than that would need a sample below zero, and the samples are
    unsigned 16-bit, so it WRAPS: the sample reappears near 65 535.

The recorded waveform therefore contains a full-scale POSITIVE spike one or two
samples after the peak. A flat-top test does not catch it, `amp` is whatever the
last un-wrapped sample on the rising edge happened to be (so it is randomly
UNDER-reported, not clipped to a constant), and the fit sees a shape no template
matches.

This counts, per detector: how many blocks contain a wrapped sample, when in the
20 ms window they occur, and -- separately -- the flash region, where wall
saturation is expected and understood.

    python adc_range_census.py <raw_head.bin> [...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc  # noqa: E402

PAD = 32768          # zero-suppression fill value, not a measurement
HIGH = 60000         # a negative-going detector never legitimately goes here
LOW = 400            # about to wrap
FLASH_END = 100_000  # samples (ns); the flash and its recovery live before this


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    st = {}
    for path in sys.argv[1:]:
        for _o, tag, _v, pay in iter_banks(path):
            if tag != 'ACQC':
                continue
            det, _chan, blks = parse_acqc(pay, with_samples=True)
            for start, s in blks:
                if len(s) < 20:
                    continue
                d = st.setdefault(det, dict(blocks=0, wrapped=0, wrapped_late=0,
                                            low=0, nsamp=0, base=[]))
                d['blocks'] += 1
                d['nsamp'] += len(s)
                real = s != PAD
                if len(d['base']) < 2000 and real.any():
                    d['base'].append(float(np.median(s[real][:40])))
                hi = (s > HIGH) & real
                if hi.any():
                    d['wrapped'] += 1
                    # sample index within the block -> time in the 20 ms window
                    if start + int(np.argmax(hi)) > FLASH_END:
                        d['wrapped_late'] += 1
                if ((s < LOW) & real).any():
                    d['low'] += 1

    print(f'{"det":6s} {"blocks":>8s} {"baseline":>9s} {"blocks with":>12s} '
          f'{"of those,":>11s} {"blocks":>8s}')
    print(f'{"":6s} {"":>8s} {"[ADC]":>9s} {"a WRAP":>12s} '
          f'{"late-time":>11s} {"near 0":>8s}')
    print('-' * 60)
    for det, d in sorted(st.items()):
        if d['blocks'] < 5:
            continue
        base = np.median(d['base']) if d['base'] else float('nan')
        print(f'{det:6s} {d["blocks"]:8d} {base:9.0f} '
              f'{d["wrapped"]:7d} {d["wrapped"] / d["blocks"]:5.2%} '
              f'{d["wrapped_late"]:11d} {d["low"]:8d}')
    print(f'\nWRAP = a real (non-{PAD}) sample above {HIGH} in a negative-going '
          f'detector.\n"late-time" excludes the first {FLASH_END / 1000:.0f} us, '
          f'i.e. the flash and its recovery,\nwhere wall saturation is expected '
          f'and already understood.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
