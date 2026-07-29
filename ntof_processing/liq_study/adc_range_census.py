#!/usr/bin/env python3
"""SUPERSEDED 2026-07-29 evening -- kept for the record, do not use.

The premise below ("the samples are unsigned 16-bit") is wrong: they are signed
int16, as ntoflib reads them and as the DAQ settings imply. Decoded correctly
there is no wrap -- what this script counts as a wrap is a pulse crossing zero,
and the real amplitude ceiling is ~63 800 ADC rather than the baseline. Use
`saturation_examples.py` and `saturation_clip_or_wrap.py` instead, and see
`../FINDINGS_2026-07-29_signed_decoding.md`.

Census of ADC WRAP-AROUND at the ends of range, in the raw stream1 samples.

Found 2026-07-29 while checking the liquid photon-statistics floor (T6). The
largest liquid pulses were described in the report as "reaching the ~31 000 ADC
rail". They do reach a rail, but they do NOT clip flat there, and that
difference matters:

  * the samples are unsigned 16-bit, and every channel sits on a baseline far
    from both ends of that range, so the largest measurable amplitude is the
    distance from the baseline to the near end;
  * a pulse bigger than that WRAPS modulo 65536 -- it reappears at the opposite
    end of the range instead of being clamped.

Which end depends on the POLARITY, and it is not the same for all of them
(measured on isolated late-time pulses, and confirmed by the sign of the shipped
pulse-shape templates):

  * PSS, LIQ are NEGATIVE-going on a baseline of ~30 700-31 200: a big pulse
    runs down through 0 and reappears near 65 535 (under-range wrap);
  * WAL, PKUP are POSITIVE-going on a baseline of ~34 000-35 000: a big pulse
    runs up through 65 535 and reappears near 0 (over-range wrap).

So a threshold test alone cannot identify a wrap -- on a positive-going wall a
sample above 60 000 is a perfectly legitimate large pulse. What identifies one is
the DISCONTINUITY: a step of more than 20 000 ADC to the neighbouring sample,
which no real pulse at this sampling rate produces. That test is
polarity-independent, and it is what this script uses.

Either way there is no flat top, so a clipping test does not catch it; `amp` is
whatever the last un-wrapped sample on the rising edge happened to be (so it is
randomly UNDER-reported, not clipped to a constant); and the fit sees a shape no
template matches.

This reports, per detector: the measured polarity, how many blocks contain a
wrap, and how many of those fall outside the flash region -- inside it,
saturation is expected and understood.

    python adc_range_census.py <raw_head.bin> [...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc  # noqa: E402

PAD = 32768          # zero-suppression fill value, not a measurement
HIGH = 60000         # near the top of the unsigned 16-bit range
LOW = 400            # near the bottom
JUMP = 20000         # a step no real pulse makes between adjacent samples
FLASH_END = 100_000  # samples (ns); the flash and its recovery live before this


def wraps(v):
    """Indices where the trace steps across the 0/65535 boundary.

    A wrap is an extreme sample with a step of more than JUMP to a neighbour --
    no real pulse moves that far in 1 ns at this sampling rate.

    Which SIDE of the range it ran off cannot be decided from the crossing
    itself: a monotone run through the boundary unwraps smoothly under either
    hypothesis (subtract 65536 from the high samples, or add it to the low
    ones). What settles it is the polarity of the detector, i.e. which way its
    ordinary pulses depart from baseline -- so this script measures that
    separately and reports the two side by side.
    """
    extreme = np.flatnonzero((v > HIGH) | (v < LOW))
    step = np.abs(np.diff(v))
    return [int(i) for i in extreme
            if (i > 0 and step[i - 1] > JUMP) or
               (i < len(v) - 1 and step[i] > JUMP)]


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
                                            up=0, down=0, big=0, base=[]))
                d['blocks'] += 1
                real = s != PAD
                if not real.any():
                    continue
                v = s[real].astype(np.int64)
                base = float(np.median(v[:40]))
                if len(d['base']) < 2000:
                    d['base'].append(base)
                if ((v > HIGH) | (v < LOW)).any():
                    d['big'] += 1              # extreme, wrapped or not
                else:
                    # polarity, from ordinary in-range blocks only: which way
                    # does the largest excursion from baseline go?
                    d['up' if v.max() - base > base - v.min() else 'down'] += 1
                w = wraps(v)
                if not w:
                    continue
                d['wrapped'] += 1
                # sample index within the block -> time in the 20 ms window
                if start + w[0] > FLASH_END:
                    d['wrapped_late'] += 1

    print(f'{"det":6s} {"blocks":>8s} {"baseline":>9s} {"polarity":>16s} '
          f'{"extreme":>8s} {"blocks with":>13s} {"of those,":>11s}')
    print(f'{"":6s} {"":>8s} {"[ADC]":>9s} {"(measured)":>16s} '
          f'{"blocks":>8s} {"a WRAP":>13s} {"late-time":>11s}')
    print('-' * 78)
    for det, d in sorted(st.items()):
        if d['blocks'] < 5:
            continue
        base = np.median(d['base']) if d['base'] else float('nan')
        n_pol = d['up'] + d['down']
        pol = (f'{"POSITIVE" if d["up"] > d["down"] else "NEGATIVE"} '
               f'{max(d["up"], d["down"]) / n_pol:4.0%}') if n_pol else '-'
        print(f'{det:6s} {d["blocks"]:8d} {base:9.0f} {pol:>16s} {d["big"]:8d} '
              f'{d["wrapped"]:8d} {d["wrapped"] / d["blocks"]:5.2%} '
              f'{d["wrapped_late"]:11d}')
    print(f'\nWRAP = an extreme sample (above {HIGH} or below {LOW}) with a step '
          f'of more than\n{JUMP} ADC to a neighbour, i.e. the trace crosses the '
          f'0/65535 boundary. A threshold\nalone will not do: on a POSITIVE-going '
          f'detector a sample above {HIGH} is an ordinary\nlarge pulse, which is '
          f'why "extreme blocks" exceeds the wrap count for the walls.\n'
          f'\nPOLARITY is measured on the in-range blocks only -- the fraction of '
          f'them whose\nlargest excursion from baseline goes that way. It decides '
          f'which end of the range\na wrap ran off: NEGATIVE (PSS, LIQ) runs below '
          f'0 and reappears near 65535,\nPOSITIVE (WAL, PKUP) runs past 65535 and '
          f'reappears near 0.\n\n"late-time" excludes the first '
          f'{FLASH_END / 1000:.0f} us, i.e. the flash and its recovery, where\n'
          f'saturation is expected and already understood.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
