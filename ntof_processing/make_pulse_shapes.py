#!/usr/bin/env python3
"""Build averaged pulse-shape templates for the PSA (AMPLITUDE OPTION=2).

The templates shipped with the current UserInput (`X17_WAL*_Signal_*.txt`,
`X17_LIQ*_Signal_*.txt`) are each a SINGLE raw pulse, and they are too short:
314 ns for the walls (whose pulse is still at ~1 % of peak at 500 ns) and
24-59 ns for the liquids.  A template that stops inside the tail biases every
fitted amplitude/area and cripples the pileup deconvolution, which is exactly
where the liquids were said to be weak.

This script builds one template per tree by median-averaging thousands of
clean, isolated pulses from the raw stream, aligned on the 50 % leading-edge
crossing with sub-sample interpolation.

Output format matches the existing files: two columns, `t_ns value`, sampled
at 1 ns, in RAW detector polarity (walls positive, plastics/liquids negative),
scaled to a peak amplitude of 1000.

Usage:
    python make_pulse_shapes.py <outdir> <raw_head.bin> [<raw_head.bin> ...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc  # noqa: E402

POLARITY = {'WAL': +1, 'PSS': -1, 'LIQ': -1}
# Per family: (ns before the 50 % crossing, ns after).  MATCH THE SPAN TO THE
# PULSE, not to "longer is safer".  Measured on run 224572 and confirmed by the
# fit chi2 of the first reprocessing:
#   WAL  tail is 4-6 % of peak at 200 ns, 0.5 % at 500 -> a long template is
#        needed, and 720-861 ns lowered chi2 p50 on all four walls
#   LIQ  FWHM 6 ns, tail 0.1-0.4 % at 200 ns -> a 551 ns template made chi2 p50
#        MORE THAN DOUBLE (1.77 -> 4.04 on LIQD) because the fit is then
#        dominated by ~500 ns of baseline noise.  Keep it short.
#   PSS  FWHM 13 ns; same argument as LIQ.
SPAN = {'WAL': (60, 800), 'PSS': (20, 80), 'LIQ': (20, 60)}
T_LO, T_HI = 1_000_000, 15_000_000        # late-time window: flat baseline
# One template per amplitude regime, mirroring how the PSA uses several shapes
# (cf. AveragePulse{Low,Med,High}Amp.dat in the PSA repository).
AMP_BINS = {'WAL': [(300, 900), (900, 2500), (2500, 20000)],
            'PSS': [(300, 900), (900, 2500), (2500, 20000)],
            'LIQ': [(300, 1000), (1000, 3000), (3000, 20000)]}
ISOLATION = 0.10          # nothing above this fraction of the peak before it
REBOUND = 0.06            # a rise of this fraction above the running minimum
                          # after the peak means a second pulse -> reject
ISO_GUARD = 20            # ns before the peak exempt from the isolation test


def collect(paths):
    out = {}
    for path in paths:
        for _o, tag, _v, pay in iter_banks(path):
            if tag != 'ACQC':
                continue
            det, chan, blks = parse_acqc(pay, with_samples=True)
            if det[:3] not in POLARITY:
                continue
            for start, s in blks:
                if T_LO <= start < T_HI and len(s) > 400:
                    out.setdefault(det, []).append(s.astype(np.float64))
    return out


def template(blocks, pol, pre, post, amp_lo, amp_hi):
    stack = []
    for s in blocks:
        base = np.median(np.concatenate([s[:50], s[-50:]]))
        d = pol * (s - base)
        i = int(np.argmax(d))
        amp = d[i]
        if not (amp_lo <= amp <= amp_hi):
            continue
        # Isolation levels are floored at a few times the baseline noise, so
        # that low-amplitude pulses are not rejected by noise alone (which
        # would silently empty the lowest amplitude bin).
        rms = np.std(s[:50] - np.median(s[:50]))
        iso = max(ISOLATION * amp, 5.0 * rms)
        reb = max(REBOUND * amp, 5.0 * rms)
        # nothing of consequence before the pulse ...
        before = d[:max(0, i - ISO_GUARD)]
        if before.size and before.max() > iso:
            continue
        # ... and no SECOND pulse riding on the (legitimately long) tail: after
        # the peak the signal must decrease monotonically to within `reb`
        after = d[i + ISO_GUARD:]
        if after.size and (after - np.minimum.accumulate(after)).max() > reb:
            continue
        # 50 % crossing on the leading edge, linearly interpolated
        lead = d[:i + 1]
        j = np.flatnonzero(lead >= 0.5 * amp)
        if not len(j):
            continue
        j0 = j[0]
        if j0 == 0:
            continue
        y0, y1 = lead[j0 - 1], lead[j0]
        t50 = (j0 - 1) + (0.5 * amp - y0) / (y1 - y0) if y1 != y0 else j0
        grid = np.arange(-pre, post + 1, dtype=float) + t50
        if grid[0] < 0:
            continue
        # ZS blocks vary in length; take what this block covers and leave the
        # rest NaN so every sample of the template averages what it can
        row = np.full(grid.shape, np.nan)
        ok = grid <= len(d) - 1
        # need the peak and a bit of tail -- but never more than the span itself,
        # or a short template (LIQ/PSS) rejects every pulse
        if ok.sum() < min(len(grid), pre + 150):
            continue
        row[ok] = np.interp(grid[ok], np.arange(len(d)), d) / amp
        stack.append(row)
    if len(stack) < 50:
        return None, len(stack)
    arr = np.array(stack)
    n_per = np.sum(~np.isnan(arr), axis=0)
    tpl = np.nanmedian(arr, axis=0)
    tpl[n_per < 50] = np.nan
    keep = ~np.isnan(tpl)
    return tpl[keep], len(stack)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    blocks = collect(sys.argv[2:])
    print(f'{"tree":6s} {"bin":>3s} {"amp range":>13s} {"nused":>7s} {"len":>5s} '
          f'{"tail@200":>9s} {"tail@500":>9s}')
    for det in sorted(blocks):
        fam = det[:3]
        pre, post = SPAN[fam]
        for k, (lo, hi) in enumerate(AMP_BINS[fam]):
            tpl, n = template(blocks[det], POLARITY[fam], pre, post, lo, hi)
            if tpl is None:
                print(f'{det:6s} {k:3d} {f"{lo}-{hi}":>13s} {n:7d}   '
                      f'too few clean pulses -- NOT WRITTEN')
                continue
            t = np.arange(-pre, -pre + len(tpl))
            pk = int(np.argmax(tpl))

            def tail(x, tpl=tpl, pk=pk):
                return tpl[pk + x] if pk + x < len(tpl) else np.nan
            print(f'{det:6s} {k:3d} {f"{lo}-{hi}":>13s} {n:7d} {len(tpl):5d} '
                  f'{tail(200):9.4f} {tail(500):9.4f}')
            # raw polarity, peak amplitude 1000, times starting at 0
            vals = POLARITY[fam] * tpl * 1000.0
            path = outdir / f'X17_{det}_Signal_avg{k}.txt'
            with open(path, 'w') as f:
                for ti, vi in zip(t - t[0], vals):
                    f.write(f'{ti:.1f}\t{vi:.6f}\n')
    print(f'\nwrote templates to {outdir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
