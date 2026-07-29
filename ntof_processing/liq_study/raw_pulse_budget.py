#!/usr/bin/env python3
"""T4, answered at the population level: does v12 over-count liquid pulses?

The per-hit version of this test -- take a v12-only hit, look at the raw trace
where it says a pulse is -- could not be made to work. The PSA's `tof` and the
raw sample index agree only loosely: stacking the raw trace on every hit shows
the pulse peak at a stable per-detector lag (+28 ns LIQA, +21 ns LIQD, +29 ns
LIQB), and the bunch identification is certain (raw bunch 161 scores 20 % of
large isolated peaks against PSA bunch 161, versus a 1.5 % background over all
other bunches), but only ~20 % of unambiguous large raw pulses have a PSA hit at
that lag, and individual overlays show hits on stretches of flat baseline. Until
that is understood, no per-hit classification from this repo should be believed.
See the WARNING in FINDINGS_liquids.md.

This measures the same thing without needing per-hit alignment, and is therefore
not affected: it only requires a hit to land in the right BLOCK, and blocks are
~1000 ns long.

    Over exactly the same zero-suppressed samples, count
      * how many resolvable pulses the RAW data contains -- local maxima above
        `n_sigma` that dominate their own +-`dom` ns, an UPPER BOUND, since
        ripple on a big pulse's tail can also produce a local maximum;
      * how many hits each processing reports.

    A processing that invents pulses shows a ratio above 1. One that merges
    pileup shows a ratio well below 1.

This is the yield question stated as a budget, and it is the one number that
says whether the +14-21 % is recovery or invention.

    python raw_pulse_budget.py <raw.bin>[,<raw.bin>...] label=file.root [...]
"""
import sys
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh  # noqa: E402

TREES = ('LIQA', 'LIQB', 'LIQC', 'LIQD')
T_LO, T_HI = 1_000_000, 18_000_000
PAD = 32768
AMP_CUT = 50.0        # the UserInput's own LIQ amplitude elimination threshold

# (label, absolute ADC threshold or None, n_sigma or None, dominance half-width).
# The middle row is the one to read: it counts raw local maxima at exactly the
# amplitude the processing itself keeps, so it is the like-for-like ceiling.
# Baseline noise is ~20 ADC rms, so amp 50 is only ~2.5 sigma -- a 5-sigma raw
# count is a much HARDER cut than the processing uses and would flatter it.
SETTINGS = (('>5sig/3', None, 5.0, 3),
            ('>50ADC/3', AMP_CUT, None, 3),
            ('>50ADC/6', AMP_CUT, None, 6))


def count_raw(d, rms, thr_adc, n_sigma, dom):
    thr = thr_adc if thr_adc is not None else n_sigma * rms
    x = d[dom:-dom]
    ok = x > thr
    for k in range(1, dom + 1):
        ok &= (x >= d[dom - k:-dom - k]) & (x > d[dom + k:len(d) - dom + k])
    return int(ok.sum())


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    raws = sys.argv[1].split(',')
    specs = [s.split('=', 1) for s in sys.argv[2:]]

    blocks, cur = {}, None
    for raw in raws:
        for _o, tag, _v, pay in iter_banks(raw):
            if tag == 'EVEH':
                cur = parse_eveh(pay)['words'][1]
            elif tag == 'ACQC' and cur is not None:
                det, _c, blks = parse_acqc(pay, with_samples=True)
                if det not in TREES:
                    continue
                for start, s in blks:
                    if T_LO <= start < T_HI and len(s) > 100:
                        blocks.setdefault((det, cur), []).append(
                            (int(start), s.astype(float)))

    psa = {lab: uproot.open(p) for lab, p in specs}
    hdr = (f'{"tree":6s} {"samples":>12s} ' +
           ' '.join(f'{"raw " + lab:>12s}' for lab, _t, _n, _d in SETTINGS) +
           ''.join(f' {lab:>10s}' for lab, _ in specs))
    print(hdr)
    print('-' * len(hdr))
    tot = {}
    for tree in TREES:
        keys = [k for k in blocks if k[0] == tree]
        if not keys:
            continue
        nsamp = 0
        nraw = [0] * len(SETTINGS)
        nhit = {lab: 0 for lab, _ in specs}
        arr = {lab: psa[lab][tree].arrays(['BunchNumber', 'tof'], library='np')
               for lab, _ in specs}
        for det, bunch in keys:
            sub = {lab: arr[lab]['tof'][arr[lab]['BunchNumber'] == bunch]
                   .astype(float) for lab, _ in specs}
            # A raw chunk taken from file N covers whatever bunches file N held,
            # which need not be in the PSA PARTIAL being graded. Counting raw
            # pulses from a bunch the partial does not contain adds pulses with
            # no possible hits and silently drags every ratio down.
            if any(len(v) == 0 for v in sub.values()):
                continue
            for start, s in blocks[(det, bunch)]:
                real = s != PAD
                sr = s[real]
                if len(sr) < 100:
                    continue
                base = float(np.percentile(sr, 90))
                d = np.where(real, base - s, 0.0)
                dif = np.diff(sr)
                rms = 1.4826 * float(np.median(np.abs(dif - np.median(dif)))) / np.sqrt(2)
                if not np.isfinite(rms) or rms <= 0:
                    continue
                nsamp += len(s)
                for j, (_lab, thr, ns_, dm) in enumerate(SETTINGS):
                    nraw[j] += count_raw(d, rms, thr, ns_, dm)
                for lab, _ in specs:
                    t = sub[lab]
                    nhit[lab] += int(((t >= start) & (t < start + len(s))).sum())
        if nsamp == 0:
            continue
        print(f'{tree:6s} {nsamp:12,d} ' +
              ' '.join(f'{v:12,d}' for v in nraw) +
              ''.join(f' {nhit[lab]:10,d}' for lab, _ in specs))
        ref = nraw[1]   # the >50 ADC row: the like-for-like ceiling
        print(f'{"":6s} {"ratio to >50ADC/3":>12s} ' + ' ' * 39 +
              ''.join(f' {nhit[lab] / max(ref, 1):10.2f}' for lab, _ in specs))
        tot.setdefault('raw', [0] * len(SETTINGS))
        for j in range(len(SETTINGS)):
            tot['raw'][j] += nraw[j]
        for lab, _ in specs:
            tot[lab] = tot.get(lab, 0) + nhit[lab]

    if tot:
        print(f'\nALL LIQUIDS: raw resolvable pulses '
              f'{tot["raw"][1]:,} (>50 ADC, dominating +-3 ns)')
        for lab, _ in specs:
            print(f'   {lab:20s} {tot[lab]:10,d} hits   '
                  f'ratio {tot[lab] / max(tot["raw"][1], 1):.2f}')
        print('\nratio > 1 : the processing reports more pulses than the raw '
              'data resolves\n            -- it is inventing them')
        print('ratio < 1 : the processing is still merging pileup; moving the '
              'ratio UP\n            toward 1 without crossing it is recovery, '
              'not invention')
        print('\nThe raw count is an UPPER BOUND: ripple on a large pulse\'s '
              'tail also makes\na local maximum. The stricter columns bracket '
              'how much that matters.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
