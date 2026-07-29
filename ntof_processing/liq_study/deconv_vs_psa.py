#!/usr/bin/env python3
"""How much MORE can we get from the raw liquid waveforms than the PSA gives us?

The claim "the liquids are pileup-limited" is only useful with a number
attached. This measures one: run an iterative multi-pulse deconvolution on the
raw stream1 waveforms, and compare -- on exactly the same blocks -- against the
hits the PSA reports for the same (tree, bunch, time window).

The comparison is apples to apples because the DAQ zero-suppression means the
PSA and this code see the identical set of samples.

Method: greedy matched-filter deconvolution. Repeatedly take the largest peak
of the template-matched residual, fit its amplitude and sub-sample time by
least squares, subtract the scaled template, and stop when no peak exceeds the
threshold. That is the standard approach the PSA's pulse-by-pulse recognition
does NOT do -- it recognises candidates from the derivative and fits them one
at a time, so heavily overlapped pulses are merged rather than separated.

    python deconv_vs_psa.py <raw_head.bin> <psa_file.root> <bunches> [outdir]
        bunches: comma-separated, e.g. 161,162,163
"""
import sys
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path.home() / 'PycharmProjects/nTof_x17_DAQ/stream1_monitor'))
from ntof_raw import iter_banks, parse_acqc, parse_eveh  # noqa: E402

TREES = ('LIQA', 'LIQB', 'LIQC', 'LIQD')
T_LO, T_HI = 1_000_000, 18_000_000
AMP_THR = 50.0             # same amplitude threshold the UserInput uses
PRE, POST = 20, 200


def build_template(blocks):
    """Median normalised pulse from clean isolated pulses in this sample."""
    rows = []
    for s in blocks:
        base = np.median(np.concatenate([s[:50], s[-50:]]))
        d = -(s - base)
        i = int(np.argmax(d))
        a = d[i]
        if a < 300 or i < PRE + 5 or i + POST + 5 > len(d):
            continue
        rms = np.std(s[:50] - np.median(s[:50]))
        before = d[:max(0, i - 8)]
        if before.size and before.max() > max(0.10 * a, 5 * rms):
            continue
        after = d[i + 8:]
        if after.size and (after - np.minimum.accumulate(after)).max() > max(0.06 * a, 5 * rms):
            continue
        rows.append(d[i - PRE:i + POST] / a)
    if len(rows) < 30:
        return None
    m = np.median(np.array(rows), axis=0)
    return m / m.max()


def deconvolve(d, tmpl, rms, max_pulses=400):
    """Greedy matched-filter deconvolution. Returns (peak times, amplitudes).

    `mode='valid'` is used deliberately: its output index i means "template
    STARTS at sample i", which is an unambiguous alignment. `mode='same'`
    centres the template on the output index, which is NOT the template's peak
    for an asymmetric pulse -- getting that wrong misplaces every fit by the
    difference and makes the deconvolution find essentially nothing.
    """
    res = d.astype(float).copy()
    n, L = len(res), len(tmpl)
    if n < L + 2:
        return np.array([]), np.array([])
    tp = int(np.argmax(tmpl))
    den = float((tmpl * tmpl).sum())
    times, amps = [], []
    thr = max(AMP_THR, 4.0 * rms)
    for _ in range(max_pulses):
        mf = np.correlate(res, tmpl, mode='valid')      # index = template start
        i = int(np.argmax(mf))
        a = float(mf[i] / den)                          # least-squares amplitude
        if a < thr:
            break
        res[i:i + L] -= a * tmpl
        times.append(i + tp)                            # report the pulse peak
        amps.append(a)
    return np.array(times), np.array(amps)


def main():
    raw, psa_path = sys.argv[1], sys.argv[2]
    bunches = [int(b) for b in sys.argv[3].split(',')]

    # ---- raw side -----------------------------------------------------------
    blocks = {}
    cur = None
    for _o, tag, _v, pay in iter_banks(raw):
        if tag == 'EVEH':
            cur = parse_eveh(pay)['words'][1]
        elif tag == 'ACQC' and cur in bunches:
            det, chan, blks = parse_acqc(pay, with_samples=True)
            if det not in TREES:
                continue
            for start, s in blks:
                if T_LO <= start < T_HI:
                    blocks.setdefault((det, cur), []).append((start, s.astype(float)))

    # ---- PSA side -----------------------------------------------------------
    f = uproot.open(psa_path)
    print(f'{"tree":5s} {"bunch":>6s} {"blocks":>7s} {"samples":>9s} '
          f'{"PSA hits":>9s} {"deconv":>7s} {"ratio":>6s}')
    tot_psa = tot_dec = 0
    for tree in TREES:
        a = f[tree].arrays(['BunchNumber', 'tof', 'tflash', 'amp'], library='np')
        tmpl_src = [s for (d, b), v in blocks.items() if d == tree for _, s in v]
        tmpl = build_template(tmpl_src)
        if tmpl is None:
            print(f'{tree}: too few clean pulses to build a template')
            continue
        for b in bunches:
            v = blocks.get((tree, b))
            if not v:
                continue
            # PSA hits in the same absolute-time window
            m = (a['BunchNumber'] == b) & (a['tof'] >= T_LO) & (a['tof'] < T_HI)
            n_psa = int(m.sum())
            n_dec, nsamp = 0, 0
            for start, s in v:
                base = np.median(np.concatenate([s[:30], s[-30:]]))
                d = -(s - base)
                rms = float(np.std(s[:30] - np.median(s[:30])))
                t, amp = deconvolve(d, tmpl, rms)
                n_dec += len(t)
                nsamp += len(s)
            tot_psa += n_psa
            tot_dec += n_dec
            print(f'{tree:5s} {b:6d} {len(v):7d} {nsamp:9d} {n_psa:9d} '
                  f'{n_dec:7d} {n_dec/max(n_psa,1):6.2f}')
    print(f'\nTOTAL  PSA {tot_psa}   deconvolution {tot_dec}   '
          f'ratio {tot_dec/max(tot_psa,1):.2f}')
    print('ratio > 1 means the raw waveforms contain resolvable pulses the PSA '
          'is not reporting')
    return 0


if __name__ == '__main__':
    sys.exit(main())
