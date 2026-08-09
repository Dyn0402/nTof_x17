#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
segment_diagnose.py -- why does this segment have no coincidence peak?

    python segment_diagnose.py run_118 stat090_0003 224642 --ntof-source DIR

A segment can fail the coarse search for three quite different reasons, and the
error message cannot tell them apart:

  1. too few events            -- nothing to fit, and the fix is to skip it
  2. the pairing is wrong      -- this DREAM sub-run does not overlap this
                                  n_TOF run, and the fix is the coverage map
  3. the BUNCH ASSIGNMENT is offset -- the pairing is right but DREAM triggers
                                  are being compared against candidates from a
                                  different bunch, so no dt can ever work

Case 3 is invisible to every other tool: the join reports a healthy event count,
candidates are produced at the normal rate, and the residual histogram is simply
flat. It is also completely recoverable once known, which is why it is worth
separating.

So this builds the candidate list ONCE (the expensive part) and then re-runs the
coarse search with the DREAM bunch numbers shifted by -N..+N. A tall peak at a
non-zero shift is case 3 and names the offset; a flat scan at every shift is
case 1 or 2.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ntof_processing.slim_pipeline import clockfit as cf        # noqa: E402
from ntof_processing.slim_pipeline.slim import (                # noqa: E402
    Segment, _bind_ntof, join_events, pass1_candidates)


def xcorr_full_burst(ev_b, ev_t, cd_b, cd_t, bin_ns=1000.0, burst_ms=80.0,
                     max_bunches=300, log=print):
    """Is there a correlation at ANY lag across the whole 80 ms burst?

    The coarse search looks in +-50 us, and a bunch shift moves whole bunches,
    so between them they cover "the offset is small" and "the offset is a
    bunch". Neither can see a MILLISECOND offset -- which is exactly what a
    mis-tagged flash reference produces, and this experiment has a documented
    n_TOF tflash mis-tag.

    So: bin DREAM trigger times and candidate times per bunch at `bin_ns`, and
    cross-correlate by FFT. That is O(N log N) and covers every lag in
    +-`burst_ms` at once, which brute force cannot.
    """
    nb = int(burst_ms * 1e6 / bin_ns)
    acc = np.zeros(2 * nb, dtype=np.float64)
    bunches = np.unique(ev_b)[:max_bunches]
    used = 0
    for b in bunches:
        te = ev_t[ev_b == b]
        tc = cd_t[cd_b == b]
        if te.size < 5 or tc.size < 5:
            continue
        a = np.bincount(np.clip((te / bin_ns).astype(int), 0, nb - 1),
                        minlength=nb).astype(np.float64)
        c = np.bincount(np.clip((tc / bin_ns).astype(int), 0, nb - 1),
                        minlength=nb).astype(np.float64)
        A = np.fft.rfft(a, 2 * nb)
        C = np.fft.rfft(c, 2 * nb)
        acc += np.fft.irfft(np.conj(A) * C, 2 * nb)
        used += 1
    if not used:
        log('  cross-correlation: no bunch had enough of both')
        return None
    lags = (np.arange(2 * nb) - 0) * bin_ns
    lags[nb:] -= 2 * nb * bin_ns                      # negative lags fold back
    o = np.argsort(lags)
    lags, acc = lags[o], acc[o]
    med = float(np.median(acc))
    mad = float(np.median(np.abs(acc - med))) * 1.4826
    i = int(np.argmax(acc))
    z = (acc[i] - med) / max(mad, 1e-9)
    log(f'  cross-correlation over +-{burst_ms:g} ms at {bin_ns/1000:g} us '
        f'bins, {used} bunches:')
    log(f'     tallest lag {lags[i]/1e6:+.4f} ms, height {acc[i]:.0f}, '
        f'median {med:.0f}, robust z {z:.1f}')
    top = np.argsort(acc)[::-1][:5]
    log('     top 5 lags: ' + '  '.join(
        f'{lags[j]/1e6:+.3f} ms (z {(acc[j]-med)/max(mad,1e-9):.0f})'
        for j in sorted(top, key=lambda j: -acc[j])))
    return dict(lag_ns=float(lags[i]), z=float(z), used=used)


def diagnose(dream_run, dream_subrun, ntof_run, source=None, span=5,
             log=print):
    seg = Segment(dream_run, dream_subrun, ntof_run,
                  ntof_source=Path(source) if source else None)
    # MUST come before join_events: the join reads PKUP through ntof_io, which
    # resolves paths against its module-level default until this rebinds it to
    # the staged copy. run_segment does the same, in the same order.
    io, files = _bind_ntof(seg)
    log(f'  bound to {len(files)} n_TOF file(s) under {seg.ntof_source}')
    ev = join_events(seg, log=log)
    phys = ~ev['is_flash'].to_numpy()
    ev_b = ev['BunchNumber'].to_numpy().astype(np.int64)[phys]
    ev_t = ev['t_since_flash_ns'].to_numpy().astype(np.float64)[phys]
    if ev_b.size == 0:
        log('no physics events joined -- case 1 (nothing to fit)')
        return
    bunches = np.unique(ev_b)
    log(f'  DREAM bunches {bunches.min()}..{bunches.max()} '
        f'({bunches.size} distinct), {ev_b.size:,} physics events')
    log(f'  t_since_flash: min {ev_t.min()/1e6:.2f} ms, '
        f'max {ev_t.max()/1e6:.2f} ms, median {np.median(ev_t)/1e6:.2f} ms')

    cd, _, _ = pass1_candidates(seg, bunches, log=log)
    cb = cd['bunch']
    log(f'  n_TOF candidate bunches {cb.min()}..{cb.max()}, '
        f'{cd["t"].size:,} candidates')

    log(f'\n  bunch-shift scan (peak counts / floor / S/N at each shift)')
    log(f'  {"shift":>6} {"cands +-50us":>13} {"peak":>7} {"floor":>7} '
        f'{"S/N":>7} {"at ns":>9}')
    best = None
    for s in range(-span, span + 1):
        try:
            _, info = cf.bootstrap(ev_b + s, ev_t, cb, cd['t'],
                                   log=lambda *a, **k: None)
            ok = True
        except RuntimeError:
            # bootstrap raises when there is no peak; redo the histogram by
            # hand so the scan still reports the numbers it saw.
            ei, r, _ = cf.residuals(ev_b + s, ev_t, cb, cd['t'],
                                    cf.K_SEED, cf.T0_SEED,
                                    search=cf.BOOT_SEARCH_NS)
            if r.size == 0:
                log(f'  {s:>+6} {0:>13}   (no candidates)')
                continue
            edges = np.arange(-cf.BOOT_SEARCH_NS,
                              cf.BOOT_SEARCH_NS + cf.BOOT_BIN_NS,
                              cf.BOOT_BIN_NS)
            h, _ = np.histogram(r, bins=edges)
            c = 0.5 * (edges[:-1] + edges[1:])
            i = int(h.argmax())
            far = np.abs(c - c[i]) > cf.BOOT_FLOOR_GAP_NS
            fl = float(np.median(h[far])) if far.any() else 0.0
            info = dict(counts=int(h[i]), floor=fl,
                        snr=h[i] / max(fl, 1.0), peak_ns=float(c[i]),
                        n_candidates=int(r.size))
            ok = False
        mark = '  <-- PEAK' if info['snr'] >= cf.BOOT_MIN_SNR else ''
        log(f'  {s:>+6} {info["n_candidates"]:>13,} {info["counts"]:>7,} '
            f'{info["floor"]:>7.0f} {info["snr"]:>7.1f} '
            f'{info["peak_ns"]:>+9.0f}{mark}')
        if best is None or info['snr'] > best[1]['snr']:
            best = (s, info, ok)

    log('')
    s, info, ok = best
    if info['snr'] < cf.BOOT_MIN_SNR:
        # Nothing within +-50 us at any bunch shift. Before blaming the
        # pairing, look at every lag in the burst -- a ms-scale offset is
        # invisible to both scans above.
        log('  no peak within +-50 us at any bunch shift; widening to the '
            'whole burst\n')
        xc = xcorr_full_burst(ev_b, ev_t, cb, cd['t'], log=log)
        if xc and xc['z'] >= 8 and abs(xc['lag_ns']) > 2 * cf.BOOT_SEARCH_NS:
            log(f'\n  VERDICT: THE OFFSET IS {xc["lag_ns"]/1e6:+.4f} ms, far '
                f'outside the +-50 us coarse search. The pairing is correct '
                f'and the data is recoverable -- this is a flash-reference '
                f'problem, not missing coincidence.')
        elif xc and xc['z'] >= 8:
            log(f'\n  VERDICT: correlation found at {xc["lag_ns"]/1e3:+.1f} us '
                f'(z {xc["z"]:.0f}) -- inside the coarse range, so revisit the '
                f'bootstrap thresholds rather than the pairing.')
        else:
            log(f'\n  VERDICT: no correlation at ANY lag in the burst. This '
                f'DREAM sub-run genuinely has no coincidence with n_TOF '
                f'{ntof_run} -- either the pairing is wrong, or DREAM was not '
                f'triggering on the n_TOF coincidence during these hours.')
    elif s == 0:
        log(f'  VERDICT: peak at shift 0 -- the pairing and the bunch '
            f'assignment are both fine. If the production run failed, the '
            f'cause is downstream of the coarse search.')
    else:
        log(f'  VERDICT: BUNCH ASSIGNMENT IS OFFSET BY {s:+d}. The pairing is '
            f'correct; DREAM events are being compared against candidates '
            f'{abs(s)} bunch(es) {"later" if s < 0 else "earlier"}. '
            f'S/N {info["snr"]:.0f} at that shift versus '
            f'{[i for i in [0]][0]} at zero. This is recoverable -- fix the '
            f'bunch join, do not discard the data.')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('dream_run')
    ap.add_argument('dream_subrun')
    ap.add_argument('ntof_run', type=int)
    ap.add_argument('--ntof-source', default=None)
    ap.add_argument('--span', type=int, default=5)
    a = ap.parse_args()
    diagnose(a.dream_run, a.dream_subrun, a.ntof_run, a.ntof_source, a.span)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
