#!/usr/bin/env python3
"""At what overhang does the offset bootstrap break, and does geometry matter?

The bootstrap bug (see README, fixed in bunch_join 2026-08-12) mediates over
bursts that have no pulse in the n_TOF run, so it should break when those
bursts are the MAJORITY. "Should" is a prediction; this measures it.

Method: take one real segment's burst epochs and one real pulse list, then
CONSTRUCT hypothetical segments at any overhang fraction by trimming bursts
(moves the sub-run's start/end) or pulses (moves the run's bounds). For each,
compare the buggy median-over-all against the correct median-over-matched.

Result (2026-08-12, run_79/stat090_0002 x 224573 as the source):

    THE THRESHOLD IS SHARP AT 50 % AND GEOMETRY-INDEPENDENT.

    start-side overhang only:  51.9 % -> -30.4 s CORRUPT, 49.9 % -> clean
    end-side overhang only:    crosses between 43.5 % and 59.8 %
    both sides:                corrupt throughout

The corruption grows continuously from zero as the fraction crosses 50 %
(the median walking into the edge of the clipped population), so there is no
regime where a >50 % segment is quietly fine. This killed the hypothesis that
one-sided overhang might behave differently from two-sided at equal fraction.

**THE FRACTION IS OF BURSTS, NOT OF TIME**, and the two diverge whenever the
burst density is not uniform -- which is exactly what a beam gap, a DAQ pause
or a parasitic-only stretch produces. Thinning the pre-run bursts while
holding the sub-run's wall-clock extent FIXED:

    thin   time-overhang   burst-overhang   delta_all    corrupted?
     1x       76.6 %          76.5 %        -957.971       YES
     2x       76.6 %          62.0 %        -536.169       YES
     3x       76.6 %          52.2 %        -116.766       YES
     4x       76.6 %          45.1 %          +0.835       no
     8x       76.6 %          29.1 %          +0.836       no

Time-overhang is identical in every row; the corruption tracks the bursts.
So `overlap_frac` from the coverage map is a wall-clock PROXY: a good one
where density is roughly uniform, and wrong for exactly those segments that
straddle a beam gap. A 61 % time overhang can be a sub-50 % burst overhang.
Quote recovery predictions in bursts when it matters.

It also closes the mechanism quantitatively: the corrupted delta is always the
true delta plus an INTEGER number of PS periods. For the exemplar,
delta_bad - delta_true = -958.808 s = -799.0068 x 1.2 s -- 799 grid steps and
8 ms, and the campaign logged 8.3 ms residual rms for that segment. That is
why a wrong lock still shows healthy residuals, and it is why the failure is
always loud: the delta lands either grid-aligned (full-looking join on wrong
bunches -> no coincidence -> the clock fit fails) or off-grid (nothing clears
MATCH_TOL_S -> the join comes back empty). Neither path writes a passing
product, which is the basis for "these bugs destroy data, they do not corrupt
it".
"""
import glob

import numpy as np
import uproot

from ntof_dream_merge.bunch_join import (burst_epochs, dream_events,
                                         MATCH_TOL_S)

SRC = '/media/dylan/data/x17/ntof_reproc/v11_pssfit_width_224573/*.root'
DELTA_TRUE = 0.8371808528900146      # this segment's correct offset


def pulses(pattern=SRC):
    b, t = [], []
    for p in sorted(glob.glob(pattern)):
        a = uproot.open(p)['PKUP'].arrays(['BunchNumber', 'psTime'],
                                          library='np')
        b.append(a['BunchNumber'])
        t.append(a['psTime'] / 1e9)
    o = np.argsort(np.concatenate(b))
    return np.concatenate(t)[o]


def probe(ep, ps, delta_true=DELTA_TRUE):
    """(overhang fraction, buggy delta, correct delta) for one hypothetical."""
    cand = ep - delta_true
    k = np.clip(np.searchsorted(ps, cand), 1, len(ps) - 1)
    k = np.where(np.abs(ps[k - 1] - cand) <= np.abs(ps[k] - cand), k - 1, k)
    sel = np.abs(ps[k] - cand) < MATCH_TOL_S
    if sel.sum() < 3:
        return float(np.mean(~sel)), np.nan, np.nan
    return (float(np.mean(~sel)), float(np.median(ep - ps[k])),
            float(np.median((ep - ps[k])[sel])))


def main():
    _, epoch, _ = burst_epochs('run_79', 'stat090_0002',
                               dream_events('run_79', 'stat090_0002'))
    ps = pulses()
    rows = [('start-side only', [(epoch[i:], ps) for i in
                                 (0, 300, 400, 500, 527, 540, 560, 600, 800)]),
            ('end-side only', [(epoch[epoch - DELTA_TRUE >= ps[0]], ps[:m])
                               for m in (500, 250, 200, 180, 160, 140, 100)]),
            ('both sides', [(epoch, ps[:m]) for m in (300, 200, 120, 80)])]
    for name, cases in rows:
        print(f'\n{name}:')
        print('   overhang%   delta_all     delta_ok   corrupted?')
        for ep, p in cases:
            f, da, dok = probe(ep, p)
            if np.isnan(da):
                continue
            print('     %5.1f %12.3f %11.3f      %s'
                  % (100 * f, da, dok, 'YES' if abs(da - dok) > 1.2 else 'no'))

    # bursts vs time: thin the pre-run bursts, hold the wall-clock extent fixed
    print('\nbursts, not time (pre-run bursts thinned, extent unchanged):')
    print('   thin   time-oh   burst-oh    delta_all   corrupted?')
    early = epoch - DELTA_TRUE < ps[0]
    for thin in (1, 2, 3, 4, 6, 8):
        ep = np.sort(np.concatenate([epoch[early][::thin], epoch[~early]]))
        f, da, dok = probe(ep, ps)
        f_time = (ps[0] - (ep.min() - DELTA_TRUE)) / (ep.max() - ep.min())
        print('    %dx     %5.1f %%   %5.1f %%  %11.3f      %s'
              % (thin, 100 * f_time, 100 * f, da,
                 'YES' if abs(da - dok) > 1.2 else 'no'))


if __name__ == '__main__':
    main()
