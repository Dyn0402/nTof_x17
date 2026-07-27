#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
match_window.py -- the DREAM->n_TOF accept window, calibrated against rate.

STRATEGY. The DREAM trigger timestamp is not precise and does not need to be: all
it has to do is identify WHICH n_TOF coincidence fired the trigger, after which
the n_TOF time is the time. So the accept window should be as wide as the DREAM
timing spread genuinely requires and no wider, and "no wider" is rate-dependent --
at 40-80 ms the n_TOF singles rate is ~0.1 hits/us and a wide window is free,
while at 1-3 ms it is ~11 hits/us and the same window matches everything by
accident. Late times therefore MEASURE the window; early times must live inside it.

WHAT THE LATE TIMES SAY. Nearest-match distribution for t_since_flash > 40 ms
(accidental floor 2 % at +-100 ns), after removing the 108.9 ppm clock drift:

    0-150 ns    main band        \\  nothing at all between 150 and 250 ns,
    250-450 ns  second band      /   and ZERO counts anywhere from 500 ns to 20 us

so the required window is +-500 ns, and inside it the matches sit in two discrete
bands rather than a continuum. Accepting BOTH bands:

    main only            40.8 %
    second band only     31.0 %
    both                 27.0 %
    neither               1.2 %   <- the true inefficiency

**98.8 % of DREAM events match**, not the 66 % previously reported. That 66 % was
an artifact of a +-100 ns window, which by construction discarded the whole second
band -- an inefficiency of the cut, not of the data.

WHAT THE SECOND BAND IS. It is a WALL-ONLY effect. Splitting by tree:

    PSSA-D  satellite/main = 0.00-0.01   (plastics: no second band at all)
    WALA-D  satellite/main = 0.68-1.97   (walls: as large as the prompt band)

So the SiPM walls deliver a hit ~330 ns after the trigger that the plastics never
do -- consistent with delayed light in the bar/WLS fibre or SiPM afterpulsing, or
with the PSA timing the later lobe of a double-peaked wall pulse. It is NOT a
wall-vs-plastic misalignment (time_align.py measures that at 0.0 +- 0.5 ns here).
Since 31 % of events have ONLY the delayed wall hit, the band has to be accepted,
not vetoed.

THRESHOLDS. None are applied on the n_TOF side anywhere in this module: every hit
in the official trees is a candidate. The only threshold in play is n_TOF's own PSA
(amp >= ~50 ADC in run224572). Adding a wall/plastic amplitude cut is the obvious
lever if purity at early times needs improving -- untested so far.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from ntof_dream_merge.ntof_io import read_bunches   # noqa: E402

# Measured at t > 40 ms where the accidental floor is 2 %. Both bands are needed.
BANDS = ((-150.0, 150.0), (250.0, 450.0))
SEARCH_NS = 20_000.0     # wide enough to contain the drift correction at 80 ms
FAR_LO, FAR_HI = 5_000.0, 20_000.0   # sideband for the accidental-density estimate

TREES = tuple(f'{k}{a}' for k in ('WAL', 'PSS') for a in 'ABCD')


def nearest_residuals(ntof_run: int, events, bunches, k: float, t0: float,
                      trees=TREES, search_ns: float = SEARCH_NS):
    """
    Per DREAM event: the nearest-candidate residual in each tree, plus the local
    accidental density.

    The search window is centred on the PREDICTED match position
    (t + k*t + t0), not on the raw event time -- at 80 ms the drift term is 8.7 us,
    so a window centred on the raw time misses the match entirely.
    """
    eids = events['eventId'].to_numpy()
    ets = events['t_since_flash_ns'].to_numpy().astype(float)
    pos = {int(e): i for i, e in enumerate(eids)}
    best = np.full((len(eids), len(trees)), np.nan)
    dens = np.zeros(len(eids))

    for ti, tree in enumerate(trees):
        h = read_bunches(ntof_run, tree, bunches, branches=('BunchNumber',))
        o = np.lexsort((h['t_since_flash_ns'], h['BunchNumber']))
        cb, ct = h['BunchNumber'][o], h['t_since_flash_ns'][o]
        for b, g in events.groupby('BunchNumber'):
            s, e = np.searchsorted(cb, [b, b + 1])
            tt = ct[s:e]
            if tt.size == 0:
                continue
            et = g['t_since_flash_ns'].to_numpy().astype(float)
            ee = g['eventId'].to_numpy()
            pred = et + k * et + t0
            lo = np.searchsorted(tt, pred - search_ns)
            hi = np.searchsorted(tt, pred + search_ns)
            for j in range(et.size):
                if hi[j] <= lo[j]:
                    continue
                i = pos[int(ee[j])]
                r = tt[lo[j]:hi[j]] - pred[j]
                best[i, ti] = r[np.abs(r).argmin()]
                far = (np.abs(r) > FAR_LO) & (np.abs(r) < FAR_HI)
                dens[i] += far.sum() / (2 * (FAR_HI - FAR_LO))
    return eids, ets, best, dens


def in_bands(res, bands=BANDS):
    """Boolean mask: residual falls in any accept band (NaN -> False)."""
    out = np.zeros_like(res, dtype=bool)
    for lo, hi in bands:
        with np.errstate(invalid='ignore'):
            out |= (res >= lo) & (res <= hi)
    return out


def band_width(bands=BANDS) -> float:
    return float(sum(hi - lo for lo, hi in bands))


def calibrate(ets, best, dens, bands=BANDS, edges=(1, 3, 10, 20, 40, 80)):
    """
    Efficiency and accidental probability per time-since-flash bin.

    P_acc is the chance that a random n_TOF hit lands somewhere in the accept
    bands: 1 - exp(-density * total_band_width), with the density measured from
    each event's own 5-20 us sidebands. It is the false-match probability for an
    event that had no real partner, so it bounds the contamination.
    """
    hit = in_bands(best, bands).any(axis=1)
    w = band_width(bands)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ets >= lo * 1e6) & (ets < hi * 1e6)
        if m.sum() == 0:
            continue
        d = float(dens[m].mean())
        rows.append(dict(t_lo=lo, t_hi=hi, n=int(m.sum()),
                         dens_per_us=d * 1000, eff=float(hit[m].mean()),
                         p_acc=float(1 - np.exp(-d * w))))
    return rows


if __name__ == '__main__':
    from ntof_dream_merge.bunch_join import dream_event_to_bunch

    run = sys.argv[1] if len(sys.argv) > 1 else 'run_79'
    sub = sys.argv[2] if len(sys.argv) > 2 else 'stat090_0000'
    nt = int(sys.argv[3]) if len(sys.argv) > 3 else 224572
    nb = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    K, T0 = 1.089e-4, -197.5

    ev = dream_event_to_bunch(run, sub, nt)
    bunches = np.sort(ev.loc[ev['BunchNumber'] > 0, 'BunchNumber'].unique())[:nb]
    sel = ev[(ev['BunchNumber'].isin(bunches)) & (~ev['is_flash'])]
    eids, ets, best, dens = nearest_residuals(nt, sel, bunches, K, T0)

    print(f'{run}/{sub} <-> {nt}: {len(eids):,} events, {len(bunches)} bunches')
    print(f'accept bands {BANDS} -> {band_width():.0f} ns total\n')
    print('  t bin (ms)      n   n_TOF rate   efficiency   P(false match)')
    for r in calibrate(ets, best, dens):
        print(f'  {r["t_lo"]:4d}-{r["t_hi"]:<4d} {r["n"]:7d} '
              f'{r["dens_per_us"]:8.2f}/us {r["eff"]:11.1%} {r["p_acc"]:14.1%}')
    hit = in_bands(best).any(axis=1)
    print(f'\n  overall: {hit.mean():.1%} of events matched')
