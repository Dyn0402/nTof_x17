#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-event BEAM PULSE INTENSITY for the run_67 scan, as a 4th/5th slice axis.

Every DREAM event is matched to its PS pulse by `ntof_july_analysis/pulse_match.py`
(cluster the sub-run's trigger times at a 0.5 s gap -> one cluster per pulse, fit
the clock offset against the beam_watcher per-pulse log, every event inherits its
cluster's intensity). run_67 is flash-anchored with one gamma flash per PS pulse,
so a pulse_match cluster IS a burst IS a beam pulse.

The July pulses are strongly BIMODAL — ~410e10 and ~850e10 — so the split is not a
free parameter: `E10_SPLIT = 600e10` cleanly separates them, the same constant and
convention used by `ntof_july_analysis/track_rate_hv_time_intensity/
intensity_split.py`. Measured on m090On_dr500_r530_056: 406-415e10 and 852-864e10
with nothing in between, match_frac 1.000.

*** THE TWO BANDS ARE NOT EQUALLY POPULATED. *** Run-wide (all 65 sub-runs,
rebuilt maps): LOW 20.5 %, HIGH 79.4 %, unmatched 0.03 %. So the LOW band costs
roughly 4x in statistics, and the per-sub-run fraction varies a lot — one
m090 sub-run sits at 12 % LOW. Hence `slide_plots.py` defaults the intensity
build to 2x the pooled boxcar width; read `n` before believing a LOW-vs-HIGH
difference in any single cell.

Public API:
    attach(ev)  -> ev + columns `e10` (float, NaN if unmatched) and
                   `iband` ('low' / 'high' / '' when unmatched)
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_JULY = os.path.dirname(_HERE)
sys.path.insert(0, os.path.dirname(_JULY))
sys.path.insert(0, _JULY)
sys.path.insert(0, _HERE)

import pulse_match as PM  # noqa: E402

# same constant as track_rate_hv_time_intensity/intensity_split.py — keep them
# equal, the two analyses are meant to be comparable
E10_SPLIT = 600.0
BANDS = ['low', 'high']
BAND_COLOR = {'low': '#0072B2', 'high': '#D55E00'}
BAND_LABEL = {'low': f'LOW  (<{E10_SPLIT:.0f}e10, ~410)',
              'high': f'HIGH (≥{E10_SPLIT:.0f}e10, ~850)'}

# A sub-run whose clusters mostly failed to match a logged pulse has an
# unreliable clock fit; its intensities are not trustworthy enough to slice on.
MIN_MATCH_FRAC = 0.80


def per_subrun_e10(subrun, run='run_67', verbose=True):
    """{eventId: intensity_e10} for one sub-run, or None if unusable."""
    try:
        r = PM.match_subrun(run, subrun)
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f'    {subrun}: pulse match FAILED ({e!r}) — no intensity')
        return None
    if r is None or r.get('match_frac', 0.0) < MIN_MATCH_FRAC:
        if verbose:
            frac = 0.0 if r is None else r.get('match_frac', 0.0)
            print(f'    {subrun}: match_frac {frac:.2f} < {MIN_MATCH_FRAC} '
                  f'— clock fit unreliable, intensity dropped')
        return None
    return r['event_e10']


def attach(ev, run='run_67', verbose=True):
    """Add `e10` and `iband` to an events table (keyed on subrun+eventId).

    Unmatched events get e10=NaN and iband='' — they are NOT silently assigned
    to a band; downstream code must filter on iband explicitly so an intensity
    split never quietly inherits the unmatched population.
    """
    ev = ev.copy()
    e10 = np.full(len(ev), np.nan)
    subs = ev['subrun'].to_numpy()
    eids = ev['eventId'].to_numpy()
    ok_subs, bad_subs = 0, []
    for sub in pd.unique(subs):
        m = subs == sub
        mp = per_subrun_e10(sub, run=run, verbose=verbose)
        if mp is None:
            bad_subs.append(sub)
            continue
        ok_subs += 1
        e10[m] = [mp.get(int(i), np.nan) for i in eids[m]]
    ev['e10'] = e10
    band = np.full(len(ev), '', dtype=object)
    good = np.isfinite(e10)
    band[good & (e10 < E10_SPLIT)] = 'low'
    band[good & (e10 >= E10_SPLIT)] = 'high'
    ev['iband'] = band
    if verbose:
        n = len(ev)
        print(f'  beam intensity: {ok_subs} sub-run(s) matched'
              + (f', {len(bad_subs)} dropped' if bad_subs else ''))
        print(f'    events: low={np.sum(band == "low")} '
              f'({100 * np.mean(band == "low"):.1f} %), '
              f'high={np.sum(band == "high")} '
              f'({100 * np.mean(band == "high"):.1f} %), '
              f'unmatched={np.sum(band == "")} '
              f'({100 * np.mean(band == ""):.1f} %) of {n}')
    return ev
