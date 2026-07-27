#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fake_trigger_study.py -- are the unmatched DREAM events misfired triggers?

HYPOTHESIS. DREAM fires some triggers spuriously. Those events would carry no
correlation whatsoever with n_TOF: whatever hits sit near them are ordinary
background, indistinguishable in rate, in amplitude, and in time from a random
instant in the same bunch. Real triggers instead have a genuine scintillator
partner. If true, the DREAM event sample is a mixture of the two.

METHOD -- a random-time control. For every DREAM event at t inside bunch b, make a
CONTROL pseudo-event in the SAME bunch at t + CONTROL_SHIFT_NS. The shift is large
enough to decorrelate completely (the real correlation dies out by 500 ns; there is
zero excess out to 20 us) but small compared with the timescale on which the n_TOF
singles rate changes, so the control samples the same local rate environment as its
event. The control therefore measures exactly what a spurious trigger would look
like, using the same data, the same bunches, and the same estimator.

Everything is then a comparison of real vs control:
  * P(match) -- if a class of DREAM events matches at the control rate, that class
    is consistent with carrying no information at all.
  * the 2x2 of wall-match x plastic-match -- a mixture of "all real" and "all fake"
    makes a specific prediction for the off-diagonal terms.
  * the amplitude spectrum of the matched hits -- a spurious trigger picks up
    background pulses, so its matched-hit spectrum is the ordinary singles
    spectrum; a real trigger picks up the pulse that fired it, which is
    trigger-level. Subtracting the control spectrum from the real one leaves the
    genuine partners.

Run late (t > 20 ms) by default, where the accidental floor is small enough that
the classes separate; the same machinery works at any time slice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from ntof_dream_merge.ntof_io import read_bunches                       # noqa: E402
from ntof_dream_merge.dream_trigger import (load_thresholds, load_adc_mv,  # noqa: E402
                                            measure_tb_offsets, singles_candidates,
                                            D_PMTS, ARMS)

K, T0 = 1.089e-4, -197.5
BANDS = ((-150.0, 150.0), (250.0, 450.0))
CONTROL_SHIFT_NS = 100_000.0     # 100 us: decorrelated, same local rate
SEARCH_NS = 1_000.0


def _inband(r):
    m = np.zeros_like(r, dtype=bool)
    for lo, hi in BANDS:
        m |= (r >= lo) & (r <= hi)
    return m


def _match(cb, ct, ev, shift=0.0, amps=None):
    """
    Per event: does a candidate fall in the accept bands, and the biggest matched
    amplitude if so. `shift` displaces the predicted time to build the control.
    """
    eids = ev['eventId'].to_numpy()
    pos = {int(e): i for i, e in enumerate(eids)}
    hit = np.zeros(len(eids), bool)
    amp = np.full(len(eids), np.nan)
    order = np.lexsort((ct, cb))
    cb, ct = cb[order], ct[order]
    ca = amps[order] if amps is not None else None
    for b, g in ev.groupby('BunchNumber'):
        s, e = np.searchsorted(cb, [b, b + 1])
        tt = ct[s:e]
        if tt.size == 0:
            continue
        aa = ca[s:e] if ca is not None else None
        et = g['t_since_flash_ns'].to_numpy().astype(float)
        ee = g['eventId'].to_numpy()
        pred = et + K * et + T0 + shift
        lo = np.searchsorted(tt, pred - SEARCH_NS)
        hi = np.searchsorted(tt, pred + SEARCH_NS)
        for j in range(et.size):
            if hi[j] <= lo[j]:
                continue
            m = _inband(tt[lo[j]:hi[j]] - pred[j])
            if m.any():
                i = pos[int(ee[j])]
                hit[i] = True
                if aa is not None:
                    amp[i] = np.nanmax([amp[i], aa[lo[j]:hi[j]][m].max()])
    return hit, amp


def study(ntof_run: int, ev, bunches, thr, adc):
    """Real-vs-control match rates and matched amplitudes, wall and plastic."""
    res = {}
    wall_hit = np.zeros(len(ev), bool); wall_ctl = np.zeros(len(ev), bool)
    pls_hit = np.zeros(len(ev), bool); pls_ctl = np.zeros(len(ev), bool)
    pls_amp = np.full(len(ev), np.nan); pls_amp_ctl = np.full(len(ev), np.nan)

    for arm in ARMS:
        off = measure_tb_offsets(ntof_run, bunches, arm)
        cb, ct = singles_candidates(ntof_run, bunches, arm, thr, adc,
                                    tb_off=off, require_plastic=False)
        if ct.size:
            h, _ = _match(cb, ct, ev)
            c, _ = _match(cb, ct, ev, shift=CONTROL_SHIFT_NS)
            wall_hit |= h; wall_ctl |= c

        p = read_bunches(ntof_run, f'PSS{arm}', bunches,
                         branches=('BunchNumber', 'detn', 'amp'))
        sel = np.isin(p['detn'], D_PMTS[arm])
        pb, pt = p['BunchNumber'][sel], p['t_since_flash_ns'][sel]
        pa = p['amp'][sel]
        h, a = _match(pb, pt, ev, amps=pa)
        c, ac = _match(pb, pt, ev, shift=CONTROL_SHIFT_NS, amps=pa)
        pls_hit |= h; pls_ctl |= c
        pls_amp = np.fmax(pls_amp, a); pls_amp_ctl = np.fmax(pls_amp_ctl, ac)

    res.update(wall=wall_hit, wall_ctl=wall_ctl, plastic=pls_hit,
               plastic_ctl=pls_ctl, plastic_amp=pls_amp,
               plastic_amp_ctl=pls_amp_ctl, n=len(ev))
    return res


def report(r, thr):
    n = r['n']
    print(f'{n:,} DREAM events\n')
    print('                        real     control     excess')
    for lab, a, b in (('wall SINGLES match ', r['wall'], r['wall_ctl']),
                      ('plastic hit match  ', r['plastic'], r['plastic_ctl'])):
        print(f'  {lab} {a.mean():7.1%}   {b.mean():7.1%}   {a.mean()-b.mean():+7.1%}')

    print('\n  2x2, wall x plastic (real / control):')
    for w in (True, False):
        row = []
        for p in (True, False):
            rr = ((r['wall'] == w) & (r['plastic'] == p)).mean()
            cc = ((r['wall_ctl'] == w) & (r['plastic_ctl'] == p)).mean()
            row.append(f'{rr:6.1%}/{cc:6.1%}')
        print(f'    wall {"Y" if w else "N"}:  plastic Y {row[0]}    plastic N {row[1]}')

    # The decisive one: among events with NO wall match, is the plastic rate the
    # control rate? If yes those events carry no n_TOF information at all.
    nw = ~r['wall']
    nwc = ~r['wall_ctl']
    if nw.sum():
        print(f'\n  among the {nw.sum():,} events with NO wall match:')
        print(f'    plastic match rate {r["plastic"][nw].mean():.1%}   '
              f'(control, no-wall subset: {r["plastic_ctl"][nwc].mean():.1%})')
    yw = r['wall']
    if yw.sum():
        print(f'  among the {yw.sum():,} events WITH a wall match:')
        print(f'    plastic match rate {r["plastic"][yw].mean():.1%}')

    # Fake fraction. A spurious trigger behaves like the control, so within the
    # no-wall class the plastic rate is a two-component mixture: real events match
    # at the rate the wall-matched ones do, fakes at the control rate. Solving for
    # the fake weight bounds how much of the sample can be spurious.
    if nw.sum() and yw.sum():
        p_real, p_ctl = r['plastic'][yw].mean(), r['plastic_ctl'][nwc].mean()
        p_obs = r['plastic'][nw].mean()
        if p_real > p_ctl:
            phi = float(np.clip((p_real - p_obs) / (p_real - p_ctl), 0, 1))
            print(f'\n  fake-trigger fraction: {phi:.1%} of the no-wall class '
                  f'= {phi*nw.mean():.1%} of all events')
            print(f'    (hard upper bound from the wall rate alone: '
                  f'{1 - r["wall"].mean():.1%})')

    a = r['plastic_amp'][np.isfinite(r['plastic_amp'])]
    c = r['plastic_amp_ctl'][np.isfinite(r['plastic_amp_ctl'])]
    print('\n  matched plastic amplitude (ADC), real vs control:')
    for q in (10, 50, 90, 99):
        print(f'    p{q:<3d}  real {np.percentile(a, q):9.1f}   '
              f'control {np.percentile(c, q):9.1f}')
    trig = np.mean([thr['plastic'][k] for k in ARMS]) / 0.0307
    print(f'    fraction above trigger level (~{trig:.0f} ADC): '
          f'real {(a > trig).mean():.1%}   control {(c > trig).mean():.1%}')


if __name__ == '__main__':
    from ntof_dream_merge.bunch_join import dream_event_to_bunch

    run = sys.argv[1] if len(sys.argv) > 1 else 'run_79'
    sub = sys.argv[2] if len(sys.argv) > 2 else 'stat090_0000'
    nt = int(sys.argv[3]) if len(sys.argv) > 3 else 224572
    nb = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    tmin = float(sys.argv[5]) if len(sys.argv) > 5 else 20e6

    ev_all = dream_event_to_bunch(run, sub, nt)
    bunches = np.sort(ev_all.loc[ev_all['BunchNumber'] > 0, 'BunchNumber'].unique())[:nb]
    ev = ev_all[(ev_all['BunchNumber'].isin(bunches)) & (~ev_all['is_flash'])
                & (ev_all['t_since_flash_ns'] > tmin)].reset_index(drop=True)
    thr, adc = load_thresholds(run, sub), load_adc_mv()
    print(f'{run}/{sub} <-> {nt}, {len(bunches)} bunches, t > {tmin/1e6:.0f} ms, '
          f'control shift {CONTROL_SHIFT_NS/1e3:.0f} us\n')
    report(study(nt, ev, bunches, thr, adc), thr)
