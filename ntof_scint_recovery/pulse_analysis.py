#!/usr/bin/env python3
"""
pulse_analysis.py -- the pulse layer of the recovery: from the per-run pulse
histograms (pulses.py --hist, condor), how the found pulses' amplitude, low edge
and width come back after the flash, per detector and beam intensity.

    python -m ntof_scint_recovery.pulse_analysis [--hists DIR]

Production runs only (quick.PRODUCTION_FROM), live channels only: a channel is
dropped from a run when it found < LIVE_FRAC of its campaign-median pulse count
(whole families were unpowered in some runs).

Per (detector, class) on the pulses' log-time grid, pooled over the detector's
channels:

``rate``   pulses per bunch per us
``q05, q50, q90, q99``  amplitude quantiles, mV
``w50``    median width (samples above half height), ns
and the same for the 30-32 ms reference bin.

Recovery times, each "the first time after which the curve stays inside the
band out to 10 ms", the band set by the 1-10 ms plateau (median of the curve
there):

``t_amp90``  the upper amplitude scale back to >= 90 % of its plateau (and stays)
``t_amp97``  ... >= 97 %
             The scale is the 90 % quantile of the pulses ABOVE A FLOOR (``AMP_FLOOR``:
             clear of the threshold and of the low-amplitude population, whose share
             changes with the particle mix), interpolated within bins in log amplitude.
``t_edge``   q05 within +-2 mV of its plateau      -- the effective threshold is back
``t_width``  w50 within +-15 % of its plateau      -- pulses have their normal shape

Caveat, the one that matters: a quantile of the amplitude spectrum moves when
the particle mix changes, not only when the gain does.  These are upper bounds
on the time to full amplitude, to be checked against a fixed reference (see
ANALYSIS_PLAN.md).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from ntof_scint_recovery.common import out_dir
from ntof_scint_recovery.quick import DETS, PRODUCTION_FROM

LIVE_FRAC = 0.05
#: amplitude floor of the gain-scale quantile, mV, per family
AMP_FLOOR = {'WAL': 30.0, 'PSS': 50.0, 'LIQ': 30.0}
PLATEAU_US = (1e3, 1e4)
CLS = ('par', 'ded')


def load(hdir: Path):
    runs, HA, HW, NB = [], [], [], []
    meta = None
    for f in sorted(hdir.glob('pulsehist_*.npz')):
        r = int(f.stem.split('_')[1])
        if r < PRODUCTION_FROM:
            continue
        z = np.load(f)
        meta = z
        runs.append(r)
        HA.append(z['HA'])
        HW.append(z['HW'])
        NB.append((int(z['n_par']), int(z['n_ded'])))
    return (np.array(runs), np.array(HA), np.array(HW), np.array(NB),
            meta['t_edges_us'], meta['a_edges'], meta['w_edges'], [tuple(c) for c in meta['channels']])


def quantile(h, edges, q):
    """Quantile of each row of a histogram (log-bin centres), NaN if < 50 entries."""
    c = np.cumsum(h, axis=-1)
    tot = c[..., -1]
    ctr = np.sqrt(edges[:-1] * edges[1:]) if edges[0] > 0 else 0.5 * (edges[:-1] + edges[1:])
    idx = np.argmax(c >= q * np.maximum(tot, 1)[..., None], axis=-1)
    v = ctr[idx].astype(float)
    v[tot < 50] = np.nan
    return v


def cond_quantile(h, edges, floor, q, nmin=100):
    """q-quantile of the amplitudes above ``floor``, per row, interpolated linearly
    in log amplitude inside the bin where the cumulative crosses q."""
    lo = np.searchsorted(edges, floor)
    hh = h[..., lo:]
    e = np.log(edges[lo:])
    c = np.cumsum(hh, axis=-1)
    tot = c[..., -1]
    out = np.full(h.shape[:-1], np.nan)
    for idx in np.ndindex(h.shape[:-1]):
        if tot[idx] < nmin:
            continue
        cc = np.r_[0, c[idx]] / tot[idx]
        out[idx] = np.exp(np.interp(q, cc, e))
    return out


def t_within(tc, y, lo, hi, tmax=1e4):
    m = (tc < tmax) & np.isfinite(y)
    t, v = tc[m], y[m]
    if not len(t):
        return np.nan
    bad = np.nonzero((v < lo) | (v > hi))[0]
    if not len(bad):
        return float(t[0])
    return float(t[bad[-1] + 1]) if bad[-1] + 1 < len(t) else np.nan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--hists', type=Path, default=out_dir('pulsehist'))
    a = ap.parse_args(argv)
    runs, HA, HW, NB, te, ae, we, chans = load(a.hists)
    # live mask per (run, channel): pulses found against the channel's median over runs
    tot = HA.sum(axis=(1, 3, 4))                                    # run, channel
    med = np.median(tot, axis=0)
    live = tot > LIVE_FRAC * np.maximum(med, 1)
    tc = np.sqrt(te[:-1] * te[1:])
    nlog = 100                                                      # log bins; then [1e4,3e4], ref
    tcl, dt = tc[:nlog], np.diff(te)[:nlog]
    out = {'tc_us': tcl}
    rows = []
    for di, det in enumerate(DETS):
        cis = [i for i, c in enumerate(chans) if c[0] == di]
        for k, c in enumerate(CLS):
            ha = np.zeros(HA.shape[-2:])
            hw = np.zeros(HW.shape[-2:])
            nb = 0
            for ci in cis:
                m = live[:, ci]
                ha += HA[m, k, ci].sum(axis=0)
                hw += HW[m, k, ci].sum(axis=0)
                nb += NB[m, k].sum()
            if nb == 0:
                continue
            key = f'{det}_{c}'
            q = {f'q{int(p * 100):02d}': quantile(ha, ae, p) for p in (0.05, 0.5, 0.9, 0.99)}
            w50 = quantile(hw, we, 0.5)
            out[key + '_rate'] = ha[:nlog].sum(axis=1) / nb / dt
            for n, v in q.items():
                out[f'{key}_{n}'] = v[:nlog]
                out[f'{key}_{n}_ref'] = v[-1]
            out[key + '_w50'] = w50[:nlog]
            out[key + '_w50_ref'] = w50[-1]
            out[key + '_nb'] = nb
            out[key + '_ha'] = ha
            pl = (tcl > PLATEAU_US[0]) & (tcl < PLATEAU_US[1])
            g = cond_quantile(ha, ae, AMP_FLOOR[det[:3]], 0.9)
            gpl = np.nanmedian(g[:nlog][pl])
            out[key + '_gain'] = g[:nlog] / gpl
            out[key + '_gain_ref'] = g[-1] / gpl
            p05, pw = np.nanmedian(out[f'{key}_q05'][pl]), np.nanmedian(out[key + '_w50'][pl])
            rows.append(dict(det=det, cls=c, channel_bunches=int(nb),
                             scale_plateau_mV=gpl, scale_ref_over_plateau=float(g[-1] / gpl),
                             t_amp90_us=t_within(tcl, out[key + '_gain'], 0.90, 1e9),
                             t_amp97_us=t_within(tcl, out[key + '_gain'], 0.97, 1e9),
                             q05_plateau_mV=p05, w50_plateau_ns=pw,
                             t_edge_us=t_within(tcl, out[f'{key}_q05'], p05 - 2, p05 + 2),
                             t_width_us=t_within(tcl, out[key + '_w50'], 0.85 * pw, 1.15 * pw)))
    o = out_dir('pulses')
    np.savez_compressed(o / 'pulse_curves.npz', a_edges=ae, w_edges=we, t_edges_us=te, **out)
    with open(o / 'pulse_recovery.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{len(runs)} production runs, live channel-runs {live.mean():.1%} -> {o}')
    for r in rows:
        print(f"  {r['det']} {r['cls']}: scale {r['scale_plateau_mV']:5.0f} mV (ref/plateau {r['scale_ref_over_plateau']:.2f}) | "
              f"amp90 {r['t_amp90_us']:7.1f}  amp97 {r['t_amp97_us']:7.1f}  width {r['t_width_us']:6.1f} us")
    return 0


if __name__ == '__main__':
    sys.exit(main())
