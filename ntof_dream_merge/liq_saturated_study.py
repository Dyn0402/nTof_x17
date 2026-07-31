#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
liq_saturated_study.py -- the liquid hits we cut: how many, and what survives?

`liq_coincidence.py` drops saturated liquid hits (`satuflag` set, or
`amp` > 63 800 -- see FINDINGS_2026-07-29_signed_decoding.md). That is right for
anything amplitude-based, because a flagged hit's `amp` is a fit extrapolation
through the excluded samples. But a clipped physics-time liquid pulse is only
2-5 samples wide at the rail, so its ARRIVAL TIME may be untouched. This asks:

  1. how many saturated hits are there, at physics time, per tree;
  2. is their timing usable? -- the same DREAM coincidence test
     `liq_coincidence.py` runs, restricted to saturated hits. If they peak at
     the same -5..-25 ns residual with an excess over the shifted control, then
     clipping costs amplitude, not time, and they are recoverable as time hits;
  3. what amplitude information is left -- `amp` is only a lower bound of
     ~63 800, but the clip WIDTH (fwhm/fwtm) orders them, so the relation is
     reported for whoever wants to calibrate it against raw waveforms later.

Note `area` carries nothing extra: with AMPLITUDE OPTION=2 the PSA takes both
`amp` and `area` from the fitted template, so area = amp x integral(shape) and
`area/amp` is one constant per pulse shape. The MEASURED pair is `amp_0`/`area_0`
(PSA guide, "Finding the amplitude and area"). Use those to recover anything.

Usage:
    python liq_saturated_study.py <parts-dir-or-file> <match.npz> [--coinc 100]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

ARMS = ('A', 'B', 'C', 'D')
SHIFT_NS = 100_000.0   # ceilings come from ntof_io.saturation_ceiling()
FLASH_END_NS = 1e6


AMP_MATCH = 20_000.0       # clean control must be amplitude-matched: near-threshold
                           # pulses have fwhm ~2 ns against ~6 ns for large ones,
                           # so an all-amplitude control fakes a width difference.


def _peak_and_rate(res, win, coinc, n_ev):
    """(rate per event in +-coinc of the peak, peak position [ns], n in peak)."""
    if res.size == 0 or n_ev == 0:
        return 0.0, np.nan, 0
    edges = np.arange(-win, win + 10.0, 10.0)
    h = np.histogram(res, bins=edges)[0]
    c = (edges[:-1] + edges[1:]) / 2
    pk = c[int(np.argmax(h))]
    n = int((np.abs(res - pk) < coinc).sum())
    return n / n_ev, float(pk), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('target')
    ap.add_argument('npz')
    ap.add_argument('--ntof-run', type=int, default=224572)
    ap.add_argument('--win', type=float, default=2000.0)
    ap.add_argument('--coinc', type=float, default=100.0)
    args = ap.parse_args()

    import ntof_dream_merge.ntof_io as ntof_io
    import ntof_dream_merge.tflash_repair as rep

    p = Path(args.target).resolve()
    files = (sorted(p.glob(f'run{args.ntof_run}_[0-9]*.root'),
                    key=lambda q: int(q.stem.split('_')[-1]))
             if p.is_dir() else [p])
    ntof_io.ntof_paths = lambda r: files          # type: ignore
    ntof_io.ntof_path = lambda r: files[0]        # type: ignore
    # Persistent, per-variant and fingerprinted: same isolation as the mkdtemp
    # this replaces, but the bunch index survives between runs (~7 s/tree to
    # rebuild over 16 partials, so ~30 s for the four LIQ trees).
    rep.CACHE_DIR = ntof_io.CACHE_DIR = ntof_io.variant_cache(p, files)
    ntof_io._TFLASH_FIX_CACHE.clear()

    d = np.load(args.npz)
    ok = (d['arm'] >= 0) & np.isfinite(d['t_ntof_ns'])
    bunch, tw, arm = d['bunch'][ok], d['t_ntof_ns'][ok], d['arm'][ok]
    bunches = np.unique(bunch)
    print(f'{ok.sum():,} exclusively-matched DREAM events, '
          f'bunches {bunches.min()}-{bunches.max()}')

    print('\n1. the saturated population (in these bunches)')
    print(f'{"tree":5} {"hits":>10} {"flagged":>9} {"over ceil":>10} '
          f'{"saturated":>10} {"frac":>9} | {"at physics time":>16} {"frac of phys":>13}')
    store = {}
    for liq in ARMS:
        t = ntof_io.read_bunches(args.ntof_run, f'LIQ{liq}', bunches,
                                 branches=('BunchNumber', 'amp', 'satuflag',
                                           'fwhm', 'fwtm', 'tof'),
                                 repair_tflash=False)
        sf = t['satuflag'].astype(bool)
        over = t['amp'] > ntof_io.saturation_ceiling(f'LIQ{liq}')
        sat = ntof_io.saturated(f'LIQ{liq}', t['amp'], sf)
        phys = t['tof'] > FLASH_END_NS
        n = sf.size
        print(f'LIQ{liq} {n:10,} {int(sf.sum()):9,} {int(over.sum()):10,} '
              f'{int(sat.sum()):10,} {sat.mean()*100:8.4f}% | '
              f'{int((sat & phys).sum()):16,} {(sat & phys).sum()/max(phys.sum(),1)*100:12.4f}%')
        store[liq] = (t, sf, over, sat, phys)

    print('\n2. is the TIMING of saturated hits usable?')
    print('   same-arm residual t_LIQ - t_wall, saturated hits only, vs the')
    print('   +100 us shifted control. clean = the cut population for reference.')
    print(f'{"arm/liq":9} {"n_ev":>7} | {"sat: n_pk":>9} {"rate":>8} {"ctl":>8} {"peak":>6} '
          f'| {"clean: n_pk":>11} {"rate":>8} {"ctl":>8} {"peak":>6}')
    for a_i, a in enumerate(ARMS):
        m_ev = arm == a_i
        n_ev = int(m_ev.sum())
        if n_ev == 0:
            continue
        t, sf, over, sat, phys = store[a]
        for lab, keep in (('sat', sat), ('clean', ~sat)):
            lt = t['t_since_flash_ns'][keep]
            lb = t['BunchNumber'][keep]
            o = np.lexsort((lt, lb))
            lt, lb = lt[o], lb[o]
            res = {'sig': [], 'ctl': []}
            for tag, shift in (('sig', 0.0), ('ctl', SHIFT_NS)):
                for b in np.unique(bunch[m_ev]):
                    s, e = np.searchsorted(lb, [b, b + 1])
                    tt = lt[s:e]
                    if tt.size == 0:
                        continue
                    for t0 in tw[m_ev][bunch[m_ev] == b] + shift:
                        res[tag].append(tt[np.searchsorted(tt, t0 - args.win):
                                           np.searchsorted(tt, t0 + args.win)] - t0)
                res[tag] = (np.concatenate(res[tag]) if res[tag]
                            else np.array([]))
            r_s, pk, n_pk = _peak_and_rate(res['sig'], args.win, args.coinc, n_ev)
            r_c, _, _ = _peak_and_rate(res['ctl'], args.win, args.coinc, n_ev)
            store.setdefault('res', {})[(a, lab)] = (r_s, r_c, pk, n_pk)
        (rs, rc, pk, n1), (rs2, rc2, pk2, n2) = (store['res'][(a, 'sat')],
                                                 store['res'][(a, 'clean')])
        print(f'{a}/LIQ{a:6} {n_ev:7,} | {n1:9,} {rs:8.5f} {rc:8.5f} {pk:6.0f} '
              f'| {n2:11,} {rs2:8.5f} {rc2:8.5f} {pk2:6.0f}')
    print('  n_pk is the raw hit count in the peak window: a zero rate with n_pk = 0')
    print('  is a statistics limit, not evidence that saturated hits mistime.')

    print('\n3. what amplitude information is left on a saturated hit')
    print('   amp is a fit extrapolation -- treat it as a lower bound of 63 800.')
    print('   clip width orders them; calibrating it needs raw waveforms.')
    print(f'   clean control is amplitude-matched (amp > {AMP_MATCH:,.0f}).')
    print(f'{"tree":5} {"n_sat":>7} | {"amp p50":>12} {"amp p90":>12} | '
          f'{"fwhm p50":>9} {"fwtm p50":>9} | {"clean fwhm":>10} {"clean fwtm":>10} '
          f'{"n_clean":>9}')
    for liq in ARMS:
        t, sf, over, sat, phys = store[liq]
        if sat.sum() == 0:
            continue
        ctl = ~sat & (t['amp'] > AMP_MATCH)
        f = lambda k, m: np.percentile(t[k][m], 50) if m.sum() else np.nan
        print(f'LIQ{liq} {int(sat.sum()):7,} | '
              f'{np.percentile(t["amp"][sat], 50):12.0f} '
              f'{np.percentile(t["amp"][sat], 90):12.0f} | '
              f'{f("fwhm", sat):9.1f} {f("fwtm", sat):9.1f} | '
              f'{f("fwhm", ctl):10.1f} {f("fwtm", ctl):10.1f} {int(ctl.sum()):9,}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
