#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_survey.py -- is every n_TOF detector on the same clock?

The matching study is about to tighten an accept window from +-150 ns towards a
few tens of ns, so every internal misalignment above a few nanoseconds stops
being bookkeeping and starts costing efficiency. This measures all four levels
of it on the CANDIDATE processing, in situ, with no calibration carried in from
another run:

  [1] ABSOLUTE, across detectors -- each tree's flash time against the beam
      pickup, tflash(tree) - tflash(PKUP), per bunch. This is the only estimator
      that compares detectors with no common particle, and it is what puts arm A
      and arm D on one time base. Cross-checked against the PKUP-referenced
      calibration in ntof_processing/flash_timing/, which was measured on the
      seven divert-off runs and transported here.
  [2] WALL vs PLASTIC, per arm and per wall channel -- prompt coincidences of
      late hits, the same estimator mx_july_beam_qa/calib/time_offsets_*.json
      uses. This is the alignment the SINGLES trigger emulation depends on: the
      wall .AND. plastic coincidence is only 20 ns wide.
  [3] WITHIN A WALL BAR, top vs bottom -- the analogue sum that gets
      discriminated is formed from two cables that need not be equal.
  [4] LIQUID vs WALL, per arm -- the liquids are the next subsystem to enter the
      merge, so their offset has to be known before their coincidences mean
      anything.

Plus the systematic that decides which time base the study should use at all:
how far the candidate file's OWN stored tflash sits from the laptop-side repair,
per tree and per bunch. If that is a few ns the two are interchangeable and the
file's own tflash is preferred (it is what the campaign will analyse); if it is
not, the repair has to stay in the chain.

USAGE
    python align_survey.py [--nb 300] [--json out.json]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from study_common import use_variant, DATA, NTOF_RUN, ROOT

use_variant()

from ntof_dream_merge import fast_singles as fs             # noqa: E402
from ntof_dream_merge import tflash_repair as rep           # noqa: E402
from ntof_dream_merge.ntof_io import bunch_edges          # noqa: E402

# fs.read_bunches honours fs.REPAIR_TFLASH, which is False: this whole survey is
# a measurement OF the candidate file's time base, so nothing may be repaired
# on the way in.
read_bunches = fs.read_bunches
assert fs.REPAIR_TFLASH is False

ARMS = ('A', 'B', 'C', 'D')
LATE_NS = 100_000.0      # out of the flash, where singles dominate
MAX_DT = 500.0           # search window for the coincidence estimator
CORE = 30.0              # core half-width for the sigma/median


def _peak(d: np.ndarray, rng=MAX_DT, bins=2000):
    """Modal position, core median and core sigma of a residual set."""
    d = d[np.isfinite(d) & (np.abs(d) < rng)]
    if d.size < 50:
        return dict(n=int(d.size), peak=np.nan, median=np.nan, sigma=np.nan)
    h, e = np.histogram(d, bins=bins, range=(-rng, rng))
    c = 0.5 * (e[1:] + e[:-1])
    pk = float(c[h.argmax()])
    core = d[np.abs(d - pk) < CORE]
    return dict(n=int(d.size), peak=pk,
                median=float(np.median(core)) if core.size else np.nan,
                sigma=float(core.std()) if core.size else np.nan)


def _dt_nearest(b_a, t_a, b_b, t_b):
    """t_b - t_a for the nearest b to each a, same bunch, within MAX_DT."""
    if t_a.size == 0 or t_b.size == 0:
        return np.array([])
    k_b = fs._pack(b_b, t_b)
    o = np.argsort(k_b, kind='stable')
    k_b, t_bs = k_b[o], np.asarray(t_b)[o]
    k = fs._nearest(fs._pack(b_a, t_a), k_b, MAX_DT)
    m = k >= 0
    return t_bs[k[m]] - np.asarray(t_a)[m]


def sample_bunches(nb: int) -> np.ndarray:
    """Bunches spread over the whole run, not a contiguous block at one end."""
    e = bunch_edges(NTOF_RUN, 'WALA')
    good = np.flatnonzero(np.diff(e) > 0) + 1
    step = max(1, len(good) // nb)
    return good[::step][:nb]


def flash_alignment(bunches) -> dict:
    """[1] tflash(tree) - tflash(PKUP), per bunch, from the cached tables."""
    tab = rep.tflash_tables(NTOF_RUN)
    pk = tab['PKUP']
    out = {}
    for tree in rep.ALL_TREES:
        if tree == 'PKUP':
            continue
        d = tab[tree] - pk
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        med = float(np.median(d))
        core = d[np.abs(d - med) < 100]
        out[tree] = dict(n=int(d.size), median=med,
                         std_core=float(core.std()) if core.size else np.nan,
                         frac_out_100ns=float(1 - core.size / d.size))
    return out


def calib_reference() -> dict:
    """The PKUP-referenced constants measured on the divert-off runs."""
    p = ROOT / 'ntof_processing' / 'flash_timing' / 'data' / \
        'flash_timing_calibration.json'
    if not p.exists():
        return {}
    j = json.loads(p.read_text())
    out = {}
    for tree, v in j.get('transport_monitor', {}).get('per_tree', {}).items():
        out[tree] = dict(C_ns=v['C_ns'], std_over_runs_ns=v.get('std_over_runs_ns'))
    con = j.get('constants', {})
    if isinstance(con, dict):
        for k, v in con.items():
            if isinstance(v, dict) and 'C_ns' in v:
                out.setdefault(k, {})['C_ns'] = v['C_ns']
    return out


def coincidence_alignment(bunches) -> dict:
    """[2]-[4]: wall-plastic, per wall channel, top/bottom, liquid-wall."""
    res = dict(station={}, channel={}, pss_bar={}, liq={}, topbottom={})
    for arm in ARMS:
        w = read_bunches(NTOF_RUN, f'WAL{arm}', bunches,
                         branches=('BunchNumber', 'detn'))
        p = read_bunches(NTOF_RUN, f'PSS{arm}', bunches,
                         branches=('BunchNumber', 'detn', 'amp'))
        q = read_bunches(NTOF_RUN, f'LIQ{arm}', bunches,
                         branches=('BunchNumber', 'amp', 'satuflag'))
        mw = w['t_since_flash_ns'] > LATE_NS
        mp = p['t_since_flash_ns'] > LATE_NS
        wb, wt, wd = (w['BunchNumber'][mw], w['t_since_flash_ns'][mw],
                      w['detn'][mw])
        pb, pt, pd = (p['BunchNumber'][mp], p['t_since_flash_ns'][mp],
                      p['detn'][mp])

        # [2] station and per-wall-channel: dt = t_pss - t_wall
        res['station'][arm] = _peak(_dt_nearest(wb, wt, pb, pt))
        for ch in range(1, 9):
            s = wd == ch
            if s.sum() < 100:
                continue
            res['channel'][f'{arm}{ch}'] = _peak(_dt_nearest(wb[s], wt[s], pb, pt))
        for bar in (1, 2):
            s = pd == bar
            if s.sum() < 100:
                continue
            res['pss_bar'][f'{arm}{bar}'] = _peak(
                _dt_nearest(wb, wt, pb[s], pt[s]))

        # [3] top vs bottom of the same bar
        res['topbottom'][arm] = {}
        for g in range(4):
            it, ib = wd == 2 * g + 1, wd == 2 * g + 2
            if it.sum() < 100 or ib.sum() < 100:
                continue
            res['topbottom'][arm][g] = _peak(
                _dt_nearest(wb[it], wt[it], wb[ib], wt[ib]), rng=200.0, bins=800)

        # [4] liquid vs wall. Saturated liquid hits are dropped: their amp is a
        # fit extrapolation, and inside the flash their time is unreliable too.
        import ntof_dream_merge.ntof_io as nio
        sat = nio.saturated(f'LIQ{arm}', q['amp'], q['satuflag'])
        mq = (q['t_since_flash_ns'] > LATE_NS) & ~sat
        res['liq'][arm] = _peak(_dt_nearest(wb, wt, q['BunchNumber'][mq],
                                            q['t_since_flash_ns'][mq]))
    return res


def repair_systematic() -> dict:
    """How far the file's own stored tflash is from the laptop repair."""
    tab = rep.tflash_tables(NTOF_RUN)
    fix = rep.corrected_tflash(NTOF_RUN)
    out = {}
    for tree in rep.ALL_TREES:
        d = tab[tree] - fix[tree]
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        out[tree] = dict(n=int(d.size), median=float(np.median(d)),
                         rms=float(d.std()),
                         p99_abs=float(np.percentile(np.abs(d), 99)),
                         frac_gt_25ns=float((np.abs(d) > 25).mean()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--nb', type=int, default=300)
    ap.add_argument('--json', default=str(DATA / 'alignment.json'))
    args = ap.parse_args()

    bunches = sample_bunches(args.nb)
    print(f'run {NTOF_RUN} (v12_liqpileup), {len(bunches)} bunches spread over '
          f'{bunches.min()}-{bunches.max()}')
    print("time base: the file's own stored tflash (repair OFF)\n")

    fl = flash_alignment(bunches)
    cal = calib_reference()
    print('[1] ABSOLUTE: flash time vs the beam pickup, tflash(tree) - tflash(PKUP)')
    print('    tree    median     sigma   >100 ns   calibration   diff')
    for tree, v in fl.items():
        c = cal.get(tree, {}).get('C_ns')
        cs = f'{c:11.1f}' if c is not None else '          -'
        ds = f'{v["median"] - c:+7.1f}' if c is not None else '      -'
        print(f'    {tree}  {v["median"]:9.1f} {v["std_core"]:8.1f} '
              f'{v["frac_out_100ns"]:8.2%} {cs} {ds}')
    print('    (the calibration column is C_ns of flash_timing, sign-flipped: it '
          'quotes\n     t_flash - tof_PKUP, this table quotes the same difference)')

    co = coincidence_alignment(bunches)
    print('\n[2] WALL vs PLASTIC, per arm (t_pss - t_wall, late hits)')
    print('    arm      peak    median     sigma        n')
    for a, v in co['station'].items():
        print(f'    {a}    {v["peak"]:+8.1f} {v["median"]:+9.2f} {v["sigma"]:9.2f} '
              f'{v["n"]:8,}')
    ch = np.array([v['median'] for v in co['channel'].values()])
    print(f'    per wall channel: RMS {ch.std():.2f} ns, '
          f'range {ch.min():+.2f} .. {ch.max():+.2f} ns over {ch.size} channels')
    pb = np.array([v['median'] for v in co['pss_bar'].values()])
    print(f'    per plastic bar : RMS {pb.std():.2f} ns, '
          f'range {pb.min():+.2f} .. {pb.max():+.2f} ns over {pb.size} bars')

    print('\n[3] WITHIN A WALL BAR, top - bottom (ns)')
    print('    arm    seg0    seg1    seg2    seg3')
    for a, v in co['topbottom'].items():
        print(f'    {a}  ' + '  '.join(
            f'{v[g]["peak"]:+6.1f}' if g in v else '     -' for g in range(4)))

    print('\n[4] LIQUID vs WALL, per arm (t_liq - t_wall, late hits)')
    print('    arm      peak    median     sigma        n')
    for a, v in co['liq'].items():
        print(f'    {a}    {v["peak"]:+8.1f} {v["median"]:+9.2f} {v["sigma"]:9.2f} '
              f'{v["n"]:8,}')

    rs = repair_systematic()
    print('\n[5] SYSTEMATIC: stored tflash - laptop-repaired tflash, per bunch')
    print('    tree    median      RMS    p99|d|   >25 ns')
    for tree, v in rs.items():
        print(f'    {tree}  {v["median"]:8.1f} {v["rms"]:8.1f} {v["p99_abs"]:9.1f} '
              f'{v["frac_gt_25ns"]:8.2%}')

    out = dict(run=NTOF_RUN, variant='v12_liqpileup', repair_tflash=False,
               n_bunches=int(len(bunches)), flash=fl, calibration=cal,
               coincidence=co, repair_systematic=rs)
    with open(args.json, 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print(f'\n-> {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
