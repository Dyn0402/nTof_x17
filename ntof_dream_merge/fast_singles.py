#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fast_singles.py -- the DREAM SINGLES trigger emulation, vectorised.

Same physics as `dream_trigger.singles_candidates` (read that module for what the
N1081B chain is and why the thresholds and the top/bottom offsets are what they
are); this is a rewrite whose cost does not grow with the number of bunches.

WHY. `dream_trigger` loops over bunches and inside each bunch masks the WHOLE
hit array (`wb == b`), so the work is O(N_hits x N_bunches). That is fine for the
60-100 bunch demonstrations it was written for and hopeless for the 2061-bunch
reference pair: the same shape of mistake that made `liq_coincidence` take
1 h 52 min before it was rewritten. Here bunch and time are packed into one
sorted float64 key, exactly as `liq_coincidence.window_residuals` does, so a
"nearest partner within +-dt, same bunch" query is two `searchsorted` calls over
the whole run at once. Cross-bunch matching is impossible by construction:
|t_since_flash| <= ~8e7 ns against a KEY_SCALE of 1e9, so consecutive bunches'
key blocks are 8.4e8 ns apart against coincidence windows of 20-25 ns.

Validated bit-identical against `dream_trigger.singles_candidates` -- same
candidate bunches and times, both legs -- by `match_study/scripts/validate_fast.py`.

It also returns more than the pair of arrays the original did: the wall analogue
SUM in mV, the segment that fired, and the plastic partner's amplitude and dt.
Those are what an amplitude-based purity cut would use, and they cost nothing to
carry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from ntof_dream_merge.ntof_io import read_bunches as _read_bunches  # noqa: E402
from ntof_dream_merge.dream_trigger import (ARMS, TB_MAX_NS, PULSE_NS,  # noqa: E402
                                            TB_LATE_NS, D_PMTS_FALLBACK,
                                            load_thresholds, load_adc_mv)

# Which time base every read in this module uses. `dream_trigger` inherits
# ntof_io's default (the laptop-side tflash repair ON), which is right for the
# OFFICIAL processing and wrong as a test of a reprocessed one: v12's own stored
# tflash is what the campaign will analyse, and the 07-29 headline (95.7 % / 0.5 %)
# was measured with the repair OFF. Set it explicitly rather than inherit it.
REPAIR_TFLASH = False


def read_bunches(run, tree, bunches, branches):
    return _read_bunches(run, tree, bunches, branches=branches,
                         repair_tflash=REPAIR_TFLASH)

# |t_since_flash| stays under ~8e7 ns (80 ms), so 1e9 keeps every bunch's keys
# disjoint with an order of magnitude to spare. Guarded, not assumed, in _pack.
KEY_SCALE = 1e9


def _pack(bunch, t):
    b = np.asarray(bunch, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if t.size and np.abs(t).max() >= 0.4 * KEY_SCALE:
        raise ValueError(f'time base reaches {np.abs(t).max():.3g} ns, too large '
                         f'for KEY_SCALE={KEY_SCALE:.3g}')
    return b * KEY_SCALE + t


def _nearest(key_a, key_b, max_dt):
    """For each a, index into the SORTED key_b of the nearest key within max_dt.

    -1 where there is none. Ties break towards the earlier partner, matching
    dream_trigger._pair_nearest.
    """
    if key_a.size == 0 or key_b.size == 0:
        return np.full(key_a.size, -1, np.int64)
    j = np.searchsorted(key_b, key_a)
    j0 = np.clip(j - 1, 0, key_b.size - 1)
    j1 = np.clip(j, 0, key_b.size - 1)
    d0, d1 = np.abs(key_b[j0] - key_a), np.abs(key_b[j1] - key_a)
    pick = np.where(d0 <= d1, j0, j1)
    return np.where(np.minimum(d0, d1) <= max_dt, pick, -1)


def measure_tb_offsets(ntof_run: int, bunches, arm: str,
                       late_ns: float = TB_LATE_NS) -> dict:
    """Per-segment modal (t_top - t_bottom), measured in situ from late hits.

    Vectorised twin of dream_trigger.measure_tb_offsets. These offsets are a
    cabling difference of either ~0 or ~+-32..40 ns; pairing the two bar ends in
    a bare +-15 ns window around zero keeps only ~28 % of genuine pairs.
    """
    w = read_bunches(ntof_run, f'WAL{arm}', bunches,
                     branches=('BunchNumber', 'detn'))
    late = w['t_since_flash_ns'] > late_ns
    tw, dw, bw = (w['t_since_flash_ns'][late], w['detn'][late],
                  w['BunchNumber'][late])
    out = {}
    for g in range(4):
        it = np.flatnonzero(dw == 2 * g + 1)
        ib = np.flatnonzero(dw == 2 * g + 2)
        if it.size == 0 or ib.size == 0:
            out[g] = 0.0
            continue
        kb = _pack(bw[ib], tw[ib])
        o = np.argsort(kb, kind='stable')
        kb, ib = kb[o], ib[o]
        ka = _pack(bw[it], tw[it])
        k = _nearest(ka, kb, 200.0)
        m = k >= 0
        if not m.any():
            out[g] = 0.0
            continue
        d = tw[it][m] - tw[ib[k[m]]]
        h, e = np.histogram(d, bins=400, range=(-200, 200))
        out[g] = float(0.5 * (e[1:] + e[:-1])[h.argmax()])
    return out


def singles_candidates(ntof_run: int, bunches, arm: str, thr: dict,
                       adc_mv: dict, tb_off: dict | None = None,
                       require_plastic: bool = True,
                       wall_thr_mv: float | None = None,
                       plastic_thr_mv: float | None = None) -> dict:
    """
    Every reconstructed sector SINGLES of one arm, as arrays.

    Returns a dict with `bunch`, `t` (the wall leg's time: the mean of the two
    bar ends, which is what the 428F analogue sum presents to the
    discriminator), `wall_mv` (the discriminated SUM), `seg`, and -- when
    require_plastic -- `pss_dt` and `pss_mv` for the plastic partner. Sorted by
    (bunch, t).

    wall_thr_mv / plastic_thr_mv override the hardware thresholds; that is for
    the threshold-scan study only, since the hardware values are what actually
    triggered DREAM.
    """
    wthr = thr['wall'][arm] if wall_thr_mv is None else wall_thr_mv
    pthr = thr['plastic'][arm] if plastic_thr_mv is None else plastic_thr_mv

    w = read_bunches(ntof_run, f'WAL{arm}', bunches,
                     branches=('BunchNumber', 'detn', 'amp'))
    p = read_bunches(ntof_run, f'PSS{arm}', bunches,
                     branches=('BunchNumber', 'detn', 'amp'))
    wmv = w['amp'] * adc_mv[f'WAL{arm}'][(w['detn'] - 1).astype(int)]
    pmv = p['amp'] * adc_mv[f'PSS{arm}'][(p['detn'] - 1).astype(int)]
    wt, wb, wd = w['t_since_flash_ns'], w['BunchNumber'], w['detn']
    pt, pb = p['t_since_flash_ns'], p['BunchNumber']

    psel = np.isin(p['detn'], thr.get('pmts', D_PMTS_FALLBACK)[arm]) & (pmv > pthr)
    pt, pb, pmv = pt[psel], pb[psel], pmv[psel]

    if tb_off is None:
        tb_off = measure_tb_offsets(ntof_run, bunches, arm)

    fb, ft, fs, fg = [], [], [], []
    for g in range(4):
        it = np.flatnonzero(wd == 2 * g + 1)
        ib = np.flatnonzero(wd == 2 * g + 2)
        if it.size == 0 or ib.size == 0:
            continue
        kb = _pack(wb[ib], wt[ib])
        o = np.argsort(kb, kind='stable')
        kb, ib = kb[o], ib[o]
        # pair around the MEASURED offset, not around zero
        ka = _pack(wb[it], wt[it] - tb_off.get(g, 0.0))
        k = _nearest(ka, kb, TB_MAX_NS)
        m = k >= 0
        if not m.any():
            continue
        i_t, i_b = it[m], ib[k[m]]
        s = wmv[i_t] + wmv[i_b]              # the ANALOG SUM is discriminated
        hit = s > wthr
        if not hit.any():
            continue
        fb.append(wb[i_t][hit])
        ft.append(0.5 * (wt[i_t][hit] + wt[i_b][hit]))
        fs.append(s[hit])
        fg.append(np.full(int(hit.sum()), g, np.int8))

    empty = dict(bunch=np.array([], np.int64), t=np.array([]),
                 wall_mv=np.array([]), seg=np.array([], np.int8),
                 pss_dt=np.array([]), pss_mv=np.array([]))
    if not fb:
        return empty
    cb, ct = np.concatenate(fb).astype(np.int64), np.concatenate(ft)
    cs, cg = np.concatenate(fs), np.concatenate(fg)
    o = np.lexsort((ct, cb))
    cb, ct, cs, cg = cb[o], ct[o], cs[o], cg[o]

    if not require_plastic:
        return dict(bunch=cb, t=ct, wall_mv=cs, seg=cg,
                    pss_dt=np.full(ct.size, np.nan),
                    pss_mv=np.full(ct.size, np.nan))

    kp = _pack(pb, pt)
    o = np.argsort(kp, kind='stable')
    kp, pt_s, pmv_s = kp[o], pt[o], pmv[o]
    k = _nearest(_pack(cb, ct), kp, PULSE_NS)     # M3: wall .AND. plastic
    sel = k >= 0
    if not sel.any():
        return empty
    return dict(bunch=cb[sel], t=ct[sel], wall_mv=cs[sel], seg=cg[sel],
                pss_dt=pt_s[k[sel]] - ct[sel], pss_mv=pmv_s[k[sel]])


def all_arms(ntof_run: int, bunches, thr, adc_mv, offsets=None,
             require_plastic: bool = True, **kw) -> dict:
    """singles_candidates over A-D, concatenated and sorted, with an `arm` index."""
    parts = []
    for ai, arm in enumerate(ARMS):
        off = None if offsets is None else offsets.get(arm)
        d = singles_candidates(ntof_run, bunches, arm, thr, adc_mv, tb_off=off,
                               require_plastic=require_plastic, **kw)
        d['arm'] = np.full(d['t'].size, ai, np.int8)
        parts.append(d)
    keys = ('bunch', 't', 'wall_mv', 'seg', 'pss_dt', 'pss_mv', 'arm')
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    o = np.lexsort((out['t'], out['bunch']))
    return {k: v[o] for k, v in out.items()}
