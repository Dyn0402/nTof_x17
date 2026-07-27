#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dream_trigger.py -- rebuild the DREAM SINGLES trigger from the n_TOF hit data.

Matching DREAM events against every n_TOF hit is hopeless at early times: the
singles rate over the eight wall/plastic trees is ~11 hits/us at 1-3 ms, one hit
every 88 ns, so any window wide enough for the DREAM timing spread matches by
accident. But DREAM did not trigger on every hit -- it triggered on the N1081B
SINGLES chain, which is a small fraction of them. Reconstructing THAT is what
makes the match unambiguous.

The chain (mx_july_beam_qa/30_trigger_emulation.py has the same physics, run on
the mid-July runs; this module reads the per-sub-run n1081b_config.json so the
thresholds are the ones that were actually loaded, not a remembered constant):

  M1  per wall, per bar segment g=0..3: the 428F linear fan-in forms the ANALOG
      SUM of the two bar ends, amp(detn 2g+1) + amp(detn 2g+2) in mV, with the two
      ends required within TB_MAX. Discriminate the SUM -- not the individual
      ends -- at the wall threshold; OR the four segments.
  M2  per wall: each plastic BAR above its threshold; OR them. Which bars are in
      play is read from the sub-run's lemo_enables, not assumed -- for run_79 all
      four arms have both.
  M3  wall .AND. plastic within PULSE -> the sector SINGLES that triggered DREAM.

The wall channel layout was verified from the data rather than assumed: an 8x8
channel coincidence matrix gives strongest partners 1<->2, 3<->4, 5<->6, 7<->8 in
every wall, so the four top and four bottom channels are INTERLEAVED in detn
(1,3,5,7 top; 2,4,6,8 bottom) and the (2g+1, 2g+2) segment pairing is right.

THRESHOLDS ACTUALLY LOADED for run_79 (n1081b_config.json, polled 18:07:21 on
2026-07-26; board .240 = M1 wall OR, .241 = M2 plastic L1, sections A-D = arms):

    wall     A 25   B 35   C 34   D 36    mV, on the top+bottom segment SUM
    plastic  A 118  B 139  C 157  D 134   mV, on a single PMT

The wall values match the half-MIP set 30_trigger_emulation carries (25/35.3/33.5/
36.0). The plastic values do NOT: that script uses 65/78/86/83 mV, the 0.5-MIP set
from mid-July, whereas run_79's run_config says "plastic 0.90 MIP" and the board
readback confirms it at ~1.8x. Using the stored 0.5-MIP numbers here would let
through roughly twice the plastic rate the hardware did.

WHAT IT BUYS (100 bunches of run_79/stat090_0000 <-> 224572, accept bands
+-150 ns and +250..450 ns, efficiency = distinct DREAM events matched):

                          candidates/bunch   efficiency   P(false) 1-3ms / 40-80ms
    every hit, no cuts           ~3900          99.4 %        99.7 %  /   5.6 %
    M1 wall SINGLES              ~11600         88.8 %        28.9 %  /   0.1 %

so the real wall threshold cuts the early-time false-match rate by 3.4x for ~11 %
of the efficiency. Past 10 ms it is 0.1-4 % false at ~86-89 % efficient.

THE PLASTIC LEG DOES NOT WORK AS A REQUIREMENT. Demanding M2 as the hardware does
gives 12.7 % efficiency against the wall's 88.3 %, and the reason is not the
threshold: for late events, where matches are clean, only ~52 % of DREAM events
have ANY plastic hit in the band, above threshold or not, versus 96.5 % with a
wall segment sum over threshold. When a plastic hit is present it almost always
passes. Use the wall as the matcher and the plastic as a confirming tag.

Excluded so far (see mapping_and_deadtime.py): tree mismapping (the 4x4 wall x
plastic coincidence matrix is strongly diagonal), PSA dead time (the plastic PSA
resolves pulses 5-6 ns apart with no truncation), a too-high PSA amplitude cut
(real -- the plastic edge is 100 ADC against the walls' 50 -- but it sits at
~3.1 mV, 40x below the 118-157 mV discriminator), and spurious DREAM triggers
(fake_trigger_study bounds them at ~2 %). What is left is geometric: the two
20x30 cm bars cover only 30 cm of the wall's 50 cm in v, so ~60 % of wall-crossing
particles also cross a plastic, close to the measured ~52 %.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from common.beam_july_paths import RUNS_DIR                 # noqa: E402
from ntof_dream_merge.ntof_io import read_bunches           # noqa: E402

M1_IP, M2_IP = '192.168.10.240', '192.168.10.241'
ARMS = ('A', 'B', 'C', 'D')

# Which plastic bars fed the trigger, per arm. Do NOT hardcode this: the D-L /
# PSSD1 input was broken in mid-July, which is where 30_trigger_emulation's
# {'D': (2,)} comes from, but it was repaired before run_79 -- SEC_D reads back
# lemo 0 AND lemo 1 enabled, and in the data PSSD1 is the STRONGER partner of WALD
# (coincidence excess 615 vs 133). Read the enables from the sub-run's config.
# Separately: this only ever governed the TRIGGER. The digitiser records both bars
# regardless, so hit-level selection should always use both.
D_PMTS_FALLBACK = {a: (1, 2) for a in ARMS}

# Top/bottom match window for the analog sum. 30_trigger_emulation uses a bare
# +-15 ns, which is wrong for run224572: the two ends of a bar are NOT simultaneous
# here. Measured per (arm, segment) from late hits, the offsets are discrete --
# either ~0 or ~+-32..40 ns, a cabling difference, with sigma ~4 ns once removed:
#
#   wall A  +38.5  -31.5   +0.5  +34.5     wall C  +34.5  -31.5   +0.5  +39.5
#   wall B   -0.5  +38.5  -28.5   +1.5     wall D  +32.5   -1.5   +0.5  +32.5
#
# A bare +-15 ns window therefore keeps only 27.6 % of genuine top/bottom pairs and
# silently guts the wall trigger. Measure the offset, subtract it, then pair.
TB_MAX_NS = 25.0     # around the MEASURED per-segment offset, not around zero
PULSE_NS = 20.0      # discriminated logic-pulse width == coincidence window
TB_LATE_NS = 100_000.0   # sample late hits when measuring the offsets

ADC_MV_DEFAULT = _HERE.parent / 'mx_july_beam_qa' / 'calib' / 'adc_to_mv_run224524.json'


def load_thresholds(run: str, subrun: str) -> dict:
    """Wall and plastic discriminator thresholds as loaded for this sub-run."""
    p = RUNS_DIR / run / subrun / 'n1081b_config.json'
    if not p.exists():
        raise FileNotFoundError(
            f'{p} missing -- rsync it from the DAQ machine; it is per sub-run and '
            'is the only record of what the discriminators were actually set to.')
    cfg = json.loads(p.read_text())
    b = cfg['boards']
    wall = {a: abs(b[M1_IP]['sections'][f'SEC_{a}']['input_configuration']['data']
                   ['threshold']) for a in ARMS}
    plastic = {a: abs(b[M2_IP]['sections'][f'SEC_{a}']['input_configuration']['data']
                      ['threshold']) for a in ARMS}
    # lemo 0/1 of M2 section <arm> are that arm's two plastic bars -> detn 1/2
    pmts = {}
    for a in ARMS:
        fc = b[M2_IP]['sections'][f'SEC_{a}'].get('function_configuration', {})
        en = (fc.get('data', fc) or {}).get('lemo_enables')
        pmts[a] = (tuple(e['lemo'] + 1 for e in en if e['enable'] and e['lemo'] < 2)
                   if en else D_PMTS_FALLBACK[a])
    return dict(wall=wall, plastic=plastic, pmts=pmts, polled_at=cfg['polled_at'])


def load_adc_mv(path: Path = ADC_MV_DEFAULT) -> dict:
    f = json.loads(Path(path).read_text())['factors']
    return {tree: np.array([v[str(i)] for i in sorted(map(int, v))])
            for tree, v in f.items()}


def _pair_nearest(t_a, t_b, max_dt):
    """For each a, index of the nearest b within max_dt (-1 if none)."""
    if t_a.size == 0 or t_b.size == 0:
        return np.full(t_a.size, -1, np.int64)
    o = np.argsort(t_b)
    tb = t_b[o]
    j = np.searchsorted(tb, t_a)
    j0 = np.clip(j - 1, 0, tb.size - 1)
    j1 = np.clip(j, 0, tb.size - 1)
    d0, d1 = np.abs(tb[j0] - t_a), np.abs(tb[j1] - t_a)
    pick = np.where(d0 <= d1, j0, j1)
    ok = np.minimum(d0, d1) <= max_dt
    return np.where(ok, o[pick], -1)


def measure_tb_offsets(ntof_run: int, bunches, arm: str) -> dict:
    """Per-segment t_top - t_bottom, measured in situ from late hits."""
    w = read_bunches(ntof_run, f'WAL{arm}', bunches, branches=('BunchNumber', 'detn'))
    late = w['t_since_flash_ns'] > TB_LATE_NS
    tw, dw, bw = w['t_since_flash_ns'][late], w['detn'][late], w['BunchNumber'][late]
    out = {}
    for g in range(4):
        ds = []
        for b in np.unique(bw):
            m = bw == b
            a = np.sort(tw[m & (dw == 2 * g + 1)])
            c = np.sort(tw[m & (dw == 2 * g + 2)])
            if a.size == 0 or c.size == 0:
                continue
            j = np.searchsorted(c, a)
            j0, j1 = np.clip(j - 1, 0, c.size - 1), np.clip(j, 0, c.size - 1)
            d0, d1 = a - c[j0], a - c[j1]
            d = np.where(np.abs(d0) <= np.abs(d1), d0, d1)
            ds.append(d[np.abs(d) < 200])
        if not ds:
            out[g] = 0.0
            continue
        d = np.concatenate(ds)
        h, e = np.histogram(d, bins=400, range=(-200, 200))
        out[g] = float(0.5 * (e[1:] + e[:-1])[h.argmax()])
    return out


def singles_candidates(ntof_run: int, bunches, arm: str, thr: dict,
                       adc_mv: dict, tb_off: dict | None = None,
                       require_plastic: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    (BunchNumber, t_since_flash_ns) of every reconstructed sector-SINGLES.

    Time is the wall leg's, i.e. the mean of the two bar ends, which is what the
    428F sum presents to the discriminator.
    """
    w = read_bunches(ntof_run, f'WAL{arm}', bunches,
                     branches=('BunchNumber', 'detn', 'amp'))
    p = read_bunches(ntof_run, f'PSS{arm}', bunches,
                     branches=('BunchNumber', 'detn', 'amp'))
    wmv = w['amp'] * adc_mv[f'WAL{arm}'][(w['detn'] - 1).astype(int)]
    pmv = p['amp'] * adc_mv[f'PSS{arm}'][(p['detn'] - 1).astype(int)]
    wt, wb, wd = w['t_since_flash_ns'], w['BunchNumber'], w['detn']
    pt, pb, pd = p['t_since_flash_ns'], p['BunchNumber'], p['detn']

    psel = np.isin(pd, thr.get("pmts", D_PMTS_FALLBACK)[arm]) & (pmv > thr['plastic'][arm])
    pt, pb = pt[psel], pb[psel]
    if tb_off is None:
        tb_off = measure_tb_offsets(ntof_run, bunches, arm)

    cb, ct = [], []
    for b in np.unique(bunches):
        mw = wb == b
        if not mw.any():
            continue
        tw_b, mv_b, dn_b = wt[mw], wmv[mw], wd[mw]
        fire_t = []
        for g in range(4):
            it = np.flatnonzero(dn_b == 2 * g + 1)
            ib = np.flatnonzero(dn_b == 2 * g + 2)
            if it.size == 0 or ib.size == 0:
                continue
            # pair around the MEASURED offset, not around zero
            k = _pair_nearest(tw_b[it] - tb_off.get(g, 0.0), tw_b[ib], TB_MAX_NS)
            m = k >= 0
            if not m.any():
                continue
            i_t, i_b = it[m], ib[k[m]]
            s = mv_b[i_t] + mv_b[i_b]              # the ANALOG SUM is discriminated
            hit = s > thr['wall'][arm]
            if hit.any():
                fire_t.append(0.5 * (tw_b[i_t][hit] + tw_b[i_b][hit]))
        if not fire_t:
            continue
        wf = np.sort(np.concatenate(fire_t))
        if not require_plastic:
            cb.append(np.full(wf.size, b))
            ct.append(wf)
            continue
        pf = np.sort(pt[pb == b])
        if pf.size == 0:
            continue
        k = _pair_nearest(wf, pf, PULSE_NS)        # M3: wall .AND. plastic
        sel = k >= 0
        if sel.any():
            cb.append(np.full(int(sel.sum()), b))
            ct.append(wf[sel])
    if not cb:
        return np.array([], np.int64), np.array([])
    return np.concatenate(cb), np.concatenate(ct)


if __name__ == '__main__':
    from ntof_dream_merge.bunch_join import dream_event_to_bunch

    run = sys.argv[1] if len(sys.argv) > 1 else 'run_79'
    sub = sys.argv[2] if len(sys.argv) > 2 else 'stat090_0000'
    nt = int(sys.argv[3]) if len(sys.argv) > 3 else 224572
    nb = int(sys.argv[4]) if len(sys.argv) > 4 else 60

    thr = load_thresholds(run, sub)
    print(f'thresholds loaded for {run}/{sub} (polled {thr["polled_at"]}):')
    print('  wall    ' + '  '.join(f'{a} {thr["wall"][a]:5.0f} mV' for a in ARMS))
    print('  plastic ' + '  '.join(f'{a} {thr["plastic"][a]:5.0f} mV' for a in ARMS))

    ev = dream_event_to_bunch(run, sub, nt)
    bunches = np.sort(ev.loc[ev['BunchNumber'] > 0, 'BunchNumber'].unique())[:nb]
    adc_mv = load_adc_mv()
    sel = ev[(ev['BunchNumber'].isin(bunches)) & (~ev['is_flash'])]
    print(f'\n{len(bunches)} bunches, {len(sel):,} DREAM events '
          f'({len(sel)/len(bunches):.0f}/burst)')
    tot = 0
    for arm in ARMS:
        cb, ct = singles_candidates(nt, bunches, arm, thr, adc_mv)
        tot += ct.size
        print(f'  arm {arm}: {ct.size:7,} SINGLES ({ct.size/len(bunches):7.1f}/bunch)')
    print(f'  total  : {tot:7,} ({tot/len(bunches):.1f}/bunch) vs '
          f'{len(sel)/len(bunches):.0f} DREAM events/burst')
