"""
09_late_inclusive.py — Two products from one pass, both with the apples-to-apples
selection (post-recovery bunches, tof - tflash > 0.1 ms):

  1. Inclusive per-channel amplitude spectra for all WAL/PSS channels (same
     binning as 01/07) -> fair comparison against the coincidence spectra.
  2. Wall-only top-bottom coincidence amplitude spectra per 4-bar group
     (signal window centered on each group's dt peak from the 06 cache, with
     sidebands) -> tests whether A/D walls see MIPs WITHOUT involving the
     plastic: a MIP peak here but not in wall-plastic coincidences indicts the
     plastic (gain/threshold/acceptance), not the SiPMs.

Usage: python 09_late_inclusive.py [run_file]
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import uproot

BASE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('coinc', BASE / '02_coincidence_scan.py')
coinc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coinc)

RUN_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'x17/beam_july/data/run224404.root'
CACHE = BASE / 'cache'

LATE_TOF = 1e5
AMP_EDGES = np.geomspace(40, 8e4, 301)
W_SIG = 8.0
SB_LO, SB_HI = 20.0, 80.0
SB_SCALE = (2 * W_SIG) / (2 * (SB_HI - SB_LO))   # sidebands on both sides
DT_MAX = SB_HI + 20


def group_dt_peaks(run_stem):
    """Top-bottom dt peak per (arm, group) from the 06 cache."""
    d = np.load(CACHE / f'06_wallgeom_{run_stem}.npz')
    cen = 0.5 * (d['dt_edges'][:-1] + d['dt_edges'][1:])
    side = np.abs(cen) > 60
    mu = {}
    for st in 'ABCD':
        h = d[f'intra_{st}']
        for g in range(4):
            hh = h[2 * g, 2 * g + 1]
            sub = np.clip(hh - hh[side].mean(), 0, None)
            win = np.abs(cen - cen[np.argmax(sub)]) <= 10
            mu[(st, g)] = float(np.average(cen[win], weights=sub[win]))
    return mu


def pair_idx(t_ref, t_other):
    lo = np.searchsorted(t_other, t_ref - DT_MAX)
    hi = np.searchsorted(t_other, t_ref + DT_MAX)
    counts = hi - lo
    tot = int(counts.sum())
    if tot == 0:
        return (np.empty(0, np.int64),) * 2
    ref_idx = np.repeat(np.arange(len(t_ref)), counts)
    within = np.arange(tot) - np.repeat(np.cumsum(counts) - counts, counts)
    return ref_idx, np.repeat(lo, counts) + within


def main():
    f = uproot.open(RUN_FILE)
    idx = f['index'].arrays(['BunchNumber', 'PulseIntensity'], library='np')
    intensity = idx['PulseIntensity'][np.argsort(idx['BunchNumber'])]
    good_bunches = coinc.select_bunches(RUN_FILE.stem, intensity)
    mu = group_dt_peaks(RUN_FILE.stem)
    print(f'{len(good_bunches)} good bunches, tof-tflash > {LATE_TOF:.0f} ns')

    results = {}
    nb = len(AMP_EDGES) - 1
    for tree in ['WALA', 'WALB', 'WALC', 'WALD', 'PSSA', 'PSSB', 'PSSC', 'PSSD']:
        n_ch = 8 if tree.startswith('WAL') else 2
        st = tree[3]
        dat = {k: [] for k in ('BunchNumber', 'tof', 'detn', 'amp')}
        for chunk in f[tree].iterate(list(dat) + ['tflash'], library='np',
                                     step_size='300 MB'):
            keep = np.isin(chunk['BunchNumber'], good_bunches) & \
                   ((chunk['tof'] - chunk['tflash']) > LATE_TOF)
            for k in dat:
                dat[k].append(chunk[k][keep])
        arrs = {k: np.concatenate(v) for k, v in dat.items()}
        order = np.lexsort((arrs['tof'], arrs['BunchNumber']))
        arrs = {k: v[order] for k, v in arrs.items()}

        # 1. inclusive late spectra
        h_inc = np.histogram2d(arrs['detn'], arrs['amp'],
                               bins=[np.arange(0.5, n_ch + 1), AMP_EDGES])[0]
        results[f'{tree}_amp_late'] = h_inc
        print(f'{tree}: {len(arrs["tof"]):,} late hits', flush=True)

        # 2. wall-only top-bottom pairs per group
        if tree.startswith('WAL'):
            h_tb = np.zeros((4, 2, 2, nb))   # (group, top/bot, sig/side, amp)
            for g in range(4):
                m_top = arrs['detn'] == 2 * g + 1
                m_bot = arrs['detn'] == 2 * g + 2
                bt, tt, at_ = arrs['BunchNumber'][m_top], arrs['tof'][m_top], arrs['amp'][m_top]
                bb, tb, ab = arrs['BunchNumber'][m_bot], arrs['tof'][m_bot], arrs['amp'][m_bot]
                for bn in good_bunches:
                    bn = int(bn)
                    ta1, ta2 = np.searchsorted(bt, [bn, bn + 1])
                    tb1, tb2 = np.searchsorted(bb, [bn, bn + 1])
                    if ta1 == ta2 or tb1 == tb2:
                        continue
                    ri, oi = pair_idx(tt[ta1:ta2], tb[tb1:tb2])
                    if len(ri) == 0:
                        continue
                    dtc = (tb[tb1:tb2][oi] - tt[ta1:ta2][ri]) - mu[(st, g)]
                    for j, mask in enumerate((np.abs(dtc) <= W_SIG,
                                              (np.abs(dtc) >= SB_LO) & (np.abs(dtc) <= SB_HI))):
                        if not mask.any():
                            continue
                        for k, amps in enumerate((at_[ta1:ta2][ri], ab[tb1:tb2][oi])):
                            ai = np.clip(np.digitize(amps[mask], AMP_EDGES) - 1, 0, nb - 1)
                            np.add.at(h_tb, (g, k, j, ai), 1)
                results[f'{tree}_topbot'] = h_tb
            print(f'  {tree} top-bottom pairs done', flush=True)

    np.savez_compressed(CACHE / f'09_late_{RUN_FILE.stem}.npz',
                        amp_edges=AMP_EDGES, sb_scale=SB_SCALE, **results)
    print(f'Cached -> {CACHE / f"09_late_{RUN_FILE.stem}.npz"}')


if __name__ == '__main__':
    main()
