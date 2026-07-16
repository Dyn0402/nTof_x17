"""18_trigger_threshold.py — Trigger-threshold study on top-bottom SUM amplitudes.

The wall trigger will fire on the analog sum of each 4-bar group's top+bottom
SiPMs, with ONE threshold per wall. This script builds, per (arm, group), after
the duplication veto of 17:

  * trigger-candidate sample: prompt top-bottom pairs (|dt - mu_g| <= 8 ns,
    late tof), observable sum_mV = amp_top*f_top + amp_bot*f_bot;
  * MIP subsample: candidates with a true same-arm plastic coincidence
    (|dt_cal| <= 8 ns, positive sideband +20..+120 ns subtracted);
  * plastic tag efficiency eps_p per group from the high-sum region
    (1.3-2.5x MIP-sum peak) where candidates are MIP-dominated.

Then vs threshold T (1 mV steps):
  efficiency_g(T)  = MIP subsample fraction above T          (eps_p cancels)
  purity_g(T)      = [tagged_sub(>=T)/eps_p] / candidates(>=T), capped at 1
  rate_g(T)        = accepted candidates per bunch (also unvetoed variant ->
                     extra rate if the duplication short is NOT yet fixed)

Cache: cache/18_trigsum_<run>.npz.  Figures: 18b_trigger_figs.py.

Usage: python 18_trigger_threshold.py [run_file]
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
import hitcache

spec = importlib.util.spec_from_file_location('coinc', BASE / '02_coincidence_scan.py')
coinc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coinc)

RUN_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'x17/beam_july/data/run224460.root'
W_TB = 8.0                       # top-bottom pair window around mu_g
W_SIG, SB_LO, SB_HI = 8.0, 20.0, 120.0
SB_SCALE = (2 * W_SIG) / (SB_HI - SB_LO)
DT_MAX = SB_HI + 25
LATE_TOF = 1e5
XT_DT, RATIO_LO, RATIO_HI = 4.0, 1 / 3.0, 3.0
SUM_EDGES = np.arange(0.0, 300.5, 1.0)     # mV
NS = len(SUM_EDGES) - 1

intensity = coinc.bunch_intensity(RUN_FILE)
good_bunches = coinc.select_bunches(RUN_FILE.stem, intensity)
n_bunches = len(good_bunches)
offs = json.loads((BASE / 'calib' / f'time_offsets_{RUN_FILE.stem}.json').read_text())['stations']
fac = json.loads((BASE / 'calib' / f'adc_to_mv_{RUN_FILE.stem}.json').read_text())['factors']

d6 = np.load(BASE / 'cache' / f'06_wallgeom_{RUN_FILE.stem}.npz')
cen6 = 0.5 * (d6['dt_edges'][:-1] + d6['dt_edges'][1:])
side6 = np.abs(cen6) > 60
MU = {}
for st in 'ABCD':
    h6 = d6[f'intra_{st}']
    for g in range(4):
        hh = h6[2 * g, 2 * g + 1]
        sub = np.clip(hh - hh[side6].mean(), 0, None)
        win = np.abs(cen6 - cen6[np.argmax(sub)]) <= 10
        MU[(st, g)] = float(np.average(cen6[win], weights=sub[win]))

print(f'{n_bunches} good bunches', flush=True)

results = {}
for st in 'ABCD':
    wal = hitcache.load(RUN_FILE, f'WAL{st}', ['BunchNumber', 'tof', 'detn', 'amp'],
                        good_bunches=good_bunches)
    late = wal['tof'] > 1.2e4 + LATE_TOF
    wal = {k: v[late] for k, v in wal.items()}
    key_w = hitcache.bunch_key(wal['BunchNumber'], wal['tof'])

    pss = hitcache.load(RUN_FILE, f'PSS{st}',
                        ['BunchNumber', 'tof', 'detn', 'amp', 'tflash'],
                        good_bunches=good_bunches)
    latep = (pss['tof'] - pss['tflash']) > LATE_TOF
    pss = {k: v[latep] for k, v in pss.items()}
    key_p = hitcache.bunch_key(pss['BunchNumber'], pss['tof'])
    print(f'arm {st}: {len(key_w):,} wall / {len(key_p):,} late pss', flush=True)

    # duplication flags (rule of 17)
    flagged = np.zeros(len(key_w), dtype=bool)
    ch_idx = [np.nonzero(wal['detn'] == c + 1)[0] for c in range(8)]
    for c in range(8):
        for c2 in (c - 2, c + 2):
            if not (0 <= c2 < 8):
                continue
            ia, ib = ch_idx[c], ch_idx[c2]
            if not len(ia) or not len(ib):
                continue
            for ri, oi in hitcache.iter_pairs(key_w[ia], key_w[ib], -XT_DT, XT_DT,
                                              wal['tof'][ia], wal['tof'][ib]):
                ratio = wal['amp'][ib][oi] / np.maximum(wal['amp'][ia][ri], 1)
                ok = (ratio >= RATIO_LO) & (ratio <= RATIO_HI)
                flagged[ia[ri[ok]]] = True

    h_cand = np.zeros((4, NS))          # vetoed trigger candidates
    h_cand_raw = np.zeros((4, NS))      # without veto (rate impact if unfixed)
    h_tag = np.zeros((4, 2, NS))        # (group, sig/side) tagged, vetoed
    for g in range(4):
        it, ib_ = ch_idx[2 * g], ch_idx[2 * g + 1]
        ft = fac[f'WAL{st}'][str(2 * g + 1)]
        fb = fac[f'WAL{st}'][str(2 * g + 2)]
        mu = MU[(st, g)]
        pr_t, pr_b = [], []
        for ri, oi in hitcache.iter_pairs(key_w[it], key_w[ib_], mu - W_TB,
                                          mu + W_TB, wal['tof'][it], wal['tof'][ib_]):
            pr_t.append(ri)
            pr_b.append(oi)
        ri = np.concatenate(pr_t) if pr_t else np.empty(0, np.int64)
        oi = np.concatenate(pr_b) if pr_b else np.empty(0, np.int64)
        _, first = np.unique(ri, return_index=True)     # dedupe on top hit
        ri, oi = ri[first], oi[first]
        i_t, i_b = it[ri], ib_[oi]
        sum_mv = wal['amp'][i_t] * ft + wal['amp'][i_b] * fb
        si = np.clip(np.digitize(sum_mv, SUM_EDGES) - 1, 0, NS - 1)
        ok = ~(flagged[i_t] | flagged[i_b])
        np.add.at(h_cand_raw[g], si, 1)
        np.add.at(h_cand[g], si[ok], 1)

        # plastic tag on the vetoed candidates
        i_t, i_b, si, sum_mv = i_t[ok], i_b[ok], si[ok], sum_mv[ok]
        t_pair = 0.5 * (wal['tof'][i_t] + wal['tof'][i_b])
        key_pair = hitcache.bunch_key(wal['BunchNumber'][i_t], t_pair)
        off_gb = np.array([0.5 * (offs[st][f'WAL{st}{2 * g + 1}_PSS{st}{pc + 1}']['offset_ns'] +
                                  offs[st][f'WAL{st}{2 * g + 2}_PSS{st}{pc + 1}']['offset_ns'])
                           for pc in range(2)])
        order = np.argsort(key_pair)
        inv_t = t_pair[order]
        inv_si = si[order]
        for pj, oj in hitcache.iter_pairs(key_p, key_pair[order], -DT_MAX, DT_MAX,
                                          pss['tof'], inv_t):
            dt_cal = (inv_t[oj] - pss['tof'][pj]) - off_gb[pss['detn'][pj] - 1]
            for j, m in enumerate((np.abs(dt_cal) <= W_SIG,
                                   (dt_cal >= SB_LO) & (dt_cal <= SB_HI))):
                np.add.at(h_tag, (g, j, inv_si[oj[m]]), 1)
    results[f'{st}_cand'] = h_cand
    results[f'{st}_cand_raw'] = h_cand_raw
    results[f'{st}_tag'] = h_tag
    print(f'  {st}: candidates {h_cand.sum(axis=1).astype(int)}', flush=True)

np.savez_compressed(BASE / 'cache' / f'18_trigsum_{RUN_FILE.stem}.npz',
                    sum_edges=SUM_EDGES, sb_scale=SB_SCALE, n_bunches=n_bunches,
                    **results)
print(f'Cached -> cache/18_trigsum_{RUN_FILE.stem}.npz')

# ---------------------------------------------------------------- summary
cen = 0.5 * (SUM_EDGES[:-1] + SUM_EDGES[1:])
kern = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
kern /= kern.sum()
print('\n=== per (arm, group): MIP-sum peak, eps_p, and threshold table ===')
for st in 'ABCD':
    cand = results[f'{st}_cand']
    tag = results[f'{st}_tag']
    print(f'\nWAL{st}:')
    for g in range(4):
        sub = tag[g, 0] - SB_SCALE * tag[g, 1]
        sm = np.convolve(sub, kern, mode='same')
        m = cen > 15
        pk = cen[m][np.argmax(sm[m])]
        hi = (cen >= 1.3 * pk) & (cen <= 2.5 * pk)
        eps = sub[hi].sum() / max(cand[g][hi].sum(), 1)
        mip = sub / max(eps, 1e-9)
        ceff = np.cumsum(sub[::-1])[::-1] / max(sub.sum(), 1)
        cmip = np.cumsum(mip[::-1])[::-1]
        ccand = np.cumsum(cand[g][::-1])[::-1]
        pur = np.minimum(cmip / np.maximum(ccand, 1), 1)
        print(f'  g{g + 1}: MIP-sum peak {pk:5.1f} mV, eps_p {eps:.3f}; '
              f'thr(eff 99/95/90%) = '
              f'{cen[np.searchsorted(-ceff, -0.99)]:5.1f}/'
              f'{cen[np.searchsorted(-ceff, -0.95)]:5.1f}/'
              f'{cen[np.searchsorted(-ceff, -0.90)]:5.1f} mV; '
              f'purity@those = {pur[np.searchsorted(-ceff, -0.99)]:.2f}/'
              f'{pur[np.searchsorted(-ceff, -0.95)]:.2f}/'
              f'{pur[np.searchsorted(-ceff, -0.90)]:.2f}')
