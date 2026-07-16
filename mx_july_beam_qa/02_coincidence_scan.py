"""
02_coincidence_scan.py — WAL x PSS coincidence time-difference analysis.

For every (wall tree, plastic tree) pair, histogram dt = t_wall - t_pss for all hit
pairs in the same bunch within +-DT_MAX, split by
  - time since gamma flash of the plastic hit (log-decade regions), and
  - pulse type (dedicated / parasitic).
The window scan (coincidences vs half-width W) is then just a cumulative integral of
the dt histogram, with the combinatorial level measured from the flat sidebands.

Only beam-on bunches after the SiPM wall recovery (bunch > WALL_RECOVERY_BUNCH, see
README flag) are used.

Usage: python 02_coincidence_scan.py [run_file]
"""

import sys
import time
from pathlib import Path

import numpy as np
import uproot

RUN_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'x17/beam_july/data/run224404.root'
BASE = Path(__file__).parent
CACHE = BASE / 'cache'

WALL_TREES = ['WALA', 'WALB', 'WALC', 'WALD']
PSS_TREES = ['PSSA', 'PSSB', 'PSSC', 'PSSD']

DT_EDGES = np.arange(-150.0, 150.0 + 0.5, 1.0)          # ns, 1 ns bins
DT_MAX = DT_EDGES[-1]
# time since gamma flash (pss hit), ns: pre-flash, <1us, 1-10us, 10-100us, 0.1-1ms, >1ms
REGION_EDGES = np.array([-np.inf, 0, 1e3, 1e4, 1e5, 1e6, np.inf])
REGION_LABELS = ['pre-flash', '0-1us', '1-10us', '10-100us', '0.1-1ms', '>1ms']
DEDICATED_THRESH = 6e12
WALL_RECOVERY_BUNCH = 2212


def select_bunches(run_stem, bunch_intensity):
    qa = np.load(CACHE / f'01_qa_{run_stem}.npz')
    wall = sum(qa[f'{t}_hits_per_bunch'].sum(axis=0) for t in WALL_TREES)
    pss = sum(qa[f'{t}_hits_per_bunch'].sum(axis=0) for t in PSS_TREES)
    bunch_no = np.arange(1, len(wall) + 1)
    good = (pss > 0.5 * np.median(pss[pss > 1000])) & \
           (wall > 0.5 * np.median(wall[wall > 1000])) & \
           (bunch_no > WALL_RECOVERY_BUNCH)
    return bunch_no[good]


def load_tree(f, tree_name, good_bunches, need_tflash):
    """Return (bunch, tof) sorted by (bunch, tof), restricted to good bunches,
    plus {bunch: tflash} if requested."""
    good = set(good_bunches.tolist())
    branches = ['BunchNumber', 'tof'] + (['tflash'] if need_tflash else [])
    bunches, tofs, tflash_map = [], [], {}
    for chunk in f[tree_name].iterate(branches, library='np', step_size='300 MB'):
        b, t = chunk['BunchNumber'], chunk['tof']
        if need_tflash:
            first = np.unique(b, return_index=True)
            for bn, fi in zip(*first):
                tflash_map.setdefault(int(bn), float(chunk['tflash'][fi]))
        keep = np.isin(b, good_bunches)
        bunches.append(b[keep])
        tofs.append(t[keep])
    b = np.concatenate(bunches)
    t = np.concatenate(tofs)
    order = np.lexsort((t, b))
    return b[order], t[order], tflash_map


def pair_dts(t_ref, t_other):
    """All (t_other - t_ref) within +-DT_MAX; t_other must be sorted."""
    lo = np.searchsorted(t_other, t_ref - DT_MAX)
    hi = np.searchsorted(t_other, t_ref + DT_MAX)
    counts = hi - lo
    tot = int(counts.sum())
    if tot == 0:
        return np.empty(0), np.empty(0, dtype=np.int64)
    ref_idx = np.repeat(np.arange(len(t_ref)), counts)
    within = np.arange(tot) - np.repeat(np.cumsum(counts) - counts, counts)
    other_idx = np.repeat(lo, counts) + within
    return t_other[other_idx] - t_ref[ref_idx], ref_idx


def main():
    t0 = time.time()
    f = uproot.open(RUN_FILE)
    idx = f['index'].arrays(['BunchNumber', 'PulseIntensity'], library='np')
    order = np.argsort(idx['BunchNumber'])
    intensity = idx['PulseIntensity'][order]
    good_bunches = select_bunches(RUN_FILE.stem, intensity)
    is_ded = {int(bn): intensity[bn - 1] > DEDICATED_THRESH for bn in good_bunches}
    print(f'{len(good_bunches)} good bunches (beam-on, walls active, bunch>{WALL_RECOVERY_BUNCH}); '
          f'{sum(is_ded.values())} dedicated')

    data = {}
    for tn in WALL_TREES + PSS_TREES:
        data[tn] = load_tree(f, tn, good_bunches, need_tflash=tn in PSS_TREES)
        print(f'  loaded {tn}: {len(data[tn][0]):,} hits in good bunches '
              f'({time.time() - t0:.0f}s)', flush=True)

    n_reg, n_dt = len(REGION_LABELS), len(DT_EDGES) - 1
    h = {(w, p): np.zeros((2, n_reg, n_dt)) for w in WALL_TREES for p in PSS_TREES}
    n_pss_hits = {p: np.zeros((2, n_reg)) for p in PSS_TREES}   # for normalization

    # per-tree bunch slice bounds
    slices = {tn: {int(bn): (int(a), int(b_)) for bn, a, b_ in zip(
        good_bunches,
        np.searchsorted(data[tn][0], good_bunches, side='left'),
        np.searchsorted(data[tn][0], good_bunches, side='right'))}
        for tn in WALL_TREES + PSS_TREES}

    for i, bn in enumerate(good_bunches):
        bn = int(bn)
        ded = int(is_ded[bn])
        for p in PSS_TREES:
            a, b_ = slices[p][bn]
            if a == b_:
                continue
            t_pss = data[p][1][a:b_]
            tflash = data[p][2].get(bn, np.nan)
            region = np.digitize(t_pss - tflash, REGION_EDGES) - 1
            n_pss_hits[p][ded] += np.bincount(region, minlength=n_reg)
            for w in WALL_TREES:
                aw, bw = slices[w][bn]
                if aw == bw:
                    continue
                dt, ref_idx = pair_dts(t_pss, data[w][1][aw:bw])
                if len(dt) == 0:
                    continue
                h[(w, p)][ded] += np.histogram2d(
                    region[ref_idx], dt, bins=[np.arange(-0.5, n_reg), DT_EDGES])[0]
        if i % 200 == 0:
            print(f'  bunch {i}/{len(good_bunches)} ({time.time() - t0:.0f}s)', flush=True)

    out = {f'{w}_{p}': h[(w, p)] for w in WALL_TREES for p in PSS_TREES}
    out.update({f'npss_{p}': n_pss_hits[p] for p in PSS_TREES})
    np.savez_compressed(CACHE / f'02_coinc_{RUN_FILE.stem}.npz',
                        dt_edges=DT_EDGES, region_edges=REGION_EDGES,
                        region_labels=np.array(REGION_LABELS),
                        good_bunches=good_bunches,
                        n_dedicated=sum(is_ded.values()),
                        n_parasitic=len(good_bunches) - sum(is_ded.values()),
                        **out)
    print(f'Cached -> {CACHE / f"02_coinc_{RUN_FILE.stem}.npz"} ({time.time() - t0:.0f}s)')

    # quick significance table: peak (|dt|<=10ns around max) vs sideband, summed regions
    print(f'\nPair excess summary (all regions, ded+para):')
    print(f'{"pair":12s} {"peak dt":>8s} {"in +-10ns":>10s} {"comb.exp":>10s} {"excess":>10s}')
    for w in WALL_TREES:
        for p in PSS_TREES:
            hh = h[(w, p)].sum(axis=(0, 1))
            cen = 0.5 * (DT_EDGES[:-1] + DT_EDGES[1:])
            side = (np.abs(cen) > 100)
            base = hh[side].mean()
            imax = np.argmax(hh - base)
            win = np.abs(cen - cen[imax]) <= 10
            n_in, n_exp = hh[win].sum(), base * win.sum()
            print(f'{w}-{p:6s} {cen[imax]:8.1f} {n_in:10.0f} {n_exp:10.0f} '
                  f'{n_in - n_exp:10.0f}')


if __name__ == '__main__':
    main()
