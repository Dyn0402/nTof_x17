"""
06_wall_geometry_test.py — Test the assumed SiPM-wall geometry in data.

Claims under test (see README):
  1. detn pairs (1,2),(3,4),(5,6),(7,8) are top/bottom readouts of the same 4-bar
     group -> within-wall coincidence matrix should be block-diagonal in pairs,
     with a narrow dt peak per group (top-bottom transit/cable skew).
  2. groups are ordered in u -> coupling should fall with group distance.
  3. cross-arm channel matrices for the strongest arm pairs (A-D, B-C) -> facing
     arms should correlate mirrored-u (group 1 <-> group 4) if truly opposite.

Within-wall pairs use dt = t(ch j) - t(ch i), late-TOF hits only (>0.1 ms after
flash approximated by tof > tflash_typ + 1e5 using the wall tof directly).

Usage: python 06_wall_geometry_test.py [run_file]
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

import hitcache

BASE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('coinc', BASE / '02_coincidence_scan.py')
coinc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coinc)

RUN_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'x17/beam_july/data/run224404.root'
CACHE = BASE / 'cache'

DT_EDGES = np.arange(-100.0, 100.5, 1.0)
DT_MAX = DT_EDGES[-1]
LATE_TOF = 1.2e4 + 1e5   # ns since acq start: flash ~11.7us + 0.1ms


def load_wall(run_file, tree_name, good_bunches):
    d = hitcache.load(run_file, tree_name, ['BunchNumber', 'tof', 'detn'],
                      good_bunches=good_bunches)
    late = d['tof'] > LATE_TOF
    return {k: v[late] for k, v in d.items()}


def hist_pairs(dat_a, dat_b, n_a=8, n_b=8, same=False):
    """(n_a, n_b, ndt) dt histograms; same=True excludes self-pairs."""
    n_dt = len(DT_EDGES) - 1
    ta, tb = dat_a['tof'], dat_b['tof']
    cha = dat_a['detn'].astype(np.int64) - 1
    chb = dat_b['detn'].astype(np.int64) - 1
    key_a = hitcache.bunch_key(dat_a['BunchNumber'], ta)
    key_b = hitcache.bunch_key(dat_b['BunchNumber'], tb)
    acc = np.zeros(n_a * n_b * n_dt, dtype=np.int64)
    for ri, oi in hitcache.iter_pairs(key_a, key_b, -DT_MAX, DT_MAX, ta, tb):
        if same:
            keep = ri != oi
            ri, oi = ri[keep], oi[keep]
        dt = tb[oi] - ta[ri]
        dtb = np.clip(np.digitize(dt, DT_EDGES) - 1, 0, n_dt - 1)
        acc += np.bincount((cha[ri] * n_b + chb[oi]) * n_dt + dtb,
                           minlength=len(acc))
    return acc.reshape(n_a, n_b, n_dt).astype(float)


def main():
    intensity = coinc.bunch_intensity(RUN_FILE)
    good_bunches = coinc.select_bunches(RUN_FILE.stem, intensity)
    print(f'{len(good_bunches)} good bunches, late-TOF wall hits only')

    results = {}
    for pair in [('A', 'D'), ('B', 'C')]:
        dat = {st: load_wall(RUN_FILE, f'WAL{st}', good_bunches) for st in pair}
        for st in pair:
            print(f'  WAL{st}: {len(dat[st]["tof"]):,} late hits', flush=True)
            results[f'intra_{st}'] = hist_pairs(dat[st], dat[st], same=True)
            print(f'  intra-{st} done', flush=True)
        results[f'cross_{pair[0]}{pair[1]}'] = hist_pairs(dat[pair[0]], dat[pair[1]])
        print(f'  cross {pair[0]}-{pair[1]} done', flush=True)

    np.savez_compressed(CACHE / f'06_wallgeom_{RUN_FILE.stem}.npz',
                        dt_edges=DT_EDGES, **results)
    print(f'Cached -> {CACHE / f"06_wallgeom_{RUN_FILE.stem}.npz"}')

    # summary: top-bottom pairs
    cen = 0.5 * (DT_EDGES[:-1] + DT_EDGES[1:])
    side = np.abs(cen) > 60
    print(f'\nTop-bottom (odd,even) same-group pairs, dt = t(even) - t(odd):')
    print(f'{"pair":12s} {"excess":>10s} {"peak[ns]":>9s} {"rms[ns]":>8s} '
          f'{"frac of odd-ch pairs":>20s}')
    for st in 'ABCD':
        h = results[f'intra_{st}']
        for g in range(4):
            hh = h[2 * g, 2 * g + 1]
            base = hh[side].mean()
            sub = np.clip(hh - base, 0, None)
            ipk = np.argmax(sub)
            win = np.abs(cen - cen[ipk]) <= 10
            exc = sub[win].sum()
            mu = np.average(cen[win], weights=sub[win]) if exc > 0 else np.nan
            rms = np.sqrt(np.average((cen[win] - mu) ** 2, weights=sub[win])) if exc > 0 else np.nan
            row_exc = []
            for cb in range(8):
                hb = h[2 * g, cb]
                bb = hb[side].mean()
                sb = np.clip(hb - bb, 0, None)
                iw = np.abs(cen - cen[np.argmax(sb)]) <= 10
                row_exc.append(sb[iw].sum())
            frac = exc / max(sum(row_exc), 1)
            print(f'WAL{st} g{g + 1} ({2 * g + 1},{2 * g + 2}) {exc:>9.0f} {mu:>9.2f} '
                  f'{rms:>8.2f} {frac:>19.2f}')


if __name__ == '__main__':
    main()
