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

import hitcache

RUN_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'x17/beam_july/data/run224404.root'
BASE = Path(__file__).parent
CACHE = BASE / 'cache'

WALL_TREES = ['WALA', 'WALB', 'WALC', 'WALD']
PSS_TREES = ['PSSA', 'PSSB', 'PSSC', 'PSSD']
LIQ_TREES = ['LIQA', 'LIQB', 'LIQC', 'LIQD']    # liquids, from run224489 on

DT_EDGES = np.arange(-150.0, 150.0 + 0.5, 1.0)          # ns, 1 ns bins
DT_MAX = DT_EDGES[-1]
# LIQ never timed in: scan generously to even find the peak (same-arm pairs only)
DT_EDGES_LIQ = np.arange(-400.0, 400.0 + 0.5, 1.0)
DT_MAX_LIQ = DT_EDGES_LIQ[-1]
# time since gamma flash (pss hit), ns: pre-flash, <1us, 1-10us, 10-100us, 0.1-1ms, >1ms
REGION_EDGES = np.array([-np.inf, 0, 1e3, 1e4, 1e5, 1e6, np.inf])
REGION_LABELS = ['pre-flash', '0-1us', '1-10us', '10-100us', '0.1-1ms', '>1ms']
DEDICATED_THRESH = 6e12
# hard bunch masks for known per-run DAQ outages (README flag); default 0 = no mask
WALL_RECOVERY_BUNCH_BY_RUN = {'run224404': 2212}
WALL_RECOVERY_BUNCH = WALL_RECOVERY_BUNCH_BY_RUN.get(
    (Path(sys.argv[1]).stem if len(sys.argv) > 1 else 'run224404'), 0)  # legacy alias


def bunch_intensity(run_file):
    """Per-bunch PulseIntensity sorted by bunch number (index tree; PKUP
    fallback for an empty index, as in run224460). Thin hitcache wrapper,
    kept here because downstream scripts import it from this module."""
    return hitcache.bunch_intensity(run_file)


def select_bunches(run_stem, bunch_intensity):
    recovery = WALL_RECOVERY_BUNCH_BY_RUN.get(run_stem, 0)
    qa = np.load(CACHE / f'01_qa_{run_stem}.npz')
    wall = sum(qa[f'{t}_hits_per_bunch'].sum(axis=0) for t in WALL_TREES)
    pss = sum(qa[f'{t}_hits_per_bunch'].sum(axis=0) for t in PSS_TREES)
    bunch_no = np.arange(1, len(wall) + 1)
    good = (pss > 0.5 * np.median(pss[pss > 1000])) & \
           (wall > 0.5 * np.median(wall[wall > 1000])) & \
           (bunch_no > recovery)
    return bunch_no[good]


def load_tree(run_file, tree_name, good_bunches, need_tflash):
    """Return (bunch, tof) sorted by (bunch, tof), restricted to good bunches,
    plus the first tflash per bunch (array indexed by bunch number, nan when
    the bunch has no hits) if requested."""
    branches = ['BunchNumber', 'tof'] + (['tflash'] if need_tflash else [])
    d = hitcache.load(run_file, tree_name, branches, good_bunches=good_bunches)
    tflash = None
    if need_tflash:
        n_bunches = int(good_bunches.max()) if len(good_bunches) else 0
        tflash = hitcache.first_by_bunch(d['BunchNumber'], d['tflash'], n_bunches)
    return d['BunchNumber'], d['tof'], tflash


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
    intensity = bunch_intensity(RUN_FILE)
    good_bunches = select_bunches(RUN_FILE.stem, intensity)
    ded_by_bunch = np.zeros(int(good_bunches.max()) + 1 if len(good_bunches) else 1,
                            dtype=np.int64)
    ded_by_bunch[good_bunches] = intensity[good_bunches - 1] > DEDICATED_THRESH
    n_ded = int(ded_by_bunch[good_bunches].sum())
    print(f'{len(good_bunches)} good bunches (beam-on, walls active, bunch>{WALL_RECOVERY_BUNCH}); '
          f'{n_ded} dedicated')

    n_reg, n_dt = len(REGION_LABELS), len(DT_EDGES) - 1
    h = {}
    n_pss_hits = {}   # for normalization

    for p in PSS_TREES:
        b_p, t_p, tflash = load_tree(RUN_FILE, p, good_bunches, need_tflash=True)
        # region/dedicated tag per pss hit; pad one region slot for out-of-range
        region = np.digitize(t_p - tflash[b_p], REGION_EDGES) - 1
        region = np.clip(region, 0, n_reg)          # nan tflash -> pad slot n_reg
        ded = ded_by_bunch[b_p]
        n_pss_hits[p] = np.bincount(ded * (n_reg + 1) + region,
                                    minlength=2 * (n_reg + 1)
                                    ).reshape(2, n_reg + 1)[:, :n_reg].astype(float)
        key_p = hitcache.bunch_key(b_p, t_p)
        print(f'  loaded {p}: {len(b_p):,} hits in good bunches '
              f'({time.time() - t0:.0f}s)', flush=True)
        for w in WALL_TREES:
            b_w, t_w, _ = load_tree(RUN_FILE, w, good_bunches, need_tflash=False)
            key_w = hitcache.bunch_key(b_w, t_w)
            acc = np.zeros(2 * n_reg * n_dt, dtype=np.int64)
            for ri, oi in hitcache.iter_pairs(key_p, key_w, -DT_MAX, DT_MAX,
                                              t_p, t_w):
                dt = t_w[oi] - t_p[ri]
                dtb = np.digitize(dt, DT_EDGES) - 1
                ok = (dtb >= 0) & (dtb < n_dt) & (region[ri] < n_reg)
                idx = (ded[ri[ok]] * n_reg + region[ri[ok]]) * n_dt + dtb[ok]
                acc += np.bincount(idx, minlength=len(acc))
            h[(w, p)] = acc.reshape(2, n_reg, n_dt).astype(float)
            print(f'  {w}x{p}: {int(h[(w, p)].sum()):,} pairs '
                  f'({time.time() - t0:.0f}s)', flush=True)

    # --- same-arm LIQ pairings (wide window; LIQ trees exist from run224489 on)
    d = hitcache.cache_dir(RUN_FILE)
    liq_avail = [t for t in LIQ_TREES
                 if d is not None and (d / f'{t}_bunch.npy').exists()]
    n_dt_liq = len(DT_EDGES_LIQ) - 1
    h_liq, n_liq_hits = {}, {}
    for lq in liq_avail:
        arm = lq[-1]
        b_l, t_l, tflash_l = load_tree(RUN_FILE, lq, good_bunches, need_tflash=True)
        region_l = np.digitize(t_l - tflash_l[b_l], REGION_EDGES) - 1
        region_l = np.clip(region_l, 0, n_reg)
        ded_l = ded_by_bunch[b_l]
        n_liq_hits[lq] = np.bincount(ded_l * (n_reg + 1) + region_l,
                                     minlength=2 * (n_reg + 1)
                                     ).reshape(2, n_reg + 1)[:, :n_reg].astype(float)
        key_l = hitcache.bunch_key(b_l, t_l)
        print(f'  loaded {lq}: {len(b_l):,} hits in good bunches '
              f'({time.time() - t0:.0f}s)', flush=True)
        for other in (f'WAL{arm}', f'PSS{arm}'):
            b_o, t_o, _ = load_tree(RUN_FILE, other, good_bunches, need_tflash=False)
            key_o = hitcache.bunch_key(b_o, t_o)
            acc = np.zeros(2 * n_reg * n_dt_liq, dtype=np.int64)
            for ri, oi in hitcache.iter_pairs(key_l, key_o, -DT_MAX_LIQ, DT_MAX_LIQ,
                                              t_l, t_o):
                dt = t_o[oi] - t_l[ri]
                dtb = np.digitize(dt, DT_EDGES_LIQ) - 1
                ok = (dtb >= 0) & (dtb < n_dt_liq) & (region_l[ri] < n_reg)
                idx = (ded_l[ri[ok]] * n_reg + region_l[ri[ok]]) * n_dt_liq + dtb[ok]
                acc += np.bincount(idx, minlength=len(acc))
            h_liq[(other, lq)] = acc.reshape(2, n_reg, n_dt_liq).astype(float)
            print(f'  {other}x{lq}: {int(h_liq[(other, lq)].sum()):,} pairs '
                  f'({time.time() - t0:.0f}s)', flush=True)

    out = {f'{w}_{p}': h[(w, p)] for w in WALL_TREES for p in PSS_TREES}
    out.update({f'npss_{p}': n_pss_hits[p] for p in PSS_TREES})
    out.update({f'{o}_{lq}': v for (o, lq), v in h_liq.items()})
    out.update({f'nliq_{lq}': n_liq_hits[lq] for lq in liq_avail})
    if liq_avail:
        out['dt_edges_liq'] = DT_EDGES_LIQ
    np.savez_compressed(CACHE / f'02_coinc_{RUN_FILE.stem}.npz',
                        dt_edges=DT_EDGES, region_edges=REGION_EDGES,
                        region_labels=np.array(REGION_LABELS),
                        good_bunches=good_bunches,
                        n_dedicated=n_ded,
                        n_parasitic=len(good_bunches) - n_ded,
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
    for (o, lq), v in h_liq.items():
        hh = v.sum(axis=(0, 1))
        cen = 0.5 * (DT_EDGES_LIQ[:-1] + DT_EDGES_LIQ[1:])
        side = (np.abs(cen) > 350)
        base = hh[side].mean()
        imax = np.argmax(hh - base)
        win = np.abs(cen - cen[imax]) <= 10
        n_in, n_exp = hh[win].sum(), base * win.sum()
        print(f'{o}-{lq:6s} {cen[imax]:8.1f} {n_in:10.0f} {n_exp:10.0f} '
              f'{n_in - n_exp:10.0f}')


if __name__ == '__main__':
    main()
