"""build_sum.py — Top+bottom SiPM SUM spectra per wall group, for trigger-threshold
studies. (New product: the 07/09 caches only store top and bottom MARGINALS; a sum
needs the per-event JOINT, so this re-reads the run.)

MEMORY-SAFE: processes ONE arm per invocation and writes a per-arm cache, so peak
RAM is one arm's compact columns (~0.6 GB) and a crash never loses other arms. The
`run_all.sh` driver launches A/B/C/D as separate processes; merge_and_plot reads the
four per-arm files. Columns are read in 150 MB chunks, masked to good+late hits on
the fly, and stored as int32/float32/int8 (no full-arm float64 copy).

Per arm and per 4-bar group (detn pairs (1,2)(3,4)(5,6)(7,8)):
  - pair each top-SiPM hit (odd detn) with same-group bottom-SiPM hits, same bunch,
    centred on the group's top-bottom dt peak (mu, from the 06 cache);
  - signal |dt_cal|<=8 ns = MIP+accidentals, sideband 20..80 ns (x2) = accidentals;
  - histogram S = a_top + a_bot (ADC) for signal/sideband, plus the (a_top,a_bot) 2D;
  - inclusive single-SiPM amp per channel (single-ended background reference).

Selection: beam-on & wall-active good bunches, tof - tflash > 0.1 ms (late, off-flash).

Usage: python build_sum.py <ARM A|B|C|D> [run_file]
"""

import gc
import importlib.util
import sys
from pathlib import Path

import numpy as np
import uproot

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location('coinc', BASE / '02_coincidence_scan.py')
coinc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coinc)

ARM = sys.argv[1].upper()
assert ARM in 'ABCD'
RUN_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    Path.home() / 'x17/beam_july/data/run224460.root'
CACHE_SRC = BASE / 'cache'
OUT = Path(__file__).parent / 'cache'
OUT.mkdir(exist_ok=True)

LATE_TOF = 1e5
W_SIG = 8.0
SB_LO, SB_HI = 20.0, 80.0
SB_SCALE = (2 * W_SIG) / (2 * (SB_HI - SB_LO))
DT_MAX = SB_HI + 20
STRIDE = 1e9                                    # bunch key stride (ns), > 20 ms window

AMP_EDGES = np.geomspace(40, 8e4, 301)
SUM_EDGES = np.geomspace(60, 1.6e5, 301)
J_EDGES = np.geomspace(40, 8e4, 81)


def group_dt_peaks(run_stem, st):
    d = np.load(CACHE_SRC / f'06_wallgeom_{run_stem}.npz')
    cen = 0.5 * (d['dt_edges'][:-1] + d['dt_edges'][1:])
    side = np.abs(cen) > 60
    h = d[f'intra_{st}']
    mu = []
    for g in range(4):
        hh = h[2 * g, 2 * g + 1]
        sub = np.clip(hh - hh[side].mean(), 0, None)
        win = np.abs(cen - cen[np.argmax(sub)]) <= 10
        mu.append(float(np.average(cen[win], weights=sub[win])))
    return mu


def load_arm_compact(tree, good_bunches):
    """Chunked read of one WAL arm; keep only good+late hits, compact dtypes.
    Returns bunch(int32), tof(float64), detn(int8), amp(float32), sorted by
    (bunch, tof). Peak memory ~ one arm's kept columns, no full-arm float64 copy."""
    good = np.asarray(good_bunches)
    gb, gt, gd, ga = [], [], [], []
    f = uproot.open(RUN_FILE)
    for ch in f[tree].iterate(['BunchNumber', 'tof', 'detn', 'amp', 'tflash'],
                              library='np', step_size='150 MB'):
        keep = np.isin(ch['BunchNumber'], good) & \
               ((ch['tof'] - ch['tflash']) > LATE_TOF)
        if keep.any():
            gb.append(ch['BunchNumber'][keep].astype(np.int32))
            gt.append(ch['tof'][keep].astype(np.float64))
            gd.append(ch['detn'][keep].astype(np.int8))
            ga.append(ch['amp'][keep].astype(np.float32))
        del ch
    bunch = np.concatenate(gb); tof = np.concatenate(gt)
    detn = np.concatenate(gd); amp = np.concatenate(ga)
    del gb, gt, gd, ga; gc.collect()
    order = np.lexsort((tof, bunch))
    return bunch[order], tof[order], detn[order], amp[order]


def iter_pairs(key_ref, key_other, dt_lo, dt_hi, max_pairs=8_000_000):
    lo = np.searchsorted(key_other, key_ref + dt_lo)
    hi = np.searchsorted(key_other, key_ref + dt_hi)
    counts = (hi - lo).astype(np.int64)
    cum = np.cumsum(counts)
    start, n = 0, len(counts)
    while start < n:
        base = cum[start - 1] if start else 0
        end = max(int(np.searchsorted(cum, base + max_pairs, side='right')),
                  start + 1)
        c = counts[start:end]
        tot = int(c.sum())
        if tot:
            ref_idx = np.repeat(np.arange(start, end), c)
            offs = np.repeat(np.cumsum(c) - c, c)
            other_idx = np.repeat(lo[start:end], c) + (np.arange(tot) - offs)
            yield ref_idx, other_idx
        start = end


def main():
    stem = RUN_FILE.stem
    intensity = coinc.bunch_intensity(RUN_FILE)
    good = coinc.select_bunches(stem, intensity)
    mu = group_dt_peaks(stem, ARM)
    print(f'{stem} arm {ARM}: {len(good)} good bunches; mu={np.round(mu,1)}',
          flush=True)

    bunch, tof, detn, amp = load_arm_compact(f'WAL{ARM}', good)
    print(f'  loaded {len(tof):,} good+late hits', flush=True)

    nS, nA, nJ = len(SUM_EDGES) - 1, len(AMP_EDGES) - 1, len(J_EDGES) - 1
    h_single = np.zeros((8, nA))
    for c in range(8):
        h_single[c] = np.histogram(amp[detn == c + 1], AMP_EDGES)[0]

    key = bunch.astype(np.float64) * STRIDE + tof
    h_sum = np.zeros((4, 2, nS))
    h_2d = np.zeros((4, 2, nJ, nJ))
    n_pairs = np.zeros((4, 2), np.int64)
    for g in range(4):
        mt, mb = detn == 2 * g + 1, detn == 2 * g + 2
        kt, at = key[mt], amp[mt]
        kb, ab, tb = key[mb], amp[mb], tof[mb]
        tt = tof[mt]
        if len(kt) == 0 or len(kb) == 0:
            continue
        m = mu[g]
        for ref, oth in iter_pairs(kt, kb, m - DT_MAX, m + DT_MAX):
            dtc = (tb[oth] - tt[ref]) - m
            s = (at[ref].astype(np.float64) + ab[oth])
            for j, mask in enumerate((np.abs(dtc) <= W_SIG,
                                      (np.abs(dtc) >= SB_LO) &
                                      (np.abs(dtc) <= SB_HI))):
                if not mask.any():
                    continue
                h_sum[g, j] += np.histogram(s[mask], SUM_EDGES)[0]
                h_2d[g, j] += np.histogram2d(at[ref][mask], ab[oth][mask],
                                             [J_EDGES, J_EDGES])[0]
                n_pairs[g, j] += int(mask.sum())
        del mt, mb, kt, at, kb, ab, tb, tt
        print(f'  grp{g+1}: sig={n_pairs[g,0]:,} side={n_pairs[g,1]:,}',
              flush=True)

    np.savez_compressed(OUT / f'sum_{stem}_{ARM}.npz',
                        sum_edges=SUM_EDGES, amp_edges=AMP_EDGES, j_edges=J_EDGES,
                        sb_scale=SB_SCALE, n_good=len(good), mu=np.array(mu),
                        h_sum=h_sum, h_2d=h_2d, h_single=h_single)
    print(f'  cached -> {OUT / f"sum_{stem}_{ARM}.npz"}', flush=True)


if __name__ == '__main__':
    main()
