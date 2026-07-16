"""
16_wall_vertical.py — Top/bottom SiPM combination and vertical-position analysis
for the SiPM-wall bars (vertical bars, read at top & bottom; ~50 cm long).

For each arm, each 4-bar group g (top channel detn=2g+1, bottom detn=2g+2):
  * match every top hit to its nearest bottom hit within +-TB_MAX ns (a real
    through-bar hit fires both ends);
  * MIP-tag the matched pair by a plastic coincidence on the mean wall time
    tw = (t_top+t_bot)/2, using the average of the two channels' stored offsets;
  * accumulate, in the plastic signal (|dt'|<=8 ns) and sideband (+20..+120 ns)
    windows so everything can be sideband-subtracted:
      - top-only / bottom-only / arithmetic-mean / geometric-mean amp spectra
      - geometric-mean spectrum split by which plastic bar tagged it
      - ln(A_top/A_bot)         -> vertical position from amplitude asymmetry
      - dt_tb = t_bot - t_top   -> vertical position from timing
      - 2D (dt_tb, ln A_top/A_bot) -> do the two position estimators agree?
      - 2D (A_top, A_bot)          -> attenuation / Landau-cancellation check

Everything reconstructed in ONE pass over the run file.  Sideband scale 16/100.

Usage: python 16_wall_vertical.py [run_file]
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import uproot

BASE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('coinc', BASE / '02_coincidence_scan.py')
coinc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coinc)

RUN_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'x17/beam_july/data/run224404.root'
NBUNCH = int(sys.argv[2]) if len(sys.argv) > 2 else 0     # 0 = all; >0 = smoke test
CACHE = BASE / 'cache'

W_SIG = 8.0
SB_LO, SB_HI = 20.0, 120.0
SB_SCALE = (2 * W_SIG) / (SB_HI - SB_LO)
DT_MAX = SB_HI + 25
TB_MAX = 15.0                       # top-bottom match window (ns)
LATE_TOF = 1e5

AMP_EDGES = np.geomspace(40, 8e4, 201)
A2_EDGES = np.geomspace(40, 8e4, 101)
LR_EDGES = np.linspace(-2.0, 2.0, 81)        # ln(A_top/A_bot)
DT_EDGES = np.linspace(-15, 15, 121)         # t_bot - t_top, 0.25 ns bins
LR2_EDGES = np.linspace(-1.5, 1.5, 61)
DT2_EDGES = np.linspace(-12, 12, 61)
VDT_EDGES = np.linspace(-15, 15, 21)         # coarse timing bins for geo-spectrum-vs-v
NA, NLR, NDT = len(AMP_EDGES) - 1, len(LR_EDGES) - 1, len(DT_EDGES) - 1
NA2 = len(A2_EDGES) - 1
NVDT = len(VDT_EDGES) - 1


def load_arm(f, st, good):
    dat = {}
    for tree, br in ((f'WAL{st}', ['BunchNumber', 'tof', 'detn', 'amp']),
                     (f'PSS{st}', ['BunchNumber', 'tof', 'detn', 'amp', 'tflash'])):
        out = {k: [] for k in br}
        for chunk in f[tree].iterate(br, library='np', step_size='300 MB'):
            keep = np.isin(chunk['BunchNumber'], good)
            if 'tflash' in br:
                keep &= (chunk['tof'] - chunk['tflash']) > LATE_TOF
            for k in br:
                out[k].append(chunk[k][keep])
        a = {k: np.concatenate(v) for k, v in out.items()}
        order = np.lexsort((a['tof'], a['BunchNumber']))
        dat[tree[:3]] = {k: v[order] for k, v in a.items()}
    return dat['WAL'], dat['PSS']


def match_nearest(t_top, t_bot):
    """Match each top hit to its nearest bottom hit within +-TB_MAX. Returns
    (top_idx, bot_idx) into the given per-bunch arrays."""
    if len(t_top) == 0 or len(t_bot) == 0:
        return np.empty(0, int), np.empty(0, int)
    j = np.searchsorted(t_bot, t_top)
    j0, j1 = np.clip(j - 1, 0, len(t_bot) - 1), np.clip(j, 0, len(t_bot) - 1)
    d0, d1 = np.abs(t_bot[j0] - t_top), np.abs(t_bot[j1] - t_top)
    jb = np.where(d0 <= d1, j0, j1)
    db = np.minimum(d0, d1)
    keep = db <= TB_MAX
    return np.nonzero(keep)[0], jb[keep]


def pair_idx(t_ref, t_other):
    lo = np.searchsorted(t_other, t_ref - DT_MAX)
    hi = np.searchsorted(t_other, t_ref + DT_MAX)
    counts = hi - lo
    tot = int(counts.sum())
    if tot == 0:
        return np.empty(0, int), np.empty(0, int)
    ref_idx = np.repeat(np.arange(len(t_ref)), counts)
    within = np.arange(tot) - np.repeat(np.cumsum(counts) - counts, counts)
    return ref_idx, np.repeat(lo, counts) + within


def main():
    t0 = time.time()
    f = uproot.open(RUN_FILE)
    intensity = coinc.bunch_intensity(RUN_FILE)
    good = coinc.select_bunches(RUN_FILE.stem, intensity)
    if NBUNCH:
        good = good[:NBUNCH]
    offs = json.loads((BASE / 'calib' / f'time_offsets_{RUN_FILE.stem}.json').read_text())['stations']
    print(f'{len(good)} bunches; sideband scale {SB_SCALE:.3f}', flush=True)

    res = {}
    for st in 'ABCD':
        wal, pss = load_arm(f, st, good)
        # per (top ch, bar) and (bottom ch, bar) offsets -> pair offset = mean, per group,bar
        off_pair = np.zeros((4, 2))
        for g in range(4):
            for bar in range(2):
                ot = offs[st][f'WAL{st}{2 * g + 1}_PSS{st}{bar + 1}']['offset_ns']
                ob = offs[st][f'WAL{st}{2 * g + 2}_PSS{st}{bar + 1}']['offset_ns']
                off_pair[g, bar] = 0.5 * (ot + ob)

        # accumulators: [sig/side, group, ...]
        spec = {k: np.zeros((2, 4, NA)) for k in ('top', 'bot', 'arith', 'geo')}
        geo_bar = np.zeros((2, 4, 2, NA))
        h_lr = np.zeros((2, 4, NLR))
        h_dt = np.zeros((2, 4, NDT))
        h2_dtlr = np.zeros((2, 4, len(DT2_EDGES) - 1, len(LR2_EDGES) - 1))
        h2_ab = np.zeros((2, 4, NA2, NA2))
        geo_v = np.zeros((2, 4, 2, NVDT, NA2))   # geo-mean spectrum vs v(timing) per (group,bar)

        wb = wal['BunchNumber']
        pb = pss['BunchNumber']
        for bn in good:
            bn = int(bn)
            wa, wz = np.searchsorted(wb, [bn, bn + 1])
            pa, pz = np.searchsorted(pb, [bn, bn + 1])
            if wa == wz or pa == pz:
                continue
            det = wal['detn'][wa:wz]
            tofw = wal['tof'][wa:wz]
            ampw = wal['amp'][wa:wz]
            t_p = pss['tof'][pa:pz]
            bar_p = pss['detn'][pa:pz] - 1
            for g in range(4):
                mt = det == 2 * g + 1
                mb = det == 2 * g + 2
                t_top, a_top = tofw[mt], ampw[mt]
                t_bot, a_bot = tofw[mb], ampw[mb]
                ti, bi = match_nearest(t_top, t_bot)
                if len(ti) == 0:
                    continue
                at, ab = a_top[ti], a_bot[bi]
                good_amp = (at > 0) & (ab > 0)
                if not good_amp.any():
                    continue
                ti, bi, at, ab = ti[good_amp], bi[good_amp], at[good_amp], ab[good_amp]
                tw = 0.5 * (t_top[ti] + t_bot[bi])
                dttb = t_bot[bi] - t_top[ti]
                # plastic coincidence on mean time
                ri, oi = pair_idx(tw, t_p)
                if len(ri) == 0:
                    continue
                bp = bar_p[oi]
                dt_cal = (tw[ri] - t_p[oi]) - off_pair[g, bp]
                sigm = np.abs(dt_cal) <= W_SIG
                sidem = (dt_cal >= SB_LO) & (dt_cal <= SB_HI)
                AT, AB = at[ri], ab[ri]
                DTB = dttb[ri]
                LR = np.log(AT / AB)
                GEO = np.sqrt(AT * AB)
                ARI = 0.5 * (AT + AB)
                for j, msk in ((0, sigm), (1, sidem)):
                    if not msk.any():
                        continue
                    def dig(x, e):
                        return np.clip(np.digitize(x[msk], e) - 1, 0, len(e) - 2)
                    np.add.at(spec['top'], (j, g, dig(AT, AMP_EDGES)), 1)
                    np.add.at(spec['bot'], (j, g, dig(AB, AMP_EDGES)), 1)
                    np.add.at(spec['arith'], (j, g, dig(ARI, AMP_EDGES)), 1)
                    np.add.at(spec['geo'], (j, g, dig(GEO, AMP_EDGES)), 1)
                    np.add.at(geo_bar, (j, g, bp[msk], dig(GEO, AMP_EDGES)), 1)
                    np.add.at(h_lr, (j, g, dig(LR, LR_EDGES)), 1)
                    np.add.at(h_dt, (j, g, dig(DTB, DT_EDGES)), 1)
                    np.add.at(h2_dtlr, (j, g, dig(DTB, DT2_EDGES), dig(LR, LR2_EDGES)), 1)
                    np.add.at(h2_ab, (j, g, dig(AT, A2_EDGES), dig(AB, A2_EDGES)), 1)
                    np.add.at(geo_v, (j, g, bp[msk], dig(DTB, VDT_EDGES),
                                      dig(GEO, A2_EDGES)), 1)
        for k, v in spec.items():
            res[f'{st}_{k}'] = v
        res[f'{st}_geobar'] = geo_bar
        res[f'{st}_lr'] = h_lr
        res[f'{st}_dt'] = h_dt
        res[f'{st}_dtlr'] = h2_dtlr
        res[f'{st}_ab'] = h2_ab
        res[f'{st}_geov'] = geo_v
        print(f'  arm {st}: {spec["geo"][0].sum():,.0f} sig / {spec["geo"][1].sum():,.0f} '
              f'sideband matched-pair coincidences ({time.time() - t0:.0f}s)', flush=True)

    tag = f'_n{NBUNCH}' if NBUNCH else ''
    np.savez_compressed(CACHE / f'16_vertical_{RUN_FILE.stem}{tag}.npz',
                        amp_edges=AMP_EDGES, a2_edges=A2_EDGES, lr_edges=LR_EDGES,
                        dt_edges=DT_EDGES, lr2_edges=LR2_EDGES, dt2_edges=DT2_EDGES,
                        vdt_edges=VDT_EDGES, sb_scale=SB_SCALE, tb_max=TB_MAX, **res)
    print(f'Cached -> {CACHE / f"16_vertical_{RUN_FILE.stem}{tag}.npz"} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
