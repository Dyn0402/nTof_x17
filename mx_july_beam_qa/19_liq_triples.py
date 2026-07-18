"""
19_liq_triples.py — WAL x PSS x LIQ triple coincidences: plastic MIP attempt +
LIQ gain-vs-position map (run224489, first run with the liquids).

Physics: a wall-plastic tag selects wall MIPs only (the plastic sits BEHIND the
wall in that pair). Requiring in addition a LIQ hit BEHIND the plastic (stack:
MM -> SiPM wall -> plastics -> LS layer, as-built 2026-07-15) selects
through-going particles, so the PLASTIC spectrum in triples is a genuine
plastic-MIP candidate spectrum — the calibration the wall-plastic pairs could
never give.

Method (per arm, one pass):
  * wall-plastic pairs as in 07/12: late pss (tof - tflash > 0.1 ms), dt_cal =
    (t_wall - t_pss) - offset;  signal |dt_cal| <= 8 ns, sideband +20..+120 ns
    (scale 16/100).
  * every pair is then matched to LIQ hits near the plastic time: dt_liq =
    (t_liq - t_pss) - liq_peak(arm), the per-arm peak measured from the 02
    cache (LIQ leads the plastic by ~30 ns).  LIQ signal |dt_liq| <= 10 ns,
    LIQ sideband 25 <= |dt_liq| <= 125 ns (scale 20/200).
  * 2x2 sideband combinations are accumulated separately so the true triple
    spectrum is the double subtraction
        S = SS - s_l*SB - s_w*BS + s_w*s_l*BB.
  * plastic/wall/LIQ amplitude spectra per HV-scan step (12's step windows, a
    trailing slot collects out-of-window bunches; sum slots for inclusive).
  * LIQ gain-vs-position (T6): per triple, horizontal = wall group (4) x
    plastic bar (2); vertical = ln(A_top/A_bot) of the tagged wall hit and its
    partner channel (odd detn = top, README flag), nearest partner within
    +-15 ns, 6 bins in [-1.5, 1.5].  LIQ spectra accumulated per position bin
    (combo-split for the same double subtraction).
  * per-channel LIQ timing (T7): dt(t_wall - t_liq) per wall channel and
    dt(t_pss - t_liq) per bar, 1 ns bins +-150 ns, late LIQ hits.

Cache: cache/19_triples_<run>.npz.  Figures: 19b_triples_figs.py.

Usage: python 19_liq_triples.py [run_file]
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
import hitcache

spec = importlib.util.spec_from_file_location('coinc', BASE / '02_coincidence_scan.py')
coinc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coinc)
spec12 = importlib.util.spec_from_file_location('hv', BASE / '12_plastic_hv_scan.py')
hv = importlib.util.module_from_spec(spec12)
spec12.loader.exec_module(hv)

RUN_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'x17/beam_july/data/run224489.root'
CACHE = BASE / 'cache'

LATE_TOF = 1e5
W_SIG = 8.0                       # wall-plastic signal half-width (dt_cal)
SB_LO, SB_HI = 20.0, 120.0        # wall-plastic sideband
S_WP = (2 * W_SIG) / (SB_HI - SB_LO)
WL_SIG = 10.0                     # LIQ signal half-width around per-arm peak
WL_SB_LO, WL_SB_HI = 25.0, 125.0  # LIQ double sideband (both sides)
S_LQ = (2 * WL_SIG) / (2 * (WL_SB_HI - WL_SB_LO))
TB_MAX = 15.0                     # wall top-bottom partner window

AMP_EDGES = np.geomspace(40, 8e4, 301)     # plastic/wall spectra (as 12)
LAMP_EDGES = np.geomspace(40, 8e4, 151)    # LIQ / position-binned spectra
LR_EDGES = np.linspace(-1.5, 1.5, 7)       # ln(A_top/A_bot), 6 vertical bins
DT_EDGES = np.arange(-150.0, 150.5, 1.0)   # per-channel LIQ timing
NB, NLB, NLR = len(AMP_EDGES) - 1, len(LAMP_EDGES) - 1, len(LR_EDGES) - 1
NDT = len(DT_EDGES) - 1


def liq_peaks(run_stem):
    """Per-arm (t_liq - t_pss) peak from the 02 cache (stored as t_pss - t_liq)."""
    d = np.load(CACHE / f'02_coinc_{run_stem}.npz')
    ce = d['dt_edges_liq']
    cen = 0.5 * (ce[:-1] + ce[1:])
    pk = {}
    for arm in 'ABCD':
        h = d[f'PSS{arm}_LIQ{arm}'].sum(axis=(0, 1))
        base = h[np.abs(cen) > 350].mean()
        sub = h - base
        ipk = np.argmax(sub)
        win = np.abs(cen - cen[ipk]) <= 12
        w = np.clip(sub[win], 0, None)
        pk[arm] = -float(np.average(cen[win], weights=w))
    return pk


def partner_arrays(wal):
    """Per wall hit: nearest same-bunch partner-channel hit within +-TB_MAX ->
    (partner amp, t_partner - t_hit), nan when absent. detn pairs (1,2)(3,4)...
    view the same 4-bar group top/bottom; odd = top (README flag)."""
    t, ch, amp = wal['tof'], wal['detn'].astype(np.int64), wal['amp']
    key = hitcache.bunch_key(wal['BunchNumber'], t)
    n = len(t)
    p_amp = np.full(n, np.nan, np.float32)
    p_dt = np.full(n, np.nan, np.float32)
    partner = ch + np.where(ch % 2 == 1, 1, -1)
    for ri, oi in hitcache.iter_pairs(key, key, -TB_MAX, TB_MAX, t, t):
        m = ch[oi] == partner[ri]
        ri, oi = ri[m], oi[m]
        dt = (t[oi] - t[ri]).astype(np.float32)
        # keep the nearest |dt| per hit: sort worst->best so best lands last
        order = np.argsort(-np.abs(dt))
        p_amp[ri[order]] = amp[oi[order]]
        p_dt[ri[order]] = dt[order]
    return p_amp, p_dt


def main():
    t0 = time.time()
    run_stem = RUN_FILE.stem
    intensity = coinc.bunch_intensity(RUN_FILE)
    good_bunches = coinc.select_bunches(run_stem, intensity)
    n_bunches = len(intensity)
    step_of = hv.bunch_step_map(RUN_FILE, n_bunches)
    n_step = len(hv.STEPS) + 1                 # trailing slot: outside windows
    step_of = np.where(step_of < 0, n_step - 1, step_of)
    offs_json = json.loads(
        (BASE / 'calib' / f'time_offsets_{run_stem}.json').read_text())
    pk = liq_peaks(run_stem)
    print(f'{len(good_bunches)} good bunches; LIQ peaks (t_liq - t_pss): '
          + ', '.join(f'{a}:{-v:+.1f}ns' for a, v in pk.items()))

    R = {
        # (step, arm, wp 0=sig/1=side, liq 0=sig/1=side, bar, amp)
        'pss_amp': np.zeros((n_step, 4, 2, 2, 2, NB)),
        'wal_amp': np.zeros((n_step, 4, 2, 2, NB)),
        'liq_amp': np.zeros((n_step, 4, 2, 2, NLB)),
        # position map: (arm, wp, liq, group, bar, lrbin, amp)
        'liq_pos': np.zeros((4, 2, 2, 4, 2, NLR, NLB)),
        'n_novert': np.zeros((4, 2, 2)),       # triples without a vertical est.
        # timing: per wall channel/bar LIQ dt (late liq hits, no plastic tag)
        'walliq_dt': np.zeros((4, 8, NDT)),
        'pssliq_dt': np.zeros((4, 2, NDT)),
    }

    for ai, st in enumerate('ABCD'):
        off = np.zeros((8, 2))
        for wc in range(8):
            for pc in range(2):
                off[wc, pc] = offs_json['stations'][st][
                    f'WAL{st}{wc + 1}_PSS{st}{pc + 1}']['offset_ns']

        wal = hitcache.load(RUN_FILE, f'WAL{st}',
                            ['BunchNumber', 'tof', 'detn', 'amp'],
                            good_bunches=good_bunches)
        pss = hitcache.load(RUN_FILE, f'PSS{st}',
                            ['BunchNumber', 'tof', 'detn', 'amp', 'tflash'],
                            good_bunches=good_bunches)
        liq = hitcache.load(RUN_FILE, f'LIQ{st}',
                            ['BunchNumber', 'tof', 'amp', 'tflash'],
                            good_bunches=good_bunches)
        late = (pss['tof'] - pss['tflash']) > LATE_TOF
        pss = {k: v[late] for k, v in pss.items()}
        print(f'arm {st}: {len(wal["tof"]):,} wall / {len(pss["tof"]):,} late pss '
              f'/ {len(liq["tof"]):,} liq ({time.time() - t0:.0f}s)', flush=True)

        p_amp, p_dt = partner_arrays(wal)
        ch_w = wal['detn'].astype(np.int64)
        # vertical estimator per wall hit: ln(A_top/A_bot), odd detn = top
        with np.errstate(divide='ignore', invalid='ignore'):
            lr_all = np.where(ch_w % 2 == 1,
                              np.log(wal['amp'] / p_amp),
                              np.log(p_amp / wal['amp']))

        t_p, t_w, t_l = pss['tof'], wal['tof'], liq['tof']
        bar = pss['detn'].astype(np.int64) - 1
        key_p = hitcache.bunch_key(pss['BunchNumber'], t_p)
        key_w = hitcache.bunch_key(wal['BunchNumber'], t_w)
        key_l = hitcache.bunch_key(liq['BunchNumber'], t_l)
        s_p = step_of[pss['BunchNumber']]

        # ---- per-channel LIQ timing (no plastic tag), late liq
        late_l = (t_l - liq['tflash']) > LATE_TOF
        key_ll, t_ll = key_l[late_l], t_l[late_l]
        for ri, oi in hitcache.iter_pairs(key_ll, key_w, -150, 150, t_ll, t_w):
            dt = t_w[oi] - t_ll[ri]
            db = np.clip(np.digitize(dt, DT_EDGES) - 1, 0, NDT - 1)
            np.add.at(R['walliq_dt'][ai], ((ch_w[oi] - 1), db), 1)
        for ri, oi in hitcache.iter_pairs(key_ll, key_p, -150, 150, t_ll, t_p):
            dt = t_p[oi] - t_ll[ri]
            db = np.clip(np.digitize(dt, DT_EDGES) - 1, 0, NDT - 1)
            np.add.at(R['pssliq_dt'][ai], (bar[oi], db), 1)
        print(f'  timing hists done ({time.time() - t0:.0f}s)', flush=True)

        # ---- wall-plastic pairs, then LIQ matching
        # dt = t_w - t_p; dt_cal window [-W_SIG, SB_HI] -> dt in [min(off)-8, max(off)+120]
        dt_lo, dt_hi = off.min() - W_SIG - 2, off.max() + SB_HI + 2
        n_trip = 0
        for ri, oi in hitcache.iter_pairs(key_p, key_w, dt_lo, dt_hi, t_p, t_w):
            dt_cal = (t_w[oi] - t_p[ri]) - off[ch_w[oi] - 1, bar[ri]]
            wp = np.where(np.abs(dt_cal) <= W_SIG, 0,
                          np.where((dt_cal >= SB_LO) & (dt_cal <= SB_HI), 1, -1))
            keep = wp >= 0
            ri, oi, wp = ri[keep], oi[keep], wp[keep]
            # LIQ candidates around the plastic time (peak-centred window)
            ctr = key_p[ri] + pk[st]
            lo = np.searchsorted(key_l, ctr - WL_SB_HI - 0.1)
            hi = np.searchsorted(key_l, ctr + WL_SB_HI + 0.1)
            cnt = (hi - lo).astype(np.int64)
            tot = int(cnt.sum())
            if tot == 0:
                continue
            e = np.repeat(np.arange(len(ri)), cnt)          # pair index, expanded
            li = np.repeat(lo, cnt) + (np.arange(tot)
                                       - np.repeat(np.cumsum(cnt) - cnt, cnt))
            dt_l = (t_l[li] - t_p[ri[e]]) - pk[st]
            lq = np.where(np.abs(dt_l) <= WL_SIG, 0,
                          np.where((np.abs(dt_l) >= WL_SB_LO)
                                   & (np.abs(dt_l) <= WL_SB_HI), 1, -1))
            k = lq >= 0
            e, li, lq = e[k], li[k], lq[k]
            rie, oie, wpe = ri[e], oi[e], wp[e]
            n_trip += int(((wpe == 0) & (lq == 0)).sum())

            s = s_p[rie]
            ab_p = np.clip(np.digitize(pss['amp'][rie], AMP_EDGES) - 1, 0, NB - 1)
            ab_w = np.clip(np.digitize(wal['amp'][oie], AMP_EDGES) - 1, 0, NB - 1)
            ab_l = np.clip(np.digitize(liq['amp'][li], LAMP_EDGES) - 1, 0, NLB - 1)
            np.add.at(R['pss_amp'], (s, ai, wpe, lq, bar[rie], ab_p), 1)
            np.add.at(R['wal_amp'], (s, ai, wpe, lq, ab_w), 1)
            np.add.at(R['liq_amp'], (s, ai, wpe, lq, ab_l), 1)

            # position map: needs a valid vertical estimator on the wall hit
            lr = lr_all[oie]
            has_v = np.isfinite(lr)
            R['n_novert'][ai] += np.bincount(
                (wpe[~has_v] * 2 + lq[~has_v]), minlength=4).reshape(2, 2)
            hv_ = has_v
            grp = (ch_w[oie[hv_]] - 1) // 2
            vb = np.clip(np.digitize(lr[hv_], LR_EDGES) - 1, 0, NLR - 1)
            np.add.at(R['liq_pos'],
                      (ai, wpe[hv_], lq[hv_], grp, bar[rie[hv_]], vb, ab_l[hv_]), 1)
        print(f'  arm {st}: {n_trip:,} raw sig-sig triples '
              f'({time.time() - t0:.0f}s)', flush=True)

    np.savez_compressed(
        CACHE / f'19_triples_{run_stem}.npz',
        amp_edges=AMP_EDGES, lamp_edges=LAMP_EDGES, lr_edges=LR_EDGES,
        dt_edges=DT_EDGES, s_wp=S_WP, s_lq=S_LQ,
        w_sig=W_SIG, wl_sig=WL_SIG,
        liq_peak=np.array([pk[a] for a in 'ABCD']),
        step_labels=np.array([s[0] for s in hv.STEPS] + ['other']),
        **R)
    print(f'Cached -> {CACHE / f"19_triples_{run_stem}.npz"} '
          f'({time.time() - t0:.0f}s)')

    # quick look: inclusive double-subtracted plastic spectra per PMT
    cen = np.sqrt(AMP_EDGES[:-1] * AMP_EDGES[1:])
    print('\nInclusive triple-tagged plastic spectra (double sideband-subtracted):')
    print(f'{"PMT":8s} {"n_triple":>10s} {"median":>8s} {"mode":>8s}  (ADC)')
    kern = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
    kern /= kern.sum()
    for ai, st in enumerate('ABCD'):
        for b in range(2):
            h = R['pss_amp'][:, ai, :, :, b].sum(axis=0)
            sub = (h[0, 0] - S_LQ * h[0, 1] - S_WP * h[1, 0]
                   + S_WP * S_LQ * h[1, 1])
            n = sub.sum()
            c = np.cumsum(np.clip(sub, 0, None))
            med = cen[np.searchsorted(c, c[-1] / 2)] if c[-1] > 100 else np.nan
            sm = np.convolve(sub, kern, mode='same')
            msk = cen > 150
            mode = cen[msk][np.argmax(sm[msk])] if n > 500 else np.nan
            print(f'PSS{st}{b + 1}   {n:10.0f} {med:8.0f} {mode:8.0f}')


if __name__ == '__main__':
    main()
