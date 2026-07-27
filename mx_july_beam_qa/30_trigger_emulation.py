"""30_trigger_emulation.py — Emulate the hardware wall "singles" trigger from the
recorded n_TOF hit-level data, and produce singles/wall vs time (averaged/event).

Emulates the N1081B trigger chain (see nTof_x17_DAQ/n1081b/n1081b_module_map.py):

  M1  per wall X, per bar-segment g (g=0..3): the 428F linear fan-in forms the
      analog SUM top+bottom = amp(detn 2g+1) + amp(detn 2g+2) (mV). Discriminate
      each segment sum at WALL_THR[X]; OR(seg0..3) -> "wall X fired".
  M2  per wall X: each plastic PMT (PSSX detn 1,2) discriminated at PLA_THR[X];
      OR -> "plastic X fired". Wall D uses PSSD2 only (D-L / PSSD1 input is broken).
  M3  per wall X: wall X AND plastic X within a 20 ns coincidence window
      (hardware = 20 ns input gate&delay on both legs) -> sector-X trigger =
      the per-wall "SINGLES" count.

Every discriminated signal becomes a 20 ns logic pulse; the OR is a union of
pulses and the AND is their overlap, so pulses closer than 20 ns MERGE (dead
time) exactly as in hardware. Wall/plastic legs are aligned by the measured
per-wall coincidence-peak offset (the hardware +20 ns wall-leg delay).

Thresholds = current standing config adopted 2026-07-19 (run224524 is post-recal):
  walls   half-MIP  (daq/calibrations/wal_trigger/thresholds_halfMIP_run224503.json)
  plastic 0.5-MIP   (daq/calibrations/pss/mip_thresholds_y88.json, per-arm)

Output: cache/30_trigemul_<run>.npz + figures/30_trigger/singles_vs_time_<run>.png
Usage:  python 30_trigger_emulation.py [run_file]
"""
import sys
from pathlib import Path

import numpy as np

import hitcache
from adc_mv import mv_factors

BASE = Path(__file__).parent
CACHE = BASE / 'cache'
OUT = BASE / 'figures' / '30_trigger'
OUT.mkdir(parents=True, exist_ok=True)

RUN_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'x17/beam_july/data/run224524.root'
STEM = RUN_FILE.stem

# ---- trigger constants (digitizer-equivalent mV; treated ~1:1 with hardware) ----
WALL_THR = {'A': 25.0, 'B': 35.3, 'C': 33.5, 'D': 36.0}   # per-wall, on top+bottom segment SUM
PLA_THR = {'A': 65.0, 'B': 78.0, 'C': 86.0, 'D': 83.0}    # per-arm, on single-PMT amplitude
D_PMTS = {'A': [1, 2], 'B': [1, 2], 'C': [1, 2], 'D': [2]}  # wall D: PSSD1 (D-L) input broken
TB_MAX = 15.0        # ns, top-bottom analog-sum match window (per 23f_sipm_sums)
PULSE = 20.0         # ns, discriminated logic-pulse width == coincidence window
OFF_WIN = 80.0       # ns, search window for the wall<->plastic timing offset
SIDEBAND = 500.0     # ns, plastic-leg delay for the accidental-coincidence estimate

# ---- time axis (tof = time since acquisition start; gamma flash ~ 11 us) ----
TOF_LO, TOF_HI = 1e3, 2.0e7                        # 1 us .. 20 ms
TOF_EDGES = np.geomspace(TOF_LO, TOF_HI, 24 * 8 + 1)   # ~24 log bins/decade
TOF_CEN = np.sqrt(TOF_EDGES[:-1] * TOF_EDGES[1:])
TOF_W_US = np.diff(TOF_EDGES) / 1e3                # bin width in us


def select_good_bunches(run_file):
    """Beam-on bunches: PulseIntensity above half the median of active bunches."""
    inten = hitcache.bunch_intensity(run_file)          # sorted by bunch number
    bunch_no = np.arange(1, len(inten) + 1)
    active = inten[inten > 0]
    thr = 0.5 * np.median(active) if len(active) else 0.0
    good = bunch_no[inten > thr]
    return good, inten


def merge_pulses(t_sorted, w):
    """Union of [t, t+w] over sorted times -> (starts, ends). Equal widths, so a
    new pulse begins wherever the gap to the previous fire exceeds w."""
    if len(t_sorted) == 0:
        return np.empty(0), np.empty(0)
    new = np.ones(len(t_sorted), bool)
    new[1:] = np.diff(t_sorted) > w
    starts = t_sorted[new]
    idx = np.flatnonzero(new)
    ends = np.maximum.reduceat(t_sorted, idx) + w       # last (max) time in group + w
    return starts, ends


def or_edges(bunch_sorted, t_sorted, w):
    """Leading edges of the OR pulse train: a (bunch,tof)-sorted fire list ->
    the times of the merged 20 ns output pulses (new pulse at each bunch change
    or gap > w). Fully vectorized (== per-bunch merge_pulses starts)."""
    if len(t_sorted) == 0:
        return np.empty(0)
    new = np.ones(len(t_sorted), bool)
    new[1:] = (bunch_sorted[1:] != bunch_sorted[:-1]) | (np.diff(t_sorted) > w)
    return t_sorted[new]


def intersect(ws, we, ps, pe):
    """Leading edges of the AND (overlap) of two interval sets sorted by start."""
    out = []
    i = j = 0
    nW, nP = len(ws), len(ps)
    while i < nW and j < nP:
        lo = ws[i] if ws[i] > ps[j] else ps[j]
        hi = we[i] if we[i] < pe[j] else pe[j]
        if lo < hi:
            out.append(lo)
        if we[i] < pe[j]:
            i += 1
        else:
            j += 1
    return out


def wall_fire_times(run_file, arm, good):
    """(bunch, t) of every M1 'wall fired' pulse: a bar-segment whose top+bottom
    analog sum crosses WALL_THR[arm], time = 0.5*(t_top+t_bot)."""
    fac = mv_factors(run_file)[f'WAL{arm}']
    d = hitcache.load(run_file, f'WAL{arm}', ['BunchNumber', 'tof', 'detn', 'amp'], good)
    mv = d['amp'] * fac[(d['detn'] - 1).astype(int)]
    key = hitcache.bunch_key(d['BunchNumber'], d['tof'])
    b_all, t_all = [], []
    for g in range(4):
        it = np.flatnonzero(d['detn'] == 2 * g + 1)
        ib = np.flatnonzero(d['detn'] == 2 * g + 2)
        if not len(it) or not len(ib):
            continue
        # nearest bottom hit within +-TB_MAX for each top hit
        best_dt = np.full(len(it), np.inf)
        best_ob = np.full(len(it), -1, np.int64)
        for ri, oi in hitcache.iter_pairs(key[it], key[ib], -TB_MAX, TB_MAX,
                                          d['tof'][it], d['tof'][ib]):
            adt = np.abs(d['tof'][ib][oi] - d['tof'][it][ri])
            better = adt < best_dt[ri]
            best_dt[ri[better]] = adt[better]
            best_ob[ri[better]] = oi[better]
        m = best_ob >= 0
        i_t = it[m]
        i_b = ib[best_ob[m]]
        s = mv[i_t] + mv[i_b]
        fire = s > WALL_THR[arm]
        b_all.append(d['BunchNumber'][i_t][fire])
        t_all.append(0.5 * (d['tof'][i_t][fire] + d['tof'][i_b][fire]))
    if not b_all:
        return np.empty(0, np.int64), np.empty(0)
    b = np.concatenate(b_all)
    t = np.concatenate(t_all)
    o = np.lexsort((t, b))
    return b[o], t[o]


def plastic_fire_times(run_file, arm, good):
    """(bunch, t) of every M2 'plastic fired' pulse: a PMT hit above PLA_THR[arm]."""
    fac = mv_factors(run_file)[f'PSS{arm}']
    d = hitcache.load(run_file, f'PSS{arm}', ['BunchNumber', 'tof', 'detn', 'amp'], good)
    mv = d['amp'] * fac[(d['detn'] - 1).astype(int)]
    sel = np.isin(d['detn'], D_PMTS[arm]) & (mv > PLA_THR[arm])
    b, t = d['BunchNumber'][sel], d['tof'][sel]
    o = np.lexsort((t, b))
    return b[o], t[o]


def measure_offset(wb, wt, pb, pt):
    """Peak of (t_plastic - t_wall) over near pairs -> hardware leg alignment."""
    kw = hitcache.bunch_key(wb, wt)
    kp = hitcache.bunch_key(pb, pt)
    edges = np.arange(-OFF_WIN, OFF_WIN + 1, 1.0)
    h = np.zeros(len(edges) - 1)
    for ri, oi in hitcache.iter_pairs(kw, kp, -OFF_WIN, OFF_WIN, wt, pt):
        h += np.histogram(pt[oi] - wt[ri], bins=edges)[0]
    cen = 0.5 * (edges[:-1] + edges[1:])
    return float(cen[np.argmax(h)]), cen, h


def group_bounds(bunch_sorted):
    """{bunch: (lo, hi)} slice bounds into a (bunch,tof)-sorted array."""
    ub, lo = np.unique(bunch_sorted, return_index=True)
    hi = np.append(lo[1:], len(bunch_sorted))
    return dict(zip(ub.tolist(), zip(lo.tolist(), hi.tolist())))


def main():
    good, inten = select_good_bunches(RUN_FILE)
    n_ev = len(good)
    print(f'{STEM}: {n_ev} good (beam-on) bunches of {len(inten)}', flush=True)

    # gamma-flash time (per-run median tflash) -> lets downstream express the
    # singles time as "time past the flash" for the sim comparison.
    ftf = hitcache.load(RUN_FILE, 'PSSA', ['tflash'], good)['tflash']
    flash_ns = float(np.median(ftf[ftf > 0]))
    print(f'  flash time (median tflash) = {flash_ns / 1e3:.2f} us', flush=True)

    singles_hist = {}       # arm -> AND counts vs tof
    singles_tof = {}        # arm -> raw AND (coincidence) times (tof, ns)
    singles_acc_tof = {}    # arm -> accidental AND times (plastic delayed +SIDEBAND)
    wallor_tof = {}         # arm -> SiPM-wall OR output-pulse times (M1 singles)
    plasor_tof = {}         # arm -> plastic OR output-pulse times (M2 singles)
    offsets = {}
    per_ev = {}
    acc_ev = {}
    for arm in 'ABCD':
        wb, wt = wall_fire_times(RUN_FILE, arm, good)
        pb, pt = plastic_fire_times(RUN_FILE, arm, good)
        off, _, _ = measure_offset(wb, wt, pb, pt)
        offsets[arm] = off
        pt_al = pt - off                              # align plastic into wall frame
        print(f'  WAL{arm}: {len(wt):,} wall fires, {len(pt):,} plastic fires, '
              f'offset {off:+.0f} ns', flush=True)

        # per-detector singles = 20 ns OR-merged output pulses (M1 wall, M2 plastic)
        wallor_tof[arm] = or_edges(wb, wt, PULSE).astype(np.float32)
        plasor_tof[arm] = or_edges(pb, pt, PULSE).astype(np.float32)

        # coincidence (M3 AND) = overlap of the two 20 ns OR pulse trains
        # real coincidence, plus an accidental estimate with the plastic OR pulse
        # train rigidly delayed by SIDEBAND (>> 20 ns correlation, << ms rate scale)
        wgb = group_bounds(wb)
        pgb = group_bounds(pb)
        edges_hit, edges_acc = [], []
        for bunch, (wl, wh) in wgb.items():
            pb_ = pgb.get(bunch)
            if pb_ is None:
                continue
            pl, ph = pb_
            ws, we = merge_pulses(wt[wl:wh], PULSE)
            ps, pe = merge_pulses(pt_al[pl:ph], PULSE)
            edges_hit.extend(intersect(ws, we, ps, pe))
            edges_acc.extend(intersect(ws, we, ps + SIDEBAND, pe + SIDEBAND))
        edges_hit = np.asarray(edges_hit)
        edges_acc = np.asarray(edges_acc)
        singles_tof[arm] = edges_hit.astype(np.float32)
        singles_acc_tof[arm] = (edges_acc - SIDEBAND).astype(np.float32)   # back to real tof
        singles_hist[arm] = np.histogram(edges_hit, bins=TOF_EDGES)[0].astype(float)
        per_ev[arm] = len(edges_hit) / n_ev
        acc_ev[arm] = len(edges_acc) / n_ev
        print(f'    -> {len(edges_hit):,} singles ({per_ev[arm]:.1f}/ev), '
              f'accidental {acc_ev[arm]:.1f}/ev ({100*acc_ev[arm]/max(per_ev[arm],1e-9):.0f}%); '
              f'wall-OR {len(wallor_tof[arm]):,}, plastic-OR {len(plasor_tof[arm]):,}',
              flush=True)

    np.savez_compressed(CACHE / f'30_trigemul_{STEM}.npz',
                        tof_edges=TOF_EDGES, n_ev=n_ev, flash_ns=flash_ns,
                        offsets=np.array([offsets[a] for a in 'ABCD']),
                        per_ev=np.array([per_ev[a] for a in 'ABCD']),
                        **{f'singles_{a}': singles_hist[a] for a in 'ABCD'},
                        **{f'singles_tof_{a}': singles_tof[a] for a in 'ABCD'},
                        **{f'singles_acc_tof_{a}': singles_acc_tof[a] for a in 'ABCD'},
                        **{f'wallor_tof_{a}': wallor_tof[a] for a in 'ABCD'},
                        **{f'plasor_tof_{a}': plasor_tof[a] for a in 'ABCD'})
    print(f'\nCached -> cache/30_trigemul_{STEM}.npz')
    figure(singles_hist, per_ev, offsets, n_ev)


def figure(singles_hist, per_ev, offsets, n_ev):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c', 'D': '#d62728'}
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for arm in 'ABCD':
        rate = singles_hist[arm] / n_ev / TOF_W_US          # singles per event per us
        ax.plot(TOF_CEN / 1e3, rate, drawstyle='steps-mid', color=colors[arm],
                label=f'WAL{arm}: {per_ev[arm]:.1f} singles/event '
                      f'(thr {WALL_THR[arm]:.0f}/{PLA_THR[arm]:.0f} mV, off {offsets[arm]:+.0f} ns)')
    ax.axvspan(10.8, 11.9, color='0.85', zorder=0, label='gamma flash')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('time since acquisition start  [us]')
    ax.set_ylabel('emulated singles per event per us')
    ax.set_title(f'Emulated wall SINGLES trigger vs time — {STEM}  '
                 f'({n_ev} beam-on bunches)\nM1(sum>thr).OR . AND . M2(PMT>thr).OR, '
                 f'20 ns logic pulses')
    ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    p = OUT / f'singles_vs_time_{STEM}.png'
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f'-> {p}')


if __name__ == '__main__':
    main()
