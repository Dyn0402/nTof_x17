#!/usr/bin/env python3
"""Gamma-flash response of the micromegas channel in the n_TOF DAQ.

Inputs are the merged per-run products built by `extract_mm.py` + `merge_mm.py`
from /eos/experiment/ntof/data/x17/mm_raw_2026-07 (see NTOF_MICROMEGAS_SIGNALS.md).

Everything downstream of the raw obeys the stream1 sample semantics:
signed int16, ZS fill code -32768, 259 pre-samples, 1 sample = 1 ns.

Writes results.json + figures/ next to this file.
"""
import json
import os
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
DATA = pathlib.Path('/media/dylan/data/x17/ntof_mm_flash')
FIG = HERE / 'figures'

# --- calibration, from the runs' own DAQsettings -------------------------------
# fullScalemV over ADCrange=65536 codes; baseline parked at +200 mV, negative
# rail at -252 mV, so the largest measurable negative excursion is 452 mV.
MV_PER_COUNT = {'MMA': 504.149 / 65536, 'MMB': 503.637 / 65536}
BASELINE_MV = 200.0
NEG_RAIL_MV = -504.149 / 2
HEADROOM_MV = BASELINE_MV - NEG_RAIL_MV
ZS_THR_MV = {'MMA': 0.01, 'MMB': 4.0}
R_TERM = 50.0                       # ohms, assumed direct termination

DEC = 10                            # stored trace is decimated 10x (10 ns bins)
RUNS = [
    dict(run=224302, live='MMB', label='224302', beam=True,
         dream='run_12 (partial)', gas='Ar/CF4/Iso 88/10/2',
         span='2026-07-05 22:32 - 07-06 01:14'),
    dict(run=224325, live='MMA', label='224325', beam=False,
         dream='none (gap between run_17 and run_18)', gas='Ar/Iso 95/5',
         span='2026-07-08 20:14 - 21:42'),
    dict(run=224327, live='MMA', label='224327', beam=True,
         dream='run_18 (full)', gas='Ar/Iso 95/5',
         span='2026-07-09 00:34 - 03:38'),
]
COL = {224302: '#2f6f9f', 224325: '#7a7a7a', 224327: '#c0632c'}


def load(run):
    return np.load(DATA / f'mm_{run}.npz')


def align(d, live, key):
    """Per-bunch scalar aligned to the index-tree bunch order."""
    vb = d[f'{live}_{key}_bunch']
    v = d[f'{live}_{key}']
    order = np.argsort(vb)
    return vb[order], v[order]


def bunch_intensity(d):
    ib = d['idx_bunch']
    o = np.argsort(ib)
    return ib[o], d['idx_intensity'][o] / 1e12, d['idx_secs'][o]


def recovery(trace, mv_per_count):
    """Per-bunch time at which the trace last exceeds the 4 mV ZS threshold,
    i.e. when a normal-sized hit would again stand clear of the flash tail."""
    thr = 4.0 / mv_per_count
    t_ns = np.arange(trace.shape[1]) * DEC
    out = np.full(len(trace), np.nan)
    for i, tr in enumerate(trace):
        over = np.flatnonzero(np.nan_to_num(tr) > thr)
        if over.size:
            out[i] = t_ns[over[-1]]
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    res = {'calibration': {
        'mv_per_count': MV_PER_COUNT, 'baseline_mV': BASELINE_MV,
        'neg_rail_mV': NEG_RAIL_MV, 'headroom_mV': HEADROOM_MV,
        'termination_ohm': R_TERM,
        'fC_per_count_ns': {k: v * 1e-3 / R_TERM * 1e15 * 1e-9
                            for k, v in MV_PER_COUNT.items()}},
        'runs': {}}

    # ---------------- figure 1: mean flash waveform ---------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    traces = {}
    for r in RUNS:
        d = load(r['run'])
        live = r['live']
        tr = d[f'{live}_trace']
        mv = MV_PER_COUNT[live]
        t = np.arange(tr.shape[1]) * DEC / 1000.0          # us
        mt = np.nanmean(tr, axis=0) * mv
        traces[r['run']] = (t, mt)
        ax[0].plot(t, mt, color=COL[r['run']], lw=1.2,
                   label=f"{r['label']} {live}" + ('' if r['beam'] else ' (beam off)'))
        ax[1].plot(t, mt, color=COL[r['run']], lw=1.2)
    ax[0].set_xlabel('time in acquisition window (us)')
    ax[0].set_ylabel('mean negative-going signal (mV)')
    ax[0].set_title('Gamma flash, bunch-averaged')
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)
    ax[1].set_xlim(11.5, 30)
    ax[1].set_ylim(-3, 8)
    ax[1].axhline(0, color='k', lw=.6)
    ax[1].axhline(4.0, color='crimson', ls=':', lw=1,
                  label='4 mV zero-suppression threshold')
    ax[1].set_xlabel('time in acquisition window (us)')
    ax[1].set_ylabel('mean signal (mV)')
    ax[1].set_title('Tail, zoomed')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / 'flash_waveform.png', dpi=140)
    plt.close(fig)

    # ---------------- per-run scalars ----------------------------------------
    for r in RUNS:
        run, live = r['run'], r['live']
        d = load(run)
        mv = MV_PER_COUNT[live]
        pb, peak = align(d, live, 'peak')
        _, integ = align(d, live, 'integral')
        _, npos = align(d, live, 'n_pos_rail')
        _, nneg = align(d, live, 'n_neg_rail')
        ib, inten, secs = bunch_intensity(d)
        pos = np.searchsorted(ib, pb)
        pos = np.clip(pos, 0, len(ib) - 1)
        inten_b = inten[pos]
        secs_b = secs[pos]

        peak_mv = peak * mv
        # charge from the stored (decimated) trace over the flash lobe
        tr = d[f'{live}_trace']
        tb = d[f'{live}_trace_bunch']
        o = np.argsort(tb)
        tr = tr[o]
        lobe = np.nansum(tr[:, 1100:2000], axis=1) * DEC          # counts*ns
        # counts*ns -> mV*ns -> V*s -> C -> pC
        q_pc = lobe * mv * 1e-3 * 1e-9 / R_TERM * 1e12            # pC
        rec = recovery(tr, mv)

        sat = int((nneg > 0).sum())
        info = dict(
            n_bunch=int(len(peak)), live=live, beam=bool(r['beam']),
            dream=r['dream'], gas=r['gas'], span=r['span'],
            zs_threshold_mV=ZS_THR_MV[live],
            peak_mV=dict(p1=float(np.percentile(peak_mv, 1)),
                         p50=float(np.median(peak_mv)),
                         p99=float(np.percentile(peak_mv, 99)),
                         max=float(peak_mv.max())),
            charge_pC=dict(p50=float(np.median(q_pc)),
                           p99=float(np.percentile(q_pc, 99))),
            railed_bunches=sat,
            railed_fraction=float(sat / len(peak)),
            headroom_used_p99=float(np.percentile(peak_mv, 99) / HEADROOM_MV),
            recovery_to_4mV_ns=dict(
                p50=float(np.nanmedian(rec)), p99=float(np.nanpercentile(rec, 99))),
            peak_time_ns=float(np.median(align(d, live, 'peak_t')[1])),
            intensity_1e12=dict(p50=float(np.median(inten_b)),
                                frac_zero=float((inten_b < 0.1).mean())),
        )

        # dedicated vs parasitic
        if r['beam']:
            hi = inten_b > 6.0
            lo = (inten_b > 2.0) & (inten_b < 6.0)
            if hi.sum() > 20 and lo.sum() > 20:
                info['dedicated'] = dict(
                    n=int(hi.sum()), intensity=float(np.median(inten_b[hi])),
                    peak_mV=float(np.median(peak_mv[hi])),
                    charge_pC=float(np.median(q_pc[hi])))
                info['parasitic'] = dict(
                    n=int(lo.sum()), intensity=float(np.median(inten_b[lo])),
                    peak_mV=float(np.median(peak_mv[lo])),
                    charge_pC=float(np.median(q_pc[lo])))
                info['charge_ratio_ded_par'] = (
                    info['dedicated']['charge_pC'] / info['parasitic']['charge_pC'])
                info['intensity_ratio'] = (
                    info['dedicated']['intensity'] / info['parasitic']['intensity'])

        # post-flash zero-suppressed block rate
        z = d[f'{live}_zs']
        edges = np.array([3e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7])
        h, _ = np.histogram(z[:, 1], bins=edges)
        rate = h / np.diff(edges) * 1e6 / len(peak)          # blocks per bunch per ms
        info['zs_rate_per_bunch_per_ms'] = [
            dict(t_lo_ms=float(edges[i] / 1e6), t_hi_ms=float(edges[i + 1] / 1e6),
                 rate=float(rate[i])) for i in range(len(h))]
        info['zs_first_block_ns'] = float(z[:, 1].min())
        info['zs_blocks_per_bunch'] = float(len(z) / len(peak))
        res['runs'][run] = info

        np.savez_compressed(DATA / f'perbunch_{run}.npz', bunch=pb, peak_mv=peak_mv,
                            charge_pc=q_pc, intensity=inten_b, secs=secs_b,
                            recov=rec, nneg=nneg, npos=npos)

    # ---------------- figure 2: post-flash rate ------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for r in RUNS:
        info = res['runs'][r['run']]
        xs, ys = [], []
        for b in info['zs_rate_per_bunch_per_ms']:
            xs += [b['t_lo_ms'], b['t_hi_ms']]
            ys += [b['rate'], b['rate']]
        ax.plot(xs, ys, color=COL[r['run']], lw=1.6,
                label=f"{r['label']} {r['live']}"
                      + ('' if r['beam'] else ' (beam off)')
                      + f"  thr {info['zs_threshold_mV']} mV")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('time after the proton pulse (ms)')
    ax.set_ylabel('zero-suppressed blocks per bunch per ms')
    ax.set_title('The channel keeps recording from the first instant it is allowed to')
    ax.axvline(0.03, color='k', ls='--', lw=.8)
    ax.text(0.031, ax.get_ylim()[0] * 1.4, ' end of the mandatory 30 us block',
            fontsize=7.5, rotation=90, va='bottom')
    ax.grid(alpha=.3, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'post_flash_rate.png', dpi=140)
    plt.close(fig)

    # ---------------- figure 3: individual flash traces ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, r in zip(axes, [RUNS[0], RUNS[2]]):
        d = load(r['run'])
        live = r['live']
        mv = MV_PER_COUNT[live]
        tr = d[f'{live}_trace']
        tb = d[f'{live}_trace_bunch']
        o = np.argsort(tb)
        tr = tr[o]
        ib, inten, _ = bunch_intensity(d)
        pos = np.clip(np.searchsorted(ib, np.sort(tb)), 0, len(ib) - 1)
        it = inten[pos]
        t = np.arange(tr.shape[1]) * DEC / 1000.0
        for sel, c, lab in ((it > 6, '#c0392b', 'dedicated'),
                            ((it > 2) & (it < 6), '#2f6f9f', 'parasitic')):
            idx = np.flatnonzero(sel)[:6]
            for j, i in enumerate(idx):
                ax.plot(t, tr[i] * mv, color=c, lw=.7, alpha=.65,
                        label=lab if j == 0 else None)
        ax.axhline(4.0, color='k', ls=':', lw=1)
        ax.set_xlim(11, 20)
        ax.set_xlabel('time in acquisition window (us)')
        ax.set_ylabel('negative-going signal (mV)')
        ax.set_title(f"{r['label']} {live} - single bunches")
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / 'single_traces.png', dpi=140)
    plt.close(fig)

    # ---------------- HV response --------------------------------------------
    import csv as _csv
    import datetime as _dt

    def _secs(ts):
        t = _dt.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
        return ((t.month - 7) * 31 + (t.day - 1)) * 86400 + \
            t.hour * 3600 + t.minute * 60 + t.second

    hvjobs = [(224302, 'hv_224302_run_12.csv'), (224327, 'hv_224327_run_18.csv')]
    res['hv'] = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (run, fname) in zip(axes, hvjobs):
        pb = np.load(DATA / f'perbunch_{run}.npz')
        rows = list(_csv.DictReader(open(DATA / fname)))
        hts = np.array([_secs(r['timestamp']) for r in rows])
        o = np.argsort(hts)
        hts, rows = hts[o], [rows[i] for i in o]
        idx = np.clip(np.searchsorted(hts, pb['secs']), 0, len(hts) - 1)
        vr = np.array([float(rows[i]['A_resist_v'] or 'nan') for i in idx])
        vd = np.array([float(rows[i]['A_drift_v'] or 'nan') for i in idx])
        inside = (pb['secs'] >= hts.min()) & (pb['secs'] <= hts.max())
        q, inten = pb['charge_pc'], pb['intensity']
        table = []
        for cls, sel, mkr in (('dedicated', inten > 6, 'o'),
                              ('parasitic', (inten > 2) & (inten < 6), 's')):
            for drift in sorted(set(np.round(vd[np.isfinite(vd)]))):
                xs, ys, ns = [], [], []
                for v in sorted(set(np.round(vr[np.isfinite(vr)]))):
                    m = sel & inside & (np.round(vd) == drift) & (np.round(vr) == v)
                    if m.sum() < 15:
                        continue
                    xs.append(v)
                    ys.append(float(np.median(q[m])))
                    ns.append(int(m.sum()))
                    table.append(dict(cls=cls, drift_V=float(drift), resist_V=float(v),
                                      n=int(m.sum()), charge_pC=float(np.median(q[m])),
                                      peak_mV=float(np.median(pb['peak_mv'][m]))))
                if len(xs) > 1:
                    ax.plot(xs, ys, marker=mkr, lw=1.3,
                            label=f'{cls}, drift {drift:.0f} V')
        res['hv'][run] = table
        ax.set_yscale('log')
        ax.set_xlabel('amplification (resistive) voltage, common ladder (V)')
        ax.set_ylabel('median flash charge (pC, into 50 ohm)')
        ax.set_title(f'{run}')
        ax.grid(alpha=.3, which='both')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'charge_vs_hv.png', dpi=140)
    plt.close(fig)

    # ---------------- intensity linearity ------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    res['linearity'] = {}
    for run in (224302, 224327):
        tab = res['hv'][run]
        pts = {}
        for row in tab:
            key = (row['drift_V'], row['resist_V'])
            pts.setdefault(key, {})[row['cls']] = row['charge_pC']
        xs, ys = [], []
        for key, v in sorted(pts.items()):
            if 'dedicated' in v and 'parasitic' in v and v['parasitic'] > 0:
                xs.append(v['parasitic'])
                ys.append(v['dedicated'] / v['parasitic'])
        if xs:
            ax.plot(xs, ys, 'o-', color=COL[run], label=f'{run}')
            res['linearity'][run] = [dict(parasitic_charge_pC=x, ratio=y)
                                     for x, y in zip(xs, ys)]
    ax.axhline(2.05, color='k', ls='--', lw=1,
               label='proton-intensity ratio 2.05 (proportional response)')
    ax.axhline(1.0, color='grey', ls=':', lw=1, label='no intensity dependence')
    ax.set_xscale('log')
    ax.set_xlabel('median flash charge on parasitic pulses (pC) - a proxy for gain')
    ax.set_ylabel('charge ratio, dedicated / parasitic')
    ax.set_title('The flash response compresses as the signal grows')
    ax.grid(alpha=.3, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'intensity_linearity.png', dpi=140)
    plt.close(fig)

    # ---------------- conditions, from the DREAM run_config.json -------------
    res['conditions'] = {
        224302: dict(span='2026-07-05 22:32 - 07-06 01:14 (2.70 h)',
                     beam='on: 1741 dedicated (8.50e12) + 1270 parasitic (4.14e12), 56 empty',
                     dream='run_12, covering only 00:07-01:14 (1256 of 3067 bunches); '
                           'no DREAM run before that',
                     gas='Ar/CF4/Iso 88/10/2', trigger='PS Pickup',
                     hv='drift 800 V fixed; amplification ladder 525 -> 540 V'),
        224325: dict(span='2026-07-08 20:14 - 21:42 (1.47 h)',
                     beam='OFF: PulseIntensity 0 for all 5277 bunches, PKUP flat, lsaCycle -1',
                     dream='none - falls in the gap between run_17 (ended 18:18) '
                           'and run_18 (started 23:46)',
                     gas='Ar/Iso 95/5 (from the runs either side)', trigger='n/a',
                     hv='not monitored during this window'),
        224327: dict(span='2026-07-09 00:34 - 03:38 (3.06 h)',
                     beam='on: 1499 dedicated (8.51e12) + 1907 parasitic (4.14e12), 38 empty',
                     dream='run_18, fully contained (23:46 07-08 -> 08:58 07-09)',
                     gas='Ar/Iso 95/5',
                     trigger='External TCM/N1081B: gamma flash on each 1.2 s cycle '
                             '+ random trigger within 30 ms after the flash',
                     hv='drift 600 then 800 V; amplification ladder 465 -> 480 V'),
    }

    # ---------------- integrity: signed decode, no wrap, no clipping ---------
    # produced by wrapcheck.py over the first 12 raw files of each beam run
    res['integrity'] = dict(
        sample_type='int16 (ntoflib ReaderStructACQC.h declares std::vector<int16_t>)',
        checked_samples=56_647_470,
        samples_at_positive_rail=0,
        samples_at_negative_rail_or_fill=0,
        sample_to_sample_jumps_over_20000=0,
        largest_jump_counts=5883,
        railed_bunches={'224302': 4, '224325': 0, '224327': 1},
        note='A wrap would show as a >20000-count step between adjacent samples '
             '(the signature used in liq_study/adc_range_census.py). None occur.')

    # HV states are fully degenerate across the four chambers
    res['chamber_identity'] = dict(
        determinable=False,
        reason='All four MX17 chambers step in lockstep: every HV state in run_18 is a '
               'joint (A,B,C,D) setting and each ladder spans the same 20 V, so the '
               'flash charge correlates identically with all four. The response does '
               'follow the common ladder exponentially, so the channel is on one of '
               'the four chambers - but which one cannot be decided from this data.')

    with open(HERE / 'results.json', 'w') as f:
        json.dump(res, f, indent=1)
    print('figures:', sorted(p.name for p in FIG.glob('*.png')))
    print(json.dumps(res['linearity'], indent=1))


if __name__ == '__main__':
    main()
