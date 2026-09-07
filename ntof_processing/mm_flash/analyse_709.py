#!/usr/bin/env python3
"""Detector-A drift x amplification scan seen on the n_TOF micromegas channel.

Run 224709, 2026-08-09 17:05-19:38. MMA = strip 32 of MX17 detector A, cable Y8.
Only detector A is scanned, so unlike the July runs the response is attributable
to a named chamber.

Consumes /media/dylan/data/x17/ntof_mm_flash/mm_224709.npz (from merge_709.py)
and hv_plateaus_224709.csv. Writes results_709.json + figures/.
"""
import csv
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
DATA = pathlib.Path('/media/dylan/data/x17/ntof_mm_flash')
FIG = HERE / 'figures'

# This run's own calibration, from the MODH channel record:
# fullScale 5043.7915 mV over 65536 codes -- 10x coarser than the July runs.
MV_PER_COUNT = 5043.7915 / 65536
FC_PER_COUNT_NS = MV_PER_COUNT * 1e-3 * 1e-9 / 50.0 * 1e15
ZS_FILL = -32768
SETTLE_S = 45           # discard this much after each HV change
FLASH_LO, FLASH_HI = 11_000, 20_000     # charge integration window, ns


def load():
    d = np.load(DATA / 'mm_224709.npz')
    wall = d['wall']
    hh = wall[:, 0] // 10000
    mm = (wall[:, 0] // 100) % 100
    ss = wall[:, 0] % 100
    secs = hh * 3600 + mm * 60 + ss
    return d, secs


def plateaus():
    rows = []
    with open(DATA / 'hv_plateaus_224709.csv') as f:
        for r in csv.DictReader(f):
            rows.append(dict(start=int(r['start_s']), end=int(r['end_s']),
                             drift=int(r['A_drift_V']), resist=int(r['A_resist_V']),
                             imon=float(r['A_resist_imon_med']),
                             label=r['start'][11:16]))
    return rows


def main():
    d, secs = load()
    flash = d['flash'].astype(np.int32)
    stats = d['stats']
    ev = stats[:, 0].astype(int)
    base = stats[:, 1]
    peak_ct = stats[:, 2]
    peak_t = stats[:, 3]
    n_pos = stats[:, 5]
    n_neg = stats[:, 6]

    # deviation traces, baseline-subtracted, fill masked
    dev = base[:, None] - flash[ev]
    dev[flash[ev] <= ZS_FILL] = 0.0

    charge_ct_ns = dev[:, FLASH_LO:FLASH_HI].sum(axis=1)
    charge_pc = charge_ct_ns * MV_PER_COUNT * 1e-3 * 1e-9 / 50.0 * 1e12
    peak_mv = peak_ct * MV_PER_COUNT

    # Recovery: last sample above a threshold that is meaningful on THIS range.
    # 4 mV is only 52 counts here (the July runs were 10x more sensitive), which
    # sits inside the single-sample noise, so a bare 4 mV cut just finds the last
    # noise excursion. Use max(4 mV, 5 sigma) with sigma from the pre-flash region.
    sigma_ct = dev[:, :10_000].std(axis=1)
    thr_ct = np.maximum(4.0 / MV_PER_COUNT, 5 * sigma_ct)
    rec = np.full(len(dev), np.nan)
    for i in range(len(dev)):
        o = np.flatnonzero(dev[i] > thr_ct[i])
        if o.size:
            rec[i] = o[-1]

    # beam intensity proxy: PKUP flash peak (0.996 correlated with PulseIntensity
    # on the July runs, where both were available)
    pk = np.zeros(len(d['bunch']))
    pkup = d['pkup']
    pk[pkup[:, 0].astype(int)] = pkup[:, 1]
    pk_ev = pk[ev]
    # The distribution is cleanly bimodal with an empty gap between the two
    # proton-intensity classes; a handful of bunches carry no beam at all.
    valid = pk_ev > 1000
    split = 0.5 * (np.percentile(pk_ev[valid], 10) + np.percentile(pk_ev[valid], 90))
    dedicated = valid & (pk_ev > split)
    parasitic = valid & ~dedicated
    intensity_ratio = float(np.median(pk_ev[dedicated]) / np.median(pk_ev[parasitic]))

    res = dict(
        run=224709,
        channel='MMA = strip 32 of MX17 detector A, cable Y8',
        calibration=dict(full_scale_mV=5043.7915, mV_per_count=MV_PER_COUNT,
                         fC_per_count_ns=FC_PER_COUNT_NS,
                         baseline_counts=float(np.median(base)),
                         baseline_mV=float(np.median(base) * MV_PER_COUNT)),
        n_bunch=int(len(ev)),
        settle_s=SETTLE_S,
        wall_span=[int(secs.min()), int(secs.max())],
        pkup_split=float(split),
        n_dedicated=int(dedicated.sum()), n_parasitic=int(parasitic.sum()),
        n_no_beam=int((~valid).sum()), intensity_ratio_from_pkup=intensity_ratio,
        railed_bunches=int(((n_pos > 0) | (n_neg > 0)).sum()),
        peak_mV=dict(p50=float(np.median(peak_mv)), p99=float(np.percentile(peak_mv, 99)),
                     max=float(peak_mv.max())),
        flash_peak_time_ns=float(np.median(peak_t)),
        recovery_ns=dict(p50=float(np.nanmedian(rec)),
                         p99=float(np.nanpercentile(rec, 99)),
                         definition='last sample above max(4 mV, 5 sigma_pre-flash)'),
        noise_sigma_mV=float(np.median(sigma_ct) * MV_PER_COUNT),
    )

    # Recovery measured on the bunch-averaged trace, where noise is not the limit:
    # the time the mean falls below 4 mV and below 1 % of its own peak.
    def mean_recovery(mask):
        if mask.sum() < 5:
            return None
        mt = dev[mask].mean(axis=0)
        pk = mt.max()
        pi = int(np.argmax(mt))
        out = {}
        for lbl, level in (('4mV', 4.0 / MV_PER_COUNT), ('1pct', 0.01 * pk)):
            o = np.flatnonzero(mt > level)
            out[lbl] = float(o[-1] - pi) if o.size else None
        out['peak_mV'] = float(pk * MV_PER_COUNT)
        out['n'] = int(mask.sum())
        return out

    # ---- per-plateau table ---------------------------------------------------
    rows = []
    for p in plateaus():
        m = (secs[ev] >= p['start'] + SETTLE_S) & (secs[ev] <= p['end'])
        for cls, sel in (('dedicated', m & dedicated), ('parasitic', m & parasitic)):
            if sel.sum() < 5:
                continue
            rows.append(dict(drift=p['drift'], resist=p['resist'], cls=cls,
                             label=p['label'], n=int(sel.sum()),
                             recovery=mean_recovery(sel),
                             charge_pC=float(np.median(charge_pc[sel])),
                             charge_mad=float(np.median(np.abs(
                                 charge_pc[sel] - np.median(charge_pc[sel])))),
                             peak_mV=float(np.median(peak_mv[sel])),
                             imon_uA=p['imon']))
    res['scan'] = rows

    # ---- gain fits, charge = A exp(k V) at each drift ------------------------
    fits = {}
    for drift in sorted({r['drift'] for r in rows}):
        for cls in ('dedicated', 'parasitic'):
            pts = [(r['resist'], r['charge_pC']) for r in rows
                   if r['drift'] == drift and r['cls'] == cls and r['charge_pC'] > 0]
            if len(pts) < 3:
                continue
            v = np.array([p[0] for p in pts], float)
            q = np.array([p[1] for p in pts], float)
            k, lna = np.polyfit(v, np.log(q), 1)
            fits[f'{drift}_{cls}'] = dict(
                drift=int(drift), cls=cls, n_points=len(pts),
                slope_per_V=float(k), e_fold_V=float(1 / k) if k else None,
                gain_per_10V=float(np.exp(10 * k)))
    res['gain_fits'] = fits

    # ---- drift dependence, at amplification voltages common to all drifts ----
    by = {}
    for r in rows:
        by.setdefault((r['cls'], r['resist']), {})[r['drift']] = r['charge_pC']
    drift_rows = []
    for (cls, resist), v in sorted(by.items()):
        if len(v) < 3:
            continue
        q = np.array([v[d] for d in sorted(v)], float)
        drift_rows.append(dict(cls=cls, resist=resist,
                               charges={str(d): float(v[d]) for d in sorted(v)},
                               spread_pct=float(100 * (q.max() - q.min()) / q.mean())))
    res['drift_dependence'] = dict(
        rows=drift_rows,
        max_spread_pct=float(max(r['spread_pct'] for r in drift_rows)) if drift_rows else None,
        note='drift 500/600/700 V compared at the same amplification voltage')

    # ---- stability: the 570 V point was visited three times, ~30 min apart ---
    rep = [r for r in rows if r['resist'] == 570 and r['cls'] == 'dedicated']
    res['repeat_point'] = [dict(drift=r['drift'], at=r['label'], n=r['n'],
                                charge_pC=r['charge_pC']) for r in sorted(rep, key=lambda x: x['label'])]

    # ---- post-flash rate -----------------------------------------------------
    z = d['zs']
    edges = np.array([3e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7])
    h, _ = np.histogram(z[:, 1], bins=edges)
    res['zs_rate_per_bunch_per_ms'] = [
        dict(t_lo_ms=float(edges[i] / 1e6), t_hi_ms=float(edges[i + 1] / 1e6),
             rate=float(h[i] / np.diff(edges)[i] * 1e6 / len(ev)))
        for i in range(len(h))]
    res['zs_first_block_ns'] = float(z[:, 1].min())
    res['zs_blocks_per_bunch'] = float(len(z) / len(ev))

    # ---- figures -------------------------------------------------------------
    COLD = {700: '#2f6f9f', 600: '#c0632c', 500: '#2e7d4f'}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for drift in sorted({r['drift'] for r in rows}, reverse=True):
        for cls, mk, ls in (('dedicated', 'o', '-'), ('parasitic', 's', '--')):
            pts = sorted([(r['resist'], r['charge_pC']) for r in rows
                          if r['drift'] == drift and r['cls'] == cls])
            if len(pts) < 2:
                continue
            ax[0].plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, ls=ls,
                       color=COLD[drift], lw=1.3, ms=4,
                       label=f'drift {drift} V, {cls}')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('detector A amplification voltage (V)')
    ax[0].set_ylabel('median flash charge (pC into 50 ohm)')
    ax[0].set_title('Detector-A scan, flash charge')
    ax[0].grid(alpha=.3, which='both')
    ax[0].legend(fontsize=7.5)

    for drift in sorted({r['drift'] for r in rows}, reverse=True):
        pts = {}
        for r in rows:
            if r['drift'] == drift:
                pts.setdefault(r['resist'], {})[r['cls']] = r['charge_pC']
        xs = sorted(v for v in pts if len(pts[v]) == 2)
        if len(xs) < 2:
            continue
        ax[1].plot(xs, [pts[v]['dedicated'] / pts[v]['parasitic'] for v in xs],
                   'o-', color=COLD[drift], lw=1.3, ms=4, label=f'drift {drift} V')
    ax[1].axhline(1.0, color='grey', ls=':', lw=1)
    ax[1].set_xlabel('detector A amplification voltage (V)')
    ax[1].set_ylabel('charge ratio, dedicated / parasitic')
    ax[1].set_title('Intensity response across the scan')
    ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'scan_709.png', dpi=140)
    plt.close(fig)

    # mean flash waveform at three amplification points, drift 700
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    t = np.arange(dev.shape[1]) / 1000.0
    for p in plateaus():
        if p['drift'] != 700 or p['resist'] not in (565, 540, 515, 500):
            continue
        m = (secs[ev] >= p['start'] + SETTLE_S) & (secs[ev] <= p['end']) & dedicated
        if m.sum() < 5:
            continue
        mt = dev[m].mean(axis=0) * MV_PER_COUNT
        ax[0].plot(t, mt, lw=1.1, label=f"{p['resist']} V  (n={m.sum()})")
        ax[1].plot(t, mt, lw=1.1)
    for a in ax:
        a.set_xlabel('time in acquisition window (us)')
        a.set_ylabel('mean signal (mV)')
        a.grid(alpha=.3)
    ax[0].set_xlim(11, 20)
    ax[0].set_title('Flash, dedicated pulses, drift 700 V')
    ax[0].legend(fontsize=8, title='amplification')
    ax[1].set_xlim(11.5, 30)
    ax[1].set_ylim(-6, 12)
    ax[1].axhline(4.0, color='crimson', ls=':', lw=1)
    ax[1].axhline(0, color='k', lw=.6)
    ax[1].set_title('Tail against the 4 mV scale')
    fig.tight_layout()
    fig.savefig(FIG / 'flash_709.png', dpi=140)
    plt.close(fig)

    with open(HERE / 'results_709.json', 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'scan'}, indent=1)[:2200])
    print(f'\nscan rows: {len(rows)}')


if __name__ == '__main__':
    main()
