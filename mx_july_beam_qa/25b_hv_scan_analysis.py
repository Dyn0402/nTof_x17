#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
25b_hv_scan_analysis.py — resist-HV scan physics analysis for run_55.

Consumes cache/25_run55/*.npz from 25_hv_scan_extract.py.

Definitions
-----------
MIP-like cluster : 3 ≤ n_strips ≤ 20 and extent ≤ 25 mm (kills the ³He-capture
                   blobs and the wide coherent-noise clusters).
MIP track        : an x-plane and a y-plane MIP-like cluster whose sample
                   ranges overlap within ±2 samples (loose time coherence).
                   Amplitude = sum_amp(x) + sum_amp(y).
Time batches     : the DAQ FIFO samples the 30 ms beam gate in three bursts —
                   b0 ~0–0.5 ms (at flash), b1 ~6–14 ms, b2 ~15–30 ms.
                   2–6 ms is unsampled (DAQ readout dead time).

Outputs → figures/25_hv_scan/ + calib/25_hv_scan_summary.json
"""

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CACHE = os.path.join(os.path.dirname(__file__), 'cache', '25_run55')
FIGDIR = os.path.join(os.path.dirname(__file__), 'figures', '25_hv_scan')
CALIB = os.path.join(os.path.dirname(__file__), 'calib')

MIP_NMIN, MIP_NMAX, MIP_EXT = 3, 20, 25.0
SMP_SLACK = 2
HVS = [520, 525, 530, 535, 540, 545, 550, 555, 560]
DETNAMES = 'ABCD'
DETCOL = {'A': 'tab:blue', 'B': 'tab:orange', 'C': 'tab:green',
          'D': 'tab:red'}


def load_all():
    fs = sorted(glob.glob(os.path.join(CACHE, '*.npz')))
    Z = [np.load(f) for f in fs]
    D = {}
    n = [len(z['t_ms']) for z in Z]
    for k in Z[0].files:
        if getattr(Z[0][k], 'ndim', 0) >= 1 and len(Z[0][k]) == n[0]:
            D[k] = np.concatenate([z[k] for z in Z])
        elif Z[0][k].ndim == 0:
            D[k] = np.concatenate([np.full(m, z[k]) for z, m in zip(Z, n)])
    return D


def mip_track(D, di):
    """Per-event MIP-track flag + amplitude for detector di (vectorized)."""
    xn = D[f'd{di}_x_n']; yn = D[f'd{di}_y_n']
    xe = D[f'd{di}_x_extent']; ye = D[f'd{di}_y_extent']
    xa = D[f'd{di}_x_sum_amp']; ya = D[f'd{di}_y_sum_amp']
    xlo = D[f'd{di}_x_smp_lo']; xhi = D[f'd{di}_x_smp_hi']
    ylo = D[f'd{di}_y_smp_lo']; yhi = D[f'd{di}_y_smp_hi']

    xok = (xn >= MIP_NMIN) & (xn <= MIP_NMAX) & (xe <= MIP_EXT)
    yok = (yn >= MIP_NMIN) & (yn <= MIP_NMAX) & (ye <= MIP_EXT)

    nev, nc = xn.shape
    best_amp = np.zeros(nev)
    has = np.zeros(nev, bool)
    # all x-cluster × y-cluster pairs (nc×nc = 25, cheap)
    for i in range(nc):
        for j in range(nc):
            ov = (np.minimum(xhi[:, i], yhi[:, j])
                  - np.maximum(xlo[:, i], ylo[:, j]))
            ok = (xok[:, i] & yok[:, j] & (ov >= -SMP_SLACK))
            amp = xa[:, i] + ya[:, j]
            better = ok & (~has | (amp > best_amp))
            best_amp[better] = amp[better]
            has |= ok
    return has, best_amp


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(CALIB, exist_ok=True)
    D = load_all()
    t = D['t_ms']
    good = ~D['is_first'].astype(bool)
    hv = D['resist_v'].astype(int)
    cyc = D['cycle'].astype(int)

    trk, amp = {}, {}
    for di in range(4):
        trk[di], amp[di] = mip_track(D, di)

    b1 = good & (t >= 6) & (t < 14)
    b2 = good & (t >= 15) & (t < 30)
    b0 = good & (t >= 0) & (t < 2)

    summary = {'definitions': {
        'mip_cluster': f'{MIP_NMIN}<=n<={MIP_NMAX}, extent<={MIP_EXT}mm',
        'track': 'x+y MIP clusters, sample overlap >= -2',
        'b1': '6-14 ms', 'b2': '15-30 ms'}}

    # ------------------------------------------------------------------ fig 1
    # trigger + track time structure (all HVs pooled): where do we sample?
    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    bins = np.concatenate([np.arange(0, 2, 0.1), np.arange(2, 30.5, 0.5)])
    axs[0].hist(t[good], bins=bins, color='0.6',
                label='all triggers (scint doubles)')
    axs[0].set_yscale('log')
    axs[0].set_ylabel('triggers / bin')
    axs[0].legend()
    axs[0].set_title('run_55: trigger sampling of the 30 ms beam gate '
                     '(DAQ FIFO batches; 2–6 ms unsampled)')
    for di, dn in enumerate(DETNAMES):
        num, _ = np.histogram(t[good & trk[di]], bins=bins)
        den, _ = np.histogram(t[good], bins=bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(den > 20, num / den, np.nan)
        cc = 0.5 * (bins[1:] + bins[:-1])
        axs[1].step(cc, r * 100, where='mid', color=DETCOL[dn],
                    label=f'det {dn}')
    axs[1].set_xlabel('time since gamma flash [ms]')
    axs[1].set_ylabel('MIP-track rate per trigger [%]')
    axs[1].legend()
    for ax in axs:
        ax.axvspan(2, 6, color='red', alpha=0.08)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '01_time_structure.png'), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------ fig 2
    # turn-on: track rate vs t per HV (one panel per det)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    tbins = [0, 0.25, 0.5, 1, 2, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 30]
    cc = 0.5 * (np.array(tbins[1:]) + np.array(tbins[:-1]))
    cmap = plt.get_cmap('viridis')
    for di, dn in enumerate(DETNAMES):
        ax = axs[di // 2, di % 2]
        for k, h in enumerate(HVS):
            m = good & (hv == h)
            num, _ = np.histogram(t[m & trk[di]], bins=tbins)
            den, _ = np.histogram(t[m], bins=tbins)
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.where(den > 30, num / den, np.nan)
            ax.plot(cc, r * 100, 'o-', ms=3, lw=1,
                    color=cmap(k / (len(HVS) - 1)),
                    label=f'{h} V' if di == 0 else None)
        ax.axvspan(2, 6, color='red', alpha=0.08)
        ax.set_title(f'det {dn}')
        if di // 2:
            ax.set_xlabel('time since flash [ms]')
        if not di % 2:
            ax.set_ylabel('MIP-track rate / trigger [%]')
    axs[0, 0].legend(fontsize=7, ncol=2, title='resist HV')
    fig.suptitle('MIP-track turn-on vs time since gamma flash')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '02_turnon_vs_time.png'), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------ fig 3
    # efficiency proxy vs HV, b1 and b2, per cycle markers
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    res = {}
    for di, dn in enumerate(DETNAMES):
        ax = axs[di // 2, di % 2]
        for m, lab, col in [(b1, 'b1 6–14 ms', 'tab:blue'),
                            (b2, 'b2 15–30 ms (pile-up!)', 'tab:red')]:
            ys, es = [], []
            for h in HVS:
                mm = m & (hv == h)
                n, k = mm.sum(), (mm & trk[di]).sum()
                p = k / n if n else np.nan
                ys.append(p * 100)
                es.append(100 * np.sqrt(p * (1 - p) / n) if n else np.nan)
            ax.errorbar(HVS, ys, yerr=es, fmt='o-', color=col, label=lab)
            res[f'{dn}_{lab.split()[0]}'] = dict(zip(map(str, HVS), ys))
        # per-cycle spread at b1 (systematic check)
        for c in range(3):
            ys = [100 * (b1 & (hv == h) & (cyc == c) & trk[di]).sum()
                  / max((b1 & (hv == h) & (cyc == c)).sum(), 1)
                  for h in HVS]
            ax.plot(HVS, ys, '.', color='tab:blue', alpha=0.35, ms=4)
        ax.set_title(f'det {dn}')
        if di // 2:
            ax.set_xlabel('resist HV [V]')
        if not di % 2:
            ax.set_ylabel('MIP-track rate / trigger [%]')
        ax.legend(fontsize=8)
    fig.suptitle('Efficiency proxy vs resist HV (dots: single cycles, b1)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '03_eff_vs_hv.png'), dpi=130)
    plt.close(fig)
    summary['track_rate_pct'] = res

    # ------------------------------------------------------------------ fig 4
    # amplitude (median MIP-track amp) vs HV at b1; log-y gain curve
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    gain = {}
    for di, dn in enumerate(DETNAMES):
        ys = []
        for h in HVS:
            m = b1 & (hv == h) & trk[di]
            ys.append(np.median(amp[di][m]) if m.sum() > 20 else np.nan)
        gain[dn] = dict(zip(map(str, HVS), ys))
        axs[0].plot(HVS, ys, 'o-', color=DETCOL[dn], label=f'det {dn}')
        axs[1].semilogy(HVS, ys, 'o-', color=DETCOL[dn], label=f'det {dn}')
    for ax in axs:
        ax.set_xlabel('resist HV [V]')
        ax.legend()
    axs[0].set_ylabel('median MIP-track amplitude (x+y sum) [ADC]')
    axs[0].set_title('b1 (6–14 ms) amplitude vs HV')
    axs[1].set_title('log scale (exp. gain rise ⇒ straight line)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '04_amp_vs_hv.png'), dpi=130)
    plt.close(fig)
    summary['median_amp_b1'] = gain

    # ------------------------------------------------------------------ fig 5
    # amplitude vs time: is the earliest sampled batch suppressed?
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    tbins5 = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 30]
    cc5 = 0.5 * (np.array(tbins5[1:]) + np.array(tbins5[:-1]))
    for di, dn in enumerate(DETNAMES):
        ax = axs[di // 2, di % 2]
        for k, h in enumerate([560, 550, 540, 530, 520]):
            ys = []
            for lo, hi in zip(tbins5[:-1], tbins5[1:]):
                m = good & (hv == h) & trk[di] & (t >= lo) & (t < hi)
                ys.append(np.median(amp[di][m]) if m.sum() > 15 else np.nan)
            ax.plot(cc5, ys, 'o-', ms=3,
                    color=plt.get_cmap('viridis')(k / 4),
                    label=f'{h} V' if di == 0 else None)
        ax.set_title(f'det {dn}')
        if di // 2:
            ax.set_xlabel('time since flash [ms]')
        if not di % 2:
            ax.set_ylabel('median MIP-track amp [ADC]')
    axs[0, 0].legend(fontsize=8)
    fig.suptitle('MIP-track amplitude vs time since flash '
                 '(b2 ≥15 ms carries capture pile-up, esp. det D)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '05_amp_vs_time.png'), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------ fig 6
    # amplitude spectra at b1 per HV (shape check: Landau-ish = tracks)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    abins = np.linspace(0, 40000, 60)
    for di, dn in enumerate(DETNAMES):
        ax = axs[di // 2, di % 2]
        for k, h in enumerate([560, 550, 540]):
            m = b1 & (hv == h) & trk[di]
            if m.sum() > 30:
                ax.hist(amp[di][m], bins=abins, histtype='step', density=True,
                        color=plt.get_cmap('viridis')(k / 2),
                        label=f'{h} V (n={m.sum()})')
        ax.set_title(f'det {dn}')
        ax.legend(fontsize=8)
        if di // 2:
            ax.set_xlabel('MIP-track amplitude [ADC]')
    fig.suptitle('b1 MIP-track amplitude spectra')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '06_amp_spectra_b1.png'), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------ fig 7
    # context: capture/blob activity vs time (per-trigger prob of a big
    # cluster, n>20 strips or extent>25) — the pile-up landscape
    fig, ax = plt.subplots(figsize=(9, 5))
    bins7 = [0, 0.5, 2, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 30]
    cc7 = 0.5 * (np.array(bins7[1:]) + np.array(bins7[:-1]))
    for di, dn in enumerate(DETNAMES):
        big = ((D[f'd{di}_x_n'] > MIP_NMAX)
               | (D[f'd{di}_x_extent'] > MIP_EXT)).any(axis=1)
        num, _ = np.histogram(t[good & big], bins=bins7)
        den, _ = np.histogram(t[good], bins=bins7)
        with np.errstate(divide='ignore', invalid='ignore'):
            ax.plot(cc7, np.where(den > 30, num / den, np.nan) * 100, 'o-',
                    color=DETCOL[dn], label=f'det {dn}')
    ax.axvspan(2, 6, color='red', alpha=0.08)
    ax.set_xlabel('time since flash [ms]')
    ax.set_ylabel('events with big/blob x-cluster [%]')
    ax.set_title('Capture-blob / coherent-noise activity vs time (pile-up '
                 'landscape, all HVs pooled)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '07_blob_activity.png'), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------ fig 8
    # THE saturation-recovery figure: alive fraction (any thresholded MM
    # pulse) vs time, per HV.  Flat curve = recovered before 8 ms; rising
    # curve = still recovering inside the gate (long-saturation regime).
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    tb8 = [(0, 0.5), (8, 10), (10, 12), (20, 22), (22, 24), (24, 28)]
    cc8 = [0.25, 9, 11, 21, 23, 26]
    for di, dn in enumerate(DETNAMES):
        ax = axs[di // 2, di % 2]
        nthr = D[f'd{di}_nthr']
        for k, h in enumerate(HVS):
            ys = []
            for lo, hi in tb8:
                m = good & (hv == h) & (t >= lo) & (t < hi)
                ys.append((nthr[m] > 0).mean() * 100 if m.sum() > 40
                          else np.nan)
            ax.plot(cc8, ys, 'o-', ms=3, lw=1,
                    color=plt.get_cmap('viridis')(k / (len(HVS) - 1)),
                    label=f'{h} V' if di == 0 else None)
        ax.axvspan(2, 6, color='red', alpha=0.08)
        ax.set_title(f'det {dn}')
        if di // 2:
            ax.set_xlabel('time since flash [ms]')
        if not di % 2:
            ax.set_ylabel('alive fraction (any MM pulse) [%]')
    axs[0, 0].legend(fontsize=7, ncol=2, title='resist HV')
    fig.suptitle('Post-flash recovery: occupancy vs time — rising curves = '
                 'still saturated inside the gate')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '08_alive_recovery.png'), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------ fig 9
    # b1 vs b2 MIP-track rate per HV: points above the diagonal = still
    # suppressed at ~10 ms (C/D at 555–560); below = in-gate degradation (A).
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for di, dn in enumerate(DETNAMES):
        r1 = [100 * (b1 & (hv == h) & trk[di]).sum()
              / max((b1 & (hv == h)).sum(), 1) for h in HVS]
        r2 = [100 * (b2 & (hv == h) & trk[di]).sum()
              / max((b2 & (hv == h)).sum(), 1) for h in HVS]
        sc = ax.scatter(r1, r2, c=HVS, cmap='viridis', s=40, zorder=3,
                        edgecolor=DETCOL[dn], linewidths=1.8,
                        label=f'det {dn} (edge color)')
        ax.plot(r1, r2, '-', color=DETCOL[dn], alpha=0.4, lw=1)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.5)
    ax.set_xlabel('MIP-track rate at 6–14 ms [%]')
    ax.set_ylabel('MIP-track rate at 15–30 ms [%]')
    plt.colorbar(sc, ax=ax, label='resist HV [V]')
    ax.legend(fontsize=8)
    ax.set_title('Early vs late rate: above diagonal = suppressed at 10 ms')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '09_early_vs_late.png'), dpi=130)
    plt.close(fig)

    # ----------------------------------------------------------------- fig 10
    # cluster multiplicity per event vs time, per HV: mean number of x-plane
    # clusters with n>=3 strips (MIP-like and blob counted separately)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    tb10 = [(0, 0.5), (8, 10), (10, 12), (20, 22), (22, 24), (24, 28)]
    cc10 = [0.25, 9, 11, 21, 23, 26]
    for di, dn in enumerate(DETNAMES):
        ax = axs[di // 2, di % 2]
        xn = D[f'd{di}_x_n']; xe = D[f'd{di}_x_extent']
        n_mip = ((xn >= MIP_NMIN) & (xn <= MIP_NMAX)
                 & (xe <= MIP_EXT)).sum(axis=1)
        n_blob = ((xn > MIP_NMAX) | ((xn >= MIP_NMIN)
                                     & (xe > MIP_EXT))).sum(axis=1)
        for k, h in enumerate(HVS):
            ys, yb = [], []
            for lo, hi in tb10:
                m = good & (hv == h) & (t >= lo) & (t < hi)
                ys.append(n_mip[m].mean() if m.sum() > 40 else np.nan)
                yb.append(n_blob[m].mean() if m.sum() > 40 else np.nan)
            col = plt.get_cmap('viridis')(k / (len(HVS) - 1))
            ax.plot(cc10, ys, 'o-', ms=3, lw=1, color=col,
                    label=f'{h} V' if di == 0 else None)
            ax.plot(cc10, yb, 's--', ms=3, lw=1, color=col, alpha=0.5)
        ax.axvspan(2, 6, color='red', alpha=0.08)
        ax.set_title(f'det {dn}  (solid: MIP-like, dashed: blob)')
        if di // 2:
            ax.set_xlabel('time since flash [ms]')
        if not di % 2:
            ax.set_ylabel('mean x-clusters / event')
    axs[0, 0].legend(fontsize=7, ncol=2, title='resist HV')
    fig.suptitle('Cluster multiplicity vs time — MIP-like (3–20 strips, '
                 '≤25 mm) vs blob (wider/bigger) x-clusters')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '10_cluster_multiplicity.png'), dpi=130)
    plt.close(fig)

    # ----------------------------------------------------------------- fig 11
    # composition of the LARGEST x-cluster per event vs HV at b1
    fig, axs = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True)
    classes = ['no cluster', '1–2 strips', 'MIP-like', '21–60 strips',
               'blob (>60 | wide)']
    ccol = ['0.85', '0.6', 'tab:green', 'tab:orange', 'tab:red']
    for di, dn in enumerate(DETNAMES):
        ax = axs[di]
        xn0 = D[f'd{di}_x_n'][:, 0]
        xe0 = D[f'd{di}_x_extent'][:, 0]
        fr = np.zeros((len(HVS), 5))
        for k, h in enumerate(HVS):
            m = b1 & (hv == h)
            tot = max(m.sum(), 1)
            c_no = (xn0[m] == 0).sum()
            c_sm = ((xn0[m] > 0) & (xn0[m] < 3)).sum()
            c_mip = ((xn0[m] >= 3) & (xn0[m] <= MIP_NMAX)
                     & (xe0[m] <= MIP_EXT)).sum()
            c_mid = ((xn0[m] > MIP_NMAX) & (xn0[m] <= 60)).sum()
            c_big = tot - c_no - c_sm - c_mip - c_mid
            fr[k] = np.array([c_no, c_sm, c_mip, c_mid, c_big]) / tot * 100
        bot = np.zeros(len(HVS))
        for j in range(5):
            ax.bar([str(h) for h in HVS], fr[:, j], bottom=bot,
                   color=ccol[j], label=classes[j] if di == 0 else None,
                   width=0.8)
            bot += fr[:, j]
        ax.set_title(f'det {dn}')
        ax.tick_params(axis='x', rotation=60, labelsize=7)
        if di == 0:
            ax.set_ylabel('fraction of b1 triggers [%]')
    axs[0].legend(fontsize=7, loc='lower left')
    fig.suptitle('What the largest x-cluster looks like at 8–12 ms, vs '
                 'resist HV')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '11_cluster_classes.png'), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------- summary
    print('\n=== b1 (6–14 ms) MIP-track rate per trigger [%] ===')
    print('HV  |' + ''.join(f'   {d}  ' for d in DETNAMES))
    for h in HVS:
        row = [f'{100*(b1 & (hv==h) & trk[di]).sum()/max((b1 & (hv==h)).sum(),1):5.1f}'
               for di in range(4)]
        print(f'{h} |' + ' '.join(row))
    print('\n=== b1 median MIP amp [ADC] ===')
    for h in HVS:
        row = []
        for di in range(4):
            m = b1 & (hv == h) & trk[di]
            row.append(f'{np.median(amp[di][m]):7.0f}' if m.sum() > 20
                       else '      -')
        print(f'{h} |' + ' '.join(row))

    with open(os.path.join(CALIB, '25_hv_scan_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print('\nfigures →', FIGDIR)
    print('donzo')


if __name__ == '__main__':
    main()
