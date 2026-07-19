#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
26b_zs_analysis.py — ZS threshold optimization from the run_55 simulation.

Consumes cache/26_run55/*.npz (from 26_zs_sim_extract.py).

Observables
-----------
retention      : kept channels / 4096 per event (channel-level volume)
sample volume  : (samples above thr + ZsChkSmp per kept channel) / (4096*32)
strip survival : fraction of MIP-track-cluster strips (25's clusters, built
                 from >400 ADC hits) whose CM-corrected waveform max crosses
                 N*sigma_aux
cluster surv.  : clusters keeping >= 3 strips (still reconstructable)

All under firmware-faithful CM correction (per Dream chip, median).

Outputs → figures/26_zs/ + calib/26_zs_summary.json
"""

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CACHE = os.path.join(os.path.dirname(__file__), 'cache', '26_run55')
FIGDIR = os.path.join(os.path.dirname(__file__), 'figures', '26_zs')
CALIB = os.path.join(os.path.dirname(__file__), 'calib')

NCH_TOT = 4096
NSMP = 32
ZSCHK = 4                    # ZsChkSmp: extra samples read out per crossing
FULL_EV_MB = 4096 * 32 * 2 / 1e6   # payload of a no-ZS event (16-bit samples)
READ_MS_FULL = 2.0           # measured: ~8 ms FIFO drain / ~4 events (no ZS)
DETNAMES = 'ABCD'
DET_FEUS = {0: [2, 3], 1: [4, 5], 2: [6, 7], 3: [0, 1]}  # det -> feu idx (0-based)
DETCOL = {'A': 'tab:blue', 'B': 'tab:orange', 'C': 'tab:green',
          'D': 'tab:red'}
MIN_STRIPS = 3


def load_all():
    fs = sorted(glob.glob(os.path.join(CACHE, '*.npz')))
    Z = [np.load(f) for f in fs]
    grid = Z[0]['zs_grid']
    ev = {}
    for k in ['t_ms', 'is_first', 'n_ch_kept', 'n_smp_above',
              'n_ch_kept_nocm5', 'base_shift']:
        ev[k] = np.concatenate([z[k] for z in Z])
    n = [len(z['t_ms']) for z in Z]
    ev['hv'] = np.concatenate([np.full(m, int(z['resist_v']))
                               for z, m in zip(Z, n)])
    cl_meta = np.concatenate([z['cl_meta'] for z in Z])
    cl_surv = np.concatenate([z['cl_surv'] for z in Z])
    cl_hv = np.concatenate([np.full(len(z['cl_meta']), int(z['resist_v']))
                            for z in Z])
    sig_nat = np.stack([z['sig_nat'] for z in Z])   # (subrun, feu, ch)
    sig_aux_like = None
    return grid, ev, cl_meta, cl_surv, cl_hv, sig_nat


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    grid, ev, cl_meta, cl_surv, cl_hv, sig_nat = load_all()
    t = ev['t_ms']
    good = ~ev['is_first'].astype(bool)
    hv = ev['hv']
    b1 = good & (t >= 6) & (t < 14)
    b2 = good & (t >= 15)
    b0 = good & (t < 2)
    nZ = len(grid)

    kept_tot = ev['n_ch_kept'].sum(axis=1)          # (ev, N)
    smp_tot = ev['n_smp_above'].sum(axis=1)
    vol_ch = kept_tot / NCH_TOT
    vol_smp = (smp_tot + ZSCHK * kept_tot) / (NCH_TOT * NSMP)
    nocm = ev['n_ch_kept_nocm5'].sum(axis=1) / NCH_TOT

    summary = {'zs_grid': grid.tolist(),
               'assumes': 'CM per Dream (Feu_RunCtrl_CM=1), sigmas from '
                          'beam-off pedestal run (thr.aux)'}

    # ------------------------------------------------------------------ fig 1
    # WHY CM: retention with/without CM at 5 sigma, vs time
    fig, ax = plt.subplots(figsize=(9, 5))
    tb = [(0, 0.5), (6, 10), (10, 14), (15, 20), (20, 24), (24, 28)]
    cc = [0.25, 8, 12, 17.5, 22, 26]
    k5 = int(np.argmin(np.abs(grid - 5.0)))
    for arr, lab, col in [(nocm, 'no CM (as configured!), 5σ', 'tab:red'),
                          (vol_ch[:, k5], 'CM per Dream, 5σ', 'tab:green'),
                          (vol_ch[:, int(np.argmin(np.abs(grid - 3.0)))],
                           'CM per Dream, 3σ', 'tab:blue')]:
        ys = [np.median(arr[good & (t >= lo) & (t < hi)]) * 100
              for lo, hi in tb]
        ax.plot(cc, ys, 'o-', color=col, label=lab)
    ax.set_xlabel('time since flash [ms]')
    ax.set_ylabel('channels kept [%]')
    ax.set_title('ZS retention: common-mode correction is MANDATORY '
                 '(beam CM wander = 10–20× beam-off σ)')
    ax.legend()
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '01_cm_mandatory.png'), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------ fig 2
    # retention vs threshold, by time batch (medians + p90)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for m, lab, col in [(b0, '0–2 ms', 'tab:purple'),
                        (b1, '6–14 ms (thermal peak)', 'tab:red'),
                        (b2, '15–30 ms', 'tab:orange'),
                        (good, 'all triggers', 'k')]:
        axs[0].plot(grid, np.median(vol_ch[m], axis=0) * 100, 'o-',
                    color=col, label=lab)
        axs[0].plot(grid, np.percentile(vol_ch[m], 90, axis=0) * 100, '--',
                    color=col, alpha=0.5)
        axs[1].plot(grid, np.median(vol_smp[m], axis=0) * 100, 'o-',
                    color=col, label=lab)
    for ax, ttl in zip(axs, ['channel retention', 'sample-level volume '
                             f'(tpc ZS, +{ZSCHK} smp/crossing)']):
        ax.set_xlabel('ZS threshold [σ, beam-off]')
        ax.set_ylabel('kept [%]')
        ax.set_title(ttl)
        ax.set_yscale('log')
        ax.legend(fontsize=8)
    fig.suptitle('Data volume vs ZS threshold (CM on; solid median, '
                 'dashed p90)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '02_volume_vs_thr.png'), dpi=130)
    plt.close(fig)
    summary['vol_ch_median_b1'] = dict(zip(map(str, grid),
                                       (np.median(vol_ch[b1], axis=0)
                                        * 100).round(2).tolist()))
    summary['vol_smp_median_b1'] = dict(zip(map(str, grid),
                                        (np.median(vol_smp[b1], axis=0)
                                         * 100).round(3).tolist()))

    # ------------------------------------------------------------------ fig 3
    # per-detector retention vs threshold at b1, per HV group
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for di, dn in enumerate(DETNAMES):
        ax = axs[di // 2, di % 2]
        keep_det = ev['n_ch_kept'][:, DET_FEUS[di], :].sum(axis=1) / 1024
        for k, hvs in enumerate([[520, 525, 530], [540, 545],
                                 [555, 560]]):
            m = b1 & np.isin(hv, hvs)
            ax.plot(grid, np.median(keep_det[m], axis=0) * 100, 'o-',
                    color=plt.get_cmap('viridis')(k / 2),
                    label=f'{hvs[0]}–{hvs[-1]} V' if di == 0 else None)
        ax.set_yscale('log')
        ax.set_title(f'det {dn}')
        if di // 2:
            ax.set_xlabel('ZS threshold [σ]')
        if not di % 2:
            ax.set_ylabel('channels kept [%]')
    axs[0, 0].legend(fontsize=8, title='resist HV')
    fig.suptitle('Per-detector retention at 6–14 ms vs threshold and HV')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '03_retention_det_hv.png'), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------ fig 4
    # MIP strip + cluster survival vs threshold
    cm_ev, cm_det, cm_t, cm_n = (cl_meta[:, 0], cl_meta[:, 1],
                                 cl_meta[:, 3], cl_meta[:, 4])
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    res_surv = {}
    for di, dn in enumerate(DETNAMES):
        m = cm_det == di
        strip_frac = cl_surv[m].sum(axis=0) / cm_n[m].sum()
        clus_ok = (cl_surv[m] >= MIN_STRIPS).mean(axis=0)
        axs[0].plot(grid, strip_frac * 100, 'o-', color=DETCOL[dn],
                    label=f'det {dn} (n={m.sum()})')
        axs[1].plot(grid, clus_ok * 100, 'o-', color=DETCOL[dn])
        res_surv[dn] = {'strip_pct': (strip_frac * 100).round(2).tolist(),
                        'cluster_pct': (clus_ok * 100).round(2).tolist()}
    axs[0].set_ylabel('MIP-cluster strips surviving [%]')
    axs[1].set_ylabel(f'clusters keeping ≥{MIN_STRIPS} strips [%]')
    for ax in axs:
        ax.set_xlabel('ZS threshold [σ, beam-off]')
        ax.axhline(99, color='0.7', ls=':')
    axs[0].legend(fontsize=8)
    fig.suptitle('Real track-hit survival under ZS (strips from >400 ADC '
                 'clusters; CM on)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '04_strip_survival.png'), dpi=130)
    plt.close(fig)
    summary['survival'] = res_surv

    # ------------------------------------------------------------------ fig 5
    # survival vs HV (are low-HV/small-amp tracks at risk?) at 3.5 and 5 sigma
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, N in zip(axs, [3.5, 5.0]):
        k = int(np.argmin(np.abs(grid - N)))
        for di, dn in enumerate(DETNAMES):
            ys = []
            hvs = sorted(set(cl_hv))
            for h in hvs:
                m = (cm_det == di) & (cl_hv == h)
                ys.append(100 * cl_surv[m][:, k].sum() / cm_n[m].sum()
                          if m.sum() > 30 else np.nan)
            ax.plot(hvs, ys, 'o-', color=DETCOL[dn], label=f'det {dn}')
        ax.set_title(f'{N}σ')
        ax.set_xlabel('resist HV [V]')
        ax.axhline(99, color='0.7', ls=':')
    axs[0].set_ylabel('MIP strips surviving [%]')
    axs[0].legend(fontsize=8)
    fig.suptitle('Strip survival vs HV — lower gain ⇒ smaller pulses ⇒ '
                 'more ZS loss')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '05_survival_vs_hv.png'), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------ fig 6
    # readout-time / gap-coverage estimate
    fig, ax = plt.subplots(figsize=(9, 5))
    # volume -> per-event readout estimate; overhead floor ~5% of full
    for m, lab, col in [(b1, '6–14 ms', 'tab:red'), (b2, '15–30 ms',
                        'tab:orange'), (b0, '0–2 ms', 'tab:purple')]:
        vol = np.median(vol_smp[m], axis=0)
        t_read = READ_MS_FULL * np.maximum(vol, 0.02)
        ax.plot(grid, t_read, 'o-', color=col,
                label=f'{lab} (median volume)')
    ax.axhline(READ_MS_FULL, color='k', ls='--',
               label='no-ZS readout (~2 ms/event)')
    ax.set_xlabel('ZS threshold [σ]')
    ax.set_ylabel('est. readout time per event [ms]')
    ax.set_yscale('log')
    ax.set_title('Estimated per-event readout vs threshold '
                 '(2 ms full-event drain, 2% overhead floor)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '06_readout_estimate.png'), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------ fig 7
    # in-beam residual (post-CM) sigma vs beam-off sigma per FEU
    fig, ax = plt.subplots(figsize=(9, 5))
    med_nat = np.median(sig_nat, axis=0)            # (feu, ch)
    for f in range(8):
        ok = med_nat[f] > 0
        ax.plot(np.full(ok.sum(), f + 1)
                + np.random.uniform(-0.25, 0.25, ok.sum()),
                med_nat[f][ok], '.', ms=2, alpha=0.3)
    ax.set_xlabel('FEU')
    ax.set_ylabel('post-CM in-beam σ per channel [ADC]')
    ax.set_yscale('log')
    ax.set_title('Residual noise after Dream-level CM correction '
                 '(compare: beam-off σ p50 ≈ 4 ADC)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, '07_residual_sigma.png'), dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------- summary
    print('=== volume + survival vs N (b1, CM on) ===')
    print('N    ch-kept%  smp-vol%  |  strip surv% A / B / C / D')
    for k, N in enumerate(grid):
        s = '  '.join(f'{100*cl_surv[cm_det==di][:, k].sum()/cm_n[cm_det==di].sum():5.1f}'
                      for di in range(4))
        print(f'{N:3.1f}  {np.median(vol_ch[b1][:, k])*100:7.2f} '
              f'{np.median(vol_smp[b1][:, k])*100:8.3f}  |  {s}')
    with open(os.path.join(CALIB, '26_zs_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print('\nfigures →', FIGDIR)
    print('donzo')


if __name__ == '__main__':
    main()
