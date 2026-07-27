#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PART 1 — run_67 time-since-flash distribution per plastic threshold, overlaid on
the GEANT in-gate IPC production spectrum.

This is the run_67 analogue of flash_comb/ipc_vs_runs, with two differences:
  * run_67 is a SINGLES trigger inside the N93B ~1->81 ms gate with the FEU
    watermark dropped to 2, so the recorded events form a CONTINUOUS, broad
    time-since-flash distribution (no deadtime comb, no discrete teeth). The
    green per-tooth live windows of the reference plot are therefore meaningless
    here — the DAQ is continuously live across the whole gate, so we shade the
    single live gate instead.
  * The scan axis of interest is the PLASTIC THRESHOLD. HV does not shape the
    time-since-flash distribution (it is set by the N93B gate and the beam), so
    we POOL over all drift/resist and draw one distribution per threshold.

Output -> <ANALYSIS_DIR>/July_HV_Scan/run67_scan/flash_timing/
  ipc_vs_thresholds.png   all three thresholds overlaid on the IPC curve (headline)
  ipc_vs_m{mip}.png       one per threshold, single-run ipc_vs_runs style
  timing_summary.csv      events/spill per window per threshold

Run: .venv/bin/python ntof_july_analysis/run67_scan/flash_timing.py
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import scan_lib as L  # noqa: E402

OUT = os.path.join(L.OUT_BASE, 'flash_timing')
GEANT = Path('/home/mx17/CLionProjects/MX17_Full_Geant')
IPC_NPZ = GEANT / 'analysis/reweight/ipc_ingate_spectrum.npz'
TMAX = 82.0          # show the whole N93B gate (reference stopped at 30 ms)
BW = 1.0             # histogram bin width [ms]
trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def load_ipc():
    Z = np.load(IPC_NPZ, allow_pickle=True)
    return dict(t=Z['t_ms'], dndt=Z['dNdt_ipc_per_pulse_per_ms'],
                bt=0.5 * (Z['bin_t_lo'] + Z['bin_t_hi']),
                bd=Z['bin_ipc_per_pulse'] / (Z['bin_t_hi'] - Z['bin_t_lo']),
                ipc_pulse=float(Z['ipc_per_pulse_ingate']),
                ipc_day=float(Z['ipc_per_day_ingate']))


def load_events():
    """Flash-ok, non-leader events with dt_ms + spill counts, per threshold."""
    ev, _, _ = L.load_all()
    ev = ev[ev.flash_ok & ~ev.is_leader].copy()
    # a spill is a unique (subrun, burst); pool over HV -> spills per threshold.
    n_spill = ev.groupby('mip').apply(
        lambda d: d.drop_duplicates(['subrun', 'burst']).shape[0],
        include_groups=False)
    return ev, n_spill


def _draw_ipc(ax, ipc):
    ax.fill_between(ipc['t'], 0, ipc['dndt'], color='#cfe3ef', zorder=1)
    ax.plot(ipc['t'], ipc['dndt'], color='#1f6fb4', lw=2.2, zorder=3,
            label='in-gate IPC production (reweighted Geant4)')
    ax.plot(ipc['bt'], ipc['bd'], 'o', ms=3.0, color='#d62728', zorder=4,
            label='IPC raw bins (10/decade)')
    # thermal peak marker
    m = ipc['t'] > 3.5
    tp = ipc['t'][m][np.argmax(ipc['dndt'][m])]
    ax.axvline(tp, color='k', ls=':', lw=1.1, zorder=3)
    ax.annotate(f'IPC thermal peak\n{tp:.1f} ms', xy=(tp, ipc['dndt'][m].max()),
                xytext=(tp + 8, ipc['dndt'].max() * 0.85), fontsize=8.5,
                color='#b1301a', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#b1301a'))
    ax.set_xlim(0, TMAX)
    ax.set_ylim(0, ipc['dndt'].max() * 1.18)
    ax.set_xlabel('neutron arrival time  t  [ms]   (t = 0 is the gamma flash)',
                  fontsize=11)
    ax.set_ylabel('IPC pairs / pulse / ms')
    ax.grid(alpha=0.25, zorder=0)


def _shade_gate(ax):
    ax.axvspan(0, L.READOUT_START_MS, color='0.6', alpha=0.30, hatch='//',
               zorder=0, label=f'gate closed (t < {L.READOUT_START_MS:g} ms)')
    ax.axvspan(L.READOUT_START_MS, L.GATE_CLOSE_MS, color='#31a354', alpha=0.10,
               zorder=0, label=f'N93B live gate ({L.READOUT_START_MS:g}-'
                               f'{L.GATE_CLOSE_MS:g} ms)')


def _hist(ev_mip, n_spill):
    bb = np.arange(0, TMAX + BW, BW)
    hh, _ = np.histogram(ev_mip.dt_ms.to_numpy(), bins=bb)
    return bb[:-1], hh / max(n_spill, 1)


def fig_combined(ev, n_spill, ipc):
    fig, ax = plt.subplots(figsize=(13, 6.6))
    _draw_ipc(ax, ipc)
    _shade_gate(ax)
    ax2 = ax.twinx()
    ymax = 0
    for mip in L.MIP_LEVELS:
        d = ev[ev.mip == mip]
        if d.empty:
            continue
        x, ys = _hist(d, int(n_spill.get(mip, 1)))
        tot = float((d.dt_ms >= L.READOUT_START_MS).sum()) / max(int(n_spill.get(mip, 1)), 1)
        ax2.step(x, ys, where='post', color=L.MIP_COLOR[mip], lw=1.8, zorder=5,
                 label=f'{L.MIP_LABEL[mip]}  ({tot:.0f} ev/spill)')
        # ignore the 1-2 ms opening transient when autoscaling
        steady = x >= (L.READOUT_START_MS + 1.0)
        ymax = max(ymax, ys[steady].max() if steady.any() else ys.max())
    ax2.set_ylim(0, ymax * 1.55)
    ax2.set_ylabel(f'run_67 recorded events / spill  (per {BW:g} ms bin)',
                   color='#333333')
    # window-set edges as light guides
    for lo, hi in L.WINDOW_SETS['broad']:
        for xv in (lo, hi):
            ax.axvline(xv, color='#888', ls='--', lw=0.8, alpha=0.5, zorder=1)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc='upper right', framealpha=0.94)
    ax.set_title('run_67 — time-since-flash distribution per plastic threshold, '
                 'vs the in-gate IPC spectrum\n'
                 'SINGLES trigger, N93B 1-81 ms gate, FEU Hwm 2 (no deadtime comb) '
                 '— HV-pooled.  Dashed = broad window edges.',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    p = os.path.join(OUT, 'ipc_vs_thresholds.png')
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return p


def fig_single(mip, ev, n_spill, ipc):
    d = ev[ev.mip == mip]
    if d.empty:
        return None
    ns = int(n_spill.get(mip, 1))
    fig, ax = plt.subplots(figsize=(12, 6.2))
    _draw_ipc(ax, ipc)
    _shade_gate(ax)
    # captured fraction of in-gate IPC: run_67's gate covers ~1-81 ms, so nearly
    # all of the 1-31 ms IPC spectrum lies inside it -> report it explicitly.
    tot_ipc = trapz(ipc['dndt'], ipc['t'])
    inside = (ipc['t'] >= L.READOUT_START_MS) & (ipc['t'] <= L.GATE_CLOSE_MS)
    frac = 100 * trapz(np.where(inside, ipc['dndt'], 0.0), ipc['t']) / tot_ipc
    ax2 = ax.twinx()
    x, ys = _hist(d, ns)
    ax2.step(x, ys, where='post', color=L.MIP_COLOR[mip], lw=1.8, zorder=5,
             label=f'{L.MIP_LABEL[mip]} recorded events')
    steady = x >= (L.READOUT_START_MS + 1.0)
    ymax = ys[steady].max() if steady.any() else ys.max()
    ax2.set_ylim(0, ymax * 1.9)
    ax2.set_ylabel(f'run_67 recorded events / spill (per {BW:g} ms bin)',
                   color=L.MIP_COLOR[mip])
    ax2.tick_params(axis='y', colors=L.MIP_COLOR[mip])
    if ys.max() > ymax * 1.9:
        ax2.annotate(f'opening bin {ys.max():.0f} ev/spill (off scale)',
                     xy=(L.READOUT_START_MS, ymax * 1.9 * 0.8), xytext=(6, 0),
                     textcoords='offset points', fontsize=8,
                     color=L.MIP_COLOR[mip], ha='left', va='center')
    tot = float((d.dt_ms >= L.READOUT_START_MS).sum()) / ns
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc='center right', framealpha=0.94)
    ax.text(0.985, 0.985,
            f'total recorded\n{tot:.0f} events / pulse\n({L.READOUT_START_MS:g}-'
            f'{TMAX:g} ms, {ns} spills)',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            color='#333', fontweight='bold', zorder=8,
            bbox=dict(boxstyle='round,pad=0.32', fc='white', ec='#666', lw=1.2))
    ax.text(0.985, 0.72,
            f'$\\int$ = {ipc["ipc_pulse"]:.2e} IPC/pulse = {ipc["ipc_day"]:.2f} '
            f'IPC/day\ninside the 1-81 ms gate: {frac:.0f}% of in-gate IPC',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec='#1f6fb4', alpha=0.93))
    ax.set_title(f"run_67 @ {L.MIP_LABEL[mip]} plastic threshold — recorded events "
                 f"vs the in-gate IPC spectrum\nSINGLES + PS-flash, N93B 1-81 ms "
                 f"gate, HV-pooled", fontsize=12, fontweight='bold')
    fig.tight_layout()
    p = os.path.join(OUT, f'ipc_vs_m{mip}.png')
    fig.savefig(p, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return p


def summary_table(ev, n_spill):
    rows = []
    for mip in L.MIP_LEVELS:
        d = ev[ev.mip == mip]
        if d.empty:
            continue
        ns = int(n_spill.get(mip, 1))
        for setname, wins in L.WINDOW_SETS.items():
            for lo, hi in wins:
                n = int(((d.dt_ms >= lo) & (d.dt_ms < hi)).sum())
                rows.append(dict(mip=mip, mip_label=L.MIP_LABEL[mip],
                                 window_set=setname, win_lo=lo, win_hi=hi,
                                 n_events=n, n_spill=ns,
                                 ev_per_spill=round(n / ns, 3)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'timing_summary.csv'), index=False)
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    ipc = load_ipc()
    ev, n_spill = load_events()
    print('spills per threshold:', {int(k): int(v) for k, v in n_spill.items()})
    print('  ->', fig_combined(ev, n_spill, ipc))
    for mip in L.MIP_LEVELS:
        p = fig_single(mip, ev, n_spill, ipc)
        if p:
            print('  ->', p)
    df = summary_table(ev, n_spill)
    print('\n=== events/spill per broad window per threshold ===')
    piv = (df[df.window_set == 'broad']
           .pivot(index=['win_lo', 'win_hi'], columns='mip', values='ev_per_spill'))
    print(piv.to_string())


if __name__ == '__main__':
    main()
