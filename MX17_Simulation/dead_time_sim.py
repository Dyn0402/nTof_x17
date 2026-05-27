#!/usr/bin/env python3
"""
Standalone DAQ Dead Time Simulation for MX17 at n-TOF EAR2.

Simulates X17 candidate acceptance vs. neutron TOF/energy accounting for DAQ
dead time after each trigger. Three event sources per spill:
  - X17 candidates:             TOF distribution from calculation tables
  - IPC pairs:                  same TOF distribution, higher rate
  - Single-particle background: flat in TOF (parameterized count per spill)

All events are assumed to fire a trigger (conservative). The dead time
after each trigger is:
    dead_time = N_READOUT_CHANNELS * CLOCK_PERIOD_NS * N_SAMPLES
optionally reduced by ZS_DEADTIME_REDUCTION when zero suppression is on.

Gamma flash at spill start either fires an automatic trigger (no veto) or is
suppressed by a hardware veto window (VETO_ENABLED).
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from X17CalculationParser import X17CalculationParser
from neutron_energy_vs_flight_time import energy_eV_to_time_s, time_s_to_energy_eV

# ===========================================================================
# Parameters
# ===========================================================================

CALCULATION_TABLES_DIR = '/media/dylan/data/x17/calculation_tables/'
FILE_NAME = 'results_3He'
DISTANCE_M = 19.5          # m, EAR2 flight path

# DAQ
N_SAMPLES = 50            # ADC samples per readout window
CLOCK_PERIOD_NS = 40.0     # ns, ADC clock period
N_READOUT_CHANNELS = 64    # number of channels (sets dead time length)

# Zero suppression
ZERO_SUPPRESSION = False       # True = zero suppression active
ZS_DEADTIME_REDUCTION = 0.5   # fractional dead time reduction when ZS on

# Beam spill timing
VETO_ENABLED = True        # hardware veto at start of spill
VETO_WINDOW_US = 45.0       # µs, veto window (covers gamma flash + fastest neutrons)

# TOF simulation range
MAX_TOF_MS = 20.0          # ms, upper limit (where thermal neutrons arrive)

# Event rates per spill (Poisson means)
IPC_PER_PULSE = 100.0
X17_FRACTION_OF_IPC = 0.025   # X17 rate = IPC_PER_PULSE * X17_FRACTION_OF_IPC
SINGLE_BG_PER_PULSE = 200.0   # single-particle bg events per spill (flat in TOF)

# Simulation
N_PULSES = 200000
SEED = 42

# ===========================================================================


def main():
    rng = np.random.default_rng(SEED)

    # Dead time per trigger
    dead_time_ns = N_READOUT_CHANNELS * CLOCK_PERIOD_NS * N_SAMPLES
    if ZERO_SUPPRESSION:
        dead_time_ns *= (1.0 - ZS_DEADTIME_REDUCTION)
    dead_time_s = dead_time_ns * 1e-9

    c_light = 299792458.0
    t_flash_s = DISTANCE_M / c_light   # gamma flash arrival (~65 ns)
    t_max_s = MAX_TOF_MS * 1e-3
    t_veto_end_s = t_flash_s + VETO_WINDOW_US * 1e-6

    x17_per_pulse = IPC_PER_PULSE * X17_FRACTION_OF_IPC

    print("=" * 65)
    print("MX17 DAQ Dead Time Simulation")
    print("=" * 65)
    print(f"  Dead time/trigger  : {dead_time_ns/1e6:.4f} ms  ({dead_time_ns:.0f} ns)")
    print(f"  Gamma flash at     : {t_flash_s*1e9:.1f} ns")
    if VETO_ENABLED:
        print(f"  Veto window        : enabled, {VETO_WINDOW_US:.1f} µs")
    else:
        print(f"  Veto window        : disabled (gamma flash fires trigger)")
    if ZERO_SUPPRESSION:
        print(f"  Zero suppression   : on, {ZS_DEADTIME_REDUCTION*100:.0f}% dead time reduction")
    else:
        print(f"  Zero suppression   : off")
    print(f"  Rates/spill        : X17={x17_per_pulse:.2f},  IPC={IPC_PER_PULSE:.1f},  bg={SINGLE_BG_PER_PULSE:.1f}")
    print(f"  N pulses           : {N_PULSES}")

    # Load X17 TOF spectrum
    t_low, t_high, x17_vals, e_low, e_up = load_x17_tof_data(t_flash_s, t_max_s)
    n_bins = len(x17_vals)
    cdf = _build_cdf(x17_vals)

    print(f"  X17 TOF bins loaded: {n_bins}  "
          f"({t_low[0]*1e6:.2f} – {t_high[-1]*1e6:.2f} µs)")

    # Per-bin efficiency accumulators aligned with X17 data bins
    x17_gen_per_bin = np.zeros(n_bins)
    x17_acc_per_bin = np.zeros(n_bins)
    total_gen = 0
    total_acc = 0

    # Simulate pulses
    try:
        from tqdm import tqdm
        pulse_iter = tqdm(range(N_PULSES), desc='Simulating', unit='spill',
                          dynamic_ncols=True)
    except ImportError:
        pulse_iter = range(N_PULSES)

    for _ in pulse_iter:
        n_x17 = int(rng.poisson(x17_per_pulse))
        n_ipc = int(rng.poisson(IPC_PER_PULSE))
        n_bg  = int(rng.poisson(SINGLE_BG_PER_PULSE))

        tof_x17 = _sample_tof(cdf, t_low, t_high, n_x17, rng)
        tof_ipc = _sample_tof(cdf, t_low, t_high, n_ipc, rng)
        tof_bg  = rng.uniform(t_flash_s, t_max_s, n_bg)

        # Pre-compute bin indices for X17 events for fast accumulation
        x17_bin_idx = _tof_to_bin_indices(tof_x17, t_low, n_bins)

        # Build combined sorted event list; bin_idx = -1 means not X17
        n_total = n_x17 + n_ipc + n_bg
        all_tof = np.empty(n_total)
        all_bin = np.full(n_total, -1, dtype=np.intp)

        all_tof[:n_x17] = tof_x17
        all_bin[:n_x17] = x17_bin_idx
        all_tof[n_x17:n_x17 + n_ipc] = tof_ipc
        all_tof[n_x17 + n_ipc:] = tof_bg

        order = np.argsort(all_tof, kind='stable')
        all_tof = all_tof[order]
        all_bin = all_bin[order]

        # Accumulate generated X17 counts per bin
        total_gen += n_x17
        for bidx in x17_bin_idx:
            if 0 <= bidx < n_bins:
                x17_gen_per_bin[bidx] += 1

        # Dead-time walk
        if VETO_ENABLED:
            dead_until = t_veto_end_s
        else:
            dead_until = t_flash_s + dead_time_s  # gamma flash fires trigger

        for t, bidx in zip(all_tof, all_bin):
            if t < dead_until:
                continue
            dead_until = t + dead_time_s
            if bidx >= 0:
                total_acc += 1
                x17_acc_per_bin[bidx] += 1

    # Results
    overall_eff = total_acc / total_gen if total_gen > 0 else 0.0
    print("\n--- Results ---")
    print(f"  X17 generated : {total_gen:,}")
    print(f"  X17 accepted  : {total_acc:,}")
    print(f"  X17 lost      : {total_gen - total_acc:,}  "
          f"({(total_gen - total_acc)/total_gen*100:.1f}%)")
    print(f"  Overall eff   : {overall_eff*100:.2f}%")

    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = np.where(x17_gen_per_bin > 0,
                              x17_acc_per_bin / x17_gen_per_bin,
                              np.nan)
        eff_err = np.where(x17_gen_per_bin > 0,
                           np.sqrt(efficiency * (1.0 - np.where(np.isnan(efficiency), 0, efficiency))
                                   / x17_gen_per_bin),
                           np.nan)

    # Fraction of X17 TOF distribution that falls inside the veto window
    veto_x17_fraction = _veto_x17_fraction(t_low, t_high, x17_vals,
                                           t_flash_s, t_veto_end_s)

    plot_results(t_low, t_high, x17_vals, efficiency, eff_err,
                 x17_gen_per_bin, dead_time_s, overall_eff,
                 t_flash_s, t_veto_end_s, veto_x17_fraction)
    plt.show()
    print("Done.")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_x17_tof_data(t_flash_s, t_max_s):
    """Load X17 spectrum, convert to TOF bins sorted by ascending TOF."""
    parser = X17CalculationParser(CALCULATION_TABLES_DIR + FILE_NAME)
    df = parser.get_dataframe()

    e_low = df['elow [eV]'].values
    e_up  = df['eup [eV]'].values
    x17   = df['X17 [1/day]'].values

    # Higher energy → lower TOF
    t_lo = energy_eV_to_time_s(e_up,  distance=DISTANCE_M)
    t_hi = energy_eV_to_time_s(e_low, distance=DISTANCE_M)

    order = np.argsort(t_lo)
    t_lo  = t_lo[order];  t_hi  = t_hi[order]
    x17   = x17[order];   e_low = e_low[order];  e_up = e_up[order]

    mask = (t_hi > t_flash_s) & (t_lo < t_max_s) & (x17 > 0)
    return t_lo[mask], t_hi[mask], x17[mask], e_low[mask], e_up[mask]


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _build_cdf(weights):
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return cdf


def _sample_tof(cdf, t_low, t_high, n, rng):
    if n == 0:
        return np.empty(0)
    u = rng.uniform(0.0, 1.0, n)
    idx = np.searchsorted(cdf, u)
    idx = np.clip(idx, 0, len(t_low) - 1)
    return t_low[idx] + rng.uniform(0.0, 1.0, n) * (t_high[idx] - t_low[idx])


def _tof_to_bin_indices(tof_arr, t_low, n_bins):
    """Map TOF values to their data-bin indices."""
    if len(tof_arr) == 0:
        return np.empty(0, dtype=np.intp)
    idxs = np.searchsorted(t_low, tof_arr, side='right') - 1
    return np.clip(idxs, 0, n_bins - 1).astype(np.intp)


# ---------------------------------------------------------------------------
# Veto fraction
# ---------------------------------------------------------------------------

def _veto_x17_fraction(t_low, t_high, x17_vals, t_flash_s, t_veto_end_s):
    """Fraction of the X17 TOF distribution that falls within the veto window."""
    total = x17_vals.sum()
    if total == 0:
        return 0.0
    veto_weight = 0.0
    for lo, hi, w in zip(t_low, t_high, x17_vals):
        overlap_lo = max(lo, t_flash_s)
        overlap_hi = min(hi, t_veto_end_s)
        if overlap_hi > overlap_lo:
            bin_width = hi - lo
            veto_weight += w * (overlap_hi - overlap_lo) / bin_width
    return veto_weight / total


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(t_low, t_high, x17_vals, efficiency, eff_err,
                 x17_gen_per_bin, dead_time_s, overall_eff,
                 t_flash_s, t_veto_end_s, veto_x17_fraction):
    t_mid = (t_low + t_high) / 2.0
    t_err = (t_high - t_low) / 2.0
    valid = ~np.isnan(efficiency)
    low_stat = valid & (x17_gen_per_bin < 10)

    xmin = min(t_low[0] * 1e6 * 0.7, t_flash_s * 1e6 * 0.5)
    xmax = t_high[-1] * 1e6 * 1.4

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.subplots_adjust(hspace=0.04)

    # --- Panel 1: X17 spectrum ---
    ax1 = axes[0]
    ax1.errorbar(t_mid * 1e6, x17_vals, xerr=t_err * 1e6,
                 fmt='o', ms=4, capsize=2, color='C0', label='X17 rate [1/day]')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('X17 rate [1/day]', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(which='both', ls=':', lw=0.5, alpha=0.7)
    ax1.set_xlim(xmin, xmax)

    title = (
        f'MX17 DAQ Dead-Time Acceptance  —  '
        f'n_samples={N_SAMPLES}, clock={CLOCK_PERIOD_NS:.0f} ns, '
        f'channels={N_READOUT_CHANNELS},  dead_time={dead_time_s*1e3:.3f} ms\n'
        f'ZS={"on (%.0f%% reduction)" % (ZS_DEADTIME_REDUCTION*100) if ZERO_SUPPRESSION else "off"},  '
        f'veto={"%.1f µs" % VETO_WINDOW_US if VETO_ENABLED else "off"},  '
        f'IPC/spill={IPC_PER_PULSE:.0f},  bg/spill={SINGLE_BG_PER_PULSE:.0f},  '
        f'overall X17 eff = {overall_eff*100:.1f}%'
    )
    ax1.set_title(title, fontsize=9)

    # Energy axis on top of panel 1
    _add_energy_top_axis(ax1, xmin, xmax)

    # --- Panel 2: X17 acceptance efficiency ---
    ax2 = axes[1]
    if valid.any():
        ax2.errorbar(
            t_mid[valid] * 1e6, efficiency[valid] * 100,
            yerr=eff_err[valid] * 100,
            fmt='o-', ms=4, capsize=2, color='C1',
            label=f'X17 acceptance per energy bin'
        )
    if low_stat.any():
        for t in t_mid[low_stat]:
            ax2.axvline(t * 1e6, color='gray', lw=0.8, ls=':', alpha=0.5)
        ax2.plot([], [], color='gray', lw=0.8, ls=':', alpha=0.5,
                 label='< 10 X17 generated (low stat)')

    ax2.axhline(0, color='gray', lw=1.0, ls='-', zorder=0)

    # Veto window shading and annotation (both panels)
    veto_start_us = t_flash_s * 1e6
    veto_end_us   = t_veto_end_s * 1e6
    for ax in (ax1, ax2):
        ax.axvspan(veto_start_us, veto_end_us, color='red', alpha=0.15,
                   label='Veto window' if ax is ax2 else '_nolegend_')

    if VETO_ENABLED:
        annot = (f'Veto window\n'
                 f'{VETO_WINDOW_US:.1f} µs\n'
                 f'{veto_x17_fraction*100:.1f}% of X17\n'
                 f'candidates lost')
    else:
        annot = (f'Gamma flash trigger\n'
                 f'dead time = {dead_time_s*1e3:.3f} ms\n'
                 f'{veto_x17_fraction*100:.1f}% of X17\n'
                 f'candidates lost')
    ax2.annotate(
        annot,
        xy=(veto_end_us, 50), xytext=(veto_end_us * 2.5, 65),
        xycoords='data', textcoords='data',
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.4', fc='mistyrose', ec='red', alpha=0.85),
        arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
    )

    ax2.set_ylim(-5, 105)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylabel('X17 acceptance [%]', fontsize=11)
    ax2.set_xlabel(f'Neutron TOF [µs]  (flight path {DISTANCE_M:.1f} m)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(which='both', ls=':', lw=0.5, alpha=0.7)

    fig.tight_layout()
    return fig


def _add_energy_top_axis(ax, xmin_us, xmax_us):
    """Add a neutron energy axis (eV) along the top of ax."""
    energy_ticks_eV = np.array([1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
    t_ticks_s  = energy_eV_to_time_s(energy_ticks_eV, distance=DISTANCE_M)
    t_ticks_us = t_ticks_s * 1e6

    mask = (t_ticks_us >= xmin_us) & (t_ticks_us <= xmax_us)
    t_ticks_us       = t_ticks_us[mask]
    energy_ticks_eV  = energy_ticks_eV[mask]

    ax_top = ax.twiny()
    ax_top.set_xscale('log')
    ax_top.set_xlim(xmin_us, xmax_us)
    ax_top.set_xticks(t_ticks_us)
    ax_top.xaxis.set_major_formatter(plt.NullFormatter())
    ax_top.set_xticklabels([
        f"$10^{{{int(np.log10(E))}}}$" if E >= 1 else f"{E:g}"
        for E in energy_ticks_eV
    ])
    ax_top.set_xlabel('Neutron Energy [eV]', fontsize=10)
    ax_top.tick_params(axis='x', which='major', length=6)
    ax_top.tick_params(axis='x', which='minor', length=0)


if __name__ == '__main__':
    main()
