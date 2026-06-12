"""
Best-case X17 spectrum for the HIGH-ENERGY (MeV-region) measurement
===================================================================
Produces the stacked angular-separation spectrum (IPC background + X17 on
top, expected counts over the full exposure) using the MeV-region production
statistics from the full Geant4 neutron campaign (Run B-full, 5e8 neutrons,
docs/report/mev_note.tex), instead of the thermal-anchored numbers baked
into MX17_Simulator (IPC_PER_PULSE etc.).

Production statistics (MX17_Full_Geant/analysis/mev/mev_rates.json):
    full range (1 meV - 100 MeV) : IPC 6.79e-2 /pulse, X17 1.70e-3 /pulse
                                   -> 32.7 X17/day produced
    0.2-2 MeV window             : IPC 4.66e-2 /pulse, X17 1.16e-3 /pulse
                                   -> 22.5 X17/day produced
These are produced-pair rates BEFORE acceptance; the fast MC applies
geometry, trigger and resolution itself (same convention as
run_significance_study.py).

Best-case assumptions
---------------------
  * full-range production statistics (ENERGY_WINDOW = 'full')
  * 30-day exposure (RUN_DAYS x PULSES_PER_DAY from MX17_Simulator)
  * no gamma-flash veto / DAQ dead-time loss (the 0.2-2 MeV gate sits
    1.0-3.2 us after the flash; best case assumes the detectors have
    recovered -- see dead_time_sim.py for the pessimistic counterpart)
  * no random combinatorial background (n_random = 0): the in-gate single
    rates (3e5 (n,p)t + 9e4 H-captures per pulse) assumed rejected
  * Config A (standard stack) -- highest double-trigger efficiency, which
    maximises the angle-channel statistics
  * wide 200 ns coincidence window (essentially no genuine pair lost)
  * Geant4-derived response smearing from the newest response JSON found
    (drop the new-geometry export from analyze_pairs.py --export-response
    into this directory, or edit RESPONSE_CANDIDATES below)

Run:  venv/bin/python run_best_case_high_energy.py
"""

import io
import os
import sys
import json
import contextlib
import dataclasses

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MX17_Simulator import (MicromegasSimulation, TOTAL_PULSES, RUN_DAYS,
                            PULSES_PER_DAY, IPC_PER_PULSE as IPC_PER_PULSE_OLD,
                            X17_PER_PULSE as X17_PER_PULSE_OLD)
from detector_config import cfg_A, cfg_B, N_WORKERS
from run_significance_study import z_asimov, best_window

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Best-case knobs ──────────────────────────────────────────────────────────
ENERGY_WINDOW = 'full'        # 'full' (1 meV-100 MeV) or 'window' (0.2-2 MeV)
CONFIG        = 'A'           # 'A' (standard stack) or 'B' (back-scint first)
N_MC_EVENTS   = 400_000       # MC readout windows per source
BG_SYST       = 0.10          # fractional background systematic for Z quote
SEED          = 314

ANGLE_BINS = np.arange(0.0, 184.0, 4.0)     # deg
MASS_BINS  = np.arange(0.0, 26.0, 1.0)      # MeV

# ── High-energy production rates (Run B-full) ────────────────────────────────
# Values frozen from mev_rates.json (12 June 2026); overridden by the live
# JSON below when the Geant4 repo is present.
MEV_RATES_JSON = os.path.expanduser(
    '~/CLionProjects/MX17_Full_Geant/analysis/mev/mev_rates.json')
HIGH_E_RATES = {
    'full':   {'ipc_per_pulse': 6.786e-2, 'x17_per_pulse': 1.696e-3,
               'x17_per_day': 32.72, 'label': 'full range (1 meV-100 MeV)'},
    'window': {'ipc_per_pulse': 4.657e-2, 'x17_per_pulse': 1.164e-3,
               'x17_per_day': 22.46, 'label': '0.2-2 MeV window'},
}


def load_high_e_rates():
    """Refresh the frozen rates from mev_rates.json if it is available."""
    rates = {k: dict(v) for k, v in HIGH_E_RATES.items()}
    if os.path.exists(MEV_RATES_JSON):
        with open(MEV_RATES_JSON) as fp:
            d = json.load(fp)
        rates['full']['ipc_per_pulse'] = d['ipc_per_pulse']
        rates['full']['x17_per_pulse'] = d['x17_per_pulse']
        rates['full']['x17_per_day']   = d['x17_per_day']
        w = d['window_0p2_2MeV']
        rates['window']['ipc_per_pulse'] = w['ipc_per_pulse']
        rates['window']['x17_per_day']   = w['x17_per_day']
        rates['window']['x17_per_pulse'] = w['x17_per_day'] / d['pulses_per_day']
        print(f"[Rates] Loaded live rates from {MEV_RATES_JSON}")
    else:
        print(f"[Rates] {MEV_RATES_JSON} not found -- using frozen values")
    return rates


# ── Geant4 detector response: prefer the new-geometry export ─────────────────
# First existing file wins.  The new-geometry response from the currently
# running analyze_pairs.py (pairs_v2_step_target) should be dropped here under
# one of the first names; until then the old-geometry tables are used.
RESPONSE_CANDIDATES = [
    os.path.join(_HERE, 'geant4_response_new_geom.json'),
    os.path.join(_HERE, 'geant4_response_step_target.json'),
    os.path.join(_HERE, 'geant4_response.json'),
    os.path.join(_HERE, 'geant4_response_old_geom.json'),
]


def pick_response():
    for path in RESPONSE_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


# ── MC running (run_significance_study pattern) ──────────────────────────────

def run_source(cfg, source, seed):
    """Run one config with only `source` pairs; return reco samples and the
    produced-pair count for exposure scaling."""
    kw = dict(n_signal=0.0, n_background_pairs=0.0, n_random=0.0,
              n_events=N_MC_EVENTS, seed=seed)
    if source == 'signal':
        kw['n_signal'] = 1.0
    else:
        kw['n_background_pairs'] = 1.0
    c = dataclasses.replace(cfg, **kw)
    sim = MicromegasSimulation(c)
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run(n_workers=N_WORKERS)

    key = 'signal' if source == 'signal' else 'background_pair'
    n_prod = (sim.pair_stats['signal_produced'] if source == 'signal'
              else sim.pair_stats['bg_produced'])

    sel_d  = np.array([s == key for s in sim.pair_sources_double], bool)
    sel_bs = np.array([s == key for s in sim.pair_sources_both_bs], bool)
    return {
        'n_produced':  n_prod,
        'angles_trig': np.asarray(sim.angular_separations_double)[sel_d]
                       if len(sel_d) else np.array([]),
        'angles_bs':   np.asarray(sim.angular_separations_both_bs)[sel_bs]
                       if len(sel_bs) else np.array([]),
        'masses_bs':   np.asarray(sim.inv_masses_both_bs)[sel_bs]
                       if len(sel_bs) else np.array([]),
        'pair_stats':  dict(sim.pair_stats),
    }


def expected_hist(samples, bins, n_produced_mc, n_produced_exp):
    h, _ = np.histogram(samples, bins=bins)
    return h * (n_produced_exp / max(n_produced_mc, 1))


# ── Plotting ─────────────────────────────────────────────────────────────────

def stacked_channel(fig, gs_main, gs_ratio, cen, width, b, s, xlabel, unit,
                    z_stat, z_syst, channel_title, mark_x=None):
    """One stacked channel (IPC + X17 on top) with a per-bin S/sqrt(B) panel."""
    ax  = fig.add_subplot(gs_main)
    axr = fig.add_subplot(gs_ratio, sharex=ax)

    ax.bar(cen, b, width=width, color='#9dc3e6', edgecolor='#4a90d9',
           lw=0.4, label=f'IPC background  (N={b.sum():,.0f})')
    ax.bar(cen, s, width=width, bottom=b, color='#e84040',
           edgecolor='#a02020', lw=0.4,
           label=f'X17 on top, unscaled  (N={s.sum():,.0f})')
    tot = s + b
    ax.errorbar(cen, tot, yerr=np.sqrt(tot), fmt='none', ecolor='k',
                elinewidth=0.7, capsize=0, alpha=0.6,
                label=r'$\pm\sqrt{N}$ expected stat. error')
    if mark_x is not None:
        ax.axvline(mark_x[0], color='red', ls=':', lw=1.2, label=mark_x[1])
    ax.set_ylabel(f'Expected pairs / {RUN_DAYS} d / {unit}')
    ax.set_title(f'{channel_title}\n'
                 f'Z = {z_stat:.2f}$\\sigma$ stat-only,  '
                 f'{z_syst:.2f}$\\sigma$ with {BG_SYST:.0%} bg syst')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.setp(ax.get_xticklabels(), visible=False)

    with np.errstate(divide='ignore', invalid='ignore'):
        zbin = np.where(b > 0, s / np.sqrt(b), 0.0)
    axr.bar(cen, zbin, width=width, color='#e84040', alpha=0.8)
    axr.axhline(0, color='k', lw=0.6)
    axr.set_xlabel(xlabel)
    axr.set_ylabel(r'S/$\sqrt{B}$ / bin')
    axr.grid(alpha=0.3)
    return ax


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    rates = load_high_e_rates()[ENERGY_WINDOW]
    cfg   = {'A': cfg_A, 'B': cfg_B}[CONFIG]

    response = pick_response()
    if response is not None:
        cfg = dataclasses.replace(cfg, g4_response_path=response)
    resp_name = os.path.basename(response) if response else 'none (unsmeared)'

    ipc_pp, x17_pp = rates['ipc_per_pulse'], rates['x17_per_pulse']
    n_x17_exp = x17_pp * TOTAL_PULSES
    n_ipc_exp = ipc_pp * TOTAL_PULSES

    print('=' * 72)
    print('BEST-CASE HIGH-ENERGY X17 MEASUREMENT  (MeV-region statistics)')
    print('=' * 72)
    print(f'  Production window  : {rates["label"]}')
    print(f'  Exposure           : {RUN_DAYS} days = {TOTAL_PULSES:,} pulses '
          f'({PULSES_PER_DAY:.3g}/day)')
    print(f'  IPC produced       : {ipc_pp:.3e}/pulse -> {n_ipc_exp:,.0f}')
    print(f'  X17 produced       : {x17_pp:.3e}/pulse -> {n_x17_exp:,.0f} '
          f'({rates["x17_per_day"]:.1f}/day)')
    print(f'  vs old (thermal-anchored) rates: IPC x{ipc_pp/IPC_PER_PULSE_OLD:.2f}, '
          f'X17 x{x17_pp/X17_PER_PULSE_OLD:.2f}')
    print(f'  Detector config    : {CONFIG} '
          f'({"standard stack" if CONFIG == "A" else "back-scint first"})')
    print(f'  Geant4 response    : {resp_name}')
    print(f'  MC statistics      : {N_MC_EVENTS:,} windows per source\n')

    print('Running signal MC ...')
    sig = run_source(cfg, 'signal', SEED)
    print('Running IPC MC ...')
    bg  = run_source(cfg, 'ipc', SEED + 1)

    s_ang = expected_hist(sig['angles_trig'], ANGLE_BINS,
                          sig['n_produced'], n_x17_exp)
    b_ang = expected_hist(bg['angles_trig'],  ANGLE_BINS,
                          bg['n_produced'],  n_ipc_exp)
    s_angbs = expected_hist(sig['angles_bs'], ANGLE_BINS,
                            sig['n_produced'], n_x17_exp)
    b_angbs = expected_hist(bg['angles_bs'],  ANGLE_BINS,
                            bg['n_produced'],  n_ipc_exp)
    s_mas = expected_hist(sig['masses_bs'], MASS_BINS,
                          sig['n_produced'], n_x17_exp)
    b_mas = expected_hist(bg['masses_bs'],  MASS_BINS,
                          bg['n_produced'],  n_ipc_exp)

    z_angle      = z_asimov(s_ang, b_ang)
    z_angle_syst = z_asimov(s_ang, b_ang, BG_SYST)
    z_mass       = z_asimov(s_mas, b_mas)
    z_mass_syst  = z_asimov(s_mas, b_mas, BG_SYST)
    # combined: mass channel + angle channel on the non-BS remainder
    s_ang_nbs = np.clip(s_ang - s_angbs, 0, None)
    b_ang_nbs = np.clip(b_ang - b_angbs, 0, None)
    z_comb      = float(np.hypot(z_mass,      z_asimov(s_ang_nbs, b_ang_nbs)))
    z_comb_syst = float(np.hypot(z_mass_syst,
                                 z_asimov(s_ang_nbs, b_ang_nbs, BG_SYST)))

    st = sig['pair_stats']
    print('\n--- Results ---')
    print(f'  Both-MM eff (signal)     : {st["signal_efficiency"]*100:.1f}%')
    print(f'  Trigger eff (of both-MM) : {st["trigger_efficiency"]*100:.1f}%')
    print(f'  Full calorimetry         : {st["calorimetry_complete_fraction"]*100:.1f}%')
    print(f'  Angle channel: S={s_ang.sum():,.0f}  B={b_ang.sum():,.0f}  '
          f'Z={z_angle:.2f} (stat), {z_angle_syst:.2f} ({BG_SYST:.0%} syst)')
    print(f'  Mass  channel: S={s_mas.sum():,.0f}  B={b_mas.sum():,.0f}  '
          f'Z={z_mass:.2f} (stat), {z_mass_syst:.2f} ({BG_SYST:.0%} syst)')
    print(f'  Combined     : Z={z_comb:.2f} (stat), {z_comb_syst:.2f} '
          f'({BG_SYST:.0%} syst)')
    lo, hi, S, B, zsb = best_window(s_ang, b_ang, ANGLE_BINS)
    print(f'  Best angle window: {lo:.0f}-{hi:.0f} deg  '
          f'S={S:.0f}  B={B:.0f}  S/sqrt(B)={zsb:.2f}')

    # ── Figure: angle channel (left) + mass channel (right), stacked ────────
    a_cen = 0.5 * (ANGLE_BINS[:-1] + ANGLE_BINS[1:])
    m_cen = 0.5 * (MASS_BINS[:-1]  + MASS_BINS[1:])

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.06, wspace=0.22)

    stacked_channel(
        fig, gs[0, 0], gs[1, 0], a_cen, 4.0, b_ang, s_ang,
        'Reco opening angle [deg]', '4°', z_angle, z_angle_syst,
        'Angle channel — scint double trigger')
    stacked_channel(
        fig, gs[0, 1], gs[1, 1], m_cen, 1.0, b_mas, s_mas,
        'Reco invariant mass [MeV]', 'MeV', z_mass, z_mass_syst,
        'Mass channel — both back-scint',
        mark_x=(16.8, r'$m_{X17}$ = 16.8 MeV'))

    fig.suptitle(
        f'Best-case high-energy X17 measurement — {rates["label"]}, '
        f'{RUN_DAYS} d exposure, Config {CONFIG}\n'
        f'produced: {n_x17_exp:,.0f} X17 / {n_ipc_exp:,.0f} IPC '
        f'({rates["x17_per_day"]:.1f} X17/day, Run B-full Geant4)  |  '
        f'no dead-time/veto loss, no combinatorial bg  |  '
        f'response: {resp_name}  |  '
        f'combined Z = {z_comb:.1f}$\\sigma$ stat, '
        f'{z_comb_syst:.1f}$\\sigma$ w/ {BG_SYST:.0%} syst',
        fontsize=11)

    out_dir = os.path.join(_HERE, 'results', 'best_case_high_energy')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir,
                       f'best_case_high_energy_{ENERGY_WINDOW}_cfg{CONFIG}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f'\n[Output] Saved {out}')
    plt.show()


if __name__ == '__main__':
    main()
