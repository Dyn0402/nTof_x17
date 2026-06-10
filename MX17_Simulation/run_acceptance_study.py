"""
Scintillator Stack Acceptance Study
====================================
Compares two detector stack orderings for X17 e+e- trigger efficiency and
calorimetry completeness.

Config A — standard (Geant4 default):
    MM → scint_wall(3mm) → LS-1(2cm) → LS-2(2cm) → back_scint(2cm)
    Trigger: scint_wall AND LS-1 on ≥2 arms

Config B — back_scint first:
    MM → scint_wall(3mm) → back_scint(2cm) → LS-1(2cm) → LS-2(2cm)
    Trigger: scint_wall AND back_scint on ≥2 arms

Metrics (for signal X17 e+e- pairs):
  1. trigger_miss_fraction: fraction of pairs where both e± hit MM
       but the event fails the scint_double trigger.
       Indicates what fraction of detectable X17 events we never read out.

  2. calorimetry_incomplete_fraction: fraction of pairs where both hit MM
       but ≥1 particle misses back_scint on its arm.
       Indicates what fraction of detectable events lack full calorimetry.

Particles are generated uniformly in the He-3 capsule volume
  (cylinder: r=1.5 cm, half-length=4 cm along beam = Y axis).
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Allow running from any directory
sys.path.insert(0, os.path.dirname(__file__))
from MX17_Simulator import (
    MicromegasSimulation, SimConfig,
    GaussianSpectrum, calculate_solid_angle_coverage,
)
from detector_config import cfg_A as cfg_a, cfg_B as cfg_b, N_WORKERS, N_EVENTS, SEED


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("Config A: standard order  (MM → scint_wall → LS-1 → LS-2 → back_scint)")
print("          Trigger: scint_wall AND LS-1 on ≥2 arms")
print("=" * 70)
sim_a = MicromegasSimulation(cfg_a)
sim_a.run(n_workers=N_WORKERS)

print()
print("=" * 70)
print("Config B: back_scint first (MM → scint_wall → back_scint → LS-1 → LS-2)")
print("          Trigger: scint_wall AND back_scint on ≥2 arms")
print("=" * 70)
sim_b = MicromegasSimulation(cfg_b)
sim_b.run(n_workers=N_WORKERS)


# ──────────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────────

def _stack_str(sim: MicromegasSimulation) -> str:
    """One-line summary of detector positions."""
    d_mm, d_sc, d_ls1, d_ls2, d_bk = sim._get_stack_distances()
    return (f"MM@{d_mm:.1f}  SW@{d_sc:.2f}  "
            f"LS1@{d_ls1:.2f}  LS2@{d_ls2:.2f}  BS@{d_bk:.2f}  [cm]")


def print_results(label: str, sim: MicromegasSimulation):
    ps  = sim.pair_stats
    st  = sim.scint_trigger_stats
    cfg = sim.cfg
    sl  = cfg.trigger_second_layer

    print(f"\n{'─' * 70}")
    print(f"  {label}")
    print(f"{'─' * 70}")
    print(f"  Stack: {_stack_str(sim)}")
    print(f"  Trigger 2nd layer: {sl}")

    print(f"\n  Per-track acceptance (all particle types hitting MM):")
    n_mm = st['n_tracks_mm']
    print(f"    MM hits                    : {n_mm:>8,}")
    print(f"    + scint wall               : {st['n_tracks_scint_wall']:>8,}"
          f"  ({st['scint_wall_efficiency']*100:5.1f}%  of MM)")
    ly2_label = 'back_scint' if sl == 'back_scint' else 'LS-1      '
    print(f"    + {ly2_label} [trigger]   : {st['n_tracks_scint_single']:>8,}"
          f"  ({st['scint_single_efficiency']*100:5.1f}%  of MM)")
    print(f"    + back_scint               : {st['n_tracks_back_scint']:>8,}"
          f"  ({st['back_scint_efficiency']*100:5.1f}%  of MM)")

    print(f"\n  Signal X17 pair metrics (e+e- ~120° opening angle):")
    n_prod = ps['signal_produced']
    n_det  = ps['signal_detected']
    n_trig = ps['signal_triggered']
    n_calo = ps['signal_both_backscint']
    print(f"    Pairs generated            : {n_prod:>8,}")
    print(f"    Both hit MM                : {n_det:>8,}  ({ps['signal_efficiency']*100:5.1f}%  of generated)")
    print(f"    + scint_double trigger     : {n_trig:>8,}  ({ps['trigger_efficiency']*100:5.1f}%  of both-MM)")
    print(f"    Trigger miss fraction      : {ps['trigger_miss_fraction']*100:5.1f}%")
    print(f"    Both hit back_scint        : {n_calo:>8,}  ({ps['calorimetry_complete_fraction']*100:5.1f}%  of both-MM)")
    print(f"    Calorimetry incomplete     : {ps['calorimetry_incomplete_fraction']*100:5.1f}%")


print_results("Config A — standard (back_scint after LS)", sim_a)
print_results("Config B — back_scint first (before LS)",   sim_b)


# ──────────────────────────────────────────────────────────────────────────────
# Comparison table
# ──────────────────────────────────────────────────────────────────────────────

a, b = sim_a.pair_stats, sim_b.pair_stats
sa, sb = sim_a.scint_trigger_stats, sim_b.scint_trigger_stats

print(f"\n{'=' * 70}")
print("COMPARISON SUMMARY")
print(f"{'=' * 70}")
print(f"  {'Metric':<35s}  {'Config A':>9s}  {'Config B':>9s}  {'Δ (B−A)':>9s}")
print(f"  {'─' * 67}")

def _row(name, va, vb, pct=True):
    suffix = '%' if pct else ''
    delta  = vb - va
    sign   = '+' if delta >= 0 else ''
    if pct:
        print(f"  {name:<35s}  {va*100:8.1f}%  {vb*100:8.1f}%  {sign}{delta*100:.1f}%")
    else:
        print(f"  {name:<35s}  {va:9.3f}  {vb:9.3f}  {sign}{delta:.3f}")

_row('MM detection efficiency',          a['signal_efficiency'],              b['signal_efficiency'])
_row('Scint_wall per-track eff.',        sa['scint_wall_efficiency'],         sb['scint_wall_efficiency'])
_row('Trigger layer per-track eff.',     sa['scint_single_efficiency'],       sb['scint_single_efficiency'])
_row('Back-scint per-track eff.',        sa['back_scint_efficiency'],         sb['back_scint_efficiency'])
_row('Trigger efficiency (both-MM)',     a['trigger_efficiency'],             b['trigger_efficiency'])
_row('Trigger miss fraction',            a['trigger_miss_fraction'],          b['trigger_miss_fraction'])
_row('Calorimetry complete (both-MM)',   a['calorimetry_complete_fraction'],  b['calorimetry_complete_fraction'])
_row('Calorimetry incomplete fraction',  a['calorimetry_incomplete_fraction'],b['calorimetry_incomplete_fraction'])

print()
print(f"  Config A stack: {_stack_str(sim_a)}")
print(f"  Config B stack: {_stack_str(sim_b)}")
print(f"  Events: {N_EVENTS:,}  |  Signal per event (mean): {cfg_a.n_signal}")
print(f"  He capsule: r={cfg_a.he_radius_cm} cm, ±{cfg_a.he_half_length_cm} cm along beam")


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def _plot_eff_into_ax(sim: MicromegasSimulation, ax, title: str):
    st  = sim.scint_trigger_stats
    cfg = sim.cfg
    sl  = cfg.trigger_second_layer
    if sl == 'back_scint':
        trig_label = f'+ back_scint\n[trigger]\n({cfg.back_scint_size_u:.0f}×{cfg.back_scint_size_v:.0f}×2 cm)'
    else:
        trig_label = f'+ LS-1\n[trigger]\n({cfg.liq_scint_size[0]:.0f}×{cfg.liq_scint_size[1]:.0f} cm)'
    sw = cfg.scint_wall_size
    labels = [
        f'MM\n({cfg.detector_size[0]:.0f}×{cfg.detector_size[1]:.0f} cm)',
        f'+ scint wall\n({sw[0]:.0f}×{sw[1]:.0f} cm)',
        trig_label,
        f'+ back_scint\n({cfg.back_scint_size_u:.0f}×{cfg.back_scint_size_v:.0f}×2 cm)',
    ]
    values = [
        st['n_tracks_mm'],
        st['n_tracks_scint_wall'],
        st['n_tracks_scint_single'],
        st['n_tracks_back_scint'],
    ]
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor='k', lw=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f'{val:,}', ha='center', va='bottom', fontsize=8)
    effs = [1.0, st['scint_wall_efficiency'], st['scint_single_efficiency'],
            st['back_scint_efficiency']]
    for bar, eff in zip(bars, effs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f'{eff*100:.1f}%', ha='center', va='center',
                fontsize=8, color='white', fontweight='bold')
    ax.set_ylabel('Track count')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')


# Figure 1: geometry diagrams + per-track acceptance bars (2×2)
fig, axes = plt.subplots(2, 2, figsize=(12, 9.5),
                          gridspec_kw={'width_ratios': [1, 1.5], 'hspace': 0.2, 'wspace': 0.15,
                                       'left': 0.05, 'right': 0.99, 'top': 0.97,
                                       'bottom': 0.07})

sim_a.plot_geometry(ax=axes[0, 0])
axes[0, 0].set_title('Config A — standard')
_plot_eff_into_ax(sim_a, axes[0, 1], 'Config A  |  per-track acceptance')

sim_b.plot_geometry(ax=axes[1, 0])
axes[1, 0].set_title('Config B — back_scint first')
_plot_eff_into_ax(sim_b, axes[1, 1], 'Config B  |  per-track acceptance')

fig.savefig(os.path.join(os.path.dirname(__file__), 'results', 'acceptance', 'acceptance_study.png'),
            dpi=150, bbox_inches='tight')
print(f"\n[Output] Saved acceptance_study.png")

# Summary bar chart of the three key fractions
fig_summary, ax_sum = plt.subplots(figsize=(6, 5))
metrics = [
    ('MM detection\nefficiency',    a['signal_efficiency'],             b['signal_efficiency']),
    ('Trigger\nefficiency',         a['trigger_efficiency'],            b['trigger_efficiency']),
    ('Calo.\ncomplete fraction',    a['calorimetry_complete_fraction'], b['calorimetry_complete_fraction']),
]
x     = np.arange(len(metrics))
width = 0.35
labels_m = [m[0] for m in metrics]
vals_a   = [m[1] * 100 for m in metrics]
vals_b   = [m[2] * 100 for m in metrics]

bars_a = ax_sum.bar(x - width/2, vals_a, width, label='Config A (standard)',
                    color='#2196F3', alpha=0.85, edgecolor='k', lw=0.5)
bars_b = ax_sum.bar(x + width/2, vals_b, width, label='Config B (bs first)',
                    color='#E91E63', alpha=0.85, edgecolor='k', lw=0.5)

for bar in list(bars_a) + list(bars_b):
    ax_sum.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)

ax_sum.set_xticks(x)
ax_sum.set_xticklabels(labels_m, fontsize=9)
ax_sum.set_ylabel('Fraction [%]')
ax_sum.set_title('Signal X17 pair acceptance metrics: Config A vs B\n'
                 '(He capsule source, both particles hit MM as denominator)')
ax_sum.legend(fontsize=9)
ax_sum.grid(True, alpha=0.3, axis='y')
ax_sum.set_ylim(0, min(110, max(vals_a + vals_b) * 1.15))
plt.tight_layout()
fig_summary.savefig(os.path.join(os.path.dirname(__file__), 'results', 'acceptance', 'acceptance_study_summary.png'),
                    dpi=150, bbox_inches='tight')
print(f"[Output] Saved acceptance_study_summary.png")

plt.show()
