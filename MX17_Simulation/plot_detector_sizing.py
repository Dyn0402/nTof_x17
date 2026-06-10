"""
Detector sizing study: hit projections onto scintillator surfaces.

Shows where particles actually land on each scintillator layer, conditioned
on hitting upstream detectors. Used to determine the minimum required detector
dimensions (can bars on the edges be removed?).

Isotropic random single tracks from the He-3 capsule are used to sample the
full geometric acceptance independent of signal kinematics.

Outputs:
  sizing_config_A.png  — Config A: scint wall projections from MM and LS-1 tracks
  sizing_config_B.png  — Config B: scint wall and LS-2 projections from MM and BS tracks
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MX17_Simulator import MicromegasSimulation, SimConfig, ParticleGenerator, Propagator
from detector_config import GEO, SEED

# ── Sizing study parameters ───────────────────────────────────────────────────
N_EVENTS_SIZING = 100_000   # readout windows
N_RANDOM        = 3.0       # Poisson mean random tracks per window

SIM_BASE = dict(
    **GEO,
    n_signal=0.0,
    n_random=N_RANDOM,
    n_background_pairs=0.0,
    n_events=N_EVENTS_SIZING,
    coincidence_window=200.0,
    time_spread=2000.0,
    spatial_resolution=0.5,
    time_resolution=5.0,
    seed=SEED,
)

sim_A = MicromegasSimulation(SimConfig(**SIM_BASE,
                                        stack_order='standard',
                                        trigger_second_layer='liq_scint_1'))
sim_B = MicromegasSimulation(SimConfig(**SIM_BASE,
                                        stack_order='back_scint_first',
                                        trigger_second_layer='back_scint'))


# ── Data collection ───────────────────────────────────────────────────────────

def collect_projections(sim: MicromegasSimulation, n_events: int) -> list[dict]:
    """
    Propagate n_events readout windows.

    Returns one dict per (particle, arm) combination where the particle hits
    the MM on that arm:
      side:    arm index (0–3)
      mm:      (u, v) in arm-local coordinates [cm]
      sw:      (u, v) on scint wall or None
      ls1:     (u, v) on LS-1 or None
      ls2:     (u, v) on LS-2 or None
      has_bs:  True if particle also hit back_scint on this arm
    """
    cfg = sim.cfg
    gen = ParticleGenerator(cfg)
    prop = Propagator(sim.detectors, cfg)
    records: list[dict] = []

    for evt_i in range(n_events):
        if evt_i % 10_000 == 0:
            print(f'  event {evt_i:,} / {n_events:,}', flush=True)
        for p in gen.generate_event(evt_i * cfg.time_spread):
            p_hits = prop.propagate(p)

            # Group hits by arm side; keep first hit per detector type per side
            by_side: dict[int, dict] = {}
            for h in p_hits:
                s, dt = h.detector_id, h.detector_type
                if s not in by_side:
                    by_side[s] = {}
                if dt not in by_side[s]:
                    by_side[s][dt] = h.true_pos.copy()

            for side, det_hits in by_side.items():
                if 'mm' not in det_hits:
                    continue
                records.append({
                    'side':   side,
                    'mm':     det_hits['mm'],
                    'sw':     det_hits.get('scint_wall'),
                    'ls1':    det_hits.get('liq_scint_1'),
                    'ls2':    det_hits.get('liq_scint_2'),
                    'has_bs': 'back_scint' in det_hits,
                })

    return records


def get_uv(records: list[dict], target: str,
           require_bs: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (u, v) hit arrays on `target` detector.
    If require_bs, only include records where has_bs is True.
    target: 'sw', 'ls1', or 'ls2'
    """
    us, vs = [], []
    for r in records:
        if r.get(target) is None:
            continue
        if require_bs and not r['has_bs']:
            continue
        uv = r[target]
        us.append(uv[0])
        vs.append(uv[1])
    return np.array(us), np.array(vs)


# ── Plotting ──────────────────────────────────────────────────────────────────

def hit_panel(ax, u: np.ndarray, v: np.ndarray,
              half_u: float, half_v: float, title: str,
              cmap: str = 'inferno', n_bins: int = 60) -> None:
    """
    2D hit density on a detector face with the detector boundary overlaid.

    Colour = hit density (normalised). Cyan rectangle = current detector edge.
    Annotation shows hit count and fraction landing within the boundary.
    """
    margin = 1.2
    lim_u = half_u * margin
    lim_v = half_v * margin

    if len(u) < 10:
        ax.text(0.5, 0.5, 'Too few hits', transform=ax.transAxes,
                ha='center', va='center', fontsize=10)
        ax.set_title(title, fontsize=8)
        return

    h = ax.hist2d(u, v, bins=n_bins,
                   range=[[-lim_u, lim_u], [-lim_v, lim_v]],
                   cmap=cmap, density=True)
    plt.colorbar(h[3], ax=ax, shrink=0.8, pad=0.02, label='Hit density [a.u.]')

    # Current detector boundary
    ax.add_patch(Rectangle((-half_u, -half_v), 2 * half_u, 2 * half_v,
                            fc='none', ec='cyan', lw=2.0, zorder=5))
    ax.axhline(0, color='white', lw=0.5, alpha=0.3, ls=':')
    ax.axvline(0, color='white', lw=0.5, alpha=0.3, ls=':')

    inside = float(np.mean((np.abs(u) <= half_u) & (np.abs(v) <= half_v))) * 100
    ax.set_title(f'{title}\nn = {len(u):,}   {inside:.0f}% within boundary',
                  fontsize=8)
    ax.set_xlabel('u  (transverse, = Z_lab for arm 0)  [cm]', fontsize=7)
    ax.set_ylabel('v  (beam = Y_lab)  [cm]', fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-lim_u, lim_u)
    ax.set_ylim(-lim_v, lim_v)
    ax.set_aspect('equal')


# ── Collect ───────────────────────────────────────────────────────────────────
print(f'Collecting Config A projections ({N_EVENTS_SIZING:,} events) …')
recs_A = collect_projections(sim_A, N_EVENTS_SIZING)
print(f'  {len(recs_A):,} MM hits collected')

print(f'Collecting Config B projections ({N_EVENTS_SIZING:,} events) …')
recs_B = collect_projections(sim_B, N_EVENTS_SIZING)
print(f'  {len(recs_B):,} MM hits collected')


# ── Config A figure: scint wall projections ───────────────────────────────────
cfg_a = sim_A.cfg
sw_hu = cfg_a.scint_wall_size[0] / 2
sw_hv = cfg_a.scint_wall_size[1] / 2

fig_a, axes_a = plt.subplots(1, 2, figsize=(13, 6))
fig_a.suptitle(
    'Config A (standard: SW → LS-1 → LS-2 → BS)  —  Scint wall coverage\n'
    'Cyan rectangle = current detector boundary  |  '
    f'Isotropic single tracks from He capsule  ({N_EVENTS_SIZING:,} events)',
    fontsize=10, fontweight='bold',
)

# Left: all MM-hitting particles → scint wall
u, v = get_uv(recs_A, 'sw')
hit_panel(axes_a[0], u, v, sw_hu, sw_hv,
          'MM tracks → Scint wall\n(all particles reaching MM, showing SW hit position)')

# Right: particles also reaching LS-1 → scint wall (trigger coincidence region)
u, v = get_uv(recs_A, 'sw', require_bs=False)
# recompute with LS-1 filter manually
us_ls1, vs_ls1 = [], []
for r in recs_A:
    if r['sw'] is not None and r['ls1'] is not None:
        us_ls1.append(r['sw'][0])
        vs_ls1.append(r['sw'][1])
hit_panel(axes_a[1], np.array(us_ls1), np.array(vs_ls1), sw_hu, sw_hv,
          'LS-1 tracks → Scint wall\n(trigger-coincidence: MM ∩ LS-1, showing SW hit position)')

plt.tight_layout()
out_a = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'sizing', 'sizing_config_A.png')
fig_a.savefig(out_a, dpi=150, bbox_inches='tight')
print(f'[Output] sizing_config_A.png')


# ── Config B figure: scint wall and LS-2 projections ─────────────────────────
cfg_b = sim_B.cfg
sw_hu_b  = cfg_b.scint_wall_size[0]    / 2
sw_hv_b  = cfg_b.scint_wall_size[1]    / 2
ls2_hu_b = cfg_b.liq_scint_2_size[0]  / 2
ls2_hv_b = cfg_b.liq_scint_2_size[1]  / 2

fig_b, axes_b = plt.subplots(2, 2, figsize=(14, 12))
fig_b.suptitle(
    'Config B (back-scint first: SW → BS → LS-1 → LS-2)  —  Coverage study\n'
    'Cyan rectangle = current detector boundary  |  '
    f'Isotropic single tracks from He capsule  ({N_EVENTS_SIZING:,} events)',
    fontsize=10, fontweight='bold',
)

# Row 1: scint wall coverage
u, v = get_uv(recs_B, 'sw')
hit_panel(axes_b[0, 0], u, v, sw_hu_b, sw_hv_b,
          'MM tracks → Scint wall\n(all particles reaching MM)')

u_bs, v_bs = [], []
for r in recs_B:
    if r['sw'] is not None and r['has_bs']:
        u_bs.append(r['sw'][0])
        v_bs.append(r['sw'][1])
hit_panel(axes_b[0, 1], np.array(u_bs), np.array(v_bs), sw_hu_b, sw_hv_b,
          'Back scint tracks → Scint wall\n(trigger-coincidence: MM ∩ SW ∩ BS)')

# Row 2: LS-2 coverage
u, v = get_uv(recs_B, 'ls2')
hit_panel(axes_b[1, 0], u, v, ls2_hu_b, ls2_hv_b,
          'MM tracks → LS-2\n(all particles reaching MM)')

u_bs2, v_bs2 = [], []
for r in recs_B:
    if r['ls2'] is not None and r['has_bs']:
        u_bs2.append(r['ls2'][0])
        v_bs2.append(r['ls2'][1])
hit_panel(axes_b[1, 1], np.array(u_bs2), np.array(v_bs2), ls2_hu_b, ls2_hv_b,
          'Back scint tracks → LS-2\n(calorimetry-complete: MM ∩ BS ∩ LS-2)')

plt.tight_layout()
out_b = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'sizing', 'sizing_config_B.png')
fig_b.savefig(out_b, dpi=150, bbox_inches='tight')
print(f'[Output] sizing_config_B.png')

plt.show()
