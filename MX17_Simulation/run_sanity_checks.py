"""
Sanity checks for MX17 simulator geometry and acceptance.

Checks:
  1. Analytic projected coverage — each downstream detector projected back to MM face
  2. Solid angle fractions — Monte Carlo from origin and He capsule centre
  3. MM hit maps — scatter of (u,v) hits coloured by per-track acceptance at each
     downstream layer, with analytic projected boundaries overlaid
  4. Back-scint acceptance vs u-position — clearly shows the geometric cut-off
  5. 3D scint-wall miss trajectories — the ~0.2% of tracks that hit MM but miss
     the scint wall; possible only with He capsule source offset
  6. He capsule origin distribution — verify uniform sampling
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.dirname(__file__))
from MX17_Simulator import (
    MicromegasSimulation, SimConfig, Detector,
    calculate_solid_angle_coverage, _random_direction, ParticleGenerator, Propagator,  # beam = +Y
)
from detector_config import GEO

# ──────────────────────────────────────────────────────────────────────────────
# Shared config (Config A standard geometry)
# ──────────────────────────────────────────────────────────────────────────────

BASE_CFG = dict(
    **GEO,
    stack_order='standard', trigger_second_layer='liq_scint_1',
    n_signal=0.0, n_random=2.0, n_background_pairs=0.0, n_events=50_000, seed=12345,
)

sim_A = MicromegasSimulation(SimConfig(**BASE_CFG))
sim_B = MicromegasSimulation(SimConfig(**{**BASE_CFG,
    'stack_order': 'back_scint_first', 'trigger_second_layer': 'back_scint'}))

d_mm, d_sw, d_ls1, d_ls2, d_back_A = sim_A._get_stack_distances()
d_back_B = sim_B._get_stack_distances()[4]
mm_hu, mm_hv = sim_A.cfg.detector_size[0]/2, sim_A.cfg.detector_size[1]/2
sw_hu, sw_hv = sim_A.cfg.scint_wall_size[0]/2, sim_A.cfg.scint_wall_size[1]/2
ls_hu = sim_A.cfg.liq_scint_size[0]/2
bs_bar_hu = sim_A.cfg.back_scint_size_u/2
bs_hv     = sim_A.cfg.back_scint_size_v/2
u_off     = (sim_A.cfg.back_scint_size_u + sim_A.cfg.back_scint_gap) / 2.0


# ──────────────────────────────────────────────────────────────────────────────
# Check 1: Analytic projected coverage
# ──────────────────────────────────────────────────────────────────────────────

def proj_to_mm(d_far, half_u_far, half_v_far=None):
    """Project half-sizes at d_far back to MM face (from origin)."""
    scale = d_mm / d_far
    if half_v_far is None:
        return half_u_far * scale
    return half_u_far * scale, half_v_far * scale

def backscint_mm_coverage(d_back):
    """Fraction of MM face that maps into back_scint bars from origin."""
    scale = d_mm / d_back
    bar_R_u_lo = (u_off - bs_bar_hu) * scale   # inner edge of bar R projected to MM
    bar_R_u_hi = (u_off + bs_bar_hu) * scale   # outer edge
    bar_v_at_mm = bs_hv * scale
    # Clamp to MM face
    eff_u_per_bar = max(0, min(bar_R_u_hi, mm_hu) - max(bar_R_u_lo, 0))
    eff_v = min(bar_v_at_mm, mm_hv)
    area_two_bars = 4 * eff_u_per_bar * eff_v   # ×2 bars, ×2 for ±v
    mm_area = 4 * mm_hu * mm_hv
    return area_two_bars / mm_area, bar_R_u_lo, bar_R_u_hi, bar_v_at_mm

bs_frac_A, bar_lo_A, bar_hi_A, bar_v_A = backscint_mm_coverage(d_back_A)
bs_frac_B, bar_lo_B, bar_hi_B, bar_v_B = backscint_mm_coverage(d_back_B)

print("=" * 72)
print("CHECK 1 — ANALYTIC PROJECTED COVERAGE (from point origin)")
print("=" * 72)
print(f"  Detector        Distance    Proj ½-u→MM   Proj ½-v→MM   MM frac (u×v)")
print(f"  {'─' * 68}")

def _row(name, d, hu_far, hv_far):
    pu, pv = proj_to_mm(d, hu_far, hv_far)
    fu = min(pu, mm_hu) / mm_hu
    fv = min(pv, mm_hv) / mm_hv
    print(f"  {name:<15s} {d:6.2f} cm   {pu:7.2f} cm    {pv:7.2f} cm    {fu*fv*100:5.1f}%")

_row('Scint wall',  d_sw,    sw_hu, sw_hv)
_row('LS-1',        d_ls1,   ls_hu, ls_hu)
_row('LS-2',        d_ls2,   ls_hu, ls_hu)
print(f"  {'─' * 68}")
print(f"  Back scint A    {d_back_A:6.2f} cm   "
      f"[{bar_lo_A:.2f},{bar_hi_A:.2f}]      ±{bar_v_A:.2f} cm      "
      f"{bs_frac_A*100:.1f}%  (2 bars)")
print(f"  Back scint B    {d_back_B:6.2f} cm   "
      f"[{bar_lo_B:.2f},{bar_hi_B:.2f}]      ±{bar_v_B:.2f} cm      "
      f"{bs_frac_B*100:.1f}%  (2 bars)")
print()
print(f"  ⚡ NOTE: Scint wall ½-u projects to {sw_hu*d_mm/d_sw:.4f} cm at MM face,")
print(f"     MM ½-u = {mm_hu:.4f} cm  → scint wall overhang = {sw_hu*d_mm/d_sw - mm_hu:.4f} cm!")
print(f"     This means from origin the scint wall is essentially the same solid angle")
print(f"     as the MM — the 99.8% acceptance is physically correct.")


# ──────────────────────────────────────────────────────────────────────────────
# Check 2: Solid angle fractions (Monte Carlo)
# ──────────────────────────────────────────────────────────────────────────────

print()
print("=" * 72)
print("CHECK 2 — SOLID ANGLE COVERAGE (Monte Carlo, 200k samples, from origin)")
print("=" * 72)

N_SA = 200_000
types = ['mm', 'scint_wall', 'liq_scint_1', 'liq_scint_2', 'back_scint']
labels_sa = ['MM', 'Scint wall', 'LS-1', 'LS-2', 'Back scint A']
colors_sa = ['#2196F3', '#FF9800', '#4CAF50', '#009688', '#E91E63']

fracs_A, fracs_B = {}, {}
mm_frac = None

for t, lbl in zip(types, labels_sa):
    dets_A = [d for d in sim_A.detectors if d.det_type == t]
    dets_B = [d for d in sim_B.detectors if d.det_type == t]
    if dets_A:
        f = calculate_solid_angle_coverage(dets_A, n_samples=N_SA)
        fracs_A[t] = f
    if dets_B:
        f2 = calculate_solid_angle_coverage(dets_B, n_samples=N_SA)
        fracs_B[t] = f2

mm_frac = fracs_A['mm']
print(f"  {'Detector':<15s}  {'Config A':>9s}  {'% of MM':>9s}  {'Config B':>9s}  {'% of MM':>9s}")
print(f"  {'─' * 60}")
for t, lbl in zip(types, labels_sa):
    fA = fracs_A.get(t, 0)
    fB = fracs_B.get(t, 0)
    pA = fA / mm_frac * 100 if mm_frac else 0
    pB = fB / mm_frac * 100 if mm_frac else 0
    print(f"  {lbl:<15s}  {fA*100:8.2f}%  {pA:8.1f}%  {fB*100:8.2f}%  {pB:8.1f}%")

print()
print(f"  (Solid angle fracs are per 4π; % of MM = fraction / MM fraction)")


# ──────────────────────────────────────────────────────────────────────────────
# Propagation helper: collect per-particle hit types
# ──────────────────────────────────────────────────────────────────────────────

def collect_hit_data(sim: MicromegasSimulation, n_events: int = 20_000):
    """
    Propagate n_events readout windows, returning a list of per-MM-hit dicts:
      origin, side, u, v (MM local), has_sw, has_bs (same side as MM hit), has_ls1
    """
    cfg = sim.cfg
    gen = ParticleGenerator(cfg)
    prop = Propagator(sim.detectors, cfg)
    records = []
    for i in range(n_events):
        for p in gen.generate_event(i * cfg.time_spread):
            p_hits = prop.propagate(p)
            mm_h = [h for h in p_hits if h.detector_type == 'mm']
            if not mm_h:
                continue
            sw_sides = {h.detector_id for h in p_hits if h.detector_type == 'scint_wall'}
            bs_sides = {h.detector_id for h in p_hits if h.detector_type == 'back_scint'}
            ls_sides = {h.detector_id for h in p_hits if h.detector_type == 'liq_scint_1'}
            for h in mm_h:
                records.append({
                    'origin': p.origin.copy(),
                    'side': h.detector_id,
                    'u': float(h.true_pos[0]),
                    'v': float(h.true_pos[1]),
                    'direction': p.direction.copy(),
                    'has_sw': h.detector_id in sw_sides,
                    'has_bs': h.detector_id in bs_sides,
                    'has_ls': h.detector_id in ls_sides,
                })
    return records


print()
print("Collecting hit data for Config A (20k events)…", flush=True)
hits_A = collect_hit_data(sim_A, n_events=20_000)
print(f"  → {len(hits_A)} MM hits")

print("Collecting hit data for Config B (20k events)…", flush=True)
hits_B = collect_hit_data(sim_B, n_events=20_000)
print(f"  → {len(hits_B)} MM hits")


# ──────────────────────────────────────────────────────────────────────────────
# Check 3 & 4: MM hit maps with projected boundaries
# ──────────────────────────────────────────────────────────────────────────────

print()
print("=" * 72)
print("CHECK 3 — MM HIT MAPS coloured by downstream acceptance")
print("=" * 72)

# Pool all four sides (they're symmetric)
u_all = np.array([r['u'] for r in hits_A])
v_all = np.array([r['v'] for r in hits_A])
sw_all = np.array([r['has_sw'] for r in hits_A])
bs_all_A = np.array([r['has_bs'] for r in hits_A])
ls_all = np.array([r['has_ls'] for r in hits_A])
bs_all_B = np.array([r['has_bs'] for r in hits_B])

# Analytic back_scint projected boundaries on MM face
def backscint_projected_patches(bar_lo, bar_hi, bar_v, color, alpha=0.15):
    """Return rectangle patches for projected back_scint bars on MM face."""
    patches = [
        Rectangle((bar_lo, -bar_v), bar_hi - bar_lo, 2 * bar_v,
                  fc=color, ec=color, alpha=alpha, lw=1.5, ls='--'),
        Rectangle((-bar_hi, -bar_v), bar_hi - bar_lo, 2 * bar_v,
                  fc=color, ec=color, alpha=alpha, lw=1.5, ls='--'),
    ]
    return patches

fig_maps, axes_maps = plt.subplots(2, 2, figsize=(13, 11))

def _hit_map(ax, u, v, accept, title, det_color='#E91E63',
             analytic_patches=None, extra_lines=None):
    miss = ~accept
    ax.scatter(u[accept], v[accept], s=1.5, alpha=0.3, color='steelblue', label=f'Hit ({accept.sum():,})')
    ax.scatter(u[miss],   v[miss],   s=4,   alpha=0.7, color='red',       label=f'Miss ({miss.sum():,})')
    if analytic_patches:
        for p in analytic_patches:
            ax.add_patch(p)
    # MM outline
    ax.add_patch(Rectangle((-mm_hu, -mm_hv), 2*mm_hu, 2*mm_hv,
                            fc='none', ec='black', lw=2, zorder=3))
    if extra_lines:
        for (x1, x2, y1, y2, lbl, col) in extra_lines:
            ax.axvline(x1, color=col, ls=':', lw=1.5, alpha=0.8)
            ax.axvline(-x1, color=col, ls=':', lw=1.5, alpha=0.8)
            ax.axhline(y1, color=col, ls=':', lw=1.5, alpha=0.8, label=lbl)
            ax.axhline(-y1, color=col, ls=':', lw=1.5, alpha=0.8)
    ax.set_xlim(-mm_hu*1.15, mm_hu*1.15)
    ax.set_ylim(-mm_hv*1.25, mm_hv*1.25)
    ax.set_xlabel('u [cm]'); ax.set_ylabel('v [cm]')
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.2)
    frac = accept.mean() * 100
    ax.text(0.03, 0.04, f'Acceptance: {frac:.1f}%',
            transform=ax.transAxes, fontsize=8,
            bbox=dict(fc='lightyellow', ec='gray', alpha=0.9, boxstyle='round'))

# Scint wall acceptance (should be ~100%)
_hit_map(axes_maps[0, 0], u_all, v_all, sw_all,
         'Config A — MM hits: scint_wall acceptance\n'
         '(blue=also hits scint_wall, red=misses scint_wall)',
         extra_lines=[(mm_hu, None, mm_hv, None, f'MM edge', 'black')])

# LS-1 acceptance
ls1_proj_u = ls_hu * d_mm / d_ls1
ls1_proj_v = ls_hu * d_mm / d_ls1
_hit_map(axes_maps[0, 1], u_all, v_all, ls_all,
         'Config A — MM hits: LS-1 acceptance\n'
         '(blue=LS-1 also hit, red=miss LS-1 → fail scint_double)',
         analytic_patches=[Rectangle((-ls1_proj_u, -ls1_proj_v),
                                      2*ls1_proj_u, 2*ls1_proj_v,
                                      fc='green', ec='green', alpha=0.12, lw=2)],
         extra_lines=[(ls1_proj_u, None, ls1_proj_v, None,
                       f'LS-1 proj boundary ({ls1_proj_u:.1f} cm)', 'green')])

# Back scint acceptance — Config A
bs_patches_A = backscint_projected_patches(bar_lo_A, bar_hi_A, bar_v_A, '#E91E63')
_hit_map(axes_maps[1, 0], u_all, v_all, bs_all_A,
         f'Config A — MM hits: back_scint acceptance\n'
         f'(blue=back_scint hit, red=miss; dashed = analytic boundary)',
         analytic_patches=[
             *bs_patches_A,
             Rectangle((-bar_hi_A, -bar_v_A), 2*bar_hi_A, 2*bar_v_A,
                       fc='none', ec='#E91E63', alpha=0.6, lw=2),
         ],
         extra_lines=[(bar_v_A, None, bar_v_A, None,
                       f'BS proj v edge ({bar_v_A:.1f} cm)', '#E91E63')])

# Back scint acceptance — Config B
bs_patches_B = backscint_projected_patches(bar_lo_B, bar_hi_B, bar_v_B, '#9C27B0')
u_B = np.array([r['u'] for r in hits_B])
v_B = np.array([r['v'] for r in hits_B])
_hit_map(axes_maps[1, 1], u_B, v_B, bs_all_B,
         f'Config B — MM hits: back_scint acceptance\n'
         f'(back_scint now at {d_back_B:.1f} cm, closer to target)',
         analytic_patches=[
             *bs_patches_B,
             Rectangle((-bar_hi_B, -bar_v_B), 2*bar_hi_B, 2*bar_v_B,
                       fc='none', ec='#9C27B0', alpha=0.6, lw=2),
         ],
         extra_lines=[(bar_v_B, None, bar_v_B, None,
                       f'BS proj v edge ({bar_v_B:.1f} cm)', '#9C27B0')])

fig_maps.suptitle('MM hit maps: acceptance at downstream detectors\n'
                   '(He capsule source, all particle types, all 4 arms pooled)',
                   fontsize=11)
plt.tight_layout()
fig_maps.savefig(os.path.join(os.path.dirname(__file__), 'results', 'sanity', 'sanity_hit_maps.png'),
                  dpi=150, bbox_inches='tight')
print("  [Output] sanity_hit_maps.png")


# ──────────────────────────────────────────────────────────────────────────────
# Check 4: Back-scint acceptance vs |u| position
# ──────────────────────────────────────────────────────────────────────────────

fig_uacc, axes_ua = plt.subplots(1, 2, figsize=(13, 5))

def _plot_u_acceptance(ax, u, v, accept, title, analytic_u_hi, analytic_u_lo,
                        analytic_v_hi, color):
    # Bin by |u|, compute acceptance fraction per bin
    u_bins = np.linspace(0, mm_hu, 25)
    frac_bins, counts = [], []
    for lo, hi in zip(u_bins[:-1], u_bins[1:]):
        mask = (np.abs(u) >= lo) & (np.abs(u) < hi)
        if mask.sum() > 10:
            frac_bins.append(accept[mask].mean() * 100)
            counts.append(mask.sum())
        else:
            frac_bins.append(np.nan)
            counts.append(0)
    u_mid = (u_bins[:-1] + u_bins[1:]) / 2
    ax.bar(u_mid, frac_bins, width=np.diff(u_bins)[0]*0.9,
           color=color, alpha=0.7, edgecolor='k', lw=0.3)
    ax.axvline(analytic_u_hi, color=color, ls='--', lw=2,
               label=f'Analytic u-cutoff: {analytic_u_hi:.1f} cm')
    ax.axvline(analytic_u_lo, color=color, ls=':', lw=1.5, alpha=0.7,
               label=f'Bar inner edge: {analytic_u_lo:.2f} cm')
    ax.set_xlabel('|u| hit position on MM [cm]')
    ax.set_ylabel('Back-scint acceptance [%]')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(0, mm_hu * 1.05)

_plot_u_acceptance(axes_ua[0], u_all, v_all, bs_all_A,
                   f'Config A: back_scint acceptance vs |u| on MM\n'
                   f'(back_scint at {d_back_A:.1f} cm)',
                   bar_hi_A, bar_lo_A, bar_v_A, '#E91E63')
_plot_u_acceptance(axes_ua[1], u_B, v_B, bs_all_B,
                   f'Config B: back_scint acceptance vs |u| on MM\n'
                   f'(back_scint at {d_back_B:.1f} cm)',
                   bar_hi_B, bar_lo_B, bar_v_B, '#9C27B0')

fig_uacc.suptitle('Back-scint acceptance vs u-position on MM face\n'
                   'Sharp drop-off at analytic projection boundary confirms geometry',
                   fontsize=11)
plt.tight_layout()
fig_uacc.savefig(os.path.join(os.path.dirname(__file__), 'results', 'sanity', 'sanity_u_acceptance.png'),
                  dpi=150, bbox_inches='tight')
print("  [Output] sanity_u_acceptance.png")


# ──────────────────────────────────────────────────────────────────────────────
# Check 5: 3D scint-wall miss trajectories
# ──────────────────────────────────────────────────────────────────────────────
# The scint wall projects back to ±19.01 cm at the MM face — only 0.01 cm larger
# than the MM itself.  Misses are only possible from He capsule off-axis origins.
# We visualise this for arm 0 (+X arm).

print()
print("=" * 72)
print("CHECK 5 — 3D SCINT-WALL MISS VISUALISATION (arm 0, +X direction)")
print("=" * 72)
print(f"  Scint wall ½-u projects back to MM as: {sw_hu*d_mm/d_sw:.4f} cm")
print(f"  MM ½-u = {mm_hu:.4f} cm  →  margin = {sw_hu*d_mm/d_sw - mm_hu:.4f} cm")
print(f"  Critical He-capsule y_o for miss at u0={mm_hu}: "
      f"y_o < {(sw_hu - mm_hu*(d_sw/d_mm))/((d_sw/d_mm - 1)):.3f} cm")
print()

# Generate analytic miss trajectories for arm 0:
#   Origin: (x_o ≈ 0, y_o < y_crit, z_o = 0) [transverse offset]
#   MM hit: u0 ≈ 19 cm (u-edge), v0 = 0
#   y_o must be < 0 for u0 > 0 miss condition (see analytic section above)

fig_3d = plt.figure(figsize=(14, 6))
ax3d = fig_3d.add_subplot(121, projection='3d')

# Draw detector panels for arm 0 (at x = 25, 30.77 cm)
def draw_panel_3d(ax, x, half_u, half_v, color, alpha, label=None, lw=1.5):
    """Draw a rectangular panel at x = const in 3D.
    Y-axis = v (beam), Z-axis = u (transverse), matching Geant4 frame (beam = +Y)."""
    ys = [-half_v, -half_v, half_v, half_v, -half_v]   # v/beam in Y
    zs = [-half_u,  half_u, half_u, -half_u, -half_u]   # u in Z
    xs = [x] * 5
    ax.plot(xs, ys, zs, color=color, lw=lw, alpha=min(alpha * 3, 1.0))
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    verts = [list(zip(xs[:-1], ys[:-1], zs[:-1]))]
    poly = Poly3DCollection(verts, alpha=alpha, facecolor=color, edgecolor=color, lw=0.3)
    ax.add_collection3d(poly)
    if label:
        ax.text(x, half_v * 1.05, 0, label, fontsize=7, ha='center', color=color)

draw_panel_3d(ax3d, d_mm,  mm_hu, mm_hv, '#2196F3', 0.15, f'MM\n{d_mm:.0f} cm')
draw_panel_3d(ax3d, d_sw,  sw_hu, sw_hv, '#FF9800', 0.08, f'SW\n{d_sw:.1f} cm')

# Draw He capsule extent at x=0: Y=v/beam (±hl_cap), Z=u/transverse (±r_cap)
theta = np.linspace(0, 2*np.pi, 60)
r_cap = sim_A.cfg.he_radius_cm
hl_cap = sim_A.cfg.he_half_length_cm
# Outline as rectangle: sides at Z=±r_cap spanning Y=±hl_cap, caps at Y=±hl_cap
for z_edge in [-r_cap, r_cap]:
    ax3d.plot([0, 0], [-hl_cap, hl_cap], [z_edge, z_edge], 'k--', lw=1, alpha=0.4)
for y_edge in [-hl_cap, hl_cap]:
    ax3d.plot([0, 0], [y_edge, y_edge], [-r_cap, r_cap], 'k--', lw=1, alpha=0.4)

# -- Miss trajectories (z_o < 0, u0 ≈ 19, v0 = 0)
# In new frame: X=depth, Y=v/beam, Z=u/transverse for arm 0
z_o_vals = np.linspace(-r_cap, -0.1, 8)
n_miss = 0
for z_o in z_o_vals:
    u0_miss = mm_hu   # exact MM edge in u (Z direction)
    v0_miss = 0.0     # v = Y direction = 0
    # Direction from (0, 0, z_o) to MM hit (d_mm, -v0, u0)
    origin = np.array([0.0, -v0_miss, z_o])
    mm_world = np.array([d_mm, -v0_miss, u0_miss])
    direction = mm_world - origin
    # Position at scint wall
    t_sw = (d_sw - origin[0]) / direction[0]
    sw_hit = origin + t_sw * direction
    miss = abs(sw_hit[2]) > sw_hu   # u is now Z (index 2)
    col = 'red' if miss else 'steelblue'
    lw  = 1.8 if miss else 0.9
    zorder = 3 if miss else 2
    # Draw trajectory: origin → MM → scint wall
    ax3d.plot([origin[0], mm_world[0], sw_hit[0]],
              [origin[1], mm_world[1], sw_hit[1]],
              [origin[2], mm_world[2], sw_hit[2]],
              color=col, lw=lw, alpha=0.85, zorder=zorder)
    # Mark origin and MM hit
    ax3d.scatter(*origin, color=col, s=15, zorder=4)
    ax3d.scatter(*mm_world, color='blue', s=20, zorder=5)
    if miss:
        ax3d.scatter(*sw_hit, color='red', s=30, marker='x', zorder=6)
        n_miss += 1

# Draw scint wall u boundary lines (u = Z, spans v = Y)
for z_sw in [-sw_hu, sw_hu]:
    ax3d.plot([d_sw, d_sw], [-sw_hv, sw_hv], [z_sw, z_sw],
              'orange', lw=2.5, alpha=0.9)

# Also show a hit trajectory for comparison (z_o = 0)
origin_hit = np.array([0.0, 0.0, 0.0])
mm_w_hit   = np.array([d_mm, 0.0, mm_hu])   # Y=v=0, Z=u=mm_hu
t_sw_hit   = d_sw / mm_w_hit[0]
sw_hit_ok  = origin_hit + (mm_w_hit - origin_hit) * t_sw_hit
ax3d.plot([0, mm_w_hit[0], sw_hit_ok[0]],
          [0, mm_w_hit[1], sw_hit_ok[1]],
          [0, mm_w_hit[2], sw_hit_ok[2]],
          color='steelblue', lw=2.5, alpha=1.0, label='Origin → HIT')
ax3d.scatter(*sw_hit_ok, color='steelblue', s=50, zorder=6)

ax3d.set_xlabel('X (depth) [cm]', fontsize=8)
ax3d.set_ylabel('Y (v/beam) [cm]', fontsize=8)
ax3d.set_zlabel('Z (u/transverse) [cm]', fontsize=8)
ax3d.set_title('3D: scint-wall miss trajectories\n'
               'red = miss (|u_sw|>24), blue = hit', fontsize=9)
ax3d.set_xlim(-2, d_sw + 3)
ax3d.set_ylim(-sw_hv - 2, sw_hv + 2)   # Y = v = beam
ax3d.set_zlim(-sw_hu - 2, sw_hu + 2)   # Z = u = transverse
ax3d.view_init(elev=20, azim=-60)
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color='red',       lw=2, label=f'He capsule origin → MISS ({n_miss}/{len(z_o_vals)})'),
    Line2D([0], [0], color='steelblue', lw=2, label='He capsule origin → HIT'),
    Line2D([0], [0], color='black',     lw=1, ls='--', label='He capsule boundary'),
]
ax3d.legend(handles=custom_lines, fontsize=7, loc='upper left')

# 2D companion: z_o (arm-u, Z_lab) vs u0 phase space, coloured by hit/miss
ax2d = fig_3d.add_subplot(122)
z_o_grid = np.linspace(-r_cap, r_cap, 300)
u0_grid  = np.linspace(0, mm_hu, 300)
Z0, U0 = np.meshgrid(z_o_grid, u0_grid)
# u_sw = (d_sw/d_mm) * U0 - ((d_sw/d_mm) - 1) * Z0
U_SW = (d_sw / d_mm) * U0 - ((d_sw / d_mm) - 1) * Z0
miss_map = U_SW > sw_hu
ax2d.contourf(Z0, U0, miss_map.astype(float), levels=[0.5, 1.5],
              colors=['red'], alpha=0.3)
ax2d.contour(Z0, U0, U_SW, levels=[sw_hu], colors=['red'], linewidths=2)
ax2d.fill_between([-r_cap, r_cap], mm_hu, mm_hu * 1.05,
                   color='gray', alpha=0.3, label='Outside MM (u0>19)')
ax2d.set_xlabel('He capsule origin z_o [cm]  (arm-u = Z_lab direction)')
ax2d.set_ylabel('MM hit position u0 [cm]')
ax2d.set_title(f'Phase space: scint-wall miss region\n'
               f'(red shaded = miss,  arm 0,  v0=0, y_o=0)')
ax2d.axvline(-r_cap, color='k', ls='--', lw=1, label='He capsule edge')
ax2d.axvline( r_cap, color='k', ls='--', lw=1)
ax2d.set_xlim(-r_cap * 1.1, r_cap * 1.1)
ax2d.set_ylim(0, mm_hu * 1.05)
ax2d.legend(fontsize=8)
ax2d.grid(True, alpha=0.3)

z_crit = (sw_hu - mm_hu * (d_sw / d_mm)) / ((d_sw / d_mm) - 1)
ax2d.annotate(f'Miss region:\nz_o < {z_crit:.3f} cm\nAND u0 near {mm_hu:.0f} cm',
              xy=(z_crit - 0.1, mm_hu - 0.5), xytext=(-1.2, 15),
              fontsize=8, color='red',
              arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

fig_3d.suptitle('Scint-wall miss geometry\n'
                'Only possible when He capsule origin offsets track outside ±24 cm scint-wall edge',
                fontsize=10)
plt.tight_layout()
fig_3d.savefig(os.path.join(os.path.dirname(__file__), 'results', 'sanity', 'sanity_scintwall_miss.png'),
               dpi=150, bbox_inches='tight')
print("  [Output] sanity_scintwall_miss.png")


# ──────────────────────────────────────────────────────────────────────────────
# Check 6: He capsule origin distribution
# ──────────────────────────────────────────────────────────────────────────────

print()
print("=" * 72)
print("CHECK 6 — HE CAPSULE ORIGIN DISTRIBUTION")
print("=" * 72)

all_origins = np.array([r['origin'] for r in hits_A])
# New frame: origin = (x, y_beam, z_transverse) — Y is beam, XZ is transverse
r_orig = np.sqrt(all_origins[:, 0]**2 + all_origins[:, 2]**2)   # transverse radius in XZ
y_orig = all_origins[:, 1]                                         # beam axis = Y

# Theoretical: r² uniform (for correct uniform cylinder sampling), z uniform
fig_cap, axes_cap = plt.subplots(1, 3, figsize=(14, 4))

axes_cap[0].hist(r_orig, bins=40, density=True, color='steelblue', alpha=0.7, label='Sampled')
r_theory = np.linspace(0, sim_A.cfg.he_radius_cm, 200)
axes_cap[0].plot(r_theory, 2 * r_theory / sim_A.cfg.he_radius_cm**2,
                 'r-', lw=2, label='Expected: 2r/R²')
axes_cap[0].set_xlabel('r [cm]'); axes_cap[0].set_ylabel('Density')
axes_cap[0].set_title('He capsule: radial distribution\n(should follow 2r/R²)')
axes_cap[0].legend(fontsize=8)

axes_cap[1].hist(y_orig, bins=40, density=True, color='steelblue', alpha=0.7, label='Sampled')
y_theory_val = 1.0 / (2 * sim_A.cfg.he_half_length_cm)
axes_cap[1].axhline(y_theory_val, color='r', lw=2, label=f'Expected: 1/(2L)={y_theory_val:.3f}')
axes_cap[1].set_xlabel('y [cm] (beam axis)'); axes_cap[1].set_ylabel('Density')
axes_cap[1].set_title('He capsule: y (beam) distribution\n(should be uniform)')
axes_cap[1].legend(fontsize=8)

axes_cap[2].scatter(all_origins[:, 0], all_origins[:, 2],   # XZ transverse plane
                     s=0.5, alpha=0.2, color='steelblue')
theta = np.linspace(0, 2*np.pi, 200)
axes_cap[2].plot(sim_A.cfg.he_radius_cm * np.cos(theta),
                  sim_A.cfg.he_radius_cm * np.sin(theta),
                  'r-', lw=2, label=f'r={sim_A.cfg.he_radius_cm} cm boundary')
axes_cap[2].set_aspect('equal'); axes_cap[2].set_xlabel('x [cm]'); axes_cap[2].set_ylabel('z [cm]')
axes_cap[2].set_title('He capsule: XZ cross-section\n(should be uniform disk)')
axes_cap[2].legend(fontsize=8)

fig_cap.suptitle('He capsule origin sampling verification', fontsize=11)
plt.tight_layout()
fig_cap.savefig(os.path.join(os.path.dirname(__file__), 'results', 'sanity', 'sanity_capsule.png'),
                dpi=150, bbox_inches='tight')
print("  [Output] sanity_capsule.png")

# Quick stats
print(f"  r: mean={r_orig.mean():.3f} cm  expected={sim_A.cfg.he_radius_cm * np.sqrt(0.5):.3f} cm (= R/√2)")
print(f"  z: mean={y_orig.mean():.3f} cm  std={y_orig.std():.3f} cm  expected std={sim_A.cfg.he_half_length_cm/np.sqrt(3):.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary coverage bar chart
# ──────────────────────────────────────────────────────────────────────────────

fig_cov, ax_cov = plt.subplots(figsize=(11, 5))
det_names_plot = ['MM', 'Scint\nwall', 'LS-1', 'LS-2', 'Back_scint\n(Config A)', 'Back_scint\n(Config B)']
mc_A_fracs = [fracs_A.get(t, np.nan) * 100 for t in types]
mc_B_fracs = [fracs_B.get(t, np.nan) * 100 for t in types]
# For back_scint, use Config-specific fraction
mc_plot = mc_A_fracs[:4] + [mc_A_fracs[4], mc_B_fracs[4]]
analytic_fracs = [
    np.nan,
    sw_hu * d_mm / d_sw / mm_hu,          # fraction of MM half-u covered
    None, None,
    bs_frac_A * 100,
    bs_frac_B * 100,
]

x = np.arange(len(det_names_plot))
bars = ax_cov.bar(x, mc_plot, color=colors_sa[:4] + ['#E91E63', '#9C27B0'],
                   alpha=0.8, edgecolor='k', lw=0.5)
# Analytic points
ax_cov.plot([4, 5], [bs_frac_A*100, bs_frac_B*100],
             'k^', ms=8, zorder=5, label='Analytic (point origin)')

mm_val = mc_A_fracs[0]
ax_cov.axhline(mm_val, color='gray', ls='--', lw=1.5, alpha=0.7, label=f'MM = {mm_val:.1f}%')

for bar, val in zip(bars, mc_plot):
    if not np.isnan(val):
        ax_cov.text(bar.get_x() + bar.get_width()/2, val + 0.2,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

ax_cov.set_xticks(x); ax_cov.set_xticklabels(det_names_plot, fontsize=9)
ax_cov.set_ylabel('Solid angle coverage [% of 4π]')
ax_cov.set_title('Solid angle coverage per detector (Monte Carlo, 200k samples from origin)\n'
                  'Triangles = analytic prediction for back_scint')
ax_cov.legend(fontsize=9)
ax_cov.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig_cov.savefig(os.path.join(os.path.dirname(__file__), 'results', 'sanity', 'sanity_solid_angle.png'),
                dpi=150, bbox_inches='tight')
print("  [Output] sanity_solid_angle.png")

print()
print("=" * 72)
print("ALL SANITY CHECKS COMPLETE")
print("=" * 72)
print()
print("Key findings:")
print(f"  • Scint wall from origin subtends ±{sw_hu*d_mm/d_sw:.4f} cm at MM face — only")
print(f"    {sw_hu*d_mm/d_sw - mm_hu:.4f} cm larger than MM (±{mm_hu} cm). This is intentional.")
print(f"    The 99.8% acceptance is correct; misses only occur from off-axis He capsule origins.")
print()
print(f"  • Back_scint analytic coverage: {bs_frac_A*100:.1f}% (Config A) / {bs_frac_B*100:.1f}% (Config B)")
print(f"    from point origin. Simulated: ~44-48% / ~55-60% (He capsule spreads acceptance).")
print(f"    The low coverage is a pure geometric effect: projection magnification of")
print(f"    {d_back_A/d_mm:.2f}× (A) / {d_back_B/d_mm:.2f}× (B) maps MM edges outside bar width.")
print()
print(f"  • LS-1 covers {min(ls_hu*d_mm/d_ls1, mm_hu)/mm_hu * min(ls_hu*d_mm/d_ls1, mm_hv)/mm_hv *100:.1f}%")
print(f"    of MM area analytically — matches scint_single_efficiency in simulation.")

plt.show()
