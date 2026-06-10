"""
Detailed detector geometry figures for mechanical review.

Active volumes only — no PCBs, mechanical frames, shielding, or support
structures are shown. All dimensions are of the active (sensitive) volume.

Outputs:
  geometry_detail_A.png  — Config A: standard stack order
  geometry_detail_B.png  — Config B: back_scint first
  geometry_detail_face.png — Face-on view (same active dimensions both configs)
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))
from MX17_Simulator import MicromegasSimulation, SimConfig
from detector_config import GEO

# ──────────────────────────────────────────────────────────────────────────────
# Configs  (geometry only — no events needed)
# ──────────────────────────────────────────────────────────────────────────────

_GEO = dict(
    **GEO,
    # Dummy event settings — no simulation is run
    n_signal=0.0, n_random=0.0, n_background_pairs=0.0, n_events=1, seed=0,
)

sim_A = MicromegasSimulation(SimConfig(**_GEO, stack_order='standard',
                                        trigger_second_layer='liq_scint_1'))
sim_B = MicromegasSimulation(SimConfig(**_GEO, stack_order='back_scint_first',
                                        trigger_second_layer='back_scint'))

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette  (shared)
# ──────────────────────────────────────────────────────────────────────────────
COL = {
    'mm':         '#2196F3',
    'scint_wall': '#FF9800',
    'liq_scint':  '#4CAF50',
    'liq_scint2': '#009688',
    'back_scint': '#E91E63',
    'he':         '#9E9E9E',
    'dim':        '#333333',
    'air':        '#BBBBBB',
}


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _layer_positions(sim):
    """
    Return a list of dicts describing each active volume layer, in stack order.
    Keys: name, color, front, back, hu (half-u), hv (half-v), thick, center
    """
    cfg = sim.cfg
    d_mm, d_sc, d_ls1, d_ls2, d_bk = sim._get_stack_distances()

    # d_mm is the inner (target-side) face of the MM drift gap
    mm_front = d_mm
    mm_back  = d_mm + cfg.mm_drift_gap

    def _layer(name, color, d_ctr, thick, hu, hv):
        return dict(name=name, color=color,
                    front=d_ctr - thick / 2, back=d_ctr + thick / 2,
                    center=d_ctr, thick=thick, hu=hu, hv=hv)

    # MM is special: front face IS d_mm (not center-based)
    mm = dict(name='MM (drift)', color=COL['mm'],
              front=mm_front, back=mm_back, center=(mm_front + mm_back) / 2,
              thick=cfg.mm_drift_gap,
              hu=cfg.detector_size[0] / 2, hv=cfg.detector_size[1] / 2)

    sw = _layer('Scint wall', COL['scint_wall'], d_sc, cfg.scint_wall_thickness,
                cfg.scint_wall_size[0] / 2, cfg.scint_wall_size[1] / 2)
    ls1 = _layer('LS-1', COL['liq_scint'], d_ls1, cfg.liq_scint_thickness,
                 cfg.liq_scint_size[0] / 2, cfg.liq_scint_size[1] / 2)
    ls2 = _layer('LS-2', COL['liq_scint2'], d_ls2, cfg.liq_scint_2_thickness,
                 cfg.liq_scint_2_size[0] / 2, cfg.liq_scint_2_size[1] / 2)

    u_off   = (cfg.back_scint_size_u + cfg.back_scint_gap) / 2.0
    bar_hu  = cfg.back_scint_size_u / 2.0
    bs = _layer('Back scint', COL['back_scint'], d_bk, cfg.back_scint_thickness,
                u_off + bar_hu, cfg.back_scint_size_v / 2)
    bs['u_off'] = u_off
    bs['bar_hu'] = bar_hu
    bs['gap'] = cfg.back_scint_gap

    if cfg.stack_order == 'standard':
        return [mm, sw, ls1, ls2, bs]
    else:
        return [mm, sw, bs, ls1, ls2]


def _air_gaps(layers):
    """Return list of (front_det, back_det, gap_cm) between consecutive layers."""
    gaps = []
    for i in range(len(layers) - 1):
        gap = layers[i + 1]['front'] - layers[i]['back']
        gaps.append((layers[i], layers[i + 1], gap))
    return gaps


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _dim_arrow_h(ax, x1, x2, y, label, color=COL['dim'],
                 fs=7.5, lw=1.1, tick_h=0.5):
    """Horizontal ↔ dimension arrow at height y, with end ticks and label."""
    if abs(x2 - x1) < 1e-6:
        return
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='<->', color=color, lw=lw,
                                mutation_scale=8, shrinkA=0, shrinkB=0))
    for xp in [x1, x2]:
        ax.plot([xp, xp], [y - tick_h, y + tick_h], color=color, lw=lw * 0.7)
    ax.text((x1 + x2) / 2, y + tick_h * 0.55, label,
            ha='center', va='bottom', fontsize=fs, color=color,
            bbox=dict(fc='white', ec='none', pad=0.5))


def _dim_arrow_v(ax, y1, y2, x, label, color=COL['dim'],
                 fs=7.5, lw=1.1, tick_w=0.4, y_frac=0.5):
    """Vertical ↔ dimension arrow at x-position x.
    y_frac (0–1) sets where along the arrow the label is placed, allowing
    vertical staggering so labels at neighbouring arrows don't overlap."""
    if abs(y2 - y1) < 1e-6:
        return
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='<->', color=color, lw=lw,
                                mutation_scale=8, shrinkA=0, shrinkB=0))
    for yp in [y1, y2]:
        ax.plot([x - tick_w, x + tick_w], [yp, yp], color=color, lw=lw * 0.7)
    y_label = y1 + (y2 - y1) * y_frac
    ax.text(x + tick_w * 1.8, y_label, label,
            ha='left', va='center', fontsize=fs, color=color, zorder=10,
            bbox=dict(fc='white', ec='none', pad=0.8, alpha=0.9))


def _leader(ax, x, y, x_txt, y_txt, label, color, fs=7.5):
    """Annotation line from (x,y) to (x_txt, y_txt) with label."""
    ax.annotate(label, xy=(x, y), xytext=(x_txt, y_txt),
                ha='left', va='center', fontsize=fs, color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=0.8,
                                connectionstyle='arc3,rad=0.0'))


# ──────────────────────────────────────────────────────────────────────────────
# Side-profile panel
# ──────────────────────────────────────────────────────────────────────────────

def draw_side_profile(ax, sim, config_label):
    """
    Top-down cross-section of one arm (+X direction).
    Horizontal axis = depth from target [cm].
    Vertical axis   = u (transverse, = Z_lab for arm 0) [cm].
    The beam axis Y (= v direction) is perpendicular to this plane (into page).
    """
    cfg    = sim.cfg
    layers = _layer_positions(sim)
    u_off  = (cfg.back_scint_size_u + cfg.back_scint_gap) / 2.0
    bar_hu = cfg.back_scint_size_u / 2.0

    # ── active volumes ────────────────────────────────────────────────────────
    for lay in layers:
        if lay['name'] == 'Back scint':
            # Upper bar
            ax.add_patch(Rectangle((lay['front'], u_off - bar_hu),
                                    lay['thick'], 2 * bar_hu,
                                    fc=lay['color'], ec='k', alpha=0.65, lw=0.8, zorder=3))
            # Lower bar
            ax.add_patch(Rectangle((lay['front'], -(u_off + bar_hu)),
                                    lay['thick'], 2 * bar_hu,
                                    fc=lay['color'], ec='k', alpha=0.65, lw=0.8, zorder=3))
            # Gap label
            ax.text(lay['center'], cfg.back_scint_gap / 2,
                    f"gap\n{cfg.back_scint_gap * 10:.0f} mm",
                    ha='center', va='center', fontsize=6.5, color='k', zorder=5,
                    bbox=dict(fc='white', ec='none', pad=0.2, alpha=0.9, boxstyle='round'))
        else:
            # Draw at true scale; ensure a minimum visible width via bold edge
            draw_thick = max(lay['thick'], 0.0)
            ax.add_patch(Rectangle((lay['front'], -lay['hu']),
                                    draw_thick, 2 * lay['hu'],
                                    fc=lay['color'], ec=lay['color'], alpha=0.65, lw=2.5, zorder=3))

    # ── position ruler at top ─────────────────────────────────────────────────
    # Absolute distances from target to each layer face, shown as tick+label above plot
    y_ruler = layers[-1]['hu'] + 3.5
    ax.axhline(y_ruler, color='#888', lw=0.5, ls=':', zorder=0)
    for lay in layers:
        for x_face, face_label in [(lay['front'], 'F'), (lay['back'], 'B')]:
            ax.plot(x_face, y_ruler, 'v', ms=4, color=lay['color'], zorder=5)
        # Center label
        ax.text(lay['center'], y_ruler + 0.5,
                f"{lay['center']:.2f}", ha='center', va='bottom',
                fontsize=6.5, color=lay['color'], fontweight='bold')

    # ── inter-layer air gap dimension arrows (below detectors) ────────────────
    y_dim_base = -(max(u_off + bar_hu, layers[0]['hu'])) - 2.5
    gap_row_step = 2.0
    for row_i, (prev, nxt, gap) in enumerate(_air_gaps(layers)):
        y_dim = y_dim_base - row_i * gap_row_step
        # Leader lines from detector faces down to dimension row
        ax.plot([prev['back'],  prev['back']],  [y_dim_base - 0.2, y_dim + 0.2],
                color=COL['air'], lw=0.6, ls='--', alpha=0.6)
        ax.plot([nxt['front'], nxt['front']], [y_dim_base - 0.2, y_dim + 0.2],
                color=COL['air'], lw=0.6, ls='--', alpha=0.6)
        _dim_arrow_h(ax, prev['back'], nxt['front'], y_dim,
                     f"air gap\n{gap:.2f} cm",
                     color=COL['air'], fs=6.5, tick_h=0.35)

    # ── per-detector active dimension annotation (thickness arrow above) ──────
    y_thick_base = max(u_off + bar_hu, layers[0]['hu']) + 1.5
    thick_row_step = 2.2
    for row_i, lay in enumerate(layers):
        # y_t = y_thick_base + row_i * thick_row_step
        y_t = y_thick_base + 4 * thick_row_step if row_i != 0 else y_thick_base
        ax.plot([lay['front'], lay['front']], [lay['hu'], y_t - 0.2],
                color=lay['color'], lw=0.5, ls=':', alpha=0.7)
        ax.plot([lay['back'],  lay['back']],  [lay['hu'], y_t - 0.2],
                color=lay['color'], lw=0.5, ls=':', alpha=0.7)
        _dim_arrow_h(ax, lay['front'], lay['back'], y_t,
                     f"{lay['name']}\nt = {lay['thick']:.2f} cm",
                     color=lay['color'], fs=6.5, tick_h=0.35)

    # ── arm axis ──────────────────────────────────────────────────────────────
    ax.axhline(0, color='k', lw=0.6, ls=':', zorder=0, alpha=0.4)

    # ── axes / labels ─────────────────────────────────────────────────────────
    x_min = layers[0]['front'] - 2.0
    x_max = layers[-1]['back'] + 1.5
    all_hu = [l['hu'] if l['name'] != 'Back scint' else (u_off + bar_hu) for l in layers]
    y_max = max(all_hu) + thick_row_step * len(layers) + 3
    y_min = y_dim_base - gap_row_step * len(layers) - 1.5

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Distance from target [cm]', fontsize=9)
    ax.set_ylabel('u  (transverse, = Z_lab for arm 0)  [cm]', fontsize=9)
    ax.set_title(
        f'{config_label}\n'
        f'Top-down cross-section of single arm (+X),  beam axis Y ⊥ page',
        fontsize=9, pad=6,
    )
    ax.grid(True, alpha=0.15, zorder=0)
    handles = [mpatches.Patch(fc=l['color'], ec=l['color'], alpha=0.7, label=l['name'])
               for l in layers]
    ax.legend(handles=handles, loc='upper left', fontsize=7, framealpha=0.9)

    # Watermark note
    ax.text(0.99, 0.99,
            'ACTIVE VOLUMES ONLY\nNo mechanics / shielding / PCBs shown',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=7, color='red', alpha=0.7,
            bbox=dict(fc='lightyellow', ec='orange', alpha=0.8, boxstyle='round'))

    # ── spec table (right side) ───────────────────────────────────────────────
    spec_lines = []
    for lay in layers:
        if lay['name'] == 'Back scint':
            spec_lines.append(
                f"Back scint:  2 bars × {cfg.back_scint_size_u:.0f}×{cfg.back_scint_size_v:.0f}×{cfg.back_scint_thickness:.0f} cm\n"
                f"             center @ {lay['center']:.2f} cm,  bar gap={cfg.back_scint_gap:.1f} cm"
            )
        else:
            spec_lines.append(
                f"{lay['name']:12s}: {lay['hu']*2:.0f}×{lay['hv']*2:.0f}×{lay['thick']:.2f} cm"
                f"  (u×v×t),  center @ {lay['center']:.2f} cm"
            )
    spec_text = '\n'.join(spec_lines)
    ax.text(0.01, 0.01, spec_text,
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=6.8, family='monospace',
            bbox=dict(fc='white', ec='gray', alpha=0.9, boxstyle='round'))


# ──────────────────────────────────────────────────────────────────────────────
# Face-on view panel
# ──────────────────────────────────────────────────────────────────────────────

def draw_face_on(ax, sim, title):
    """
    Face-on view looking from detector toward target.
    u-axis (= Z_lab for +X arm) horizontal.
    v-axis (= beam / Y_lab) vertical.
    Draws all detector faces at their active u×v dimensions.
    Back-scint bars shown at their true u-offset.
    """
    cfg    = sim.cfg
    layers = _layer_positions(sim)
    u_off  = (cfg.back_scint_size_u + cfg.back_scint_gap) / 2.0
    bar_hu = cfg.back_scint_size_u / 2.0

    # Draw from outermost (largest) to innermost so smaller are visible
    for lay in reversed(layers):
        if lay['name'] == 'Back scint':
            for sign in [+1, -1]:
                u_bar_ctr = sign * u_off
                ax.add_patch(Rectangle(
                    (u_bar_ctr - bar_hu, -lay['hv']), 2 * bar_hu, 2 * lay['hv'],
                    fc=lay['color'], ec=lay['color'], alpha=0.22,
                    lw=2.0, zorder=3, ls='--'))
            # Label the gap
            ax.annotate(
                f"gap\n{cfg.back_scint_gap * 10:.0f} mm",
                xy=(0, 0), xytext=(0, lay['hv'] * 0.5),
                ha='center', va='center', fontsize=6.5, color=lay['color'],
                bbox=dict(fc='white', ec=lay['color'], alpha=0.8,
                          boxstyle='round', pad=0.3))
        else:
            ax.add_patch(Rectangle(
                (-lay['hu'], -lay['hv']), 2 * lay['hu'], 2 * lay['hv'],
                fc=lay['color'], ec=lay['color'], alpha=0.15,
                lw=2.0, zorder=3,
                ls='-'))
            # Outline only — no fill for non-MM layers to avoid clutter
            ax.add_patch(Rectangle(
                (-lay['hu'], -lay['hv']), 2 * lay['hu'], 2 * lay['hv'],
                fc='none', ec=lay['color'], alpha=0.9,
                lw=2.2, zorder=4))

    # Highlight MM outline boldly (reference detector)
    mm = next(l for l in layers if l['name'] == 'MM (drift)')
    ax.add_patch(Rectangle((-mm['hu'], -mm['hv']), 2 * mm['hu'], 2 * mm['hv'],
                            fc='none', ec=COL['mm'], lw=3.0, zorder=6))

    # He capsule cross-section: rectangle r×2hl (u=transverse, v=beam)
    r_cap = sim.cfg.he_radius_cm
    hl    = sim.cfg.he_half_length_cm
    ax.add_patch(Rectangle((-r_cap, -hl), 2 * r_cap, 2 * hl,
                            fc=COL['he'], ec='k', alpha=0.5, lw=1.0, zorder=5))
    ax.plot(0, 0, 'k+', ms=8, mew=1.5, zorder=7)
    ax.text(r_cap + 0.3, 0, f'He capsule\nr = {r_cap:.1f} cm\nL = ±{hl:.1f} cm',
            ha='left', va='center', fontsize=6.5, color='#555')

    # ── dimension arrows ──────────────────────────────────────────────────────
    # v-heights (vertical axis = beam = Y_lab): vertical arrows on the right,
    # labels staggered so they don't overlap.
    x_dim_base = max(l['hu'] for l in layers if l['name'] != 'Back scint') + 1.5
    step_u = 3.5
    y_fracs = [0.82, 0.64, 0.50, 0.36, 0.18]
    for i, lay in enumerate(layers):
        x_d = x_dim_base + i * step_u
        yf  = y_fracs[i % len(y_fracs)]
        _dim_arrow_v(ax, -lay['hv'], lay['hv'], x_d,
                     f"{lay['name']}\nv = {lay['hv']*2:.0f} cm",
                     color=lay['color'], fs=6.5, tick_w=0.4, y_frac=yf)

    # u-widths (horizontal axis = Z_lab): horizontal arrows below the detector
    y_dim_base = -(max(l['hv'] for l in layers)) - 2.5
    step_v = 2.8
    for i, lay in enumerate(layers):
        if lay['name'] == 'Back scint':
            lbl = f"Back scint  u span = {(u_off + bar_hu)*2:.1f} cm"
            _dim_arrow_h(ax, -(u_off + bar_hu), (u_off + bar_hu),
                         y_dim_base - i * step_v,
                         lbl, color=lay['color'], fs=6.5, tick_h=0.4)
        else:
            lbl = f"{lay['name']}  u = {lay['hu']*2:.0f} cm"
            _dim_arrow_h(ax, -lay['hu'], lay['hu'], y_dim_base - i * step_v,
                         lbl, color=lay['color'], fs=6.5, tick_h=0.4)

    # Back-scint bar centre-offset annotation (above bars)
    # bs = next(l for l in layers if l['name'] == 'Back scint')
    # _dim_arrow_h(ax, 0, u_off, bs['hv'] + 2.5,
    #              f"bar centre offset\n{u_off:.2f} cm",
    #              color=COL['back_scint'], fs=6.5, tick_h=0.3)

    # ── axes ──────────────────────────────────────────────────────────────────
    x_max = x_dim_base + step_u * len(layers) + 3.0
    y_min = y_dim_base - step_v * len(layers) - 1.5
    y_max = max(l['hv'] for l in layers) + 5.5

    ax.set_xlim(-(max(l['hu'] for l in layers) + 1.5), x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xlabel('u  (= Z_lab for +X arm)  [cm]', fontsize=9)
    ax.set_ylabel('v  (= beam axis Y_lab)  [cm]', fontsize=9)
    ax.set_title(title, fontsize=9, pad=6)
    ax.axhline(0, color='gray', lw=0.5, ls=':', alpha=0.5)
    ax.axvline(0, color='gray', lw=0.5, ls=':', alpha=0.5)
    ax.grid(True, alpha=0.15)

    ax.text(0.99, 0.99,
            'ACTIVE VOLUMES ONLY\nNo mechanics / shielding / PCBs shown',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=7, color='red', alpha=0.7,
            bbox=dict(fc='lightyellow', ec='orange', alpha=0.8, boxstyle='round'))

    # Legend patches
    handles = []
    for lay in layers:
        handles.append(mpatches.Patch(fc=lay['color'], ec=lay['color'],
                                       alpha=0.6, label=lay['name']))
    handles.append(mpatches.Patch(fc=COL['he'], ec='k', alpha=0.5,
                                   label=f'He-3 capsule  {2*r_cap:.0f}×{2*hl:.0f} cm (u×v)'))
    ax.legend(handles=handles, loc='lower right', fontsize=7, framealpha=0.9)


# ──────────────────────────────────────────────────────────────────────────────
# Combined per-config figure
# ──────────────────────────────────────────────────────────────────────────────

def make_config_figure(sim, config_label, filename):
    # Side profile (left, ~55%) + face-on view (right, ~45%)
    fig = plt.figure(figsize=(14, 7.5))
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.25, 1.1],
                   left=0.05, right=0.99, top=0.87, bottom=0.06, wspace=0.12)

    ax_side = fig.add_subplot(gs[0, 0])
    ax_face = fig.add_subplot(gs[0, 1])

    draw_side_profile(ax_side, sim, config_label)
    draw_face_on(ax_face, sim,
                 'Face-on view\n(looking from detector toward target)')

    fig.suptitle(
        f'MX17 Detector Stack — {config_label}\n'
        f'Active volumes only.  All dimensions in cm.  Beam axis: +Y_lab (= v, vertical in face-on view).',
        fontsize=12, fontweight='bold', y=0.97,
    )
    fig.savefig(os.path.join(os.path.dirname(__file__), 'results', 'geometry', filename),
                dpi=150, bbox_inches='tight')
    print(f'[Output] {filename}')
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

print('Generating detailed geometry figures…')
fig_a = make_config_figure(sim_A,
    'Config A — standard  (scint_wall → LS-1 → LS-2 → back_scint)',
    'geometry_detail_A.png')
fig_b = make_config_figure(sim_B,
    'Config B — back_scint first  (scint_wall → back_scint → LS-1 → LS-2)',
    'geometry_detail_B.png')

# Print numeric summary for quick reference
print()
print('=' * 70)
print('ACTIVE VOLUME POSITIONS (distance = centre of active volume from target)')
print('=' * 70)
for label, sim in [('Config A', sim_A), ('Config B', sim_B)]:
    print(f'\n  {label}')
    for lay in _layer_positions(sim):
        print(f"    {lay['name']:16s}: front={lay['front']:6.2f}  "
              f"back={lay['back']:6.2f}  centre={lay['center']:6.2f}  "
              f"u={lay['hu']*2:.0f}×v={lay['hv']*2:.0f} cm  t={lay['thick']:.2f} cm")
    print()
    gaps = _air_gaps(_layer_positions(sim))
    print(f'  Air gaps (back face → next front face):')
    for prev, nxt, gap in gaps:
        print(f"    {prev['name']:16s} → {nxt['name']:16s}: {gap:.2f} cm")

plt.show()
