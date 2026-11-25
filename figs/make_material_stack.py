#!/usr/bin/env python3
"""make_material_stack.py -- Paper I v2, Section 3: the chamber as a stack of layers, drawn three ways.

    .venv/bin/python papers/paper1_v2/figs/make_material_stack.py

Replaces Tables 1 and 2 of the v2 draft (Stephan's suggestion, 2026-11-25). One row per layer at the center of the
active area, in the order a lepton from the target meets them. Three aligned panels:
  (a) the physical thickness of the layer (log scale);
  (b) its radiation length fraction x/X0, averaged over the pattern for patterned layers (pads, strips, mesh);
  (c) the rms multiple-scattering angle it causes alone (Highland, per projection), for electrons of 4, 8, 16 MeV.
Summary rows give the totals in front of the drift volume and including the drift gas (Highland on the summed x/X0).

Writes out/material_stack.{pdf,png,json} and out/material_stack_table.tex (the full numbers for the appendix).

Sources (docs/detector_spec/CONSTRUCTION_DOSSIER.md has each one):
  window, gas gap, cathode, drift gap: docs/detector_spec/spec.py and mcs_common (drift gap 28.2 mm measured on A)
  mesh: ASADA 45/19 (45 um opening, 19 um wire), AISI 316, calendered to 20 um (Stephan, 2026-11-24). Its areal mass
        is that of two 19 um wires per 64 um pitch: 2 * pi/4 * 19^2 / 64 = 8.86 um of solid steel.
  resistive strips: 100 um wide on 488 um (Stephan, "tbc"); thickness 10 um assumed; paste as C/H/O 1.4 g/cm3.
  readout board: stack-up 3498-A (CERN MPT; docs/detector_spec/stackup_3498A), copper coverages from the Gerbers
        (spec.py): pads 0.760, Y 0.458, X 0.457, L7 lands 0.53; the L3 top copper has no copper in the active area.
  Rohacell 5 mm (grade assumed 51), back Mylar 25 um assumed. The Al plate has an aperture: nothing behind the
        active area.
Assumed values are flagged (dagger) in the figure and the table.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import paper_style as ps        # noqa: E402
import mcs_common as mc         # noqa: E402

# extra materials (PDG): X0 [g/cm2], density [g/cm3]
mc.MAT.setdefault('steel', dict(X0=13.84, rho=8.00, label='stainless steel'))
mc.MAT.setdefault('fr4', dict(X0=33.0, rho=1.70, label='glass-epoxy (G10)'))
mc.MAT.setdefault('rohacell', dict(X0=40.0, rho=0.052, label='Rohacell'))
mc.MAT.setdefault('paste', dict(X0=40.0, rho=1.40, label='resistive paste (C/H/O)'))

# colors as in the chamber figure (make_chamber_stack.py)
CU, KAP, FR4, GAS, ROH, MYL, ESL, AIR, STEEL = ('#b87333', '#f0d28a', '#cfdccd', '#d7e8f3', '#efe4c6', '#8fc8e6',
                                               '#444444', '#eeeeee', '#7d8590')
COL = dict(air=AIR, mylar=MYL, gas=GAS, kapton=KAP, Cu=CU, steel=STEEL, paste=ESL, fr4=FR4, rohacell=ROH)

# (key, label, material, thickness mm, areal fill, assumed?, group)
#   group: 'front' = in front of the drift volume, 'drift' = the drift gas, 'back' = behind the measured segment
G = mc.G_DRIFT
LAYERS = [
    ('air', 'air, target to window', 'air', mc.D_WIN, 1, False, 'front'),
    ('window', 'entrance window, Mylar', 'mylar', mc.T_WIN, 1, False, 'front'),
    ('gap5', 'gas gap', 'gas', mc.T_GAP5, 1, False, 'front'),
    ('kapton', 'drift cathode, Kapton', 'kapton', mc.T_KAP, 1, False, 'front'),
    ('cu', 'drift cathode, copper', 'Cu', mc.T_CU, 1, True, 'front'),
    ('drift', f'drift gas ({G:.1f} mm measured)', 'gas', G, 1, False, 'drift'),
    ('mesh', 'mesh, steel (calendered)', 'steel', 0.020, 2 * np.pi / 4 * 19.0 ** 2 / 64.0 / 20.0, False, 'back'),
    ('ampgap', 'amplification gap', 'gas', 0.150, 1, False, 'back'),
    ('esl', 'resistive strips', 'paste', 0.010, 100 / 488, True, 'back'),
    ('ins', 'insulator, polyimide + prepreg', 'fr4', 0.050, 1, False, 'back'),
    ('pads', 'pads, copper', 'Cu', 0.034, 0.760, False, 'back'),
    ('d2', 'prepreg', 'fr4', 0.100, 1, False, 'back'),
    ('ystr', 'Y strips, copper', 'Cu', 0.017, 0.458, False, 'back'),
    ('d3', 'prepreg', 'fr4', 0.100, 1, False, 'back'),
    ('xstr', 'X strips, copper', 'Cu', 0.017, 0.457, False, 'back'),
    ('d4', 'prepreg', 'fr4', 0.100, 1, False, 'back'),
    ('l7', 'via lands, copper', 'Cu', 0.034, 0.53, False, 'back'),
    ('d5', 'prepreg + polyimide', 'fr4', 0.050, 1, False, 'back'),
    ('roh', 'Rohacell', 'rohacell', 5.0, 1, True, 'back'),
    ('backfoil', 'back foil, Mylar', 'mylar', 0.025, 1, True, 'back'),
]
TS = (4, 8, 16)
TMARK = {4: ('o', '#9ecae1'), 8: ('o', '#2171b5'), 16: ('o', '#08306b')}


def th(x, T):
    return float(np.degrees(mc.highland(x, T)))


def fmt_t(mm):
    if mm >= 1:
        return f'{mm:g} mm'
    return f'{mm * 1000:.0f} µm'


def main():
    ps.use()
    rows = []
    for key, lab, m, t, fill, asm, grp in LAYERS:
        x = mc.x_over_X0(m, t) * fill
        rows.append(dict(key=key, label=lab, material=m, thickness_mm=t, fill=round(fill, 4), assumed=asm, group=grp,
                         X0_cm=round(mc.X0_cm(m), 3), x_over_X0_pct=100 * x,
                         theta0_deg={T: th(x, T) for T in TS}))
    xs_front = sum(r['x_over_X0_pct'] for r in rows if r['group'] == 'front') / 100
    x_drift = sum(r['x_over_X0_pct'] for r in rows if r['group'] == 'drift') / 100
    xs_back = sum(r['x_over_X0_pct'] for r in rows if r['group'] == 'back') / 100
    xs_front_nocu = xs_front - [r for r in rows if r['key'] == 'cu'][0]['x_over_X0_pct'] / 100
    board = sum(r['thickness_mm'] for r in rows if r['key'] in ('ins', 'pads', 'd2', 'ystr', 'd3', 'xstr', 'd4',
                                                                 'l7', 'd5'))
    tot = {'front': xs_front, 'front_noCu': xs_front_nocu, 'with_drift': xs_front + x_drift, 'behind': xs_back}

    # ------------------------------------------------------------------ figure
    n = len(rows)
    ysum = [n + 0.9, n + 1.9]                     # two summary rows under the layers
    fig = plt.figure(figsize=(6.1, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], left=0.335, right=0.985, top=0.915, bottom=0.075, wspace=0.12)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ylim = (ysum[-1] + 0.7, -0.7)

    def ybar(ax, i, v, col, x0, hatch=None, alpha=1.0):
        ax.add_patch(Rectangle((x0, i - 0.36), v - x0, 0.72, fc=col, ec='#555555', lw=0.4, alpha=alpha, hatch=hatch,
                               zorder=3))

    # (a) thickness
    ax = axs[0]
    for i, r in enumerate(rows):
        ybar(ax, i, r['thickness_mm'], COL[r['material']], 1e-3)
        ax.text(r['thickness_mm'] * 1.25, i, fmt_t(r['thickness_mm']), va='center', ha='left', fontsize=7,
                color=ps.INK, zorder=4)
    ax.set_xscale('log')
    ax.set_xlim(1e-3, 2e4)
    ax.set_xticks([1e-3, 1, 1e3])
    ax.set_xticklabels(['1 µm', '1 mm', '1 m'])
    ax.set_xlabel('thickness')
    # (b) x/X0
    ax = axs[1]
    for i, r in enumerate(rows):
        ybar(ax, i, r['x_over_X0_pct'], COL[r['material']], 1e-5)
    for y, (lab, x) in zip(ysum, (('front', xs_front), ('with_drift', xs_front + x_drift))):
        ybar(ax, y, 100 * x, '#bbbbbb', 1e-5)
        ax.text(100 * x * 1.2, y, f'{100 * x:.2f} %', va='center', ha='left', fontsize=7, color=ps.INK, zorder=4)
    ax.set_xscale('log')
    ax.set_xlim(1e-4, 3)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_xticklabels(['$10^{-4}$', '', '0.01', '', '1'])
    ax.set_xlabel('$x/X_0$ (%)')
    # (c) theta0
    ax = axs[2]
    for i, r in enumerate(rows):
        t = [r['theta0_deg'][T] for T in TS]
        ax.plot([t[0], t[-1]], [i, i], color='#bbbbbb', lw=1.0, zorder=2)
        for T, v in zip(TS, t):
            ax.plot(v, i, TMARK[T][0], color=TMARK[T][1], ms=4.2, mec='white', mew=0.4, zorder=4)
    for y, x in zip(ysum, (xs_front, xs_front + x_drift)):
        t = [th(x, T) for T in TS]
        ax.plot([t[0], t[-1]], [y, y], color='#bbbbbb', lw=1.0, zorder=2)
        for T, v in zip(TS, t):
            ax.plot(v, y, TMARK[T][0], color=TMARK[T][1], ms=4.2, mec='white', mew=0.4, zorder=4)
    ax.axvline(1.0, color=ps.ACCENT, lw=0.9, ls=':', zorder=1)
    ax.text(1.06, -0.55, '1°', color=ps.ACCENT, fontsize=7.5, va='top', ha='left')
    ax.set_xscale('log')
    ax.set_xlim(0.04, 12)
    ax.set_xticks([0.1, 1, 10])
    ax.set_xticklabels(['0.1°', '1°', '10°'])
    ax.set_xlabel(r'$\theta_0$ per projection')
    for T in TS:
        ax.plot([], [], TMARK[T][0], color=TMARK[T][1], ms=4.2, mec='white', mew=0.4, label=f'{T} MeV')
    ax.legend(loc='lower right', bbox_to_anchor=(1.02, 1.055), ncol=3, fontsize=7, handletextpad=0.1,
              columnspacing=0.6, borderaxespad=0.1)

    # shared row labels, group bands and separators
    i_drift = [i for i, r in enumerate(rows) if r['group'] == 'drift'][0]
    for k, ax in enumerate(axs):
        ax.set_ylim(*ylim)
        ax.set_yticks([])
        ax.grid(axis='y', visible=False)
        ax.axhspan(i_drift - 0.5, i_drift + 0.5, color='#e8f1f8', zorder=0)
        ax.axhline(i_drift + 0.5, color='#555555', lw=0.7, zorder=1)
        ax.axhline(n - 0.5 + 0.2, color='#555555', lw=0.4, zorder=1)
        ax.spines['left'].set_visible(k == 0)
        ax.tick_params(axis='x', labelsize=7.5)
        ax.text(0.0, 1.012, f'({"abc"[k]})', transform=ax.transAxes, fontweight='bold', ha='left', va='bottom')
    ax0 = axs[0]
    for i, r in enumerate(rows):
        lab = r['label'] + (r'$^\dagger$' if r['assumed'] else '')
        ax0.text(-0.03, i, lab, transform=ax0.get_yaxis_transform(), ha='right', va='center', fontsize=7.5,
                 color=ps.INK)
    for y, lab in zip(ysum, ('total in front of the drift gas', 'total including the drift gas')):
        ax0.text(-0.03, y, lab, transform=ax0.get_yaxis_transform(), ha='right', va='center', fontsize=7.5,
                 color=ps.INK, style='italic')
    # bracket for the readout board, inside panel (a) where the bars are short
    i0 = [i for i, r in enumerate(rows) if r['key'] == 'ins'][0]
    i1 = [i for i, r in enumerate(rows) if r['key'] == 'd5'][0]
    ax0.plot([2e3, 2e3], [i0 - 0.35, i1 + 0.35], color='#555555', lw=0.6, zorder=4)
    ax0.text(2.6e3, (i0 + i1) / 2, 'readout board', rotation=90, ha='left', va='center', fontsize=7, color='#555555')
    out = HERE / 'out' / 'material_stack'
    ps.save(fig, out)

    # ------------------------------------------------------------------ numbers
    js = dict(rows=[{**r, 'theta0_deg': {f'{T}MeV': round(v, 3) for T, v in r['theta0_deg'].items()},
                     'x_over_X0_pct': round(r['x_over_X0_pct'], 5)} for r in rows],
              totals_x_over_X0_pct={k: round(100 * v, 4) for k, v in tot.items()},
              totals_theta0_deg={k: {f'{T}MeV': round(th(v, T), 3) for T in TS} for k, v in tot.items()},
              readout_board_um=round(board * 1000, 1),
              mesh_equivalent_steel_um=round(2 * np.pi / 4 * 19.0 ** 2 / 64.0, 2),
              notes='Highland per projection with momentum; patterned layers averaged over their pattern; '
                    'dagger = assumed; drift gas Ar/iC4H10 95/5 at 1 atm.')
    (out.with_suffix('.json')).write_text(json.dumps(js, indent=1))

    mats = dict(air='air', mylar='Mylar', gas='Ar/iC$_4$H$_{10}$ 95/5', kapton='Kapton', Cu='copper',
                steel='steel (AISI 316)', paste='resistive paste', fr4='glass-epoxy', rohacell='Rohacell')

    def tt(mm):
        return f'{mm:g}\\,mm' if mm >= 1 else f'{mm * 1000:.0f}\\,\\textmu m'
    L = ['\\begin{tabular}{llrrrrrr}', '\\hline',
         ' & & & & & \\multicolumn{3}{c}{$\\theta_0$ (\\textdegree), $e^\\pm$ of} \\\\',
         'layer & material & thickness & fill & $x/X_0$ (\\%) & 4\\,MeV & 8\\,MeV & 16\\,MeV \\\\', '\\hline']
    for r in rows:
        lab = r['label'] + ('$^\\dagger$' if r['assumed'] else '')
        fill = '' if r['fill'] == 1 else f"{r['fill']:.2f}"
        L.append(f"{lab} & {mats[r['material']]} & {tt(r['thickness_mm'])} & {fill} & {r['x_over_X0_pct']:.4f} & "
                 + ' & '.join(f"{r['theta0_deg'][T]:.2f}" for T in TS) + ' \\\\')
        if r['key'] == 'drift':
            L.append('\\hline')
    L.append('\\hline')
    for k, lab in (('front', 'in front of the drift gas'), ('front_noCu', '\\quad without the cathode copper'),
                   ('with_drift', 'including the drift gas'), ('behind', 'behind the drift gas')):
        L.append(f"{lab} & & & & {100 * tot[k]:.4f} & " + ' & '.join(f"{th(tot[k], T):.2f}" for T in TS) + ' \\\\')
    L += ['\\hline', '\\end{tabular}',
          '% Generated by papers/paper1_v2/figs/make_material_stack.py. Fill: fraction of the area covered by the '
          'pattern (copper coverage from the Gerbers; mesh: areal mass of the weave over its 20 um thickness).']
    (HERE / 'out' / 'material_stack_table.tex').write_text('\n'.join(L) + '\n')
    print(json.dumps(js['totals_x_over_X0_pct']), json.dumps(js['totals_theta0_deg']), js['readout_board_um'])


if __name__ == '__main__':
    main()
