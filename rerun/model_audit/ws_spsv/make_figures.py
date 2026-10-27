#!/usr/bin/env python3
"""make_figures.py --stack zsstack_fixed.json --res results.json [--dp DIR] --out figures/

F16 slide figures (fixed caches only):
  f16_grid_artefact.png  the audit's v_col = 28.8 mm / uend against the depth-grid end K*60 ns: uend sat on the grid
                         end, K was set from v_span, so v_span / v_col = 1.16 at every plateau by construction
  f16_column_243.png     stacked strip-sum waveforms at 243 V/cm (head-on RAW, head-on ZS, 25.64 deg ZS) with the box (x)
                         template fits: the column lasts 2.0-2.1 us on all of them
  f16_v_of_E.png         v(E): 28.8 mm / T and 30 mm / T (model-free), the fitted slope |w|/tan, the span route, and
                         Magboltz for the SPS gas dry / +1 % / +2 % H2O
  f16_gap_eff.png        gap_eff = |w| T / tan per plateau: no drift velocity enters
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, MUTED, GRID = '#1f2933', '#7b8794', '#e4e7eb'
C = ['#1f6feb', '#d97706', '#0f766e', '#b42318', '#7c3aed']
plt.rcParams.update({'font.size': 12, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'axes.spines.top': False,
                     'axes.spines.right': False})
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
SPAN = {234.5: 14.0, 217.1: 13.2, 199.8: 12.4, 182.4: 11.8, 165.0: 11.6, 147.6: 11.6}
OLD_K = {234.5: 40, 217.1: 42, 199.8: 45, 182.4: 47, 165.0: 48, 147.6: 48}       # make_sps_dag: int(30000/v/60)+5
OLD_UEND = {234.5: 2400, 217.1: 2520, 199.8: 2700, 182.4: 2820, 165.0: 2880, 147.6: 2880}   # sps2 base, both views p50


def fig_artefact(out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    E = np.array(sorted(SPAN, reverse=True)); vs = np.array([SPAN[e] for e in E])
    grid = np.array([OLD_K[e] * 60 for e in E]); ue = np.array([OLD_UEND[e] for e in E])
    ax[0].plot(E, grid, 's', ms=11, mfc='none', color=MUTED, label='depth-grid end K·60 ns (K from v_span)')
    ax[0].plot(E, ue, 'o', color=C[0], label='fitted column end q_uend (median)')
    ax[0].set(xlabel='drift field [V/cm]', ylabel='column end [ns]', title='old F16: the column end sat on the grid end')
    ax[0].legend(frameon=False, fontsize=10)
    ax[1].plot(E, vs / (28800 / ue), 'o', color=C[3], label='v_span / v_col (the "14 %")')
    ax[1].plot(E, vs / (28800 / ((30000 / vs // 60 + 5) * 60)), '_', ms=20, color=INK, label='predicted from K formula alone')
    ax[1].set(xlabel='drift field [V/cm]', ylabel='ratio', ylim=(1.0, 1.3), title='so the constant ratio is arithmetic')
    ax[1].legend(frameon=False, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'f16_grid_artefact.png'), dpi=150); plt.close(fig)


def fig_column(ST, out):
    t = np.arange(64) * 60.0
    sets = [('run71_raw_raw700', 'x', 'head-on, RAW (run_71)'), ('run63_flat_operating_03_flat700', 'x', 'head-on, ZS (run_63)'),
            ('run62_rot25_ladder_driftscan_gap150V_d700', 'y', '25.64°, ZS (run_62), inclined view')]
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (k, v, lab) in enumerate(sets):
        r = ST[k][v]; s = np.array([np.nan if x is None else x for x in r['stack']]); m = np.array(r['model'])
        sc = 1.0 / np.nanmax(m)
        ax.plot(t, s * sc, 'o', ms=4, color=C[i], alpha=0.8)
        ax.plot(t, m * sc, '-', color=C[i], lw=2, label=f"{lab}: T = {r['T_box']:.0f} ns (flat box {r['T_flat']:.0f})")
    ax.set(xlabel='time in the 64 × 60 ns window [ns]', ylabel='strip-sum waveform (normalised)',
           title='243 V/cm: the drift column lasts 2.0–2.1 µs, head-on or inclined')
    ax.legend(frameon=False, fontsize=10, loc='upper right')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'f16_column_243.png'), dpi=150); plt.close(fig)


def magboltz():
    d = os.path.join(REPO, 'garfield_sim', 'results')
    out = {}
    for tag, f in (('dry', 'drift_velocity_beamtest_cf4_CERN.json'), ('+1 % H₂O', 'drift_velocity_beamtest_cf4_wet1_CERN.json'),
                   ('+2 % H₂O', 'drift_velocity_beamtest_cf4_wet2_CERN.json')):
        p = json.load(open(os.path.join(d, f)))['points']
        out[tag] = (np.array([x['E_Vcm'] for x in p]), np.array([x['v_um_per_ns'] for x in p]))
    return out


def rows(R, arm='z'):
    out = []
    for k, r in R.items():
        y = r.get('y') or {}
        if r['arm'] != arm or not r['tan'] or not y.get('T_box') or '_d' not in r['key'] or r['field'] is None:
            continue
        out.append((r['field'], abs(y['w']) / r['tan'], y['T_box'], y['T_flat'], r['key']))
    return sorted(out)


def fig_v(R, out):
    rr = rows(R); E = np.array([x[0] for x in rr]); vsl = np.array([x[1] for x in rr]); Tb = np.array([x[2] for x in rr])
    Tf = np.array([x[3] for x in rr]); ok = E >= 170
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, (tag, (e, v)) in enumerate(magboltz().items()):
        m = e <= 260
        if tag != 'dry':
            ax.plot(e[m], v[m], '--', color=MUTED, lw=1)
            ax.text(e[m][-1] + 2, v[m][-1], f'Magboltz {tag}', color=MUTED, fontsize=9, va='center')
    ax.errorbar(E[ok], 28800 / Tb[ok], yerr=28800 / Tb[ok] * 0.03, fmt='s', color=C[0], label='28.8 mm / T (model-free column)')
    ax.plot(E[ok], 30000 / Tb[ok], 's', mfc='none', color=C[0], label='30 mm / T')
    ax.plot(E[ok], vsl[ok], 'o', color=C[3], label='fitted slope |w|/tan 25.64° (bench model)')
    es = np.array(sorted(SPAN)); ax.plot(es[es >= 180], [SPAN[e] for e in es[es >= 180]], 'D', mfc='none', color=C[1],
                                         label='span route (old registry)')
    ax.plot(E[~ok], 28800 / Tb[~ok], 'x', color=MUTED, label='< 170 V/cm: column overruns the window')
    ax.set(xlabel='drift field (HV / 28.8 mm) [V/cm]', ylabel='drift velocity [µm/ns]', xlim=(140, 290), ylim=(6, 22),
           title='det4 at SPS, Ar/CF₄/iso 88/10/2 (wet)')
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'f16_v_of_E.png'), dpi=150); plt.close(fig)


def fig_gap(R, out, DP):
    rr = rows(R); E = np.array([x[0] for x in rr]); g = np.array([x[1] * x[2] / 1e3 for x in rr])
    gf = np.array([x[1] * x[3] / 1e3 for x in rr]); ok = E >= 170
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhspan(28.8, 30.0, color=GRID, alpha=0.8, lw=0); ax.text(176, 29.35, 'nominal 28.8–30 mm', color=MUTED, fontsize=10)
    ax.plot(E[ok], g[ok], 'o', color=C[3], label='|w|·T_box / tan  (bench model, Dp 0.020)')
    ax.plot(E[ok], gf[ok], 'o', mfc='none', color=C[3], label='|w|·T_flat / tan')
    for i, (lab, pts) in enumerate((DP or {}).items()):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], 'D', color=C[i + 1], label=lab)
    ax.set(xlabel='drift field [V/cm]', ylabel='implied gap [mm]', ylim=(22, 31), xlim=(165, 250),
           title='gap implied by slope × column duration (no drift velocity enters)')
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'f16_gap_eff.png'), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stack', required=True); ap.add_argument('--res', required=True)
    ap.add_argument('--dp', default=None, help='JSON {label: [[E, gap], ...]} from the Dp arms'); ap.add_argument('--out', required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ST = json.load(open(a.stack)); R = json.load(open(a.res)); DP = json.load(open(a.dp)) if a.dp else None
    fig_artefact(a.out); fig_column(ST, a.out); fig_v(R, a.out); fig_gap(R, a.out, DP)
    print('figures in', a.out)


if __name__ == '__main__':
    main()
