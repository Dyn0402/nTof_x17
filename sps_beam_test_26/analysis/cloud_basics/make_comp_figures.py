#!/usr/bin/env python3
"""make_comp_figures.py -- gas compositions shown as model curves against the data.

Reads the composition fits (beam_comp_fit_air_hs.json, driftscan_fit_air_hs_x.json,
bench_fit_x_free.json) and the data stacks, recomputes each model curve from the fitted
composition with the same physics code (predict.current_field, gasmodel 'air_hs'), and
draws data, the fitted composition, and the same gas without air (or, for the bench,
with air added) so the effect of the air is visible.

    make_comp_figures.py [--out DIR]
Writes DIR/figures/f9..f13*.png and DIR/compositions.json
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gasmodel as M                                         # noqa: E402
import predict as P                                          # noqa: E402
import beam_comp_fit as BC                                   # noqa: E402
import driftscan_fit as DS                                   # noqa: E402
import bench_fit as BF                                       # noqa: E402
from make_figures import style, INK, INK2, GRID, NEUTRAL, OUT  # noqa: E402

RES = os.path.join(HERE, 'results')
C3 = ['#2a78d6', '#eb6834', '#1baf7a']
BENCH_AIR_CONTRAST = 0.04          # % air: top of the bench grid (84 ppm O2, about half the beam's)


def J(n):
    return json.load(open(os.path.join(RES, n)))


def co2_estimate():
    """CO2 period (run_56, ZS 5 sigma, drift field assumed 243 V/cm): water from v = 12.33 um/ns
    (run_57, same gas), air from the late/early ratio R over 1.38 us (zs_timestack, all/mid selections),
    corrected for 5-sigma ZS by the run_71 emulation (+0.03..+0.05).  Rough by construction."""
    Gc = M.GasGrid('co2', 'air_hs')
    ws = np.linspace(Gc.W.min(), Gc.W.max(), 201)
    w = float(ws[np.argmin([abs(Gc(x, 0.0, 243.0)['v'] - 12.33) for x in ws])])
    Zt = J('zs_timestack.json')
    Rs = [Zt[k][f'R_x_{s}'][0] for k in ('r56_625V', 'r56_590V') for s in ('all', 'mid')]
    rates = [-np.log(min(R + c, 0.999)) / 1380.0 for R in Rs for c in (0.03, 0.05)]
    As = np.linspace(0.0, Gc.A.max(), 401)
    ev = np.array([Gc(w, x, 243.0)['etav'] for x in As])
    air = [float(np.interp(r, ev, As)) for r in (min(rates), np.median(rates), max(rates))]
    return dict(water=w, air=air[1], air_range=[air[0], air[2]], o2_ppm=air[1] * 2095,
                o2_range=[air[0] * 2095, air[2] * 2095], rate_range=[min(rates), max(rates)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    fd = os.path.join(a.out, 'figures'); os.makedirs(fd, exist_ok=True)
    summary = {}

    # ------------------------------------------------ F9: beam
    B = J('beam_comp_fit_air_hs.json'); G = M.GasGrid('beam', 'air_hs')
    H = J('headon_masked_k12.json'); t = np.array(H['t'])
    comp = (B['water'], B['air']); par = np.array(B['shaper']); sj = B['jitter']
    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1], width_ratios=[1, 1, 1, 0.9])
    for col, (lab, E) in enumerate((('raw700', 243.0), ('raw450', 150.0), ('raw275', 92.0))):
        ax = fig.add_subplot(gs[0, col]); axr = fig.add_subplot(gs[1, col], sharex=ax)
        for view, h, ls in (('x', '2', '-'), ('y', '8', '--')):
            S = np.array(H[lab][view]['sum'][h]); y = S / S[(t >= 1080) & (t <= 1260)].mean()
            e = np.maximum(np.array(H[lab][view]['band'][h]), 0.005)
            c, n, mc, q = BC.fit_stack(t, y, e, BC.cur(G, comp, E, B['k'], B['gap_spread']), par, sj)
            c0, _, m0, _ = BC.fit_stack(t, y, e, BC.cur(G, (comp[0], 0.0), E, B['k'], B['gap_spread']), par, sj)
            if view == 'x':
                ax.plot(t / 1e3, m0, color=NEUTRAL, ls=':', lw=1.5, label=f'same gas, no air (χ² {c0:.0f})')
                ax.plot(t / 1e3, mc, color=INK, lw=1.4, label=f'{comp[0]:.2f} % H$_2$O + {comp[1]:.3f} % air (χ² {c:.0f}/{n})')
            ax.plot(t / 1e3, y, ls, color=C3[col], lw=2, label=f'data {view.upper()} ' + ('±2' if view == 'x' else '±8'))
            axr.plot(t / 1e3, (y - mc) / e, ls, color=C3[col], lw=1.2)
        ax.set_title(f'run_71 RAW, {E:.0f} V/cm', fontsize=11, loc='left', color=INK)
        ax.set_ylim(-0.1, 1.15); ax.set_xlim(0.4, 3.84); style(ax); ax.legend(frameon=False, fontsize=7.5, loc='lower left')
        axr.axhline(0, color=INK2, lw=0.6); axr.set_ylim(-5, 5); axr.set_xlabel('time after trigger [µs]'); style(axr)
        if col == 0:
            ax.set_ylabel('signal / 1.08–1.26 µs level'); axr.set_ylabel('(data − model)/σ')
    ax = fig.add_subplot(gs[:, 3])
    Es = np.linspace(60, 275, 60)
    ax.plot(Es, [G(*comp, E)['v'] for E in Es], color=INK, lw=1.4, label='model v(E)')
    lv = {float(k): v for k, v in B['ladder_v'].items()}
    ax.errorbar(list(lv), [v[0] for v in lv.values()], yerr=[0.03 * v[0] for v in lv.values()], fmt='o', color=C3[1],
                ms=7, label='run_63 ladder')
    T243 = json.load(open(os.path.join(a.out, 'fits.json')))['raw700']['T']        # make_figures drift-end fit
    ax.errorbar([243.0], [30.0e3 / T243], yerr=[0.03 * 30.0e3 / T243], fmt='s', color=C3[0], ms=7,
                label='run_71 drift end (30 mm)')
    ax.set_xlabel('drift field [V/cm]'); ax.set_ylabel('drift velocity [µm/ns]'); ax.legend(frameon=False, fontsize=8)
    ax.set_title('the same composition', fontsize=11, loc='left', color=INK); style(ax)
    fig.suptitle(f'Beam gas (Ar/CF$_4$/iso + water + air): one composition, three fields, two views — '
                 f'{comp[0]:.2f} % H$_2$O, {comp[1]:.3f} % air = {comp[1] * 2095:.0f} ppm O$_2$', fontsize=11, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(os.path.join(fd, 'f9_beam_composition.png'), dpi=130); plt.close(fig)
    summary['beam'] = dict(water=comp[0], air=comp[1], o2_ppm=comp[1] * 2095, water_68=B['water_68'], air_68=B['air_68'],
                           chi2={k: v['chi2'] for k, v in B['per_stack'].items()},
                           chi2_noair={k: v['chi2_noair'] for k, v in B['per_stack'].items()})

    # ------------------------------------------------ F10: bench drift scan
    Dd = J('bench_driftscan.json'); F = J('driftscan_fit_air_hs_x.json'); Gb = M.GasGrid('bench', 'air_hs')
    g = np.array(Dd['grid']); par = np.array(F['shaper']); sj = F['jitter']
    fig, axs = plt.subplots(2, 6, figsize=(18, 6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    for k, (V, E) in enumerate(zip(Dd['volts'], Dd['E_Vcm'])):
        r = Dd[str(V)]['x']; y = np.array(r['stack'])
        e = np.maximum(np.nan_to_num(np.array(r['band']), nan=1.0), 0.004)
        c, n, mc, _ = DS.fit_field(g, y, e, DS.cur(Gb, (F['water'], F['air']), float(E), F['k'], F['gap_spread']), par, sj)
        ca, _, ma, _ = DS.fit_field(g, y, e, DS.cur(Gb, (F['water'], BENCH_AIR_CONTRAST), float(E), F['k'], F['gap_spread']), par, sj)
        ax = axs[0, k]
        ax.plot(g, ma, color=NEUTRAL, ls=':', lw=1.5, label=f'+{BENCH_AIR_CONTRAST} % air (χ² {ca:.0f})')
        ax.plot(g, mc, color=INK, lw=1.3, label=f'fit (χ² {c:.0f}/{n})')
        ax.plot(g, y, 'o', color=C3[0], ms=2.5, label='data X')
        ax.plot(g, np.array(Dd[str(V)]['y']['stack']), '-', color=C3[1], lw=0.8, alpha=0.8, label='data Y')
        ax.set_title(f'{V} V ({E:.0f} V/cm)', fontsize=10, loc='left', color=INK); ax.set_ylim(-0.1, 1.15); style(ax)
        ax.legend(frameon=False, fontsize=6.5, loc='lower center')
        axs[1, k].plot(g, (y - mc) / e, '-', color=C3[0], lw=1); axs[1, k].axhline(0, color=INK2, lw=0.6)
        axs[1, k].set_ylim(-4, 4); axs[1, k].set_xlabel('t [ns]'); style(axs[1, k])
    axs[0, 0].set_ylabel('signal / peak'); axs[1, 0].set_ylabel('(data − fit)/σ')
    fig.suptitle(f'Bench det3 drift scan (Ar/iso + water + air, 6-27): one composition, six fields — '
                 f'{F["water"]:.2f} % H$_2$O, no air (90 % CL ≲ 2 ppm O$_2$ statistical; ≲ 10 ppm robust)',
                 fontsize=11, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(os.path.join(fd, 'f10_bench_driftscan.png'), dpi=120); plt.close(fig)
    summary['bench_det3_driftscan'] = dict(water=F['water'], air=F['air'], k=F['k'], gap_spread=F['gap_spread'],
                                           chi2={k: v['chi2'] for k, v in F['per_field'].items()})

    # ------------------------------------------------ F11: bench chambers (single field)
    S = J('bench_stack.json'); BFj = J('bench_fit_x_free.json'); par = np.array(BFj['shaper'])
    g = np.array(S['grid'])
    fig, axs = plt.subplots(2, 4, figsize=(15, 5.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    for k, d in enumerate(('det2', 'det3', 'det4', 'det7')):
        f = BFj[d]; E, gap = BF.CH[d]
        y = np.array(S[d]['x']['stack']); e = np.maximum(np.array(S[d]['x']['band']), 0.004)
        q, c, n, mc = BF.fit_curve(g, y, e, BF.cur(Gb, (f['water'], f['air']), E, gap, f['k'], f['gap_spread']), par)
        qa, ca, _, ma = BF.fit_curve(g, y, e, BF.cur(Gb, (f['water'], BENCH_AIR_CONTRAST), E, gap, f['k'], f['gap_spread']), par)
        ax = axs[0, k]
        ax.plot(g, ma, color=NEUTRAL, ls=':', lw=1.5, label=f'+{BENCH_AIR_CONTRAST} % air (χ² {ca:.0f})')
        ax.plot(g, mc, color=INK, lw=1.3, label=f'{f["water"]:.2f} % H$_2$O, {f["air"]:.3f} % air (χ² {c:.0f}/{n})')
        ax.plot(g, y, 'o', color=C3[0], ms=2.5, label=f'data X (n={S[d]["x"]["n"]})')
        ax.set_title(f'bench {d}, {E:.0f} V/cm, gap {gap} mm', fontsize=10, loc='left', color=INK); ax.set_ylim(-0.1, 1.15)
        style(ax); ax.legend(frameon=False, fontsize=7, loc='lower center')
        axs[1, k].plot(g, (y - mc) / e, '-', color=C3[0], lw=1); axs[1, k].axhline(0, color=INK2, lw=0.6)
        axs[1, k].set_ylim(-5, 5); axs[1, k].set_xlabel('t [ns]'); style(axs[1, k])
    axs[0, 0].set_ylabel('signal / peak')
    fig.tight_layout(); fig.savefig(os.path.join(fd, 'f11_bench_chambers.png'), dpi=120); plt.close(fig)
    summary['bench_chambers'] = {d: dict(water=BFj[d]['water'], air=BFj[d]['air'], air_68=BFj[d]['air_68'],
                                         chi2=BFj[d]['chi2'], n=BFj[d]['n']) for d in ('det2', 'det3', 'det4', 'det7')}


    # ------------------------------------------------ F13: ZS beam arms -- composition fitted per block
    # The ZS distortion is ADDITIVE (zero suppression removes small and negative samples, so a ratio
    # breaks where the signal crosses zero): D(t) = ZS4-emulated - RAW, both normalised, from run_71 at
    # the nearest field.  Errors: bootstrap (+) 1 % systematic for carrying run_71's distortion over.
    Zt = J('zs_timestack.json'); tz = np.array(Zt['t']); Em = J('zs_emulate.json')
    par = np.array(B['shaper']); sj = B['jitter']
    arms = (('r63_d425', 142.0, 'raw450', 'run_63 25.64°, 142 V/cm (X head-on)', '00:22–00:30'),
            ('r63_d325', 108.0, 'raw275', 'run_63 25.64°, 108 V/cm (X head-on)', '00:30–00:37'),
            ('r63_flat700', 243.0, 'raw700', 'run_63 flat, 243 V/cm', 'after the 25° block'))
    ws = np.round(np.linspace(G.W.min(), G.W.max(), 31), 4); as_ = np.round(np.linspace(0.0, G.A.max(), 27), 4)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    zs_out = {}
    for ax, (arm, E, ref, nm, when) in zip(axs, arms):
        y = np.array(Zt[arm]['x_all'])
        e = np.sqrt(np.array(Zt[arm]['x_all_band']) ** 2 + 0.01 ** 2)
        D = np.array(Em[ref]['x']['curve_zs4_x2']) - np.array(Em[ref]['x']['curve_raw_x2'])
        yr = y - D
        chi = np.array([[BC.fit_stack(tz, yr, e, BC.cur(G, (float(w), float(x)), E, B['k'], B['gap_spread']), par, sj)[0]
                         for x in as_] for w in ws])
        i0, j0 = np.unravel_index(np.argmin(chi), chi.shape)
        cb = (float(ws[i0]), float(as_[j0]))
        ok = chi <= chi.min() + 2.30
        c, n, mc, q = BC.fit_stack(tz, yr, e, BC.cur(G, cb, E, B['k'], B['gap_spread']), par, sj)
        c0, _, m0, _ = BC.fit_stack(tz, yr, e, BC.cur(G, (cb[0], 0.0), E, B['k'], B['gap_spread']), par, sj)
        c71, _, m71, _ = BC.fit_stack(tz, yr, e, BC.cur(G, comp, E, B['k'], B['gap_spread']), par, sj)
        ax.plot(tz / 1e3, np.array(m0) + D, color=NEUTRAL, ls=':', lw=1.5, label=f'same water, no air (χ² {c0:.0f})')
        ax.plot(tz / 1e3, np.array(m71) + D, color=INK2, ls='--', lw=1.1, label=f'run_71 composition (χ² {c71:.0f})')
        ax.plot(tz / 1e3, np.array(mc) + D, color=INK, lw=1.4,
                label=f'fit {cb[0]:.2f} % H$_2$O, {cb[1]:.3f} % air (χ² {c:.0f}/{n})')
        ax.plot(tz / 1e3, y, '-', color=C3[0], lw=2, label='data X ±2 (ZS 4σ)')
        ax.set_title(f'{nm}\n{when}', fontsize=9.5, loc='left', color=INK); ax.set_xlabel('time after trigger [µs]')
        ax.set_xlim(0.4, 3.84); ax.set_ylim(-0.1, 1.2); style(ax); ax.legend(frameon=False, fontsize=7, loc='lower left')
        zs_out[arm] = dict(E=E, water=cb[0], air=cb[1], o2_ppm=cb[1] * 2095,
                           water_68=[float(ws[ok.any(1)].min()), float(ws[ok.any(1)].max())],
                           air_68=[float(as_[ok.any(0)].min()), float(as_[ok.any(0)].max())],
                           chi2=c, n=n, chi2_noair=c0, chi2_run71comp=c71)
        print(arm, zs_out[arm])
    axs[0].set_ylabel('signal / 1.08–1.26 µs level')
    fig.suptitle('Zero-suppressed run_63 (the night before run_71): composition fitted per block, ZS distortion from RAW run_71',
                 fontsize=10.5, x=0.01, ha='left')
    fig.tight_layout(); fig.savefig(os.path.join(fd, 'f13_zs_arms.png'), dpi=130); plt.close(fig)
    summary['zs_arms'] = zs_out

    # ------------------------------------------------ F12: composition map (after F13: uses the run_63 fits)
    co2 = co2_estimate(); summary['co2_period'] = co2
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    o2 = np.logspace(0, 3.6, 100)
    ax.plot(o2, o2 * 1e-4 * 0.05, color=NEUTRAL, ls='--', lw=1.2)
    ax.text(250, 0.07, 'what a bulk room-air leak brings (H$_2$O/O$_2$ ≈ 0.05)', fontsize=8, color=INK2)
    beam_pts = [('run_63 25.6°, Aug 3 00:22', zs_out['r63_d425'], 'D'), ('run_63 25.6°, 00:30', zs_out['r63_d325'], 'D'),
                ('run_63 flat, 01:00–01:54', zs_out['r63_flat700'], 'o'),
                ('run_71 RAW, 05:22–05:52', dict(o2_ppm=comp[1] * 2095, water=comp[0], air_68=B['air_68'],
                                               water_68=B['water_68']), 'o')]
    for k, (nm, r, mk) in enumerate(beam_pts):
        xe = [[r['o2_ppm'] - r['air_68'][0] * 2095], [r['air_68'][1] * 2095 - r['o2_ppm']]]
        ye = [[r['water'] - r['water_68'][0]], [r['water_68'][1] - r['water']]]
        ax.errorbar([r['o2_ppm']], [r['water']], xerr=xe, yerr=ye, fmt=mk, color=C3[0], ms=8, mfc='white' if mk == 'D' else C3[0],
                    capsize=0, zorder=3)
        off = [(-150, 14), (10, 10), (-120, 4), (-118, -14)][k]
        ax.annotate(nm, (r['o2_ppm'], r['water']), xytext=off, textcoords='offset points', fontsize=7.5, color=INK2,
                    arrowprops=dict(arrowstyle='-', color=GRID, lw=0.8))
    ax.errorbar([co2['o2_ppm']], [co2['water']], xerr=[[co2['o2_ppm'] - co2['o2_range'][0]], [co2['o2_range'][1] - co2['o2_ppm']]],
                fmt='s', color=C3[1], ms=8, capsize=0)
    ax.annotate('run_56 CO$_2$ gas, Aug 1 (ZS, rough)', (co2['o2_ppm'], co2['water']), xytext=(-40, -26),
                textcoords='offset points', fontsize=7.5, color=INK2)
    # bench: upper limits as arrows
    lim = [('bench det3 (6-field scan) and det2', 10.0, F['water'])] + \
          [(f'bench {d}', max(BFj[d]['air_68'][1] * 2095, 10.0), BFj[d]['water']) for d in ('det4', 'det7')]
    for k, (nm, ul, w) in enumerate(lim):
        ax.annotate('', xy=(ul / 4, w), xytext=(ul, w), arrowprops=dict(arrowstyle='->', color=C3[2], lw=1.6))
        ax.plot([ul], [w], '|', color=C3[2], ms=12, mew=2)
        ax.annotate(nm, (ul, w), xytext=(6, -3), textcoords='offset points', fontsize=7.5, color=INK2)
    ax.set_xscale('log'); ax.set_xlim(1, 2000); ax.set_ylim(0, 2.0)
    ax.set_xlabel('O$_2$ [ppm]  (= air fraction × 0.2095; bench: upper limits)'); ax.set_ylabel('water [%]')
    ax.set_title('Where the water and the oxygen are: beam (blue, orange) and bench (green)', fontsize=11, loc='left', color=INK)
    style(ax)
    fig.tight_layout(); fig.savefig(os.path.join(fd, 'f12_composition_map.png'), dpi=130); plt.close(fig)
    json.dump(summary, open(os.path.join(a.out, 'compositions.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1)[:2500])


if __name__ == '__main__':
    main()
