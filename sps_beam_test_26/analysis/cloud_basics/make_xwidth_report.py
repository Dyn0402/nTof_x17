#!/usr/bin/env python3
"""make_xwidth_report.py -- figures + report.html for the X width-growth study (FINDINGS §29).

Reads results/x_width.json, x_width_map.json, x_width_beam.json, iso_fit.json and the Magboltz grids
(results/iso, gasmodel 'beam' air_hs).  Writes <OUT>/report.html and <OUT>/figures/*.png, with
relative figure links so the DAQ page's Analysis tab can serve it.

    make_xwidth_report.py [--out ~/x17/cosmic_bench/cloud_basics/xwidth]
"""
import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [HERE, os.path.join(REPO, 'sept26_prelim_analysis')]
import figstyle as fs                                        # noqa: E402

R = f'{HERE}/results'
SCAN_GAP = 27.9
WET = fs.ACCENT          # Magboltz at the composition v asks for
DRY = fs.COPPER          # dry 95/5
DATA = fs.INK


def load(name):
    return json.load(open(f'{R}/{name}'))


def iso_grid():
    G = {}
    for f in glob.glob(f'{R}/iso/magboltz_benchi*_a0_E*_c15.json'):
        m = re.search(r'benchi([\dp]+)_w([\dp]+)_a0_E(\d+)', f)
        i, w, E = (float(m.group(k).replace('p', '.')) for k in (1, 2, 3))
        p = json.load(open(f))['points'][0]
        G[(i, w, E)] = (p['v_true_um_ns'], p['DT_um_rtcm'])
    return G


def curve(G, iso, water, what):
    Es = sorted({k[2] for k in G if k[0] == iso})
    ws = sorted({k[1] for k in G if k[0] == iso})
    j = 0 if what == 'v' else 1
    return np.array(Es), np.array([np.interp(water, ws, [G[(iso, w, E)][j] for w in ws]) for E in Es])


def fig_datasets(out, isof):
    rows = []
    for det, c in isof['chambers'].items():
        for d in c['rows']:
            k = d['key']
            if k.startswith('sat_det3__hv_scan') or k.startswith('sat_det3__drift'):
                continue
            rows.append((det, k, d))
    rows.sort(key=lambda r: (r[0], r[1]))
    fig, ax = fs.figure(figsize=fs.FIG)
    y = np.arange(len(rows))[::-1]
    for yi, (det, k, d) in zip(y, rows):
        ax.plot([d['DT_at_v95'], d['DT']], [yi, yi], color=fs.LINE, lw=1.2, zorder=1)
    ax.errorbar([r[2]['DT'] for r in rows], y, xerr=[r[2]['DTe'] for r in rows], fmt='o', color=DATA, ms=6,
                capsize=0, elinewidth=1.4, label='measured, X head-on stack', zorder=3)
    ax.plot([r[2]['DT_at_v95'] for r in rows], y, 'D', mfc='white', mec=WET, mew=1.6, ms=7,
            label='Magboltz, 95/5 + the water that gives v', zorder=2)
    ax.axvline(482, color=DRY, ls='--', lw=1.4, label='Magboltz, dry Ar/iso 95/5')
    ax.set_yticks(y, [f'{det}  {k.replace("sat_det3__long_run_resist_490V_drift_1000V", "sat_det3 (ref-only)")}'
                      for det, k, _ in rows])
    ax.set_xlabel('X transverse diffusion  D_T  [µm/√cm]')
    ax.set_xlim(200, 600)
    ax.legend(loc='upper left')
    fs.title(ax, 'Every bench dataset spreads in X like dry gas, not like the wet gas its v implies',
             'D_T = σ_drift(gap)/√gap; Magboltz at each run\'s field, Ar/iso 95/5 with the water that gives that run\'s v')
    fig.savefig(f'{out}/figures/x_dt_datasets.png'); plt.close(fig)


def fig_det3_field(out, isof, G, xw):
    rows = isof['chambers']['det3']['rows']
    scan = [d for d in rows if '__drift_scan' in d['key']]
    longr = [d for d in rows if d['key'] == 'sat_det3__long_run_resist_490V_drift_1000V'][0]
    hv = [d for d in rows if '__hv_scan' in d['key']]
    hvm = np.average([d['DT'] for d in hv], weights=[d['DTe'] ** -2 for d in hv])
    hve = np.sum([d['DTe'] ** -2 for d in hv]) ** -0.5
    fig, axs = plt.subplots(1, 2, figsize=fs.WIDE)
    for a in axs:
        fs.strip(a)
    jb = isof['chambers']['det3']['fits']['both']
    for ax, what, lab in ((axs[0], 'v', 'drift velocity  v  [µm/ns]'), (axs[1], 'DT', 'X diffusion  D_T  [µm/√cm]')):
        for (iso, w, c, ls, name) in ((5.0, 0.0, DRY, '--', 'dry 95/5'), (5.0, 0.95, WET, '-', '95/5 + 0.95 % water (v(E) fit)'),
                                      (jb['iso'], jb['water'], fs.MUTED, ':', f'best joint: iso {jb["iso"]:g} %, water {jb["water"]:.2f} %')):
            E, y = curve(G, iso, w, what)
            ax.plot(E, y, color=c, ls=ls, lw=1.8, label=name)
        key = 'v' if what == 'v' else 'DT'
        ax.errorbar([d['E'] for d in scan], [d[key] for d in scan], yerr=[d[key + ('e' if key == 'v' else 'e')] for d in scan],
                    fmt='o', color=DATA, ms=6, capsize=0, label='drift-scan sub-runs')
        if what == 'DT':
            ax.errorbar([347], [hvm], yerr=[hve], fmt='s', color=DATA, mfc='white', mew=1.5, ms=7, capsize=0,
                        label='amplification scan, 15 sub-runs (mean)')
        ax.errorbar([347], [longr[key]], yerr=[longr[key + 'e']], fmt='^', color=DATA, ms=7, capsize=0, label='long run')
        ax.set_xlabel('drift field  [V/cm]')
        ax.set_ylabel(lab)
        ax.set_xlim(150, 400)
    axs[0].set_ylim(10, 45); axs[1].set_ylim(150, 600)
    axs[1].legend(loc='upper left', fontsize=fs.BASE_PT * 0.8)
    fs.fig_title(fig, 'det3, six drift fields: v needs a cool (wet) gas, the X width a hot one',
                 f'joint v + D_T fit: χ² {jb["chi2"]:.0f}/{jb["ndf"]}; v alone '
                 f'{isof["chambers"]["det3"]["fits"]["v"]["chi2"]:.0f}/{isof["chambers"]["det3"]["fits"]["v"]["ndf"]}, '
                 f'D_T alone {isof["chambers"]["det3"]["fits"]["DT"]["chi2"]:.0f}/{isof["chambers"]["det3"]["fits"]["DT"]["ndf"]} '
                 '(det3 gap 27.9 mm)')
    fig.savefig(f'{out}/figures/x_det3_field.png'); plt.close(fig)


def fig_hv(out, xw):
    pts = []
    for k, r in xw.items():
        m = re.search(r'__hv_scan2?_resist_(\d+)V', k)
        if m:
            q = r['all']
            pts.append((int(m.group(1)), q['DT'] * np.sqrt(30 / SCAN_GAP), q['DT_err'] * np.sqrt(30 / SCAN_GAP), q['sig0'], q['sig0_err']))
    pts.sort()
    P = np.array(pts)
    sat = P[:, 0] > 505
    fig, axs = plt.subplots(1, 2, figsize=fs.WIDE)
    for a in axs:
        fs.strip(a)
    for ax, j, lab in ((axs[0], 1, 'X diffusion  D_T  [µm/√cm]'), (axs[1], 3, 'prompt footprint  σ0  [mm]')):
        ax.errorbar(P[~sat, 0], P[~sat, j], yerr=P[~sat, j + 1], fmt='o', color=DATA, ms=6, capsize=0, label='stack fit')
        ax.errorbar(P[sat, 0], P[sat, j], yerr=P[sat, j + 1], fmt='o', color=fs.MUTED, mfc='white', mew=1.4, ms=6, capsize=0,
                    label='> 505 V: stacks saturate')
        ax.set_xlabel('mesh (resist) voltage  [V]'); ax.set_ylabel(lab)
    axs[0].axhline(482, color=DRY, ls='--', lw=1.4, label='dry 95/5')
    axs[0].axhline(271, color=WET, lw=1.4, label='gas that matches v')
    axs[0].set_ylim(150, 900)
    axs[0].legend(loc='upper left', fontsize=fs.BASE_PT * 0.8)
    fs.fig_title(fig, 'The excess does not depend on gain: det3 X D_T is flat over 425–505 V',
                 'det3 6-27, drift 1000 V (347 V/cm), 15 sub-runs of ~170 head-on events each')
    fig.savefig(f'{out}/figures/x_hv_scan.png'); plt.close(fig)


def fig_position(out, mp):
    fig, axs = plt.subplots(2, 2, figsize=fs.FULL, sharey='row')
    for a in axs.flat:
        fs.strip(a)
    for j, coord in enumerate(('ref_mesh_x', 'ref_mesh_y')):
        for det, mk, c in (('det3', 'o', DATA), ('det4', 's', fs.ACCENT)):
            rows = mp[det]['bins'][coord]
            x = [r['mid'] for r in rows]
            axs[0, j].errorbar(x, [r['m2_early'] for r in rows], yerr=[r['m2_early_err'] for r in rows], fmt=mk + '-', color=c,
                               ms=6, capsize=0, lw=1.2, label=f'{det} (n {mp[det]["n"]})')
            axs[1, j].errorbar(x, [r['growth'] for r in rows], yerr=[r['growth_err'] for r in rows], fmt=mk + '-', color=c,
                               ms=6, capsize=0, lw=1.2, label=det)
        axs[1, j].set_xlabel({'ref_mesh_x': 'position across the X strips (ref_mesh_x) [mm]',
                              'ref_mesh_y': 'position along the X strips (ref_mesh_y) [mm]'}[coord])
        axs[1, j].axhline(0, color=fs.LINE, lw=1)
    axs[0, 0].set_ylabel('early ⟨x²⟩ [mm²]')
    axs[1, 0].set_ylabel('growth [mm²]')
    axs[0, 0].legend(loc='upper center', fontsize=fs.BASE_PT * 0.85)
    fs.fig_title(fig, 'The growth is everywhere; what changes with position is the early (prompt) width, at the edges',
                 'model-free second moment of the head-on X stack (±4 strips), quintiles of the reference position; '
                 'late = 600–900 ns after the threshold crossing')
    fig.savefig(f'{out}/figures/x_position.png'); plt.close(fig)


def fig_beam(out, bm):
    from gasmodel import GasGrid
    g = GasGrid('beam', 'air_hs')
    E = np.linspace(80, 260, 40)
    fig, ax = fs.figure(figsize=fs.FIG)
    ax.plot(E, [g(1.51, 0.078, e)['DT'] for e in E], color=WET, lw=1.8, label='Magboltz, run_71 gas (1.51 % water, 0.078 % air)')
    ax.plot(E, np.sqrt(2 * 0.02526 / E) * 1e4, color=fs.MUTED, ls=':', lw=1.6, label='thermal floor √(2kT/eE)')
    pts = [(243, bm['sps700']), (150, bm['sps450']), (92, bm['sps275'])]
    ax.errorbar([p[0] for p in pts], [p[1]['DT'] for p in pts], yerr=[p[1]['DT_err'] for p in pts], fmt='o', color=DATA, ms=7,
                capsize=0, label='det4 X, run_71 head-on (template-free fit)')
    ax.set_xlabel('drift field  [V/cm]'); ax.set_ylabel('X diffusion  D_T  [µm/√cm]')
    ax.set_ylim(100, 450)
    ax.legend(loc='upper right', fontsize=fs.BASE_PT * 0.85)
    fs.title(ax, 'The beam X also grows faster than Magboltz, most at low field',
             'at 92 V/cm the data are above the thermal floor no beam-gas composition can beat')
    fig.savefig(f'{out}/figures/x_beam.png'); plt.close(fig)


def table(rows, head):
    h = ''.join(f'<th>{x}</th>' for x in head)
    b = ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in r) + '</tr>' for r in rows)
    return f'<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/home/dylan/x17/cosmic_bench/cloud_basics/xwidth')
    a = ap.parse_args()
    os.makedirs(f'{a.out}/figures', exist_ok=True)
    fs.use()
    xw, mp, bm, isof, G = load('x_width.json'), load('x_width_map.json'), load('x_width_beam.json'), load('iso_fit.json'), iso_grid()
    fig_datasets(a.out, isof); fig_det3_field(a.out, isof, G, xw); fig_hv(a.out, xw); fig_position(a.out, mp); fig_beam(a.out, bm)

    # tables
    t1, dxs = [], []
    for det, c in sorted(isof['chambers'].items()):
        for d in c['rows']:
            if '__hv_scan' in d['key']:
                continue
            dx2 = d['DT'] ** 2 - d['DT_at_v95'] ** 2
            t1.append([det, d['key'], d['E'], f'{d["v"]:.1f} ± {d["ve"]:.1f}', f'{d["DT"]:.0f} ± {d["DTe"]:.0f}',
                       f'{d["water_95"]:.2f}', f'{d["DT_at_v95"]:.0f}', f'{np.sign(dx2) * np.sqrt(abs(dx2)):+.0f}'])
            if '__drift' not in d['key']:
                dxs.append(np.sign(dx2) * np.sqrt(abs(dx2)))
    dxlo, dxhi = min(dxs), max(dxs)
    isolo = min(isof['grid_iso'])
    t2 = [[det, f['v']['chi2'].__format__('.1f') + f' / {f["v"]["ndf"]}', f'{f["DT"]["chi2"]:.1f} / {f["DT"]["ndf"]}',
           f'{f["both"]["chi2"]:.1f} / {f["both"]["ndf"]}', f'{f["both"]["iso"]:g} %, {f["both"]["water"]:.2f} %']
          for det, f in sorted((d, c['fits']) for d, c in isof['chambers'].items())]
    t3 = [[k, f'{v["v"]:.1f}', f'{v["DT"]:.0f} ± {v["DT_err"]:.0f}', f'{v["DT_magboltz"]:.0f}', f'{v["Dx"]:+.0f} ± {v["Dx_err"]:.0f}',
           f'{v["DT_template"]:.0f}' if v.get('DT_template') else '—'] for k, v in bm.items()]
    d3 = mp['det3']
    t4 = [[c, f'{r["lo"]:.0f}–{r["hi"]:.0f}', r['n'], f'{r["m2_early"]:.2f} ± {r["m2_early_err"]:.2f}',
           f'{r["growth"]:.2f} ± {r["growth_err"]:.2f}', f'{r["free"]["sig0"]:.2f}', f'{r["free"]["DT"]:.0f} ± {r["free"]["DT_err"]:.0f}']
          for c in ('ref_mesh_x', 'ref_mesh_y') for r in d3['bins'][c]]
    j3 = isof['chambers']['det3']['fits']
    html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>X width growth</title><style>
body{{font:15px/1.5 system-ui,sans-serif;color:#1b2430;background:#fff;max-width:980px;margin:0 auto;padding:16px}}
h1{{font-size:24px}} h2{{font-size:18px;margin-top:32px}} .verdict{{border-left:4px solid #8a3f8f;padding:8px 14px;background:#f6f1f7}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0;display:block;overflow-x:auto}} td,th{{border-bottom:1px solid #e6e9ee;padding:3px 8px;text-align:left}}
img{{max-width:100%}} .cap{{color:#6a7583;font-size:13px}} code{{font-size:13px}}
</style></head><body>
<h1>Bench X width growth: not the gas that v implies</h1>
<p class="cap">cloud_basics FINDINGS §29 · generated by <code>make_xwidth_report.py</code> · 2026-10-12</p>
<div class="verdict"><b>Verdict.</b> On every bench dataset (5 chambers, 8 runs, plus det3's 24-sub-run drift and gain scans) the
X cluster widens with drift depth at D_T ≈ 400–520 µm/√cm, close to the dry Ar/iso 95/5 value (482). The nominal gas with
the water that each run's v requires (0.6–1.0 %) gives D_T ≈ 270–325. In quadrature the extra is <b>D_x ≈ {dxlo:.0f}–{dxhi:.0f}
µm/√cm on every chamber</b>. It does not depend on gain (det3, 425–505 V). It is spread over the whole active area, with no
hot spot. Letting the isobutane fraction float ({isolo:g}–8 %) can fit a single-field chamber with an iso-poor gas, but not
det3's five fields: no Ar/iso/water mixture gives det3's v(E) and D_T(E) together (joint χ² {j3['both']['chi2']:.0f}/{j3['both']['ndf']},
against {j3['v']['chi2']:.0f}/{j3['v']['ndf']} and {j3['DT']['chi2']:.0f}/{j3['DT']['ndf']} for each alone). The det4 beam X also grows
faster than its Magboltz gas, most at low field (+{bm['sps275']['Dx']:.0f} µm/√cm in quadrature at 92 V/cm, above the thermal floor).
<b>The excess is not gas diffusion at any composition tried.</b> It is a Magboltz/water-model problem common to both setups, or a
non-gas term that grows with depth.</div>

<h2>1 · Every dataset</h2>
<img src="figures/x_dt_datasets.png"><p class="cap">X-only head-on fit (σ0, Dd, T, amplitude, offset free; <code>x_width.py</code>),
bootstrap errors. Grey bar: the gap between the measurement and Magboltz Ar/iso 95/5 with the water that reproduces that run's v
(<code>iso_fit.py</code>; v is not monotonic in water above ~300 V/cm, so the closest grid value is taken). Dropped: g_det6_longer
(σ0 at its bound) and det3's 6-22 runs (T at its bound).</p>
{table(t1, ['chamber', 'dataset', 'E [V/cm]', 'v [µm/ns]', 'D_T [µm/√cm]', 'water from v [%]', 'Magboltz 95/5 at that water', 'D_x (quadrature)'])}

<h2>2 · det3: field and gain</h2>
<img src="figures/x_det3_field.png"><p class="cap">The six 15-minute drift-scan sub-runs (100 and 300 V are dropped: their drift
time runs past the 1.9 µs window, and 500 V only partly fits) with the M3 reference taken through the long run's alignment
(<code>headon_cache.py --subrun</code>; it reproduces the hits-chain reference exactly on the long run).</p>
{table(t2, ['chamber', 'χ² v only', 'χ² D_T only', 'χ² both', 'best joint (iso, water)'])}
<p class="cap">Free (iso, water) per chamber over the Magboltz grid (iso {isolo:g}–8 %). A single-field chamber has two numbers
for two parameters, so an iso-poor gas can always come close. Only det3's scan, with five fields, tests the idea, and the joint
fit fails there: its best point misses v at 243 V/cm by ~20 %.</p>
<img src="figures/x_hv_scan.png"><p class="cap">The fitted σ0 rises slowly with gain (0.3 → 0.5 mm) before the stacks saturate,
while D_T does not move.</p>

<h2>3 · Position on the chamber</h2>
<img src="figures/x_position.png">
<p>The growth is positive in every quintile across and along the strips (det3 0.10–0.41 mm², pooled 0.26 ± 0.03). Its
bin-to-bin scatter is ~2σ and it has no hot spot. The early (shallow-charge) width is what moves with position. On det3 it
nearly doubles in the outer quintiles (0.75–0.95 mm² against 0.43–0.52 in the middle), in both directions. That is a footprint
or edge effect that the fitted σ0 absorbs. It is not the excess.</p>
{table(t4, ['coord', 'range [mm]', 'n', 'early ⟨x²⟩', 'growth', 'fit σ0', 'fit D_T'])}

<h2>4 · The beam</h2>
<img src="figures/x_beam.png"><p class="cap">No beam template is on disk, so this is rc_drift's template-free model (the stack's
strip sum as the drive, arrival onset t_a free; <code>x_width_beam.py</code>). On bench data it reads 5–12 % below the template fit.</p>
{table(t3, ['dataset', 'v', 'D_T (template-free)', 'Magboltz', 'D_x', 'D_T (template fit)'])}

<h2>5 · What this does not rule out</h2>
<ul>
<li><b>A gas outside the grid.</b> Ar/iso {isolo:g}–8 % × water 0–1 % (to 1.5 % below 3 % iso). N2, CO2 or another
contaminant was not tried.</li>
<li><b>The edges.</b> At the outer quintiles the early width is inflated and the measured growth drops. That is partly the
±4-strip moment window truncating a wide cluster, so the growth there is a lower bound.</li>
<li><b>Magboltz's water cross sections.</b> If water cooled the transverse energy less than Magboltz says while slowing the
drift as it says, both setups would show what is seen. Bench and beam are both wet, so the data cannot separate this from a
non-gas term. A dry-gas bench run would.</li>
<li><b>Track-side terms.</b> The inclination leak goes into σ0, not Dd (§28, |tan| 0.05 → 0.01). δ-rays add width that does
not grow with depth. A term that grows linearly with depth and does not come from the gas is not identified.</li>
<li><b>The beam number's method.</b> At 92 V/cm the drift time (~6 µs) is 3× the window, so t_a and dd2 trade. The beam is
indicative, not a measurement.</li>
</ul>
</body></html>"""
    open(f'{a.out}/report.html', 'w').write(html)
    print('wrote', f'{a.out}/report.html')


if __name__ == '__main__':
    main()
