#!/usr/bin/env python3
"""make_report_final.py: report/report_final.html + report/figures/final_*.png from results/final.json, final_resid.npz,
footprint.csv, baseline.json, seed_census.csv and pass_hists.json.  Relative figure links."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results'); REP = os.path.join(HERE, 'report'); FIG = os.path.join(REP, 'figures')
ESTN = {'p0': 'fit p0 (mesh)', 'p_mid': 'fit at 7.5 mm', 'c_pos': 'charge centroid', 'hits': 'hits (run-53 method)'}


def um(x):
    return '—' if x is None or not np.isfinite(x) else f'{x * 1e3:.0f}'


def fig_fiducial(R, name):
    rx, ry, ty = R[f'{name}|p0|rx_all'], R[f'{name}|p0|ry_all'], R[f'{name}|ty']
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(np.clip(rx, -20, 20), bins=200, histtype='step', label='det4 Y residual (telescope x)')
    ax[0].hist(np.clip(ry, -20, 20), bins=200, histtype='step', label='det4 X residual (telescope y)')
    ax[0].axvspan(-1.5, 1.5, color='g', alpha=.15, label='cross-view fiducial ±1.5 mm')
    ax[0].set_yscale('log'); ax[0].set_xlabel('residual [mm] (clipped at ±20)'); ax[0].legend(fontsize=8)
    ax[0].set_title(f'{name}: a narrow core on a flat floor')
    b = np.arange(0, 131, 2.0); i = np.digitize(ty, b)
    f = [np.mean(np.abs(ry[i == k]) > 1.5) if (i == k).sum() > 20 else np.nan for k in range(1, len(b))]
    ax[1].step(0.5 * (b[1:] + b[:-1]), f, where='mid')
    ax[1].set_xlabel('telescope y at det4 [mm] (runs along det4 X)'); ax[1].set_ylabel('fraction with |det4 X residual| > 1.5 mm')
    ax[1].set_title('the floor is the dead amplification stripes in X'); ax[1].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'final_fiducial.png'), dpi=100); plt.close(fig)


def fig_resid(R, sets):
    fig, ax = plt.subplots(2, len(sets), figsize=(4.6 * len(sets), 7.5), squeeze=False)
    bins = np.linspace(-1.5, 1.5, 61)
    for j, s in enumerate(sets):
        for i, v in enumerate('xy'):
            for e, col in (('p0', 'C0'), ('p_mid', 'C2'), ('c_pos', 'C1'), ('hits', 'C4')):
                k = f'{s}|{e}|{v}'
                if k in R.files:
                    ax[i, j].hist(R[k], bins=bins, histtype='step', color=col, density=True, label=ESTN[e])
            ax[i, j].set_title(f'{s}: det4 {v.upper()}', fontsize=9); ax[i, j].set_xlabel('residual [mm]')
            ax[i, j].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'final_residuals.png'), dpi=95); plt.close(fig)


def fig_footprint(F):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    r66 = F[F.key.str.startswith('run66')].sort_values('resist_V')
    for v, m in (('x', 'o-'), ('y', 's--')):
        ax[0, 0].plot(r66.resist_V, r66[f'{v}_strips_mean'], m, label=f'{v.upper()}')
        ax[0, 1].plot(r66.resist_V, r66[f'{v}_lead_frac'], m, label=f'{v.upper()}')
        ax[0, 2].plot(r66.resist_V, r66[f'{v}_q_sum'], m, label=f'{v.upper()}')
    for a_, t in zip(ax[0], ('mean cluster strips', 'median lead-strip fraction', 'median cluster charge [ADC]')):
        a_.set_xlabel('resist voltage [V] (run66, CF4, drift 243 V/cm)'); a_.set_title(t); a_.legend()
    for run, m in (('run55', 'o'), ('run70', 's'), ('run71', '^')):
        d = F[F.key.str.startswith(run)].sort_values('field')
        for v, ls in (('x', '-'), ('y', '--')):
            ax[1, 0].plot(d.field, d[f'{v}_strips_mean'], m + ls, label=f'{run} {v.upper()}')
            ax[1, 1].plot(d.field, d[f'{v}_lead_frac'], m + ls, label=f'{run} {v.upper()}')
            if f'{v}_t_le_sigma_ns' in d:
                ax[1, 2].plot(d.field, d[f'{v}_t_le_sigma_ns'], m + ls, label=f'{run} {v.upper()}')
    for a_, t in zip(ax[1], ('mean cluster strips', 'median lead-strip fraction', 'σ leading edge (ftst-corrected) [ns]')):
        a_.set_xlabel('drift field [V/cm]'); a_.set_title(t); a_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'final_footprint.png'), dpi=95); plt.close(fig)


def zone_txt(V):
    z = ', '.join(f'{q["pitch"]}: {q["sigma_res"] * 1e3:.0f}±{q["err"] * 1e3:.0f}→{q["sigma_det4"] * 1e3:.0f}' for q in V['zones'])
    if V.get('fit'):
        z += f' | <b>3-zone fit: det4 {V["fit"]["sigma_det4"] * 1e3:.0f}±{V["fit"]["sigma_det4_err"] * 1e3:.0f} µm, f = {V["fit"]["f_back"]:.3f}</b>'
    return z or '—'


def main():
    os.makedirs(FIG, exist_ok=True)
    A = json.load(open(os.path.join(RES, 'final.json'))); R = np.load(os.path.join(RES, 'final_resid.npz'))
    F = pd.read_csv(os.path.join(RES, 'footprint.csv'))
    fig_fiducial(R, 'run63 flat700' if 'run63 flat700|p0|rx_all' in R.files else next(iter(A)))
    fig_resid(R, [s for s in ('run53', 'run63 flat700', 'run66 r580-r730', 'run71 raw700') if f'{s}|p0|x' in R.files])
    fig_footprint(F)
    h = ['<!DOCTYPE html><html><head><meta charset="utf-8"><title>det4 at SPS H4: final numbers</title>',
         '<style>body{font-family:sans-serif;max-width:1200px;margin:auto;padding:1em}table{border-collapse:collapse;font-size:12px}'
         'td,th{border:1px solid #bbb;padding:3px 6px}th{background:#eee}img{max-width:100%}</style></head><body>',
         '<h1>det4 (mx17_E) at SPS H4: final position resolution, footprint, timing</h1>',
         '<p>Generated by <code>rerun/sps_audit/make_report_final.py</code>. Method, cuts and the run-53 answer: '
         '<code>rerun/sps_audit/README.md</code>. Earlier audit page: <a href="report.html">report.html</a>.</p>',
         '<h2>Resolution against the uRWELL telescope</h2><p>Residual σ per back-plane zone (µm), → det4 after the telescope '
         'pointing (run-53 f = 0.301); the 3-zone fit measures f and det4 together where all three zones have ≥ 150 tracks.</p>',
         '<table><tr><th>dataset</th><th>estimator</th><th>det(A)</th><th>fiducial</th><th>det4 X: n | zones</th><th>det4 Y: n | zones</th></tr>']
    for name, D in A.items():
        for e, S in D['est'].items():
            h.append(f'<tr><td>{name}</td><td>{ESTN[e]}</td><td>{S["det_A"]:+.3f}</td><td>{S["fid_frac"]:.2f}</td>'
                     f'<td>{S["X"]["n_fid"]} | {zone_txt(S["X"])}</td><td>{S["Y"]["n_fid"]} | {zone_txt(S["Y"])}</td></tr>')
    h.append('</table><img src="figures/final_residuals.png">')
    h.append('<h2>Selection</h2><img src="figures/final_fiducial.png"><p>Cuts tested and not applied (core σ in µm, '
             'passing / failing / all, fit p0):</p><table><tr><th>dataset</th><th>test</th><th>passes</th><th>X</th><th>Y</th></tr>')
    for name, D in A.items():
        for t, T in D.get('tests', {}).items():
            f = lambda q: '/'.join(um(x[0]) for x in (q['pass_'], q['fail'], q['all']))
            h.append(f'<tr><td>{name}</td><td>{t}</td><td>{T["frac_pass"]:.2f}</td><td>{f(T["X"])}</td><td>{f(T["Y"])}</td></tr>')
    h.append('</table><h2>Footprint and timing (flat rows, no pass cut)</h2><img src="figures/final_footprint.png">')
    cols = [c for c in ('key', 'field', 'resist_V', 'n', 'x_strips_mean', 'y_strips_mean', 'x_lead_frac', 'y_lead_frac', 'x_nb', 'y_nb',
                        'x_q_sum', 'y_q_sum', 't_le_sigma_xy_ns', 'x_t_le_sigma_ns', 'y_t_le_sigma_ns') if c in F]
    h.append('<table><tr>' + ''.join(f'<th>{c}</th>' for c in cols) + '</tr>')
    for _, r in F.iterrows():
        h.append('<tr>' + ''.join(f'<td>{r[c]:.3g}</td>' if isinstance(r[c], float) else f'<td>{r[c]}</td>' for c in cols) + '</tr>')
    h.append('</table></body></html>')
    open(os.path.join(REP, 'report_final.html'), 'w').write('\n'.join(h))
    print('wrote', os.path.join(REP, 'report_final.html'))


if __name__ == '__main__':
    main()
