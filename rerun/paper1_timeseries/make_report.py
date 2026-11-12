#!/usr/bin/env python3
"""make_report.py -- the Paper I figure det_a_timeseries and report.html from results/ (timeseries.py's bins.csv and
timeseries.json).  Run from this directory after timeseries.py:  ../../.venv/bin/python make_report.py

Figure: chamber A, Runs 1+2, every bench measurement against wall-clock time, one shared time axis, six stacked panels:
(a) drift velocity (anchored reference ladder, filled; column duration, open) with the Magboltz water on the right axis,
(b) gain proxy (median track charge) and saturated tracks, (c) single-track efficiency, (d) position core at the mesh,
(e) angle s68, (f) time resolution.  Dotted horizontal lines: the integrated Runs 1+2 value of the paper.
"""
import html
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'papers'), os.path.join(REPO, 'rerun', 'paper1_gas')]
import paper_style as ps  # noqa: E402

R = os.path.join(HERE, 'results')
F = os.path.join(HERE, 'figures')
WHAT = {  # trend key -> (label, unit, format)
    'v': ('drift velocity, reference ladder (anchored)', 'µm/ns', '{:.2f}'),
    'vcol': ('drift velocity, column duration (no reference)', 'µm/ns', '{:.2f}'),
    'water': ('water (Magboltz, from the ladder v)', '%', '{:.3f}'),
    'q50': ('median track charge q_X + q_Y', 'ADC', '{:.0f}'),
    'f_sat': ('tracks with a strip at the ADC rail', '%', '{:.1f}'),
    'eff_int': ('single-track efficiency, interior', '%', '{:.2f}'),
    'eff_all': ('single-track efficiency, full area', '%', '{:.2f}'),
    'pos_x': ('position core at the mesh, X', 'mm', '{:.3f}'),
    'pos_y': ('position core at the mesh, Y', 'mm', '{:.3f}'),
    's68_x': ('angle s68, X', 'deg', '{:.3f}'),
    's68_y': ('angle s68, Y', 'deg', '{:.3f}'),
    'sigma_xy': ('time, (t_X - t_Y)/√2', 'ns', '{:.1f}'),
    'sigma_evt': ('time, event time against the trigger', 'ns', '{:.1f}'),
    't_evt_med': ('median event time (latency)', 'ns', '{:.1f}'),
}
QOF = {'v': 'v', 'vcol': 'vcol', 'water': 'v', 'eff_int': 'eff', 'eff_all': 'eff', 'pos_x': 'pos', 'pos_y': 'pos',
       's68_x': 'ang', 's68_y': 'ang', 'sigma_xy': 'tim', 'sigma_evt': 'tim', 't_evt_med': 'tim', 'q50': 'q', 'f_sat': 'q'}


def clock_ticks(ax):
    ticks = np.arange(23, 34, 1.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{int(t) % 24:02d}:00' if int(t) % 2 == 0 else '' for t in ticks])
    ax.set_xlim(22.6, 33.35)


def figure(df, J):
    ps.use()
    col = ps.DET_COLOR['det3']
    fig, axs = ps.fig_for('full', nrows=6, ncols=1, height=6.2, sharex=True)
    ms = 3.2
    leg = dict(fontsize=ps.BASE_PT * 0.72, handlelength=1.6, borderpad=0.2, labelspacing=0.2, ncol=2,
               columnspacing=0.9)
    I = J['integrated']

    def ser(ax, q, y, e, mk, filled=True, c=col, ls='-', label=None, xoff=0.0):
        for run in ('run1', 'run2'):
            d = df[(df.q == q) & (df.run == run)].sort_values('t')
            ax.errorbar(d.t + xoff, d[y], yerr=d[e] if e else None, color=c, marker=mk, ms=ms, lw=0.7, ls=ls,
                        mfc=c if filled else 'white', mew=0.8, elinewidth=0.7, capsize=0,
                        label=label if run == 'run1' else None)

    def ref(ax, y, c=ps.MUTED):
        ax.axhline(y, color=c, lw=0.7, ls=':', zorder=0)

    # (a) drift velocity
    ax = axs[0]
    ser(ax, 'v', 'v', 'v_err', 'o', label='reference ladder')
    ser(ax, 'vcol', 'v_col', 'v_col_err', 'o', filled=False, ls='--', label='column duration', xoff=0.04)
    ax.set_ylabel(r'$v_\mathrm{d}$ [µm/ns]'); ax.set_ylim(35.2, 37.1); ax.set_yticks([35.5, 36.0, 36.5, 37.0])
    ax.legend(loc='lower right', **leg)
    import gasinv as GI
    E = J['v']['E_Vcm']
    t = ax.twinx(); t.grid(False); t.spines['right'].set_visible(True); t.set_ylim(ax.get_ylim())
    ws = [0.90, 0.95, 1.00, 1.05]
    t.set_yticks([GI.v_of(w, E) for w in ws]); t.set_yticklabels([f'{w:.2f}' for w in ws])
    t.set_ylabel('H$_2$O [%]', labelpad=2, color=ps.MUTED); t.tick_params(axis='y', colors=ps.MUTED)
    # (b) gain proxy
    ax = axs[1]
    d = df[df.q == 'q'].copy(); d['q50k'] = d.q50 / 1e3; d['q50k_err'] = d.q50_err / 1e3
    df2 = pd.concat([df, d.assign(q='qk')])
    for run in ('run1', 'run2'):
        x = d[d.run == run]
        ax.errorbar(x.t, x.q50k, yerr=x.q50k_err, color=col, marker='D', ms=ms - 0.4, lw=0.7, ls='-', mew=0.8,
                    elinewidth=0.7, label='median charge' if run == 'run1' else None)
    ref(ax, I['q50'] / 1e3)
    ax.set_ylabel('charge\n[$10^3$ ADC]'); ax.set_ylim(13, 25.5); ax.set_yticks([15, 20, 25])
    t = ax.twinx(); t.grid(False); t.spines['right'].set_visible(True)
    for run in ('run1', 'run2'):
        x = d[d.run == run]
        t.plot(x.t, x.f_sat, color=ps.ACCENT, marker='*', ms=ms + 1.2, lw=0.7, ls='-')
    t.set_ylim(28, 60); t.set_yticks([30, 40, 50]); t.set_ylabel('saturated\n[%]', color=ps.ACCENT, labelpad=2)
    t.tick_params(axis='y', colors=ps.ACCENT)
    # (c) efficiency
    ax = axs[2]
    ser(ax, 'eff', 'eff_int', 'eff_int_err', 's', label='interior')
    ser(ax, 'eff', 'eff_all', 'eff_all_err', 's', filled=False, ls='--', label='full area', xoff=0.04)
    ref(ax, I['eff_int']); ref(ax, I['eff_all'])
    ax.set_ylabel('efficiency\n[%]'); ax.set_ylim(91.5, 99)
    ax.legend(loc='upper right', **leg)
    # (d) position core, (e) angle s68
    for ax, q, key, lab, lim, mk in ((axs[3], 'pos', 'pos', 'position core\n[mm]', (0.4, 0.68), 'v'),
                                     (axs[4], 'ang', 's68', r'angle $s_{68}$' + '\n[deg]', (0.7, 1.15), 'o')):
        for v in 'xy':
            ser(ax, q, f'{key}_{v}', f'{key}_{v}_err', mk, filled=(v == 'x'), ls=ps.VIEW_LS[v], label=v.upper(),
                xoff=0.04 if v == 'y' else 0.0)
            ref(ax, I[f'{key}_{v}'])
        ax.set_ylabel(lab); ax.set_ylim(*lim)
        ax.legend(loc='upper right', **leg)
    # (f) time
    ax = axs[5]
    ser(ax, 'tim', 'sigma_evt', 'sigma_evt_err', 'P', filled=False, ls='--', label='event time', xoff=0.04)
    ser(ax, 'tim', 'sigma_xy', 'sigma_xy_err', 'P', label=r'$(t_X-t_Y)/\sqrt{2}$')
    ref(ax, I['sigma_evt'])
    ref(ax, I['sigma_xy'])
    ax.set_ylabel(r'time $\sigma$' + '\n[ns]'); ax.set_ylim(10, 58)
    ax.legend(loc='upper right', **leg)
    # run boundary and labels
    r1, r2 = J['runs']['run1'], J['runs']['run2']
    for ax in axs:
        ax.axvspan(r1['t_hi'], r2['t_lo'], color=ps.GRID, lw=0, zorder=0)
        clock_ticks(ax)
    axs[0].text(0.5 * (r1['t_lo'] + r1['t_hi']), 1.02, 'Run 1', transform=axs[0].get_xaxis_transform(), ha='center',
                va='bottom', fontsize=ps.BASE_PT * 0.85)
    axs[0].text(0.5 * (r2['t_lo'] + r2['t_hi']), 1.02, 'Run 2', transform=axs[0].get_xaxis_transform(), ha='center',
                va='bottom', fontsize=ps.BASE_PT * 0.85)
    axs[-1].set_xlabel('time (27-28 June)')
    ps.panel_labels(axs, loc='upper left')
    fig.get_layout_engine().set(h_pad=0.02, hspace=0.02)
    fig.align_ylabels(axs)
    ps.save(fig, os.path.join(F, 'det_a_timeseries'), data=df2)


def fmt(key, x):
    return WHAT[key][2].format(x) if x is not None and np.isfinite(x) else '–'


def trend_table(J):
    T = J['trends']
    h = ['<table><tr><th>quantity</th><th>bin</th><th>bins</th><th>per-bin error</th><th>Runs 1+2 (integrated)</th>'
         '<th>line: first → last bin</th><th>slope / h (z)</th><th>Run 1 change (z)</th><th>Run 2 change (z)</th>'
         '<th>Run 2 − Run 1</th><th>χ²/dof, constant</th></tr>']
    ib = J['integrated']
    ival = dict(v=None, vcol=None, water=None, q50=ib['q50'], f_sat=ib['f_sat'], eff_int=ib['eff_int'],
                eff_all=ib['eff_all'], pos_x=ib['pos_x'], pos_y=ib['pos_y'], s68_x=ib['s68_x'], s68_y=ib['s68_y'],
                sigma_xy=ib['sigma_xy'], sigma_evt=None, t_evt_med=None)
    for k, (lab, unit, f) in WHAT.items():
        if k not in T:
            continue
        t = T[k]; pr = t['per_run']
        rc = lambda r: (f'{f.format(pr[r]["change"])} ({pr[r]["z"]:+.1f})' if r in pr else '–')
        h.append(f'<tr><td style="text-align:left">{html.escape(lab)} [{unit}]</td>'
                 f'<td>{J["bin_minutes"][QOF[k] if QOF[k] != "vcol" else "v"]} min</td><td>{t["n_bins"]}</td>'
                 f'<td>{f.format(t["median_err"])}</td><td>{fmt(k, ival[k]) if ival[k] is not None else "–"}</td>'
                 f'<td>{f.format(t["line_start"])} → {f.format(t["line_end"])}</td>'
                 f'<td>{t["slope_per_h"]:+.3g} ± {t["slope_err"]:.2g} ({t["z"]:+.1f})</td>'
                 f'<td>{rc("run1")}</td><td>{rc("run2")}</td>'
                 f'<td>{t["run2_minus_run1"]:+.3g} ± {t["run2_minus_run1_err"]:.2g}</td>'
                 f'<td>{t["chi2_const"]:.1f}/{t["dof_const"]}</td></tr>')
    return '\n'.join(h + ['</table>'])


def paragraph(J):
    T, I, V = J['trends'], J['integrated'], J['v']
    v1, v2 = V['anchored_whole']['run1'], V['anchored_whole']['run2']
    lad2 = T['v']['per_run']['run2']; col2 = T['vcol']['per_run']['run2']
    w = T['water']
    q = I['q50_by_run']; fs = I['f_sat_by_run']
    return (
        f'Runs 1 and 2 ran back to back for ten hours at the same HV and in the same slot. Binned in 20--30 minute '
        f'slices of wall-clock time ({T["eff_int"]["n_bins"]} bins, ${{\\sim}}1\\,300$ reference tracks each), the '
        f'detection efficiency, the position and angle resolutions and the time resolution are constant within their '
        f'statistical errors: the interior single-track efficiency stays at {I["eff_int"]:.1f}\\,\\% '
        f'($\\pm{T["eff_int"]["median_err"]:.1f}$\\,pt per bin; slope {T["eff_int"]["slope_per_h"]:+.2f}$\\pm$'
        f'{T["eff_int"]["slope_err"]:.2f}\\,pt/h), the position core at {I["pos_x"]:.2f}/{I["pos_y"]:.2f}\\,mm (X/Y), '
        f'the angle $s_{{68}}$ at {I["s68_x"]:.2f}/{I["s68_y"]:.2f}\\textdegree{{}} and the uncorrelated time '
        f'$(t_X-t_Y)/\\sqrt2$ at {I["sigma_xy"]:.1f}\\,ns, none with a trend beyond {max(abs(T[k]["z"]) for k in ("eff_int", "pos_x", "pos_y", "s68_x", "s68_y", "sigma_xy")):.1f}\\,$\\sigma$. '
        f'Two things do move. The drift velocity rises as the gas dries: by the reference ladder it is flat through Run~1 '
        f'at {T["v"]["per_run"]["run1"]["mean"]:.2f}\\,\\textmu m/ns and reaches {lad2["line_end"]:.2f}\\,\\textmu m/ns at the end of Run~2 '
        f'({100 * (lad2["line_end"] / T["v"]["per_run"]["run1"]["mean"] - 1):+.1f}\\,\\%; through Magboltz at {V["E_Vcm"]:.0f}\\,V/cm, '
        f'{T["water"]["per_run"]["run1"]["mean"]:.2f} to {T["water"]["per_run"]["run2"]["line_end"]:.2f}\\,\\% water). Inside Run~2 the ladder and '
        f'the reference-free column duration agree ({100 * lad2["change"] / lad2["line_start"]:+.1f} and '
        f'{100 * col2["change"] / col2["line_start"]:+.1f}\\,\\% over 7.5\\,h); between the runs only the ladder steps (+0.9\\,\\%). '
        f'The gain is {100 * (q["run2"] / q["run1"] - 1):.0f}\\,\\% higher in Run~2 (median charge {q["run1"] / 1e3:.1f} '
        f'against {q["run2"] / 1e3:.1f}\\,k ADC, saturated tracks {fs["run1"]:.0f} against {fs["run2"]:.0f}\\,\\%), a step '
        f'between the runs rather than a drift within them. Neither reaches the performance. Each run is reconstructed with its own velocity, and the '
        f'{abs(100 * lad2["change"] / lad2["line_start"]):.0f}\\,\\% left inside Run~2 moves a 20\\textdegree{{}} track by '
        f'${{\\sim}}0.2$\\textdegree{{}}, a fifth of the resolution; the gain step is what ${{\\sim}}1$\\,V on the resist '
        f'gives (one doubling per 17\\,V, \\cref{{fig:hvmetrics}}), and the HV scan shows nothing but the time '
        f'resolution moving over 60\\,V.')


def main():
    df = pd.read_csv(os.path.join(R, 'bins.csv'))
    J = json.load(open(os.path.join(R, 'timeseries.json')))
    figure(df, J)
    T, I, V = J['trends'], J['integrated'], J['v']
    para = paragraph(J)
    f2 = '{:.2f}'
    H = ['<!DOCTYPE html><html><head><meta charset="utf-8"><title>Paper I: chamber A Runs 1+2 against time</title>',
         '<style>body{font-family:sans-serif;max-width:1150px;margin:auto;line-height:1.45}table{border-collapse:collapse;'
         'font-size:13px;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 6px;text-align:right}th{background:#eee}'
         'img{max-width:100%}.v{background:#eef6ee;padding:10px;border-left:4px solid #2e7d32}'
         '.p{background:#f6f6fb;padding:10px;border-left:4px solid #0b3d91;font-family:serif}</style></head><body>',
         '<h1>Chamber A, Runs 1+2: every measurement against time</h1>',
         '<div class="v"><b>Verdict.</b> Over the ten hours of Runs 1+2 (6-27 22:49 → 6-28 08:47) the performance numbers '
         'are constant: efficiency, position core, angle s68 and time resolution show no trend beyond '
         f'{max(abs(T[k]["z"]) for k in ("eff_int", "pos_x", "pos_y", "s68_x", "s68_y", "sigma_xy")):.1f} σ, and every '
         'series is consistent with a constant except the efficiency (χ²/dof '
         f'{T["eff_int"]["chi2_const"]:.0f}/{T["eff_int"]["dof_const"]}, no slope). '
         f'<b>The drift velocity rises</b> (anchored reference ladder: flat at {T["v"]["per_run"]["run1"]["mean"]:.2f} µm/ns '
         f'through Run 1, {T["v"]["per_run"]["run2"]["line_end"]:.2f} at the end of Run 2, '
         f'{100 * (T["v"]["per_run"]["run2"]["line_end"] / T["v"]["per_run"]["run1"]["mean"] - 1):+.1f} %; water '
         f'{T["water"]["per_run"]["run1"]["mean"]:.2f} → {T["water"]["per_run"]["run2"]["line_end"]:.2f} %; the reference-free '
         'column duration agrees inside Run 2 but not on the step between the runs), '
         f'and <b>the gain is {100 * (I["q50_by_run"]["run2"] / I["q50_by_run"]["run1"] - 1):.0f} % higher in Run 2</b> '
         '(a step between the runs, flat inside each).</div>',
         '<h2>Figure (paper: det_a_timeseries)</h2><img src="figures/det_a_timeseries.png">',
         '<p>(a) drift velocity: reference ladder anchored to Run 1\'s timing v (filled) and the column duration '
         '(open, no reference involved; scaled to the same Run 1 value), water through Magboltz on the right; (b) median '
         'fitted charge q<sub>X</sub>+q<sub>Y</sub> and the fraction of tracks with a strip at the ADC rail; (c) single-track '
         'efficiency, interior (filled) and full area (open); (d) Gaussian core of the mesh residual (reference pointing and '
         'scattering included); (e) angle s68; (f) event time against the trigger (open) and (t<sub>X</sub>−t<sub>Y</sub>)/√2 '
         '(filled). X filled / Y open in (d), (e). Dotted lines: the Runs 1+2 value of the paper. Grey band: the 23 min '
         'between the runs.</p>',
         '<h2>Trends</h2>', trend_table(J),
         '<p>Line: weighted straight line in time over both runs; its slope error is inflated by √(χ²/dof) when above 1. '
         '"Run N change" = a line fitted inside that run, × its duration. z = slope / error.</p>',
         '<h2>Integrated numbers reproduced (Runs 1+2, same code)</h2><table><tr><th>quantity</th><th>here</th><th>paper</th></tr>'
         f'<tr><td>single-track efficiency, interior / full area</td><td>{I["eff_int"]:.2f} / {I["eff_all"]:.2f} % '
         f'(n = {I["n_int"]} / {I["n_all"]})</td><td>95.7 / 94.8 % (28 453 tracks)</td></tr>'
         f'<tr><td>position core X / Y</td><td>{I["pos_x"]:.3f} / {I["pos_y"]:.3f} mm</td><td>0.55 / 0.53 mm</td></tr>'
         f'<tr><td>angle s68 X / Y</td><td>{I["s68_x"]:.3f} / {I["s68_y"]:.3f}°</td><td>0.987 / 0.900°</td></tr>'
         f'<tr><td>(t_X − t_Y)/√2, Run 1 / Run 2</td><td>{I["sigma_xy_by_run"]["run1"]:.1f} / {I["sigma_xy_by_run"]["run2"]:.1f} ns'
         '</td><td>22.4 / 22.2 ns</td></tr>'
         f'<tr><td>median charge / saturated, Runs 1+2</td><td>{I["q50"]:.0f} ADC / {I["f_sat"]:.1f} %</td>'
         '<td>(hv_metrics 490 V scan row: 20 700 / 29.7 %)</td></tr></table>',
         '<h2>Drift velocity: what the two runs\' v really differ by</h2>',
         f'<p>The paper (§3.3) says Run 1 and Run 2 have v = 35.8 and 37.2 µm/ns. Those are two different estimators: 35.8 '
         f'is Run 1\'s timing velocity (head-on stack), 37.19 is Run 2\'s reference-ladder velocity (vrefit --fold-kw: '
         f'35.8 × Run 2\'s mean kw, from a reconstruction of up to 8 000 matched events). On one estimator the runs differ by much less: '
         f'the ladder gives {V["v_lad_whole"]["run1"]:.2f} and {V["v_lad_whole"]["run2"]:.2f} µm/ns over the whole rows '
         f'({100 * (V["v_lad_whole"]["run2"] / V["v_lad_whole"]["run1"] - 1):+.1f} %), i.e. {V["anchored_whole"]["run1"]:.2f} '
         f'→ {V["anchored_whole"]["run2"]:.2f} anchored to the timing scale (k = {V["k_anchor"]:.4f}, the same timing/ladder '
         f'ratio as chambers C and D: 0.970 / 0.984); the column duration gives '
         f'{100 * (I["T_col_by_run"]["run1"] / I["T_col_by_run"]["run2"] - 1):+.1f} %. '
         'Recommendation for the text: quote the change on one estimator (≈ +1.8 % ladder, +0.5 % column), not 35.8 against 37.2.</p>',
         f'<p>Shape: Run 1 is flat in both estimators (ladder {T["v"]["per_run"]["run1"]["change"]:+.2f}, column '
         f'{T["vcol"]["per_run"]["run1"]["change"]:+.2f} µm/ns over 2.3 h); inside Run 2 both rise '
         f'(ladder {T["v"]["per_run"]["run2"]["change"]:+.2f} ± {T["v"]["per_run"]["run2"]["change"] / T["v"]["per_run"]["run2"]["z"]:.2f}, '
         f'column {T["vcol"]["per_run"]["run2"]["change"]:+.2f} ± {T["vcol"]["per_run"]["run2"]["change"] / T["vcol"]["per_run"]["run2"]["z"]:.2f} µm/ns over 7.5 h). '
         'The ladder also steps up by ≈ 0.3 µm/ns (0.9 %) across the 23 min between the runs, where the column does not '
         'step. The gain steps up by 3.5 % at the same place. Whether that step is the gas (a change at the restart) or a '
         'difference between the two rows\' reconstructions (Run 2 is reconstructed with Run 1\'s bundle, v folded to 37.19, '
         'and its own w0/kw; the alignments are identical) cannot be decided from these products. The ladder is the '
         'estimator gas_drying uses for C and D; the column is gain- and attachment-sensitive (rerun/vt_all CROSSCHECK).</p>',
         '<h2>Bins and their statistics</h2><ul>'
         f'<li>v (ladder and column), efficiency, position, angle: {J["bin_minutes"]["v"]} min (5 bins in Run 1, 15 in Run 2; '
         '~700 tracks per view in the ladder window, ~1 300 interior rays, ~1 250 position fits, ~1 200 angles per view).'
         ' The ladder needs ~300 tracks per view in its window for 1 % on kw (vt_within_rows); 30 min gives 0.35 %, against '
         'a 1–2 % effect.</li>'
         f'<li>time resolution: {J["bin_minutes"]["tim"]} min (~450 M3-matched clean triggers, ±1.1 ns on 22 ns).</li>'
         f'<li>charge and saturation: {J["bin_minutes"]["q"]} min (~420 tracks; ±2.7 % on the median, ±2.4 pt on the fraction).</li>'
         '</ul><p>Finer bins are possible (the per-trigger tables are in results/bins.csv inputs), but at 15 min the '
         'efficiency error (±0.8 pt) and the ladder error (±0.5 %) approach the size of what they would look for.</p>',
         '<h2>Draft paragraph for §3.3 / §6 (not inserted in the .tex)</h2>', f'<div class="p">{html.escape(para)}</div>',
         '<h2>What this does not rule out / what could not be computed</h2><ul>'
         '<li><b>The cause of the gain step</b> (+3.5 % between the runs): no pressure, temperature or HV-current log of '
         'the bench is in the products; drying (less water) raises the gain, but a step at the restart is not a drying '
         'shape.</li>'
         '<li><b>The 0.9 % ladder step</b> at the row boundary (above): gas or row calibration, not separable.</li>'
         '<li><b>The timing-stack v</b> (head-on stack fit, the paper\'s "timing v") is not available per time bin: the '
         'head-on stack of a row has ~50–100 tracks (rerun/paper1_gas), too few to split. The column duration of the '
         'timing tables stands in for it (same physics, all angles, edge25 start to the 30 % trailing crossing).</li>'
         '<li><b>The event time</b> includes the trigger; the uncorrelated part does not. Neither drifts; the median event '
         f'time (latency) is constant to {T["t_evt_med"]["rms"]:.1f} ns rms.</li>'
         '<li><b>Efficiency scatter:</b> χ²/dof 29/19 for a constant (p ≈ 0.07), no slope; the bins are not independent of '
         'the M3 rate per bin, which is constant.</li>'
         '<li><b>Calibrations</b> (kernel, template, bow, alignment) are per row and cannot be re-derived per bin; the '
         'time series measures the stability of the result under the row\'s fixed calibration.</li>'
         '<li>Rays with no chamber event take their time from neighbouring trigger numbers (≤ seconds of error).</li></ul>',
         '<h2>Method and inputs</h2><p>timeseries.py (docstring has every definition and its source). Inputs, copied from EOS '
         'with fetch.py: Analysis_pass1 events tables and 03_angles of both rows, the effbudget rays (x17rerun/paper1_eff/budget), '
         'the interior_good angle tables (x17rerun/paper1_eff/angles), the per-trigger timing tables '
         '(x17rerun/paper1_timing/rows); and, from the repo, rerun/effbudget/single/results/single_track_area_events_*.parquet.</p>',
         '</body></html>']
    open(os.path.join(HERE, 'report.html'), 'w').write('\n'.join(H))
    print('report.html + figures/det_a_timeseries written')
    print(para)


if __name__ == '__main__':
    main()
