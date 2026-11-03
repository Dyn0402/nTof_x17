#!/usr/bin/env python3
"""make_report.py -- Paper I §7 timing: figures + report.html from results/ (rows_matched.csv, rows.csv, angle.json,
amplitude.json, sps_timing.json, t0_position.json).  Run from this directory: ../../.venv/bin/python make_report.py

Figures (figures/): timing.pdf/png (the paper figure: (a) HV scan, (b) amplitude curve), timing_driftscan (appendix), timing_all (appendix A3),
angle.png, drift.png.  Depths in mm use each row's timing v: the S4 golden-row v of the chamber scaled by the row's
column duration, v_row = v_S4 x T_golden / T_row (colsum's median T; relative T is what colsum measures well).
"""
import csv
import html
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
F = os.path.join(HERE, 'figures')
os.makedirs(F, exist_ok=True)
#: timing v at the golden-row conditions, S4 re-check (rerun/calib/S4_PHYSICS_INPUTS_2026-10-30.md, masked interior)
V_S4 = {'mx17_2': 38.25, 'mx17_3': 36.03, 'mx17_4': 32.70, 'mx17_6': 24.78, 'mx17_7': 36.83}
GOLD = {'mx17_2': 'o22_long_det2', 'mx17_3': 'sat_det3', 'mx17_4': 'g_det4', 'mx17_6': 'g_det6_long', 'mx17_7': 'g_det7_long'}
QUOTED = {'mx17_2': 'c26__mx17_det2_det3_overnight_6-22-26__long_run__mx17_2',
          'mx17_3': 'c26__mx17_det3_p2_det1_overnight_6-27-26__long_run_p2_det1_sanity_check__mx17_3'}
NAME = {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4', 'mx17_6': 'det6', 'mx17_7': 'det7'}
COL = {'mx17_2': '#56B4E9', 'mx17_3': '#0b3d91', 'mx17_4': '#999999', 'mx17_6': '#E69F00', 'mx17_7': '#CC79A7'}  # = paper_style
ORDER = ['mx17_3', 'mx17_2', 'mx17_6', 'mx17_7', 'mx17_4']      # det3 leads (paper order)
BAD_RUNS = ('mx17_det3_det4_overnight_6-23-26', 'mx17_det4_day_6-24-26')
#: (run, drift HV) of each chamber's mesh-HV scan
HV_RUNS = {'mx17_2': ('mx17_det2_det3_overnight_6-22-26', 1000), 'mx17_3': ('mx17_det3_saturday_scan_6-27-26', 1000),
           'mx17_6': ('mx17_det6_det7_hv_scan_6-26-26', 700), 'mx17_7': ('mx17_det6_det7_hv_scan_6-26-26', 700)}
GAP_CM = {'mx17_3': 2.88}


def load():
    m = pd.read_csv(os.path.join(R, 'rows_matched.csv'))
    m['T'] = m[['T_x', 'T_y']].mean(axis=1)
    tg = {d: float(m.loc[m.key == k, 'T'].iloc[0]) for d, k in GOLD.items() if (m.key == k).any()}
    m['v'] = [V_S4.get(d, np.nan) * tg.get(d, np.nan) / t if t > 0 else np.nan for d, t in zip(m.det, m['T'])]
    for k in ('sigma_xy', 'sigma_evt', 'sigma_common'):
        m[k + '_mm'] = m[k] * m['v'] / 1e3
    m['trig'] = np.sign(m.trig_cov_ns2) * np.sqrt(m.trig_cov_ns2.abs())
    return m


def jload(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def fig_paper(m, amp, sps):
    """The appendix figure timing_all (all five chambers) in the house style (papers/paper_style.py): det3 leads (dark blue, heavy, on top), det4 grey;
    one chamber legend above the panels, in-panel legends only for what the line style means."""
    import sys
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'papers'))
    import paper_style as ps
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ps.use()
    fig, ax = ps.fig_for('full', height=2.6, nrows=1, ncols=3, gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    csv_rows = []
    # (a) HV scans, M3-matched
    a = ax[0]
    for d in ORDER:
        if d not in HV_RUNS:
            continue
        run, dv = HV_RUNS[d]
        s = m[(m.det == d) & (m.run == run) & (m.n >= 150) & (m.cls != 'skeptical') & (m.drift_V == dv)]
        s = s.groupby('resist_V').agg(evt=('sigma_evt', 'median'), xy=('sigma_xy', 'median')).reset_index()
        a.plot(s.resist_V, s.evt, **ps.det(NAME[d], view='x', ms=3.5 if d == 'mx17_3' else 2.8))
        a.plot(s.resist_V, s.xy, **ps.det(NAME[d], view='y', ms=3.5 if d == 'mx17_3' else 2.8))
        csv_rows += [dict(panel='a', det=NAME[d], x=r.resist_V, sigma_evt=r.evt, sigma_xy=r.xy) for r in s.itertuples()]
    a.set_xlabel('resist HV [V]'); a.set_ylabel(r'$\sigma$ (s68) [ns]'); a.set_ylim(0, 140)
    # (b) amplitude curve: bench + SPS
    a = ax[1]
    for d in ORDER:
        v = amp.get('bench', {}).get(d)
        if not v:
            continue
        b = v['bins']
        a.plot([x['amp'] for x in b], [x['sigma_xy'] for x in b], **ps.det(NAME[d], ms=3 if d == 'mx17_3' else 2.4))
        csv_rows += [dict(panel='b', det=NAME[d], setup='bench', x=x['amp'], sigma_xy=x['sigma_xy']) for x in b]
    sps_h = []
    for tag, ls, lab in (('zs', ':', 'ZS'), ('raw', '-.', 'RAW run71')):
        if tag in amp.get('sps', {}):
            b = amp['sps'][tag]['bins']
            a.plot([x['amp'] for x in b], [x['sigma_xy'] for x in b], **ps.det('det4', ls=ls, mfc='white', ms=3.5))
            sps_h.append(Line2D([], [], color=ps.color('det4'), marker='v', mfc='white', ls=ls, label=f'det4 SPS {lab}'))
            csv_rows += [dict(panel='b', det='det4', setup=f'SPS {lab}', x=x['amp'], sigma_xy=x['sigma_xy']) for x in b]
    a.set_xscale('log'); a.set_yscale('log'); a.set_ylim(8, 700)
    a.set_xlabel('early amplitude, X + Y [ADC]'); a.set_ylabel(r'$(X-Y)/\sqrt{2}$ (s68) [ns]')
    a.legend(handles=sps_h, loc='lower left')
    # (c) golden-row budget, det3 first
    a = ax[2]
    ang = jload('angle.json')
    dets = [d for d in ORDER]
    keys = [QUOTED.get(d, GOLD[d]) for d in dets]
    xs = np.arange(len(keys))
    vals = [ang.get(k, {}).get('all', {}) for k in keys]
    xy = [v.get('sigma_xy', np.nan) for v in vals]; cm = [v.get('sigma_common', np.nan) for v in vals]
    ev = [v.get('sigma_evt', np.nan) for v in vals]
    tr = [np.sqrt(max(v.get('trig_cov_ns2', 0), 0)) for v in vals]
    cols = [ps.color(NAME[d]) for d in dets]
    a.bar(xs - 0.27, ev, 0.27, color=cols)
    a.bar(xs, cm, 0.27, color=cols, alpha=0.55)
    a.bar(xs + 0.27, xy, 0.27, color=cols, alpha=0.25, hatch='////', edgecolor=cols, lw=0)
    for x, t in zip(xs, tr):
        if t > 10:
            a.plot([x - 0.42, x + 0.42], [t, t], '-', color=ps.INK, lw=1.4)
            a.text(x, t + 3, 'trigger', ha='center', va='bottom', fontsize=ps.BASE_PT * 0.8)
    csv_rows += [dict(panel='c', det=NAME[d], row=k, sigma_evt=e, sigma_common=c, sigma_xy=q, trigger=t)
                 for d, k, e, c, q, t in zip(dets, keys, ev, cm, xy, tr)]
    a.set_xticks(xs); a.set_xticklabels([NAME[d] for d in dets]); a.set_ylabel(r'$\sigma$ (s68) [ns]')
    a.grid(axis='x', visible=False)
    a.set_ylim(0, max(np.nanmax(ev), max(tr)) * 1.45)
    g = '#777777'
    a.legend(handles=[Patch(fc=g, label='event time'), Patch(fc=g, alpha=0.55, label='common to X, Y'),
                      Patch(fc=g, alpha=0.25, hatch='////', ec=g, lw=0, label=r'$(X-Y)/\sqrt{2}$')],
             loc='upper left', ncol=1)
    ps.panel_labels(ax, y=1.05)
    # one legend above: the chambers, then what the line styles of (a) mean
    h = ps.det_handles([NAME[d] for d in ORDER]) + [
        Line2D([], [], color=ps.INK, marker='o', ls='-', label='event time'),
        Line2D([], [], color=ps.INK, marker='o', mfc='white', ls='--', label=r'$(X-Y)/\sqrt{2}$')]
    ps.legend_above(fig, h, ncol=7, columnspacing=0.9, handlelength=2.0)
    ps.save(fig, os.path.join(F, 'timing_all'), data=pd.DataFrame(csv_rows))


def fig_main(m, amp):
    """The main-text figure `timing`: chamber A only (Dylan 2026-11-03; all chambers are timing_all, appendix).
    (a) event time and the uncorrelated part (X-Y)/sqrt2 against the resist HV of the 6-27 scan; (b) the uncorrelated
    part against the early signal amplitude: every chamber-A row pooled (filled) and Runs 1+2 alone, one HV, binned in
    the per-track amplitude (open; walk_snr.py). 2026-11-03: the drift-field scan that was panel (c) is now its own figure,
    timing_driftscan (fig_driftscan), for the appendix on the timing details (Dylan)."""
    import sys
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'papers'))
    import paper_style as ps
    from matplotlib.lines import Line2D
    ps.use()
    ws = jload('walk_snr.json')
    fig, ax = ps.fig_for('full', height=2.4, nrows=1, ncols=2)
    d = 'mx17_3'; rows = []
    run, dv = HV_RUNS[d]
    s = m[(m.det == d) & (m.run == run) & (m.n >= 150) & (m.cls != 'skeptical') & (m.drift_V == dv)]
    s = s.groupby('resist_V').agg(evt=('sigma_evt', 'median'), xy=('sigma_xy', 'median')).reset_index()
    a = ax[0]
    a.plot(s.resist_V, s.evt, **ps.det('det3', view='x', ms=3.5))
    a.plot(s.resist_V, s.xy, **ps.det('det3', view='y', ms=3.5))
    rows += [dict(panel='a', x=r.resist_V, sigma_evt=r.evt, sigma_xy=r.xy) for r in s.itertuples()]
    a.set_xlabel('resist HV [V]'); a.set_ylabel(r'$\sigma$ (s68) [ns]'); a.set_ylim(0, 120)
    a.legend(handles=[Line2D([], [], color=ps.color('det3'), marker='o', ls='-', label='event time'),
                      Line2D([], [], color=ps.color('det3'), marker='o', mfc='white', ls='--', label=r'$(X-Y)/\sqrt{2}$')],
             loc='upper right')
    a = ax[1]
    b = amp['bench'][d]['bins']
    a.plot([x['amp'] for x in b], [x['sigma_xy'] for x in b], **ps.det('det3', ms=3.5, label='all rows'))
    rows += [dict(panel='b', x=x['amp'], sigma_xy=x['sigma_xy']) for x in b]
    if ws.get('fixed_hv'):
        f = ws['fixed_hv']
        a.errorbar([x['amp'] for x in f], [x['sigma_xy'] for x in f], yerr=[x['sigma_xy_err'] for x in f], color=ps.ACCENT,
                   marker='s', ms=3.3, mfc='white', mew=0.9, ls='none', elinewidth=0.8, zorder=6, label='Runs 1+2 (490 V)')
        rows += [dict(panel='b_fixed_hv', x=x['amp'], sigma_xy=x['sigma_xy'], err=x['sigma_xy_err']) for x in f]
    a.set_xscale('log'); a.set_yscale('log'); a.set_ylim(10, 300)
    a.set_yticks([10, 20, 50, 100, 200]); a.set_yticklabels(['10', '20', '50', '100', '200']); a.minorticks_off()
    a.set_xlabel('early amplitude, X + Y [ADC]'); a.set_ylabel(r'$(X-Y)/\sqrt{2}$ (s68) [ns]')
    a.legend(loc='lower left', handlelength=1.2)
    ps.panel_labels(ax)
    ps.save(fig, os.path.join(F, 'timing'), data=pd.DataFrame(rows))

def fig_driftscan(ws):
    """Appendix figure `timing_driftscan` (was panel (c) of `timing`): chamber A's drift-field scan at 490 V (M3-matched,
    rows_matched.csv): event time and (X-Y)/sqrt2 in ns, and the event time in mm of drift (x the row's timing v) on the
    right axis. 300 V has no mm (the column overruns the readout window, no timing v)."""
    import sys
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'papers'))
    import paper_style as ps
    from matplotlib.lines import Line2D
    ps.use()
    fig, a = ps.fig_for(4.3, height=2.5)
    ax = [a]; rows = []
    ds = [r for r in ws.get('drift_scan', []) if r['n'] >= 150]
    if ds:
        E = np.array([r['E_Vcm'] for r in ds])
        a.errorbar(E, [r['sigma_evt'] for r in ds], yerr=[r['sigma_evt_err'] for r in ds], elinewidth=0.8,
                   **ps.det('det3', view='x', ms=3.5))
        a.errorbar(E, [r['sigma_xy'] for r in ds], yerr=[r['sigma_xy_err'] for r in ds], elinewidth=0.8,
                   **ps.det('det3', view='y', ms=3.5))
        a.set_ylim(0, 120); a.set_xlim(80, 410)
        a.set_xlabel('drift field [V/cm]'); a.set_ylabel(r'$\sigma$ (s68) [ns]')
        t = a.twinx(); t.grid(False); t.spines['right'].set_visible(True)
        k = np.isfinite([r['evt_mm'] for r in ds])
        t.plot(E[k], np.array([r['evt_mm'] for r in ds])[k], color=ps.MODEL, marker='D', ms=3.0, mfc='white', ls=':',
               lw=0.9)
        t.set_ylim(0, 2.4); t.set_ylabel('event time [mm of drift]', labelpad=2)
        a.legend(handles=[Line2D([], [], color=ps.color('det3'), marker='o', ls='-', label='event time [ns]'),
                          Line2D([], [], color=ps.color('det3'), marker='o', mfc='white', ls='--', label=r'$(X-Y)/\sqrt{2}$ [ns]'),
                          Line2D([], [], color=ps.MODEL, marker='D', mfc='white', ls=':', label='event time [mm, right]')],
                 loc='upper right')
        rows += [dict(panel='driftscan', x=r['E_Vcm'], drift_V=r['drift_V'], sigma_evt=r['sigma_evt'], sigma_xy=r['sigma_xy'],
                      evt_mm=r['evt_mm'], v=r['v']) for r in ds]
    ps.save(fig, os.path.join(F, 'timing_driftscan'), data=pd.DataFrame(rows))


def fig_angle():
    ang = jload('angle.json')
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, v in ang.items():
        d = next((dd for dd, kk in {**GOLD, **QUOTED}.items() if kk == k), None)
        b = [x for x in v['bins'] if x['n'] >= 60]
        c = [(x['lo'] + x['hi']) / 2 for x in b]
        ls = '-' if k in QUOTED.values() else '--'
        ax[0].plot(c, [x['median'] - b[0]['median'] for x in b], 'o' + ls, color=COL.get(d, 'k'), label=k[:40])
        ax[1].plot(c, [x['sigma_evt'] for x in b], 'o' + ls, color=COL.get(d, 'k'))
    ax[0].set_ylabel('median event time − head-on bin [ns]'); ax[1].set_ylabel('σ_evt (s68) [ns]')
    for a in ax:
        a.set_xlabel('larger |tan θ| of the two views'); a.grid(alpha=0.3)
    ax[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(F, 'angle.png'), dpi=140); plt.close(fig)


def fig_drift(m):
    s = m[(m.det == 'mx17_3') & (m.run == 'mx17_det3_saturday_scan_6-27-26') & (m.resist_V == 490) & (m.n >= 150)]
    s = s.assign(E=s.drift_V / GAP_CM['mx17_3']).groupby('E').agg(t=('sigma_evt', 'median'), mm=('sigma_evt_mm', 'median'),
                                                                  xy=('sigma_xy', 'median')).reset_index()
    fig, a = plt.subplots(figsize=(5.5, 3.8))
    a.plot(s.E, s.t, 'o-', color=COL['mx17_3'], label='event time [ns]')
    a.plot(s.E, s.xy, 's--', color=COL['mx17_3'], mfc='none', label='X−Y / √2 [ns]')
    b = a.twinx(); b.plot(s.E, s.mm, 'd:', color='k', label='event time [mm of drift]'); b.set_ylabel('[mm]')
    a.set_xlabel('drift field [V/cm]'); a.set_ylabel('σ (s68) [ns]'); a.grid(alpha=0.3)
    a.legend(loc='upper right', fontsize=7); b.legend(loc='center right', fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(F, 'drift.png'), dpi=140); plt.close(fig)
    return s


def walk_section(ws):
    """Dylan 2026-11-03: is the HV gain amplitude walk (correctable) or signal-to-noise (not)? walk_snr.py's numbers."""
    if not ws:
        return ''
    w = ws['walk']
    fx = pd.DataFrame(ws['fixed_hv']); fx['hv_scan_ratio'] = [r['ratio'] for r in ws['fixed_over_hvscan']]
    ds = pd.DataFrame(ws['drift_scan'])
    return ('<h2>Amplitude walk or signal-to-noise? (walk_snr.py)</h2>'
            '<p><b>Signal-to-noise, not walk.</b> At one HV (Runs 1+2, 490 V), binned in the per-track early amplitude, '
            f'(X−Y)/√2 falls from {fx.sigma_xy.iloc[0]:.1f} to {fx.sigma_xy.iloc[-1]:.1f} ns and sits on the HV-scan curve '
            f'at the same amplitude (ratio {fx.hv_scan_ratio.min():.2f}–{fx.hv_scan_ratio.max():.2f}): the HV improvement is '
            'the signal size. The edge estimator fits a scaled template step, so a pure amplitude change does not move it: '
            f'the median event time moves by {w["med_evt_span_ns"]:.0f} ns over the whole amplitude range (most of it in the '
            f'top bin) and t<sub>X</sub>−t<sub>Y</sub> by {w["med_xy_span_ns"]:.1f} ns. Subtracting that walk curve changes '
            f'σ_evt {w["sigma_evt"]:.1f} → {w["sigma_evt_walk_corrected"]:.1f} ns and (X−Y)/√2 {w["sigma_xy"]:.1f} → '
            f'{w["sigma_xy_walk_corrected"]:.1f} ns. What falls with amplitude is the width (noise over the slope of the '
            'rising edge), which no amplitude correction removes. The event time is flat in amplitude at fixed HV '
            f'({", ".join(f"{q["sigma_evt"]:.0f}" for q in w["quartiles"])} ns by amplitude quartile): its common part is not '
            'set by the signal size.</p>'
            + table(fx, ['amp', 'n', 'sigma_xy', 'sigma_xy_err', 'sigma_evt', 'med_evt', 'hv_scan_ratio'],
                    dict(amp='{:.0f}', sigma_xy='{:.1f}', sigma_xy_err='{:.1f}', sigma_evt='{:.1f}', med_evt='{:+.1f}',
                         hv_scan_ratio='{:.2f}'))
            + '<p><b>Drift field (490 V, M3-matched).</b> In ns the widths fall with the field; in drift distance the event '
            'time and its common part are constant (the common part is a depth, ~1.2–1.4 mm: where the first electrons '
            'are made, common to both views), and the uncorrelated part grows slowly (it follows the early amplitude, '
            '∝ v).</p>'
            + table(ds, ['drift_V', 'E_Vcm', 'n', 'v', 'sigma_evt', 'sigma_xy', 'sigma_common', 'evt_mm', 'xy_mm', 'common_mm'],
                    dict(E_Vcm='{:.0f}', v='{:.1f}', sigma_evt='{:.1f}', sigma_xy='{:.1f}', sigma_common='{:.1f}',
                         evt_mm='{:.2f}', xy_mm='{:.2f}', common_mm='{:.2f}'))
            + f'<p>{html.escape(ws.get("note", ""))}</p>')


def table(df, cols, fmt=None):
    fmt = fmt or {}
    h = ['<table><tr>' + ''.join(f'<th>{html.escape(c)}</th>' for c in cols) + '</tr>']
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f'<td>{fmt[c].format(v) if c in fmt and pd.notna(v) else html.escape(str(v))}</td>')
        h.append('<tr>' + ''.join(cells) + '</tr>')
    return '\n'.join(h + ['</table>'])


def main():
    m = load()
    amp = jload('amplitude.json'); sps = jload('sps_timing.json'); ang = jload('angle.json'); t0p = jload('t0_position.json')
    fig_paper(m, amp, sps); fig_main(m, amp); fig_driftscan(jload('walk_snr.json')); fig_angle(); ds = fig_drift(m)
    rows = pd.read_csv(os.path.join(R, 'rows.csv'))
    g = []
    for d in ('mx17_2', 'mx17_3', 'mx17_7', 'mx17_6', 'mx17_4'):
        for k in ([QUOTED[d]] if d in QUOTED else []) + [GOLD[d]]:
            a = ang.get(k, {}).get('all')
            if not a:
                continue
            v = float(m.loc[m.key == k, 'v'].iloc[0]) if (m.key == k).any() else np.nan
            g.append(dict(chamber=NAME[d], row=k, n=a['n'], sigma_xy=a['sigma_xy'], sigma_evt=a['sigma_evt'],
                          sigma_common=a['sigma_common'], trigger=np.sign(a['trig_cov_ns2']) * np.sqrt(abs(a['trig_cov_ns2'])),
                          v=v, evt_mm=a['sigma_evt'] * v / 1e3, xy_mm=a['sigma_xy'] * v / 1e3))
    g = pd.DataFrame(g)
    byrun = m[m.n >= 150].groupby(['run', 'det']).agg(rows=('key', 'size'), n=('n', 'sum'), xy=('sigma_xy', 'median'),
                                                      evt=('sigma_evt', 'median'), trig=('trig', 'median')).reset_index()
    sps_df = pd.DataFrame([dict(row=k, n=v.get('n'), xy_edge=v.get('edge25', {}).get('sigma_xy'),
                                abs_edge=v.get('edge25', {}).get('abs_raw'), abs_ftst=v.get('edge25', {}).get('abs_plus'),
                                xy_cf20=v.get('cf20', {}).get('sigma_xy'), q=str(v.get('q_median')))
                           for k, v in sps.items() if 'edge25' in v])
    est = []
    for k in (QUOTED['mx17_3'], GOLD['mx17_3'], QUOTED['mx17_2']):
        p = rows[rows.key == k]
        if len(p):
            p = p.iloc[0]
            est.append(dict(row=k, edge25=p.sigma_evt, edge15=p.edge15_sigma_evt, edge40=p.edge40_sigma_evt, cf10=p.cf10_sigma_evt,
                            fixT=p.fixT_sigma_evt, box=p.box_sigma_evt))
    est = pd.DataFrame(est)
    f1 = '{:.1f}'; f2 = '{:.2f}'; f0 = '{:.0f}'
    H = ['<!DOCTYPE html><html><head><meta charset="utf-8"><title>Paper I §7: timing on waveforms</title>',
         '<style>body{font-family:sans-serif;max-width:1150px;margin:auto;line-height:1.45}table{border-collapse:collapse;'
         'font-size:13px;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 6px;text-align:right}th{background:#eee}'
         'img{max-width:100%}.v{background:#eef6ee;padding:10px;border-left:4px solid #2e7d32}</style></head><body>',
         '<h1>Paper I §7 — chamber timing from the waveforms (pass 1)</h1>',
         '<div class="v"><b>Verdict.</b> On a cosmic muon the chamber gives its own event time to <b>σ ≈ 40–46 ns</b> '
         '(det2, det3, det7; 1.4–1.7 mm of drift), from the start of the charge column, at any track angle. '
         'Of that, ≈ 22–28 ns per view is not shared by the two views (noise, sampling) and falls with the signal size '
         '(roughly as amplitude<sup>−0.7</sup>); '
         '≈ 35–40 ns is common to X and Y — the arrival of the first electrons next to the mesh — and sets the floor. '
         'The trigger adds nothing measurable on those runs (cov(start, end) ≈ 0), but ≈ 40–60 ns on the 6-23 / 6-24 runs (det4). '
         'At SPS (det4, CF₄, 14 µm/ns) the same estimator gives 160–200 ns per view (ZS): the low-amplitude end of the bench curve, '
         'not a measurement of the intrinsic floor.</div>',
         '<h2>The paper figure (chamber A)</h2><img src="figures/timing.png">','<h2>Appendix figure, all chambers</h2><img src="figures/timing_all.png">',
         '<p>(a) mesh-HV scans; (b) the uncorrelated part against the early amplitude, bench and SPS; (c) the golden rows split '
         'into event time, the part common to both views and X−Y/√2, with the trigger term where cov(start, end) &gt; 0.</p>',
         '<h2>Golden rows (M3-matched single tracks)</h2>',
         table(g, ['chamber', 'row', 'n', 'sigma_evt', 'sigma_xy', 'sigma_common', 'trigger', 'v', 'evt_mm', 'xy_mm'],
               dict(sigma_evt=f1, sigma_xy=f1, sigma_common=f1, trigger=f0, v=f1, evt_mm=f2, xy_mm=f2)),
         '<p>ns unless marked; trigger = sign × √|cov(start, end)| (≈ 0 means no common shift of both column ends); '
         'v = timing v of the row [µm/ns].</p>',
         '<h2>Per run (M3-matched, rows with ≥ 150 tracks)</h2>',
         table(byrun, ['run', 'det', 'rows', 'n', 'evt', 'xy', 'trig'], dict(evt=f1, xy=f1, trig=f0)),
         '<h2>Angle</h2><img src="figures/angle.png"><p>The column start does not move with the track angle (median shift ≤ 8 ns '
         'from head-on to |tan θ| = 0.5 on det2/det3), and its width is the same head-on as inclined: the estimator is a time, '
         'also at normal incidence. The wft fit\'s t0 is not (it is pinned to the trigger in the pass, σ = 0.05 ns).</p>',
         '<h2>Drift field (det3, 490 V)</h2><img src="figures/drift.png">',
         table(ds, ['E', 't', 'xy', 'mm'], dict(E=f0, t=f1, xy=f1, mm=f2)),
         walk_section(jload('walk_snr.json')),
         '<h2>Estimator choice (σ_evt, all clean triggers)</h2>',
         table(est, ['row', 'edge25', 'edge15', 'edge40', 'cf10', 'fixT', 'box'],
               dict(edge25=f1, edge15=f1, edge40=f1, cf10=f1, fixT=f1, box=f1)),
         '<p>box = colsum\'s free (t0, T) fit (t0 and T trade off); fixT = T fixed to the row median; cf10 = 10 % constant '
         'fraction; edgeNN = a template step fitted to the rising edge up to NN % of the maximum. A noise-set cut '
         '(dev_edgeabs.py) gives 37–48 ns with 1.5 % failures: the common part stays at 33–44 ns whatever the cut.</p>',
         '<h2>SPS (det4 at H4)</h2>',
         table(sps_df, ['row', 'n', 'xy_edge', 'abs_edge', 'abs_ftst', 'xy_cf20', 'q'],
               dict(xy_edge=f0, abs_edge=f0, abs_ftst=f0, xy_cf20=f0)),
         '<h2>Does the trigger error reach the positions? (det4)</h2><pre>' + html.escape(json.dumps(t0p, indent=1)[:3000]) + '</pre>',
         '<h2>What this does not rule out</h2><ul>'
         '<li>The common 35–40 ns could hold a trigger part below the method\'s reach: cov(start, end) carries an estimator '
         'bias of about −60 ns² (det3), so a trigger jitter ≲ 10 ns is not excluded.</li>'
         '<li>The physical origin of the common part (first-electron arrival: primary-cluster positions, gain of the first '
         'electrons, the threshold set relative to the column maximum) is inferred, not simulated.</li>'
         '<li>det6\'s uncorrelated part is 2× the others at the same HV; not chased (det6 depths already carry the v '
         'limitation).</li>'
         '<li>The 6-23 / 6-24 trigger term is measured on the column; its cost on det4\'s positions is not established '
         '(the signed regression matches, the angle-specific widening is not seen).</li>'
         '<li>SPS: only an upper bound; the intrinsic floor needs a higher-gain chamber or RAW data at the bench S/N.</li></ul>',
         '</body></html>']
    open(os.path.join(HERE, 'report.html'), 'w').write('\n'.join(H))
    g.to_csv(os.path.join(R, 'golden.csv'), index=False)
    print('report.html + figures written;', len(g), 'golden rows')


if __name__ == '__main__':
    main()
