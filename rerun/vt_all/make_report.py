#!/usr/bin/env python3
"""make_report.py [data dir] -- tables (results/*.csv, results/summary.json), figures and report.html of the v(t) test
on every data set (VT_PLAN_2026-10-30), from the JSONs fetch.sh pulls (vt_bench.py, vt_sps.py, vt_ntof.py).  No data
access beyond those; re-run after the DAG to refresh numbers, figures and verdicts together."""
import glob
import html
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES, FIG = os.path.join(HERE, 'results'), os.path.join(HERE, 'figures')
for _d in (RES, FIG):
    os.makedirs(_d, exist_ok=True)
DATA = sys.argv[1] if len(sys.argv) > 1 else (os.environ.get('VT_ALL_DATA')
                                              or os.path.join(os.environ.get('WORK_DATA') or os.path.expanduser('~/x17'), 'vt_all'))
sys.path.insert(0, HERE)
from vt_bench import GATE_DV, GATE_Z  # noqa: E402
SPS_GATE = 0.02            # SPS: first-to-last window change in both views, same sign
NAME = {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4', 'mx17_6': 'det6', 'mx17_7': 'det7'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e3', 'grid.linewidth': 0.6})
PAL = ['#2a6fdb', '#eda100', '#c0392b', '#1e9e6a', '#6250d6', '#8c564b', '#e377c2', '#7f7f7f', '#17becf', '#bcbd22']


def short(sub):
    return sub.replace('resist_', 'r').replace('_drift_', ' d').replace('det6_', '').replace('det7_', '')


# ---------------------------------------------------------------- bench
B = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(DATA, 'bench', '*.json'))) if not f.endswith('vt_map.json')]
rows = []
for r in B:
    f = r.get('fit') or {}
    rows.append(dict(key=r['key'], det=NAME.get(r['det'], r['det']), run=r['run'], subrun=r['subrun'], cls=r['cls'],
                     t_start_h=r['t_start'], duration_h=r['duration_h'], n_events=r['n_events'], n_kw=r.get('n_kw'),
                     n_blocks=r.get('n_blocks', 0), v_drift=r['v_drift'],
                     slope_pct_h=100 * f['slope_per_h'] if f else np.nan,
                     slope_err_pct_h=100 * f['slope_err'] * f['inflate'] if f else np.nan,
                     dv_pct=100 * r['dv_total'] if 'dv_total' in r else np.nan,
                     dv_err_pct=100 * r['dv_err'] if 'dv_err' in r else np.nan, z=f.get('z', np.nan),
                     chi2_line=f.get('chi2', np.nan), dof_line=f.get('dof', np.nan),
                     chi2_const=f.get('chi2_const', np.nan), dof_const=f.get('dof_const', np.nan),
                     needs_correction=bool(r.get('needs_correction', False)), verdict=r['verdict']))
BT = pd.DataFrame(rows).sort_values(['det', 't_start_h']) if rows else pd.DataFrame()
if len(BT):
    BT.to_csv(os.path.join(RES, 'bench_vt.csv'), index=False, float_format='%.4g')

# ---------------------------------------------------------------- SPS
S = {}
for f in sorted(glob.glob(os.path.join(DATA, 'sps', '*', 'vt_sps.json'))):
    S.update(json.load(open(f)))
srows = []
for key, r in S.items():
    bl = [b for b in r['blocks'] if b.get('n')]
    d = dict(key=key, field_Vcm=r['field_Vcm'], gas=r['gas'], raw=r['raw'], n_blocks=len(bl),
             minutes=(r['plateau'][1] - r['plateau'][0]) / 60)
    for v in 'xy':
        T = [b.get(v, {}).get('T_box') for b in bl]
        ok = [b.get(v, {}).get('defined', False) for b in bl]
        d[f'T_{v}'] = ' '.join(f'{t:.0f}' if t else '-' for t in T)
        d[f'defined_{v}'] = all(ok) and len(ok) > 1
        # v_rel(last) / v_rel(first) - 1 = T_first / T_last - 1
        d[f'dv_{v}_pct'] = 100 * (T[0] / T[-1] - 1) if len(T) > 1 and T[0] and T[-1] else np.nan
    dvs = [d[f'dv_{v}_pct'] for v in 'xy' if d[f'defined_{v}'] and np.isfinite(d[f'dv_{v}_pct'])]
    d['dv_pct'] = float(np.mean(dvs)) if dvs else np.nan
    d['dv_sys_pct'] = float(abs(dvs[0] - dvs[1]) / 2) if len(dvs) == 2 else np.nan
    d['needs_correction'] = bool(len(dvs) == 2 and np.sign(dvs[0]) == np.sign(dvs[1])
                                 and min(abs(x) for x in dvs) >= 100 * SPS_GATE)
    d['verdict'] = ('correct' if d['needs_correction'] else 'flat' if len(dvs) == 2
                    else 'one view defined' if dvs else 'column not defined')
    srows.append(d)
ST = pd.DataFrame(srows)
if len(ST):
    ST.to_csv(os.path.join(RES, 'sps_vt.csv'), index=False, float_format='%.4g')

# ---------------------------------------------------------------- n_TOF
N = []
for f in sorted(glob.glob(os.path.join(DATA, 'ntof', '*.json'))):
    N += json.load(open(f))
N = list({r['run']: r for r in N}.values())
nrows = []
for r in N:
    for arm, a in r['arms'].items():
        f = a['fit'] or {}
        nrows.append(dict(run=r['run'], arm=arm, n_subruns=r['n_subruns'], hours=r['t_end'] - r['t_start'],
                          k_run=a['k_run'], k_run_err=a['k_run_err'],
                          slope_pct_h=100 * f.get('slope_per_h', np.nan), slope_err_pct_h=100 * f.get('slope_err', np.nan),
                          dv_pct=100 * f.get('dv_total', np.nan), rms_sub_pct=100 * f.get('rms_sub', np.nan),
                          err_sub_pct=100 * f.get('median_err_sub', np.nan), chi2_const=f.get('chi2_const', np.nan),
                          dof_const=f.get('dof_const', np.nan), half_rms_pct=100 * a['half_diff']['rms'],
                          half_err_pct=100 * a['half_diff']['med_err']))
NT = pd.DataFrame(nrows).sort_values(['run', 'arm'], key=lambda s: s.str.replace('run_', '').str.zfill(4)
                                     if s.name == 'run' else s) if nrows else pd.DataFrame()
if len(NT):
    NT.to_csv(os.path.join(RES, 'ntof_vt.csv'), index=False, float_format='%.4g')

# ---------------------------------------------------------------- figures
figs = []
if len(B):
    dets = sorted({r['det'] for r in B})
    fig, axs = plt.subplots(len(dets), 1, figsize=(10, 2.6 * len(dets)), squeeze=False)
    for ax, det in zip(axs[:, 0], dets):
        runs = sorted({r['run'] for r in B if r['det'] == det})
        for i, run in enumerate(runs):
            lab = run.replace('mx17_', '')
            for r in B:
                if r['det'] != det or r['run'] != run or not r['blocks']:
                    continue
                t = np.array([b['t'] for b in r['blocks']])
                v = np.array([np.nanmean([b[p]['kw'] if b[p]['kw'] is not None else np.nan for p in 'xy'])
                              for b in r['blocks']], float) * r['v_drift']
                ax.plot(t, v, '.-', color=PAL[i % len(PAL)], ms=3, lw=0.8, label=lab)
                lab = None
        ax.set_title(f'{NAME.get(det, det)}: ladder v per time block (v_drift x kw_block, mean of the views)',
                     fontsize=9, loc='left')
        ax.set_ylabel('v [µm/ns]')
        ax.legend(fontsize=7, ncol=3, frameon=False)
    axs[-1, 0].set_xlabel('hours since 2026-06-01 00:00')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'bench_v_blocks.png'), dpi=110); plt.close(fig)
    figs.append(('bench_v_blocks.png', 'Bench: the M3-ladder drift velocity per time block, every pass-1 row, per '
                 'chamber against the clock. Each row is a connected line; colours are the runs (nights).'))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, det in enumerate(sorted(BT.det.unique())):
        q = BT[(BT.det == det) & np.isfinite(BT.dv_pct)]
        ax.errorbar(q.duration_h, q.dv_pct, q.dv_err_pct, fmt='o', ms=4, color=PAL[i], label=det, alpha=0.8)
    ax.axhspan(-100 * GATE_DV, 100 * GATE_DV, color='#e6e6e3', alpha=0.6, lw=0)
    ax.set_xscale('log'); ax.set_xlabel('row duration [h]'); ax.set_ylabel('v change over the row [%]')
    ax.set_title(f'Bench: within-row change (line fit); gate |Δ| ≥ {100 * GATE_DV:.0f} % and z ≥ {GATE_Z:.0f}',
                 fontsize=9, loc='left')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'bench_dv.png'), dpi=110); plt.close(fig)
    figs.append(('bench_dv.png', 'Bench: v change from the start to the end of each row against its duration '
                 '(error inflated by the line χ²/dof where above 1). The grey band is the 2 % gate.'))
if S:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (key, r) in enumerate(S.items()):
        for v, mk in (('x', 'o'), ('y', 's')):
            bl = [b for b in r['blocks'] if b.get(v) and b[v].get('T_box') and b[v].get('defined')]
            if len(bl) > 1:
                t = np.array([b['t'] for b in bl]); T = np.array([b[v]['T_box'] for b in bl])
                ax.plot((t - t[0]) / 60, T[0] / T, mk + '-', color=PAL[i % len(PAL)], ms=4, lw=0.8,
                        label=key if v == 'x' else None)
    ax.set_xlabel('minutes into the row'); ax.set_ylabel('v / v(first window)  (= T_first / T)')
    ax.set_title('SPS: column-duration v per wall-clock window (o = x, ■ = y; defined windows only)', fontsize=9,
                 loc='left')
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'sps_v_blocks.png'), dpi=110); plt.close(fig)
    figs.append(('sps_v_blocks.png', 'SPS: v relative to the first window of each row, from the stacked column '
                 'duration, both views.'))
if N:
    arms = sorted({a for r in N for a in r['arms']})
    fig, axs = plt.subplots(len(arms), 1, figsize=(10, 2.8 * len(arms)), squeeze=False)
    for ax, arm in zip(axs[:, 0], arms):
        for i, r in enumerate(sorted(N, key=lambda r: r['t_start'])):
            a = r['arms'].get(arm)
            if not a:
                continue
            bl = [b for b in a['blocks'] if b['half'] is None and b['k']]
            vp = a.get('v_prior', 1.0)
            t = np.array([b['t'] for b in bl]); vv = np.array([vp / b['k'] for b in bl])
            e = np.array([vp * b['k_err'] / b['k'] ** 2 for b in bl])
            ax.errorbar(t, vv, e, fmt='.', ms=3, color=PAL[i % len(PAL)], lw=0.6)
        ax.set_ylabel('v_prior / k  [µm/ns]'); ax.set_title(f'n_TOF arm {arm}: in-situ drift velocity per sub-run '
                                                            '(the reconstruction prior / the angle scale k)',
                                                            fontsize=9, loc='left')
    axs[-1, 0].set_xlabel('hours since 2026-06-01 00:00')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'ntof_k.png'), dpi=110); plt.close(fig)
    figs.append(('ntof_k.png', 'n_TOF: the in-situ v = v_prior / k per sub-run (k = v_prior / v_true from the pointing-coincident '
                 'tracks), every run tested; colours alternate between runs.'))

# ---------------------------------------------------------------- summary + html
summ = dict(gate=dict(bench_dv=GATE_DV, bench_z=GATE_Z, sps_dv=SPS_GATE))
if len(BT):
    summ['bench'] = dict(n_rows=int(len(BT)), n_measured=int(np.isfinite(BT.dv_pct).sum()),
                         n_correct=int(BT.needs_correction.sum()),
                         n_significant_small=int((BT.verdict == 'significant, < 2 %').sum()),
                         n_too_few=int(BT.verdict.str.startswith('too few').sum()),
                         correct=BT[BT.needs_correction].key.tolist())
if len(ST):
    summ['sps'] = dict(n_rows=int(len(ST)), n_correct=int(ST.needs_correction.sum()),
                       correct=ST[ST.needs_correction].key.tolist())
if len(NT):
    summ['ntof'] = dict(n_runs=int(NT.run.nunique()), arms=sorted(NT.arm.unique().tolist()),
                        median_err_sub_pct=float(NT.err_sub_pct.median()),
                        median_half_err_pct=float(NT.half_err_pct.median()))
json.dump(summ, open(os.path.join(RES, 'summary.json'), 'w'), indent=1, default=float)


def table(df, cols, fmt):
    h = '<table><tr>' + ''.join(f'<th>{html.escape(c)}</th>' for c in cols) + '</tr>'
    for _, r in df.iterrows():
        cls = ' class="flag"' if r.get('needs_correction') else ''
        h += f'<tr{cls}>' + ''.join(f'<td>{fmt.get(c, "{}").format(r[c]) if not (isinstance(r[c], float) and np.isnan(r[c])) else "–"}</td>'
                                    for c in cols) + '</tr>'
    return h + '</table>'


css = ('body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222}'
       'table{border-collapse:collapse;font-size:12px;margin:1em 0}td,th{border-bottom:1px solid #ddd;padding:2px 6px;'
       'text-align:right}th{background:#f4f4f2}tr.flag td{background:#fff3d6}figure{margin:1.5em 0}'
       'figcaption{font-size:13px;color:#555}img{max-width:100%}.box{background:#f4f4f2;padding:.6em 1em}')
b = summ.get('bench', {}); s = summ.get('sps', {}); n = summ.get('ntof', {})
H = [f'<!doctype html><meta charset="utf-8"><title>v(t) in every data set</title><style>{css}</style>',
     '<h1>v as a function of time, in every data set</h1>',
     '<p>Plan: <code>rerun/VT_PLAN_2026-10-30.md</code>. Generated by <code>rerun/vt_all/make_report.py</code>.</p>',
     '<div class="box"><b>Verdict.</b> ',
     (f'Bench: {b.get("n_correct", 0)} of {b.get("n_measured", 0)} measured pass-1 rows (of {b.get("n_rows", 0)}) need a '
      f'v(t) correction (|Δv| ≥ {100 * GATE_DV:.0f} % over the row at ≥ {GATE_Z:.0f}σ); '
      f'{b.get("n_significant_small", 0)} drift significantly but by less than {100 * GATE_DV:.0f} %; '
      f'{b.get("n_too_few", 0)} are too short to measure. ' if b else ''),
     (f'SPS: {s.get("n_correct", 0)} of {s.get("n_rows", 0)} long rows need one. ' if s else ''),
     (f'n_TOF: feasible per sub-run on arms {", ".join(n.get("arms", []))}, at {n.get("median_err_sub_pct", 0):.2f} % '
      f'per sub-run (within a sub-run, halves at {n.get("median_half_err_pct", 0):.2f} %).' if n else ''), '</div>']
if len(BT):
    H += ['<h2>Bench rows that need a correction</h2>',
          table(BT[BT.needs_correction], ['det', 'run', 'subrun', 'duration_h', 'n_blocks', 'dv_pct', 'dv_err_pct', 'z',
                                          'chi2_line', 'dof_line'],
                {'duration_h': '{:.2f}', 'dv_pct': '{:+.1f}', 'dv_err_pct': '{:.1f}', 'z': '{:.1f}', 'chi2_line': '{:.0f}',
                 'dof_line': '{:.0f}'}),
          '<h2>All bench rows</h2>',
          table(BT, ['det', 'run', 'subrun', 'cls', 'duration_h', 'n_kw', 'n_blocks', 'slope_pct_h', 'slope_err_pct_h',
                     'dv_pct', 'dv_err_pct', 'z', 'verdict'],
                {'duration_h': '{:.2f}', 'slope_pct_h': '{:+.2f}', 'slope_err_pct_h': '{:.2f}', 'dv_pct': '{:+.1f}',
                 'dv_err_pct': '{:.1f}', 'z': '{:.1f}', 'n_kw': '{:.0f}'})]
if len(ST):
    H += ['<h2>SPS rows (column duration per wall-clock window)</h2>',
          table(ST, ['key', 'gas', 'field_Vcm', 'minutes', 'n_blocks', 'T_x', 'T_y', 'dv_x_pct', 'dv_y_pct', 'verdict'],
                {'field_Vcm': '{:.0f}', 'minutes': '{:.0f}', 'dv_x_pct': '{:+.1f}', 'dv_y_pct': '{:+.1f}'})]
if len(NT):
    H += ['<h2>n_TOF runs (in-situ k per sub-run)</h2>',
          table(NT, ['run', 'arm', 'n_subruns', 'hours', 'k_run', 'slope_pct_h', 'slope_err_pct_h', 'dv_pct', 'rms_sub_pct',
                     'err_sub_pct', 'chi2_const', 'dof_const', 'half_rms_pct', 'half_err_pct'],
                {'hours': '{:.1f}', 'k_run': '{:.3f}', 'slope_pct_h': '{:+.3f}', 'slope_err_pct_h': '{:.3f}',
                 'dv_pct': '{:+.1f}', 'rms_sub_pct': '{:.2f}', 'err_sub_pct': '{:.2f}', 'chi2_const': '{:.0f}',
                 'dof_const': '{:.0f}', 'half_rms_pct': '{:.2f}', 'half_err_pct': '{:.2f}'})]
for f, c in figs:
    H.append(f'<figure><img src="figures/{f}"><figcaption>{html.escape(c)}</figcaption></figure>')
H += ['<h2>What this does not rule out</h2><ul>',
      '<li>The bench line is an interpolator: a knee inside a row (det6 6-26) shows as χ²/dof &gt; 1, which inflates '
      'the error but is not modelled. Rows the gate flags with a large χ²/dof need the piecewise model of '
      '<code>rerun/vt_within_rows</code>.</li>',
      '<li>Rows below 2 × 150 kw-window tracks (most 20-min HV-scan rows) have no time measurement; they inherit the '
      'night curve of their chamber if one exists.</li>',
      '<li>SPS: the column duration is gain- and attachment-confounded (the beam gas carries O₂); the window ratio '
      'cancels the gap but not a change of attachment. Rows whose column does not end at an edge give no v.</li>',
      '<li>n_TOF: the per-track time is the acquisition file\'s, interpolated; k is measured on arms A and C only (the '
      'is2_v2 stage-3 tables carry no B/D tracks), and k mixes v with any change of the source geometry.</li></ul>']
open(os.path.join(HERE, 'report.html'), 'w').write('\n'.join(H))
print(json.dumps(summ, indent=1, default=float))
