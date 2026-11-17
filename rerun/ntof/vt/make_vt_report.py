#!/usr/bin/env python3
"""make_vt_report.py -- P7a campaign report: drift velocity v(t) per chamber from vt_table.py's parquets.

    make_vt_report.py [--vt DIR]        # DIR = vt_table.py --out (default $WORK_DATA/ntof_pass1/vt)

Writes DIR/report.html, DIR/figures/*.png and DIR/vt_summary.csv.  Re-run after vt_table.py.
"""
import argparse
import csv
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
WD = os.environ.get('WORK_DATA') or next((p for p in ('/media/ucla/work', os.path.expanduser('~/work_data'))
                                          if os.path.isdir(p)), '/media/ucla/work')
ARMS = ('A', 'C', 'D')
COL = {'beam': '#2a6fdb', 'cosmic': '#d9622b'}
# campaign markers (local time, as the registry's t_local); the 23 July clock change is a noise event, not a gas one
MARKS = [(datetime(2026, 7, 23, 16, 47), '23 Jul RdClk 6→4'),
         (None, '27 Jul access')]        # filled from the registry's access_epoch boundary


def access_time():
    R = [r for r in csv.DictReader(open(os.path.join(HERE, '..', 'ntof_registry.csv'))) if r['t_local']]
    post = [r['t_local'] for r in R if r['access_epoch'] == 'post_access_27jul']
    return datetime.strptime(min(post), '%Y-%m-%d %H:%M')


def wstats(v, e):
    ok = np.isfinite(v) & np.isfinite(e) & (e > 0)
    v, e = v[ok], e[ok]
    if not len(v):
        return dict(n=0)
    p16, p50, p84 = np.percentile(v, [16, 50, 84])
    pull = (v - p50) / e
    return dict(n=int(len(v)), median=p50, p16=p16, p84=p84, half68=(p84 - p16) / 2, med_err=float(np.median(e)),
                excess=float(np.sqrt(max((p84 - p16) ** 2 / 4 - np.median(e) ** 2, 0))),
                vmin=float(v.min()), vmax=float(v.max()), pull68=float(np.subtract(*np.percentile(pull, [84, 16])) / 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vt', default=f'{WD}/ntof_pass1/vt')
    a = ap.parse_args()
    FIG = f'{a.vt}/figures'
    os.makedirs(FIG, exist_ok=True)
    MARKS[1] = (access_time(), MARKS[1][1])
    meta = json.load(open(f'{a.vt}/vt.meta.json'))
    T = {arm: pd.read_parquet(f'{a.vt}/vt_{arm}.parquet') for arm in ARMS if os.path.exists(f'{a.vt}/vt_{arm}.parquet')}
    S = {arm: pd.read_parquet(f'{a.vt}/vt_{arm}_subrun.parquet') for arm in T}

    rows = []
    for arm in T:
        for lvl, D in (('acq', T[arm]), ('subrun', S[arm])):
            for trig, g in D.groupby('trigger'):
                st = wstats(g.v.to_numpy(float), g.v_err.to_numpy(float))
                rows.append(dict(arm=arm, level=lvl, trigger=trig, v_bundle=float(g.v_bundle.iloc[0]),
                                 n_rows=len(g), **st))
    SUM = pd.DataFrame(rows)
    SUM.to_csv(f'{a.vt}/vt_summary.csv', index=False)

    # beam vs cosmic offset: cosmic sub-runs against the beam sub-runs within +-12 h
    off = []
    for arm in T:
        P = S[arm].dropna(subset=['v'])
        b, c = P[P.trigger == 'beam'], P[P.trigger == 'cosmic']
        for _, r in c.iterrows():
            near = b[(b.t_start - r.t_start).abs() < pd.Timedelta(hours=12)]
            if len(near):
                off.append(dict(arm=arm, run=r.run, subrun=r.subrun, v_cos=r.v, v_beam=float(near.v.median()),
                                ratio=r.v / float(near.v.median())))
    OFF = pd.DataFrame(off)

    figs = []
    for arm in T:
        fig, ax = plt.subplots(figsize=(11, 3.6))
        for trig, g in T[arm].dropna(subset=['v']).groupby('trigger'):
            ax.errorbar(g.t_start, g.v, yerr=g.v_err, fmt='.', ms=3, lw=0.5, alpha=0.5, color=COL[trig],
                        label=f'{trig}, per acquisition ({len(g)})')
        for trig, g in S[arm].dropna(subset=['v']).groupby('trigger'):
            ax.errorbar(g.t_start, g.v, yerr=g.v_err, fmt='o', ms=4, mfc='white', lw=0.8, color=COL[trig],
                        label=f'{trig}, per sub-run ({len(g)})')
        vb = float(T[arm].v_bundle.iloc[0])
        ax.axhline(vb, color='k', lw=0.8, ls='--', label=f'frozen bundle v {vb:.2f}')
        for t, lab in MARKS:
            ax.axvline(t, color='grey', lw=0.8, ls=':')
            ax.text(t, 1.0, ' ' + lab, transform=ax.get_xaxis_transform(), fontsize=8, va='top', color='grey')
        ax.set_ylabel('v (µm/ns)')
        ax.set_title(f'Chamber {arm}: drift velocity from the column duration (HV fixed all campaign)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        vv = T[arm].v.dropna()
        if len(vv):
            lo, hi = np.percentile(vv, [1, 99])
            pad = 0.15 * (hi - lo)
            ax.set_ylim(min(lo, vb) - pad, max(hi, vb) + pad)
        ax.legend(fontsize=7, ncol=3, loc='lower left')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fn = f'vt_{arm}.png'
        fig.savefig(f'{FIG}/{fn}', dpi=130)
        plt.close(fig)
        figs.append((arm, fn))
    if len(OFF):
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for arm, g in OFF.groupby('arm'):
            ax.hist(g.ratio, bins=np.arange(0.9, 1.1, 0.004), histtype='step', lw=1.4, label=f'{arm} ({len(g)})')
        ax.axvline(1, color='k', lw=0.6)
        ax.set_xlabel('v cosmic sub-run / median v of beam sub-runs within 12 h')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f'{FIG}/vt_beam_cosmic.png', dpi=130)
        plt.close(fig)

    # common mode: A, C (and D) on the same sub-runs, and the hour-of-day residual about a 48 h running median
    m = S['A'].merge(S['C'], on=['run', 'subrun', 'trigger'], suffixes=('A', 'C'))
    if 'D' in S:
        m = m.merge(S['D'][['run', 'subrun', 'v']].rename(columns={'v': 'vD'}), on=['run', 'subrun'], how='left')
    cm = []
    for trig, g in m.dropna(subset=['vA', 'vC']).groupby('trigger'):
        rA, rC = g.vA / g.vA.median() - 1, g.vC / g.vC.median() - 1
        d = dict(trigger=trig, n=len(g), corr_AC=np.corrcoef(rA, rC)[0, 1], slope_CA=np.polyfit(rA, rC, 1)[0],
                 rms_A=100 * rA.std(), rms_C=100 * rC.std())
        gd = g.dropna(subset=['vD']) if 'vD' in g else g.iloc[:0]
        d['corr_CD'] = np.corrcoef(gd.vD, gd.vC)[0, 1] if len(gd) > 10 else np.nan
        d['n_D'] = len(gd)
        cm.append(d)
    CM = pd.DataFrame(cm)
    CM.to_csv(f'{a.vt}/vt_common_mode.csv', index=False)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    hod = {}
    for arm in ('A', 'C'):
        b = S[arm][S[arm].trigger == 'beam'].dropna(subset=['v']).sort_values('t_start').set_index('t_start')
        r = (b.v / b.v.rolling('48h', center=True).median() - 1) * 100
        h = r.groupby(r.index.hour).median()
        hod[arm] = h
        ax.plot(h.index, h.values, 'o-', ms=3, label=arm)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('hour of day (local)'); ax.set_ylabel('v / 48 h running median − 1 (%)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{FIG}/vt_hour_of_day.png', dpi=130); plt.close(fig)
    cmtxt = ''.join(f"<li>{r.trigger} sub-runs (n = {r.n}): corr(A, C) = {r.corr_AC:.2f}, C moves {r.slope_CA:.2f}× A "
                    f"(rms A {r.rms_A:.1f} %, C {r.rms_C:.1f} %)"
                    + (f"; corr(C, D) = {r.corr_CD:.2f} (n = {r.n_D})" if np.isfinite(r.corr_CD) else "; D: no beam v")
                    + "</li>"
                    for r in CM.itertuples())
    hodtxt = ', '.join(f"{arm} {h.max() - h.min():.1f} % peak-to-peak (high at {h.idxmax():02d} h, low at {h.idxmin():02d} h)"
                       for arm, h in hod.items())

    def f(x, d=2):
        return '–' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'

    sub = SUM[SUM.level == 'subrun']
    lines = []
    for arm in T:
        for _, r in sub[sub.arm == arm].iterrows():
            if r.n:
                lines.append(f"<li><b>{arm} {r.trigger}</b>: median {f(r['median'])} µm/ns over {r.n} sub-runs, 68 % "
                             f"spread ±{f(r.half68)} ({f(100 * r.half68 / r['median'], 1)} %), range {f(r.vmin)}–"
                             f"{f(r.vmax)}; frozen bundle {f(r.v_bundle)} ({f(100 * (r['median'] / r.v_bundle - 1), 1)} %)</li>")
    offtxt = ''.join(f"<li>{arm}: cosmic/beam = {f(g.ratio.median(), 3)} (n = {len(g)}, 68 % ±"
                     f"{f(np.subtract(*np.percentile(g.ratio, [84, 16])) / 2, 3)})</li>" for arm, g in OFF.groupby('arm')) \
        if len(OFF) else '<li>no cosmic sub-run within 12 h of a beam sub-run</li>'

    def table(D):
        h = ('<tr><th>arm</th><th>level</th><th>trigger</th><th>rows</th><th>with v</th><th>median</th><th>p16</th>'
             '<th>p84</th><th>±68 %/2</th><th>median stat err</th><th>excess over stat</th><th>min</th><th>max</th>'
             '<th>bundle v</th></tr>')
        b = ''.join(f'<tr><td>{r.arm}</td><td>{r.level}</td><td>{r.trigger}</td><td>{r.n_rows}</td><td>{r.n}</td>'
                    f'<td>{f(r.get("median"))}</td><td>{f(r.get("p16"))}</td><td>{f(r.get("p84"))}</td>'
                    f'<td>{f(r.get("half68"))}</td><td>{f(r.get("med_err"))}</td><td>{f(r.get("excess"))}</td>'
                    f'<td>{f(r.get("vmin"))}</td><td>{f(r.get("vmax"))}</td><td>{f(r.v_bundle)}</td></tr>'
                    for _, r in D.iterrows())
        return f'<table>{h}{b}</table>'

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>n_TOF v(t)</title>
<style>body{{font-family:sans-serif;max-width:1150px;margin:auto;padding:0 16px;line-height:1.4}}
table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #bbb;padding:3px 7px;text-align:right}}
img{{max-width:100%}}.cap{{font-size:13px;color:#444}}</style></head><body>
<h1>n_TOF pass 1: drift velocity v(t) (P7a)</h1>
<p>Generated by <code>rerun/ntof/vt/make_vt_report.py</code> from <code>vt_table.py</code> ({meta['n_files']:,} vt_acq
outputs, condor 4677358 on the pass-1 stage-2 tracks). v = timing-free gap / column duration T (stacked-median box fit,
pass-1 X template), per acquisition (≥ {meta['min_n']} clean tracks per view) and pooled per sub-run.</p>
<h2>Headline (per sub-run)</h2><ul>{''.join(lines)}</ul>
<h2>A common gas effect</h2><ul>{cmtxt}</ul>
<p>Daily cycle on the beam sub-runs, after removing a 48 h running median: {hodtxt}.</p>
<p><img src="figures/vt_hour_of_day.png"></p>
<h2>Beam vs cosmic selection offset</h2><ul>{offtxt}</ul>
<p class="cap">Beam and cosmic samples select different tracks (angle, depth, multiplicity), so their T differ; the two
classes are kept apart and never pooled.</p>
<h2>Tables</h2>{table(SUM)}
<p class="cap">"excess over stat" = sqrt(half68² − median stat err²): the scatter not explained by the bootstrap
error, i.e. real v(t) variation plus unmodelled systematics. Gap error (common scale): {meta['gap_err_mm']} mm.</p>
<h2>Figures</h2>
{''.join(f'<p><img src="figures/{fn}"><br><span class="cap">Chamber {arm}. Small points: per acquisition; open circles: per sub-run; dashed: the frozen bundle v used by the pass. Dotted lines: the 23 July readout-clock change (noise, not gas) and the 27 July access.</span></p>' for arm, fn in figs)}
{'<p><img src="figures/vt_beam_cosmic.png"><br><span class="cap">Cosmic sub-runs against nearby beam sub-runs.</span></p>' if len(OFF) else ''}
<h2>What this does not rule out</h2><ul>
{''.join(f'<li>{c}</li>' for c in meta['caveats'])}
<li>T measures v only through the timing-free gap; an error in the gap is a common scale on all of a chamber's v
(A ±0.7 %, D ±3.8 %, C not quoted).</li>
<li>HV is the same for every acquisition of each chamber (registry), so no point is flagged for an HV change; a gain
drift at fixed HV (temperature, pressure) would still move T and read as v.</li>
<li>B has no v (column not a box).</li></ul>
</body></html>"""
    open(f'{a.vt}/report.html', 'w').write(html)
    print(SUM.to_string())
    if len(OFF):
        print(OFF.groupby('arm').ratio.describe())
    print(f'{a.vt}/report.html')


if __name__ == '__main__':
    main()
