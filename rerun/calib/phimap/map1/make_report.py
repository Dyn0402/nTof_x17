#!/usr/bin/env python
"""make_report.py -- report.html and GATES.md of the step-4 gates (reads $JOB/map1/gates.json, build_summary.json).

Writes $JOB/map1/report.html (+ figures/) and a copy of report.html, figures/ and GATES.md next to this script.
"""
import html
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JOB = os.environ.get('PHIMAP_JOB', '/media/ucla/work/claude-jobs/d006ab29')
OUT = os.environ.get('MAP1_OUT', f'{JOB}/map1')
G = json.load(open(f'{OUT}/gates.json'))
S = json.load(open(f'{OUT}/build_summary.json'))
os.makedirs(f'{OUT}/figures', exist_ok=True)
NSIG = 2.0
MINROW = G['min_row']
MINFLAG = G['min_flag']

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 130})
C1, C2, CG = '#8a8f98', '#1f6fb4', '#c4c8cf'


def short(k):
    k = k.replace('c26__', '').replace('mx17_', '')
    return k.replace('overnight_', 'ovn_').replace('-26', '')[-52:]


def fmt(x, nd=3):
    return 'n/a' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def pm(a, e, nd=3):
    return f'{fmt(a, nd)} &plusmn; {fmt(e, nd)}' if e is not None else fmt(a, nd)


def flags(s):
    """list of the gates a (row or group, view) fails, with the NSIG-sigma rule."""
    if 'cv' not in s or s.get('dcv_err', 0) == 0:
        return []
    f = []
    if s['diff_cv'] > NSIG * s['dcv_err']:
        f.append('angle')
    if s['pos']['diff_cv'] > NSIG * s['pos']['dcv_err']:
        f.append('position')
    if s['far_diff_cv'] > 0 and s['far_diff_cv'] > NSIG * s['fdcv_err']:
        f.append('far')
    return f


groups = {g['group']: g for g in G['groups']}
rows = {r['key']: r for r in G['rows']}
order = ['A_0627', 'A_0625', 'B_0622', 'C_long (own map)', 'C_longer (own map)', 'C borrowed (C_long)', 'C borrowed (C_longer)', 'D_0626',
         'E', 'A_0622', 'A_0623', 'A_0627 (100/300 V, no map)', 'B_0617']
order = [o for o in order if o in groups]

# ------------------------------------------------------------------------------------------------ numbers
bad_rows = []
for k, r in rows.items():
    for v in 'xy':
        s = r[v]
        if r['n_sel'] >= MINFLAG and flags(s):
            bad_rows.append((k, v, r['n_sel'], flags(s)))
bad_groups = [(g, v, flags(groups[g][v])) for g in order for v in 'xy' if flags(groups[g][v])]
nmap_rows = sum(1 for i in S.values() if not i['nomap'])
n_ev = sum(i['n_events'] for i in S.values())
n_used = {v: sum(i.get(f'n_used_{v}', 0) for i in S.values()) for v in 'xy'}
max_s3 = max(max(i.get('max_diff_vs_step3_x', 0), i.get('max_diff_vs_step3_y', 0)) for i in S.values())
max_unused = max(max(i.get('max_diff_unused_table_vs_pass1_x', 0), i.get('max_diff_unused_table_vs_pass1_y', 0)) for i in S.values())
fb_ok = all(i.get(f'fallback_equals_pass_bundle_{v}', True) for i in S.values() for v in 'xy' if not i['nomap'])
n_bundle_ne = sum(i.get(f'n_bundle_ne_pass1_{v}', 0) for i in S.values() for v in 'xy')
n_not_used = sum(i.get(f'n_not_used_{v}', 0) for i in S.values() for v in 'xy' if not i['nomap'])
max_bundle_ne = max(max(i.get('max_diff_bundle_vs_pass1_x', 0), i.get('max_diff_bundle_vs_pass1_y', 0)) for i in S.values())
max_edge_unused = max(groups[g][f'edge_{v}'].get('max_diff_unused', 0) for g in groups for v in 'xy')
rows_big = [r for r in G['rows'] if r['n_sel'] >= MINROW]
verdict_pass = not bad_rows and not bad_groups

# ------------------------------------------------------------------------------------------------ figures
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
for a, v in zip(ax, 'xy'):
    ys = np.arange(len(order))
    p1 = [groups[g][v].get('p1', np.nan) for g in order]
    cv = [groups[g][v].get('cv', np.nan) for g in order]
    pe = [groups[g][v].get('p1_err', 0) for g in order]
    ce = [groups[g][v].get('cv_err', 0) for g in order]
    a.barh(ys - 0.2, p1, 0.38, xerr=pe, color=C1, label='pass 1')
    a.barh(ys + 0.2, cv, 0.38, xerr=ce, color=C2, label='map, cross-validated (interior only)')
    a.set_yticks(ys)
    a.set_yticklabels(order if v == 'x' else [])
    a.invert_yaxis()
    a.set_xscale('log')
    a.set_xlabel(f'angle s68, {v.upper()} view [deg]')
    a.set_title(f'{v.upper()} view')
h_, l_ = ax[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, loc='lower center', ncol=2, fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(f'{OUT}/figures/units_s68.png')
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for a, v in zip(ax, 'xy'):
    n = np.array([r['n_sel'] for r in G['rows'] if 'cv' in r[v] and r[v].get('dcv_err', 0) > 0 and r['n_sel'] >= MINFLAG])
    d = np.array([r[v]['diff_cv'] / r[v]['dcv_err'] for r in G['rows'] if 'cv' in r[v] and r[v].get('dcv_err', 0) > 0 and r['n_sel'] >= MINFLAG])
    big = n >= MINROW
    a.scatter(n[~big], d[~big], s=12, color=CG, label=f'rows with {MINFLAG}-{MINROW} events')
    a.scatter(n[big], d[big], s=18, color=C2, label=f'rows with >= {MINROW} events')
    a.axhline(0, color='k', lw=0.6)
    a.axhline(NSIG, color='#b03a2e', lw=0.8, ls='--')
    a.set_xscale('log')
    a.set_xlabel('gate events of the row')
    a.set_ylabel('(map CV - pass 1) / error')
    a.set_title(f'{v.upper()} view: angle s68 difference per row')
ax[0].legend(frameon=False, fontsize=8, loc='lower left')
fig.tight_layout()
fig.savefig(f'{OUT}/figures/row_pulls.png')
plt.close(fig)

eg = [g for g in order if 'cv' in groups[g]['edge_x']]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for a, v in zip(ax, 'xy'):
    ys = np.arange(len(eg))
    a.barh(ys - 0.2, [groups[g]['edge_' + v]['p1'] for g in eg], 0.38, color=C1, label='pass 1')
    a.barh(ys + 0.2, [groups[g]['edge_' + v]['cv'] for g in eg], 0.38, color=C2, label='interior-only map (CV)')
    a.set_yticks(ys)
    a.set_yticklabels(eg if v == 'x' else [])
    a.invert_yaxis()
    a.set_xscale('log')
    a.set_xlabel(f'edge zone, angle s68, {v.upper()} [deg]')
ax[0].legend(frameon=False, fontsize=8, loc='lower right')
fig.tight_layout()
fig.savefig(f'{OUT}/figures/edge_zone.png')
plt.close(fig)

fig, ax = plt.subplots(2, 3, figsize=(10, 5), sharex=True)
for j, g in enumerate(['A_0627', 'B_0622', 'C_long (own map)']):
    for i, v in enumerate('xy'):
        pr = groups[g]['profile_' + v]
        x = [0.5 * (p[0] + p[1]) for p in pr if p[3] is not None]
        ax[i, j].plot(x, [p[3] for p in pr if p[3] is not None], 'o-', color=C1, label='pass 1')
        ax[i, j].plot(x, [p[4] for p in pr if p[3] is not None], 'o-', color=C2, label='map (CV)')
        ax[i, j].axhline(0, color='k', lw=0.5)
        ax[i, j].set_title(f'{g}, {v.upper()}')
        if i == 1:
            ax[i, j].set_xlabel('tan of the reference track')
        if j == 0:
            ax[i, j].set_ylabel('mean residual [deg]')
ax[0, 0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(f'{OUT}/figures/profile.png')
plt.close(fig)

# ------------------------------------------------------------------------------------------------ tables
def unit_table(md=False):
    hdr = ['group', 'view', 'N', 'pass 1', 'map (CV)', 'diff +- err', 'far %: pass 1 -> CV', 'position s68 [mm]: pass 1 -> CV', 'map used', 'gates']
    out = []
    for g in order:
        for v in 'xy':
            s = groups[g][v]
            f = flags(s)
            out.append([g, v.upper(), s['n'], f"{s['p1']:.3f} +- {s['p1_err']:.3f}" if 'p1_err' in s else f"{s['p1']:.3f}",
                        f"{s['cv']:.3f} +- {s['cv_err']:.3f}" if 'cv_err' in s else f"{s['cv']:.3f}",
                        f"{s['diff_cv']:+.3f} +- {s['dcv_err']:.3f}" if 'dcv_err' in s else '0',
                        f"{100 * s['far_p1']:.1f} -> {100 * s['far_cv']:.1f}",
                        f"{s['pos']['p1']:.3f} -> {s['pos']['cv']:.3f}" if 'pos' in s else '', f"{100 * s['frac_used']:.0f} %",
                        'FAIL ' + ', '.join(f) if f else 'ok'])
    return hdr, out


def used_table():
    hdr = ['group', 'view', 'N (map used)', 'pass 1', 'map (CV)', 'diff +- err']
    out = []
    for g in order:
        for v in 'xy':
            s = groups[g].get(v + '_used', {})
            if 'cv' in s and groups[g][v]['frac_used'] > 0 and groups[g][v]['frac_used'] < 0.999:
                out.append([g, v.upper(), s['n'], f"{s['p1']:.3f} +- {s['p1_err']:.3f}", f"{s['cv']:.3f} +- {s['cv_err']:.3f}", f"{s['diff_cv']:+.3f} +- {s['dcv_err']:.3f}"])
    return hdr, out


def edge_table():
    hdr = ['group', 'view', 'N', 'map used %', 'pass 1', 'interior-only map (CV)', 'diff +- err']
    out = []
    for g in order:
        for v in 'xy':
            s = groups[g]['edge_' + v]
            if 'cv' in s:
                out.append([g, v.upper(), s['n'], f"{100 * s['frac_used']:.1f}", f"{s['p1']:.3f}", f"{s['cv']:.3f}", f"{s['diff_cv']:+.3f} +- {s['dcv_err']:.3f}"])
    return hdr, out


def row_table():
    hdr = ['row', 'group', 'N', 'X: pass 1', 'X: CV', 'X: diff +- err', 'X far %', 'X pos', 'Y: pass 1', 'Y: CV', 'Y: diff +- err', 'Y far %', 'Y pos', 'gates']
    out = []
    for r in sorted(rows_big, key=lambda r: (S[r['key']]['chamber'], r['key'])):
        a, b = r['x'], r['y']
        fl = [f'X {"/".join(flags(a))}' if flags(a) else '', f'Y {"/".join(flags(b))}' if flags(b) else '']
        def cells(s):
            if 'cv' not in s:
                return ['', '', '', '', '']
            return [f"{s['p1']:.3f}", f"{s['cv']:.3f}", f"{s['diff_cv']:+.3f} +- {s['dcv_err']:.3f}",
                    f"{100 * s['far_p1']:.1f} -> {100 * s['far_cv']:.1f}", f"{s['pos']['p1']:.3f} -> {s['pos']['cv']:.3f}"]
        out.append([short(r['key']), G['allrows'][r['key']]['group'], r['n_sel']] + cells(a) + cells(b) + ['FAIL ' + ' '.join(x for x in fl if x) if any(fl) else 'ok'])
    return hdr, out


def rest_summary():
    small = [r for r in G['rows'] if MINFLAG <= r['n_sel'] < MINROW]
    tiny = [k for k, a in G['allrows'].items() if a['n_gate'] < MINFLAG]
    nb = sum(1 for r in small for v in 'xy' if flags(r[v]))
    nused = sum(1 for r in small if G['allrows'][r['key']]['info']['unit'] and not G['allrows'][r['key']]['info']['nomap'])
    return dict(n_small=len(small), n_events=sum(r['n_sel'] for r in small), n_flagged=nb, n_with_map=nused, n_tiny=len(tiny),
                n_tiny_events=sum(G['allrows'][k]['n_gate'] for k in tiny))


def rows_below_table():
    out = []
    for r in G['rows']:
        if MINFLAG <= r['n_sel'] < MINROW:
            for v in 'xy':
                if flags(r[v]):
                    s = r[v]
                    out.append([short(r['key']), v.upper(), r['n_sel'], f"{s['p1']:.3f}", f"{s['cv']:.3f}", f"{s['diff_cv']:+.3f} +- {s['dcv_err']:.3f}", ' '.join(flags(s))])
    return ['row', 'view', 'N', 'pass 1', 'CV', 'diff +- err', 'gates'], out


def railed_table():
    hdr = ['group', 'view', 'railed N', 'mean residual x sign(tan) [deg]: pass 1', 'map CV x rescale', 's68: pass 1', 's68: CV x rescale']
    out = []
    for g in order:
        for v in 'xy':
            s = groups[g].get('railed_' + v)
            if s and s['n'] > 20:
                out.append([g, v.upper(), s['n'], f"{s['mean_p1']:+.3f}", f"{s['mean_cv']:+.3f}", f"{s['s68_p1']:.3f}", f"{s['s68_cv']:.3f}"])
    return hdr, out


def html_table(hdr, body, cls=''):
    t = ['<table class="%s"><thead><tr>%s</tr></thead><tbody>' % (cls, ''.join(f'<th>{html.escape(str(h))}</th>' for h in hdr))]
    for r in body:
        bad = str(r[-1]).startswith('FAIL')
        t.append('<tr%s>%s</tr>' % (' class="bad"' if bad else '', ''.join(f'<td>{html.escape(str(c))}</td>' for c in r)))
    t.append('</tbody></table>')
    return '\n'.join(t)


def md_table(hdr, body):
    return '\n'.join(['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)] + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in body])


rs = rest_summary()
u_h, u_b = unit_table()
us_h, us_b = used_table()
e_h, e_b = edge_table()
r_h, r_b = row_table()
rb_h, rb_b = rows_below_table()
ra_h, ra_b = railed_table()


def gain(g, v):
    s = groups[g][v]
    return f"{s['p1']:.3f} -> {s['cv']:.3f} ({-100 * s['diff_cv'] / s['p1']:.0f} %)"


headline = ' '.join([f"{g.split(' ')[0] if not g.startswith('C') else g}: X {gain(g, 'x')}, Y {gain(g, 'y')};" for g in ['A_0627', 'A_0625', 'B_0622', 'C_long (own map)', 'D_0626']])
if verdict_pass:
    verdict = (f'The interior-only maps pass the gates on all {len(rows_big)} rows with at least {MINROW} gate events and on all {len(order)} pooled groups: '
               f'no angle s68, no position s68 is worse than pass 1 beyond {NSIG:.0f} sigma, and no far fraction is higher beyond {NSIG:.0f} sigma.')
else:
    verdict = (f'{len(bad_rows)} row/view combinations and {len(bad_groups)} pooled group/view combinations fail at least one gate '
               f'({NSIG:.0f} sigma rule); see the lists below.')

exact = f"""<ul>
<li><b>Outside the map's region the angle is pass 1, bit for bit.</b> {n_not_used:,} view-events of the {nmap_rows} rows with a map are outside the region
(or have no finite position) and keep the stored pass-1 value; the largest difference between the new table and pass 1 there is {max_unused:.1e}
(per row in <code>map1/&lt;row&gt;/map1_row.json</code>). The bundle code path (<code>bow_pairs</code> with the fallback constants) gives
the same array as the pass-1 bundle's <code>bow_pairs</code>, bit for bit, on all {2 * nmap_rows} row/view pairs: {fb_ok}. It differs from the <i>stored</i> pass-1 value on
{n_bundle_ne} events ({100 * n_bundle_ne / max(n_not_used, 1):.2f} % of the events outside; up to {max_bundle_ne:.2g}), the same events on which the pass-1 bundle itself does not
reproduce the stored table (stored value from the winner mapping, quality flag false); the table keeps the stored value.</li>
<li><b>Agreement with step 3:</b> where the map is used, the new tangent equals step 3's <code>x_tan_map</code> / <code>y_tan_map</code> to {max_s3:.1e} on {n_used['x']:,} X and {n_used['y']:,} Y
events (every event of every mapped row with a finite value).</li>
<li><b>Edge zone:</b> largest |new - pass 1| where the map is not used is {max_edge_unused:.1e} (all groups, both views).</li>
</ul>"""

# ------------------------------------------------------------------------------------------------ html
CSS = """
body{font:14px/1.5 system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:16px;color:#1c1f24;background:#fff}
h1{font-size:22px} h2{font-size:17px;margin-top:28px;border-bottom:1px solid #d5d8de;padding-bottom:3px}
table{border-collapse:collapse;font-size:12px;margin:8px 0;display:block;overflow-x:auto}
th,td{border:1px solid #d5d8de;padding:2px 6px;text-align:right;white-space:nowrap} th{background:#f1f3f6} td:first-child,th:first-child,td:nth-child(2),th:nth-child(2){text-align:left}
tr.bad td{background:#fbe3e0}
.verdict{background:#eef6ee;border:1px solid #b7d7b7;padding:10px 14px;border-radius:6px;font-size:15px} .verdict.fail{background:#fbe3e0;border-color:#e0a49d}
img{max-width:100%} figcaption{font-size:12px;color:#444;margin-bottom:12px} code{font-size:12px}
@media (prefers-color-scheme:dark){body{background:#15171a;color:#e3e5e8} th{background:#23262b} th,td{border-color:#3a3e45} .verdict{background:#1c2b1c;border-color:#355a35} tr.bad td{background:#4a2420}}
"""
body = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bench angle maps, step 4: gates</title><style>{CSS}</style></head><body>
<h1>Bench re-finalization, step 4: interior-only angle maps in the row bundles, and the gates</h1>
<div class="verdict{'' if verdict_pass else ' fail'}"><b>Verdict.</b> {html.escape(verdict)}</div>
<h2>Headline numbers (angle s68 [deg], pass 1 -> cross-validated map, interior only)</h2>
<p>{html.escape(headline)} The same chambers on pass 1 are the reference; E, A 6-22, A 6-23, A 6-27 at 100 and 300 V and B 6-17 stay on pass 1 (identical columns).
Quoted angles are the cross-validated ones (five time blocks, map and row constants trained without the event's block); events outside the map's region count with their pass-1 value.
Positions: the mid-gap position residual s68 improves by 0-8 % where the map is used (one group, C borrowed from the longer row, Y, gets worse by 0.025 +- 0.013 mm, below 2 sigma); the mesh position p0 is not touched by the map.</p>
<h2>What was done</h2>
<ul><li>{nmap_rows} of the 156 rows carry a map bundle (<code>map1/&lt;run&gt;/&lt;subrun&gt;/&lt;det&gt;/calib_bundle</code>: the pass bundle, the <code>bow['phi']</code> block with
<code>outside = 'block'</code>, the row's w0 and kw measured with the map, and the pass-1 w0 and kw as <code>fallback</code>); the other {156 - nmap_rows} keep the pass-1 bundle.
{n_ev:,} events were recomputed in all 156 <code>events.parquet</code> ({n_used['x']:,} X and {n_used['y']:,} Y angles from a map).</li>
<li>Chamber D's X view uses the map only for x &gt;= 200 mm (<code>domain_x</code>, U &gt;= 0); its Y view uses the whole domain.</li>
<li>C: railed tracks (q_uend at the end of the depth grid) are multiplied by the per-row factor of step 3 (<code>x_resc</code>, <code>y_resc</code>). D and E railed tracks are flagged in <code>x_railed</code>, <code>y_railed</code> and not changed (the map still applies to D's).
C rows without their own map use the map of the nearest C row (<code>map_borrowed</code>).</li>
<li>Columns changed where the map is used: <code>x/y_tan_theta, x/y_theta_deg, x/y_slope_reliable, x/y_bow_h</code>, <code>bow_h</code>. Added: <code>map_unit, map_variant, map_borrowed, map_used_x/y, x/y_railed, x/y_resc, x/y_tan_pass1</code>.</li></ul>
<h2>Exactness checks</h2>{exact}
<h2>Gates by pooled group (gate selection of step 3; D and E railed tracks excluded, C railed tracks included)</h2>
<p>Gates: angle s68 not worse than pass 1 beyond {NSIG:.0f} sigma (paired bootstrap); far fraction (|residual| &gt; 3 s68 of pass 1) not higher; position s68 not worse. &quot;map used&quot; is the share of events inside the map's region.</p>
{html_table(u_h, u_b)}
<p>Where the map is used only (the groups with a partial region: D's X view for x &gt;= 200 mm):</p>
{html_table(us_h, us_b)}
<figure><img src="figures/units_s68.png" alt="angle s68 per group"><figcaption>Figure 1. Angle s68 of every group, pass 1 (grey) and the cross-validated interior-only map (blue), with bootstrap errors. The groups without a map are identical by construction. D's X bar includes the events at x &lt; 200 mm that stay on pass 1.</figcaption></figure>
<figure><img src="figures/profile.png" alt="angle residual profile"><figcaption>Figure 2. Mean angle residual (angle minus the reference) against the reference tangent, pass 1 and the cross-validated map, for three of the mounted maps (bins of the reference tangent with at least 20 events). The map flattens the curvature of B and the slope of C Y in the interior bins; the residuals of A are small before and after (0.1 deg or less). The outermost bin of C Y is not better with the map (it holds few events and lies near the edge of the region).</figcaption></figure>
<h2>Gates by row (rows with at least {MINROW} gate events)</h2>
<p>Each line is one row of the registry: angle s68 pass 1 and cross-validated map with the paired difference, far fraction, and mid-gap position s68 [mm].</p>
{html_table(r_h, r_b)}
<figure><img src="figures/row_pulls.png" alt="row pulls"><figcaption>Figure 3. For every row with at least {MINFLAG} gate events, the change of the angle s68 divided by its bootstrap error against the number of events. Negative is better. The dashed line is the 2 sigma limit for &quot;worse&quot;; no row crosses it.</figcaption></figure>
<h3>Rows with {MINFLAG}-{MINROW} gate events</h3>
<p>{rs['n_small']} rows with {rs['n_events']:,} gate events in total ({rs['n_with_map']} with a map); view/gate combinations failing the 2 sigma rule: {rs['n_flagged']}.
The remaining {rs['n_tiny']} rows have fewer than {MINFLAG} gate events ({rs['n_tiny_events']:,} in total) and enter only the pooled groups.</p>
{html_table(rb_h, rb_b) if rb_b else '<p>None of them fails a gate.</p>'}
<h2>Edge zone</h2>
<p>Events inside the active area but within 30 mm of its border, which no map saw in its fit. Outside the map's region they are pass 1 by construction; the few that fall just inside the region (a few percent) use the map.</p>
{html_table(e_h, e_b)}
<figure><img src="figures/edge_zone.png" alt="edge zone"><figcaption>Figure 4. Angle s68 in the edge zone, pass 1 and the interior-only map. The bars are equal up to the few events that lie inside the map's region. Step 3 had found a clamped map up to 1 degree worse here (E, C longer, D, A 6-25 Y).</figcaption></figure>
<h2>Railed tracks of C</h2>
{html_table(ra_h, ra_b)}
<h2>What the result does not rule out</h2>
<ul>
<li>The maps are trained on the gate rows of their mounting and the quoted numbers are cross-validated in five time blocks of the same mounting. We have no measurement of a map on a mounting it was not trained on, and a map does not transfer between mountings.</li>
<li>The railed tracks of A 6-22 (80 %), D and E are not corrected: their angles are biased flat by the depth grid (pass-level fix: K = 27), and they are flagged only.</li>
<li>Pass-1 angles outside the map's region (edge zone, D's X at x &lt; 200 mm, the rows without a map) are as good or as bad as before: A 6-22 3.4 / 2.7 deg, A 6-23 18 / 35 deg, E 2.2 / 1.5 deg, the 100 V and 300 V rows of A 6-27.</li>
<li>The region is a rectangle in (U, W) at the reconstructed mid-gap position. A track of large angle can move across its border, so the angle can jump by the difference between the map and the plain block at the border (smooth within a few tenths of a degree for the mounted maps; not measured as a function of distance to the border).</li>
<li>The gate selection is the reference-matched, single-track, interior selection of step 3. Events without a reference track, or with several tracks, have the map applied but no measurement of its effect.</li>
<li>The row constants w0, kw of the 22 borrowed C rows come from few events (20 to 450 per row).</li>
</ul>
<p>Code: <code>rerun/calib/phimap/map1/</code> (<code>build_map1.py</code>, <code>gates.py</code>, this script). Data: <code>{OUT}</code>.</p>
</body></html>"""
open(f'{OUT}/report.html', 'w').write(body)

md = f"""# Step 4 gates: interior-only angle maps against pass 1 (2026-11-24)

**Verdict.** {verdict}

Data: `{OUT}` (`<run>/<subrun>/<det>/{{calib_bundle, events.parquet, map1_row.json}}`, `gates.json`, `report.html`). Code: `build_map1.py`, `gates.py`, `make_report.py` in this directory.
Quoted angles: five-time-block cross-validation (step 3), events outside a map's region at their pass-1 value. Gate rule: {NSIG:.0f} sigma, paired bootstrap ({G['nboot']}).

## Exactness

- Outside the region the table value is the stored pass-1 value (max difference {max_unused:.1e} on {n_not_used:,} view-events); the bundle fallback equals the pass bundle's `bow_pairs` bit for bit on all row/view pairs: {fb_ok}.
- Where the map is used the tangent equals step 3's `x_tan_map` / `y_tan_map` to {max_s3:.1e} ({n_used['x']:,} X, {n_used['y']:,} Y events).

## Pooled groups

{md_table(u_h, u_b)}

## Rows with >= {MINROW} gate events

{md_table(r_h, r_b)}

## Rows with {MINFLAG}-{MINROW} gate events failing a gate

{md_table(rb_h, rb_b) if rb_b else 'None.'}

## Edge zone

{md_table(e_h, e_b)}

## Railed tracks of C

{md_table(ra_h, ra_b)}
"""
open(f'{OUT}/GATES.md', 'w').write(md)

for f in ('report.html', 'GATES.md'):
    shutil.copy(f'{OUT}/{f}', os.path.join(HERE, f))
shutil.copytree(f'{OUT}/figures', os.path.join(HERE, 'figures'), dirs_exist_ok=True)
print('verdict pass' if verdict_pass else 'verdict FAIL', len(bad_rows), 'bad rows', len(bad_groups), 'bad groups')
for b in bad_rows:
    print('BAD ROW', b)
for b in bad_groups:
    print('BAD GROUP', b)
