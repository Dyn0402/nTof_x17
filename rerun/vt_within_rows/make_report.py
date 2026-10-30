#!/usr/bin/env python3
"""make_report.py -- figures, results/summary.json and report.html of the within-row v(t) study, from the results of
vt_measure.py and vt_apply.py (no data access).  Re-run after either to refresh numbers, figures and verdict together."""
import html
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vt_common import RES, FIG, HERE, NAME, COL  # noqa: E402
from vt_apply import vlad, OVERNIGHT, GAP  # noqa: E402

B = json.load(open(os.path.join(RES, 'blocks.json')))
M = json.load(open(os.path.join(RES, 'models.json')))
H = json.load(open(os.path.join(RES, 'heldout.json')))
C = json.load(open(os.path.join(RES, 'consequences.json')))
DETS = ('mx17_6', 'mx17_7')
PLATEAU_T = 12.4          # h: the second half of the HV scan, after the knee (both chambers flat within 2 %)
SCAN_T = 10.4             # h: the overnight run ends, the HV scan starts
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e3', 'grid.linewidth': 0.6})


def blocks_of(det, half='all'):
    t, v, e, sub = [], [], [], []
    for r in B:
        if r['det'] != det:
            continue
        for b in r[half]['blocks']:
            vv, ee = vlad(b)
            if np.isfinite(vv):
                t.append(b['t']); v.append(vv); e.append(ee); sub.append(r['subrun'])
    o = np.argsort(t)
    return np.array(t)[o], np.array(v)[o], np.array(e)[o], np.array(sub)[o]


def angle_bias_deg(eps, theta_deg):
    th = np.radians(theta_deg)
    return np.degrees(np.arctan((1 + eps) * np.tan(th)) - th)


S = dict(model=C['model'], heldout_chi2=C['heldout_chi2'], heldout_nblocks=C['heldout_nblocks'], chamber={}, rows=[])
for d in DETS:
    t, v, e, sub = blocks_of(d)
    pl = t >= PLATEAU_T
    vp = float(np.average(v[pl], weights=1 / e[pl] ** 2))
    S['chamber'][d] = dict(v_first=float(v[0]), t_first=float(t[0]), v_plateau=vp,
                           rise_pct=float((vp / v[0] - 1) * 100), night_fit={k: M['all'][d][k] for k in
                                                                            ('vinf', 'D', 'tau', 'err', 'chi2', 'ndf')})
    # gain insensitivity: the plateau rows (one HV each, gain x ~60) -- v_lad against log10 of the row's charge
    q = {c['subrun']: np.median([b['q_sum'] for b in c['gas_blocks']]) for c in C['rows'] if c['det'] == d
         and c['gas_blocks']}
    rr = [(np.log10(q[r['subrun']]), vlad(dict(x=r['all']['row']['x'], y=r['all']['row']['y']))[0])
          for r in B if r['det'] == d and r['t_start'] >= PLATEAU_T and r['subrun'] in q]
    lq, vv = np.array(rr).T
    sl = np.polyfit(lq, vv / vv.mean(), 1)[0]
    S['chamber'][d]['plateau_rows'] = len(rr)
    S['chamber'][d]['plateau_q_range'] = [float(10 ** lq.min()), float(10 ** lq.max())]
    S['chamber'][d]['plateau_rms_pct'] = float(np.std(vv / vv.mean()) * 100)
    S['chamber'][d]['v_per_decade_q_pct'] = float(sl * 100)
# same shape?  the fractional deficit 1 - v / v_plateau of det6 against det7's at the same clock time, before the scan
t6, v6, e6, _ = blocks_of('mx17_6'); t7, v7, e7, _ = blocks_of('mx17_7')
k = t6 < SCAN_T
d6 = 1 - v6[k] / S['chamber']['mx17_6']['v_plateau']; s6 = e6[k] / S['chamber']['mx17_6']['v_plateau']
d7 = 1 - np.interp(t6[k], t7, v7) / S['chamber']['mx17_7']['v_plateau']
s7 = np.interp(t6[k], t7, e7) / S['chamber']['mx17_7']['v_plateau']
for _ in range(3):        # k from a fit through the origin with both errors (effective variance)
    kk = S.get('_k', 2.5)
    w = 1 / (s6 ** 2 + (kk * s7) ** 2 + (0.01 * kk) ** 2)
    S['_k'] = float(np.sum(w * d6 * d7) / np.sum(w * d7 * d7))
kk = S.pop('_k')
w = 1 / (s6 ** 2 + (kk * s7) ** 2 + (0.01 * kk) ** 2)
S['deficit_ratio'] = dict(k=kk, k_err=float(1 / np.sqrt(np.sum(w * d7 * d7))),
                          chi2=float(np.sum(w * (d6 - kk * d7) ** 2)), ndf=int(k.sum() - 1),
                          corr=float(np.corrcoef(d6, d7)[0, 1]), n=int(k.sum()))
for c in sorted(C['rows'], key=lambda c: (c['det'], c['t_start'])):
    if c['subrun'] not in OVERNIGHT:
        continue
    h = next(h for h in H if h['det'] == c['det'] and h['subrun'] == c['subrun'])
    r = next(r for r in B if r['det'] == c['det'] and r['subrun'] == c['subrun'])
    bv = [vlad(b)[0] for b in r['all']['blocks']]
    epsmax = max(abs(c['v_rel_min'] - 1), abs(c['v_rel_max'] - 1))
    S['rows'].append(dict(
        det=c['det'], subrun=c['subrun'], t_start=c['t_start'], t_end=c['t_end'], v_drift=c['v_drift'],
        v_block_first=bv[0], v_block_last=bv[-1], n_blocks=len(bv),
        block_change_pct=(bv[-1] / bv[0] - 1) * 100, r_min=c['v_rel_min'], r_max=c['v_rel_max'],
        dtheta30_max_deg=float(angle_bias_deg(epsmax, 30.0)), dz_cathode_max_mm=GAP[c['det']] * epsmax,
        dmid_p95_mm=max(c[p]['dmid_p95_mm'] for p in 'xy'),
        dtheta_rms_deg={p: c[p]['dtheta_rms_deg'] for p in 'xy'},
        s68_before={p: c[p]['s68_before_deg'] for p in 'xy'}, s68_after={p: c[p]['s68_after_deg'] for p in 'xy'},
        held={m: {p: dict(eps_rms=h[m][p]['eps_rms'], chi2=h[m][p]['chi2'], n_blocks=h[m][p]['n_blocks'],
                          s68=h[m][p]['s68_deg']) for p in 'xy'} for m in ('const', 'linear', 'blocks', 'night')},
        z_end=c['z_end_blocks'], gas=c['gas_blocks']))
json.dump(S, open(os.path.join(RES, 'summary.json'), 'w'), indent=1, default=float)

# ---------------------------------------------------------------- figures
# 1. the night: ladder v per block, both chambers, with the per-chamber night curve and the deficit comparison
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
for d in DETS:
    t, v, e, sub = blocks_of(d)
    ax[0].errorbar(t, v, e, fmt='o', ms=4, color=COL[d], ecolor=COL[d], elinewidth=1, label=NAME[d])
    f = M['all'][d]; tg = np.linspace(0.8, 15.4, 300)
    ax[0].plot(tg, f['vinf'] - f['D'] * np.exp(-tg / f['tau']), color=COL[d], lw=1.2, alpha=0.6)
    ax[0].axhline(S['chamber'][d]['v_plateau'], color=COL[d], lw=0.8, ls=':')
    ax[0].text(15.4, v[-1] + 0.6, NAME[d], color='#333', ha='right', fontsize=9)
for lo, hi, lab in ((0.97, 1.97, 'short'), (1.97, 3.97, 'longer'), (3.98, 6.05, '530→505 V'),
                    (6.05, 10.34, 'long (golden)'), (10.4, 15.2, 'HV scan 500→400, 495→405 V')):
    ax[0].axvspan(lo, hi, color='#f0f0ec' if lab.startswith(('short', '530', 'HV')) else 'white', zorder=0)
    ax[0].text(0.5 * (lo + hi), 16.2, lab, ha='center', fontsize=7.5, color='#555')
ax[0].set(xlabel='clock [h since 2026-06-26 00:00]', ylabel='ladder v = v_drift × kw  [µm/ns]', ylim=(15.5, 41),
          title='Drift velocity through the 6-26 night (M3 ladder, ~300 tracks/block)')
ax[0].legend(loc='center right', frameon=False)
x = np.linspace(0, 0.3, 10)
ax[1].scatter(d7, d6, s=16, color=COL['mx17_6'])
ax[1].plot(x, S['deficit_ratio']['k'] * x, color='#555', lw=1)
ax[1].set(xlabel='det7 deficit 1 − v/v_plateau', ylabel='det6 deficit 1 − v/v_plateau',
          title=f"Same time-shape: det6 = {S['deficit_ratio']['k']:.2f} × det7 (r = {S['deficit_ratio']['corr']:.2f})")
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'vt_night.png'), dpi=130); plt.close(fig)

# 2. within the overnight rows: block v / row v - 1 with the applied model (all events) and the held-out score
fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
for i, d in enumerate(DETS):
    for j, sub in enumerate(OVERNIGHT):
        a = ax[i, j]
        r = next(r for r in B if r['det'] == d and r['subrun'] == sub)
        vr = vlad(dict(x=r['all']['row']['x'], y=r['all']['row']['y']))[0]
        for p, mk in (('x', 'o'), ('y', 's')):
            tb = [b['t'] for b in r['all']['blocks']]
            vb = np.array([b[p]['v_lad'] for b in r['all']['blocks']]) / r['all']['row'][p]['v_lad'] - 1
            eb = np.array([b[p]['v_lad_err'] for b in r['all']['blocks']]) / r['all']['row'][p]['v_lad']
            a.errorbar(tb, 100 * vb, 100 * eb, fmt=mk, ms=4, color=COL[d], mfc='white' if p == 'y' else COL[d],
                       elinewidth=0.8, label=f'{p} view')
        f = M['all'][d]; tg = np.linspace(r['t_start'], r['t_end'], 100)
        cur = f['vinf'] - f['D'] * np.exp(-tg / f['tau'])
        c = next(c for c in C['rows'] if c['det'] == d and c['subrun'] == sub)
        a.plot(tg, 100 * (cur / cur.mean() - 1), color='#333', lw=1.2, label=f"applied ({C['model']})")
        a.axhline(0, color='#999', lw=0.8, ls='--')
        hs = next(h for h in H if h['det'] == d and h['subrun'] == sub)
        a.set_title(f"{NAME[d]} {sub}  (row v {vr:.1f} µm/ns)", fontsize=9.5)
        a.text(0.02, 0.04, 'held-out block rms: ' + ', '.join(
            f"{m} {100 * np.mean([hs[m][p]['eps_rms'] for p in 'xy']):.1f}%" for m in ('const', C['model'])),
            transform=a.transAxes, fontsize=7.5, color='#333')
        if i == 1:
            a.set_xlabel('clock [h]')
        if j == 0:
            a.set_ylabel('block v / row v − 1  [%]')
ax[0, 0].legend(frameon=False, fontsize=8, loc='upper left')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'vt_within_rows.png'), dpi=130); plt.close(fig)

# 3. the gas signature: total charge (gain) and charge-arrival shape u50/uend over the overnight rows, both chambers
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for d in DETS:
    for i, c in enumerate(c for c in C['rows'] if c['det'] == d and c['subrun'] in OVERNIGHT):
        g = c['gas_blocks']; t = [b['t'] for b in g]
        lab = NAME[d] if i == 0 else None
        ax[0].plot(t, [b['q_sum'] for b in g], 'o-', ms=3.5, lw=1, color=COL[d], label=lab)
        ax[1].plot(t, [b['u50_uend'] for b in g], 'o-', ms=3.5, lw=1, color=COL[d], label=lab)
ax[0].set(yscale='log', xlabel='clock [h]', ylabel='median x+y charge [ADC·samples]',
          title='Gain at fixed HV (short, longer, long_run)')
ax[1].set(xlabel='clock [h]', ylabel='median u50 / u_end', title='Charge-arrival shape (falls with attachment)')
for a in ax:
    a.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'vt_gas.png'), dpi=130); plt.close(fig)

# ---------------------------------------------------------------- report.html
e = html.escape
def tr(cells, th=False):
    t = 'th' if th else 'td'
    return '<tr>' + ''.join(f'<{t}>{c}</{t}>' for c in cells) + '</tr>'


c6, c7, dr = S['chamber']['mx17_6'], S['chamber']['mx17_7'], S['deficit_ratio']
r6 = {r['subrun']: r for r in S['rows'] if r['det'] == 'mx17_6'}
verdict = (f"Within-row v(t) is real and post-correctable. det6's ladder v rises {c6['v_first']:.1f} → "
           f"{c6['v_plateau']:.1f} µm/ns (+{c6['rise_pct']:.0f} %) from 01:00 to the 10:30 knee on 6-26, "
           f"{r6['long_run']['block_change_pct']:+.0f} % inside the golden long_run; det7 rises with the same time-shape, "
           f"{c7['v_first']:.1f} → {c7['v_plateau']:.1f} µm/ns (+{c7['rise_pct']:.0f} %), and both stop at the same clock "
           f"time. det6's deficit is {dr['k']:.2f} ± {dr['k_err']:.2f} × det7's at every time (r = {dr['corr']:.2f}): "
           "one cause common to both chambers, the gas. The correction, one smooth curve per chamber "
           f"applied as v → v·r(t) in the angle mapping, wins the held-out test (block χ² {C['heldout_chi2'][C['model']]:.0f} "
           f"against {C['heldout_chi2']['const']:.0f} for one v per row, ~{C['heldout_nblocks']} blocks) and moves det6 "
           f"angles by up to {max(r['dtheta30_max_deg'] for r in S['rows']):.1f}° at 30°.")
rows_html = ''.join(tr([
    NAME[r['det']], r['subrun'], f"{r['t_start']:.2f}–{r['t_end']:.2f}", f"{r['v_drift']:.2f}",
    f"{r['v_block_first']:.1f} → {r['v_block_last']:.1f} ({r['block_change_pct']:+.1f} %)",
    f"{r['r_min']:.3f}–{r['r_max']:.3f}", f"{r['dtheta30_max_deg']:.2f}", f"{r['dz_cathode_max_mm']:.1f}",
    f"{r['dmid_p95_mm']:.2f}",
    ' / '.join(f"{r['s68_before'][p]:.3f}→{r['s68_after'][p]:.3f}" for p in 'xy')]) for r in S['rows'])
held_html = ''.join(tr([NAME[r['det']], r['subrun']] + [
    ' / '.join(f"{100 * r['held'][m][p]['eps_rms']:.1f}" for p in 'xy') for m in ('const', 'linear', 'blocks', 'night')]
    + [' / '.join(f"{r['held'][m][p]['s68']:.3f}" for p in 'xy') for m in ('const', C['model'])]) for r in S['rows'])
ch_html = ''.join(tr([NAME[d], f"{S['chamber'][d]['v_first']:.1f} at {S['chamber'][d]['t_first']:.1f} h",
                      f"{S['chamber'][d]['v_plateau']:.1f}", f"+{S['chamber'][d]['rise_pct']:.0f} %",
                      f"{S['chamber'][d]['plateau_rows']} rows, charge ×{S['chamber'][d]['plateau_q_range'][1] / S['chamber'][d]['plateau_q_range'][0]:.0f}: "
                      f"rms {S['chamber'][d]['plateau_rms_pct']:.1f} %, slope {S['chamber'][d]['v_per_decade_q_pct']:+.1f} %/decade",
                      f"τ {S['chamber'][d]['night_fit']['tau']:.1f} ± {S['chamber'][d]['night_fit']['err'][2]:.1f} h, "
                      f"χ² {S['chamber'][d]['night_fit']['chi2']:.0f}/{S['chamber'][d]['night_fit']['ndf']}"]) for d in DETS)
page = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>v(t) within the det6/det7 6-26 rows</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;color:#222;line-height:1.45}}
table{{border-collapse:collapse;margin:.6em 0 1.4em;font-size:.88em}}td,th{{border-bottom:1px solid #ddd;padding:3px 9px;text-align:left}}
th{{background:#f4f4f1}}.verdict{{background:#f7f5ee;border-left:4px solid #eda100;padding:.7em 1em}}img{{max-width:100%}}
figcaption{{font-size:.85em;color:#555;margin-bottom:1.6em}}</style></head><body>
<h1>Drift velocity within a row: det6 / det7, 6-26 night</h1>
<p class="verdict">{e(verdict)}</p>
<h2>Headline numbers</h2>
<table>{tr(['chamber', 'first block', 'plateau v (≥ 12.4 h)', 'rise', 'gain independence (plateau rows)', 'one-exponential night fit'], True)}{ch_html}</table>
<p>Deficit test: 1 − v/v_plateau of det6 against det7's at the same clock time, {dr['n']} det6 blocks before 10.4 h:
det6 = {dr['k']:.2f} ± {dr['k_err']:.2f} × det7, χ² {dr['chi2']:.0f}/{dr['ndf']} (1 % block floor), r = {dr['corr']:.2f}.</p>
<h2>What it does to the overnight rows (applied model: {e(C['model'])}, built on all events)</h2>
<table>{tr(['det', 'row', 'clock [h]', 'row v [µm/ns]', 'first → last block v', 'r(t) range', 'max Δθ at 30° [°]', 'max Δz at cathode [mm]', 'mid-gap shift p95 [mm]', 'angle s68 x / y [°] before→after'], True)}{rows_html}</table>
<h2>Held-out test (models from even event_id, scored on odd)</h2>
<table>{tr(['det', 'row', 'const rms % x/y', 'linear', 'blocks', 'night', 's68 const x/y [°]', f's68 {C["model"]}'], True)}{held_html}</table>
<p>Summed block χ² over the overnight rows (both views, ~{C['heldout_nblocks']} blocks):
{', '.join(f'{m} {v:.1f}' for m, v in C['heldout_chi2'].items())}. linear, blocks and night are equivalent; one v per row is not.</p>
<h2>Figures</h2>
<figure><img src="figures/vt_night.png"><figcaption><b>Left:</b> the M3-ladder drift velocity (v_drift × kw, mean of the two views)
in blocks of ~300 tracks, over every det6/det7 row of 6-26 (shaded: the row groups). Thin curves: the one-exponential night fit
per chamber; dotted: the plateau. The 530→505 V and HV-scan rows change the mesh HV by up to 130 V and the gain by ~60×
without leaving the time curve. <b>Right:</b> det6's fractional deficit against det7's at the same time, before the scan.</figcaption></figure>
<figure><img src="figures/vt_within_rows.png"><figcaption>Within each overnight row: block v relative to the row v, per view
(filled x, open y), and the applied correction r(t) − 1. short_run has one block per half, so its held-out score is empty (0.0). The row ends sit up to ±9 % (det6) and ±3 % (det7) off the row's single v.</figcaption></figure>
<figure><img src="figures/vt_gas.png"><figcaption>The gas signature at fixed HV. det6's gain rises ×25 and its charge-arrival
shape u50/u_end recovers from 0.23 to 0.49 by ~4 h (an electronegative contaminant clearing: early, charge from the cathode side
is lost); det7 shows neither, yet its v rises 24 %. v keeps rising in both until ~10.5 h, long after det6's attachment and gain have settled (4–10 h: u50/u_end flat, v +20 %), so the v rise is not the attachment.</figcaption></figure>
<h2>Method</h2>
<ul><li>Clock: acq_tag stem (minute) + DREAM timestamp × 10 ns, from the pass-1 event tables (Analysis_pass1, code 6a0bb1f, read-only).</li>
<li>Estimator: the one that gives every row its kw (<code>rerun/w12/set_w0kw.py</code>) on the row's own 03_angles per-event table,
after the tan-form bow (t′ = (tan_ref + a)·S): w0 = median(w − v t′) over |t′| &lt; 0.30, kw = median((w − w0)/(v t′)) over 0.10 &lt; |t′| &lt; 0.40.
Run on the whole row it reproduces the applied kw to ≤ 1 σ on all 48 rows (golden rows: their unfolded 1.08). Per time block it gives v_lad(t) = v_drift × kw(t).</li>
<li>Correction: tan = (w − w0)/(kw · v · r(t) · S) − a, i.e. <code>wft.reco.v_rel_t</code>'s v_map['t'] (t_h, v_rel) per row; w is
independent of v (GC), so nothing is refitted. Mid-gap position p0 + 15 mm·tan and depth z = v r(t) u move with it.</li>
<li>Models: one v per row (pass 1), a line per row, a piecewise-linear through the row's blocks, one exponential per chamber over the night.
Held-out: built on even event_id, scored on odd by the residual scale of the odd half's own blocks.</li></ul>
<h2>What this does not rule out / limitations</h2>
<ul><li><b>The cause inside "the gas" is not identified.</b> The common time-shape and simultaneous knee point to the gas supply, not to
the chambers; that det6's deficit is ×{dr['k']:.1f} det7's fits a larger dose of the same contaminant in det6 (e.g. its place in the gas chain
or its own outgassing), but the bench gas order and flow on 6-26 are not recorded, and the bench gas composition is known only as
a fit (beam-gas-attachment study: bench 0.95 % H₂O, no O₂). Water drying out would raise v in both; det6's early attachment and
gain loss (clearing by ~4 h) is an additional electronegative component that det7 did not see. No new data can be taken.</li>
<li><b>The two plateaus differ</b> (det6 {c6['v_plateau']:.1f}, det7 {c7['v_plateau']:.1f} µm/ns at the same 700 V drift): a chamber-side difference
(field, gap, method scale), not addressed here.</li>
<li><b>The column-end depth does not confirm the ladder.</b> z_end = v·u_end in det6's long_run falls slightly with time under one v and
rises after the correction: the column duration is gain- and attachment-confounded (GC found the ladder moving 1.68× faster than 1/T
on det6). The ladder is gain-independent on the plateau rows (table above), so it is the measure of v here; the absolute depth scale
carries this disagreement.</li>
<li>One exponential does not describe det6's knee at ~10.5 h (χ² {c6['night_fit']['chi2']:.0f}/{c6['night_fit']['ndf']}); inside each row it is as good as a line
(held-out). It is a smooth interpolator, not a gas model; its τ is not a gas time constant.</li>
<li>The angle resolution improves modestly (det6 long_run x 1.50 → 1.46°); most of the gain is in the angle <i>scale</i> at the row
ends (up to ~2° at 30° on det6), which the s68 around the median only partly shows.</li>
<li>Rows shorter than ~2 blocks (HV-scan rows, 20 min) get the night curve; their within-row change is ≤ 1–2 %.</li>
<li>Only det6/det7 6-26. det4's golden (6-24) and its 6-23 rows, and det2 6-23, are the other GC candidates; not done here.</li></ul>
<p>Products: corrected tables (event_id, acq_tag, t_h, v_rel, x/y_tan_vt, x/y_theta_deg_vt, x/y_mid_vt, z_end_vt) and vt_map.json per row in
<code>/eos/user/d/dneff/x17rerun/vt_correction/&lt;run&gt;/&lt;subrun&gt;/&lt;det&gt;/</code>. Join on event_id (unique within each of these rows).
Scripts: <code>rerun/vt_within_rows/</code>.</p>
</body></html>"""
open(os.path.join(HERE, 'report.html'), 'w').write(page)
print(json.dumps({k: v for k, v in S.items() if k != 'rows'}, indent=1, default=float))
for r in S['rows']:
    print(NAME[r['det']], r['subrun'], f"{r['block_change_pct']:+.1f}%  r {r['r_min']:.3f}-{r['r_max']:.3f} "
          f"dth30 {r['dtheta30_max_deg']:.2f} dz {r['dz_cathode_max_mm']:.1f} mm")
