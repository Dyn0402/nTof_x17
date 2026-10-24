#!/usr/bin/env python3
"""make_report.py [out_dir] -- report.html + figures for the starting-angle / v-consistency study (CHECKS C18/C19),
built from the result files in ~/x17/cosmic_bench/vladder (seedfit + vladder outputs)."""
import glob, json, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

L = os.path.expanduser('~/x17/cosmic_bench/vladder/')
OUT = sys.argv[1] if len(sys.argv) > 1 else L + 'report'
FIG = os.path.join(OUT, 'figures'); os.makedirs(FIG, exist_ok=True)
DETS = ['det3', 'det2', 'det4', 'det6', 'det7']
TIMING = dict(det2=(39.3, 1.2), det3=(35.8, 0.6), det4=(33.0, 0.7), det6=(25.9, 0.8), det7=(36.8, 1.1))
GAP = dict(det2=31.12, det3=28.08, det4=30.48, det7=29.44, det6=None)
C = dict(bare='#9aa0a6', reco='#4c78a8', pivot='#e45756', reco_2s='#72b7b2', r06='#f58518')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def ladder(tr, w, nb=100):
    m = np.isfinite(w)
    for _ in range(6):
        s, c = np.polyfit(tr[m], w[m], 1); r = w - s * tr - c
        m = np.abs(r) < 3 * np.percentile(np.abs(r[m]), 68.27)
    rng = np.random.default_rng(0); idx = np.flatnonzero(m)
    e = np.std([np.polyfit(tr[j], w[j], 1)[0] for j in (rng.choice(idx, len(idx)) for _ in range(nb))])
    th = np.degrees(np.arctan((w - c) / s)) - np.degrees(np.arctan(tr)); far = np.abs(th) > 5
    s68 = np.percentile(np.abs(th[~far] - np.median(th[~far])), 68.27)
    return dict(v=float(s), err=float(e), s68=float(s68), far=float(far.mean()))


def rows(f, p, names):
    E = json.load(open(f))['events']
    return [e[p] for e in E.values() if p in e and all(n in e[p] and 'chi2' in e[p][n] for n in names)]


# ---- fitter comparison (sx_<det>_reco.json carries prod, pivot, reco, reco_shear, reco_2s, pivot_fast)
FN = [('prod', 'bare start'), ('reco', 'production reco'), ('pivot_fast', 'pivot (new)'), ('reco_2s', 'reco two-sided')]
fitcmp = {}
for d in DETS:
    for p in 'xy':
        R = rows(L + f'sx_{d}_reco.json', p, [n for n, _ in FN])
        tr = np.array([r['tr'] for r in R]); Cm = np.array([[r[n]['chi2'] for n, _ in FN] for r in R])
        wb = np.array([r[FN[int(np.argmin(c))][0]]['w'] for r, c in zip(R, Cm)])
        for n, lab in FN:
            w = np.array([r[n]['w'] for r in R])
            fitcmp[(d, p, n)] = dict(wrong=float(np.mean(np.abs(w - wb) > 1.0)), sec=float(np.mean([r[n]['sec'] for r in R])),
                                     **ladder(tr, w))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, p in zip(axs, 'xy'):
    x = np.arange(len(DETS)); bw = 0.2
    for i, (n, lab) in enumerate(FN):
        ax.bar(x + (i - 1.5) * bw, [100 * fitcmp[(d, p, n)]['wrong'] for d in DETS], bw, label=lab,
               color=[C['bare'], C['reco'], C['pivot'], C['reco_2s']][i])
    ax.set_xticks(x, DETS); ax.set_title(f'{p.upper()} view'); ax.set_ylabel('fits in the wrong w basin (%)' if p == 'x' else '')
axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f'{FIG}/fitter_wrong_basin.png', dpi=130); plt.close(fig)

# ---- ladder vs timing (pivot on all 1300 events; bare/reco on 400; old kernel pivot on 400)
LM = {d: json.load(open(L + f'ladder_map_{d}.json')) for d in DETS}
old = {}
for d in ('det3', 'det4', 'det7'):
    for p in 'xy':
        R = rows(L + f'sx_{d}_r06.json', p, ['pivot'])
        old[(d, p)] = ladder(np.array([r['tr'] for r in R]), np.array([r['pivot']['w'] for r in R]))
        old[(d, p)]['chi2'] = float(np.median([r['pivot']['chi2'] for r in R]))
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, p in zip(axs, 'xy'):
    for i, d in enumerate(DETS):
        vt, ve = TIMING[d]
        ax.add_patch(plt.Rectangle((i - 0.4, -100 * ve / vt), 0.8, 200 * ve / vt, color='#dddddd', zorder=0))
        pts = [('bare', fitcmp[(d, p, 'prod')], -0.24), ('reco', fitcmp[(d, p, 'reco')], -0.08),
               ('pivot', LM[d][p]['all'], 0.08)]
        if (d, p) in old:
            pts.append(('r06', old[(d, p)], 0.24))
        for k, r, dx in pts:
            ax.errorbar(i + dx, 100 * (r['v'] / vt - 1), 100 * r['err'] / vt, fmt='o', color=C[k], ms=5,
                        label={'bare': 'bare start (RC)', 'reco': 'production reco (RC)', 'pivot': 'pivot, 1300 ev (RC)',
                               'r06': 'pivot, old lp/delay kernel'}[k] if i == 2 else None)
    ax.axhline(0, color='k', lw=0.6); ax.set_xticks(range(len(DETS)), DETS)
    ax.set_title(f'{p.upper()} view'); ax.set_ylabel('reco ladder v vs timing v (%)' if p == 'x' else '')
axs[0].legend(frameon=False, fontsize=8, loc='lower left')
fig.tight_layout(); fig.savefig(f'{FIG}/ladder_vs_timing.png', dpi=130); plt.close(fig)

# ---- regional maps: ladder v (X, Y) and stacked T
TM = {d: {(r['ix'], r['iy']): r for r in json.load(open(L + f'tmap_{d}.json'))} for d in DETS}
fig, axs = plt.subplots(3, 5, figsize=(13, 7.4))
for j, d in enumerate(DETS):
    for i, (lab, getter) in enumerate((('ladder v, X', lambda g: g['v']), ('ladder v, Y', None), ('stacked T (ns)', None))):
        ax = axs[i, j]; A = np.full((3, 3), np.nan)
        for ix in range(3):
            for iy in range(3):
                if i < 2:
                    g = [g for g in LM[d]['xy'[i]]['grid'] if g['ix'] == ix and g['iy'] == iy][0]
                    A[2 - iy, ix] = g['v']
                elif (ix, iy) in TM[d]:
                    A[2 - iy, ix] = TM[d][(ix, iy)]['T']
        if i < 2:
            vm = np.nanmean(A); im = ax.imshow(A, cmap='RdBu_r', vmin=vm * 0.88, vmax=vm * 1.12)
        else:
            vm = np.nanmean(A); im = ax.imshow(A, cmap='RdBu', vmin=vm * 0.8, vmax=vm * 1.2)
        for (r, c), val in np.ndenumerate(A):
            if np.isfinite(val):
                ax.text(c, r, f'{val:.1f}' if i < 2 else f'{val:.0f}', ha='center', va='center', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(lab)
        if i == 0:
            ax.set_title(d)
fig.suptitle('Per-region drift velocity (3 x 3 regions of 133 mm; top row = high y). Red = fast for v, slow (long T) for T.', fontsize=10)
fig.tight_layout(); fig.savefig(f'{FIG}/regional_maps.png', dpi=120); plt.close(fig)

# ---- det7 X: failures vs position, adjacency test per connector
ev7 = pickle.load(open(L + 'det7/cache.pkl', 'rb'))
R = {}
for f in ('sf_det7_400.json', 'sp_det7_rest.json'):
    R.update(json.load(open(L + f))['events'])
xs, far7 = [], []
for k, e in R.items():
    r = e['x']; th = np.degrees(np.arctan(r['pivot']['w'] / LM['det7']['x']['all']['v'])) - np.degrees(np.arctan(r['tr']))
    xs.append(ev7[int(k)]['ref_mesh_x']); far7.append(abs(th) > 5)
xs, far7 = np.array(xs), np.array(far7)
ed = np.arange(0, 401, 25)
fr = [far7[(xs >= a) & (xs < b)].mean() * 100 if ((xs >= a) & (xs < b)).sum() >= 8 else np.nan for a, b in zip(ed[:-1], ed[1:])]


def adjacency(det):
    ev = pickle.load(open(L + det + '/cache.pkl', 'rb'))
    pos = {}; rws = []
    for e in ev.values():
        P = e['x']; W = np.asarray(P['W'], float)
        if W.max() >= 3500:
            continue
        pk = W.max(1); nz = np.maximum(np.asarray(P['noise']), 3)
        rws.append({int(c): (q if q > 3 * n else 0.0) for c, q, n in zip(P['ch'], pk, nz)})
        for c, q in zip(P['ch'], P['pos']):
            pos[int(c)] = float(q)
    chs = np.array(sorted(pos)); idx = {c: i for i, c in enumerate(chs)}; n = len(chs)
    num = np.zeros((n, n)); ni = np.zeros((n, n)); nj = np.zeros((n, n))
    for dd in rws:
        ks = np.array([idx[c] for c in dd]); a = np.array([dd[c] for c in dd])
        num[np.ix_(ks, ks)] += np.outer(a, a); sq = a ** 2
        ni[np.ix_(ks, ks)] += sq[:, None]; nj[np.ix_(ks, ks)] += sq[None, :]
    S = num / np.sqrt(np.maximum(ni * nj, 1e-9)); np.fill_diagonal(S, 0)
    out = []
    for blk in range(8):
        m = np.flatnonzero((chs >= 64 * blk) & (chs < 64 * blk + 64)); ok = tot = 0
        for i in m:
            for j in np.argsort(S[i])[::-1][:2]:
                if S[i, j] > 0:
                    tot += 1; ok += round(abs(pos[chs[j]] - pos[chs[i]]) / 0.78) == 1
        out.append(100 * ok / tot if tot >= 20 else np.nan)
    return out


adj = {d: adjacency(d) for d in ('det3', 'det7')}
fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
axs[0].bar(ed[:-1] + 12.5, fr, 22, color=C['pivot'])
for b in range(1, 8):
    axs[0].axvline(50 * b, color='#bbbbbb', lw=0.6, ls=':')
axs[0].set_xlabel('M3 x at the mesh (mm); dotted = 64-channel connector edges'); axs[0].set_ylabel('det7 X fits > 5 deg off (%)')
x = np.arange(8)
axs[1].bar(x - 0.2, adj['det3'], 0.4, label='det3 X', color=C['reco']); axs[1].bar(x + 0.2, adj['det7'], 0.4, label='det7 X', color=C['r06'])
axs[1].set_xticks(x, [f'c{i}' for i in x]); axs[1].set_ylabel('most similar channels that are\nmapped neighbours (%)')
axs[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f'{FIG}/det7_connectors.png', dpi=130); plt.close(fig)

# ---- one landscape
Ld = json.load(open(L + 'land_det3_x.json'))
fig, axs = plt.subplots(1, 3, figsize=(11, 3.0))
for ax, k in zip(axs, list(Ld)[2:5]):
    d = Ld[k]; ws = np.array(d['ws']); p0s = np.array(d['p0s']); M = np.array(d['M'])
    im = ax.imshow(np.log10(M - M.min() + 1).T, origin='lower', aspect='auto', cmap='viridis',
                   extent=[ws[0], ws[-1], p0s[0] - d['prod']['p0'], p0s[-1] - d['prod']['p0']])
    ax.plot(d['prod']['w'], 0, 'wx', ms=8, label='bare fit'); ax.plot(d['seed']['w'], d['seed']['p0'] - d['prod']['p0'], 'r+', ms=10, label='lowest found')
    ax.axvline(d['prod']['tr'] * 35.8, color='w', lw=0.6, ls='--')
    ax.set_xlabel('w (um/ns)'); ax.set_ylabel('p0 - bare fit p0 (mm)')
axs[0].legend(fontsize=7, loc='upper left')
fig.suptitle('det3 X: log chi2 over (w, p0) at the trigger t0, three events the bare start missed (dashed: M3 angle x v)', fontsize=9)
fig.tight_layout(); fig.savefig(f'{FIG}/landscape.png', dpi=120); plt.close(fig)

# ---- gaps
gap_rows = []
k3 = None
for d in DETS:
    T = json.load(open(L + f'tmap_{d}.json'))
    allT = None
for d in DETS:
    pass

# ---- HTML
def tr_(cells, th=False):
    t = 'th' if th else 'td'
    return '<tr>' + ''.join(f'<{t}>{c}</{t}>' for c in cells) + '</tr>'


fit_tab = [tr_(['fitter'] + [f'{d} X' for d in DETS] + ['time / plane'], True)]
for n, lab in FN:
    fit_tab.append(tr_([lab] + [f"{100 * fitcmp[(d, 'x', n)]['wrong']:.1f} % · {fitcmp[(d, 'x', n)]['s68']:.2f}°" for d in DETS]
                       + [f"{min(fitcmp[(d, 'x', n)]['sec'] for d in DETS):.1f}–{max(fitcmp[(d, 'x', n)]['sec'] for d in DETS):.1f} s"]))
lad_tab = [tr_(['chamber', 'timing v', 'reco ladder X (pivot, RC)', 'reco ladder Y', 'old kernel X / Y', 'median χ² RC vs old (X / Y)'], True)]
for d in DETS:
    vt, ve = TIMING[d]
    a, b = LM[d]['x']['all'], LM[d]['y']['all']
    o = (f"{old[(d, 'x')]['v']:.1f} / {old[(d, 'y')]['v']:.1f}" if (d, 'x') in old else '–')
    rc_chi = (f"{np.median([r['pivot']['chi2'] for r in rows(L + f'sf_{d}_400.json', 'x', ['pivot'])]) / 1e3:.1f}k / "
              f"{np.median([r['pivot']['chi2'] for r in rows(L + f'sf_{d}_400.json', 'y', ['pivot'])]) / 1e3:.1f}k")
    oc = (f" vs {old[(d, 'x')]['chi2'] / 1e3:.1f}k / {old[(d, 'y')]['chi2'] / 1e3:.1f}k" if (d, 'x') in old else '')
    lad_tab.append(tr_([d, f'{vt} ± {ve}', f"{a['v']:.2f} ± {a['err']:.2f} ({100 * (a['v'] / vt - 1):+.1f} %)",
                        f"{b['v']:.2f} ± {b['err']:.2f} ({100 * (b['v'] / vt - 1):+.1f} %)", o, rc_chi + oc]))

html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drift velocity consistency</title>
<style>
:root{{--bg:#fff;--fg:#1d1f23;--mut:#5f6368;--line:#e3e5e8;--acc:#c0392b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16181b;--fg:#e8eaed;--mut:#9aa0a6;--line:#33363b;--acc:#ff8a7a}} img{{background:#fff;border-radius:4px}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,sans-serif;max-width:980px;margin:0 auto;padding:16px}}
h1{{font-size:1.5rem;margin:.4em 0}} h2{{font-size:1.15rem;margin-top:1.8em;border-bottom:1px solid var(--line);padding-bottom:.2em}}
.verdict{{border-left:4px solid var(--acc);padding:.5em 1em;background:color-mix(in srgb,var(--acc) 8%,transparent)}}
table{{border-collapse:collapse;font-size:.88rem;margin:.6em 0;display:block;overflow-x:auto}} td,th{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;white-space:nowrap}}
img{{max-width:100%}} figcaption{{color:var(--mut);font-size:.85rem;margin-bottom:1em}} code{{font-size:.85em}}
</style></head><body>
<h1>Drift velocity: why the estimators disagree, and the starting-angle fix</h1>
<p style="color:var(--mut)">MX17 bench re-run, 2026-10-24 · CHECKS C18/C19 · generated by <code>rerun/seedfit/make_report.py</code></p>
<div class="verdict"><b>Verdict.</b> det3 is consistent: the reconstruction's own drift velocity is
{LM['det3']['x']['all']['v']:.2f} ± {LM['det3']['x']['all']['err']:.2f} (X) and {LM['det3']['y']['all']['v']:.2f} ± {LM['det3']['y']['all']['err']:.2f} (Y) µm/ns
against the timing v of 35.8 ± 0.6. The calibration fit's free v is not a measurement (χ² flat along v). On det2/4/6/7 four
measured effects stop the estimators closing: no dead-strip masks in any bench bundle (small), a scrambled channel order in
det7 X connectors 0/1/3, a drift velocity that varies across det4/6/7, and an RC kernel that fits det4/det7 much worse than
the old one. The starting-angle problem has an opt-in fix in <code>wft.reco</code> (<code>WFT_PIVOT_START=1</code>).</div>

<h2>Three different "v"s</h2>
<ul><li><b>Timing v</b> = gap / drift time. Gap from the charge extent of inclined tracks (no timing), T from head-on stacks. Kernel-free.</li>
<li><b>Reconstruction (ladder) v</b> = slope of the fitted transverse speed w against the M3 angle. The per-event fit never uses v;
the bundle's v × kw is only an angle scale.</li>
<li><b>Calibration fit with v free</b>: χ² nearly flat along v, so its minimum is set by how the M3 reference pins the line. Not a measurement.</li></ul>

<h2>Reconstruction v against timing v</h2>
<table>{''.join(lad_tab)}</table>
<figure><img src="figures/ladder_vs_timing.png" alt="ladder vs timing"><figcaption>Grey band: timing v error. The bare start (last night's evaluation tool) reads X 1–4 % low;
production and pivot agree. Where the old lp/delay kernel fits better (det4, det7: half to three quarters of the χ²), its ladder lands on timing.</figcaption></figure>

<h2>The starting-angle problem</h2>
<p>At the right mesh position p0, χ² along w has one narrow well (3–5 µm/ns). But p0 is the mesh <i>end</i> of the track, up to
w × (half the column) ≈ 5 mm from the charge, so a fit started at w = 0 on the brightest strip sits in a diagonal valley beside a
broad w ≈ 0 shelf. Two more defects: the RC kernel's χ² is a 5 ns staircase in t0, and the ±5 ns t0 prior is invisible against χ² of 10⁴–10⁶.</p>
<figure><img src="figures/landscape.png" alt="chi2 landscapes"><figcaption>χ² over (w, p0) for three det3 X events the bare start missed.</figcaption></figure>
<p><b>Pivot start:</b> a straight track passes through its own charge centroid (x<sub>c</sub>, t<sub>c</sub>), model-free (t<sub>c</sub> = charge-weighted
sample time minus the impulse response's own time centroid, since convolution adds means). Scan w with the line forced through it and refine the two best wells.</p>
<table>{''.join(fit_tab)}</table>
<figure><img src="figures/fitter_wrong_basin.png" alt="wrong basin"><figcaption>Share of fits whose w is more than 1 µm/ns from the lowest-χ² solution any fitter found (400 held-out events per chamber).
Table cells: wrong-basin share · angle s68. Two-sided reaches lower χ² but costs 3–5× with no resolution gain.</figcaption></figure>

<h2>Why det2/4/6/7 do not close</h2>
<ol><li><b>No dead masks / gains in the bench bundles.</b> Derived from the data (<code>strip_health.py</code>): masking moves the ladder ≤ 0.5 %; gains help det2/det4 X by 1–2 % but cost det3 resolution.</li>
<li><b>det7 X connectors 0, 1, 3 are scrambled.</b> True neighbours sit 5/10/15 channels apart; positions survive but angles are lost there.</li>
<li><b>v varies over det4/6/7.</b> X and Y ladder maps (independent readouts) agree, and the stacked drift time anti-correlates with them.</li>
<li><b>RC kernel mismatch on det4/det7</b> (table above). det6's timing v assumed a 30 mm gap; ladder × T implies ≈ 27.5–28 mm, i.e. v ≈ 24, as the ladder reads.</li></ol>
<figure><img src="figures/det7_connectors.png" alt="det7 connectors"><figcaption>Left: det7 X failures by position, stripes along whole connectors. Right: adjacency test — a channel's two most similar channels are its mapped neighbours on det3 everywhere, but not on det7 connectors 0, 1, 3.</figcaption></figure>
<figure><img src="figures/regional_maps.png" alt="regional maps"><figcaption>Ladder v (X, Y) and stacked drift time T per region. Same pattern in both views; T long where v is low.</figcaption></figure>

<h2>What this does not rule out</h2>
<ul><li>The timing v carries 2–4 % (gap, stack-end definition); det6's gap is unmeasured and det7's long/longer runs differ by 12 %.</li>
<li>The kernel comparison uses the old bundles as delivered; which kernel is right on det4/det7 is judged by χ² only.</li>
<li>det7 X's channel scramble is identified but not mapped (≈150 events per connector); its pattern is not yet understood.</li>
<li>Regional v uses 3 × 3 regions with 60–280 events each; the per-region timing T comes from head-on tracks only.</li>
<li>All of this is the bench June data with RC it0 bundles; the pass will use final bundles.</li></ul>
</body></html>'''
open(os.path.join(OUT, 'report.html'), 'w').write(html)
print('wrote', OUT)
