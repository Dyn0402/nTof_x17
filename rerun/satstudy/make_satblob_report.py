#!/usr/bin/env python3
"""make_satblob_report.py -- report.html for satblob.py (saturated delta blobs: per-channel rail, bottom rail, track +
compact-blob model), generated from satblob.json / satblob_scored.parquet / satblob_blob.parquet.

    ../../.venv/bin/python satblob.py score && ../../.venv/bin/python make_satblob_report.py
-> ~/x17/cosmic_bench/satstudy/satblob_report/{report.html, figures/*.png}
"""
import html, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import satfit as S  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(S.OUT, 'satblob_report'); FIG = os.path.join(OUT, 'figures')

ARMS = [  # arm, label, what changes
    ('prod', 'production', 'SAT = 3550 censored, everything else fitted (satfit.py re-fit)'),
    ('rail', 'per-channel top rail', 'censor at 4095 − ped − CM instead of 3550'),
    ('bot', 'bottom rail censored', 'SAT 3550 + raw ADC 0 samples censored (one-sided)'),
    ('railbot', 'both rails', 'rail + bot'),
    ('mt_prod', 'multi-start, track only', 'own optimiser, every start, production mask'),
    ('mt_railbot', 'multi-start, track only, both rails', ''),
    ('mt_railbot@0.05', 'multi-start, track only, both rails, 5 % model error', 'σ² = noise² + (0.05 W)²'),
    ('blob_prod', 'track + blob (ΔBIC > 0)', 'production mask'),
    ('blob_railbot', 'track + blob, both rails', ''),
    ('blob_railbot@0.05', 'track + blob, both rails, 5 % model error', ''),
]


def pct(x):
    return '—' if x is None or x != x else f'{100 * x:.1f} %'


def main():
    os.makedirs(FIG, exist_ok=True)
    J = json.load(open(os.path.join(S.OUT, 'satblob.json')))
    F = pd.read_parquet(os.path.join(S.OUT, 'satblob_scored.parquet'))
    B = pd.read_parquet(os.path.join(S.OUT, 'satblob_blob.parquet'))
    arms = [a for a in ARMS if f'sat|{a[0]}' in J]

    # ---- figure 1: agreement per arm
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(arms))
    ax.barh(y, [100 * J[f'sat|{a}']['agree'] for a, *_ in arms], color='#3b6fb6', label='350 saturated views')
    ax.scatter([100 * J[f'ctrl|{a}']['agree'] if f'ctrl|{a}' in J else np.nan for a, *_ in arms], y, color='#d1495b',
               zorder=3, label='150 controls')
    ax.set_yticks(y, [l for _, l, _ in arms], fontsize=8); ax.invert_yaxis(); ax.set_xlim(50, 100)
    ax.axvline(100 * J['sat|prod']['agree'], color='grey', lw=0.8, ls=':')
    ax.set_xlabel('views agreeing with M3 (|Δθ| ≤ 5°, |Δp0| ≤ 3 mm) [%]'); ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'agree.png'), dpi=120); plt.close(fig)

    # ---- figure 2: threshold scan per blob mask
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for mk, col in (('prod', '#888'), ('railbot', '#3b6fb6'), ('railbot@0.05', '#2a9d8f')):
        sc = J.get(f'scan|{mk}')
        if not sc:
            continue
        ax.plot([100 * s['ctrl_added'] for s in sc], [100 * s['sat_agree'] for s in sc], 'o-', color=col,
                label=f'blob, {mk} mask')
        for s in sc:
            ax.annotate(f"{s['thr']}", (100 * s['ctrl_added'], 100 * s['sat_agree']), fontsize=6,
                        textcoords='offset points', xytext=(3, 3))
    ax.axhline(100 * J['sat|prod']['agree'], color='grey', ls=':', lw=0.8, label='production')
    ax.set_xlabel('controls given a blob [%] (false-blob rate)'); ax.set_ylabel('saturated views agreeing [%]')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'scan.png'), dpi=120); plt.close(fig)

    # ---- figure 3: angle residual distributions on the saturated views
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bins = np.linspace(-30, 30, 61)
    for a, col in (('prod', '#888'), ('railbot', '#e9c46a'), ('blob_railbot', '#3b6fb6'),
                   ('blob_railbot@0.05', '#2a9d8f')):
        h = F[(F.arm == a) & (F.group == 'sat')]
        if len(h):
            ax.hist(np.clip(h.dl, -30, 30), bins, histtype='step', color=col, lw=1.4, label=a)
    ax.set_xlabel('fitted − M3 angle, ladder-scaled [deg]'); ax.set_ylabel('saturated views'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'dl.png'), dpi=120); plt.close(fig)

    # ---- figure 4: example failures with the blob fit drawn
    import appendix_figs as A
    A.md.wf(28537); A.md.row(28537)
    Bf = B[(B['mask'] == 'railbot') & (B.group == 'sat')].merge(
        F[F.arm == 'blob_railbot'][['event_id', 'view', 'agree']], on=['event_id', 'view'])
    Bf = Bf[~Bf.agree & (Bf.dbic > 0) & Bf.event_id.isin(list(A.md._WF))]
    ex = Bf.sample(min(3, len(Bf)), random_state=5) if len(Bf) else Bf
    if len(ex):
        fig, axs = plt.subplots(1, len(ex), figsize=(3.3 * len(ex), 3.3), squeeze=False)
        for ax, (_, r) in zip(axs[0], ex.iterrows()):
            A.evd_mpl(ax, int(r.event_id), r['view'], 14, fit=True, mark=True)
            tt = np.array([r.t0_TB, r.t0_TB + 30 / A.md.V_D])
            ax.plot(r.p0_TB + r.w_TB * (tt - tt[0]), tt, color='#ffb000', lw=1.3, ls='-.', label='track + blob fit')
            ax.plot([r.pb], [r.t0_TB + (r.kb + 0.5) * 60], marker='o', ms=9, mfc='none', mec='#ffb000', mew=1.5,
                    label='blob')
            ax.set_title(f"event {int(r.event_id)} {r['view']}", fontsize=8)
        axs[0][0].legend(fontsize=6, loc='lower left')
        fig.tight_layout(); fig.savefig(os.path.join(FIG, 'examples.png'), dpi=120); plt.close(fig)
    pinrows = []
    for mk in ('prod', 'railbot'):
        for nm in ('fail', 'agree'):
            r = J.get(f'pinned|{mk}|{nm}')
            if r:
                pinrows.append(f"<tr><td>{mk}</td><td>{'disagree with M3' if nm == 'fail' else 'agree'}</td><td>{r['n']}</td>"
                               f"<td>{pct(r['ref_worse_TB'])}</td><td>{r['ratio_TB_med']:.2f}</td>"
                               f"<td>{pct(r['blob_gain_free_med'])}</td><td>{pct(r['blob_gain_ref_med'])}</td></tr>")

    # ---- tables
    def arm_rows(g):
        out = []
        for a, l, w in arms:
            k = f'{g}|{a}'
            if k not in J:
                continue
            r = J[k]; sw = J.get(f'switch|{g}|{a}', {})
            out.append(f"<tr><td>{html.escape(l)}<div class=sub>{html.escape(w)}</div></td><td>{r['n']}</td>"
                       f"<td><b>{pct(r['agree'])}</b></td><td>{sw.get('gained', '')}</td><td>{sw.get('lost', '')}</td>"
                       f"<td>{r['dl_med']:+.2f}°</td><td>{r['dl_sigma68']:.2f}°</td><td>{pct(r['steeper_frac'])}</td>"
                       f"<td>{pct(r['far_steeper']) if r['far_steeper'] is not None else '—'}</td></tr>")
        return '\n'.join(out)

    hdr = ('<tr><th>arm</th><th>n</th><th>agree</th><th>gained</th><th>lost</th><th>median Δθ</th>'
           '<th>68 % |Δθ|</th><th>steeper than M3</th><th>…of the failures</th></tr>')
    blobrows = []
    for mk in ('prod', 'railbot', 'railbot@0.05'):
        for g in ('sat', 'ctrl'):
            r = J.get(f'blobrate|{g}|{mk}')
            if r:
                blobrows.append(f"<tr><td>{mk}</td><td>{g}</td><td>{r['n']}</td><td>{pct(r['added'])}</td>"
                                f"<td>{r['dbic_q']['10']:.0f} / {r['dbic_q']['50']:.0f} / {r['dbic_q']['90']:.0f}</td>"
                                f"<td>{pct(r['qblob_frac_med']) if r['qblob_frac_med'] is not None else '—'}</td></tr>")
    scanrows = []
    for mk in ('prod', 'railbot', 'railbot@0.05'):
        for s in J.get(f'scan|{mk}', []):
            scanrows.append(f"<tr><td>{mk}</td><td>{s['thr']}</td><td>{pct(s['sat_agree'])}</td>"
                            f"<td>{pct(s['ctrl_agree'])}</td><td>{pct(s['sat_added'])}</td><td>{pct(s['ctrl_added'])}</td></tr>")
    # rail facts
    rf = J['railfacts']
    p = J['sat|prod']['agree']
    g = lambda k: J.get(k, {})
    sw = lambda a: f"+{g(f'switch|sat|{a}').get('gained', 0)} / −{g(f'switch|sat|{a}').get('lost', 0)}"
    pin = g('pinned|railbot|fail')
    verdict = (
        f"<b>The track + compact-blob model does not recover the saturated failures.</b> On the saturated views, "
        f"agreement with M3 goes {pct(p)} (production) → {pct(g('sat|blob_railbot').get('agree'))} (track + blob, both "
        f"rails); most of that step is the multi-start alone ({pct(g('sat|mt_prod').get('agree'))}, {sw('mt_prod')} "
        f"views), the blob itself moves {sw('blob_railbot')} views against production. With ΔBIC &gt; 0 a blob is "
        f"added to {pct(g('blobrate|ctrl|railbot').get('added'))} of unsaturated controls (model error, not blobs); "
        f"no ΔBIC threshold separates the two groups usefully (scan below). A 5 % model-error term makes it worse "
        f"({pct(g('sat|blob_railbot@0.05').get('agree'))}). "
        f"The rail fixes on their own, production fit: per-channel top rail {sw('rail')}, bottom rail censored "
        f"{sw('bot')}, both {sw('railbot')} ({pct(g('sat|railbot').get('agree'))}); controls unchanged. "
        + (f"Pinned to the M3 line, the same model has a higher χ² than its free optimum in "
           f"{pct(pin.get('ref_worse_TB'))} of the still-failing views (median ×{pin.get('ratio_TB_med', float('nan')):.1f}, "
           f"against ×{g('pinned|railbot|agree').get('ratio_TB_med', float('nan')):.1f} on views that DO agree, where it "
           f"only measures how stiff a bright-pulse χ² is to M3's own error): the failure is the model, not the "
           f"optimiser. The displays show why: the saturated deposit spans 5–10 strips and several hundred ns, a short "
           f"δ TRACK rather than a point, so one compact deposit cannot absorb it."
           if pin else ''))
    css = ('body{font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222;'
           'background:#fff}table{border-collapse:collapse;margin:8px 0 18px;font-size:13px}td,th{border:1px solid #ccc;'
           'padding:3px 7px;text-align:right}td:first-child{text-align:left}.sub{color:#777;font-size:11px}'
           'img{max-width:100%}.verdict{background:#eef4fb;border-left:4px solid #3b6fb6;padding:10px 14px}'
           '@media (prefers-color-scheme: dark){body{background:#16181d;color:#ddd}td,th{border-color:#444}'
           '.verdict{background:#1f2a3a}}')
    doc = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Saturated delta blobs</title><style>{css}</style></head><body>
<h1>Saturated δ blobs on det3: rails and a track + blob model</h1>
<p class=sub>Generated by rerun/satstudy/make_satblob_report.py from satblob.py outputs. Golden row sat_det3 (6-27, 490 V),
phys bundle, pre-pass final_phys table. Same 350 saturated views (≥ 3 saturated hits) + 150 unsaturated controls as
satfit.py. Agreement with M3 = the C22 rule (|Δθ| ≤ 5° ladder-scaled, |p0 − M3| ≤ 3 mm).</p>
<div class=verdict>{verdict}</div>
<h2>Agreement per arm</h2><img src="figures/agree.png" alt="agreement per arm">
<h3>Saturated views</h3><table>{hdr}{arm_rows('sat')}</table>
<h3>Controls</h3><table>{hdr}{arm_rows('ctrl')}</table>
<p class=sub>gained / lost: views that switch from disagree to agree (and back) against production on the same view.
Arms prod/rail/bot/railbot are the production fit (wft.reco.fit_plane, pivot start) with only the censor mask changed;
mt_* and blob_* are the analysis-side optimiser (satblob_core.py): every start of the view, track-only (mt) or
track + one compact deposit kept when ΔBIC = n ln(χ²_T/χ²_TB) − 5 ln n &gt; 0.</p>
<h2>The rail facts</h2>
<ul><li>Raw strip maxima above 3800 sit exactly at <b>4095</b> (13 086 strips): the rail is 4095 − pedestal − CM per
channel and sample (3650–3900 after subtraction), rebuilt from the pedestals and the block CM of every event (match to the
stored waveforms ≤ 2 ADC, float16).</li>
<li>In the {rf['n_views']} saturated windows: {rf['both']:,} samples are censored by both rules;
{rf['prod_only']:,} are ≥ 3550 but below their own rail (real measurements production throws away) and
{rf['rail_only']:,} sit ON the rail but below 3550 (a clipped plateau the common mode pulled down: production fits them
as exact data).</li>
<li>Bottom rail (raw ADC 0): {rf['n_bot']:,} samples in {rf['views_with_bot']} of {rf['n_views']} saturated windows
({pct(rf['views_with_bot'] / max(rf['n_views'], 1))}), fitted as data by production.</li></ul>
<h2>Blob model selection</h2><table><tr><th>mask</th><th>group</th><th>n</th><th>blob added</th><th>ΔBIC p10/50/90</th>
<th>blob share of charge</th></tr>{''.join(blobrows)}</table>
<img src="figures/scan.png" alt="threshold scan">
<table><tr><th>mask</th><th>ΔBIC &gt;</th><th>sat agree</th><th>ctrl agree</th><th>sat with blob</th><th>ctrl with blob</th></tr>
{''.join(scanrows)}</table>
<h2>Is it the optimiser or the model? Track pinned to M3</h2>
<p>The same fit with the track fixed to the M3 line (p0, angle from the reference; t0, the depth profile and the blob
free). If the pinned χ² is LOWER than the free optimum the optimiser missed the right basin; if it is higher, the
model itself prefers the wrong line.</p>
<table><tr><th>mask</th><th>views</th><th>n</th><th>pinned χ² higher than free</th><th>median χ²_pinned / χ²_free</th>
<th>median χ² drop from the blob, free</th><th>… pinned</th></tr>{''.join(pinrows)}</table>
<img src="figures/examples.png" alt="example failures with the blob fit">
<p class=sub>Three random saturated views that still disagree after the track + blob fit (both rails, ΔBIC &gt; 0):
M3 dashed red, production fit white, track + blob fit orange dash-dot, the blob's position and depth circled.
Outlined: samples ≥ 3550 (red) and on the bottom rail (blue).</p>
<h2>Angle residuals</h2><img src="figures/dl.png" alt="angle residuals">
<h2>What this does not rule out</h2>
<ul>
<li>One compact deposit only (3 depth bins, point-like transverse). A δ that travels several mm sideways, or two blobs,
is outside the model.</li>
<li>The bench χ²/dof is model-error dominated (~35 on clean tracks, ~500 on saturated ones), so the Gaussian BIC is not a
calibrated test: the false-blob rate on controls is the honest measure, and the threshold should be set from it.</li>
<li>350 + 150 views, one chamber (det3), one run condition; the fits do not include the column guard (WFT_COL_GUARD) of
the pass, as satfit.py's prod arm.</li>
<li>Agreement is against M3, which is itself wrong for ~0.8 % of tracks (bent between the doublets).</li>
<li>Event-level detection (both views, n_dropped) is not re-derived here: these are per-view numbers.</li>
</ul></body></html>"""
    open(os.path.join(OUT, 'report.html'), 'w').write(doc)
    print('wrote', os.path.join(OUT, 'report.html'))


if __name__ == '__main__':
    main()
