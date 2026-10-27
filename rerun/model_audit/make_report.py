#!/usr/bin/env python3
"""make_report.py [--root ~/x17/cosmic_bench/model_audit] -- the model-audit report (RERUN_PLAN §12 B4), built from the
JSON results in <root>: fleet_tables.json, angle_response.json, dt_xy_recal.json, gate_compare.json,
heldout_compare.json, sps_summary.json, plus FINDINGS.md (the ledger, rendered as the verdict section).
Writes <root>/report.html and <root>/figures/*.png (relative links).
"""
import argparse, html, json, os, re
import numpy as np

DETS = ['det2', 'det3', 'det4', 'det6', 'det7']


def load(root, name):
    p = os.path.join(root, name)
    return json.load(open(p)) if os.path.exists(p) else None


def md_table_to_html(md):
    rows = [l for l in md.strip().splitlines() if l.startswith('|')]
    out = []
    for i, l in enumerate(rows):
        cells = [c.strip() for c in l.strip('|').split('|')]
        if all(set(c) <= set('-: ') for c in cells):
            continue
        tag = 'th' if i == 0 else 'td'
        out.append('<tr>' + ''.join(f'<{tag}>{html.escape(c)}</{tag}>' for c in cells) + '</tr>')
    return '<table>' + '\n'.join(out) + '</table>'


def md_to_html(md):
    """Minimal markdown: ##/### headings, paragraphs, tables, `code`, **bold**, lists."""
    out, para, table = [], [], []

    def flush():
        nonlocal para, table
        if table:
            out.append(md_table_to_html('\n'.join(table))); table = []
        if para:
            t = ' '.join(para); t = html.escape(t)
            t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t); t = re.sub(r'`(.+?)`', r'<code>\1</code>', t)
            out.append(f'<p>{t}</p>'); para = []
    for l in md.splitlines():
        if l.startswith('|'):
            if para:
                flush()
            table.append(l); continue
        if table:
            flush()
        m = re.match(r'^(#{1,4}) (.*)', l)
        if m:
            flush(); out.append(f'<h{len(m.group(1)) + 1}>{html.escape(m.group(2))}</h{len(m.group(1)) + 1}>'); continue
        if not l.strip():
            flush(); continue
        para.append(l.strip())
    flush()
    return '\n'.join(out)


def fig_angle_response(root, ar, ft):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    edges = np.array(ar['edges']); mid = 0.5 * (edges[:-1] + edges[1:])
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i, p in enumerate('xy'):
        ax = axs[0, i]
        for det in DETS:
            k = f'{det}_{p}'
            if k in ar:
                ax.plot(mid, ar[k]['bias_bins'], 'o-', label=det)
        ax.axhline(0, color='k', lw=0.5); ax.set_xlabel('tan θ (M3)'); ax.set_ylabel('median θ_fit − θ_ref [deg]'); ax.set_title(f'{p.upper()}: signed angle bias per tan bin (|Δθ| < 5°)')
        ax.legend(fontsize=8); ax.set_ylim(-1.8, 1.0)
        ax = axs[1, i]
        for det in DETS:
            v = ft[det]['views'][p]
            b = np.array(v['implied_v_bins']); bins = [(0.08, 0.14), (0.14, 0.20), (0.20, 0.28), (0.28, 0.45)]
            x = [0.5 * (a + c) for a, c in bins]
            ax.plot(x, b / v['ladder_v'], 'o-', label=f"{det} (ladder {v['ladder_v']:.1f})")
        ax.axhline(1, color='k', lw=0.5); ax.set_xlabel('|tan θ| bin'); ax.set_ylabel('implied v / ladder v'); ax.set_title(f'{p.upper()}: implied drift velocity vs angle (compression signature)')
        ax.legend(fontsize=8); ax.set_ylim(0.85, 1.15)
    fig.tight_layout(); p_ = os.path.join(root, 'figures', 'angle_response.png'); fig.savefig(p_, dpi=110); plt.close(fig)
    return 'figures/angle_response.png'


def fig_ladder(root, ft):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, det in enumerate(DETS):
        r = ft[det]; vt = r['v_timing']
        for j, p in enumerate('xy'):
            v = r['views'][p]
            ax.errorbar(i + (j - 0.5) * 0.25, v['ladder_v'] / vt, yerr=v['ladder_v_err'] / vt, fmt='o' if p == 'x' else 's', color=f'C{i}',
                        label=('X' if p == 'x' else 'Y') if i == 0 else None)
    ax.axhline(1, color='k', lw=0.5); ax.set_xticks(range(len(DETS))); ax.set_xticklabels(DETS); ax.set_ylabel('reco ladder v / timing v')
    ax.set_title('Angle scale of the fit against the kernel-free timing v (0.10 < |tan| < 0.45)'); ax.legend()
    fig.tight_layout(); p_ = os.path.join(root, 'figures', 'ladder_vs_timing.png'); fig.savefig(p_, dpi=110); plt.close(fig)
    return 'figures/ladder_vs_timing.png'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default=os.path.expanduser('~/x17/cosmic_bench/model_audit'))
    ap.add_argument('--findings', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FINDINGS.md'))
    ap.add_argument('--inline', default=None, help='also write a self-contained copy (figures as data URIs) to this path')
    a = ap.parse_args(); root = a.root; os.makedirs(os.path.join(root, 'figures'), exist_ok=True)
    ft, ar, dt = load(root, 'fleet_tables.json'), load(root, 'angle_response.json'), load(root, 'dt_xy_recal.json')
    gate, ho, sps = load(root, 'gate_compare.json'), load(root, 'heldout_compare.json'), load(root, 'sps_summary.json')
    S = []
    S.append('<h1>Model audit of the physical RC waveform reconstruction (B4), night of 2026-10-27</h1>')
    S.append('<p class="lead">Independent audit of <code>wft</code> (phys kernel, promoted <code>calib_bundle</code>s) on the five '
             'June golden rows and SPS det4, against the four figures of merit: angle resolution vs M3/uRWELL, detection efficiency / '
             'loss budget, drift velocity / depth scale, position resolution and x–y pairing. Code, scripts and this report: '
             '<code>rerun/model_audit/</code>. The findings ledger (<code>FINDINGS.md</code>) is rendered below with each item\'s status.</p>')
    # verdict from FINDINGS.md
    fm = open(a.findings).read()
    S.append('<h2>Verdict and findings</h2>' + md_to_html(fm.split('\n', 1)[1] if fm.startswith('#') else fm))
    if ft:
        S.append('<h2>Fleet consistency on the finalize tables (final_physmask, promoted constants)</h2>')
        S.append(md_to_html(open(os.path.join(root, 'fleet_tables.md')).read()))
        S.append(f'<img src="{fig_ladder(root, ft)}">')
        if ar:
            S.append(f'<img src="{fig_angle_response(root, ar, ft)}">')
    if dt:
        S.append('<h2>x–y t0 offsets re-measured (dt_xy) and pairing widths</h2><table><tr><th>det</th><th>old dt_xy</th><th>new (n, median, rσ)</th><th>pairing rσ lq / u50 / u90 / t0</th></tr>')
        for det in DETS:
            r = dt[det]; new = {k: (v['n'], round(v['median'], 1), round(v['rsig'], 1)) for k, v in r['dt_xy_new'].items()}
            rs = r['xy_pairing']['rsig']
            S.append(f"<tr><td>{det}</td><td>{ {k: round(v, 1) for k, v in r['dt_xy_old'].items()} }</td><td>{new}</td>"
                     f"<td>{rs['lq']:.3f} / {rs['u50']:.0f} / {rs['u90']:.0f} / {rs['t0']:.0f}</td></tr>")
        S.append('</table>')
    for name, title in (('gate_compare.md', 'Fleet gate of the prototype switches (8000 M3-matched events per golden row; interior, X+Y, 5 mm / 3 mm)'),
                        ('heldout_compare_env.md', 'Held-out arms, environment switches (1320 events per chamber, event-paired, arm − base): t0 interpolation, model fraction, saturated solve'),
                        ('heldout_compare_ts.md', 'Held-out arm: two-sided start'),
                        ('heldout_compare_cli.md', 'Held-out arms, model terms: X asymmetric copy, narrow σ0_x, no σ0 profile, diffusion ×0.7 / ×1.3, RC diffusion ×1.5, K = 30, t0 pinned'),
                        ('sps_summary.md', 'SPS det4 known-angle test (run_63 25.64° drift ladder, bench det4 bundle, self-seeded; sps_res = no ZS censoring, sps2 = ZS censored)')):
        p = os.path.join(root, name)
        if os.path.exists(p):
            S.append(f'<h2>{title}</h2>' + md_to_html(open(p).read()))
    lg = os.path.join(root, 'steep_probe.log')
    if os.path.exists(lg):
        S.append('<h2>Steep tracks: model or optimiser? (steep_probe.py, held-out |tan| ≥ 0.25)</h2>'
                 '<p>χ² at the M3 angle (p0 re-scanned, t0 and profile fitted) minus χ² at the fit: positive = the model prefers the fitted '
                 '(shallower) angle; the fraction with negative values is the optimiser-miss rate.</p><pre>' + html.escape(open(lg).read()) + '</pre>')
    S.append('<h2>What this audit does not settle</h2><ul>'
             '<li>The drift-time-dependent part of the angle compression (SPS 0.89 → 0.65 from 235 to 148 V/cm): which model term carries it is not identified; the low-field span v is itself an upper bound.</li>'
             '<li>Position resolution is reference-limited (M3 ≈ 0.40 mm); no pointing-estimator defect was looked for beyond the alignment residuals.</li>'
             '<li>SPS: no uRWELL-referenced position test tonight (angle test only); the run_61 32-sample inclined set truncates the column and was not used.</li>'
             '<li>The noise-band mechanism, the X asymmetry mechanism and det6 are stated limitations of the plan (§7); nothing here changes that.</li></ul>')
    css = ('body{font-family:system-ui,sans-serif;max-width:1500px;margin:1em auto;padding:0 1em;line-height:1.4}'
           'table{border-collapse:collapse;font-size:12px;margin:.6em 0}td,th{border:1px solid #bbb;padding:2px 6px;text-align:right}'
           'th{background:#eee}td:first-child,th:first-child{text-align:left}img{max-width:100%}code{background:#f3f3f3;padding:0 3px}'
           '.lead{font-size:1.05em}h2{border-bottom:1px solid #ccc;margin-top:1.6em}')
    doc = f'<!doctype html><html><head><meta charset="utf-8"><title>Model audit B4 — 2026-10-27</title><style>{css}</style></head><body>' + '\n'.join(S) + '</body></html>'
    open(os.path.join(root, 'report.html'), 'w').write(doc)
    print('wrote', os.path.join(root, 'report.html'))
    if a.inline:
        import base64
        def emb(m):
            b = base64.b64encode(open(os.path.join(root, m.group(1)), 'rb').read()).decode()
            return f'<img src="data:image/png;base64,{b}"'
        doc2 = re.sub(r'<img src="(figures/[^"]+)"', emb, doc)
        doc2 = doc2.replace('<head>', '<head><meta name="description" content="Independent audit of the physical RC waveform reconstruction on the June bench and SPS det4 (B4, 2026-10-27): fleet consistency, held-out model arms, prototype gate, SPS known-angle test; findings F1-F21 with status.">', 1)
        open(a.inline, 'w').write(doc2); print('wrote', a.inline)


if __name__ == '__main__':
    main()
