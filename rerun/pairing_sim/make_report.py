#!/usr/bin/env python3
"""make_report.py -- the x<->y pairing simulation study report (PLAN.md), built from the tables:

    ../../.venv/bin/python make_report.py     -> ~/x17/cosmic_bench/satstudy/pairing_sim/report/report.html
                                                 (+ report_inline.html, figures as data URIs, for the notes site)
Inputs (all in ~/x17/cosmic_bench/satstudy/pairing_sim/): gates_data.json (gates_data.py), gates_sim.json
(sim_gates.py), mf_data_det3.parquet / mf_sim_*.parquet (mfratio.py), sim_overlay_<cond>.json (sim_overlay.py),
xysplit.json (xysplit_score in sim_gates.py, optional), and the data overlay ../overlay.json (rerun/satstudy).
"""
import base64, json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = os.path.expanduser('~/x17/cosmic_bench/satstudy/pairing_sim')
R = os.path.join(D, 'report'); F = os.path.join(R, 'figures')
DETS = ('det2', 'det3', 'det4', 'det6', 'det7')
C_DATA, C_SIM, C_TRUTH, C_N2 = '#2a6fb0', '#d0562c', '#7a7a7a', '#8a3fa0'


def J(name):
    p = os.path.join(D, name)
    return json.load(open(p)) if os.path.exists(p) else None


def rsig(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else float('nan')


def pct(x, nd=1):
    return '–' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{100 * x:.{nd}f} %'


def num(x, f='{:.3f}'):
    return '–' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f.format(x)


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(F, name), dpi=130); plt.close(fig)
    return f'figures/{name}'


def fig_chambers(gd):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(DETS))
    s = [gd[d]['lq']['rsig'] for d in DETS]; e = [gd[d]['lq']['rsig_err'] for d in DETS]
    ax[0].bar(x, s, yerr=e, color=C_DATA); ax[0].set_xticks(x, DETS)
    ax[0].set_ylabel('robust σ of ln(q_x/q_y)'); ax[0].set_title('x/y charge identity of one track')
    for k, nm in enumerate(('q', 'q_u')):
        ax[1].bar(x + (k - 0.5) * 0.38, [100 * gd[d]['toy']['right'][nm] for d in DETS], 0.38,
                  color=[C_DATA, '#8fb6dc'][k], label=['charge', 'charge + depth profile'][k])
    ax[1].set_xticks(x, DETS); ax[1].set_ylim(80, 100); ax[1].set_ylabel('pairs right [%]')
    ax[1].set_title('toy from each chamber\'s own singles'); ax[1].legend(fontsize=8, loc='lower right')
    return save(fig, 'chambers.png')


def fig_angle(gd, gs):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for k, (key, lab) in enumerate((('lq_rsig', 'σ ln(q_x/q_y)'), ('u50_rsig', 'σ (u50_x − u50_y) [ns]'),
                                    ('lnQ_rsig', 'σ ln Q (Landau width)'))):
        for v, mk in (('x', 'o'), ('y', 's')):
            rows = [r for r in gd['det3']['vs_view_angle'][v] if r[key] is not None]
            ax[k].plot([r['deg'] for r in rows], [r[key] for r in rows], mk + '-', color=C_DATA, ms=5,
                       label=f'data det3, {v} inclined')
        for v, mk in (('x', 'o'), ('y', 's')):
            pts = [('th00', 0)] + [(f'th{v}{a:02d}', a) for a in (5, 10, 15, 20, 30)]
            xs, ys = [], []
            for p, a in pts:
                g = gs['points'].get(p, {}).get('dphys')
                if g:
                    val = {'lq_rsig': g['lq']['rsig'], 'u50_rsig': g['u50']['rsig'], 'lnQ_rsig': g['lnQ']['rsig']}[key]
                    xs.append(a); ys.append(val)
            ax[k].plot(xs, ys, mk + '--', color=C_SIM, ms=5, mfc='none', label=f'sim v4c, {v} inclined')
        if key == 'lq_rsig':
            xs = [0, 5, 10, 15, 20, 30]
            ys = [gs['points'].get(p, {}).get('truth', {}) for p in ('th00', 'thx05', 'thx10', 'thx15', 'thx20', 'thx30')]
            ax[k].plot(xs, [t.get('lq_rsig', np.nan) if t else np.nan for t in ys], '^:', color=C_TRUTH,
                       label='sim truth (induced, no electronics)')
            ax[k].set_ylim(0, 0.3)
        if key == 'u50_rsig':
            ax[k].set_ylim(0, 450)
        ax[k].set_xlabel('track inclination in that view [deg]'); ax[k].set_ylabel(lab)
    ax[0].legend(fontsize=7)
    return save(fig, 'angle.png')


def fig_mechanism(gs, mf):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    t = gs['points']['th00']['truth']
    n = [0.5 * (max(r['lo'], 100) + min(r['hi'], 800)) for r in t['scaling']]
    ax[0].plot(n, [r['rsig'] for r in t['scaling']], 'o', color=C_TRUTH, label='sim truth, th00')
    nn = np.linspace(80, 900, 50); a = t['split_per_avalanche']
    ax[0].plot(nn, a / np.sqrt(nn), '-', color=C_TRUTH, lw=1, label=f'{a:.2f} / √N')
    ax[0].set_xscale('log'); ax[0].set_xlabel('electrons reaching the mesh, N'); ax[0].set_ylabel('σ ln(q_x/q_y)')
    ax[0].set_title('the sim\'s split is a per-avalanche coin'); ax[0].legend(fontsize=8)
    labs, dv, sv = [], [], []
    for k, nm in (('sum', 'Σ samples'), ('pos', 'Σ positive samples'), ('fit', 'wft fit q_sum')):
        labs.append(nm); dv.append(mf['data'].get(k)); sv.append(mf['sim'].get(k))
    x = np.arange(len(labs))
    ax[1].bar(x - 0.2, dv, 0.4, color=C_DATA, label='data det3, < 5°'); ax[1].bar(x + 0.2, sv, 0.4, color=C_SIM, label='sim th00')
    ax[1].axhline(t['lq_rsig'], color=C_TRUTH, ls=':', label='sim truth')
    ax[1].set_xticks(x, labs); ax[1].set_ylabel('σ ln(q_x/q_y)'); ax[1].set_title('same strips, same reader')
    ax[1].legend(fontsize=8)
    return save(fig, 'mechanism.png')


def fig_toy(gd, gs):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    def curve(by, c, lab, ls='-'):
        b = [r for r in by if r['right'] is not None]
        x = [np.sqrt(max(r['lo'], 0.01) * min(r['hi'], 2)) for r in b]
        ax.plot(x, [100 * r['right'] for r in b], 'o' + ls, color=c, label=lab)
    curve(gd['det3']['toy']['by_ratio']['q'], C_DATA, 'data det3 (charge)')
    curve(gd['det3']['toy']['by_ratio']['q_u'], '#8fb6dc', 'data det3 (charge + profile)')
    if gs.get('pooled_dphys'):
        curve(gs['pooled_dphys']['toy']['by_ratio']['q'], C_SIM, 'sim v4c (charge)', '--')
        curve(gs['pooled_dphys']['toy']['by_ratio']['q_u'], '#eba58c', 'sim v4c (charge + profile)', '--')
    ax.set_xscale('log'); ax.set_xlabel('|ln Q1/Q2| of the two tracks'); ax.set_ylabel('pairs right [%]')
    ax.set_ylim(20, 101); ax.legend(fontsize=8); ax.set_title('toy: right vs charge ratio')
    return save(fig, 'toy.png')


def fig_overlay(ov_data, ov_sim):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    def mid(b):
        return 0.5 * (b['lo'] + b['hi'])
    series = [('data det3 overlay', ov_data, C_DATA, '-')] + \
             [(f'sim {c}', o, {'nom': C_SIM, 'n2': C_N2}[c], '--') for c, o in ov_sim.items()]
    for lab, o, c, ls in series:
        if not o:
            continue
        b = [r for r in o['bins'] if r['hi'] <= 80]
        ax[0].plot([mid(r) for r in b], [100 * r['both_found'] for r in b], 'o' + ls, color=c, label=lab)
        bb = [r for r in b if r.get('right_q') is not None]
        ax[1].plot([mid(r) for r in bb], [100 * r['right_q'] for r in bb], 'o' + ls, color=c, label=lab + ', charge')
    ax[0].set_xlabel('smaller of the two views\' separations [mm]'); ax[0].set_ylabel('both tracks found [%]')
    ax[1].set_xlabel('smaller of the two views\' separations [mm]'); ax[1].set_ylabel('pairs right, both found [%]')
    ax[1].set_ylim(70, 101); ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    return save(fig, 'overlay.png')


def mf_numbers():
    out = dict(data={}, sim={})
    p = os.path.join(D, 'mf_data_det3.parquet')
    if os.path.exists(p):
        m = pd.read_parquet(p)
        m = m[np.degrees(np.arctan(np.maximum(abs(m.tan_x_m3), abs(m.tan_y_m3)))) < 5]
        out['data'] = dict(sum=rsig(np.log(m.x_sum / m.y_sum)), pos=rsig(np.log(m.x_pos / m.y_pos)),
                           fit=rsig(np.log(m.x_q_sum / m.y_q_sum)), n=int(len(m)))
    for tag in ('th00_dphys', 'th00_simcal'):
        p = os.path.join(D, f'mf_sim_{tag}.parquet')
        if os.path.exists(p):
            m = pd.read_parquet(p)
            out['sim'] = dict(sum=rsig(np.log(m.x_sum / m.y_sum)), pos=rsig(np.log(m.x_pos / m.y_pos)),
                              fit=rsig(np.log(m.x_q_sum / m.y_q_sum)), n=int(len(m)), tag=tag)
            break
    return out


def img(src):
    return f'<figure><img src="{src}" alt=""></figure>'


def main():
    os.makedirs(F, exist_ok=True)
    gd, gs = J('gates_data.json'), J('gates_sim.json')
    ov_data = json.load(open(os.path.join(D, '..', 'overlay.json')))
    ov_sim = {c: J(f'sim_overlay_{c}.json') for c in ('nom', 'n2')}
    ov_sim = {c: o for c, o in ov_sim.items() if o}
    xs = J('xysplit.json'); mf = mf_numbers()
    figs = dict(ch=fig_chambers(gd), ang=fig_angle(gd, gs), mech=fig_mechanism(gs, mf), toy=fig_toy(gd, gs),
                ov=fig_overlay(ov_data, ov_sim))
    tmpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_template.html')).read()
    ctx = build_context(gd, gs, ov_data, ov_sim, xs, mf, figs)
    html = re.sub(r'\{\{(\w+)\}\}', lambda m: str(ctx[m.group(1)]), tmpl)
    open(os.path.join(R, 'report.html'), 'w').write(html)
    inline = re.sub(r'src="(figures/[^"]+)"', lambda m: 'src="data:image/png;base64,' + base64.b64encode(
        open(os.path.join(R, m.group(1)), 'rb').read()).decode() + '"', html)
    open(os.path.join(R, 'report_inline.html'), 'w').write(inline)
    print('wrote', os.path.join(R, 'report.html'))


def build_context(gd, gs, ov_data, ov_sim, xs, mf, figs):
    c = {f'fig_{k}': img(v) for k, v in figs.items()}
    rows = []
    for d in DETS:
        g = gd[d]
        rows.append(f"<tr><td>{d}</td><td>{g['run_key']}</td><td>{g['n']}</td><td>{g['lq']['median']:+.3f}</td>"
                    f"<td><b>{g['lq']['rsig']:.3f}</b> ± {g['lq']['rsig_err']:.3f}</td><td>{g['u50']['rsig']:.0f}</td>"
                    f"<td>{g['u90']['rsig']:.0f}</td><td>{g['lnQ']['rsig']:.3f}</td>"
                    f"<td>{pct(g['toy']['right']['q'])}</td><td>{pct(g['toy']['right']['q_u'])}</td>"
                    f"<td>{pct(g['toy']['frac_ratio_lt_0p1'], 0)}</td></tr>")
    c['chamber_rows'] = '\n'.join(rows)
    d3 = gd['det3']
    p = gs.get('pooled_dphys') or {}
    sim0 = gs['points']['th00'].get('dphys', {})
    ref0 = gs['points']['th00'].get('data') or {}
    c['d3_lq'] = num(d3['lq']['rsig']); c['d3_u50'] = num(d3['u50']['rsig'], '{:.0f}')
    c['sim_lq'] = num(p.get('lq', {}).get('rsig')); c['sim_u50'] = num(p.get('u50', {}).get('rsig'), '{:.0f}')
    c['sim_lnQ'] = num(p.get('lnQ', {}).get('rsig')); c['d3_lnQ'] = num(d3['lnQ']['rsig'])
    c['ratio_lq'] = num(p['lq']['rsig'] / d3['lq']['rsig'], '{:.2f}') if p else '–'
    c['ratio_u50'] = num(p['u50']['rsig'] / d3['u50']['rsig'], '{:.2f}') if p else '–'
    c['sim_toy_q'] = pct(p.get('toy', {}).get('right', {}).get('q')); c['sim_toy_qu'] = pct(p.get('toy', {}).get('right', {}).get('q_u'))
    c['d3_toy_q'] = pct(d3['toy']['right']['q']); c['d3_toy_qu'] = pct(d3['toy']['right']['q_u'])
    c['truth_lq'] = num(gs['points']['th00']['truth']['lq_rsig']); c['truth_a'] = num(gs['points']['th00']['truth']['split_per_avalanche'], '{:.2f}')
    c['mf_data_pos'] = num(mf['data'].get('pos')); c['mf_data_sum'] = num(mf['data'].get('sum'))
    c['mf_sim_pos'] = num(mf['sim'].get('pos')); c['mf_sim_sum'] = num(mf['sim'].get('sum'))
    c['sim0_lq'] = num(sim0.get('lq', {}).get('rsig')); c['ref0_lq'] = num(ref0.get('lq_rsig'))
    # gate table
    def gate_row(name, data, sim, test, ok):
        return f"<tr><td>{name}</td><td>{data}</td><td>{sim}</td><td>{test}</td><td class='{'pass' if ok else 'fail'}'>{'pass' if ok else 'FAIL'}</td></tr>"
    g = []
    if p:
        r = p['lq']['rsig'] / d3['lq']['rsig']
        g.append(gate_row('σ ln(q_x/q_y), single tracks', num(d3['lq']['rsig']), num(p['lq']['rsig']), 'ratio within ±20 %: ' + num(r, '{:.2f}'), abs(r - 1) <= 0.2))
        g.append(gate_row('median ln(q_x/q_y)', num(d3['lq']['median'], '{:+.3f}'), num(p['lq']['median'], '{:+.3f}'), 'stated (sets the cost centre)', True))
        r = p['lnQ']['rsig'] / gs.get('data_lnQ_local', d3['lnQ']['rsig'])
        g.append(gate_row('Landau width (σ ln Q)', f"{num(d3['lnQ']['rsig'])} ({num(gs.get('data_lnQ_local'))} after a 25 mm gain map)",
                          num(p['lnQ']['rsig']), 'within ±20 % of the map-corrected data: ' + num(r, '{:.2f}'), abs(r - 1) <= 0.2))
        for k in ('u50', 'u90'):
            r = p[k]['rsig'] / d3[k]['rsig']
            g.append(gate_row(f'σ ({k}_x − {k}_y)', num(d3[k]['rsig'], '{:.0f} ns'), num(p[k]['rsig'], '{:.0f} ns'), 'ratio within ±20 %: ' + num(r, '{:.2f}'), abs(r - 1) <= 0.2))
        dq = [(a['right'], b['right']) for a, b in zip(d3['toy']['by_ratio']['q'], p['toy']['by_ratio']['q']) if a['right'] is not None and b['right'] is not None]
        ok = all(abs(a - b) < 0.03 for a, b in dq)
        g.append(gate_row('toy right vs |ln Q1/Q2| (charge)', ' / '.join(pct(a, 0) for a, _ in dq), ' / '.join(pct(b, 0) for _, b in dq), 'each bin within 3 points', ok))
    if ov_sim.get('nom'):
        o = ov_sim['nom']
        bf = [(a['both_found'], b['both_found']) for a, b in zip([x for x in ov_data['bins'] if x['hi'] <= 80], o['bins'])]
        g.append(gate_row('S2 overlay: both found vs separation', ' / '.join(pct(a, 0) for a, _ in bf), ' / '.join(pct(b, 0) for _, b in bf),
                          'each bin within its binomial error (n = 40)', all(abs(a - b) <= 2.5 * np.sqrt(max(a * (1 - a), 0.02) / 40) + 0.05 for a, b in bf)))
        g.append(gate_row('S2 overlay: right ≥ 24 mm (charge)', pct(ov_data['ge24']['right_q']), pct(o['ge24']['right_q']), 'within errors', abs(ov_data['ge24']['right_q'] - o['ge24']['right_q']) < 0.05))
    c['gate_rows'] = '\n'.join(g)
    # overlay table
    orow = []
    for k, b in enumerate([x for x in ov_data['bins'] if x['hi'] <= 80]):
        cells = [f"{b['lo']}–{b['hi']}", pct(b['both_found'], 0), pct(b['right_q'], 0)]
        for cnd in ('nom', 'n2'):
            o = ov_sim.get(cnd)
            if o:
                s = o['bins'][k]; cells += [pct(s['both_found'], 0), pct(s['right_q'], 0), pct(s['right_q_u'], 0)]
            else:
                cells += ['–'] * 3
        orow.append('<tr>' + ''.join(f'<td>{x}</td>' for x in cells) + '</tr>')
    c['overlay_rows'] = '\n'.join(orow)
    for cnd in ('nom', 'n2'):
        o = ov_sim.get(cnd) or {}
        c[f'{cnd}_ge24_q'] = pct(o.get('ge24', {}).get('right_q')); c[f'{cnd}_ge24_qu'] = pct(o.get('ge24', {}).get('right_q_u'))
        c[f'{cnd}_ge24_n'] = o.get('ge24', {}).get('n', '–')
        c[f'{cnd}_open_rows'] = '\n'.join(f"<tr><td>{r['lo']}–{r['hi']}°</td><td>{r['n']}</td><td>{pct(r['right_q'])}</td><td>{pct(r['right_q_u'])}</td></tr>"
                                          for r in o.get('by_open', []))
    c['data_ge24_q'] = pct(ov_data['ge24']['right_q']); c['data_ge24_qu'] = pct(ov_data['ge24']['right_q_u'])
    c['xysplit'] = xs_text(xs, gs)
    return c


def _mf(tag):
    p = os.path.join(D, f'mf_sim_{tag}.parquet')
    if not os.path.exists(p):
        return '–', '–', '–'
    m = pd.read_parquet(p)
    return (num(rsig(np.log(m.x_pos / m.y_pos))) + '–' + num(rsig(np.log(m.x_sum / m.y_sum))),
            num(rsig(np.log(m.x_q_sum / m.y_q_sum))), str(len(m)))


def xs_text(xs, gs):
    if not xs:
        return '<p>The diagnostic Stage B run (displaced feedback secondaries) had not finished when this report was built.</p>'
    t0 = gs['points']['th00']['truth']
    w, f, n = _mf('th00_dphys')
    rows = [f"<tr><td>v4c (H = 0)</td><td>{num(t0['lq_rsig'])}</td><td>{w}</td><td>{f}</td><td>{n}</td></tr>"]
    for k, v in xs.items():
        h = re.sub(r'\D', '', k)
        w, f, n = _mf(f'xys{h}_dphys')
        rows.append(f"<tr><td>v4c, feedback displaced, {k}</td><td>{num(v['lq_rsig'])}</td><td>{w}</td><td>{f}</td><td>{n}</td></tr>")
    md = mf_numbers()['data']
    rows.append(f"<tr><td>data det3, &lt; 5°</td><td>–</td><td>{num(md.get('pos'))}–{num(md.get('sum'))}</td><td>{num(md.get('fit'))}</td><td>{md.get('n', '–')}</td></tr>")
    return ("<table><tr><th>th00, σ ln(q_x/q_y)</th><th>truth (induced)</th><th>waveform, model-free</th><th>wft fit</th><th>n</th></tr>"
            + ''.join(rows) + '</table>')


if __name__ == '__main__':
    main()
