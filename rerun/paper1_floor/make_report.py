#!/usr/bin/env python3
"""make_report.py -- report.html + figures of the §8 floor study from results/*.json (and the frames for the
per-angle panel).  Re-run after analyze.py / position_ms.py / xasym_sys.py / hits_arm.py / ytoy_collect.py."""
import html
import json
import os

import numpy as np

import floorlib as C

R = os.path.join(C.HERE, 'results')
FIG = os.path.join(C.HERE, 'figures')
DETS = list(C.CALIB)


def J(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fig_budget(B, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels, parts = [], []
    names = ['toy (all modelled terms)', 'M3 0.05', 'saturated planes', 'bright tail', 'topology',
             'remainder on clean events (fit-visible, ~angle-independent)']
    for d in DETS:
        for p in ('x', 'y'):
            v = B[f'{d}|calib']['views'][p]
            meas2 = v['measured'] ** 2
            toy2 = (v['toy_all'] or 0) ** 2
            lad = {L['step']: max(L['delta2'], 0.0) for L in v['ladder']}
            row = [toy2, 0.05 ** 2, lad.get('sat', 0), lad.get('amp', 0), lad.get('topo', 0), v['U_cond'] ** 2]
            labels.append(f'{d} {p.upper()}')
            parts.append(np.array(row) / max(sum(row), 1e-9) * meas2)
    parts = np.array(parts)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    cols = ['C0', '0.6', 'C1', 'C2', 'C4', 'C3']
    bottom = np.zeros(len(labels))
    for i, n in enumerate(names):
        ax.bar(labels, parts[:, i], bottom=bottom, color=cols[i], label=n)
        bottom += parts[:, i]
    ax.set_ylabel('variance of the angle residual [deg$^2$]\n(after the 2D tilt map)')
    ax.legend(fontsize=7, ncol=2)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_angle(ytoy, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analyze import derived, ladder
    X = {}
    for det in DETS:
        lf = os.path.join(C.REPO, 'rerun', 'model_audit', 'followups', 'fd_floor_realism', 'results',
                          f'ladder_pass1_ig_n1500_{det}.json')
        X[det] = json.load(open(lf))['rungs'] if os.path.exists(lf) else None
    fig, ax = plt.subplots(2, 5, figsize=(13, 5), sharex=True)
    edges = (0, 3, 6, 10, 15, 25)
    for j, det in enumerate(DETS):
        F = C.load(C.CALIB[det])
        for i, p in enumerate(('x', 'y')):
            g = derived(C.sample(F, p))
            tl = ladder(g)[-1][1]
            th = np.abs(np.degrees(np.arctan(g.tan_ref)))
            xs, ya, yt = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (th >= lo) & (th < hi)
                if m.sum() >= 100:
                    xs.append(0.5 * (lo + hi))
                    ya.append(C.s68(g.dt[m]))
                    yt.append(C.s68(g.dt[m & tl]) if (m & tl).sum() >= 80 else np.nan)
            a = ax[i, j]
            a.plot(xs, ya, 'o-', color='0.4', label='data, all')
            a.plot(xs, yt, 's-', color='C3', label='data, toy-like')
            T = X[det] if p == 'x' else ((ytoy or {}).get(det) or {}).get('rungs')
            if T:
                for rung, st, lab in (('L0', ':', 'toy floor'), ('L8', '--', 'toy, all terms')):
                    if rung in T:
                        pa = T[rung].get('per_angle', T[rung])
                        ks = [k for k in ('3', '9', '16') if k in pa]
                        a.plot([float(k) for k in ks], [pa[k]['s68'] for k in ks], st, color='C0', label=lab)
            a.set_title(f'{det} {p.upper()}', fontsize=9)
            a.set_ylim(0, 2.6)
            a.grid(alpha=0.3)
            if j == 0:
                a.set_ylabel('s68 [deg]')
            if i == 1:
                a.set_xlabel('|theta_ref| [deg]')
    ax[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_pos(P, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for i, p in enumerate(('x', 'y')):
        a = ax[i]
        for j, det in enumerate(DETS):
            o = P[f'{det}|calib'][p]
            b = o[o['matched']]['bins_um']
            a.plot([x for x, _ in b], [y for _, y in b], 'o-', color=f'C{j}', label=f"{det} (chi2->0 {o['chi2_0_um']:.0f})")
            a.errorbar([0], [o['chi2_0_um']], yerr=[o['chi2_0_err_um']], fmt='x', color=f'C{j}')
        a.axhspan(190, 210, color='k', alpha=0.12, label='SPS det4 head-on 190-210')
        a.set_title(f'view {p.upper()}: position residual vs the M3 chi2 of the matching coordinate', fontsize=9)
        a.set_xlabel('M3 chi2 (quintile mean)')
        a.grid(alpha=0.3)
    ax[0].set_ylabel('core sigma of p0 - M3 [um]')
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def table(rows, head):
    h = '<table><tr>' + ''.join(f'<th>{html.escape(x)}</th>' for x in head) + '</tr>'
    for r in rows:
        h += '<tr>' + ''.join(f'<td>{x}</td>' for x in r) + '</tr>'
    return h + '</table>'


def main():
    os.makedirs(FIG, exist_ok=True)
    B, P, X, Fl, Y = J('floor_budget.json'), J('position_ms.json'), J('xasym_sys.json'), J('fleet.json'), J('ytoy.json')
    fig_budget(B, os.path.join(FIG, 'floor_budget.png'))
    fig_angle(Y, os.path.join(FIG, 'angle_form.png'))
    fig_pos(P, os.path.join(FIG, 'position_ms.png'))
    H = ['<!DOCTYPE html><html><head><meta charset="utf-8"><title>§8 floor study</title><style>body{font-family:sans-serif;'
         'max-width:1100px;margin:auto;line-height:1.45}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid'
         ' #ccc;padding:3px 7px;text-align:right}th{background:#eee}img{max-width:100%}.v{background:#eef6ee;padding:8px'
         ' 12px;border-left:4px solid #393}</style></head><body>',
         '<h1>Paper I §8: what the unexplained angle term is made of</h1>',
         '<p>Generated by <code>rerun/paper1_floor/make_report.py</code>; numbers from <code>results/*.json</code>. '
         'Pass 1 (Analysis_pass1, read-only), calibration golden rows, interior_good, v(t) arm, FD convention.</p>']
    H.append('<div class="v"><b>Verdict.</b> See README.md for the full statement. The unexplained term is not the '
             'reference (no M3-chi2 dependence), not a per-event common cause (X-Y |r| &lt; 0.1), not position structure '
             'finer than the tilt map and not the X copy (Y has it too). Event conditions the toy does not generate '
             '(saturated planes, the bright tail, topology) are 12-37 % of it (65-92 % on det7). The rest, 0.56-0.63 deg on '
             'det2/3/7 X (1.42 det4), is present on clean events, roughly angle-independent, and fit-visible: a best-chi2 '
             'selection removes the same share of it as of the toy\'s own error, so it behaves like an under-modelled '
             'stochastic part of the waveform (the data chi2/dof is 4-15x the toy\'s and grows as Q^1.4-2), not like the '
             'chi2-blind geometry the 10-27 study proposed. The bench position resolution agrees with SPS once M3\'s own '
             'scattering is measured.</div>')
    rows = []
    for d in DETS:
        for p in ('x', 'y'):
            v = B[f'{d}|calib']['views'][p]
            fr = {L['step']: L['delta2'] / max(v['U'] ** 2, 1e-9) for L in v['ladder'][1:]}
            m3 = v['m3_term'][v['m3_term']['matched']]['delta2']
            st = v.get('chi2_select_share_toy')
            rows.append([f'{d} {p.upper()}', f"{v['measured']:.2f}", '—' if v['toy_all'] is None else f"{v['toy_all']:.2f}",
                         f"{v['U']:.2f}", f"{100 * fr.get('sat', 0):.0f}", f"{100 * fr.get('amp', 0):.0f}",
                         f"{100 * fr.get('topo', 0):.0f}", f"<b>{v['U_cond']:.2f}</b>",
                         f"{100 * v['chi2_select_share_data']:.0f} / {'—' if st is None else f'{100 * st:.0f}'}",
                         f"{np.sqrt(max(v['fine_cov'], 0)):.2f}", f"{np.sqrt(max(v['nonlin_cov'], 0)):.2f}",
                         f"{100 * m3 / max(v['U'] ** 2, 1e-9):+.0f}",
                         '' if v['angle_form_toylike'].get('a') is None else f"{v['angle_form_toylike']['a']:.2f}"])
    H.append('<h2>The ranked budget (calibration rows)</h2>')
    H.append(table(rows, ['row', 'measured', 'toy_all', 'U', 'saturated %', 'bright %', 'topology %', 'remainder',
                          'chi2-select share data / toy %', 'fine rms', 'non-lin rms', 'M3 %', 'a (θ→0)']))
    H.append('<p>%: share of U² removed by each cumulative cut; remainder = U on clean events. chi2-select: the share of '
             'the clean-event variance a best-chi2 quarter (at the same amplitude, angle and strip count) removes, in data '
             'and in the toy (pure selection). det6 is 92 % saturated: its ladder is not representative. For Y, toy_all is '
             'the Y toy (approximate: X generation fitted as Y; an upper estimate of the Y toy).</p>')
    H.append('<img src="figures/floor_budget.png"><p>Variance of the angle residual after the tilt map, split into the toy, M3, '
             'the event-condition terms and what is left on toy-like events.</p>')
    H.append('<img src="figures/angle_form.png"><p>Data (all and toy-like events) against the toy floor and the full toy, '
             'per angle bin.</p>')
    rows = []
    for k, r in P.items():
        for p, o in r.items():
            f = lambda v: '—' if v is None else f'{v:.0f}'
            rows.append([f'{k} {p.upper()}', f"{o['core_all_um']:.0f}", f"{o['ms_um']:.0f}", f(o['P_um']),
                         f"{f(o['chamber_um'])} ± {f(o['chamber_err_um'])}",
                         f"{f(o['chamber_near_normal_um'])} ± {f(o['chamber_near_normal_err_um'])}"])
    H.append('<h2>Bench position against SPS</h2>')
    H.append(table(rows, ['row', 'core σ all [µm]', 'M3 scattering [µm]', 'M3 pointing P [µm]', 'chamber, all angles',
                          'chamber, |θ| < 5°']))
    H.append('<img src="figures/position_ms.png"><p>The position residual grows with the M3 chi2: the reference\'s own '
             'scattering between its doublets. Its chi2 → 0 limit minus the pointing is the chamber.</p>')
    if X:
        rows = [[k, f"{v['shift_um_rms']:.0f}"] + [f"{v['arms'][a]['mesh_core_um']:.0f}" for a in ('none', 'K', 'K-1s', 'K+1s')]
                for k, v in X.items() if isinstance(v, dict) and 'arms' in v]
        H.append('<h2>wft.xasym ± systematic</h2>')
        H.append(table(rows, ['row', 'shift rms [µm]', 'X core: none', 'K', 'K−1σ', 'K+1σ']))
    if Fl:
        rows = []
        for d, r in Fl.items():
            for p, v in r['views'].items():
                rows.append([f'{d} {p.upper()}', v['n_common'], f"{v['H0']['s68']:.2f}", f"{v['W']['d_vt']['s68']:.2f}",
                             f"{v['hits_v_implied_um_ns']:.1f}", f"{v['pos_core_um']['H0']:.0f}", f"{v['pos_core_um']['W']:.0f}"])
        H.append('<h2>Fleet: H0 hits against W waveforms, same events</h2>')
        H.append(table(rows, ['row', 'events', 'H0 s68', 'W s68', 'H0 implied v [µm/ns]', 'H0 pos [µm]', 'W pos [µm]']))
        H.append('<img src="figures/fleet.png">')
    H.append('<h2>What this does not rule out</h2><ul><li>The angle-independent remainder is not identified; it is bounded, '
             'not explained.</li><li>The Y toy uses the X generation (no measured Y tails).</li><li>The bench chamber '
             'position rests on the M3 pointing P (±10 % moves the near-normal chamber value by ~±50 µm) and on a linear '
             'sigma²(chi2).</li><li>det6 (92 % saturated) has no representative toy-like sample.</li></ul></body></html>')
    open(os.path.join(C.HERE, 'report.html'), 'w').write('\n'.join(H))
    print('report.html written')


if __name__ == '__main__':
    main()
