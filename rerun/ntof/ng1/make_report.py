#!/usr/bin/env python3
"""make_report.py -- report.html for the N-G1 gate, built from ng1_score.py's results/ (re-run after any re-score).

    make_report.py            # -> $WORK_DATA/ntof_pass1/ng1_pass/report.html (+ figures/)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ng1_score import W, OUT, REF_Y, TAN_MAX   # noqa: E402

FIG = f'{W}/figures'
TOL_PT = 0.5


def load():
    R = dict(eff=pd.read_csv(f'{OUT}/eff_run145.csv').set_index('arm'),
             wall=pd.read_csv(f'{OUT}/wall_run145.csv'),
             split=pd.read_csv(f'{OUT}/wall_split_run145.csv').set_index('arm'),
             ceff=pd.read_csv(f'{OUT}/cos_eff_truth.csv').set_index('arm'),
             cres=pd.read_csv(f'{OUT}/cos_res_truth.csv'),
             line=json.load(open(f'{OUT}/line_run149.json')),
             lbins=pd.read_csv(f'{OUT}/line_bins_run149.csv'),
             srcy=json.load(open(f'{OUT}/srcy_run145.json')),
             focus=pd.read_csv(f'{OUT}/focus_run145.csv'))
    return R


def verdicts(R):
    V, N = {}, {}
    E, Wl, S = R['eff'], R['wall'].set_index(['chain', 'arm']), R['split']
    cr = R['cres'].set_index(['arm', 'view'])
    for arm in 'ABCD':
        v, n = {}, {}
        n['eff'] = dict(pass_=E.loc[arm, 'pass_per_trig'], is2=E.loc[arm, 'is2_per_trig'])
        n['conf'] = dict(pass_=Wl.loc[('pass', arm), 'conf_per_trig'],
                         is2=Wl.loc[('is2_v2', arm), 'conf_per_trig'] if ('is2_v2', arm) in Wl.index else np.nan,
                         purity=Wl.loc[('pass', arm), 'conf_per_gated'],
                         purity_is2=Wl.loc[('is2_v2', arm), 'conf_per_gated'] if ('is2_v2', arm) in Wl.index else np.nan)
        if arm in 'AC':
            d_eff = 100 * (n['eff']['pass_'] - n['eff']['is2'])
            d_conf = 100 * (n['conf']['pass_'] - n['conf']['is2'])
            d_same, e_same = S.loc[arm, 'd_common_pt'], S.loc[arm, 'd_common_err_pt']
            ok = d_eff >= -TOL_PT and d_conf >= -TOL_PT
            tag = 'PASS' if ok and d_same >= -TOL_PT else ('PASS (same-track purity at the tolerance)' if ok else 'FAIL')
            v['C1'] = (tag, f'gated/trigger {d_eff:+.1f} pt, wall-confirmed/trigger {d_conf:+.1f} pt; '
                            f'same tracks {d_same:+.2f} ± {e_same:.2f} pt')
            sc = [cr.loc[(arm, ax), 'd_s68_scaled'] for ax in 'xy']
            se = [cr.loc[(arm, ax), 'd_s68_scaled_err'] for ax in 'xy']
            raw = [cr.loc[(arm, ax), 's68_pass_raw_0.0_0.6'] - cr.loc[(arm, ax), 's68_is2_raw_0.0_0.6'] for ax in 'xy']
            rat = [cr.loc[(arm, ax), 'ratio_pass'] for ax in 'xy']
            ok_s = all(s <= 2 * e for s, e in zip(sc, se))
            ok_r = all(r <= 0.05 for r in raw)
            v['C2'] = ('PASS' if ok_s and ok_r else ('PASS on resolution; raw needs P7a v(t)' if ok_s else 'FAIL'),
                       f'Δs68 (pass − is2, rescaled) x {sc[0]:+.2f} ± {se[0]:.2f}°, y {sc[1]:+.2f} ± {se[1]:.2f}°; '
                       f'raw x {raw[0]:+.2f}°, y {raw[1]:+.2f}° (pass scale fit/true x {rat[0]:.3f}, y {rat[1]:.3f})')
            p = R['srcy']['pass'][arm]
            sig = (p['y_cross_mm'] - REF_Y[arm]) / p['err_mm']
            dc = R['srcy'][f'{arm}_common_pass_minus_is2']
            v['C3'] = ('PASS' if abs(sig) <= 2 else ('PASS within the ±3 mm estimator systematic' if
                                                      abs(p['y_cross_mm'] - REF_Y[arm]) <= 3 else 'FAIL'),
                       f"{p['y_cross_mm']:.1f} ± {p['err_mm']:.1f} mm vs {REF_Y[arm]:.0f} ({sig:+.1f}σ stat); "
                       f"pass − is2_v2 on the same events {dc:+.1f} mm")
            F = R['focus'].set_index(['chain', 'arm', 'tan_lo', 'tan_hi'])
            fs, fe = F.loc[('pass', arm, 0.1, 0.5), 's_best'], F.loc[('pass', arm, 0.1, 0.5), 's_err']
            fi = F.loc[('is2_v2', arm, 0.1, 0.5), 's_best']
            mu = R['line'][f'{arm}x']['ratio']
            mu_i = cr.loc[(arm, 'x'), 'ratio_is2']
            t, te = fs * mu, fe * mu
            n['g5'] = dict(focus=fs, focus_err=fe, muon=mu, tension=t, tension_err=te, focus_is2=fi,
                           tension_is2=fi * mu_i)
            v['G5'] = ('QUANTIFIED, open', f'focus s {fs:.3f} ± {fe:.3f} × muon fit/true {mu:.3f} → tension '
                                           f'{100 * (t - 1):+.1f} ± {100 * te:.1f} % (is2_v2 same run '
                                           f'{100 * (fi * mu_i - 1):+.1f} %)')
        else:
            v['C1'] = ('n/a (no is2_v2 reference)', f"gated/trigger {100 * n['eff']['pass_']:.1f} %, wall-confirmed/"
                                                   f"trigger {100 * n['conf']['pass_']:.1f} %, purity "
                                                   f"{n['conf']['purity']:.2f}")
            for c in ('C2', 'C3', 'G5'):
                v[c] = ('n/a', 'no A–C line, band-crossing guard or focus for this arm' if arm == 'D'
                        else 'positions only (no field-shaping ring): no angle criteria')
        V[arm], N[arm] = v, n
    return V, N


def figures(R):
    os.makedirs(FIG, exist_ok=True)
    E, Wl = R['eff'], R['wall'].set_index(['chain', 'arm'])
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.0))
    x = np.arange(4)
    for i, (chain, lab, col) in enumerate((('pass', 'pass 1', '#2b6cb0'), ('is2_v2', 'is2_v2', '#c05621'))):
        g = [E.loc[a, 'pass_per_trig' if chain == 'pass' else 'is2_per_trig'] for a in 'ABCD']
        c = [Wl.loc[(chain, a), 'conf_per_trig'] if (chain, a) in Wl.index else np.nan for a in 'ABCD']
        ax[0].bar(x + (i - 0.5) * 0.38, np.array(g) * 100, 0.36, color=col, label=lab)
        ax[1].bar(x + (i - 0.5) * 0.38, np.array(c) * 100, 0.36, color=col, label=lab)
    for a, t in zip(ax, ('gated tracks per trigger [%]', 'wall-confirmed per trigger [%]')):
        a.set_xticks(x, list('ABCD')); a.set_title(t, fontsize=9); a.tick_params(labelsize=8)
    ax[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f'{FIG}/eff_wall.png', dpi=150); plt.close(fig)

    B = R['cres']
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    lab = [f'{r.arm} {r.view}' for r in B.itertuples()]
    x = np.arange(len(B))
    for i, (k, l, col) in enumerate((('s68_pass_scaled_0.0_0.6', 'pass 1 (rescaled)', '#2b6cb0'),
                                     ('s68_is2_scaled_0.0_0.6', 'is2_v2 (rescaled)', '#c05621'),
                                     ('s68_pass_raw_0.0_0.6', 'pass 1 (own v)', '#90cdf4'),
                                     ('s68_is2_raw_0.0_0.6', 'is2_v2 (own v)', '#fbd38d'))):
        ax.bar(x + (i - 1.5) * 0.2, B[k], 0.19, yerr=B[k.replace('s68_', 's68err_')], color=col, label=l, capsize=2)
    ax.set_xticks(x, lab); ax.set_ylabel('68 % |Δθ| vs A–C truth [deg]', fontsize=8); ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=2); fig.tight_layout(); fig.savefig(f'{FIG}/cos_s68.png', dpi=150)
    plt.close(fig)

    F = R['focus']
    F = F[F.tan_hi - F.tan_lo < 0.15]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for (chain, arm), g in F.groupby(['chain', 'arm']):
        ax.errorbar((g.tan_lo + g.tan_hi) / 2 + (0.006 if chain == 'pass' else -0.006), g.s_best, g.s_err,
                    marker='o' if arm == 'A' else 's', ls='-' if chain == 'pass' else '--',
                    color='#2b6cb0' if chain == 'pass' else '#c05621', label=f'{chain} {arm}', ms=4, capsize=2)
    ax.axhline(1, color='0.6', lw=0.8)
    ax.set_xlabel('|tan x|', fontsize=8); ax.set_ylabel('sharpest focus scale s', fontsize=8)
    ax.tick_params(labelsize=8); ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(f'{FIG}/focus.png', dpi=150); plt.close(fig)


def table(df, fmt='{:.4g}'):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return '' if not np.isfinite(v) else fmt.format(v)
        return str(v)
    h = ''.join(f'<th>{c}</th>' for c in df.columns)
    b = ''.join('<tr>' + ''.join(f'<td>{f(v)}</td>' for v in r) + '</tr>' for r in df.itertuples(index=False))
    return f'<table><tr>{h}</tr>{b}</table>'


def main():
    R = load()
    V, N = verdicts(R)
    figures(R)
    hold = [a for a in 'AC' if any(V[a][c][0].startswith('FAIL') for c in ('C1', 'C2', 'C3'))]
    head = ('No arm fails N-G1; none is held.' if not hold else f'Held: {", ".join(hold)}.')
    vt = pd.DataFrame([dict(arm=a, criterion=c, verdict=V[a][c][0], numbers=V[a][c][1])
                       for a in 'ABCD' for c in ('C1', 'C2', 'C3', 'G5')])
    line = R['line']
    lb = R['lbins']; lb = lb[lb.n > 10]
    css = ('body{font-family:system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1a202c;'
           'background:#fff;line-height:1.45}table{border-collapse:collapse;font-size:13px;margin:8px 0 16px}'
           'td,th{border:1px solid #cbd5e0;padding:3px 7px;text-align:left}th{background:#edf2f7}'
           'img{max-width:100%}.v{font-size:18px;font-weight:600}code{background:#edf2f7;padding:0 3px}')
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>N-G1 gate, n_TOF pass 1</title>
<style>{css}</style></head><body>
<h1>N-G1 gate — n_TOF pass 1 against is2_v2</h1>
<p class="v">{head} A and C pass every criterion; C's angles need the P7a v(t) scale and C's source y sits 3 mm
above is2_v2's. G5 is quantified, not closed: A +{100 * (N['A']['g5']['tension'] - 1):.1f} %, C
+{100 * (N['C']['g5']['tension'] - 1):.1f} % (is2_v2 on the same run +{100 * (N['A']['g5']['tension_is2'] - 1):.1f} /
+{100 * (N['C']['g5']['tension_is2'] - 1):.1f} %).</p>
<p>Pass 1 = EOS <code>Analysis_pass1/stage2/</code> (code ed0b1162, bundles frozen 11-07). run_145 beam, all 24
acquisitions, A B C D; run_149 cosmic, all 87 sub-runs (A–C line) and the 14 sub-runs of the in-situ truth (paired with
is2_v2's cosmic recipe <code>is2ts_s3</code>). Criteria from <code>NTOF_PASS_PLAN_2026-11-02.md</code> §3 (N-G1):
C1 efficiency and wall-confirmed fraction ≥ is2_v2 − {TOL_PT} pt; C2 A–C line s68 ≤ is2_v2; C3 source y within its
error of −31 (A) / −35 (C) mm; G5 the capsule-focus vs muon scale tension closed or quantified. |tan| &lt; {TAN_MAX}
on both chains everywhere. <b>D's beam numbers use the L2 pin, which is being rescanned (condor 4674646): they may
change.</b></p>
<h2>Verdict per arm and criterion</h2>{table(vt)}
<h2>C1 — efficiency and wall confirmation (run_145 beam)</h2>
<p>One trigger set per sub-run for both chains (the union of the pass's four arm tables and is2_v2's stage-3 events).
Wall confirmation = SiPM-wall match minus the same-width off-time control (<code>det_a_scint.match_run</code>), on gated
tracks predicted on the wall.</p>
<img src="figures/eff_wall.png" alt="gated and wall-confirmed tracks per trigger">
{table(R['eff'].reset_index())}{table(R['wall'])}
<p>Per gated track the pass confirms less often (A {N['A']['conf']['purity']:.3f} vs {N['A']['conf']['purity_is2']:.3f},
C {N['C']['conf']['purity']:.3f} vs {N['C']['conf']['purity_is2']:.3f}). That is composition: on the same single-track
events both chains confirm equally, and the tracks only the pass has are about half confirmed, against a control near 1 %:</p>
{table(R['split'].reset_index())}
<p>Cosmic truth events (run_149, 815 per arm): both chains fit both planes of every event (fit-level efficiency 1.000).
After the quality rule the pass keeps A {R['ceff'].loc['A', 'pass_eff']:.3f} / C {R['ceff'].loc['C', 'pass_eff']:.3f}
against {R['ceff'].loc['A', 'is2_eff']:.3f} / {R['ceff'].loc['C', 'is2_eff']:.3f}. The rule is relative (χ²/dof ≤ 5 ×
the table's median); is2ts_s3 was run on an allowlist of the truth events only, so its median and hence its cut are up to
2–3× looser on C (thresholds below). Not a like-for-like efficiency; the beam comparison above is.</p>
{table(R['ceff'].reset_index())}
<h2>C2 — angle resolution against the A–C line</h2>
<p>Paired on the truth events both chains keep (test two thirds), per view, all |tan| &lt; 0.6. "Rescaled" = each chain
divided by its own median fit/true on the train third (resolution only); "own v" = as reconstructed.</p>
<img src="figures/cos_s68.png" alt="angle residual s68 per arm and view">
{table(R['cres'][['arm', 'view', 'ratio_pass', 'ratio_is2', 's68_pass_scaled_0.0_0.6', 's68_is2_scaled_0.0_0.6',
                  'd_s68_scaled', 'd_s68_scaled_err', 's68_pass_raw_0.0_0.6', 's68_is2_raw_0.0_0.6']])}
<p>The pass's own A–C line over all of run_149 ({line['n_ac_line']:,} lines): 3D residual s68 A
{line['A_res68']:.2f} ± {line['A_res68_err']:.2f}°, C {line['C_res68']:.2f} ± {line['C_res68_err']:.2f}°; angle scale
fit/true A x {line['Ax']['ratio']:.4f} / y {line['Ay']['ratio']:.4f}, C x {line['Cx']['ratio']:.4f} / y
{line['Cy']['ratio']:.4f}. C reads ~4 % steep at the frozen v 28.1 µm/ns: the known C scale (0.99–1.09 by run,
accepted at the freeze, corrected at table level by P7a).</p>{table(lb, '{:.3g}')}
<h2>C3 — source y (run_145, ≥ 20 ms band crossing, <code>y_guard.late_y_crossing</code>)</h2>
{table(pd.DataFrame([dict(chain=c, sample=s, arm=a, y_mm=(R['srcy'][c] if s == 'all' else R['srcy']['common'][c])[a]['y_cross_mm'],
                          err_mm=(R['srcy'][c] if s == 'all' else R['srcy']['common'][c])[a]['err_mm'],
                          n=(R['srcy'][c] if s == 'all' else R['srcy']['common'][c])[a]['n'])
                     for c in ('pass', 'is2_v2') for s in ('all', 'same events') for a in 'AC']))}
<p>A agrees with −31. C reads −33.2, 4σ (statistical) above −35 and 3.0 mm above is2_v2 on the same events, so the shift is
C's reconstruction (positions or y tan), not the added tracks. It is inside the estimator's ±3 mm A/C systematic
(the reference values themselves span −31 … −36 across chains), and it brings C toward A (A − C 1.7 mm, is2_v2 3.8).</p>
<h2>G5 — capsule focus vs the muon scale</h2>
<p>run_145, 20–80 ms, x view, source offset dx = −9.5 mm (<code>focus_offset.py</code>): the tan scale s that makes the
median miss at the capsule axis smallest. The muon side is the pass's own A–C cosmic line (fit/true on x). Tension =
s × (fit/true): 1 means the capsule and the muons agree.</p>
<img src="figures/focus.png" alt="sharpest focus scale per tan bin">{table(R['focus'], '{:.3f}')}
<h2>What this does not rule out</h2>
<ul>
<li>One beam run (run_145) and one cosmic run (run_149). Other periods are not scored here; the pre-freeze checks
(plan §6 11-06) covered efficiency and pointing on runs 79, 81, 116, 162.</li>
<li>B and D have no is2_v2 reference (is2_v2 reconstructed A and C only), so N-G1 cannot pass or fail them; their
absolute numbers are given. D's beam numbers are on the L2 pin under rescan.</li>
<li>Wall confirmation says a track is a real particle that reached the wall, not that its angle is right (G6: electrons
scatter). The extra pass tracks are ~half confirmed: some of the rest may be real particles that miss the wall
acceptance, some not.</li>
<li>The cosmic efficiency comparison is not like-for-like (relative quality rule on an allowlist table); only the fit
level is.</li>
<li>C3's reference values carry a ±3 mm systematic of the estimator (efficiency tilts, chamber-to-chamber); which of
the pass's −33.2 or is2_v2's −35.8 is closer to the truth is not decided here.</li>
<li>G5: the focus estimator is unbiased to ±3 % in Geant4 with the simulated source; a source model that differs
from the data (vertex distribution, electron angle response) would move s. The remaining 8 % on A is not explained.</li>
</ul>
<p style="color:#718096;font-size:12px">Generated by rerun/ntof/ng1/make_report.py from results/ of ng1_score.py.</p>
</body></html>'''
    open(f'{W}/report.html', 'w').write(html)
    vt.to_csv(f'{OUT}/verdicts.csv', index=False)
    print(vt.to_string(index=False))
    print(f'-> {W}/report.html')


if __name__ == '__main__':
    main()
