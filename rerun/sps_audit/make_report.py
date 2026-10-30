#!/usr/bin/env python3
"""make_report.py [--tel DIR]: report/report.html + report/figures/*.png from results/ (and the telescope tables in
--tel for the scatter figure).  Relative figure links, so the page works from disk, the DAQ page or a copy."""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
REP = os.path.join(HERE, 'report')
FIG = os.path.join(REP, 'figures')
DEF_TEL = '/media/dylan/data/x17/sps_audit/tel'


def um(x):
    return '—' if x is None or not np.isfinite(x) else f'{x * 1e3:.0f}'


def fig_pass_hists():
    H = json.load(open(os.path.join(RES, 'pass_hists.json')))
    e = np.array(H['edges']); c = 0.5 * (e[1:] + e[:-1])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for k, col in (('run71_raw__raw700', 'C3'), ('run70_flat_drift__d700', 'C0'), ('run55_flatdrift__d700', 'C2')):
        for i, v in enumerate('xy'):
            h = np.array(H[k][v], float)
            ax[i].step(c, h / h.sum(), where='mid', color=col, label=k)
    for i, v in enumerate('XY'):
        ax[i].set_xlabel(f'pass-1 det4 {v} p0 [mm]'); ax[i].set_ylabel('fraction / mm'); ax[i].set_yscale('log')
    for x, lab in ((158.34, 'ch 372'), (250.38, 'ch 510')):
        ax[1].axvline(x, color='k', ls=':', lw=1); ax[1].text(x + 1, 0.2, lab, fontsize=8)
    ax[1].legend(fontsize=8, loc='upper right')
    ax[0].set_title('X: the RAW row and its ZS twins agree'); ax[1].set_title('Y: the RAW row sits on the two oscillating strips')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'pass_y_pileup.png'), dpi=110); plt.close(fig)


def fig_scatter(tel):
    pan = [('r71_', 'run71 raw700, followup telescope: pre-fix seed (red) / fixed (blue)', True),
           ('r71v2_', 'run71, ZS-like telescope, fixed seed', False), ('r66_', 'run66 ZS (CF4, same mount)', False)]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for k, (pref, title, old) in enumerate(pan):
        fs = sorted(glob.glob(os.path.join(tel, pref + '[0-9][0-9][0-9].parquet')))
        if not fs:
            continue
        d = pd.concat([pd.read_parquet(f) for f in fs])
        if 'fx_size' in d:
            d = d[(d.fx_n == 1) & (d.fy_n == 1) & (d.bx_n == 1) & (d.by_n == 1)]
        tx = d.fx_p + (d.bx_p - d.fx_p) * 1120 / 1370; ty = d.fy_p + (d.by_p - d.fy_p) * 1120 / 1370
        if old:
            ax[0, k].plot(tx, np.where(d.old_seed_y >= 0, d.old_y_p0, d.new_y_p0), '.', color='C3', ms=2, alpha=.5, label='pre-fix seed')
        ax[0, k].plot(tx, d.new_y_p0, '.', color='C0', ms=2, alpha=.4, label='fixed seed')
        ax[1, k].plot(ty, d.new_x_p0, '.', color='C0', ms=2, alpha=.4)
        ax[0, k].set_title(title, fontsize=10); ax[0, k].set_ylim(90, 310); ax[1, k].set_ylim(90, 310)
        ax[0, k].set_xlabel('telescope x at det4 [mm]'); ax[0, k].set_ylabel('det4 Y (pass p0) [mm]')
        ax[1, k].set_xlabel('telescope y at det4 [mm]'); ax[1, k].set_ylabel('det4 X (pass p0) [mm]')
        if old:
            ax[0, k].legend(fontsize=8, markerscale=4)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'scatter.png'), dpi=90); plt.close(fig)


def fig_resid(A):
    R = np.load(os.path.join(RES, 'audit_resid.npz'))
    sets = [k for k in ('run66 r755+r780 (ZS32, CF4)', 'run63 flat700 (ZS64, CF4)', 'run71 raw700 (RAW64, CF4)') if k in A]
    fig, ax = plt.subplots(2, len(sets), figsize=(5.2 * len(sets), 7.5), squeeze=False)
    bins = np.linspace(-3, 3, 61)
    for j, s in enumerate(sets):
        for i, v in enumerate('xy'):
            for est, col in (('pass', 'C0'), ('pass_mid', 'C2'), ('centroid', 'C1'), ('pass_old', 'C3')):
                key = f'{s}|{est}|{v}'
                if key not in R.files:
                    continue
                r = R[key]; r = r[np.isfinite(r)]
                ax[i, j].hist(r, bins=bins, histtype='step', color=col, label=f'{est} (n {len(r)})')
            ax[i, j].set_title(f'{s}: det4 {v.upper()}', fontsize=9); ax[i, j].set_xlabel('residual to telescope [mm]')
            ax[i, j].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'residuals.png'), dpi=100); plt.close(fig)


def zone_txt(V):
    return ', '.join(f'{q["pitch"]}: {q["sigma_res"] * 1e3:.0f}→{q["sigma_det4"] * 1e3:.0f} (n {q["n"]})' for q in V['zones']) or '—'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tel', default=DEF_TEL)
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    A = json.load(open(os.path.join(RES, 'audit.json')))
    B = json.load(open(os.path.join(RES, 'baseline.json')))
    P = pd.read_csv(os.path.join(RES, 'pile_rows.csv'))
    S = pd.read_csv(os.path.join(RES, 'seed_census.csv'), keep_default_na=False) if os.path.exists(os.path.join(RES, 'seed_census.csv')) else None
    fig_pass_hists()
    if os.path.isdir(a.tel):
        fig_scatter(a.tel)
    fig_resid(A)
    h = ['<!DOCTYPE html><html><head><meta charset="utf-8"><title>SPS det4 audit 2026-10-30</title>',
         '<style>body{font-family:sans-serif;max-width:1150px;margin:auto;padding:1em}table{border-collapse:collapse;font-size:13px}'
         'td,th{border:1px solid #bbb;padding:3px 7px}th{background:#eee}img{max-width:100%}.v{background:#fff6d6;padding:.6em;border:1px solid #e0c060}</style>',
         '</head><body><h1>det4 at SPS H4: audit of the pass-1 SPS chain (2026-10-30)</h1>',
         '<p>Generated by <code>rerun/sps_audit/make_report.py</code> from <code>results/</code>. The text and the reasoning are in '
         '<code>rerun/sps_audit/README.md</code>.</p>']
    # baseline
    h.append('<h2>1. The run-53 measurement is reproduced by the audit\'s machinery</h2><table><tr><th>sample</th><th>subset</th><th>n</th>'
             '<th>det(A)</th><th>det4 X zones (res→det4, µm)</th><th>det4 Y zones</th><th>3-zone fit det4 Y</th></tr>')
    for name, D in B.items():
        for sub in ('all', 'all_seeded', 'amax300', 'amax_lt300'):
            R = D[sub]; fit = R['det4_Y'].get('fit')
            h.append(f'<tr><td>{name}</td><td>{sub}</td><td>{R["n"]}</td><td>{R["det_A"]:+.4f}</td><td>{zone_txt(R["det4_X"])}</td>'
                     f'<td>{zone_txt(R["det4_Y"])}</td><td>{"f=%.3f, %s ± %s µm" % (fit["f_back"], um(fit["sigma_det4"]), um(fit["sigma_det4_err"])) if fit else "—"}</td></tr>')
    h.append('</table>')
    # pile-up
    h.append('<h2>2. The bug: oscillating channels seed the Y window</h2><img src="figures/pass_y_pileup.png">')
    cols = ['key', 'n_y_ok', 'excess_ch372', 'excess_ch510', 'excess_total']
    h.append('<p>Pass-1 rows, Y fits in the 4.5 mm bin of each oscillating strip minus the neighbouring bins (top 22):</p><table><tr>' +
             ''.join(f'<th>{c}</th>' for c in cols) + '</tr>')
    for _, r in P.sort_values('excess_total', ascending=False).head(22).iterrows():
        h.append('<tr>' + ''.join(f'<td>{r[c]:.3f}</td>' if isinstance(r[c], float) else f'<td>{r[c]}</td>' for c in cols) + '</tr>')
    h.append('</table>')
    if S is not None:
        h.append('<p>Events whose pre-fix seed the fix excludes (seed_census.py), i.e. the rows to re-run:</p><table><tr><th>row</th><th>events</th>'
                 '<th>changed</th><th>fraction</th><th>hot channels (this row)</th><th>pre-fix seeds on excluded channels</th></tr>')
        for _, r in S[S.n_changed > 0].sort_values('frac', ascending=False).iterrows():
            h.append(f'<tr><td>{r.key}</td><td>{r.n_events}</td><td>{r.n_changed}</td><td>{float(r.frac):.4f}</td><td>{r.hot}</td><td>{r.seeds_on_excluded}</td></tr>')
        h.append('</table>')
    # resolution
    h.append('<h2>3. det4 against the telescope, per dataset and estimator</h2><img src="figures/scatter.png">'
             '<table><tr><th>dataset</th><th>estimator</th><th>det(A)</th><th>roll</th><th>both-view core</th>'
             '<th>det4 X: n, zones res→det4 (µm)</th><th>det4 Y: n, zones</th></tr>')
    for name, D in A.items():
        for est, E in D['estimators'].items():
            h.append(f'<tr><td>{name}</td><td>{est}</td><td>{E["det_A"]:+.3f}</td><td>{E["roll_deg"]:+.1f}°</td><td>{E["both_core"]:.2f}</td>'
                     f'<td>{E["det4_X"]["n_fid"]}: {zone_txt(E["det4_X"])}</td><td>{E["det4_Y"]["n_fid"]}: {zone_txt(E["det4_Y"])}</td></tr>')
    h.append('</table><img src="figures/residuals.png">')
    h.append('</body></html>')
    open(os.path.join(REP, 'report.html'), 'w').write('\n'.join(h))
    print('wrote', os.path.join(REP, 'report.html'))


if __name__ == '__main__':
    main()
