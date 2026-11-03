#!/usr/bin/env python3
"""hv_metrics.py --rows DIR [--out DIR] -- Paper I: what chamber A's resist-HV scan (6-27, 1000 V drift) changes.

Dylan 2026-11-03: chamber A (det3) looks on its efficiency plateau down to 430 V, so why run at 490 V? Efficiency
alone does not say; this puts every per-row metric of the scan on one HV axis:
  (a) interior efficiency (rerun/effbudget/pass1 rows.csv eff_ig, R = 5 mm) and the tracks lost to discharges
      (figures/eff_vs_hv_discharge.csv);
  (b) angle resolution s68 per view, pass arm (paper1_res rows: views.<v>.arms.pass.all.s68);
  (c) position residual core at the mesh per view (same rows: arms.pass.position.mesh.core; M3 pointing included);
  (d) time resolution: the uncorrelated part (t_X - t_Y)/sqrt2 and the event time (rerun/paper1_timing rows.csv).
Result (2026-11-03): (a)-(c) are flat from 425 to ~505 V; only (d) moves, 109 -> 23 ns from 425 to 490 V.

DIR holds the paper1_res per-row JSONs (EOS x17rerun/paper1_res/results_18efcd0d/rows, fetched over xrootd).
Writes figures/hv_metrics.{pdf,png,csv} and results/hv_metrics.json.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'papers'))
import paper_style as ps  # noqa: E402

RUN = 'mx17_det3_saturday_scan_6-27-26'
GOLDEN_V = 490


def load(rows_dir):
    T = pd.read_csv(os.path.join(REPO, 'rerun', 'effbudget', 'pass1', 'results', 'rows.csv')).set_index('key')
    tim = pd.read_csv(os.path.join(REPO, 'rerun', 'paper1_timing', 'results', 'rows.csv')).set_index('key')
    dis = pd.read_csv(os.path.join(HERE, 'figures', 'eff_vs_hv_discharge.csv'))
    dis = dis[(dis.det == 'mx17_3') & (dis.run == RUN)].set_index('resist_V')
    out = []
    for f in sorted(glob.glob(os.path.join(rows_dir, f'*{RUN}*mx17_3*.json'))):
        d = json.load(open(f))
        if d['drift_V'] != 1000 or not d['subrun'].startswith('hv_scan'):
            continue
        k, V = d['key'], d['resist_V']
        r = dict(key=k, subrun=d['subrun'], resist_V=V)
        for v in 'xy':
            a = d['views'][v]['arms']['pass']
            r[f's68_{v}'], r[f's68_{v}_err'], r[f'n_{v}'] = a['all']['s68'], a['all']['s68_err'], a['all']['n']
            pm = a['position']['mesh']
            r[f'pos_core_{v}'], r[f'pos_core_{v}_err'] = pm['core'], pm.get('core_err', np.nan)
        if k in T.index:
            r['eff_ig'], r['eff_ig_err'] = T.loc[k, 'eff_ig'], T.loc[k, 'err_ig']
        if k in tim.index:
            for c in ('sigma_xy', 'sigma_xy_err', 'sigma_evt', 'sigma_evt_err', 'n_clean'):
                r[c] = tim.loc[k, c]
        if V in dis.index:
            r['loss_discharge_pct'] = float(np.atleast_1d(dis.loc[V, 'loss_discharge_pct'])[0])
        out.append(r)
    return pd.DataFrame(out).sort_values('resist_V').reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', required=True)
    ap.add_argument('--out', default=HERE)
    a = ap.parse_args()
    df = load(a.rows)
    ps.use()
    col = ps.DET_COLOR['det3']
    fig, axs = ps.fig_for('full', nrows=1, ncols=4, height=2.25, sharex=True)
    V = df.resist_V.to_numpy()
    ax = axs[0]
    ax.errorbar(V, df.eff_ig, yerr=df.eff_ig_err, color=col, marker='o', ms=3, lw=1.0, label='interior efficiency')
    ax.plot(V, df.loss_discharge_pct, color=col, marker='o', ms=3, mfc='white', ls='--', lw=0.9,
            label='lost to a discharge')
    ax.set_ylim(0, 100); ax.set_ylabel('fraction of M3 tracks [%]')
    ax.legend(loc='center left', fontsize=ps.BASE_PT * 0.72, handlelength=1.5)
    for i, (q, lab, sc) in enumerate((('s68', r'angle $s_{68}$ [deg]', 1.0), ('pos_core', 'position core at mesh [mm]', 1.0))):
        ax = axs[i + 1]
        for v in 'xy':
            ax.errorbar(V, df[f'{q}_{v}'] * sc, yerr=df[f'{q}_{v}_err'] * sc, color=col, marker='o', ms=3,
                        ls=ps.VIEW_LS[v], lw=1.0, mfc=col if v == 'x' else 'white', label=v.upper())
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1.5 if q == 's68' else 0.8)
        ax.legend(loc='lower left', fontsize=ps.BASE_PT * 0.75, ncol=2)
    ax = axs[3]
    k = df.sigma_xy.notna()
    ax.errorbar(V[k], df.sigma_xy[k], yerr=df.sigma_xy_err[k], color=col, marker='o', ms=3, lw=1.0,
                label=r'$(t_X-t_Y)/\sqrt{2}$')
    ax.errorbar(V[k], df.sigma_evt[k], yerr=df.sigma_evt_err[k], color=col, marker='o', ms=3, mfc='white', ls='--',
                lw=0.9, label='event time')
    ax.set_ylim(0, 120); ax.set_ylabel(r'time $\sigma$ ($s_{68}$) [ns]')
    ax.legend(loc='upper right', fontsize=ps.BASE_PT * 0.72, handlelength=1.5)
    for ax in axs:
        ax.axvline(GOLDEN_V, color=ps.ACCENT, lw=0.8, ls=':', zorder=0)
        ax.set_xlabel('resist HV [V]'); ax.set_xlim(418, 532)
    axs[0].text(GOLDEN_V - 2, 4, 'Runs 1, 2', color=ps.ACCENT, ha='right', fontsize=ps.BASE_PT * 0.75)
    ps.panel_labels(axs)
    ps.save(fig, os.path.join(a.out, 'figures', 'hv_metrics'), data=df)
    flat = df[(df.resist_V >= 425) & (df.resist_V <= 505)]
    summ = dict(run=RUN, n_rows=int(len(df)),
                range_425_505=dict(eff_ig=[float(flat.eff_ig.min()), float(flat.eff_ig.max())],
                                   s68_x=[float(flat.s68_x.min()), float(flat.s68_x.max())],
                                   s68_y=[float(flat.s68_y.min()), float(flat.s68_y.max())],
                                   pos_core_x=[float(flat.pos_core_x.min()), float(flat.pos_core_x.max())],
                                   pos_core_y=[float(flat.pos_core_y.min()), float(flat.pos_core_y.max())]),
                sigma_xy_ns={int(r.resist_V): float(r.sigma_xy) for r in df.itertuples() if np.isfinite(r.sigma_xy)})
    os.makedirs(os.path.join(a.out, 'results'), exist_ok=True)
    json.dump(summ, open(os.path.join(a.out, 'results', 'hv_metrics.json'), 'w'), indent=1)
    print(json.dumps(summ['range_425_505'], indent=1))


if __name__ == '__main__':
    main()
