#!/usr/bin/env python3
"""hv_metrics.py --rows DIR [--out DIR] -- Paper I: what chamber A's resist-HV scan (6-27, 1000 V drift) changes.

Dylan 2026-11-03: chamber A (det3) looks on its efficiency plateau down to 430 V, so why run at 490 V? Efficiency
alone does not say; this puts every per-row metric of the scan on one HV axis:
  (a) interior efficiency (rerun/effbudget/pass1 rows.csv eff_ig, R = 5 mm) and the tracks lost to discharges
      (figures/eff_vs_hv_discharge.csv);
  (b) angle resolution s68 per view, pass arm (paper1_res rows: views.<v>.arms.pass.all.s68);
  (c) position residual core at the mesh per view (same rows: arms.pass.position.mesh.core; M3 pointing included);
  (d) time resolution: the uncorrelated part (t_X - t_Y)/sqrt2 and the event time (rerun/paper1_timing rows.csv);
  (e) a gain proxy, per track from the pass-1 events: median fitted charge q_X + q_Y and the fraction of tracks with a
      strip at the ADC rail (n_sat_strips > 0 in either view) (Dylan 2026-11-03);
  (f) the angle scale: robust slope of the fitted tan against the M3 tan per view (|tan_M3| > 0.05, 3 MAD trimmed).
Tracks: M3 rays in the interior, matched within 5 mm (paper1_eff rays_det3.parquet), joined to wft/events.parquet.
Result (2026-11-03): efficiency, angle and position are flat from 425 to ~505 V and the angle scale is 1.00 +- 0.01 at
every HV, while the charge grows x13.7 (one doubling per ~17 V, saturation 0 -> 24 %) and the time resolution
improves 109 -> 23 ns from 425 to 490 V.  Per track, the angle residual does not depend on the track's charge.

--rows   the paper1_res per-row JSONs (EOS x17rerun/paper1_res/results_18efcd0d/rows)
--events Analysis_pass1/mx17_det3_saturday_scan_6-27-26 (<subrun>/events.parquet or <subrun>/mx17_3/wft/events.parquet)
--rays   paper1_eff budget/mx17_det3_saturday_scan_6-27-26 (<subrun>/mx17_3/rays_det3.parquet)
all fetched from EOS over xrootd.  Writes figures/hv_metrics.{pdf,png,csv} and results/hv_metrics.json.
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


def _scale(x, y):
    """Robust straight-line slope of y against x (|x| > 0.05, iterated 3 MAD trim) and its error."""
    k = np.abs(x) > 0.05
    A = np.vstack([x[k], np.ones(k.sum())]).T; yk = y[k]; m = np.ones(k.sum(), bool)
    for _ in range(4):
        c, *_ = np.linalg.lstsq(A[m], yk[m], rcond=None); r = yk - A @ c
        m = np.abs(r) < 3 * 1.4826 * np.median(np.abs(r[m]))
    cov = np.linalg.inv(A[m].T @ A[m]) * np.var(r[m])
    return float(c[0]), float(np.sqrt(cov[0, 0]))


def per_track(df, events_dir, rays_dir):
    """Gain proxy and angle scale per row, from the pass-1 events of the M3-matched interior tracks."""
    add = []
    for sub in df.subrun:
        ev = [p for p in (os.path.join(events_dir, sub, 'events.parquet'),
                          os.path.join(events_dir, sub, 'mx17_3', 'wft', 'events.parquet')) if os.path.exists(p)][0]
        E = pd.read_parquet(ev)
        R = pd.read_parquet(os.path.join(rays_dir, sub, 'mx17_3', 'rays_det3.parquet')).drop_duplicates('event_id')
        R = R[(~R.not_measured) & (R.region == 'interior') & R.good_x & (R.r <= 5)]
        M = E.merge(R[['event_id', 'tan_x', 'tan_y']], on='event_id')
        M = M[(M.x_ok == True) & (M.y_ok == True)]  # noqa: E712
        q = M.x_q_sum + M.y_q_sum
        o = dict(subrun=sub, n_tracks=len(M), q50=float(q.median()),
                 q50_err=float(1.2533 * q.std() / np.sqrt(len(q))),
                 f_sat=float(((M.x_n_sat_strips > 0) | (M.y_n_sat_strips > 0)).mean()))
        for v in 'xy':
            o[f'scale_{v}'], o[f'scale_{v}_err'] = _scale(M[f'tan_{v}'].to_numpy(), M[f'{v}_tan_theta'].to_numpy())
        add.append(o)
    return df.merge(pd.DataFrame(add), on='subrun', how='left')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', required=True)
    ap.add_argument('--events', required=True)
    ap.add_argument('--rays', required=True)
    ap.add_argument('--out', default=HERE)
    a = ap.parse_args()
    df = per_track(load(a.rows), a.events, a.rays)
    ps.use()
    col = ps.DET_COLOR['det3']
    # 3 x 2, one shared HV axis (ticks on the bottom row only). Every panel's quantity has its own marker shape; the
    # X/Y panels draw X filled / Y open (ps.VIEW_LS) in that shape, so a shape names the quantity, fill names the view.
    fig, ax2d = ps.fig_for('full', nrows=3, ncols=2, height=5.4, sharex=True)
    axs = ax2d.ravel()
    V = df.resist_V.to_numpy()
    ms = 3.6
    leg = dict(fontsize=ps.BASE_PT * 0.72, handlelength=1.6, borderpad=0.3, labelspacing=0.25)

    def ser(ax, x, y, e, mk, filled=True, c=col, ls='-', **kw):
        return ax.errorbar(x, y, yerr=e, color=c, marker=mk, ms=ms, lw=0.9, ls=ls, mfc=c if filled else 'white',
                           mew=0.9, elinewidth=0.8, capsize=0, **kw)

    # (a) gain proxy: median fitted charge (log: the gain is exponential in V) and the saturated fraction
    ax = axs[0]
    ser(ax, V, df.q50, df.q50_err, 'D', label='median charge $q_X+q_Y$')
    ax.set_yscale('log'); ax.set_ylim(400, 3e5); ax.set_ylabel('median track\ncharge [ADC]')
    lo = df.resist_V <= 465                       # no saturation yet: the exponential fit
    b = np.polyfit(df.resist_V[lo], np.log2(df.q50[lo]), 1)
    vv = np.linspace(420, 490, 50)
    ax.plot(vv, 2 ** np.polyval(b, vv), color=ps.MODEL, lw=0.7, ls=':', zorder=1)
    ax.text(0.04, 0.95, f'$\\times2$ per {1 / b[0]:.1f} V', transform=ax.transAxes, va='top',
            fontsize=ps.BASE_PT * 0.75, color=ps.INK)
    t = ax.twinx(); t.grid(False); t.spines['right'].set_visible(True)
    t.plot(V, 100 * df.f_sat, color=ps.ACCENT, marker='*', ms=ms + 2, lw=0.9, ls='-')
    t.set_ylim(0, 105); t.set_ylabel('saturated\ntracks [%]', color=ps.ACCENT, labelpad=2)
    t.tick_params(axis='y', colors=ps.ACCENT)
    # (b) efficiency and discharge loss
    ax = axs[1]
    ser(ax, V, df.eff_ig, df.eff_ig_err, 's', label='interior efficiency')
    ax.plot(V, df.loss_discharge_pct, color=col, marker='s', ms=ms, mfc='white', ls='--', lw=0.9, mew=0.9,
            label='lost to a discharge')
    ax.set_ylim(0, 100); ax.set_ylabel('reference tracks [%]')
    ax.legend(loc='center left', **leg)
    # (c) time
    ax = axs[2]
    k = df.sigma_xy.notna()
    ser(ax, V[k], df.sigma_xy[k], df.sigma_xy_err[k], 'P', label=r'$(t_X-t_Y)/\sqrt{2}$')
    ser(ax, V[k], df.sigma_evt[k], df.sigma_evt_err[k], 'P', filled=False, ls='--', label='event time')
    ax.set_ylim(0, 120); ax.set_ylabel(r'time $\sigma$ ($s_{68}$) [ns]')
    ax.legend(loc='upper right', **leg)
    # (d) angle resolution, (e) angle scale, (f) position core: one marker shape each, X filled / Y open
    for ax, q, lab, lim, mk in ((axs[3], 's68', r'angle $s_{68}$ [deg]', (0, 1.5), 'o'),
                                (axs[4], 'scale', 'angle scale\n(fit / reference tan)', (0.9, 1.1), '^'),
                                (axs[5], 'pos_core', 'position core\nat mesh [mm]', (0, 0.8), 'v')):
        for v in 'xy':
            ser(ax, V, df[f'{q}_{v}'], df[f'{q}_{v}_err'], mk, filled=(v == 'x'), ls=ps.VIEW_LS[v], label=v.upper())
        ax.set_ylabel(lab); ax.set_ylim(*lim)
        ax.legend(loc='lower left', ncol=2, **leg)
    axs[4].axhline(1.0, color=ps.MUTED, lw=0.7, zorder=0)
    for ax in axs:
        ax.axvline(GOLDEN_V, color=ps.ACCENT, lw=0.8, ls=':', zorder=0)
        ax.set_xlim(418, 532)
    for ax in axs[4:]:
        ax.set_xlabel('resist HV [V]')
    axs[1].text(GOLDEN_V - 2, 14, 'Runs 1, 2', color=ps.ACCENT, ha='right', fontsize=ps.BASE_PT * 0.75)
    ps.panel_labels(axs)
    fig.get_layout_engine().set(h_pad=0.07, w_pad=0.05)
    ps.save(fig, os.path.join(a.out, 'figures', 'hv_metrics'), data=df)
    flat = df[(df.resist_V >= 425) & (df.resist_V <= 505)]
    summ = dict(run=RUN, n_rows=int(len(df)),
                range_425_505=dict(eff_ig=[float(flat.eff_ig.min()), float(flat.eff_ig.max())],
                                   s68_x=[float(flat.s68_x.min()), float(flat.s68_x.max())],
                                   s68_y=[float(flat.s68_y.min()), float(flat.s68_y.max())],
                                   pos_core_x=[float(flat.pos_core_x.min()), float(flat.pos_core_x.max())],
                                   pos_core_y=[float(flat.pos_core_y.min()), float(flat.pos_core_y.max())]),
                gain_doubling_V=float(1 / b[0]),
                q50={int(r.resist_V): float(r.q50) for r in df.itertuples()},
                f_sat={int(r.resist_V): float(r.f_sat) for r in df.itertuples()},
                angle_scale_425_510=dict(x=[float(df.scale_x[df.resist_V <= 510].min()), float(df.scale_x[df.resist_V <= 510].max())],
                                         y=[float(df.scale_y[df.resist_V <= 510].min()), float(df.scale_y[df.resist_V <= 510].max())]),
                sigma_xy_ns={int(r.resist_V): float(r.sigma_xy) for r in df.itertuples() if np.isfinite(r.sigma_xy)})
    os.makedirs(os.path.join(a.out, 'results'), exist_ok=True)
    json.dump(summ, open(os.path.join(a.out, 'results', 'hv_metrics.json'), 'w'), indent=1)
    print(json.dumps(summ['range_425_505'], indent=1))


if __name__ == '__main__':
    main()
