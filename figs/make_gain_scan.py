#!/usr/bin/env python3
"""make_gain_scan.py -- Paper I v2: chamber A's resist-HV scan (27 June, 1000 V drift) on one figure, and the numbers
that settle three disagreements between the old figures (hv_metrics, timing (a), hv_metrics_extra (a)/(c)).

    .venv/bin/python papers/paper1_v2/figs/make_gain_scan.py [--timing-rows DIR] [--events DIR]

Panels (shared resist-HV axis, dotted line at 490 V = Runs 1 and 2):
  (a) gain proxy: the rank-censored 25th percentile of the track charge (rerun/paper1_charge, cens_25: unsaturated
      events give the values, a saturated event only its rank; valid while f_sat < 0.6) on a log axis with the
      exponential fitted to the scan sub-runs 425-495 V; right axis: the fraction of tracks with a strip at the rail.
  (b) angle s68, X and Y (rerun/paper1_ops/figures/hv_metrics.csv, the paper1_res rows); right axis: the angle scale
      (robust slope of fitted against reference tan, same csv).
  (c) core of the position residual at the mesh, X and Y (same csv).
  (d) time resolution on reference-matched single tracks (rerun/paper1_timing matched_rows.py definition: clean
      triggers of timing_row.py joined to the pass-1 events table, both views ok, one track): the uncorrelated part
      s68(tX-tY)/sqrt2 and the event time s68((tX+tY)/2).  Recomputed here from the per-trigger npz with more
      bootstrap samples; the values equal rerun/paper1_timing/results/rows_matched.csv (checked, written to the json).
Inputs are all local result tables; nothing is reconstructed.  --timing-rows / --events default to the pairfix copies
under $WORK_DATA (~/work_data); without them panel (d) falls back to rows_matched.csv.
Writes out/gain_scan.{pdf,png,csv} and out/gain_scan.json.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'papers'))
sys.path.insert(0, os.path.join(REPO, 'rerun', 'paper1_timing'))
import paper_style as ps  # noqa: E402
from matched_rows import widths, s68  # noqa: E402

RUN = 'mx17_det3_saturday_scan_6-27-26'
GOLDEN_V = 490
WD = os.environ.get('WORK_DATA', os.path.expanduser('~/work_data'))
OUT = os.path.join(HERE, 'out')
rng = np.random.default_rng(17)


def wfit(V, y, e):
    """Weighted straight line y = a + b (V - Vm); returns b, its error scaled by sqrt(chi2/dof) when > 1, chi2, dof."""
    w = 1 / e ** 2
    A = np.column_stack([np.ones_like(V), V - V.mean()])
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    c = cov @ (A.T @ (w * y))
    chi2 = float(np.sum(w * (y - A @ c) ** 2)); dof = len(V) - 2
    be = float(np.sqrt(cov[1, 1]) * max(1.0, np.sqrt(chi2 / dof))) if dof > 0 else np.nan
    return dict(a=float(c[0]), b=float(c[1]), b_err=be, b_err_stat=float(np.sqrt(cov[1, 1])), Vm=float(V.mean()),
                chi2=chi2, dof=dof)


def doubling(f):
    return dict(doubling_V=float(np.log(2) / f['b']), doubling_err=float(np.log(2) / f['b'] ** 2 * f['b_err']),
                chi2=f['chi2'], dof=f['dof'])


# ---------------------------------------------------------------- (ii) gain
def gain():
    R = pd.read_csv(os.path.join(REPO, 'rerun', 'paper1_charge', 'results', 'rows.csv'))
    R = R[(R.run == RUN) & (R.drift_V == 1000)].sort_values('resist_V')
    sc = R[R.subrun.str.startswith('hv_scan')].copy()
    lr = R[R.subrun.str.startswith('long_run')]
    ok = sc[(sc.f_sat < 0.6) & np.isfinite(sc.cens_25)]

    def fit(lo, hi, extra=None):
        p = ok[(ok.resist_V >= lo) & (ok.resist_V <= hi)]
        if extra is not None:
            p = pd.concat([p, extra])
        return wfit(p.resist_V.to_numpy(float), np.log(p.cens_25.to_numpy()), (p.cens_25_err / p.cens_25).to_numpy())

    main = fit(425, 495)
    c = sc.set_index('resist_V')
    r = c.cens_25[490] / c.cens_25[425]
    r_err = r * np.hypot(c.cens_25_err[490] / c.cens_25[490], c.cens_25_err[425] / c.cens_25[425])
    # the old numbers, reproduced: paper1_charge's 16.7 V (its fit also takes in the 490 V long run, Run 1) and
    # hv_metrics' 18.8 V (median of all fitted charges, unweighted, 425-465 V)
    H = pd.read_csv(os.path.join(REPO, 'rerun', 'paper1_ops', 'figures', 'hv_metrics.csv'))
    lo = H.resist_V <= 465
    b18 = np.polyfit(H.resist_V[lo], np.log2(H.q50[lo]), 1)[0]
    g = dict(
        estimator='rank-censored 25th percentile of (q_X+q_Y) cos(theta), reference-matched single tracks >= 30 mm '
                  'inside the active area (rerun/paper1_charge cens_25); scan sub-runs only; f_sat < 0.6',
        fit_425_495=doubling(main),
        fit_425_460=doubling(fit(425, 460)), fit_470_495=doubling(fit(470, 495)),
        fit_425_465=doubling(fit(425, 465)),
        reproduce_paper1_charge_16p7_with_long_run=doubling(fit(425, 495, extra=lr[['resist_V', 'cens_25', 'cens_25_err']])),
        reproduce_hv_metrics_18p8_median_425_465=float(1 / b18),
        ratio_490_over_425=float(r), ratio_490_over_425_err=float(r_err),
        ratio_from_fit_425_495=float(np.exp(main['b'] * 65)),
        ratio_from_16p7=float(2 ** (65 / 16.7)), ratio_from_18p8=float(2 ** (65 / 18.8)),
        hv_metrics_median_ratio_490_over_425=float(H.set_index('resist_V').q50[490] / H.set_index('resist_V').q50[425]),
        average_doubling_425_490=float(65 * np.log(2) / np.log(r)),
        long_run_over_scan_490=float(lr.cens_25.iloc[0] / c.cens_25[490]),
        points={int(v): dict(cens_25=float(q), err=float(e), f_sat=float(f)) for v, q, e, f in
                zip(sc.resist_V, sc.cens_25, sc.cens_25_err, sc.f_sat)})
    return g, sc, main


def current_vs_gain(sc):
    T = pd.read_csv(os.path.join(REPO, 'rerun', 'paper1_ops', 'currents', 'results', 'currents.csv'))
    g = T[(T.run == RUN) & T.subrun.str.contains('hv_scan')].merge(sc[['subrun', 'cens_25']], on='subrun')
    g = g.assign(Vn=g.subrun.str.extract(r'resist_(\d+)V', expand=False).astype(int)).sort_values('Vn')
    floor = g[g.Vn <= 455]
    pf = np.polyfit(floor.V, floor.I_med, 1)
    out = dict(I_floor_425_455_uA=[float(floor.I_med.min()), float(floor.I_med.max())])
    k = (g.Vn >= 470) & (g.Vn <= 495)
    for name, f0 in (('total', 0 * g.V), ('above_linear_floor', np.polyval(pf, g.V))):
        ex = g.I_med - f0
        p, cov = np.polyfit(np.log(g.cens_25[k]), np.log(ex[k]), 1, cov=True)
        out[f'k_{name}_470_495'] = [float(p[0]), float(np.sqrt(cov[0, 0]))]
    a, b = g[g.Vn == 470].iloc[0], g[g.Vn == 495].iloc[0]
    out['I_ratio_495_over_470'] = float(b.I_med / a.I_med)
    out['G_ratio_495_over_470'] = float(b.cens_25 / a.cens_25)
    J = json.load(open(os.path.join(REPO, 'rerun', 'paper1_ops', 'currents', 'results', 'currents.json')))['det3_6-27_fit']
    out['old_fit'] = dict(V2_doubling_V=J['V2_doubling_V'], b_uA=J['b_uA'], b_err=J['b_err'],
                          note='fit I = b + A 2^((V-490)/V2) over 425-525 V, unweighted: set by the 490-525 V points '
                               'where the charge cannot be measured; b = 0.01 +- 0.04 uA misses the 0.17 uA measured '
                               'at 425-455 V')
    out['old_exponent_16p7_over_13p3'] = float(16.7 / J['V2_doubling_V'])
    return out


# ---------------------------------------------------------------- (i), (iii) timing
def timing(trows, events):
    M = pd.read_csv(os.path.join(REPO, 'rerun', 'paper1_timing', 'results', 'rows_matched.csv'))
    M = M[(M.run == RUN) & (M.drift_V == 1000) & M.subrun.str.startswith('hv_scan')].set_index('resist_V').sort_index()
    C = pd.read_csv(os.path.join(REPO, 'rerun', 'paper1_timing', 'results', 'rows.csv'))
    C = C[(C.run == RUN) & (C.drift_V == 1000) & C.subrun.str.startswith('hv_scan')].set_index('resist_V').sort_index()
    rows, win = [], {}
    files = sorted(glob.glob(os.path.join(trows, f'c26__{RUN}__hv_scan*_drift_1000V__mx17_3.npz')))
    for f in files:
        sub = os.path.basename(f).split('__')[2]
        V = int(sub.split('resist_')[1][:3])
        ev = os.path.join(events, RUN, sub, 'mx17_3', 'wft', 'events.parquet')
        if not os.path.exists(ev):
            continue
        z = np.load(f); tags = list(z['acq_tags'])
        D = pd.DataFrame(dict(acq_tag=[tags[i] for i in z['acq']], event_id=z['eid'], tx=z['tx'], ty=z['ty'],
                              ex=z['ex'], ey=z['ey'], amp=z['amp']))
        E = pd.read_parquet(ev, columns=['acq_tag', 'event_id', 'x_ok', 'y_ok', 'n_tracks'])
        m = D.merge(E, on=['acq_tag', 'event_id'], how='left', indicator=True)
        inE = (m._merge == 'both').to_numpy()
        mt = inE & (m.x_ok == True).to_numpy() & (m.y_ok == True).to_numpy() & (m.n_tracks == 1).to_numpy()  # noqa: E712
        r = dict(resist_V=V, n_clean=len(m), n_matched=int(mt.sum()), n_no_reference=int((~inE).sum()))
        for lab, sel in (('matched', mt), ('noref', ~inE)):
            s = m[sel]
            arr = [s[c].to_numpy(float) for c in ('tx', 'ty', 'ex', 'ey')]
            if len(s) >= 150:
                w = widths(*arr)
                bs = [widths(*[x[ii] for x in arr]) for ii in (rng.integers(0, len(s), len(s)) for _ in range(300))]
                r[f'{lab}_sigma_xy'], r[f'{lab}_sigma_evt'] = w['sigma_xy'], w['sigma_evt']
                r[f'{lab}_sigma_common'] = w['sigma_common']
                for k in ('sigma_xy', 'sigma_evt'):
                    r[f'{lab}_{k}_err'] = float(np.nanstd([b[k] for b in bs]))
                r[f'{lab}_amp50'] = float(np.median(s.amp))
        # the fixed early-amplitude window of timing_vs_hv_check.py, all clean triggers: sigma_xy AND the event time
        ok = np.isfinite(m.tx) & np.isfinite(m.ty)
        k = ok & (m.amp >= 1000) & (m.amp < 1600)
        if k.sum() >= 80:
            a, b = m.tx[k].to_numpy(float), m.ty[k].to_numpy(float)
            bs = np.array([(s68(a[i] - b[i]) / np.sqrt(2), s68((a[i] + b[i]) / 2))
                           for i in (rng.integers(0, len(a), len(a)) for _ in range(300))])
            win[V] = dict(n=int(k.sum()), sigma_xy=float(s68(a - b) / np.sqrt(2)), sigma_xy_err=float(bs[:, 0].std()),
                          sigma_evt=float(s68((a + b) / 2)), sigma_evt_err=float(bs[:, 1].std()),
                          amp_median=float(np.median(m.amp[k])))
        rows.append(r)
    T = pd.DataFrame(rows).set_index('resist_V').sort_index() if rows else None
    out = dict(sample='reference-matched single tracks: clean triggers (both views >= 5 pulse strips, one cluster, no '
                      'saturated strip) that are in the pass-1 events table with both views ok and one track')
    if T is not None:
        common = T.index.intersection(M.index[M.n >= 150])
        out['max_abs_diff_vs_rows_matched_ns'] = float(np.nanmax(np.abs(
            np.r_[T.loc[common, 'matched_sigma_xy'] - M.loc[common, 'sigma_xy'],
                  T.loc[common, 'matched_sigma_evt'] - M.loc[common, 'sigma_evt']])))
        src = T.rename(columns={'matched_sigma_xy': 'sigma_xy', 'matched_sigma_xy_err': 'sigma_xy_err',
                                'matched_sigma_evt': 'sigma_evt', 'matched_sigma_evt_err': 'sigma_evt_err'})
        out['source'] = 'recomputed from the timing_row npz + pass-1 events (300 bootstrap samples)'
    else:
        src = M[M.n >= 150]
        out['source'] = 'rerun/paper1_timing/results/rows_matched.csv (local npz not found)'
    src = src[np.isfinite(src.sigma_xy)]
    out['per_hv'] = {int(V): {k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                              for k, v in r.items() if pd.notna(v)} for V, r in (T if T is not None else src).iterrows()}
    out['all_clean_triggers_rows_csv'] = {int(V): dict(n=int(r.n_clean), sigma_xy=float(r.sigma_xy),
                                                       sigma_evt=float(r.sigma_evt))
                                          for V, r in C.iterrows() if np.isfinite(r.sigma_xy)}
    if win:
        vs = sorted(win)
        out['fixed_window_1000_1600_all_clean'] = {int(v): win[v] for v in vs}
        for q in ('sigma_xy', 'sigma_evt'):
            x = np.array([win[v][q] for v in vs]); e = np.array([win[v][q + '_err'] for v in vs])
            mu = float(np.sum(x / e ** 2) / np.sum(1 / e ** 2)); chi2 = float(np.sum(((x - mu) / e) ** 2))
            lf = wfit(np.array(vs, float), x, e)
            out[f'fixed_window_{q}_vs_constant'] = dict(mean=mu, chi2=chi2, dof=len(x) - 1,
                                                        p=float(stats.chi2.sf(chi2, len(x) - 1)),
                                                        slope_ns_per_V=lf['b'], slope_err=lf['b_err_stat'])
    J = json.load(open(os.path.join(REPO, 'rerun', 'paper1_ops', 'results', 'timing_vs_hv_check.json')))
    out['timing_vs_hv_check_chi2_dof_const'] = J['chi2_dof_const']
    return out, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timing-rows', default=os.path.join(WD, 'claude-jobs', 'pairfix', 'timing_rows'))
    ap.add_argument('--events', default=os.path.join(WD, 'claude-jobs', 'pairfix', 'pass1', 'Analysis_pass1'))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    G, sc, gfit = gain()
    I = current_vs_gain(sc)
    Tj, tim = timing(a.timing_rows, a.events)
    H = pd.read_csv(os.path.join(REPO, 'rerun', 'paper1_ops', 'figures', 'hv_metrics.csv')).sort_values('resist_V')

    ps.use()
    import matplotlib.ticker as mt
    from matplotlib.lines import Line2D
    col = ps.color('det3')
    fig, ax2d = ps.fig_for('full', nrows=2, ncols=2, height=4.0, sharex=True)
    axs = ax2d.ravel()
    ms = 3.6
    leg = dict(fontsize=ps.BASE_PT * 0.78, handlelength=1.8, borderpad=0.3, labelspacing=0.25, frameon=False)

    def ser(ax, x, y, e, mk, filled=True, c=col, ls='-', **kw):
        return ax.errorbar(x, y, yerr=e, color=c, marker=mk, ms=ms, lw=0.7, ls=ls, mfc=c if filled else 'white',
                           mew=0.9, elinewidth=0.8, capsize=0, **kw)

    # (a) charge and saturation
    ax = axs[0]
    p = sc[(sc.f_sat < 0.6) & np.isfinite(sc.cens_25)]
    ser(ax, p.resist_V, p.cens_25, p.cens_25_err, 'D', ls='none')
    vv = np.linspace(420, 500, 50)
    ax.plot(vv, np.exp(gfit['a'] + gfit['b'] * (vv - gfit['Vm'])), color=ps.MODEL, lw=0.8, ls=':', zorder=1)
    ax.set_yscale('log'); ax.set_ylim(700, 6e4)
    ax.set_ylabel('track charge, 25th\npercentile [ADC]')
    d = G['fit_425_495']
    ax.text(0.04, 0.95, f"charge: $\\times2$ per {d['doubling_V']:.1f} V,\n"
                        f"$\\times${G['ratio_490_over_425']:.0f} from 425 to 490 V",
            transform=ax.transAxes, va='top', fontsize=ps.BASE_PT * 0.78, color=ps.INK)
    t = ax.twinx(); t.grid(False); t.spines['right'].set_visible(True)
    t.plot(sc.resist_V, 100 * sc.f_sat, color=ps.ACCENT, marker='*', ms=ms + 1.5, lw=0.7, ls='-')
    t.set_ylim(0, 105); t.set_ylabel('saturated tracks [%]', color=ps.ACCENT)
    t.tick_params(axis='y', colors=ps.ACCENT)
    # (b) angle s68 (left) and angle scale (right, small)
    ax = axs[1]
    for v in 'xy':
        ser(ax, H.resist_V, H[f's68_{v}'], H[f's68_{v}_err'], 'o', filled=(v == 'x'), ls=ps.VIEW_LS[v])
    ax.set_ylim(0, 1.45); ax.set_ylabel(r'angle $s_{68}$ [deg]')
    t2 = ax.twinx(); t2.grid(False); t2.spines['right'].set_visible(True)
    for v in 'xy':
        t2.errorbar(H.resist_V, H[f'scale_{v}'], yerr=H[f'scale_{v}_err'], color=ps.MUTED, marker='^', ms=2.4,
                    mfc=ps.MUTED if v == 'x' else 'white', mew=0.7, lw=0.5, ls=ps.VIEW_LS[v], elinewidth=0.6, capsize=0)
    t2.set_ylim(0.97, 1.27); t2.set_ylabel('angle scale', color=ps.MUTED)
    t2.yaxis.set_major_locator(mt.FixedLocator([0.98, 1.0, 1.02, 1.04]))
    t2.tick_params(axis='y', colors=ps.MUTED)
    t2.axhline(1.0, color=ps.MUTED, lw=0.5, zorder=0)
    ax.legend(handles=[Line2D([], [], color=col, marker='o', ms=ms, ls='-', lw=0.7, label='X'),
                       Line2D([], [], color=col, marker='o', ms=ms, mfc='white', ls='--', lw=0.7, label='Y'),
                       Line2D([], [], color=ps.MUTED, marker='^', ms=2.6, ls='-', lw=0.5, label='scale (right)')],
              loc='upper left', ncol=3, **leg)
    # (c) position core
    ax = axs[2]
    for v in 'xy':
        ser(ax, H.resist_V, H[f'pos_core_{v}'], H[f'pos_core_{v}_err'], 'v', filled=(v == 'x'), ls=ps.VIEW_LS[v],
            label=v.upper())
    ax.set_ylim(0, 0.8); ax.set_ylabel('position core\nat the mesh [mm]')
    ax.legend(loc='lower left', ncol=2, **leg)
    # (d) time resolution
    ax = axs[3]
    ser(ax, tim.index, tim.sigma_xy, tim.sigma_xy_err, 's', ls='--', filled=False,
        label=r'uncorrelated, $(t_X-t_Y)/\sqrt{2}$')
    ser(ax, tim.index, tim.sigma_evt, tim.sigma_evt_err, 's', label=r'event time, $(t_X+t_Y)/2$')
    ax.set_ylim(0, 125); ax.set_ylabel(r'time resolution $s_{68}$ [ns]')
    ax.legend(loc='upper right', **leg)
    for ax in axs:
        ax.axvline(GOLDEN_V, color=ps.ACCENT, lw=0.8, ls=':', zorder=0)
        ax.set_xlim(418, 532)
        ax.xaxis.set_major_locator(mt.MultipleLocator(20))
    for ax in axs[2:]:
        ax.set_xlabel('resist HV [V]')
    axs[2].text(GOLDEN_V - 2, 0.70, 'Runs 1, 2', color=ps.ACCENT, ha='right', fontsize=ps.BASE_PT * 0.75)
    ps.panel_labels(axs, x=-0.15)
    fig.get_layout_engine().set(h_pad=0.06, w_pad=0.05)
    csv = pd.concat([sc[['resist_V', 'cens_25', 'cens_25_err', 'f_sat']].assign(panel='a'),
                     H[['resist_V', 's68_x', 's68_x_err', 's68_y', 's68_y_err', 'scale_x', 'scale_x_err', 'scale_y',
                        'scale_y_err', 'pos_core_x', 'pos_core_x_err', 'pos_core_y', 'pos_core_y_err']].assign(panel='b,c'),
                     tim.reset_index()[['resist_V', 'sigma_xy', 'sigma_xy_err', 'sigma_evt', 'sigma_evt_err']]
                     .assign(panel='d')], ignore_index=True)
    ps.save(fig, os.path.join(OUT, 'gain_scan'), data=csv)

    flat = H[(H.resist_V >= 425) & (H.resist_V <= 505)]
    summ = dict(
        figure='gain_scan', run='chamber A, 27 June HV scan, 1000 V drift',
        gain=G, current_vs_gain=I, timing=Tj,
        timing_quoted={int(V): dict(sigma_xy=float(r.sigma_xy), sigma_xy_err=float(r.sigma_xy_err),
                                    sigma_evt=float(r.sigma_evt), sigma_evt_err=float(r.sigma_evt_err))
                       for V, r in tim.iterrows()},
        flat_425_505=dict(s68_x=[float(flat.s68_x.min()), float(flat.s68_x.max())],
                          s68_y=[float(flat.s68_y.min()), float(flat.s68_y.max())],
                          pos_core_x=[float(flat.pos_core_x.min()), float(flat.pos_core_x.max())],
                          pos_core_y=[float(flat.pos_core_y.min()), float(flat.pos_core_y.max())],
                          scale_x=[float(flat.scale_x.min()), float(flat.scale_x.max())],
                          scale_y=[float(flat.scale_y.min()), float(flat.scale_y.max())]),
        s68_above_510={int(r.resist_V): [float(r.s68_x), float(r.s68_y)] for r in H.itertuples() if r.resist_V > 510},
        scale_all=dict(x=[float(H.scale_x.min()), float(H.scale_x.max())],
                       y=[float(H.scale_y.min()), float(H.scale_y.max())]))
    json.dump(summ, open(os.path.join(OUT, 'gain_scan.json'), 'w'), indent=1, default=float)
    print(json.dumps(dict(gain=G['fit_425_495'], ratio=G['ratio_490_over_425'], k=I['k_total_470_495'],
                          t={v: summ['timing_quoted'][v] for v in (425, 455, 490)}), indent=1, default=float))


if __name__ == '__main__':
    main()
