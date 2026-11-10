#!/usr/bin/env python3
"""make_pairing.py -- Paper I v2: the X/Y pairing figure (out/pairing.pdf, .png, .json, .csv).

(a) X/Y charge balance ln(qX/qY) of chamber A single tracks (Runs 1 and 2), inset: median and core sigma per chamber
    (golden runs, unsaturated: rerun/paper1_charge/results/summary.json, the numbers of charge_balance.pdf).
(b) ln(qX/qY) of the true (reference-tagged) and swapped X/Y pairings of the two-particle events, single-track band.
(c) fraction paired correctly against the charge ratio of the two particles, charge alone and charge + time.
(d) the time handle: u50_X - u50_Y (difference between the views of the median charge-arrival time after the fitted t0),
    true and swapped pairings, single-track band.

The per-event quantities are recomputed here from the chamber A two-track re-pass tables with the selection and
rules of rerun/effbudget/single/xy_pairing.py (imported, not copied: load(), R_IN/R_OUT, rsig); the script checks
that it reproduces that study's stored per-event verdicts (results/xy_pairing_events.parquet) exactly.

  .venv/bin/python papers/paper1_v2/figs/make_pairing.py \
      --root ~/work_data/claude-jobs/twotrack --analysis Analysis_pass1_twotrack --eff paper1_eff_twotrack
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
STUDY = os.path.join(REPO, 'rerun', 'effbudget', 'single')
sys.path[:0] = [STUDY, os.path.join(REPO, 'papers')]
import xy_pairing as xp          # noqa: E402  (load, ROWS, R_IN, R_OUT, rsig)
import paper_style as ps          # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402,F401
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import beta      # noqa: E402

OUT = os.path.join(HERE, 'out')
D3 = ps.DET_COLOR['det3']


def recompute(root, analysis, eff):
    """xy_pairing.py's loop, keeping the time differences of all four pairings. Returns (events, singles, refs)."""
    evs, sing, refs = [], [], {}
    for key, rel in xp.ROWS.items():
        C, B = xp.load(root, rel, analysis, eff)
        n = C.groupby(['event_id', 'plane']).size().unstack(fill_value=0)
        one = n.index[(n.x == 1) & (n.y == 1)]
        S = C[C.event_id.isin(one)].pivot(index='event_id', columns='plane')
        S = S[(S['d']['x'] < xp.R_IN) & (S['d']['y'] < xp.R_IN)]
        lq1 = np.log(S['q_sum']['x'].clip(lower=1) / S['q_sum']['y'].clip(lower=1))
        du1 = S['q_u50']['x'] - S['q_u50']['y']
        ref = dict(n=int(len(S)), lq_med=float(lq1.median()), lq_rsig=float(xp.rsig(lq1)),
                   du_med=float(du1.median()), du_rsig=float(xp.rsig(du1)))
        refs[key] = ref
        sing.append(pd.DataFrame(dict(row=key, lq=lq1.to_numpy(), du=du1.to_numpy())))
        rows = []
        for e, g in C[C.event_id.isin(n.index[(n.x >= 2) & (n.y >= 2)])].groupby('event_id'):
            gx = g[g.plane == 'x'].sort_values('d'); gy = g[g.plane == 'y'].sort_values('d')
            if not (gx.d.iloc[0] < xp.R_IN and gx.d.iloc[1] > xp.R_OUT and gy.d.iloc[0] < xp.R_IN and gy.d.iloc[1] > xp.R_OUT):
                continue
            xm, ym = gx.iloc[0], gy.iloc[0]
            xo = gx.iloc[1:].sort_values('q_sum').iloc[-1]; yo = gy.iloc[1:].sort_values('q_sum').iloc[-1]
            lq = lambda u, v: np.log(max(u.q_sum, 1) / max(v.q_sum, 1))
            zq = lambda u, v: abs(lq(u, v) - ref['lq_med']) / ref['lq_rsig']
            zt = lambda u, v: abs((u.q_u50 - v.q_u50) - ref['du_med']) / ref['du_rsig']
            keep_q, swap_q = zq(xm, ym) ** 2 + zq(xo, yo) ** 2, zq(xm, yo) ** 2 + zq(xo, ym) ** 2
            keep_t = keep_q + zt(xm, ym) ** 2 + zt(xo, yo) ** 2
            swap_t = swap_q + zt(xm, yo) ** 2 + zt(xo, ym) ** 2
            qa = abs(np.log((xm.q_sum + ym.q_sum) / max(xo.q_sum + yo.q_sum, 1)))
            rows.append(dict(row=key, event_id=int(e), charge=keep_q < swap_q, charge_time=keep_t < swap_t, q_asym=qa,
                             lq_true_m=lq(xm, ym), lq_true_o=lq(xo, yo), lq_swap_1=lq(xm, yo), lq_swap_2=lq(xo, ym),
                             du_true_m=xm.q_u50 - ym.q_u50, du_true_o=xo.q_u50 - yo.q_u50,
                             du_swap_1=xm.q_u50 - yo.q_u50, du_swap_2=xo.q_u50 - ym.q_u50,
                             dqcost_keep=keep_q, dqcost_swap=swap_q, cost_keep=keep_t, cost_swap=swap_t))
        evs.append(pd.DataFrame(rows))
    return pd.concat(evs, ignore_index=True), pd.concat(sing, ignore_index=True), refs


def cp(k, n, cl=0.6827):
    """Clopper-Pearson interval (fraction), central cl."""
    a = (1 - cl) / 2
    lo = beta.ppf(a, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a, k + 1, n - k) if k < n else 1.0
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.expanduser('~/work_data/claude-jobs/twotrack'))
    ap.add_argument('--analysis', default='Analysis_pass1_twotrack')
    ap.add_argument('--eff', default='paper1_eff_twotrack')
    a = ap.parse_args()
    E, S, refs = recompute(a.root, a.analysis, a.eff)

    # --- closure against the study's stored per-event verdicts
    st = pd.read_parquet(os.path.join(STUDY, 'results', 'xy_pairing_events.parquet'))
    M = st.merge(E, on=['row', 'event_id'], suffixes=('_st', ''))
    closure = dict(n_stored=int(len(st)), n_here=int(len(E)), n_common=int(len(M)),
                   charge_mismatch=int((M.charge_st != M.charge).sum()),
                   charge_time_mismatch=int((M.charge_time_st != M.charge_time).sum()),
                   max_dlq=float(np.abs(M.lq_true_m_st - M.lq_true_m).max()),
                   max_dqasym=float(np.abs(M.q_asym_st - M.q_asym).max()))
    assert closure['n_common'] == closure['n_stored'] == closure['n_here'], closure
    assert closure['charge_mismatch'] == 0 and closure['charge_time_mismatch'] == 0, closure
    prod = M.set_index(['row', 'event_id'])['prod']

    med = float(np.mean([refs[k]['lq_med'] for k in refs])); rs = float(np.mean([refs[k]['lq_rsig'] for k in refs]))
    tmed = float(np.mean([refs[k]['du_med'] for k in refs])); trs = float(np.mean([refs[k]['du_rsig'] for k in refs]))
    lq_true = np.r_[E.lq_true_m, E.lq_true_o]; lq_swap = np.r_[E.lq_swap_1, E.lq_swap_2]
    du_true = np.r_[E.du_true_m, E.du_true_o]; du_swap = np.r_[E.du_swap_1, E.du_swap_2]

    # --- (c) correct fraction against the charge ratio of the two particles
    edges = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 4.5])
    cbins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = E[(E.q_asym >= lo) & (E.q_asym < hi)]
        r = dict(lo_lnratio=lo, hi_lnratio=hi, lo_ratio=float(np.exp(lo)), hi_ratio=float(np.exp(hi)), n=int(len(g)),
                 ratio_median=float(np.exp(g.q_asym.median())) if len(g) else np.nan)
        for m in ('charge', 'charge_time'):
            k = int(g[m].sum()); l_, h_ = cp(k, len(g)) if len(g) else (np.nan, np.nan)
            r.update({f'{m}_k': k, f'{m}_frac': k / len(g) if len(g) else np.nan, f'{m}_lo': l_, f'{m}_hi': h_})
        cbins.append(r)
    CB = pd.DataFrame(cbins)

    def frac(m, sel):
        g = E[sel]; k = int(g[m].sum()); n = len(g); l_, h_ = cp(k, n)
        return dict(n=n, k=k, frac=k / n, err_binom=float(np.sqrt(k / n * (1 - k / n) / n)), cp68=[l_, h_])
    sim, dif = E.q_asym < 0.5, E.q_asym >= 0.5
    numbers = dict(
        sample='chamber A, Runs 1 and 2, interior, two-track re-pass tables; >= 2 candidates per view, the reference '
               'ray marks exactly one X and one Y candidate (< 5 mm, next > 10 mm); other particle = brightest other '
               'candidate of each view (rerun/effbudget/single/xy_pairing.py selection, recomputed here)',
        closure=closure,
        n_events=int(len(E)), n_per_run={k: int((E.row == k).sum()) for k in refs},
        single_track_reference=refs,
        charge_only=dict(all=frac('charge', np.ones(len(E), bool)), similar_lt_1p65=frac('charge', sim), different_ge_1p65=frac('charge', dif)),
        charge_time=dict(all=frac('charge_time', np.ones(len(E), bool)), similar_lt_1p65=frac('charge_time', sim),
                         different_ge_1p65=frac('charge_time', dif)),
        production_standard_reco=dict(correct=float((prod == 'correct').mean()), wrong=float((prod == 'wrong').mean()),
                                      unpaired=float((prod == 'unpaired').mean())),
        min_ratio_with_a_failure=dict(charge=float(np.exp(E.q_asym[~E.charge].max())) if (~E.charge).any() else None,
                                      charge_time=float(np.exp(E.q_asym[~E.charge_time].max())) if (~E.charge_time).any() else None),
        time_handle=dict(variable='u50_X - u50_Y: difference between the views of the median arrival time of the fitted '
                                  'charge after the fitted t0 (wft.reco._profile_summary q_u50) [ns]',
                         single_med_ns=tmed, single_rsig_ns=trs,
                         true_rsig_ns=float(xp.rsig(du_true)), swap_rsig_ns=float(xp.rsig(du_swap)),
                         true_within_2sig=float(np.mean(np.abs(du_true - tmed) < 2 * trs)),
                         swap_within_2sig=float(np.mean(np.abs(du_swap - tmed) < 2 * trs)),
                         rescued_by_time=int((~E.charge & E.charge_time).sum()), lost_by_time=int((E.charge & ~E.charge_time).sum())),
        charge_handle=dict(true_rsig=float(xp.rsig(lq_true)), swap_rsig=float(xp.rsig(lq_swap)),
                           true_within_2sig=float(np.mean(np.abs(lq_true - med) < 2 * rs)),
                           swap_within_2sig=float(np.mean(np.abs(lq_swap - med) < 2 * rs))),
        panel_c_bins=cbins,
    )
    summ = json.load(open(os.path.join(REPO, 'rerun', 'paper1_charge', 'results', 'summary.json')))
    order = ['mx17_3', 'mx17_2', 'mx17_6', 'mx17_7', 'mx17_4']       # A..E
    G = {g['det']: g for g in summ['golden']}
    ins_rows = [dict(det=d, chamber=ps.name(d), med=G[d]['xy_ln_median'], core_sigma=G[d]['xy_ln_core_sigma'],
                     n_unsat=int(round(G[d]['n'] * (1 - G[d]['f_sat'])))) for d in order]
    numbers['panel_a_inset'] = ins_rows
    numbers['panel_a_main'] = dict(n=int(len(S)), median=float(np.median(S.lq)), core_sigma=float(xp.rsig(S.lq)))

    # ---------------------------------------------------------------- figure
    ps.use()
    fig, axs = ps.fig_for('full', height=4.9, nrows=2, ncols=2)
    (aa, ab), (ac, ad) = axs
    from matplotlib.colors import to_rgb
    band = tuple(0.25 * c + 0.75 for c in to_rgb(ps.MUTED))
    small = ps.BASE_PT * 0.85

    # (a) single-track charge balance
    b = np.arange(-0.6, 0.6001, 0.02)
    h, _ = np.histogram(S.lq, b)
    aa.stairs(h / h.sum() / 0.02, b, color=D3, lw=1.4, fill=False)
    aa.stairs(h / h.sum() / 0.02, b, color=D3, alpha=0.15, fill=True, lw=0)
    aa.axvline(0, color=ps.MUTED, lw=0.7, ls='--', zorder=0.6)
    aa.set_xlim(-0.6, 0.6); aa.set_ylim(0, None)
    aa.set_xlabel(r'$\ln(q_X/q_Y)$, single tracks'); aa.set_ylabel('probability density')
    aa.text(0.03, 0.96, f"chamber A\n{len(S):,}".replace(',', ' ') + ' tracks\n'
            + f"core $\\sigma$ = {numbers['panel_a_main']['core_sigma']:.3f}",
            transform=aa.transAxes, va='top', ha='left', fontsize=small, color=ps.INK)
    ins = aa.inset_axes([0.66, 0.42, 0.32, 0.54])
    ins.axhline(0, color=ps.MUTED, lw=0.6, ls='--', zorder=0)
    for i, r in enumerate(ins_rows):
        c = ps.color(r['det'])
        ins.errorbar(i, r['med'], yerr=r['core_sigma'], color=c, marker=ps.DET_MARKER[r['det'].replace('mx17_', 'det')],
                     ms=4.5, lw=1.0, mec='white', mew=0.4)
    ins.set_xticks(range(5)); ins.set_xticklabels([r['chamber'] for r in ins_rows])
    ins.set_xlim(-0.6, 4.6); ins.set_ylim(-0.2, 0.2); ins.set_yticks([-0.2, 0, 0.2])
    ins.tick_params(labelsize=ps.BASE_PT * 0.78, length=2, pad=1); ins.grid(False)
    ins.set_title(r'median $\pm$ core $\sigma$', fontsize=ps.BASE_PT * 0.78, loc='center', pad=2)

    # (b) true vs swapped pairings, charge
    bb = np.linspace(-2.5, 2.5, 101)
    ab.axvspan(med - rs, med + rs, facecolor=band, lw=0, zorder=0.6)
    ab.hist(lq_true, bb, histtype='step', color=D3, lw=1.3)
    ab.hist(lq_swap, bb, histtype='step', color=ps.ACCENT, lw=1.0, ls='--')
    ymax = max(np.histogram(lq_true, bb)[0].max(), np.histogram(lq_swap, bb)[0].max())
    ab.set_xlim(-1.9, 1.9); ab.set_ylim(0, ymax * 1.6)
    ab.set_xlabel(r'$\ln(q_X/q_Y)$ of an X/Y segment pair'); ab.set_ylabel('segment pairs / 0.05')
    hb = [Line2D([], [], color=D3, lw=1.3, label='true'),
          Line2D([], [], color=ps.ACCENT, lw=1.0, ls='--', label='swapped'),
          Patch(facecolor=band, lw=0, label=r'single, $\pm1\sigma$')]
    ab.legend(handles=hb, loc='center right', bbox_to_anchor=(1.0, 0.62), fontsize=small, handlelength=1.3, title='pairing', title_fontsize=small, alignment='left')
    ab.text(0.02, 0.70, f"{len(E)} two-particle\nevents", transform=ab.transAxes, va='top', ha='left',
            fontsize=small, color=ps.INK)

    # (c) correct fraction vs charge ratio
    xr = np.arange(len(CB), dtype=float)          # one slot per bin; ticks at the bin edges (ratio values)
    for m, mk, lab, col, fill, dx in (('charge', 's', 'charge only', ps.MUTED, False, -0.12),
                                      ('charge_time', 'o', 'charge + time', D3, True, 0.12)):
        y = 100 * CB[f'{m}_frac']; lo = 100 * CB[f'{m}_lo']; hi = 100 * CB[f'{m}_hi']
        ac.errorbar(xr + dx, y, yerr=[y - lo, hi - y], color=col, marker=mk, ms=4, ls='none', lw=1.0,
                    mfc=col if fill else 'white', mec=col, label=lab, zorder=4 if fill else 3)
    i165 = int(np.where(np.isclose(CB.lo_lnratio, 0.5))[0][0]) - 0.5
    ac.axvline(i165, color=ps.ACCENT, lw=0.8, ls=':', zorder=1)
    ac.text(i165 + 0.1, 36, 'factor 1.65', color=ps.ACCENT, fontsize=small, ha='left', va='bottom')
    ac.set_xlim(-0.6, len(CB) - 0.4); ac.set_ylim(20, 104)
    ac.set_xticks(np.arange(len(CB) + 1) - 0.5)
    ac.set_xticklabels([f'{np.exp(e):.2f}'.rstrip('0').rstrip('.') for e in edges[:-1]] + [''],
                       fontsize=ps.BASE_PT * 0.8)
    ac.grid(axis='x', visible=False)
    for xi, n in zip(xr, CB.n):
        ac.text(xi, 21.5, f'{n}', ha='center', va='bottom', fontsize=ps.BASE_PT * 0.8, color=ps.MUTED)
    ac.text(len(CB) - 0.45, 21.5, '$n$', ha='left', va='bottom', fontsize=ps.BASE_PT * 0.8, color=ps.MUTED)
    ac.set_xlabel('charge ratio of the two particles, $Q_1/Q_2$ ($\\geq$1)')
    ac.set_ylabel('paired correctly [%]')
    ac.legend(loc='center right', bbox_to_anchor=(1.0, 0.40), fontsize=small, handlelength=1.0)

    # (d) the time handle
    bt = np.arange(-400, 400.1, 10)
    ad.axvspan(tmed - trs, tmed + trs, facecolor=band, lw=0, zorder=0.6)
    ad.hist(du_true, bt, histtype='step', color=D3, lw=1.3)
    ad.hist(du_swap, bt, histtype='step', color=ps.ACCENT, lw=1.0, ls='--')
    ymax = max(np.histogram(du_true, bt)[0].max(), np.histogram(du_swap, bt)[0].max())
    ad.set_xlim(-400, 400); ad.set_ylim(0, ymax * 1.6)
    ad.set_xlabel(r'$u_{50,X}-u_{50,Y}$ of an X/Y segment pair [ns]'); ad.set_ylabel('segment pairs / 10 ns')
    hd = [Line2D([], [], color=D3, lw=1.3, label='true pairing'),
          Line2D([], [], color=ps.ACCENT, lw=1.0, ls='--', label='swapped pairing'),
          Patch(facecolor=band, lw=0, label=r'single tracks $\pm1\sigma$')]
    ad.text(0.98, 0.75, 'half-width 16–84 %:\n' + f"single {trs:.0f} ns\ntrue {xp.rsig(du_true):.0f} ns\n"
            + f"swapped {xp.rsig(du_swap):.0f} ns", transform=ad.transAxes, va='top', ha='right', fontsize=small,
            color=ps.INK)

    ps.panel_labels([aa, ab, ac, ad])
    os.makedirs(OUT, exist_ok=True)
    data = pd.DataFrame(dict(lq_true=pd.Series(lq_true), lq_swap=pd.Series(lq_swap),
                             du_true=pd.Series(du_true), du_swap=pd.Series(du_swap)))
    ps.save(fig, os.path.join(OUT, 'pairing'), data={'pairs': data, 'panel_c': CB})
    json.dump(numbers, open(os.path.join(OUT, 'pairing.json'), 'w'), indent=1, default=float)
    print(json.dumps({k: numbers[k] for k in ('closure', 'n_events', 'charge_only', 'charge_time', 'production_standard_reco',
                                              'min_ratio_with_a_failure', 'time_handle', 'charge_handle')}, indent=1, default=float))
    print(CB.to_string())


if __name__ == '__main__':
    main()
