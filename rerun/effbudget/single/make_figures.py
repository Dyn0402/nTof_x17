#!/usr/bin/env python3
"""make_figures.py -- Paper I §5 figures from results/ (run after radius.py, xy_pairing.py, single_track_area.py).

figures/eff_vs_radius      det3 efficiency (production pick, multi-track credit) and the accidental rate vs R
figures/residuals          the full det3 residual distributions behind the radius choice (per view, radial)
figures/xy_pairing         ln(qX/qY) of the true and the swapped pairings of 552 real two-particle events
figures/loss_budget        where the det3 single-track inefficiency goes, full active area (% of single tracks)
figures/two_particle       det3 events with a second particle: how many per configuration, and the muon recovery
Style: papers/paper_style.py (one palette for the paper; det3 dark blue). Each figure writes pdf + png + csv.

Checklist pass 2026-11-02: drawn at the printed width (ps.fig_for; residuals and loss_budget full width 6.1 in, the
others 4.3 in, i.e. 0.66 of the A4 2.2 cm-margin \\linewidth of 6.54 in); no titles (the caption carries them); panel
labels; palette colours only (det3 and, for contrast, ps.ACCENT / ps.MUTED); data as markers with binomial errors where
the denominator is known; one legend, off the data. Numbers unchanged (the CSVs add error / pooled columns only).
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'papers'))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402,F401
from matplotlib.lines import Line2D  # noqa: E402

RES, OUT = os.path.join(HERE, 'results'), os.path.join(HERE, 'figures')
D3 = ps.DET_COLOR['det3']
ROWLS = {'sat_det3': '-', 'p2_det3': '--'}     # Run 1 solid, Run 2 dashed (2026-11-03)
ROWLAB = dict(ps.RUN)                          # public names: sat_det3 = Run 1, p2_det3 = Run 2 (time order)
W_MID = 4.3            # in (printed width of the single-panel figures)


def _logticks(ax, ticks):
    from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _steps(ax, edges, vals, **kw):
    """A histogram outline that leaves a gap at NaN (empty) bins instead of dropping to the axis."""
    return ax.plot(np.repeat(edges, 2)[1:-1], np.repeat(vals, 2), **kw)


def eff_vs_radius():
    """Both golden rows (line style); multi-track credit filled, production pick open; accidentals on the right axis."""
    d = json.load(open(os.path.join(RES, 'radius.json')))
    fig, ax = ps.fig_for(6.1, aspect=0.45)  # 2026-11-03: full page width, lower
    ax2 = ax.twinx(); ax2.grid(False); ax2.spines['right'].set_visible(True)
    rows = []
    for key in ('sat_det3', 'p2_det3'):
        r = d[key]; R = np.array(r['radii']); n = r['n']
        e, c = np.array(r['eff']), np.array(r['credit'])
        ee, ce = [100 * np.sqrt(q / 100 * (1 - q / 100) / n) for q in (e, c)]
        ax.errorbar(R, c, yerr=ce, color=D3, ls=ROWLS[key], marker='o', ms=3.0, lw=1.2, zorder=5)
        ax.errorbar(R, e, yerr=ee, color=D3, ls=ROWLS[key], marker='o', ms=3.0, mfc='white', lw=0.9, zorder=4)
        ax2.plot(R, r['accidental'], ls=ROWLS[key], color=ps.MUTED, marker='^', ms=2.6, lw=0.8)
        rows += [dict(row=key, n=n, R_mm=x, eff=a, eff_err=ae, credit=b, credit_err=be, accidental=f)
                 for x, a, ae, b, be, f in zip(R, e, ee, c, ce, r['accidental'])]
    ax.axvline(5.0, color=ps.ACCENT, lw=0.8, ls=':', zorder=1)
    ax.text(5.6, 63, '$R$ = 5 mm', color=ps.ACCENT, ha='left', fontsize=ps.BASE_PT * 0.85)
    ax2.text(15, 1.15, 'accidental', color=ps.MUTED, ha='center', fontsize=ps.BASE_PT * 0.85)
    ax.set_xlim(0, 30); ax.set_ylim(60, 100)  # Dylan 2026-11-03: linear R (was log)
    ax.set_xlabel('matching radius $R$ at the mesh [mm]'); ax.set_ylabel('efficiency [%]')
    ax2.set_ylabel('accidental match [%]', color=ps.MUTED); ax2.set_ylim(0, 5)
    ax2.tick_params(axis='y', colors=ps.MUTED)
    h = [Line2D([], [], color=D3, marker='o', ms=3.0, lw=1.2, label='multi-track credit'),
         Line2D([], [], color=D3, marker='o', ms=3.0, mfc='white', lw=0.9, label='production pick'),
         Line2D([], [], color=ps.INK, ls=ROWLS['sat_det3'], lw=0.9, label=ROWLAB['sat_det3']),
         Line2D([], [], color=ps.INK, ls=ROWLS['p2_det3'], lw=0.9, label=ROWLAB['p2_det3'])]
    ax.legend(handles=h, loc='center right', bbox_to_anchor=(0.99, 0.50), fontsize=ps.BASE_PT * 0.8)  # 2026-11-03: legend off the curves (top left is on the multi-track curve)
    ps.save(fig, os.path.join(OUT, 'eff_vs_radius'), data=pd.DataFrame(rows))


def residuals():
    """(a) per-view residuals, (b) radial distance with the event-mixed accidentals; both golden rows summed."""
    d = json.load(open(os.path.join(RES, 'radius.json')))
    fig, (a1, a2) = ps.fig_for('full', aspect=0.40, nrows=1, ncols=2)
    rows, tot = [], {}
    for key in ('p2_det3', 'sat_det3'):
        h = d[key]['hist']; be = np.array(h['edges_view']); br = np.array(h['edges_r'])
        for k in ('res_x', 'res_y', 'r', 'r_mixed'):
            tot[k] = tot.get(k, 0) + np.array(h[k], float)
        rows += [dict(row=key, kind='view', lo=lo, hi=hi, x=nx, y=ny) for lo, hi, nx, ny in zip(be[:-1], be[1:], h['res_x'], h['res_y'])]
        rows += [dict(row=key, kind='radial', lo=lo, hi=hi, x=n, y=m) for lo, hi, n, m in zip(br[:-1], br[1:], h['r'], h['r_mixed'])]
    rows += [dict(row='both', kind='view', lo=lo, hi=hi, x=nx, y=ny) for lo, hi, nx, ny in zip(be[:-1], be[1:], tot['res_x'], tot['res_y'])]
    rows += [dict(row='both', kind='radial', lo=lo, hi=hi, x=n, y=m) for lo, hi, n, m in zip(br[:-1], br[1:], tot['r'], tot['r_mixed'])]
    for v in ('x', 'y'):
        _steps(a1, be, np.where(tot[f'res_{v}'] > 0, tot[f'res_{v}'], np.nan), color=D3, lw=1.0, ls=ps.VIEW_LS[v], label=f'{v.upper()} view')
    w = np.diff(br)
    nz = lambda a: np.where(a > 0, a, np.nan)   # empty bins: a gap, not a drop to the axis
    _steps(a2, br, nz(tot['r']) / w, color=D3, lw=1.1, label='its own M3 ray')
    _steps(a2, br, nz(tot['r_mixed']) / w, color=ps.MUTED, lw=1.0, label='another event’s ray')
    a1.set_yscale('log'); a1.set_xlim(-20, 20); a1.set_ylim(0.6, 3e4)
    a1.set_xlabel('residual at the mesh, det3 − M3 [mm]'); a1.set_ylabel('tracks / 0.25 mm')
    for x in (-5, 5):
        a1.axvline(x, color=ps.ACCENT, lw=0.8, ls=':', zorder=1)
    a1.legend(loc='upper right')
    a2.set_xscale('log'); a2.set_yscale('log'); a2.set_xlim(0.05, 200); a2.set_ylim(0.1, 2e5)
    _logticks(a2, [0.1, 1, 10, 100])
    a2.axvline(5, color=ps.ACCENT, lw=0.8, ls=':', zorder=1)
    a2.set_xlabel('distance $|r|$ to the M3 ray [mm]'); a2.set_ylabel('tracks / mm')
    a2.legend(loc='upper right', title='det3 fit vs', title_fontsize=ps.BASE_PT * 0.85, alignment='left')
    ps.panel_labels([a1, a2])
    ps.save(fig, os.path.join(OUT, 'residuals'), data=pd.DataFrame(rows))


def xy_pairing():
    E = pd.read_parquet(os.path.join(RES, 'xy_pairing_events.parquet'))
    ref = json.load(open(os.path.join(RES, 'xy_pairing.json')))
    med = np.mean([ref[k]['reference']['lq_med'] for k in ('sat_det3', 'p2_det3')])
    rs = np.mean([ref[k]['reference']['lq_rsig'] for k in ('sat_det3', 'p2_det3')])
    true = np.r_[E.lq_true_m, E.lq_true_o]; swap = np.r_[E.lq_swap_1, E.lq_swap_2]
    fig, ax = ps.fig_for(W_MID, aspect=0.62)
    b = np.linspace(-2.5, 2.5, 101)
    ax.axvspan(med - rs, med + rs, color=ps.MUTED, alpha=0.25, lw=0, zorder=0, label=r'single tracks, $\pm1\sigma$')
    ax.hist(true, b, histtype='step', color=D3, lw=1.3, label='true pairing (M3-tagged)')
    ax.hist(swap, b, histtype='step', color=ps.ACCENT, lw=1.0, label='swapped pairing')
    ax.set_xlabel(r'$\ln(q_X/q_Y)$ of an X/Y segment pair'); ax.set_ylabel('segment pairs / 0.05')
    ok = ref['pooled']['all']
    ax.text(0.02, 0.97, f"{ok['n']} events, correct pairing\n{100 * ok['charge_time']['frac']:.1f} % "
            f"(charge alone {100 * ok['charge']['frac']:.1f} %)",
            transform=ax.transAxes, va='top', fontsize=ps.BASE_PT * 0.85, color=ps.INK)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(0, 1.3 * ax.get_ylim()[1]); ax.legend(loc='upper right')
    ps.save(fig, os.path.join(OUT, 'xy_pairing'), data=pd.DataFrame(dict(lq_true=pd.Series(true), lq_swap=pd.Series(swap))))


GROUPS = [   # (group, label, predicate on a lost single-track row); first match wins
    ('δ-rays', 'δ along the track (fit pulled off)', lambda r: r.recat == 'far:DELTA'),
    ('δ-rays', 'saturated δ blob', lambda r: r.recat == 'far:FIT_SAT'),
    ('δ-rays', 'large δ (> 50 strips, veto)', lambda r: r.recat == 'spark:shower'),
    ('chamber', 'discharge', lambda r: r.recat == 'spark:discharge'),
    ('chamber', 'noise band', lambda r: r.recat == 'band'),
    ('reconstruction', 'fit off, nothing flagged', lambda r: r.recat in ('far:FIT', 'fit_failed', 'cut:other', 'other:not_seeded',
                                                                          'other:charge_elsewhere', 'multi:m3_not_found')),
    ('reconstruction', 'second deposit, not particle-like', lambda r: r.recat == 'far:MULTI'),
    ('reconstruction', 'right angle, > 5 mm from M3', lambda r: r.recat == 'far:POS_TAIL'),
    ('edges', 'edge of the other coordinate', lambda r: r.recat == 'far:EDGE'),
    ('edges', 'edge zone: > 50 strips (veto)', lambda r: r.region != 'interior' and r['cat'] == 'spark'),
    ('edges', 'edge zone: fit off / no fit', lambda r: r.region != 'interior' and r['cat'] != 'spark'),
]


def loss_budget():
    """Markers with Poisson errors, one colour (det3); the four groups are bold headings on the y axis, not colours."""
    E = pd.concat([pd.read_parquet(os.path.join(RES, f'single_track_area_events_{k}.parquet')) for k in ('sat_det3', 'p2_det3')])
    S = E[E.cls == 'single']; L = S[~S.credit].reset_index(drop=True); n = len(S)
    rows, used = [], np.zeros(len(L), bool)
    for g, lab, f in GROUPS:
        m = L.apply(f, axis=1).to_numpy() & ~used
        used |= m
        rows.append(dict(group=g, cls=lab, n=int(m.sum()), pct=100 * m.sum() / n))
    rest = int((~used).sum())
    if rest:
        rows.append(dict(group='reconstruction', cls='other', n=rest, pct=100 * rest / n))
    T = pd.DataFrame(rows)
    T['pct_err'] = 100 * np.sqrt(T.n) / n        # Poisson
    yb, heads, y = np.zeros(len(T)), [], 0.0
    for g in dict.fromkeys(T.group):
        heads.append((y, g)); y -= 1.0
        for i in T.index[T.group == g]:
            yb[i] = y; y -= 1.0
        y -= 0.3
    fig, ax = ps.fig_for('full', aspect=0.52)
    ax.hlines(yb, 0, T.pct, color=ps.GRID, lw=2.2, zorder=2)
    ax.errorbar(T.pct, yb, xerr=T.pct_err, color=D3, marker='o', ms=4.5, ls='none', lw=1.0, zorder=4)
    for yi, p, e in zip(yb, T.pct, T.pct_err):
        ax.text(p + e + 0.025, yi, f'{p:.2f} %', va='center', fontsize=ps.BASE_PT * 0.8, color=ps.INK)
    ax.set_yticks(list(yb) + [h[0] for h in heads])
    ax.set_yticklabels(list(T.cls) + [h[1] for h in heads])
    for t in ax.get_yticklabels()[len(T):]:
        t.set_fontweight('bold')
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(min(yb) - 0.7, heads[0][0] + 0.6)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('lost single tracks [% of all det3 single tracks]')
    ax.set_xlim(0, 1.25)
    ps.save(fig, os.path.join(OUT, 'loss_budget'), data=T)


def two_particle():
    """Muon recovery per event configuration: markers with binomial errors (all det3, one colour)."""
    E = pd.concat([pd.read_parquet(os.path.join(RES, f'single_track_area_events_{k}.parquet')) for k in ('sat_det3', 'p2_det3')])
    lab = {'single': 'one\nparticle', 'two: separate in both views': 'two, separate\nin both views',
           'two: merged in one projection': 'two, merged in\none projection',
           'two: muon + separate deposit (vetoed)': 'muon + deposit\n(> 50 strips, veto)'}
    rows = []
    for c, l in lab.items():
        g = E[E.cls == c]; k = int(g.credit.sum()); nn = len(g); p = k / nn
        rows.append(dict(cls=l.replace('\n', ' '), n=nn, k=k, eff=100 * p, err=100 * np.sqrt(p * (1 - p) / nn)))
    T = pd.DataFrame(rows)
    fig, ax = ps.fig_for(W_MID, aspect=0.64)
    x = np.arange(len(T))
    ax.errorbar(x, T.eff, yerr=T.err, color=D3, marker='o', ms=5, ls='none', lw=1.2, zorder=5)
    for xi, e in zip(x, T.eff):
        ax.text(xi + 0.13, e, f'{e:.0f} %', va='center', ha='left', fontsize=ps.BASE_PT * 0.85, color=ps.INK)
    ax.set_xticks(x)
    ax.set_xticklabels([l + '\n($n$ = ' + f'{nn:,}'.replace(',', '\u2009') + ')' for l, nn in zip(lab.values(), T.n)],
                       fontsize=ps.BASE_PT * 0.8)
    ax.grid(axis='x', visible=False)
    ax.set_xlim(-0.5, len(T) - 0.5); ax.set_ylim(-4, 105)
    ax.set_ylabel('muon within 5 mm of M3 [%]')
    ps.save(fig, os.path.join(OUT, 'two_particle'), data=T)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    ps.use()
    eff_vs_radius(); residuals(); xy_pairing(); loss_budget(); two_particle()
    print('wrote', OUT)
