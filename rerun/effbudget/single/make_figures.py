#!/usr/bin/env python3
"""make_figures.py -- Paper I §5 figures from results/ (run after radius.py, xy_pairing.py, single_track_area.py).

figures/eff_vs_radius      det3 efficiency (production pick, multi-track credit) and the accidental rate vs R
figures/residuals          the full det3 residual distributions behind the radius choice (per view, radial)
figures/xy_pairing         ln(qX/qY) of the true and the swapped pairings of 552 real two-particle events
figures/loss_budget        where the det3 single-track inefficiency goes, full active area (% of single tracks)
figures/two_particle       det3 events with a second particle: how many per configuration, and the muon recovery
Style: papers/paper_style.py (one palette for the paper; det3 dark blue). Each figure writes pdf + png + csv.
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'papers'))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RES, OUT = os.path.join(HERE, 'results'), os.path.join(HERE, 'figures')
D3 = ps.DET_COLOR['det3']
ROWLS = {'p2_det3': '-', 'sat_det3': '--'}
ROWLAB = {'p2_det3': '6-27 long run', 'sat_det3': '6-27 Saturday long run'}
CREDIT_C, PROD_C = D3, '#7a9cc6'


def eff_vs_radius():
    d = json.load(open(os.path.join(RES, 'radius.json')))
    fig, ax = plt.subplots(figsize=ps.FULL)
    ax2 = ax.twinx(); ax2.grid(False); ax2.spines['right'].set_visible(True)
    rows = []
    for key in ('p2_det3', 'sat_det3'):
        r = d[key]; R = np.array(r['radii'])
        ax.plot(R, r['credit'], ROWLS[key], color=CREDIT_C, lw=1.6, label=f'multi-track credit ({ROWLAB[key]})')
        ax.plot(R, r['eff'], ROWLS[key], color=PROD_C, lw=1.4, label=f'production pick ({ROWLAB[key]})')
        ax2.plot(R, r['accidental'], ROWLS[key], color=ps.MUTED, lw=1.0)
        rows += [dict(row=key, R_mm=x, eff=e, credit=c, accidental=a) for x, e, c, a in zip(R, r['eff'], r['credit'], r['accidental'])]
    ax.axvline(5.0, color=ps.ACCENT, lw=0.9, ls=':')
    ax.text(5.4, 62, 'R = 5 mm', color=ps.ACCENT)
    ax2.text(32, 3.6, 'accidental\n(right axis)', color=ps.MUTED, ha='right', fontsize=8)
    ax.set_xscale('log'); ax.set_xlim(0.25, 40); ax.set_ylim(60, 100)
    ax.set_xlabel('matching radius R at the mesh [mm]'); ax.set_ylabel('det3 efficiency [%]')
    ax2.set_ylabel('accidental match [%]', color=ps.MUTED); ax2.set_ylim(0, 5); ax2.tick_params(colors=ps.MUTED)
    ax.legend(loc='center right', fontsize=7.5)
    ps.save(fig, os.path.join(OUT, 'eff_vs_radius'), data=pd.DataFrame(rows))


def residuals():
    d = json.load(open(os.path.join(RES, 'radius.json')))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=ps.WIDE)
    rows = []
    for key in ('p2_det3', 'sat_det3'):
        h = d[key]['hist']; be = np.array(h['edges_view']); c = 0.5 * (be[1:] + be[:-1])
        for v in ('x', 'y'):
            a1.step(c, h[f'res_{v}'], where='mid', color=D3, lw=1.0, ls=ps.VIEW_LS[v] if key == 'p2_det3' else ':',
                    label=(f'{v.upper()} view' if key == 'p2_det3' else None))
        br = np.array(h['edges_r']); w = np.diff(br); cr = np.sqrt(br[1:] * np.maximum(br[:-1], 1e-3))
        a2.step(cr, np.array(h['r']) / w, where='mid', color=D3, lw=1.1, ls=ROWLS[key],
                label=('det3 fit vs its M3 ray' if key == 'p2_det3' else None))
        a2.step(cr, np.array(h['r_mixed']) / w, where='mid', color=ps.MUTED, lw=1.0, ls=ROWLS[key],
                label=('vs another event’s M3 ray' if key == 'p2_det3' else None))
        rows += [dict(row=key, kind='view', lo=lo, hi=hi, x=nx, y=ny) for lo, hi, nx, ny in zip(be[:-1], be[1:], h['res_x'], h['res_y'])]
        rows += [dict(row=key, kind='radial', lo=lo, hi=hi, x=n, y=m) for lo, hi, n, m in zip(br[:-1], br[1:], h['r'], h['r_mixed'])]
    a1.set_yscale('log'); a1.set_xlim(-20, 20); a1.set_xlabel('residual at the mesh, det3 − M3 [mm]'); a1.set_ylabel('tracks / 0.25 mm')
    for x in (-5, 5):
        a1.axvline(x, color=ps.ACCENT, lw=0.8, ls=':')
    a1.legend(fontsize=7.5)
    a2.set_xscale('log'); a2.set_yscale('log'); a2.set_xlim(0.05, 200); a2.axvline(5, color=ps.ACCENT, lw=0.8, ls=':')
    a2.set_xlabel('radial distance |r| [mm]'); a2.set_ylabel('tracks / mm'); a2.legend(fontsize=7.5, loc='lower left')
    ps.save(fig, os.path.join(OUT, 'residuals'), data=pd.DataFrame(rows))


def xy_pairing():
    E = pd.read_parquet(os.path.join(RES, 'xy_pairing_events.parquet'))
    ref = json.load(open(os.path.join(RES, 'xy_pairing.json')))
    med = np.mean([ref[k]['reference']['lq_med'] for k in ('sat_det3', 'p2_det3')])
    rs = np.mean([ref[k]['reference']['lq_rsig'] for k in ('sat_det3', 'p2_det3')])
    true = np.r_[E.lq_true_m, E.lq_true_o]; swap = np.r_[E.lq_swap_1, E.lq_swap_2]
    fig, ax = plt.subplots(figsize=ps.FULL)
    b = np.linspace(-2.5, 2.5, 101)
    ax.axvspan(med - rs, med + rs, color=ps.DET_COLOR['det6'], alpha=0.2, lw=0, label='single tracks (±1σ)')
    ax.hist(true, b, histtype='step', color=D3, lw=1.4, label='true pairing (M3-tagged)')
    ax.hist(swap, b, histtype='step', color=ps.ACCENT, lw=1.1, label='swapped pairing')
    ax.set_xlabel(r'$\ln(q_X/q_Y)$ of an X/Y segment pair'); ax.set_ylabel('segment pairs')
    ok = ref['pooled']['all']
    ax.text(0.02, 0.97, f"{ok['n']} real two-particle events\ncorrect pairing: charge {100 * ok['charge']['frac']:.1f} %,"
            f" charge + time {100 * ok['charge_time']['frac']:.1f} %", transform=ax.transAxes, va='top', fontsize=8)
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
GCOL = {'δ-rays': D3, 'chamber': ps.ACCENT, 'reconstruction': ps.DET_COLOR['det2'], 'edges': ps.DET_COLOR['det4']}


def loss_budget():
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
    fig, ax = plt.subplots(figsize=(6.1, 3.6))
    y = np.arange(len(T))[::-1]
    ax.barh(y, T.pct, color=[GCOL[g] for g in T.group], height=0.7)
    for yi, p, k in zip(y, T.pct, T.n):
        ax.text(p + 0.02, yi, f'{p:.2f} %  ({k})', va='center', fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels(T.cls); ax.grid(axis='y', visible=False)
    ax.set_xlabel(f'% of det3 single tracks, full active area (n = {n:,}; lost {100 * len(L) / n:.1f} %)')
    ax.set_xlim(0, T.pct.max() * 1.45)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=g) for g, c in GCOL.items()], loc='upper right', fontsize=7.5)
    ps.save(fig, os.path.join(OUT, 'loss_budget'), data=T)


def two_particle():
    E = pd.concat([pd.read_parquet(os.path.join(RES, f'single_track_area_events_{k}.parquet')) for k in ('sat_det3', 'p2_det3')])
    lab = {'single': 'one particle', 'two: separate in both views': 'two, separate\nin both views',
           'two: merged in one projection': 'two, merged in\none projection',
           'two: muon + separate deposit (vetoed)': 'muon + separate deposit\n(> 50 strips, veto)'}
    rows = []
    for c, l in lab.items():
        g = E[E.cls == c]; k = int(g.credit.sum()); nn = len(g); p = k / nn
        rows.append(dict(cls=l.replace('\n', ' '), n=nn, k=k, eff=100 * p, err=100 * np.sqrt(p * (1 - p) / nn)))
    T = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=ps.FULL)
    x = np.arange(len(T))
    ax.bar(x, T.eff, yerr=T.err, color=[D3, D3, ps.DET_COLOR['det2'], ps.DET_COLOR['det4']], width=0.6)
    for xi, e, er, nn in zip(x, T.eff, T.err, T.n):
        ax.text(xi, e + er + 2.5, f'{e:.0f} %\n(n = {nn:,})', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(list(lab.values()), fontsize=7.5); ax.grid(axis='x', visible=False)
    ax.set_ylim(0, 120); ax.set_ylabel('muon reconstructed within 5 mm [%]')
    ax.set_title('det3, full active area, both golden rows (pass 1, before the two-track and veto fixes)', fontsize=7.5)
    ps.save(fig, os.path.join(OUT, 'two_particle'), data=T)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    ps.use()
    eff_vs_radius(); residuals(); xy_pairing(); loss_budget(); two_particle()
    print('wrote', OUT)
