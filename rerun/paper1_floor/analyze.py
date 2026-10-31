#!/usr/bin/env python3
"""analyze.py [--out results/floor_budget.json] -- rank the candidates for the unexplained angle term (README.md).

Sample: the calibration golden rows (and the det2/det3 headline rows), 03_angles' interior_good events, the v(t) arm;
FD's convention throughout: dt = residual after the global d(tan_ref) line and the cross-validated 2D 40 mm tilt map,
s68 over |d| <= 5 deg.  The unexplained term U = q(s68(dt), toy_all, M3 0.05) exactly as budget_pass1 (closure checked
against budget_pass1.json's tilt_2d40 to the printed digit); for Y, toy_all comes from the Y toy (ytoy.py) when present.

Candidates (each a measurement on data; the fraction is Delta^2 / U^2):
  conditions ladder, cumulative, towards the toy's own conditions (the toy generates unsaturated tracks at the row's
  median charge, one particle, no band):
    sat     drop planes with a saturated strip (K1/K-list: saturated planes 0.2-0.3 deg worse)
    amp     keep amplitude ranks 10-80 % (bright tails: noise-only weights over-trust bright samples, K1)
    topo    one candidate, one track, no noise-band strip, strip-count excess and hit multiplicity below their 90th pct
    misfit  the best quarter in chi2/dof, ranked WITHIN amplitude-decile x angle-tercile x strip-count-tercile cells
            (the waveform misfit at fixed brightness and geometry; misfit_retest.py)
    -> U_core: what the toy cannot explain on events that look like the toy's own
  structure / reference terms (on the full sample, not removed by the ladder):
    fine    position structure finer than the 40 mm tilt cells: split-half covariance of dt cell means in 1.56 mm cells
            (a per-strip time offset pattern would show here)
    nonlin  angle non-linearity: split-half covariance of dt means in 20 tan_ref quantiles
    m3      the M3 reference: s68^2 against the M3 chi2 of the matching coordinate, linear; Delta^2 = slope * <chi2>
    xy      the per-event common cause: corr(dt_x, dt_y) on the same events
  angle form: s68^2(theta) = a^2 + (b tan theta)^2 on the toy-like sample: the constant part a vs the scale part b
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import floorlib as C

YTOY = os.path.join(C.HERE, 'results', 'ytoy.json')
_TS = os.path.join(C.HERE, 'results', 'toy_chi2_select.json')
TOYSEL = json.load(open(_TS)) if os.path.exists(_TS) else {}


def split_cov(d, key, half, min_n=4):
    df = pd.DataFrame(dict(d=d, k=key, h=half))
    df = df[np.abs(df.d) <= 5]
    a = df[df.h == 0].groupby('k').d.agg(['mean', 'size'])
    b = df[df.h == 1].groupby('k').d.agg(['mean', 'size'])
    j = a.join(b, lsuffix='_e', rsuffix='_o', how='inner')
    j = j[(j.size_e >= min_n) & (j.size_o >= min_n)]
    if len(j) < 5:
        return np.nan
    w = (j.size_e + j.size_o).to_numpy(float)
    me, mo = j.mean_e.to_numpy(), j.mean_o.to_numpy()
    me, mo = me - np.average(me, weights=w), mo - np.average(mo, weights=w)
    return float(np.average(me * mo, weights=w))


def boot_cov(d, key, half, n=100, seed=3):
    """bootstrap over events of split_cov: its own error."""
    rng = np.random.default_rng(seed)
    d, key, half = np.asarray(d), np.asarray(key), np.asarray(half)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(d), len(d))
        out.append(split_cov(d[i], key[i], half[i]))
    return float(np.nanstd(out))


def derived(g):
    g = g.copy()
    g['qr'] = g.q_sum.rank(pct=True)
    g['cd'] = g.chi2 / g.dof
    g['qdec'] = pd.qcut(g.q_sum, 10, labels=False, duplicates='drop')
    g['cdr'] = g.groupby('qdec').cd.rank(pct=True)
    a = np.abs(g.tan_ref)
    g['nse'] = g.n_strips - np.polyval(np.polyfit(a, g.n_strips, 1), a)
    g['multx'] = g.r_mult - np.polyval(np.polyfit(a, g.r_mult.fillna(g.r_mult.median()), 1), a)
    # chi2/dof rank within amplitude-decile x angle-tercile x strip-count-tercile cells (misfit_retest.py): low chi2
    # at the same brightness AND the same geometry
    cell = (g.qdec.astype(str) + '|' + pd.qcut(a, 3, labels=False, duplicates='drop').astype(str) + '|'
            + pd.qcut(g.n_strips.rank(method='first'), 3, labels=False).astype(str))
    g['cdr3'] = g.groupby(cell).cd.rank(pct=True)
    return g


def ladder(g):
    steps = []
    k = np.ones(len(g), bool)
    steps.append(('all', k.copy()))
    k &= (g.n_sat_strips == 0).to_numpy()
    steps.append(('sat', k.copy()))
    k &= ((g.qr > 0.1) & (g.qr < 0.8)).to_numpy()
    steps.append(('amp', k.copy()))
    k &= ((g.n_band_strips == 0) & (g.n_candidates <= 1) & (g.n_tracks <= 1) & (g.nse < np.nanpercentile(g.nse, 90))
          & (g.multx < np.nanpercentile(g.multx, 90))).to_numpy()
    steps.append(('topo', k.copy()))
    k &= (g.cdr3 < 0.25).to_numpy()
    steps.append(('misfit', k.copy()))
    return steps


def angle_form(g, m):
    th = np.abs(np.degrees(np.arctan(g.tan_ref)))
    x, y, w = [], [], []
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 15), (15, 25)):
        mm = m & (th >= lo) & (th < hi)
        if mm.sum() >= 100:
            s, e = C.boot_s68(g.dt[mm], n=60)
            t = np.tan(np.radians(0.5 * (lo + hi)))
            x.append(t * t)
            y.append(s * s)
            w.append(1 / max(2 * s * e, 1e-4) ** 2)
    if len(x) < 3:
        return dict(a=None, b=None)
    c = np.polyfit(np.array(x), np.array(y), 1, w=np.sqrt(w))
    return dict(a=float(np.sqrt(max(c[1], 0))), b=float(np.sqrt(max(c[0], 0))), b2=float(c[0]), a2=float(c[1]))


def m3_term(g, p):
    col = 'm3_chi2y' if p == 'x' else 'm3_chi2x'
    out = {}
    for c in ('m3_chi2x', 'm3_chi2y'):
        ed = np.nanpercentile(g[c], [0, 20, 40, 60, 80, 100])
        ed[-1] += 1e-9
        x, y = [], []
        for lo, hi in zip(ed[:-1], ed[1:]):
            mm = (g[c] >= lo) & (g[c] < hi)
            x.append(float(g[c][mm].mean()))
            y.append(C.s68(g.dt[mm]) ** 2)
        sl = np.polyfit(x, y, 1)[0]
        out[c] = dict(slope=float(sl), delta2=float(sl * g[c].mean()), bins=[(round(a, 3), round(np.sqrt(b), 3))
                                                                             for a, b in zip(x, y)])
    out['matched'] = col
    return out


def one(det, key, role, ytoy):
    F = C.load(key)
    B = C.budget()['new']
    res = dict(key=key, role=role, views={})
    G = {}
    for p in ('x', 'y'):
        g = derived(C.sample(F, p))
        G[p] = g
        b = B.get(f'{det}|{role}|vt|{p}|ig') or {}
        toy = b.get('toy_all')
        toy_src = 'budget_pass1 (X toy)' if toy is not None else None
        if toy is None and ytoy and det in ytoy:
            toy, toy_src = ytoy[det]['toy_all'], 'ytoy.py (Y toy)'
        meas, merr = C.boot_s68(g.dt, n=200)
        U = C.q(C.q(meas, toy or 0.0), 0.05)
        half = (g.event_id % 2).to_numpy()
        steps = ladder(g)
        lad, prev = [], meas
        for name, m in steps:
            s, e = C.boot_s68(g.dt[m], n=150)
            u = C.q(C.q(s, toy or 0.0), 0.05)
            lad.append(dict(step=name, n=int(m.sum()), s68=s, err=e, U=u,
                            delta2=float(prev * prev - s * s) if name != 'all' else 0.0))
            prev = s
        core = np.abs(g.dt) < 2.5
        fine = split_cov(g.dt[core].to_numpy(), np.floor(g.p0[core] / 1.56).astype(int).to_numpy(), half[core])
        fine_e = boot_cov(g.dt[core].to_numpy(), np.floor(g.p0[core] / 1.56).astype(int).to_numpy(), half[core], n=40)
        nonlin = split_cov(g.dt[core].to_numpy(), pd.qcut(g.tan_ref[core], 20, labels=False).to_numpy(), half[core])
        tl = steps[-1][1]
        # the chi2 step against pure selection: the share of the variance a best-chi2 quarter removes in the toy itself
        st = {L['step']: L for L in lad}
        sel_data = float(1 - st['misfit']['s68'] ** 2 / st['topo']['s68'] ** 2)
        sel_toy = (TOYSEL.get(f'{det}|{p}|L8') or {}).get('mean_share')
        res['views'][p] = dict(n=int(len(g)), measured=meas, measured_err=merr, toy_all=toy, toy_src=toy_src, m3=0.05,
                               U=U, ladder=lad, U_core=lad[-1]['U'], U_cond=st['topo']['U'], chi2_select_share_data=sel_data,
                               chi2_select_share_toy=sel_toy, fine_cov=fine, fine_cov_err=fine_e,
                               nonlin_cov=nonlin, m3_term=m3_term(g, p), angle_form_toylike=angle_form(g, tl),
                               angle_form_all=angle_form(g, np.ones(len(g), bool)))
    j = G['x'][['event_id', 'dt']].merge(G['y'][['event_id', 'dt']], on='event_id', suffixes=('_x', '_y'))
    jj = j[(np.abs(j.dt_x) < 3) & (np.abs(j.dt_y) < 3)]
    res['xy_corr'] = float(np.corrcoef(jj.dt_x, jj.dt_y)[0, 1])
    res['xy_n'] = int(len(jj))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(C.HERE, 'results', 'floor_budget.json'))
    a = ap.parse_args()
    ytoy = json.load(open(YTOY)) if os.path.exists(YTOY) else None
    out = {}
    for det, key in C.CALIB.items():
        out[f'{det}|calib'] = one(det, key, 'calib', ytoy)
    for det, key in C.HEADLINE.items():
        out[f'{det}|headline'] = one(det, key, 'headline', ytoy)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1, default=float)
    for k, r in out.items():
        for p, v in r['views'].items():
            fr = {L['step']: round(L['delta2'] / max(v['U'] ** 2, 1e-9), 2) for L in v['ladder'][1:]}
            print(f"{k} {p}: meas {v['measured']:.3f} toy {v['toy_all']} U {v['U']:.3f} U_cond {v['U_cond']:.3f} "
                  f"U_core {v['U_core']:.3f} chi2-select share data {v['chi2_select_share_data']:.2f} toy "
                  f"{v['chi2_select_share_toy']} "
                  f"ladder fractions {fr} fine {np.sqrt(max(v['fine_cov'], 0)):.3f} nonlin "
                  f"{np.sqrt(max(v['nonlin_cov'], 0)):.3f} m3 {v['m3_term'][v['m3_term']['matched']]['delta2']:+.4f} "
                  f"angle a/b(toylike) {v['angle_form_toylike']}")
        print(f"   xy corr {r['xy_corr']:+.3f} (n {r['xy_n']})")


if __name__ == '__main__':
    main()
