#!/usr/bin/env python3
"""pg_gas.py <fit dir> -- the bench gas against time and field, from the pass-1 timing stacks (rerun/paper1_gas/README.md).

Inputs
  <fit dir>/fit_T*.json, rows.json   pg_specs.py + s4f_fit.py (condor/pg_fit.sh): per row (and per third of a long row)
                                     the head-on stack's drift duration T, S4 re-check recipe
  rerun/vt_all/results/bench_vt.csv  each row's clock (t_start_h, duration_h; hours on one campaign clock)
  rerun/vt_within_rows/results/blocks.json   6-26 M3-ladder blocks (angle-scale v per 13-40 min block, det6 + det7)
  rerun/vt_all/crosscheck/results/measure/*.json   column-duration blocks and the per-block gain of the cross-check rows
  sps_beam_test_26/analysis/cloud_basics/gaps.py    the timing-free gaps (charge extent)

Method
  v = gap / T and E = V_drift / gap per stack; Magboltz (bench Ar/iso 95/5 + water, air 0, fine grid air_merged) turns
  v at E into water (gasinv.py).  Errors: bootstrap T (+) V_SYS on T (template / t0 / end definition, FINDINGS §21);
  the gap enters through both v and E and is reported as its own band (gap +- its error).
  det3 drift scan: one water for all fields, gap profiled (FINDINGS §40 on pass 1).

    pg_gas.py <fit dir>  -> results/gas.json, figures/*.png
"""
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, os.path.join(REPO, 'sps_beam_test_26', 'analysis', 'cloud_basics')]
import gasinv as GI  # noqa: E402
import gaps  # noqa: E402

V_SYS = 0.015
DET = {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4', 'mx17_6': 'det6', 'mx17_7': 'det7'}
#: det3's interior gap from the S4 re-check (28.19 +- 0.19), every other chamber the timing-free charge-extent gap
GAP = dict(gaps.GAP_GEOM, det3=28.19)
GAP_ERR = dict(gaps.GAP_GEOM_ERR)


def load(fd):
    F = {}
    for p in sorted(glob.glob(os.path.join(fd, 'fit_T*.json'))):
        F.update(json.load(open(p)))
    return F, json.load(open(os.path.join(fd, 'rows.json')))


def clock():
    return {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'vt_all', 'results', 'bench_vt.csv')))}


def drift_V(row):
    run, sub = row.split('/')[:2]
    if 'drift_' in sub:
        return float(sub.rsplit('drift_', 1)[1].split('V')[0])
    return {'6-26': 700.0, '6-23': 600.0, '6-24': 600.0, '6-22': 1000.0}[next(d for d in ('6-26', '6-23', '6-24', '6-22')
                                                                         if d in run)]


def water_point(T, T_err, V, det, gap=None):
    g = GAP[det] if gap is None else gap
    v = g / (T * 1e-3)
    ve = v * np.hypot(T_err / T, V_SYS)
    E = V / (g / 10.0)
    w = GI.water_err(v, ve, E)
    out = dict(v=v, v_err=ve, E=E, gap=g)
    if w:
        # near the velocity maximum (det2 at 321 V/cm) two waters give one v: take the wet branch (every bench
        # fill measured so far is >= 0.4 %, FINDINGS §25/§40) and flag the point
        br = w['branches']
        wet = [b for b in br if b > 0.3]
        out.update(water=(wet[0] if wet else br[0]), water_lo=w['lo'], water_hi=w['hi'], branches=br,
                   ambiguous=len(br) > 1)
    # the gap band: the same T at gap +- its error (v and E both move)
    for s, lab in ((-1, 'gap_lo'), (1, 'gap_hi')):
        gg = g + s * GAP_ERR[det]
        ww = GI.water_of_v(gg / (T * 1e-3), V / (gg / 10.0))
        out[f'water_{lab}'] = ww[0] if ww else np.nan
    return out


def series(F, R, C):
    """Every fitted arm as a point on the clock: row arms at the row's centre, thirds at their third."""
    pts = []
    for name, r in F.items():
        if 'T' not in r or not (np.isfinite(r.get('T_err', np.nan)) and r['T_err'] > 1e-3 * r['T']):
            continue                     # not fitted, or railed at the starting T (no bootstrap spread)
        parts = name.split('|')
        key, kind = parts[0], '|'.join(parts[1:])
        row = R[key]['row']; det = DET[R[key]['det']]
        t0, dur = float(C[key]['t_start_h']), float(C[key]['duration_h'])
        if kind.startswith('in30|t'):
            i = int(kind[-1]); t = t0 + dur * (i + 0.5) / 3; dt = dur / 6
        else:
            t, dt = t0 + dur / 2, dur / 2
        p = dict(key=key, arm=kind, det=det, row=row, t=t, dt=dt, n=r['n'], T=r['T'], T_err=r['T_err'],
                 Dp=r.get('Dp'), sig0=r.get('sig0'), V=drift_V(row))
        p.update(water_point(r['T'], r['T_err'], p['V'], det))
        pts.append(p)
    return pts


def scan_fit(pts):
    """det3 6-27 drift scan + the 1000 V rows of that day: one water for every field, gap profiled."""
    S = [p for p in pts if p['det'] == 'det3' and p['arm'] == 'in30' and np.isfinite(p['T'])]
    if len(S) < 3:
        return None
    gaps_ = np.arange(27.0, 30.01, 0.05)
    best = None
    prof = []
    for g in gaps_:
        chi_w = []
        for w in np.arange(0.5, 1.6, 0.0025):
            c = 0.0
            for p in S:
                v = g / (p['T'] * 1e-3); ve = v * np.hypot(p['T_err'] / p['T'], V_SYS)
                c += ((v - GI.v_of(w, p['V'] / (g / 10.0))) / ve) ** 2
            chi_w.append((c, w))
        c, w = min(chi_w)
        prof.append((float(g), float(c), float(w)))
        if best is None or c < best[1]:
            best = (float(g), float(c), float(w))
    fixed = []
    for w in np.arange(0.5, 1.6, 0.0025):
        c = sum(((GAP['det3'] / (p['T'] * 1e-3) - GI.v_of(w, p['V'] / (GAP['det3'] / 10.0))) /
                 (GAP['det3'] / (p['T'] * 1e-3) * np.hypot(p['T_err'] / p['T'], V_SYS))) ** 2 for p in S)
        fixed.append((float(c), float(w)))
    c0, w0 = min(fixed)
    ok = [w for c, w in fixed if c <= c0 + 1.0]
    gp = [g for g, c, w in prof if c <= best[1] + 1.0]
    return dict(n=len(S), fixed=dict(gap=GAP['det3'], water=w0, water_68=[min(ok), max(ok)], chi2=c0, ndf=len(S) - 1),
                profile=dict(gap=best[0], gap_68=[min(gp), max(gp)], water=best[2], chi2=best[1], ndf=len(S) - 2),
                points=[dict(V=p['V'], E=p['E'], v=p['v'], v_err=p['v_err'], key=p['key']) for p in S])


def ladder_626():
    """The 6-26 ladder blocks (angle-scale v, x/y mean) on the campaign clock: blocks.json t is hours on 6-26 (00:00)."""
    B = json.load(open(os.path.join(REPO, 'rerun', 'vt_within_rows', 'results', 'blocks.json')))
    out = []
    for r in B:
        for b in r['all']['blocks']:
            vs = [(b[v]['v_lad'], b[v]['v_lad_err']) for v in ('x', 'y') if v in b and b[v].get('v_lad')]
            if not vs:
                continue
            w = np.array([1 / e ** 2 for _, e in vs]); v = float(np.sum(w * np.array([x for x, _ in vs])) / w.sum())
            out.append(dict(key=r['key'], det=DET[r['det']], t626=b['t'], v_lad=v, v_lad_err=float(w.sum() ** -0.5)))
    return out


T626 = 600.0        # the campaign clock (bench_vt.csv hours) at 6-26 00:00: blocks.json t is hours on 6-26
ETAV_PER_PPM = 1.55e-6   # bench Magboltz eta*v per ppm O2 at 238-250 V/cm (air_merged, 0.7-1.35 % water; per ns)


def attachment_626(pts, lad):
    """Timing stack against the M3 ladder on 6-26.  The ladder (drift-time slope across strips against the M3 angle)
    does not see attachment; the stack's drift duration does: electrons from deep in the gap are lost, the column
    reads short and gap/T reads fast.  k = timing/ladder on the attachment-free rows (t >= 6 h on 6-26) carries the
    ladder's kw; then T_true = gap / (k v_lad) and f = T / T_true is the visible fraction of the column.  For an
    exponential loss exp(-t/tau) the fitted box end sits near the half-height point, tau ~ T / ln 2 (a factor-2 model
    uncertainty; the stack model has no attachment term).  O2-equivalent = 1 / (tau ETAV_PER_PPM)."""
    out = {}
    for det in ('det6', 'det7'):
        L = sorted((b['t626'] + T626, b['v_lad'], b['v_lad_err']) for b in lad if b['det'] == det)
        tl, vl = np.array([x[0] for x in L]), np.array([x[1] for x in L])
        P = [p for p in pts if p['det'] == det and p['row'].startswith('mx17_det6_det7') and
             (p['arm'] == 'in30' or p['arm'].startswith('in30|t'))]
        rows = []
        for p in P:
            m = (tl >= p['t'] - p['dt'] - 0.05) & (tl <= p['t'] + p['dt'] + 0.05)
            vlad = float(np.mean(vl[m])) if m.any() else float(np.interp(p['t'], tl, vl))
            rows.append(dict(t=p['t'], arm=p['arm'], key=p['key'], v_stack=p['v'], v_stack_err=p['v_err'], v_lad=vlad,
                             ratio=p['v'] / vlad, ratio_err=p['v_err'] / vlad, T=p['T'], gap=p['gap'], V=p['V']))
        late = [r for r in rows if r['t'] >= T626 + 6.0]
        w = np.array([1 / r['ratio_err'] ** 2 for r in late]); x = np.array([r['ratio'] for r in late])
        k = float(np.sum(w * x) / w.sum()); k_err = float(w.sum() ** -0.5)
        chi = float(np.sum(w * (x - k) ** 2))
        k_err *= max(1.0, np.sqrt(chi / max(len(late) - 1, 1)))
        for r in rows:
            vt = k * r['v_lad']; Tt = r['gap'] / (vt * 1e-3)
            r.update(v_true=vt, T_true=Tt, frac=r['T'] / Tt)
            if r['frac'] < 0.9:
                tau = r['T'] / np.log(2)
                r.update(tau_ns=tau, o2_equiv_ppm=1.0 / (tau * ETAV_PER_PPM))
        # water against time from the anchored ladder blocks, at the row's field (700 V over the gap)
        g = GAP[det]; E = 700.0 / (g / 10.0)
        wat = []
        for t, v, e in L:
            vt = k * v; ve = np.hypot(k * e, v * k_err)
            ww = GI.water_err(vt, ve, E)
            if ww:
                wat.append(dict(t=t, v_true=vt, v_err=ve, water=ww['water'], water_lo=ww['lo'], water_hi=ww['hi']))
        out[det] = dict(k=k, k_err=k_err, k_chi2=chi, k_n=len(late), points=sorted(rows, key=lambda r: r['t']),
                        water=wat, E=E, gap=g)
    # do the two chambers carry one water history?  excess over each one's own afternoon plateau, scaled
    a = {d: out[d]['water'] for d in ('det6', 'det7')}
    pl = {d: float(np.median([w['water'] for w in a[d] if w['t'] >= T626 + 11.0])) for d in a}
    t6 = np.array([w['t'] for w in a['det6']]); e6 = np.array([w['water'] for w in a['det6']]) - pl['det6']
    t7 = np.array([w['t'] for w in a['det7']]); e7 = np.array([w['water'] for w in a['det7']]) - pl['det7']
    m = t7 < T626 + 11.0
    e6i = np.interp(t7[m], t6, e6)
    ratio = float(np.sum(e6i * e7[m]) / np.sum(e7[m] ** 2)) if np.sum(e7[m] ** 2) > 0 else np.nan
    out['shared'] = dict(plateau_water=pl, excess_ratio_det6_over_det7=ratio,
                         corr=float(np.corrcoef(e6i, e7[m])[0, 1]) if m.sum() > 2 else np.nan,
                         start_water={d: a[d][0]['water'] for d in a})
    return out


def main():
    fd = sys.argv[1]
    F, R = load(fd)
    C = clock()
    pts = series(F, R, C)
    lad = ladder_626()
    out = dict(points=pts, scan=scan_fit(pts), ladder_626=lad, att_626=attachment_626(pts, lad), gap=GAP,
               gap_err=GAP_ERR, v_sys=V_SYS, grid=GI.SOURCE)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'gas.json'), 'w'), indent=1, default=float)
    for p in sorted(pts, key=lambda p: (p['det'], p['t'])):
        if p['arm'] in ('in30',) or p['arm'].startswith('in30|t'):
            print(f"{p['det']} {p['t']:7.2f} {p['arm']:9s} n {p['n']:4d} T {p['T']:6.0f}±{p['T_err']:3.0f} v {p['v']:5.2f} "
                  f"E {p['E']:5.0f} water {p.get('water', np.nan):.3f}  {p['row'].split('/')[1][:40]}")
    sc = out['scan']
    print('det3 scan:', {k: sc[k] for k in ('fixed', 'profile')})
    for det in ('det6', 'det7'):
        A = out['att_626'][det]
        print(det, f"k {A['k']:.3f}+-{A['k_err']:.3f} (chi2 {A['k_chi2']:.1f}/{A['k_n'] - 1})")
        for r in A['points']:
            print(f"   t {r['t'] - T626:5.2f} h  stack {r['v_stack']:5.1f}  ladder {r['v_lad']:5.1f}  ratio {r['ratio']:.2f}  "
                  f"frac {r['frac']:.2f}  tau {r.get('tau_ns', np.nan):6.0f} ns  O2eq {r.get('o2_equiv_ppm', np.nan):5.0f}")
        print('   water', ' '.join(f"{w['t'] - T626:.1f}:{w['water']:.2f}" for w in A['water'][::3]))
    print('shared', out['att_626']['shared'])


if __name__ == '__main__':
    main()
