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
        return float(sub.split('drift_')[1].split('V')[0])
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
        out.update(water=w['water'], water_lo=w['lo'], water_hi=w['hi'], branches=w['branches'])
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
        if 'T' not in r:
            continue
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


def main():
    fd = sys.argv[1]
    F, R = load(fd)
    C = clock()
    pts = series(F, R, C)
    out = dict(points=pts, scan=scan_fit(pts), ladder_626=ladder_626(), gap=GAP, gap_err=GAP_ERR, v_sys=V_SYS,
               grid=GI.SOURCE)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'gas.json'), 'w'), indent=1, default=float)
    for p in sorted(pts, key=lambda p: (p['det'], p['t'])):
        if p['arm'] in ('in30',) or p['arm'].startswith('in30|t'):
            print(f"{p['det']} {p['t']:7.2f} {p['arm']:9s} n {p['n']:4d} T {p['T']:6.0f}±{p['T_err']:3.0f} v {p['v']:5.2f} "
                  f"E {p['E']:5.0f} water {p.get('water', np.nan):.3f}  {p['row'].split('/')[1][:40]}")
    print(json.dumps(out['scan'], default=float)[:600])


if __name__ == '__main__':
    main()
