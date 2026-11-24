#!/usr/bin/env python3
"""spline_map_fit.py -- fit, validate and check the n_TOF in-situ spline angle map (r3; Dylan's decision 2026-11-24).

The map replaces the 50 mm cells of angle_map.py / bd_map.py (r2).  Per chamber A, C, D: a tilt that is the gradient
of one potential (cubic tensor B-spline, 20 mm knots) and one scale per view (40 mm knots), with a thin-plate penalty
whose two smoothing parameters are chosen by 5-fold cross-validation over runs.  A and C are fitted on the A-C cosmic
lines only.  D is fitted on the B-D cosmic lines together with the horizontal capsule-pointing cells of the late beam
tracks (25 mm cells, a systematic floor, nuisances alpha, gamma, kappa that are not applied).  Positions are clamped to
+-200 mm and each field to its [0.1, 99.9] percentile over the training lines.  The model and the file format are in
rerun/ntof/stage3/spline_map.py (the evaluator), the fit in spline_map_lib.py.  The study: the re-audit report
rerun/ntof/improve/reaudit/report.html, sections 2.2-2.3 (prototype tests/W2a_map_r3/).

All inputs are UNMAPPED: the lines are built on the _nomap slopes (line_residuals.py build, bd_map.py), and the
pointing cells come from an unmapped view with the censored-window cut (the training copy of run_r3_chain.sh), or from
a mapped view through its _nomap twins.

    spline_map_fit.py points  --view <view> --out late.parquet [--jobs 3]
        the late (>= 25 ms) capsule-track sample of the pointing cells (cache, ~1 min per 100 tables)
    spline_map_fit.py fit     --lines lines.parquet --bd-lines bd_lines.parquet --points late.parquet --out map.json
        the map; D's pointing cells are built from --points (a cache from `points`, or a view directory)
    spline_map_fit.py heldout --lines ... --bd-lines ... --points ... --out heldout.csv
        held-out validation by run parity on the clean cosmic lines: clean sigma, w68, w95, edge / interior, for no
        map, the free 50 mm cells (and D's current 100 mm cells), the lines-only spline and (D) the joint spline
    spline_map_fit.py beam    --late late_<view>.parquet [--map extra.json ...] --out beam.csv
        beam validation on a late sample read from a MAPPED view: pointing of the late tracks with no map, with the
        view's own map and with extra map files applied to the unmapped slopes (fraction within 15 mm of the source,
        cell-to-cell rms of the residual centre, core width)

Numbers are in tan units unless a column says mm.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from rerun.ntof.improve import spline_map_lib as SL  # noqa: E402
from rerun.ntof.stage3 import spline_map as SM  # noqa: E402

VERSION = 'r3'
CODE = [HERE / 'spline_map_fit.py', HERE / 'spline_map_lib.py', REPO / 'rerun/ntof/stage3/spline_map.py']


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def code_version():
    def git(*a):
        try:
            return subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
    rel = [str(p.relative_to(REPO)) for p in CODE]
    return dict(git_head=git('rev-parse', '--short', 'HEAD'), dirty_files=git('status', '--porcelain', '--', *rel),
                md5={r: md5(p) for r, p in zip(rel, CODE)})


def late_sample(points, jobs=3):
    """The late-track sample from a cache file or from a view directory."""
    p = Path(points)
    if p.is_dir():
        print(f'building the late-track sample from {p}', flush=True)
        return SL.build_late_tracks(p, jobs=jobs), dict(view=str(p.resolve()))
    return pd.read_parquet(p), dict(cache=str(p.resolve()), md5=md5(p))


def read_inputs(a):
    L = pd.read_parquet(a.lines)
    B = pd.read_parquet(a.bd_lines)
    d, src = late_sample(a.points, a.jobs)
    return L, B, SL.load_late(d), src


# ================================================================================================= fit
def cmd_points(a):
    d = SL.build_late_tracks(a.view, jobs=a.jobs, out=a.out)
    print(len(d), 'late tracks', d.groupby('arm').size().to_dict(), '->', a.out)


def cmd_fit(a):
    L, B, late, psrc = read_inputs(a)
    maps, cells = {}, {}
    for arm in ('A', 'C', 'D'):
        D = SL.rows_d(B) if arm == 'D' else SL.rows_ac(L, arm)
        use_pc = arm == 'D' or a.ac_pointing
        pc = SL.pointing_cells(late, arm) if use_pc else None
        if pc is not None:
            cells[arm] = pc
        maps[arm], _ = SL.fit_arm(arm, D, pc, floor_mult=a.floor_mult)
        maps[arm]['fit']['training'] = ('B-D lines + horizontal capsule pointing cells' if use_pc and arm == 'D' else
                                        'A-C lines + horizontal capsule pointing cells' if use_pc else 'A-C lines')
    # where the map extrapolates: late beam tracks outside the lines' support
    extrap = {}
    for arm, m in maps.items():
        sup = np.array(m['support']['mask'], bool)
        g = late[late.arm == arm]
        x, y = g.x_local.to_numpy(float), g.y_local.to_numpy(float)
        ix = np.clip(np.round((x - SL.GRID[0]) / 10).astype(int), 0, len(SL.GRID) - 1)
        iy = np.clip(np.round((y - SL.GRID[0]) / 10).astype(int), 0, len(SL.GRID) - 1)
        ok = np.isfinite(x) & np.isfinite(y)
        X, Y = np.meshgrid(SL.GRID, SL.GRID, indexing='ij')
        extrap[arm] = dict(support_x=[float(X[sup].min()), float(X[sup].max())],
                           support_y=[float(Y[sup].min()), float(Y[sup].max())],
                           beam_frac_outside_support=float(np.mean(ok & ~sup[ix, iy])) if len(g) else None)
    meta = dict(
        model=SM.MODEL, version=VERSION,
        created=dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        built_by='rerun/ntof/improve/spline_map_fit.py fit', code=code_version(),
        inputs=dict(lines=dict(path=str(Path(a.lines).resolve()), md5=md5(a.lines), n_lines=int(len(L)),
                               n_runs=int(L.run.nunique())),
                    bd_lines=dict(path=str(Path(a.bd_lines).resolve()), md5=md5(a.bd_lines), n_lines=int(len(B)),
                                  n_runs=int(B.run.nunique())),
                    points=dict(psrc, n_tracks={k: int(v) for k, v in late.groupby('arm').size().items()},
                                rule='late >= 25 ms, gated on the unmapped angles, not censored, one per (sub-run, '
                                     'event, arm), not low_charge / t0_railed / split child; pre-access runs out; D '
                                     'needs its wall')),
        counts={arm: dict(n_lines=m['fit']['n_lines'], n_rows_clean=m['fit']['n_rows_clean'],
                          n_rows_kept=m['fit']['n_rows_kept'],
                          n_pointing_cells=(m['fit']['pointing'] or {}).get('n_cells', 0)) for arm, m in maps.items()},
        smoothing={arm: dict(lam_t=m['fit']['lam_t'], lam_s=m['fit']['lam_s'],
                             grid_t=SL.lam_grids(arm)[0].tolist(), grid_s=SL.lam_grids(arm)[1].tolist(),
                             rule='5-fold CV over runs (folds balanced in rows) on the clean lines; criterion = sum '
                                  'over views of the pooled held-out MAD; the joint fit keeps the lines-only choice')
                   for arm, m in maps.items()},
        model_text="t = (1 + s_p) t_true + a_p in the LOCAL tan (stage-3 tanx/tany); corrected t' = (t - a_p)/(1 + s_p)."
                   " Equivalently on GLOBAL slopes S = g_p t: S' = (S - g_p a_p)/(1 + s_p). Tilt a = grad(phi), phi a "
                   f"tensor cubic B-spline (h = {SL.HT} mm); per-view scale s_x, s_y tensor cubic B-splines "
                   f"(h = {SL.HS} mm); thin-plate penalty.",
        coefficients='phi, s_x, s_y: nb x nb arrays indexed [i_x][i_y]; uniform cubic basis, knots lo + k h; '
                     'evaluate with rerun/ntof/stage3/spline_map.py (a_x = d phi/dx, a_y = d phi/dy, x = x_local, '
                     'y = y_local in mm)',
        frames={arm: SL.FRAMES[arm] for arm in maps},
        signs='g = local-tan -> global-slope sign per view (x, y): A (+1,+1), C (+1,-1), D (-1,+1); map_view checks it',
        clamps=f'positions clamped to [{SL.LO}, {SL.HI}] mm; each field then clamped to its {list(SL.CLAMP_PCT)} '
               'percentile over the clean training lines (maps[arm].clamp); NaN positions: no correction',
        pointing=('A, C: not used (adds nothing on held-out cosmics, re-audit 2.3); D: B-D lines + horizontal capsule '
                  'pointing cells (maps.D.fit.pointing)') if not a.ac_pointing else 'A, C, D: lines + pointing',
        extrapolation=extrap,
        status='production map of the n_TOF analysis view (stage3_r3_final)' if a.status is None else a.status)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(meta=meta, maps=maps), open(out, 'w'))
    print('wrote', out, 'md5', md5(out))
    for arm, m in maps.items():
        print(arm, 'lam', m['fit']['lam_t'], m['fit']['lam_s'], 'clamp', {k: [round(v, 4) for v in c]
                                                                         for k, c in m['clamp'].items()})


# ================================================================================================= held-out
def metrics(D, R, rng):
    c = D.clean.to_numpy()
    edge = ((D.x.abs() > 125) | (D.y.abs() > 125)).to_numpy()
    runs = D.run.to_numpy()
    ur = np.unique(runs)
    rs = {r: np.flatnonzero(runs == r) for r in ur}
    boots = [np.concatenate([rs[r] for r in rng.choice(ur, len(ur))]) for _ in range(150)]
    rows = []
    for w in (0, 1):
        vw = (D.view == w).to_numpy()
        for k, e in R.items():
            sel = vw & c & np.isfinite(e)
            row = dict(view='xy'[w], model=k, n_clean=int(sel.sum()), sig_clean=SL.mad(e[sel]),
                       sig_clean_err=float(np.std([SL.mad(e[b][sel[b]]) for b in boots])))
            row['w68'], row['w95'] = SL.widths(e[sel])
            for lab, rm in (('edge', edge), ('int', ~edge)):
                s2 = sel & rm
                row[f'sig_{lab}'] = SL.mad(e[s2])
                row[f'w68_{lab}'], row[f'w95_{lab}'] = SL.widths(e[s2])
                row[f'n_{lab}'] = int(s2.sum())
            a = vw & np.isfinite(e)
            row['sig_all'] = SL.mad(e[a])
            row['tail_all'] = float(np.mean(np.abs(e[a]) > 0.15))
            rows.append(row)
    return rows


def cmd_heldout(a):
    L, B, late, _ = read_inputs(a)
    out = []
    for arm in ('A', 'C', 'D'):
        D = SL.rows_d(B) if arm == 'D' else SL.rows_ac(L, arm)
        D['part'] = D.run.str.replace('run_', '').astype(int) % 2          # run parity for every chamber
        LT, LS = SL.lam_grids(arm)
        names = ['none', 'free50'] + (['free100'] if arm == 'D' else []) + ['spline_lines'] + \
            (['spline_joint'] if arm == 'D' else [])
        R = {k: np.full(len(D), np.nan) for k in names}
        pc = SL.pointing_cells(late, arm) if arm == 'D' else None
        if pc is not None:
            base = SL.Spline(tilt='pot', scale='sep', lam_grid_t=LT, lam_grid_s=LS).fit(D[D.clean])
            P = SL.point_rows(pc, arm)
            fl = SL.floor_from(base, P)
        lam = []
        for tr in (0, 1):
            trn = D[(D.part == tr) & D.clean]
            tst = (D.part != tr).to_numpy() & D.allok.to_numpy()
            R['none'][tst] = SL.NoMap().corrected_resid(D[tst])
            R['free50'][tst] = SL.Free().fit(trn).corrected_resid(D[tst])
            if 'free100' in R:
                R['free100'][tst] = SL.Free(edges=np.arange(-200.0, 201.0, 100.0), n_min=60).fit(trn) \
                    .corrected_resid(D[tst])
            lines = SL.Spline(tilt='pot', scale='sep', lam_grid_t=LT, lam_grid_s=LS).fit(trn)
            R['spline_lines'][tst] = lines.corrected_resid(D[tst])
            lam.append((tr, lines.lam_t_sel, lines.lam_s_sel))
            if 'spline_joint' in R:
                pt = dict(P, floor=fl['floor'] * a.floor_mult, sig_line=np.mean(list(lines.sig.values())),
                          kappas=SL.KAP)
                jm = SL.Spline(tilt='pot', scale='sep', lam_t=lines.lam_t_sel, lam_s=lines.lam_s_sel,
                               point=pt).fit(trn)
                R['spline_joint'][tst] = jm.corrected_resid(D[tst])
            print(f'{arm} train parity {tr}: lam {lines.lam_t_sel:g} {lines.lam_s_sel:g}', flush=True)
        rows = metrics(D, R, np.random.default_rng(5))
        for r in rows:
            r.update(arm=arm, protocol='train one run parity, test the other (every line tested once)',
                     lam=str(lam))
        out += rows
    T = pd.DataFrame(out)
    cols = ['arm', 'view', 'model'] + [c for c in T.columns if c not in ('arm', 'view', 'model')]
    T[cols].to_csv(a.out, index=False)
    pd.set_option('display.width', 220)
    show = T.assign(**{k: T[k] * 1e3 for k in ('sig_clean', 'sig_clean_err', 'w68', 'w95', 'sig_edge', 'sig_int')})
    print(show[['arm', 'view', 'model', 'n_clean', 'sig_clean', 'sig_clean_err', 'w68', 'w95', 'sig_edge',
                'sig_int']].round(2).to_string(index=False))
    print('wrote', a.out)


# ================================================================================================= beam
def beam_metrics(arm, g, A, shc, valid, e_cells, hx, rng):
    from rerun.ntof.improve.track_quality import pointing
    from sept26_prelim_analysis.capsule_tof import crossing
    from sept26_prelim_analysis.pass1_tracks import SOURCE_XZ_MM
    src = {'A': SOURCE_XZ_MM[0], 'C': SOURCE_XZ_MM[0], 'D': SOURCE_XZ_MM[1]}
    r = shc - A['e']
    tk = SL.track_ok(A, hx)
    reg = tk & (np.abs(A['y']) <= 125)
    c0, _, n0, core = SL.centre(r[reg])
    X, Y = np.where(tk, A['x'], np.nan), np.where(tk, A['y'], np.nan)
    m = SL.cell_map(X, Y, r, SL.E25, nmin=300)
    v = valid & np.isfinite(m['c'])
    cc, ee = m['c'][v], e_cells[v]
    Xd = np.vstack([np.ones(v.sum()), ee]).T
    det = cc - Xd @ np.linalg.lstsq(Xd, cc, rcond=None)[0]
    h, n = SL.FR[arm]
    pos = g[f'p0_{h}'].to_numpy(float)
    pn = g[f'p0_{n}'].to_numpy(float)
    disp = -pn * shc
    s = np.abs(shc) >= 0.1
    pos, disp = pos[s], disp[s]
    miss = pos + disp - src[arm]
    P = pointing(miss, pos + disp[rng.permutation(len(disp))] - src[arm])
    xc, xce, _ = crossing(pos, disp)
    return dict(arm=arm, n=int(len(r)), core_tan=core, centre=c0, cellrms=float(np.std(cc)),
                cellrms_detr=float(np.std(det)), n_cells=int(v.sum()), w68_mm=P.get('w68'), f15=P.get('f1'),
                exc15=P.get('exc1'), exc15_err=P.get('exc1_err'), x_cross=xc, x_cross_err=xce, n_mm=int(s.sum()))


def cmd_beam(a):
    d = pd.read_parquet(a.late)
    if 'd_x_map' not in d:
        raise SystemExit(f'{a.late} has no mapped directions (d_<k>_map): build it from a MAPPED view')
    d = SL.load_late(d)
    extra = {Path(f).stem: SM.load(f) for f in (a.map or [])}
    rng = np.random.default_rng(7)
    rows = []
    for arm in ('A', 'C', 'D'):
        g = d[d.arm == arm]
        A = SL.arm_arrays(g, arm)
        Am = SL.arm_arrays(g, arm, mapped=True)
        A['e_normal'] = SL.normal_halfwidth(A['sh'], A['y']) + 0.06
        hx = SL.holes(A['x'], A['y'])
        pc = SL.pointing_cells(d, arm)
        specs = {'none': A['sh'], 'view': Am['sh']}
        for k, M in extra.items():
            specs[k] = SM.correct_global(M, arm, A['x'], A['y'], A['sh'], A['sv'])[0]
        for lab, shc in specs.items():
            row = beam_metrics(arm, g, A, shc, pc['valid'], pc['e'], hx, rng)
            row['map'] = lab
            rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(a.out, index=False)
    pd.set_option('display.width', 220)
    print(T[['arm', 'map', 'n', 'core_tan', 'cellrms', 'cellrms_detr', 'n_cells', 'f15', 'exc15', 'w68_mm',
             'x_cross']].round(5).to_string(index=False))
    print('wrote', a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('points')
    p.add_argument('--view', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--jobs', type=int, default=3)
    for name in ('fit', 'heldout'):
        p = sub.add_parser(name)
        p.add_argument('--lines', required=True, help='A-C lines (line_residuals.py build)')
        p.add_argument('--bd-lines', required=True, help='B-D lines (bd_map.py)')
        p.add_argument('--points', required=True, help='late-track cache (points) or an unmapped view directory')
        p.add_argument('--out', required=True)
        p.add_argument('--jobs', type=int, default=3)
        p.add_argument('--floor-mult', type=float, default=1.0, help='pointing floor x this (prototype: 1)')
        if name == 'fit':
            p.add_argument('--ac-pointing', action='store_true', help='also use the pointing cells on A and C')
            p.add_argument('--status', default=None)
    p = sub.add_parser('beam')
    p.add_argument('--late', required=True, help='late-track cache built (points) from a MAPPED view')
    p.add_argument('--map', action='append', help='extra map file applied to the unmapped slopes')
    p.add_argument('--out', required=True)
    a = ap.parse_args()
    {'points': cmd_points, 'fit': cmd_fit, 'heldout': cmd_heldout, 'beam': cmd_beam}[a.cmd](a)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
