#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracking_audit.py -- audit of the n_TOF pass-1 tracks and of the capsule imaging (2026-11-21).

Question: are the oddities of the capsule imaging (only 45-60 % of late tracks "in focus", early tracks that do not
point, D's two different z crossings, the 12-20 ms pair population) tracking effects or poor track selection?

Every number comes from the pass-1 analysis view (pass1_tracks, `gated` = true |tan| < 0.5, map and censored-window cut
included) and from Geant4 through the pass-1 reconstruction ($WORK_DATA/ntof_pass1/g4_digi).  Existing data only.

    python -m rerun.ntof.improve.tracking_audit cache      -> <out>/tracks_ACD.parquet   (A, C, D reco-gated rows, 10 min)
    python -m rerun.ntof.improve.tracking_audit tables     -> <out>/*.csv                (3 min)
    python -m rerun.ntof.improve.tracking_audit bow        -> <out>/bow_partner.csv      (run_145, 1 min)
    ~/.venvs/xrd/bin/python rerun/ntof/improve/tracking_audit.py slim   -> <out>/slim_files.csv (EOS listing, XRootD)
    python -m rerun.ntof.improve.make_audit_report         -> <out>/report.html + figures/

Out: $WORK_DATA/ntof_pass1/ana/tracking_audit/ (X17_AUDIT_OUT overrides).  Findings: README.md of this directory.

The track-quality selection tested here ("clean") is a proposal, not part of the analysis view:
    min(x_q_sum, y_q_sum) >= Q_MIN        a minimum-ionising track leaves at least this much charge in each view
    max(x_n_strips, y_n_strips) < N_MAX   a fit window wider than this is not one track
    x_cens_frac, y_cens_frac < CENS_MAX   at most half of a fit window on censored (dead / hot) strips
    chi2/dof < CHI2_REL x the median of late, wall-confirmed, pointing tracks of that chamber and view
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:          # the XRootD venv of the `slim` step has no numpy
    np = None

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

Q_MIN = 600.0
N_MAX = 30
CENS_MAX = 0.5
CHI2_REL = 5.0
ARMS = ('A', 'C', 'D')
PRE_ACCESS = ('run_79', 'run_81')
TOF = [1, 4, 8, 12, 15, 20, 30, 75]
TOF_PAIRS = [4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75]
#: the plane's distance from the axis (geometry.MM_DIST_Z + W_STRIP), as capsule_tof.D_PLANE
D_PLANE = 234.6
FID = 100.0            # |x_local| below this: the capsule can reach every strip with |tan| < 0.5
COLS = ['run', 'subrun', 'tag', 'event_id', 'arm', 'track_id', 'gated', 'gated_nomap', 'censored_window',
        'x_local', 'y_local', 'tanx', 'tany', 'p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z', 'x_t0', 'y_t0',
        't_since_flash_ns', 'chi2dof_x', 'chi2dof_y', 'x_n_strips', 'y_n_strips', 'n_cand_x', 'n_cand_y',
        'x_q_sum', 'y_q_sum', 'x_q_uend', 'y_q_uend', 'x_cens_frac', 'y_cens_frac',
        'wall_A', 'wall_C', 'wall_D', 'plastic_A', 'plastic_C', 'plastic_D']


def out_dir() -> Path:
    p = os.environ.get('X17_AUDIT_OUT')
    d = Path(p) if p else Path(os.environ.get('WORK_DATA', '/media/ucla/work')) / 'ntof_pass1' / 'ana' / 'tracking_audit'
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------------------------------------------------ cache
def cache() -> None:
    import pyarrow.dataset as ds
    from sept26_prelim_analysis import pass1_tracks as P1
    d = P1.read(COLS, arms=ARMS, filter=(ds.field('gated_reco') == True))  # noqa: E712
    d.to_parquet(out_dir() / 'tracks_ACD.parquet', index=False)
    print(f'{len(d):,} rows ->', out_dir() / 'tracks_ACD.parquet')


def load():
    """Gated beam tracks of A, C, D with the pointing quantities.  `u` is global x for A/C and global z for D."""
    import pandas as pd
    d = pd.read_parquet(out_dir() / 'tracks_ACD.parquet')
    n_all = d.groupby('run').size()
    untimed = d[d.t_since_flash_ns.isna()].groupby('run').size().reindex(n_all.index).fillna(0)
    cov = pd.DataFrame(dict(n=n_all, untimed=untimed)).reset_index()
    d = d[d.gated.fillna(False).astype(bool) & d.t_since_flash_ns.notna() & ~d.run.isin(PRE_ACCESS)].copy()
    d['ms'] = d.t_since_flash_ns / 1e6
    ac = d.arm.isin(['A', 'C']).to_numpy()
    t = np.where(ac, -d.p0_z / d.d_z, -d.p0_x / d.d_x)
    d['pos'] = np.where(ac, d.p0_x, d.p0_z)
    d['src_u'] = d.pos + t * np.where(ac, d.d_x, d.d_z)
    d['disp'] = d.src_u - d.pos
    d['src_y'] = d.p0_y + t * d.d_y
    d['wall'] = False
    for a in ARMS:
        m = d.arm == a
        d.loc[m, 'wall'] = d.loc[m, f'wall_{a}'].astype('boolean').fillna(False).astype(bool)
    d['qmin'] = np.minimum(d.x_q_sum, d.y_q_sum)
    d['nmax'] = np.maximum(d.x_n_strips, d.y_n_strips)
    d['cens'] = np.maximum(d.x_cens_frac.fillna(0), d.y_cens_frac.fillna(0))
    return d, cov


def stats(m) -> dict:
    m = np.asarray(m, float)
    m = m[np.isfinite(m)]
    if len(m) == 0:
        return dict(n=0, f15=np.nan, f30=np.nan, f90=np.nan, hw50=np.nan)
    a = np.abs(m)
    return dict(n=len(m), f15=(a < 15).mean(), f30=(a < 30).mean(), f90=(a < 90).mean(), hw50=np.median(a))


def add_selection(d):
    """Source coordinate per arm (late band crossing), the chi2 references and the `clean` flag."""
    import pandas as pd
    from sept26_prelim_analysis.capsule_tof import crossing
    rows = []
    d['miss'] = np.nan
    d['clean'] = False
    for a in ARMS:
        m = (d.arm == a).to_numpy()
        g = d[m]
        base = (g.qmin >= Q_MIN) & (g.nmax < N_MAX) & (g.cens < CENS_MAX)
        late = g[(g.ms >= 20) & base]
        u0, e0, _ = crossing(late.pos.to_numpy(), late.disp.to_numpy())
        miss = g.src_u - u0
        ref = (g.ms >= 30) & g.wall & (miss.abs() < 15) & base
        cx, cy = g.chi2dof_x[ref].median(), g.chi2dof_y[ref].median()
        clean = base & (g.chi2dof_x < CHI2_REL * cx) & (g.chi2dof_y < CHI2_REL * cy)
        d.loc[m, 'miss'] = miss.to_numpy()
        d.loc[m, 'clean'] = clean.to_numpy()
        ref_all = (g.ms >= 30) & g.wall & (miss.abs() < 15)
        parts = {'charge': g.qmin >= Q_MIN, 'width': g.nmax < N_MAX, 'censored': g.cens < CENS_MAX,
                 'chi2 x': g.chi2dof_x < CHI2_REL * cx, 'chi2 y': g.chi2dof_y < CHI2_REL * cy, 'all (clean)': clean}
        for name, c in parts.items():
            rows.append(dict(arm=a, cut=name, source_u=u0, chi2_ref_x=cx, chi2_ref_y=cy,
                             keeps_reference=c[ref_all].mean(), removes_lt12=1 - c[g.ms < 12].mean(),
                             removes_12_20=1 - c[(g.ms >= 12) & (g.ms < 20)].mean(), removes_ge20=1 - c[g.ms >= 20].mean()))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------- tables
def tof_table(d):
    import pandas as pd
    from sept26_prelim_analysis.capsule_tof import crossing
    rows = []
    for a in ARMS:
        g = d[d.arm == a]
        for lo, hi in zip(TOF[:-1], TOF[1:]):
            w = g[(g.ms >= lo) & (g.ms < hi)]
            for sel, L in (('gated', w), ('clean', w[w.clean])):
                fid = L.x_local.abs() < FID
                u0, e0, _ = crossing(L.pos.to_numpy(), L.disp.to_numpy())
                rows.append(dict(arm=a, ms_lo=lo, ms_hi=hi, sel=sel, rate_per_ms=len(L) / (hi - lo),
                                 **stats(L.miss[fid]), not_pointing=(L.miss.abs() > 60).mean() if len(L) else np.nan,
                                 wall=L.wall.mean() if len(L) else np.nan, low_charge=(L.qmin < Q_MIN).mean() if len(L) else np.nan,
                                 q_med=L.qmin.median(), chi2x_med=L.chi2dof_x.median(), nstrips_med=L.x_n_strips.median(),
                                 crossing=u0, crossing_err=e0))
    return pd.DataFrame(rows)


def charge_hist(d):
    import pandas as pd
    e = np.array([0, 100, 200, 300, 400, 500, 600, 800, 1000, 1300, 1600, 2000, 3000, 5000, 1e9])
    rows = []
    for a in ARMS:
        g = d[d.arm == a]
        for lo, hi in ((1, 8), (8, 12), (12, 20), (20, 75)):
            w = g[(g.ms >= lo) & (g.ms < hi)]
            h, _ = np.histogram(w.qmin, e)
            for k in range(len(h)):
                rows.append(dict(arm=a, ms_lo=lo, ms_hi=hi, q_lo=e[k], q_hi=e[k + 1], n=h[k], frac=h[k] / max(len(w), 1),
                                 f30=(w.miss[(w.qmin >= e[k]) & (w.qmin < e[k + 1]) & (w.x_local.abs() < FID)].abs() < 30).mean()))
    return pd.DataFrame(rows)


def pairs_table(d):
    import pandas as pd
    key = ['run', 'subrun', 'tag', 'event_id', 'arm']
    g = d[d.arm.isin(['A', 'C']) & (d.x_t0 <= 300) & (d.y_t0 <= 300)]
    rows = []
    for sel, G in (('gated', g), (f'charge >= {Q_MIN:.0f}', g[g.qmin >= Q_MIN]), ('clean', g[g.clean])):
        ev = G.groupby(key).agg(n=('qmin', 'size'), ms=('ms', 'first'), qlo=('qmin', 'min')).reset_index()
        for a in ('A', 'C'):
            E = ev[ev.arm == a]
            for lo, hi in zip(TOF_PAIRS[:-1], TOF_PAIRS[1:]):
                S = E[(E.ms >= lo) & (E.ms < hi)]
                p = S[S.n >= 2]
                rows.append(dict(arm=a, sel=sel, ms_lo=lo, ms_hi=hi, events=len(S), pairs=len(p),
                                 pair_pct=100 * len(p) / max(len(S), 1), err=100 * np.sqrt(len(p)) / max(len(S), 1),
                                 low_charge_leg=(p.qlo < Q_MIN).mean() if len(p) else np.nan))
    return pd.DataFrame(rows)


def position_hist(d):
    """Where the late tracks that do not point (|miss| > 60 mm) sit on each chamber, 5 mm bins."""
    import pandas as pd
    e = np.arange(-200, 200.01, 5.0)
    rows = []
    for a in ARMS:
        L = d[(d.arm == a) & (d.ms >= 20)]
        bad = L.miss.abs() > 60
        for col in ('x_local', 'y_local'):
            h, _ = np.histogram(L[col], e)
            hb, _ = np.histogram(L[col][bad], e)
            hc, _ = np.histogram(L[col][bad & L.clean], e)
            for k in range(len(h)):
                rows.append(dict(arm=a, coord=col, lo=e[k], n=h[k], not_pointing=hb[k], not_pointing_clean=hc[k]))
    return pd.DataFrame(rows)


def g4_tables(d):
    """The same pointing statistics on Geant4: truth gap lines and the pass-1 reconstruction of the same events."""
    import warnings
    import pandas as pd
    from ntof_cosmics.g4_digi import analyse as AN
    AN.OUT = Path(os.environ.get('WORK_DATA', '/media/ucla/work')) / 'ntof_pass1' / 'g4_digi'
    e = np.arange(-150, 150.01, 5.0)
    ey = np.arange(-140, 140.01, 10.0)
    rows, hist, prof = [], [], []
    for a in ('A', 'C'):
        for sel in ('gated', 'clean'):
            L = d[(d.arm == a) & (d.ms >= 20) & (d.x_local.abs() < FID)]
            L = L[L.clean] if sel == 'clean' else L
            rows.append(dict(sample=f'data {a}, {sel}', arm=a, **stats(L.miss)))
            if sel == 'clean':
                h, _ = np.histogram(L.miss, e)
                hist += [dict(sample=f'data {a}', lo=e[k], frac=h[k] / len(L)) for k in range(len(h))]
    for a in ARMS:
        L = d[(d.arm == a) & (d.ms >= 20) & d.clean & (d.miss.abs() < 15)]
        h, _ = np.histogram(L.src_y, ey)
        prof += [dict(sample=f'data {a}', lo=ey[k], frac=h[k] / h.sum()) for k in range(len(h))]
    for lib in ('nose', 'ear2_L19p5'):
        for a in ('A', 'C'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                R = AN.load(f'g4_{a}_{lib}_pass1')
            f = AN.foot(R)
            T = R[np.isfinite(R.tan_u) & np.isfinite(R.u_mesh) & (R.tan_u.abs() < 0.5) & (R.tan_v.abs() < 0.5)]
            mt = T.u_mesh - T.tan_u * D_PLANE - f
            fid = T.u_mesh.abs() < FID
            rows.append(dict(sample=f'Geant4 {lib} {a}, truth', arm=a, **stats(mt[fid])))
            for lo, hi in ((1, 2), (2, 4), (4, 20)):
                k = fid & (T.dom_ke_gap >= lo) & (T.dom_ke_gap < hi)
                rows.append(dict(sample=f'Geant4 {lib} {a}, truth, {lo}-{hi} MeV', arm=a, **stats(mt[k])))
            G = R[R.x_ok & R.y_ok & (R.x_tan_theta.abs() < 0.5) & (R.tan_v_reco.abs() < 0.5)]
            mr = G.xl - G.x_tan_theta * D_PLANE - f
            rows.append(dict(sample=f'Geant4 {lib} {a}, pass-1 reconstruction', arm=a, **stats(mr[G.xl.abs() < FID])))
            if lib == 'nose':
                h, _ = np.histogram(mt[fid], e)
                hist += [dict(sample=f'Geant4 truth {a}', lo=e[k], frac=h[k] / fid.sum()) for k in range(len(h))]
                h, _ = np.histogram(mr[G.xl.abs() < FID], e)
                hist += [dict(sample=f'Geant4 reconstructed {a}', lo=e[k], frac=h[k] / (G.xl.abs() < FID).sum())
                         for k in range(len(h))]
            yt = (T.v_mesh - T.tan_v * D_PLANE)[mt.abs() < 15]
            h, _ = np.histogram(yt, ey)
            prof += [dict(sample=f'Geant4 {lib} truth {a}', lo=ey[k], frac=h[k] / h.sum()) for k in range(len(h))]
            yr = (G.vl - G.tan_v_reco * D_PLANE)[mr.abs() < 15]
            h, _ = np.histogram(yr, ey)
            prof += [dict(sample=f'Geant4 {lib} reconstructed {a}', lo=ey[k], frac=h[k] / h.sum()) for k in range(len(h))]
    return pd.DataFrame(rows), pd.DataFrame(hist), pd.DataFrame(prof)


def half_max_edges(lo, frac):
    """Half-maximum edges of a flat-topped profile: where it crosses half of the mean of its five highest bins."""
    lo, frac = np.asarray(lo, float), np.asarray(frac, float)
    c = lo + 0.5 * (lo[1] - lo[0])
    top = np.sort(frac)[-5:].mean()
    i = np.where(frac >= 0.5 * top)[0]
    a, b = i[0], i[-1]
    x_lo = np.interp(0.5 * top, [frac[a - 1], frac[a]], [c[a - 1], c[a]])
    x_hi = np.interp(0.5 * top, [frac[b + 1], frac[b]], [c[b + 1], c[b]])
    return float(x_lo), float(x_hi)


def tables() -> None:
    import pandas as pd
    out = out_dir()
    d, cov = load()
    cuts = add_selection(d)
    cuts.to_csv(out / 'cuts.csv', index=False)
    print(cuts.round(3).to_string(index=False))
    tof_table(d).to_csv(out / 'tof.csv', index=False)
    charge_hist(d).to_csv(out / 'charge_hist.csv', index=False)
    pairs_table(d).to_csv(out / 'pairs_tof.csv', index=False)
    position_hist(d).to_csv(out / 'positions.csv', index=False)
    P, H, Y = g4_tables(d)
    P.to_csv(out / 'pointing_data_vs_g4.csv', index=False)
    H.to_csv(out / 'miss_hist.csv', index=False)
    Y.to_csv(out / 'height_profile.csv', index=False)
    E = []
    for s, g in Y.groupby('sample'):
        lo, hi = half_max_edges(g.lo, g.frac)
        E.append(dict(sample=s, edge_lo=lo, edge_hi=hi, length=hi - lo, centre=0.5 * (lo + hi)))
    pd.DataFrame(E).to_csv(out / 'height_edges.csv', index=False)
    cov = cov[(cov.untimed > 0) & (cov.untimed < cov.n)]          # production runs: some rows timed, some not
    cov['untimed_frac'] = cov.untimed / cov.n
    cov.to_csv(out / 'untimed_by_run.csv', index=False)
    print(P.round(3).to_string(index=False))
    print(pd.DataFrame(E).round(1).to_string(index=False))
    print(f'production rows without a time since flash: {cov.untimed.sum() / cov.n.sum():.3f}')


# -------------------------------------------------------------------------------------------- the bow-partner check
def bow() -> None:
    """wft.reco._worker_fit maps each pair's tangents with its own partner, then maps the two plane winners together.
    When the winners belong to different pairs, each gets the other pair's coordinate.  Recompute from (w, p0)."""
    import glob
    import pandas as pd
    from wft.calib import CalibrationBundle
    from wft import reco as R
    base = Path(os.environ.get('WORK_DATA', '/media/ucla/work')) / 'ntof_pass1'

    class F:
        pass
    cols = ['subrun', 'tag', 'event_id', 'arm', 'track_id', 'gated', 'x_p0', 'y_p0', 'x_w', 'y_w', 'x_tan_theta',
            'y_tan_theta', 'x_rank', 'y_rank', 't_since_flash_ns']
    d = pd.concat([pd.read_parquet(f, columns=cols)
                   for f in sorted(glob.glob(str(base / 'stage3_vt' / 'tracks_run_145_*.parquet')))], ignore_index=True)
    d = d[d.arm.isin(['A', 'C']) & d.gated.fillna(False).astype(bool) & (d.t_since_flash_ns >= 12e6)].copy()
    d['nt'] = d.groupby(['subrun', 'tag', 'event_id', 'arm']).track_id.transform('size')
    rows = []
    for arm in 'AC':
        cal = CalibrationBundle.load(str(base / 'bundles' / 'pf' / f'{arm}_E2_beam'))
        a = d[d.arm == arm]
        for name, s in (('one track in the chamber', a[a.nt == 1].sample(8000, random_state=1)),
                        ('two or more tracks in the chamber', a[a.nt >= 2])):
            dd = []
            for r in s.itertuples():
                fx, fy = F(), F()
                fx.p0, fx.w, fy.p0, fy.w = r.x_p0, r.x_w, r.y_p0, r.y_w
                U, W = R.bow_UW(cal, fx, fy)
                dd.append(max(abs(r.x_tan_theta - R.tan_bow(cal, 'x', r.x_w, U, W)),
                              abs(r.y_tan_theta - R.tan_bow(cal, 'y', r.y_w, U, W))))
            dd = np.array(dd)
            wrong = dd > 1e-6
            rows.append(dict(arm=arm, sample=name, n=len(s), wrong_frac=wrong.mean(),
                             dtan_median=np.median(dd[wrong]) if wrong.any() else 0.0,
                             dtan_p90=np.quantile(dd[wrong], 0.9) if wrong.any() else 0.0, dtan_max=dd.max()))
    t = pd.DataFrame(rows)
    t.to_csv(out_dir() / 'bow_partner.csv', index=False)
    print(t.round(4).to_string(index=False))


# -------------------------------------------------------------------------------------------- slim files on EOS
def slim() -> None:
    """How many slim ROOT files each production sub-run has on EOS.  Needs the XRootD bindings (~/.venvs/xrd) and
    <out>/untimed_by_subrun.csv is not needed: it lists every sub-run directory of the production runs."""
    import csv
    from XRootD import client
    fs = client.FileSystem('root://eospublic.cern.ch')
    base = '/eos/experiment/ntof/data/x17/july_beam/runs/'
    runs = [r['run'] for r in csv.DictReader(open(out_dir() / 'untimed_by_run.csv'))]
    rows = []
    for run in runs:
        st, ls = fs.dirlist(base + run)
        for sub in sorted(e.name for e in (ls or []) if e.name.startswith('stat')):
            st2, l2 = fs.dirlist(f'{base}{run}/{sub}/ntof_hits', flags=1)
            f = sorted((e.name, e.statinfo.size) for e in (l2 or []) if e.name.endswith('.root'))
            if f:
                rows.append(dict(run=run, subrun=sub, n_slim=len(f), first_bytes=f[0][1],
                                 all_bytes=sum(x[1] for x in f), files=';'.join(x[0] for x in f)))
    with open(out_dir() / 'slim_files.csv', 'w') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    multi = [r for r in rows if r['n_slim'] > 1]
    print(f'{len(rows)} sub-runs, {len(multi)} with more than one slim file; '
          f'{1 - sum(r["first_bytes"] for r in rows) / sum(r["all_bytes"] for r in rows):.3f} of the bytes are not in the first')


if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) > 1 else 'tables'
    {'cache': cache, 'tables': tables, 'bow': bow, 'slim': slim}[step]()
