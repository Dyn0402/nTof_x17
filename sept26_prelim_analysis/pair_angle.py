#!/usr/bin/env python3
"""
pair_angle -- what relative angle do two tracks in one chamber have, and how
much does it help to tell them apart?

A companion to `two_track_limit` (R1-R4), which bins everything in separation.
Three questions, each answered from its own source:

  1. Does the relative angle matter, with truth known?   (synthetic, R5)
     The R2 oracle (perfect forward model, white noise at the run_145 level,
     tied t0, ideal fitter) on pairs whose second track diverges from the first
     by D mm over the drift column: ``asimov`` (noise-free Delta-chi2) and
     ``oracle`` (efficiency at the 1 % false-split threshold of R2's own
     synthetic singles, tan 0.3, view x).
  2. What relative angles do the bench pairs carry?      (data: donors, R3)
     Donors are clean single tracks that point at the capsule, so their slope
     is tied to their position. Pairs of donors from different triggers carry
     that tie plus the spread of the two vertices.
  3. What relative angles do real pairs have?            (data: det_a_intra)
     Every pair of tracks in one chamber A trigger, against pairs mixed across
     triggers; plus the IPC prior (`ipc_born`) through a point-source toy of
     the four planes, which says where a common-vertex pair lands.

    python -m sept26_prelim_analysis.pair_angle asimov
    python -m sept26_prelim_analysis.pair_angle oracle --n 40 --jobs 15
    python -m sept26_prelim_analysis.pair_angle data
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import paths                     # noqa: E402
from sept26_prelim_analysis import two_track_limit as TL     # noqa: E402

ARMS = ('A', 'C')
#: base slope of track a; R2's representative inclination
TAN0 = 0.3
#: mesh separation [mm] x divergence over the drift column [mm]
D_MESH = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
DIVERGE = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
#: drift gap [mm] the in-chamber divergence is quoted over for data
GAP_MM = 30.0
#: strip plane distance from the beam axis [mm]: mylar face + 30.1 mm
L_STRIP = 204.5 + 30.1
FSR = 0.01


def out_dir() -> Path:
    d = paths.out('two_track_limit', 'pair_angle')
    d.mkdir(parents=True, exist_ok=True)
    return d


def cells():
    """(d, D) with D the signed change of separation over the column: D >= 0
    diverging, D = -2d the symmetric crossing at mid-column."""
    out = [(d, D) for d in D_MESH for D in DIVERGE if not (d == 0 and D == 0)]
    out += [(d, -2 * d) for d in D_MESH if d > 0]
    return out


# --------------------------------------------------------------------------- #
# R5: synthetic, truth known
# --------------------------------------------------------------------------- #
def _tracks(d, D, p0a=100.3):
    from sept26_prelim_analysis import two_track_synth as ts
    w = TAN0 * ts._CAL.v_drift * 1e-3
    return (p0a, w), (p0a + d, w + D / TL.UEND)


def _asimov_job(args):
    arm, d, D = args
    TL._init(arm)
    (pa, wa), (pb, wb) = _tracks(d, D)
    tr = [(pa, wa, 0.0, TL.QTOT), (pb, wb, 0.0, TL.QTOT)]
    P = TL.synth_window('x', tr, None, noise=0.0, flat=True)
    P.noise[:] = TL.NOISE
    lo, hi = min(pa, pb, pb + D) - 1, max(pa, pb, pb + D) + 1
    wm = 0.5 * (wa + wb)
    starts = [(p, w, t) for p in np.arange(lo, hi + 1e-3, 0.1) for w in (wa, wm, wb)
              for t in (-120, -60, 0, 60, 120)]
    lam, _x = TL.fit_one(P, starts)
    return dict(arm=arm, d=d, D=D, lam=float(lam),
                rms=float(TL.sep_rms((pa, wa, 0.0), (pb, wb, 0.0))))


def asimov(jobs):
    J = [(a, d, D) for a in ARMS for d, D in cells()]
    with ProcessPoolExecutor(jobs) as ex:
        R = pd.DataFrame(list(ex.map(_asimov_job, J, chunksize=1)))
    R.to_csv(out_dir() / 'r5_asimov.csv', index=False)
    print(R.pivot_table(index='d', columns=['arm', 'D'], values='lam').round(0).to_string())


def _oracle_job(args):
    arm, d, D, seed = args
    TL._init(arm)
    rng = np.random.default_rng(seed)
    p0a = float(100.0 + rng.uniform(0.0, 0.78))
    (pa, wa), (pb, wb) = _tracks(d, D, p0a)
    t = time.perf_counter()
    tracks = [(pa, wa, 0.0, TL.QTOT), (pb, wb, 0.0, TL.QTOT)]
    P = TL.synth_window('x', tracks, rng)
    mid, wm = 0.5 * (pa + pb), 0.5 * (wa + wb)
    starts = [(pa, wa, 0.0), (pb, wb, 0.0), (mid, wm, 0.0)]
    c1, x1 = TL.fit_one(P, starts)
    truth = [(pa, wa, 0.0), (pb, wb, 0.0)]
    c2, x2 = TL.fit_two(P, x1, truth)
    found, ds = TL.match([(x2[0], x2[1], x2[4]), (x2[2], x2[3], x2[4])], truth)
    return dict(arm=arm, d=d, D=D, seed=seed, dchi2=float(c1 - c2), found=bool(found),
                rms=float(TL.sep_rms((pa, wa, 0.0), (pb, wb, 0.0))),
                sec=time.perf_counter() - t)


def oracle(n: int, jobs: int, seed: int):
    J, k = [], 0
    for a in ARMS:
        for d, D in cells():
            for _i in range(n):
                J.append((a, d, D, seed + k))
                k += 1
    rows, t = [], time.time()
    with ProcessPoolExecutor(jobs) as ex:
        for i, r in enumerate(ex.map(_oracle_job, J, chunksize=2)):
            rows.append(r)
            if (i + 1) % 200 == 0:
                print(f'{i + 1}/{len(J)}  {time.time() - t:.0f} s', flush=True)
                pd.DataFrame(rows).to_parquet(out_dir() / 'r5_oracle.partial.parquet')
    R = pd.DataFrame(rows)
    R.to_parquet(out_dir() / 'r5_oracle.parquet')
    print(oracle_table(R).to_string())


def r2_threshold() -> dict:
    """R2's 1 % false-split threshold on its own synthetic singles, view x,
    tan 0.3: the same fitter and noise, so it applies to R5 unchanged."""
    r2 = pd.read_parquet(TL.out_dir() / 'r2_oracle.parquet')
    s = r2[(r2.n_true == 1) & (r2.plane == 'x') & np.isclose(r2.tan, TAN0)]
    return {a: float(np.quantile(g.dchi2, 1 - FSR)) for a, g in s.groupby('arm')}


def oracle_table(R: pd.DataFrame) -> pd.DataFrame:
    thr = r2_threshold()
    R = R.assign(ok=(R.dchi2 > R.arm.map(thr)) & R.found)
    T = R.groupby(['arm', 'd', 'D']).agg(eff=('ok', 'mean'), n=('ok', 'size'),
                                         rms=('rms', 'first')).reset_index()
    T['thr'] = T.arm.map(thr)
    return T


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def pointing(D: pd.DataFrame) -> pd.DataFrame:
    """Slope of tan against mesh position per (arm, view) on the clean donors:
    the in-model pointing scale, and the scatter about it."""
    rows = []
    for (arm, v), g in [((a, v), D[D.arm == a]) for a in ARMS for v in 'xy']:
        x, y = g[f'{v}_p0'].to_numpy(), g[f'{v}_tan_theta'].to_numpy()
        s, c = np.polyfit(x, y, 1)
        r = y - (s * x + c)
        rows.append(dict(arm=arm, view=v, n=len(g), slope=s, foot=-c / s,
                         L_eff=1 / abs(s), resid_rms=float(r.std()),
                         resid_rsig=float(0.5 * np.subtract(*np.percentile(r, [84, 16]))),
                         corr=float(np.corrcoef(x, y)[0, 1])))
    return pd.DataFrame(rows)


def ipc_toy(n: int = 3_000_000, seed: int = 5) -> pd.DataFrame:
    """Pairs from a point at the capsule centre, legs drawn from the ipc_born
    opening-angle law (M1, E0), pair axis isotropic. A pair is intra-chamber
    when both legs cross the same strip plane (a 380 x 340 mm square at
    L_STRIP, pinwheel offsets and dead strips ignored). Returns, per channel,
    the weighted separation at the strip plane and the in-chamber divergence."""
    from sept26_prelim_analysis import ipc_born as IB
    rng = np.random.default_rng(seed)
    out = []
    for kind in ('M1', 'E0'):
        S = IB.e0_lab(n) if kind == 'E0' else IB.sample(kind, n)
        th = np.radians(S.theta_deg.to_numpy())
        wgt = S.weight.to_numpy()
        # leg 1 isotropic; leg 2 at theta from it, random azimuth
        u = rng.normal(size=(len(th), 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        a = np.where(np.abs(u[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        e1 = np.cross(u, a)
        e1 /= np.linalg.norm(e1, axis=1)[:, None]
        e2 = np.cross(u, e1)
        ph = rng.uniform(0, 2 * np.pi, len(th))
        v = (np.cos(th)[:, None] * u
             + np.sin(th)[:, None] * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2))
        for axis, sign in ((2, 1), (2, -1), (0, 1), (0, -1)):
            du, dv = u[:, axis] * sign, v[:, axis] * sign
            ok = (du > 0) & (dv > 0)
            other = [i for i in range(3) if i != axis]
            with np.errstate(divide='ignore', invalid='ignore'):
                pu = L_STRIP * u[:, other] / du[:, None]
                pv = L_STRIP * v[:, other] / dv[:, None]
            # in-plane extents: y (beam, index 1) is 340 mm; the tangent is 380
            ext = np.array([[190.0 if o != 1 else 170.0 for o in other]])
            ok &= (np.abs(pu) < ext).all(1) & (np.abs(pv) < ext).all(1)
            sep = np.linalg.norm(pu - pv, axis=1)
            # same lines at the cathode, GAP_MM nearer the source
            f = (L_STRIP - GAP_MM) / L_STRIP
            out.append(pd.DataFrame(dict(kind=kind, theta_deg=S.theta_deg.to_numpy()[ok],
                                         weight=wgt[ok], sep=sep[ok],
                                         div=sep[ok] * (1 - f))))
        tot = wgt.sum()
        out[-1].attrs['tot'] = tot
        out.append(pd.DataFrame(dict(kind=[kind], theta_deg=[np.nan], weight=[0.0],
                                     sep=[np.nan], div=[np.nan], total_weight=[tot])))
    return pd.concat(out, ignore_index=True)


def data():
    od = out_dir()
    ib_dir = paths.out('intra_bench')
    D = pd.read_parquet(ib_dir / 'donors.parquet')
    PT = pointing(D)
    PT.to_csv(od / 'pointing.csv', index=False)
    D.to_parquet(od / 'donors.parquet')
    print(PT.round(4).to_string())

    # R3 per-view pairs with their relative angle, as the report scores them
    from sept26_prelim_analysis import make_two_track_limit_report as R
    L = R.load()
    R3 = L['r3']
    _T, P = R.real_ladder(R3[R3.donor_chi2dof < R.DONOR_CHI2DOF_MAX])
    P = P.loc[:, ~P.columns.duplicated()].copy()
    P['dtan'] = P.tan_b - P.tan_a
    P['dp'] = P.pb - P.pa
    P['div'] = GAP_MM * P.dtan.abs()
    s = PT.set_index(['arm', 'view']).slope
    P['div_cv'] = [GAP_MM * abs(s[(a, 'x')] * dp) for a, dp in zip(P.arm, P.dp)]
    keep = ['oid', 'arm', 'plane', 'sep', 'sep_rms', 'dp', 'dtan', 'div', 'div_cv',
            'real_ok', 'twin_ok', 'prod_ok', 'fixed_ok']
    P[keep].to_parquet(od / 'r3_pairs_angle.parquet')

    # event-level bench (fixed + profc), both views' divergence per overlay
    from sept26_prelim_analysis import intra_bench as ib
    Dn = D.set_index(['arm', 'tag', 'event_id'])
    ev = []
    for v in ('fixed_A_replace_profc', 'fixed_C_replace_profc'):
        d = ib_dir / v
        M = pd.read_parquet(d / 'overlays.parquet')
        C = pd.read_parquet(d / 'candidates.parquet')
        Sc = ib.score(M, C, pd.read_parquet(d / 'donors.parquet'))
        both = (Sc[Sc['mode'] == 'overlay'].groupby('oid').track_found.all().rename('both'))
        M = M[M['mode'] == 'overlay'].set_index('oid').join(both).reset_index()
        for p in 'xy':
            ta = Dn.loc[list(zip(M.arm, M.tag, M.a_eid)), f'{p}_tan_theta'].to_numpy()
            tb = Dn.loc[list(zip(M.arm, M.tag, M.b_eid)), f'{p}_tan_theta'].to_numpy()
            M[f'div_{p}'] = GAP_MM * np.abs(tb - ta)
        ev.append(M[['oid', 'arm', 'cls', 'sep_x', 'sep_y', 'div_x', 'div_y', 'both']])
    pd.concat(ev, ignore_index=True).to_parquet(od / 'bench_events_angle.parquet')

    # real intra-A pairs, campaign: opening angle against separation
    cols = ['mixed', 'open_deg', 'sep_plane_mm', 'both_slope', 'pointing', 'prompt']
    A = pd.read_parquet(paths.out('det_a_intra') / 'pairs.parquet', columns=cols)
    A['cv_deg'] = np.degrees(2 * np.arctan(A.sep_plane_mm / 2 / L_STRIP))
    sb = np.array([0, 12, 24, 50, 100, 150, 200, 250, 300, 400, 520])
    rows = []
    for sel, m in (('all', np.ones(len(A), bool)), ('slope', A.both_slope.to_numpy()),
                   ('slope+pointing', (A.both_slope & A.pointing).to_numpy())):
        for mixed in (False, True):
            g = A[m & (A.mixed == mixed).to_numpy()]
            h, _ = np.histogram(g.sep_plane_mm, sb)
            for i in range(len(sb) - 1):
                q = g[(g.sep_plane_mm >= sb[i]) & (g.sep_plane_mm < sb[i + 1])].open_deg
                rows.append(dict(sel=sel, mixed=mixed, lo=sb[i], hi=sb[i + 1], n=int(h[i]),
                                 frac=float(h[i] / max(len(g), 1)), n_sel=len(g),
                                 q10=q.quantile(.1) if len(q) else np.nan,
                                 q50=q.median() if len(q) else np.nan,
                                 q90=q.quantile(.9) if len(q) else np.nan))
    pd.DataFrame(rows).to_csv(od / 'intra_a_sep_angle.csv', index=False)
    # a thinned scatter for the figure
    g = A[A.both_slope & A.pointing]
    g.groupby('mixed', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 1500), random_state=3))[
        ['mixed', 'open_deg', 'sep_plane_mm']].to_parquet(od / 'intra_a_scatter.parquet')

    # IPC prior through the point-source toy
    T = ipc_toy()
    tot = T.groupby('kind').total_weight.max()
    T = T[T.weight > 0]
    edges = np.array([0, 3, 6, 12, 24, 50, 100, 200, 520])
    rows = []
    for kind, g in T.groupby('kind'):
        w = g.weight.to_numpy()
        h, _ = np.histogram(g.sep, edges, weights=w)
        for i in range(len(edges) - 1):
            rows.append(dict(kind=kind, lo=edges[i], hi=edges[i + 1],
                             frac_of_intra=float(h[i] / w.sum()),
                             intra_of_all=float(w.sum() / tot[kind])))
    pd.DataFrame(rows).to_csv(od / 'ipc_intra_sep.csv', index=False)
    fine = np.linspace(0, 120, 61)
    rows = []
    for kind, g in T.groupby('kind'):
        h, _ = np.histogram(g.sep, fine, weights=g.weight)
        for lo, hi, v in zip(fine[:-1], fine[1:], h / g.weight.sum()):
            rows.append(dict(kind=kind, lo=lo, hi=hi, frac=v))
    pd.DataFrame(rows).to_csv(od / 'ipc_intra_sep_fine.csv', index=False)
    print(pd.read_csv(od / 'ipc_intra_sep.csv').round(4).to_string())
    (od / 'data.meta.json').write_text(json.dumps(dict(
        written=time.strftime('%Y-%m-%d %H:%M'), L_strip=L_STRIP, gap_mm=GAP_MM), indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['asimov', 'oracle', 'data', 'table'])
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--seed', type=int, default=50_000)
    a = ap.parse_args()
    if a.cmd == 'asimov':
        asimov(a.jobs)
    elif a.cmd == 'oracle':
        oracle(a.n, a.jobs, a.seed)
    elif a.cmd == 'data':
        data()
    else:
        print(oracle_table(pd.read_parquet(out_dir() / 'r5_oracle.parquet')).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
