#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
z_image.py -- where the z of the pair image went, and how much of it comes back.

`vertex_image` found that perpendicular pairs image the capsule in x and not in
z.  In a perpendicular crossing x is the A or C leg and z is the D leg, so that
result is a statement about chamber D.  This module asks two follow-up
questions and measures both.

1. IS IT D, AND CAN D BE CLEANED?  Two populations in the gated track sample
   carry no pointing and were never cut:

   * **unmeasured-slope tracks.**  ``x_slope_reliable`` is `wft`'s
     ``|tan| >= TAN_MIN_SLOPE = 0.08`` -- "below this the timing carries no
     slope information" (`wft/reco.py`).  Below it the fit piles tracks up at
     tan ~ 0 (median |tan| 0.002-0.003): a horizontal line across every
     chamber's (x_local, tan) band plot, 22 % of A's gated tracks on run_145,
     4 % of C's, 23 % of D's.  The flag is recorded and gates nothing
     (`sept26_prelim_analysis/STATUS.md`, 2026-09-08).  Requiring it removes
     that pile-up AND the genuinely near-normal tracks, i.e. a band of about
     +-19 mm (0.08 x 234.6 mm, before k) around each chamber's foot point; the
     null is shuffled within the same flag, so it takes the same cut.
   * **noisy readout columns.**  Whole x columns that make "tracks" at every
     slope -- vertical stripes in the band plot, on connector boundaries.
     Found here PER RUN from the data: 2 mm columns whose occupancy exceeds 4x
     the running median over 30 mm.  A median of a third of D's tracks per run
     sit in one; 4 % of A's; none of C's.

   And a third tier: the track's own arm fired its scintillator coincidence
   (``coinc_this_arm``), which selects particles that crossed the chamber
   toward its own wall.

   Tiers are cumulative: ``all`` > ``reliable`` > ``clean`` (reliable and not
   in a hot column) > ``confirmed`` (clean and scintillator-confirmed).

   PER-LEG TIERS FOR PERPENDICULAR PAIRS.  The production trigger lights one
   arm, so a pair confirmed on BOTH legs (A and D both fired) is rare -- a few
   hundred in the campaign.  The question is about D, so the D leg's tier is
   varied and the A/C leg's is set separately (``tier`` / ``tier_other``).

2. CAN A AND C GIVE z?  A single A or C track runs along z: its drift depth is
   what gives it an ANGLE, and so its x at the capsule, but not WHERE along z
   the vertex sits.  That needs a second, non-parallel track.  Without D the
   candidates are other A and C tracks.  Intra A-A and C-C pairs turn out to
   carry no z; opposing A-C pairs carry some.  For an A-C crossing

       z = [x_A - x_C - L (t_A + t_C)] / (t_C - t_A),   L = 234.6 mm

   so its z is set by the DIFFERENCE of the two chambers' x and slopes: a
   relative x offset of A against C moves every A-C z by offset / (t_C - t_A),
   several mm per mm.  A and C's single-track band crossings already disagree
   by 2.0 mm in x (`imaging_campaign`), so ``--align-ac`` rebuilds with each
   chamber shifted onto their common x and the A-C z is re-measured.

   D-D PAIRS ARE THE MIRROR CASE: two D lines run along x, so their crossing
   fixes z well and x poorly.  Kept as their own selection so they can never
   again be pooled into an "intra" z that looks like A and C's.

THE NULL IS SHUFFLED WITHIN A STRATUM, not just within a chamber: a track's
direction is replaced by that of another track of the same chamber, the same
run AND the same (slope-reliable, hot-column, confirmed) flags, so the null for
every tier is the same selection as the data.

    python -m pair_vertex_imaging.z_image --jobs 8              # build + measure
    python -m pair_vertex_imaging.z_image --derive-only
    python -m pair_vertex_imaging.z_image --jobs 8 --align-ac   # A/C aligned
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from sept26_prelim_analysis import paths  # noqa: E402
from pair_vertex_imaging import vertex_lab as VL  # noqa: E402
from pair_vertex_imaging import vertex_image as VI  # noqa: E402

SCHEMA = 'athens26/pair_vertex_z/2'
ARMS = VI.ARMS_IMG
BUILD_CEIL_MM = VI.BUILD_CEIL_MM
N_SHUFFLE = VI.N_SHUFFLE
LEVER_MM = VL.D_PERP_MM

TRACK_COLS = VI.TRACK_COLS + ['x_local', 'x_slope_reliable', 'coinc_this_arm']

#: Hot-column finder.  2 mm is ~2.6 strips; the 15-bin (30 mm) running median
#: is wide against one noisy connector column and narrow against the pointing
#: band's lobes (~100 mm), so the band itself is never called hot.
HOT_BIN_MM = 2.0
HOT_WINDOW_BINS = 15
HOT_FACTOR = 4.0
HOT_EDGES = np.arange(-200.0, 200.0 + HOT_BIN_MM, HOT_BIN_MM)

#: Flag bits of a track's stratum code.
REL, HOT, CONF = 1, 2, 4
TIERS = ('all', 'reliable', 'clean', 'confirmed')

#: Single-track miss-distance histogram edges [mm] (as `diagnostics`).
DCA_EDGES = np.r_[np.arange(0, 120, 2.0), np.arange(120, 320, 10.0)]
BAND_RANGE = ((-200.0, 200.0), (-0.8, 0.8))
BAND_BINS = (200, 160)

PAIR_F32 = ('e1', 'e2', 'vx', 'vy', 'vz', 'sep_mm', 'vx_xz', 'vz_xz', 'vy_xz',
            'dy_cross', 'sin_psi_xz', 's_xz', 't_xz', 'open_deg')
SIN_PSI_BINS = (0.0, 0.3, 0.5, 0.7)
PAR_SELECTIONS = ('A-A', 'C-C', 'A-C', 'D-D')
PERP_SELECTIONS = ('A-D', 'C-D', 'perpendicular')

#: A fitted capsule term below this is "none wanted": centre and blur are then
#: undefined and are blanked rather than printed as if they were measured.
F_NONE = 0.005


def tier_ok(code: np.ndarray, tier: str) -> np.ndarray:
    code = np.asarray(code)
    rel = (code & REL) > 0
    hot = (code & HOT) > 0
    conf = (code & CONF) > 0
    if tier == 'all':
        return np.ones(code.shape, bool)
    if tier == 'reliable':
        return rel
    if tier == 'clean':
        return rel & ~hot
    if tier == 'confirmed':
        return rel & ~hot & conf
    raise ValueError(tier)


def hot_columns(xl: np.ndarray, arms: np.ndarray, run: str):
    """Per-track hot-column flag, and the table of hot columns found."""
    from scipy.ndimage import median_filter
    flag = np.zeros(len(xl), bool)
    rows = []
    for arm in ARMS:
        m = (arms == arm) & np.isfinite(xl)
        if m.sum() < 500:
            continue
        h, _ = np.histogram(xl[m], HOT_EDGES)
        med = median_filter(h.astype(float), size=HOT_WINDOW_BINS, mode='nearest')
        hot = h > HOT_FACTOR * np.maximum(med, 1.0)
        idx = np.clip(np.digitize(xl[m], HOT_EDGES) - 1, 0, len(h) - 1)
        flag[m] = hot[idx]
        for i in np.flatnonzero(hot):
            rows.append(dict(run=run, arm=arm, lo=HOT_EDGES[i], hi=HOT_EDGES[i + 1],
                             n=int(h[i]), local_median=float(med[i])))
        rows.append(dict(run=run, arm=arm, lo=np.nan, hi=np.nan,
                         n=int(m.sum()), local_median=np.nan,
                         frac_tracks_hot=float(flag[m].mean())))
    return flag, pd.DataFrame(rows)


def ac_alignment_shift() -> dict:
    """Global x shifts that put A's and C's single-track crossings on their mean.

    From `imaging_campaign/per_arm.csv`.  This is a HYPOTHESIS test -- that the
    2 mm A/C disagreement is a relative placement offset -- not a correction.
    """
    ic = pd.read_csv(paths.spell('out', 'imaging_campaign', 'per_arm.csv')
                     ).set_index('arm')
    xa, xc = float(ic.loc['A', 'median_mm']), float(ic.loc['C', 'median_mm'])
    mid = 0.5 * (xa + xc)
    return {'A': mid - xa, 'C': mid - xc}


# --------------------------------------------------------------------------- #
def one_run(run: str, subruns, src: str, seed: int, shift: dict) -> tuple:
    try:
        out = []
        for sub in subruns:
            p = Path(src) / f'tracks_{run}_{sub}.parquet'
            out.append(pd.read_parquet(p, columns=TRACK_COLS).assign(subrun=sub))
        t = pd.concat(out, ignore_index=True)
        t = t[t.gated & t.angle_calibrated & t.arm.isin(ARMS)].reset_index(drop=True)
        if t.empty:
            return run, None, 'no angle-calibrated A/C/D tracks'
        t['key'] = t.subrun + ':' + t.event_id.astype(str)
        # explicit copies: pandas hands back READ-ONLY arrays here, and the
        # alignment shift below writes into P
        P = np.array(t[['p0_x', 'p0_y', 'p0_z']].to_numpy(float), copy=True)
        D = np.array(t[['d_x', 'd_y', 'd_z']].to_numpy(float), copy=True)
        arms = t.arm.to_numpy()
        xl = t.x_local.to_numpy(float)

        e0 = VI.signed_miss(P, D)
        ok = np.isfinite(e0)
        resid = float(np.max(np.abs(np.abs(e0[ok]) - t.dca_axis_mm.to_numpy(float)[ok])))
        if resid > 1e-6:
            raise AssertionError(f'signed miss != dca_axis_mm by {resid:.3g} mm')
        # the alignment hypothesis moves whole chambers in global x, AFTER the
        # identity check against the stored column
        for arm, dx in (shift or {}).items():
            P[arms == arm, 0] += dx
        e0 = VI.signed_miss(P, D)

        rel = t.x_slope_reliable.fillna(False).astype(bool).to_numpy()
        conf = pd.to_numeric(t.coinc_this_arm, errors='coerce').fillna(-1).to_numpy() == 1
        hot, hot_tab = hot_columns(xl, arms, run)
        code = rel * REL + hot * HOT + conf * CONF

        TX = np.full(len(t), np.nan)
        TY = np.full(len(t), np.nan)
        SG = np.zeros(len(t))
        for arm in ARMS:
            m = arms == arm
            if m.any():
                TX[m], TY[m], SG[m] = VI.tans(arm, D[m])

        rng = np.random.default_rng(seed)
        dirs = [D]
        for _ in range(N_SHUFFLE):
            Dv = np.full_like(D, np.nan)
            for arm in ARMS:
                for c in range(8):
                    idx = np.flatnonzero((arms == arm) & (code == c)
                                         & np.isfinite(TX) & np.isfinite(TY))
                    if len(idx) < 2:
                        continue
                    perm = rng.permutation(idx)
                    Dv[idx] = VI.rebuild(arm, TX[perm], TY[perm], SG[idx])
            dirs.append(Dv)

        e1 = VI.signed_miss(P, dirs[1])
        point, band = [], {}
        for arm in ARMS:
            for c in range(8):
                m = (arms == arm) & (code == c)
                if not m.any():
                    continue
                hd = np.histogram(np.abs(e0[m][np.isfinite(e0[m])]), DCA_EDGES)[0]
                hn = np.histogram(np.abs(e1[m][np.isfinite(e1[m])]), DCA_EDGES)[0]
                point.append(dict(arm=arm, code=c, data=hd, null=hn))
                band[(arm, c)] = np.histogram2d(xl[m], TX[m], bins=BAND_BINS,
                                                range=BAND_RANGE)[0]

        frames = []
        for v, Dv in enumerate(dirs):
            ev = VI.signed_miss(P, Dv)
            keep = np.isfinite(ev) & (np.abs(ev) < BUILD_CEIL_MM)
            pr = VL.pairs_real(t.loc[keep, ['key']])
            if pr.empty:
                continue
            i, j = pr.i.to_numpy(), pr.j.to_numpy()
            swap = arms[i] > arms[j]
            i, j = np.where(swap, j, i), np.where(swap, i, j)
            p1, p2, d1, d2 = P[i], P[j], Dv[i], Dv[j]
            V, sep, _, _ = VL.dca_3d(p1, d1, p2, d2)
            cx, cz, dy, cy, sp, sxz, txz = VL.cross_xz(p1, d1, p2, d2)
            dot = np.einsum('ij,ij->i', d1, d2).clip(-1, 1)
            frames.append(pd.DataFrame(dict(
                variant=np.int8(v), arm1=arms[i], arm2=arms[j],
                code1=code[i].astype(np.int8), code2=code[j].astype(np.int8),
                e1=ev[i], e2=ev[j],
                vx=V[:, 0], vy=V[:, 1], vz=V[:, 2], sep_mm=sep,
                vx_xz=cx, vz_xz=cz, vy_xz=cy, dy_cross=dy, sin_psi_xz=sp,
                s_xz=sxz, t_xz=txz, open_deg=np.degrees(np.arccos(dot)))))
        d = pd.concat(frames, ignore_index=True)
        for c in PAIR_F32:
            d[c] = d[c].astype(np.float32)
        d['run'] = run
        return run, dict(pairs=d, hot=hot_tab, point=point, band=band), ''
    except Exception:
        return run, None, traceback.format_exc(limit=3).strip().splitlines()[-1]


def build(src: Path, jobs: int, seed: int, shift: dict):
    rs = VL.discover(src, False)
    print(f'{len(rs)} run(s) from {src}   shift {shift or "none"}\n')
    pairs, hots, bad = [], [], {}
    P, B = {}, {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(one_run, r, s, str(src), seed + k, shift): r
                for k, (r, s) in enumerate(sorted(rs.items()))}
        for f in as_completed(futs):
            run, res, err = f.result()
            if err:
                bad[run] = err
                print(f'  {run:<10} --   {err}', flush=True)
                continue
            pairs.append(res['pairs'])
            hots.append(res['hot'])
            for r in res['point']:
                k = (r['arm'], r['code'])
                if k not in P:
                    P[k] = [np.zeros(len(DCA_EDGES) - 1, np.int64)] * 2
                P[k] = [P[k][0] + r['data'], P[k][1] + r['null']]
            for k, h in res['band'].items():
                B[k] = B.get(k, 0) + h
            print(f'  {run:<10} ok   {int((res["pairs"].variant == 0).sum()):>9,} '
                  f'data pairs', flush=True)
    d = pd.concat(pairs, ignore_index=True)
    for c in ('arm1', 'arm2', 'run'):
        d[c] = d[c].astype('category')
    H = pd.concat(hots, ignore_index=True)
    rows = []
    for (arm, c), (hd, hn) in sorted(P.items()):
        for lo, hi, a, b in zip(DCA_EDGES[:-1], DCA_EDGES[1:], hd, hn):
            rows.append(dict(arm=arm, code=c, lo=lo, hi=hi, data=int(a), null=int(b)))
    return d, H, pd.DataFrame(rows), B, bad


# --------------------------------------------------------------------------- #
def names(tag: str) -> dict:
    s = f'_{tag}' if tag else ''
    return dict(pairs=f'pairs_z{s}.parquet', meta=f'pairs_z{s}.meta.json',
                hot=f'z_hot_columns{s}.csv', ph=f'z_pointing_hist{s}.csv',
                band=f'z_band{s}.npz', point=f'z_pointing{s}.csv',
                fits=f'z_fits{s}.csv')


def load(od: Path, tag: str = '') -> pd.DataFrame:
    d = pd.read_parquet(od / names(tag)['pairs'])
    a1, a2 = d.arm1.astype(str).to_numpy(), d.arm2.astype(str).to_numpy()
    pair = pd.Series(np.char.add(np.char.add(a1, '-'), a2))
    topo = {p: VL.topology(p[0], p[2]) for p in pair.unique()}
    d['pair'] = pair.astype('category').to_numpy()
    d['topo'] = pair.map(topo).astype('category').to_numpy()
    d['null'] = d.variant > 0
    d['worst_axis'] = np.maximum(np.abs(d.e1), np.abs(d.e2))
    return d


def select(d, sel, cut, tier, smin=0.0, tier_other=None):
    """Pairs of ``sel`` within ``cut``.  ``tier`` applies to both legs, unless
    ``tier_other`` is given: then, in a pair holding a D leg, ``tier`` is the D
    leg's and ``tier_other`` the other leg's.  (Legs are ordered A < C < D, so
    in A-D and C-D the D leg is always leg 2.)"""
    g = d[(d.topo == sel) if sel in VI.CLASSES else (d.pair == sel)]
    c1, c2 = g.code1.to_numpy(), g.code2.to_numpy()
    if tier_other is None:
        ok = tier_ok(c1, tier) & tier_ok(c2, tier)
    else:
        ok = tier_ok(c2, tier) & tier_ok(c1, tier_other)
    m = (g.worst_axis < cut).to_numpy() & ok
    if smin > 0:
        m &= g.sin_psi_xz.to_numpy() >= smin
    return g[m]


def _hmedian(counts, edges):
    c = np.cumsum(counts)
    if c[-1] == 0:
        return np.nan
    k = np.searchsorted(c, 0.5 * c[-1])
    lo, hi = edges[k], edges[k + 1]
    prev = c[k - 1] if k else 0
    return float(lo + (hi - lo) * (0.5 * c[-1] - prev) / max(counts[k], 1))


def pointing_table(PH: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, g in PH.groupby('arm'):
        n_all = g.data.sum()
        for tier in TIERS:
            h = g[tier_ok(g.code.to_numpy(), tier)]
            hd = h.groupby('lo').data.sum().to_numpy()
            hn = h.groupby('lo').null.sum().to_numpy()
            lo = np.sort(h.lo.unique())
            edges = np.r_[lo, DCA_EDGES[len(lo)]]
            i30 = lo < 30
            rows.append(dict(arm=arm, tier=tier, n_tracks=int(hd.sum()),
                             frac_of_all=float(hd.sum() / max(n_all, 1)),
                             med_miss=_hmedian(hd, edges),
                             med_miss_null=_hmedian(hn, edges),
                             f30=float(hd[i30].sum() / max(hd.sum(), 1)),
                             f30_null=float(hn[i30].sum() / max(hn.sum(), 1))))
    T = pd.DataFrame(rows)
    T['excess_30'] = T.f30 - T.f30_null
    return T


class BoundedProfile(VI.ProfileModel):
    """`ProfileModel` with the blur held above half a bin and the centre inside
    the window.  Unbounded, a fit with nothing to find parks on the window edge
    or collapses to a 0.3 mm spike on one bin, and a table of those reads like
    measurements."""

    def nll(self, p):
        c, s, f = p
        if not (1.0 < s < 80 and 0 <= f <= 1 and -48 < c < 48):
            return 1e30
        m = self.mu(p)
        return float((m - self.n * np.log(m)).sum())


def _coord_row(g, coord, c0):
    col = 'vx_xz' if coord == 'x' else 'vz_xz'
    a = g.loc[~g.null, col].to_numpy(float)
    b = g.loc[g.null, col].to_numpy(float)
    base = dict(n_data=len(a), capsule=c0)
    if len(a) < 200 or len(b) < 200:
        return base
    r = VI.fit_profile(a, b, c0, model_cls=BoundedProfile)
    if r.get('f', 0) < F_NONE:
        for k in ('c', 's', 'c_err', 's_err'):
            r[k] = np.nan
    return dict(base,
                band_data=float(np.mean(np.abs(a - c0) < VI.BAND_MM)),
                band_null=float(np.mean(np.abs(b - c0) < VI.BAND_MM)),
                rsig_data=VI._rsig(a), rsig_null=VI._rsig(b),
                median_data=float(np.nanmedian(a)),
                median_null=float(np.nanmedian(b)), **r)


def fit_jobs():
    jobs = []
    for tier in TIERS:                      # the D leg
        for tier_other in ('all', 'clean'):  # the A/C leg
            for sel in PERP_SELECTIONS:
                for cut in (150.0, 60.0):
                    jobs.append(('perp', sel, tier, tier_other, cut, 0.0))
    # 150 mm too: an axis-centred leg cut drags an image toward the axis, and
    # the no-cut point is the check on whether an offset is that drag
    for tier in ('all', 'clean', 'confirmed'):
        for sel in PAR_SELECTIONS:
            for cut in (150.0, 60.0):
                for smin in SIN_PSI_BINS:
                    jobs.append(('par', sel, tier, tier, cut, smin))
    return jobs


def fits(d: pd.DataFrame, cap, jobs=None) -> pd.DataFrame:
    """Every coordinate fit, centre FREE (bounded), per job."""
    rows = []
    for kind, sel, tier, tier_other, cut, smin in (jobs or fit_jobs()):
        g = (select(d, sel, cut, tier, smin, tier_other) if kind == 'perp'
             else select(d, sel, cut, tier, smin))
        for coord, c0 in (('x', cap[0]), ('z', cap[1])):
            r = _coord_row(g, coord, c0)
            rows.append(dict(kind=kind, selection=sel, tier=tier,
                             tier_other=tier_other, cut_mm=cut,
                             sin_psi_min=smin, coord=coord, **r))
        z = rows[-1]
        if 'f' in z:
            cs = ('     none wanted     ' if not np.isfinite(z['c']) else
                  f'c={z["c"]:+6.1f}±{z["c_err"]:4.1f} s={z["s"]:5.1f}')
            print(f'   {sel:<14} D:{tier:<9} other:{tier_other:<9} cut {cut:>3.0f} '
                  f'sin>={smin:.1f} n={z["n_data"]:>7,}  z {cs} f={z["f"]:.3f} '
                  f'2dlnL={z["two_dnll_vs_none"]:6.0f} '
                  f'chi2/ndf={z["chi2"] / max(z["ndf"], 1):4.1f}  '
                  f'band {100 * (z["band_data"] - z["band_null"]):+.1f}%', flush=True)
    F = pd.DataFrame(rows)
    if 'band_data' in F:
        F['band_excess'] = F.band_data - F.band_null
        F['rsig_ratio'] = F.rsig_data / F.rsig_null
    return F


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=str(paths.spell('out', 'stage3_fullpass')))
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=29)
    ap.add_argument('--derive-only', action='store_true')
    ap.add_argument('--align-ac', action='store_true',
                    help='shift A and C onto their common single-track x and '
                         'write everything under the tag "ac_aligned"')
    a = ap.parse_args()

    od = paths.out('pair_vertex')
    cap = VI.capsule_centre()
    tag = 'ac_aligned' if a.align_ac else ''
    shift = ac_alignment_shift() if a.align_ac else {}
    N = names(tag)
    if not a.derive_only:
        src = paths.require(Path(a.src), 'the stage-3 track tables')
        d, H, PH, B, bad = build(src, a.jobs, a.seed, shift)
        d.to_parquet(od / N['pairs'], index=False)
        H.to_csv(od / N['hot'], index=False)
        PH.to_csv(od / N['ph'], index=False)
        np.savez_compressed(od / N['band'], **{f'{k[0]}_{k[1]}': v for k, v in B.items()})
        json.dump(dict(schema=SCHEMA, src=str(src), seed=a.seed, tag=tag,
                       shift_x_mm=shift, build_ceil_mm=BUILD_CEIL_MM,
                       n_shuffle=N_SHUFFLE, capsule_xz=list(cap),
                       hot_bin_mm=HOT_BIN_MM, hot_window_bins=HOT_WINDOW_BINS,
                       hot_factor=HOT_FACTOR, band_range=BAND_RANGE,
                       band_bins=BAND_BINS, f_none=F_NONE,
                       n_runs=int(d.run.nunique()),
                       n_data=int((d.variant == 0).sum()),
                       n_null=int((d.variant > 0).sum()), runs_failed=bad),
                  open(od / N['meta'], 'w'), indent=1)
        del d

    PH = pd.read_csv(od / N['ph'])
    T = pointing_table(PH)
    T.to_csv(od / N['point'], index=False)
    print('\n-- single-track pointing by tier')
    print(T.to_string(index=False, float_format=lambda x: f'{x:8.3f}'))

    d = load(od, tag)
    if not tag and (od / 'pairs_image.parquet').exists():
        r = VI.load(od)
        a_ = r[(r.variant == 0) & (r.worst_axis < 30)].groupby('pair', observed=True).size()
        b_ = d[(d.variant == 0) & (d.worst_axis < 30)].groupby('pair', observed=True).size()
        print('\nsame data sample as vertex_image at 30 mm:',
              bool((a_.sort_index().to_numpy() == b_.sort_index().to_numpy()).all()))
        del r
    print('\n-- fits')
    F = fits(d, cap)
    F.to_csv(od / N['fits'], index=False)
    print(f'\nwrote -> {od}  ({N["fits"]})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
