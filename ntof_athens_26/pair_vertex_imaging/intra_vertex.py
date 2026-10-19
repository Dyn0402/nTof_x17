#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
intra_vertex.py -- do two tracks in ONE chamber come from a common vertex?

A-A and C-C pairs cannot locate the capsule in z (`z_image`): both lines run
along z, so they cross at a shallow angle.  That is a statement about imaging a
KNOWN source.  Whether the two legs share a vertex at all is a different
question, and it needs a different null.

THE TWO NULLS, and what each one answers:

  shuffled   every leg keeps its impact point, its direction is swapped with
             another track's of the same chamber and run and passes the same
             cuts again.  No leg points anywhere.  Answers "is there any source".
  mixed      two legs from DIFFERENT triggers, each passing the same cuts, so
             each one still points at the capsule on its own.  Answers "is this
             more vertex-like than two independent capsule tracks" -- the
             common-vertex question.  (`vertex_lab.pairs_mixed`, drawn from the
             tracks that form real pairs.)

THE TESTS, for legs in a chamber whose drift axis is z (A at +z, C at -z), each
leg written as x(z) = x0 + a (z - z0), y(z) = y0 + b (z - z0):

  1. SAME POINT AT THE CAPSULE'S DEPTH.  dx and dy between the two legs at
     z = cz.  Legs from one point near that depth agree to their resolution;
     two independent capsule tracks differ by the capsule's size as well.
  2. THE TWO VIEWS AGREE ON DEPTH.  The x view puts the crossing at
     z_x = where x1(z) = x2(z); the y view at z_y.  For a real vertex these
     are the same z measured twice, independently -- the strongest test here,
     and it uses no capsule position at all.  Conditioned on both views having
     a usable angle difference.
  3. WHERE THE VERTEX IS.  z_x and the legs' mean y at z_x, real against
     mixed.

CLONES FIRST.  One particle reconstructed twice -- two tracks sharing an x or
a y cluster -- would pass every test above perfectly.  A pair whose legs have
identical strip-plane fit positions in either view is flagged and removed
before anything is measured, and the fraction is reported.

LEGS: gated, angle-calibrated, slope measured and not in a noisy column in
BOTH views (`z_image`, `y_image` flags), transverse miss from the measured
capsule position < 30 mm.  The capsule position is the single-track band
crossing; nothing assumes the capsule is at the frame origin.

    python -m pair_vertex_imaging.intra_vertex --jobs 8
    python -m pair_vertex_imaging.intra_vertex --derive-only
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
from pair_vertex_imaging import z_image as Z  # noqa: E402

SCHEMA = 'athens26/intra_vertex/1'
ARMS = ('A', 'C')
POINT_MM = 30.0
CLONE_MM = 1e-3
#: Both views need this much slope difference for their depths to mean anything.
DT_MIN = 0.15
COLS = ['event_id', 'arm', 'gated', 'angle_calibrated', 'p0_x', 'p0_y', 'p0_z',
        'd_x', 'd_y', 'd_z', 'x_local', 'y_local', 'x_slope_reliable',
        'y_slope_reliable', 'x_p0', 'y_p0', 'n_cand_x', 'n_cand_y', 'coinc_this_arm']


def pair_metrics(P, D, xp0, yp0, i, j, cap, kind, arm, run):
    a = D[:, 0] / D[:, 2]
    b = D[:, 1] / D[:, 2]
    x0, y0, z0 = P[:, 0], P[:, 1], P[:, 2]
    cx, cz = cap

    def at(k, z):
        return x0[k] + a[k] * (z - z0[k]), y0[k] + b[k] * (z - z0[k])

    x1c, y1c = at(i, cz)
    x2c, y2c = at(j, cz)
    da, db = a[i] - a[j], b[i] - b[j]
    with np.errstate(divide='ignore', invalid='ignore'):
        zx = (x0[j] - x0[i] + a[i] * z0[i] - a[j] * z0[j]) / da
        zy = (y0[j] - y0[i] + b[i] * z0[i] - b[j] * z0[j]) / db
    x1x, y1x = at(i, zx)
    x2x, y2x = at(j, zx)
    V, sep, _, _ = VL.dca_3d(P[i], D[i], P[j], D[j])
    dot = np.einsum('ij,ij->i', D[i], D[j]).clip(-1, 1)
    return pd.DataFrame(dict(
        kind=kind, arm=arm, run=run,
        clone=(np.abs(xp0[i] - xp0[j]) < CLONE_MM) | (np.abs(yp0[i] - yp0[j]) < CLONE_MM),
        dx_c=(x1c - x2c).astype(np.float32), dy_c=(y1c - y2c).astype(np.float32),
        xm_c=(0.5 * (x1c + x2c) - cx).astype(np.float32),
        ym_c=(0.5 * (y1c + y2c)).astype(np.float32),
        da=da.astype(np.float32), db=db.astype(np.float32),
        zx=zx.astype(np.float32), zy=zy.astype(np.float32),
        xv=(0.5 * (x1x + x2x)).astype(np.float32), yv=(0.5 * (y1x + y2x)).astype(np.float32),
        dyv=(y1x - y2x).astype(np.float32),
        sep=sep.astype(np.float32), vz3=V[:, 2].astype(np.float32),
        vy3=V[:, 1].astype(np.float32),
        open_deg=np.degrees(np.arccos(dot)).astype(np.float32)))


def one_run(run, subruns, src, cap, seed):
    try:
        out = []
        for sub in subruns:
            out.append(pd.read_parquet(Path(src) / f'tracks_{run}_{sub}.parquet',
                                       columns=COLS).assign(subrun=sub))
        t = pd.concat(out, ignore_index=True)
        t = t[t.gated & t.angle_calibrated & t.arm.isin(ARMS)].reset_index(drop=True)
        if t.empty:
            return run, None, 'no angle-calibrated A/C tracks'
        t['key'] = t.subrun + ':' + t.event_id.astype(str)
        arms = t.arm.to_numpy()
        P = np.array(t[['p0_x', 'p0_y', 'p0_z']].to_numpy(float), copy=True)
        D = np.array(t[['d_x', 'd_y', 'd_z']].to_numpy(float), copy=True)
        xh, _ = Z.hot_columns(t.x_local.to_numpy(float), arms, run)
        yh, _ = Z.hot_columns(t.y_local.to_numpy(float), arms, run)
        quality = (t.x_slope_reliable.fillna(False).to_numpy(bool) & ~xh
                   & t.y_slope_reliable.fillna(False).to_numpy(bool) & ~yh)
        xp0 = t.x_p0.to_numpy(float)
        yp0 = t.y_p0.to_numpy(float)

        TX = np.full(len(t), np.nan)
        TY = np.full(len(t), np.nan)
        SG = np.zeros(len(t))
        for arm in ARMS:
            m = arms == arm
            if m.any():
                TX[m], TY[m], SG[m] = VI.tans(arm, D[m])
        rng = np.random.default_rng(seed)
        Dsh = np.full_like(D, np.nan)
        # the shuffle swaps BOTH tans together, with one permutation per chamber
        for arm in ARMS:
            idx = np.flatnonzero((arms == arm) & quality & np.isfinite(TX) & np.isfinite(TY))
            if len(idx) > 1:
                perm = rng.permutation(idx)
                Dsh[idx] = VI.rebuild(arm, TX[perm], TY[perm], SG[idx])

        frames = []
        n_counts = {}
        for kind, Dk in (('real', D), ('shuffled', Dsh)):
            ec = VI.signed_miss(P, Dk, cap)
            sel = quality & np.isfinite(ec) & (np.abs(ec) < POINT_MM)
            ts = t.loc[sel, ['key', 'arm']].reset_index()          # 'index' -> row in t
            pr = VL.pairs_real(ts)
            if pr.empty:
                continue
            same = ts.arm.to_numpy()[pr.i.to_numpy()] == ts.arm.to_numpy()[pr.j.to_numpy()]
            pr = pr[same].reset_index(drop=True)
            jobs = [(kind, pr)]
            if kind == 'real':
                mix = VL.pairs_mixed(ts, pr, seed + 1)
                am = ts.arm.to_numpy()
                mix = mix[am[mix.i.to_numpy()] == am[mix.j.to_numpy()]].reset_index(drop=True)
                jobs.append(('mixed', mix))
            for kk, pp in jobs:
                if pp.empty:
                    continue
                ii = ts['index'].to_numpy()[pp.i.to_numpy()]
                jj = ts['index'].to_numpy()[pp.j.to_numpy()]
                for arm in ARMS:
                    m = arms[ii] == arm
                    if m.any():
                        frames.append(pair_metrics(P, Dk, xp0, yp0, ii[m], jj[m], cap,
                                                   kk, arm, run))
                        n_counts[(kk, arm)] = int(m.sum())
        if not frames:
            return run, None, 'no intra pairs'
        return run, pd.concat(frames, ignore_index=True), ''
    except Exception:
        return run, None, traceback.format_exc(limit=3).strip().splitlines()[-1]


def build(src, jobs, seed, cap):
    rs = VL.discover(src, False)
    out, bad = [], {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(one_run, r, s, str(src), cap, seed + k): r
                for k, (r, s) in enumerate(sorted(rs.items()))}
        for f in as_completed(futs):
            run, d, err = f.result()
            if err:
                bad[run] = err
                print(f'  {run:<10} --   {err}', flush=True)
                continue
            out.append(d)
            c = d.groupby('kind').size().to_dict()
            print(f'  {run:<10} ok   {c}', flush=True)
    d = pd.concat(out, ignore_index=True)
    for c in ('kind', 'arm', 'run'):
        d[c] = d[c].astype('category')
    return d, bad


# --------------------------------------------------------------------------- #
def rsig(v):
    return VI._rsig(np.asarray(v, float))


def summary(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, kind), g in d.groupby(['arm', 'kind'], observed=True):
        n_all = len(g)
        h = g[~g.clone]
        well = h[(np.abs(h.da) > DT_MIN) & (np.abs(h.db) > DT_MIN)]
        dz = (well.zx - well.zy).to_numpy(float)
        ok = np.isfinite(dz)
        from scipy.stats import spearmanr
        rho = spearmanr(well.zx[ok], well.zy[ok]).correlation if ok.sum() > 20 else np.nan
        rows.append(dict(
            arm=arm, kind=kind, n_pairs=n_all, clone_frac=float(g.clone.mean()),
            n_noclone=len(h),
            rsig_dx_c=rsig(h.dx_c), rsig_dy_c=rsig(h.dy_c),
            f_dx5=float(np.mean(np.abs(h.dx_c) < 5)), f_dy15=float(np.mean(np.abs(h.dy_c) < 15)),
            f_both=float(np.mean((np.abs(h.dx_c) < 5) & (np.abs(h.dy_c) < 15))),
            n_well=int(ok.sum()), med_abs_zx_minus_zy=float(np.nanmedian(np.abs(dz))),
            f_zagree30=float(np.mean(np.abs(dz[ok]) < 30)) if ok.any() else np.nan,
            spearman_zx_zy=float(rho), med_open=float(np.nanmedian(h.open_deg)),
            med_sep=float(np.nanmedian(h.sep))))
    S = pd.DataFrame(rows)
    # lifts, real over each null
    for col in ('f_dx5', 'f_dy15', 'f_both', 'f_zagree30'):
        for null in ('mixed', 'shuffled'):
            ref = S[S.kind == null].set_index('arm')[col]
            S[f'lift_{col}_vs_{null}'] = [r[col] / ref.get(r.arm, np.nan) if r.kind == 'real'
                                          else np.nan for _, r in S.iterrows()]
    return S


def figures(d: pd.DataFrame, S: pd.DataFrame, cap):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    mp = os.path.join(REPO, 'mpgd26')      # plotstyle lives there, not in the package
    if mp not in sys.path:
        sys.path.insert(0, mp)
    import plotstyle as P
    from pair_vertex_imaging.make_image_figures import SEQ, foot, save
    P.use()
    KIND = {'real': ('#1b2430', '-', 2.0), 'mixed': ('#0072B2', (0, (4, 2.5)), 1.5),
            'shuffled': ('#9aa3ad', (0, (1, 1.5)), 1.5)}
    FOOT = ('ntof_athens_26/pair_vertex_imaging  |  33 runs of the condor full pass; '
            'both legs slope-measured and not noisy in x and y, pointing within 30 mm of '
            'the measured capsule, clones removed')
    h = d[~d.clone]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6))
    for i, arm in enumerate(ARMS):
        for j, (col, edges, lab) in enumerate((
                ('dx_c', np.linspace(-60, 60, 61), 'Δx of the legs at the capsule’s depth  [mm]'),
                ('dy_c', np.linspace(-200, 200, 81), 'Δy of the legs at the capsule’s depth  [mm]'),
                ('sep', np.geomspace(0.3, 300, 50), '3D closest approach of the two lines  [mm]'))):
            ax = axes[i, j]
            for kind, (c, ls, lw) in KIND.items():
                v = h[(h.arm == arm) & (h.kind == kind)][col].to_numpy(float)
                v = v[np.isfinite(v)]
                if len(v) < 30:
                    continue
                y = np.histogram(v, edges)[0] / len(v)
                ax.stairs(y, edges, color=c, ls=ls, lw=lw,
                          label=f'{kind}  n={len(v):,}  robust σ {rsig(v):.1f}')
            if col == 'sep':
                ax.set_xscale('log')
            ax.set_ylim(bottom=0)
            ax.set_xlabel(lab, fontsize=10)
            ax.set_title(f'{arm}–{arm}', loc='left', fontsize=12.5, fontweight='bold', color=P.INK)
            ax.legend(fontsize=8, loc='upper left' if col != 'dx_c' else 'upper right')
            P.strip(ax)
        axes[i, 0].set_ylabel('fraction of pairs per bin')
    fig.suptitle('Test 1: do the two legs meet at the capsule’s depth more than independent '
                 'capsule tracks do?', x=0.01, ha='left', fontsize=14, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    foot(fig, 'real: both legs in one trigger.  mixed: legs from different triggers, each '
         'pointing at the capsule on its own.  shuffled: directions swapped among tracks of '
         'the chamber, no leg points anywhere.   ' + FOOT)
    save(fig, 'intra_test1')

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.2))
    for i, arm in enumerate(ARMS):
        for j, kind in enumerate(('real', 'mixed')):
            g = h[(h.arm == arm) & (h.kind == kind)]
            g = g[(np.abs(g.da) > DT_MIN) & (np.abs(g.db) > DT_MIN)]
            ax = axes[i, j]
            H, xe, ye = np.histogram2d(g.zx, g.zy, bins=80, range=((-400, 400), (-400, 400)))
            ax.imshow(H.T + 0.5, origin='lower', extent=(xe[0], xe[-1], ye[0], ye[-1]),
                      cmap=SEQ, norm=LogNorm(vmin=0.5, vmax=max(H.max(), 2)), aspect='equal')
            ax.plot([-400, 400], [-400, 400], color=P.COPPER, lw=0.8)
            srow = S[(S.arm == arm) & (S.kind == kind)].iloc[0]
            ax.set_title(f'{arm}–{arm} {kind}: {len(g):,} pairs\nSpearman ρ(z_x, z_y) = '
                         f'{srow.spearman_zx_zy:.2f}', loc='left', fontsize=10.5, color=P.INK)
            ax.set_xlabel('depth where the x views cross, z_x  [mm]')
            ax.set_ylabel('depth where the y views cross, z_y  [mm]')
            for s in ax.spines.values():
                s.set_visible(False)
        ax = axes[i, 2]
        for kind, (c, ls, lw) in KIND.items():
            g = h[(h.arm == arm) & (h.kind == kind)]
            g = g[(np.abs(g.da) > DT_MIN) & (np.abs(g.db) > DT_MIN)]
            v = (g.zx - g.zy).to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) < 30:
                continue
            e = np.linspace(-300, 300, 61)
            ax.stairs(np.histogram(v, e)[0] / len(v), e, color=c, ls=ls, lw=lw,
                      label=f'{kind}: |z_x − z_y| < 30 mm in {100 * np.mean(np.abs(v) < 30):.0f} %')
        ax.set_xlabel('z_x − z_y  [mm]')
        ax.set_ylim(bottom=0)
        ax.set_title(f'{arm}–{arm}: the two views’ depths, differenced', loc='left',
                     fontsize=10.5, color=P.INK)
        ax.legend(fontsize=8)
        P.strip(ax)
    fig.suptitle('Test 2: the x view and the y view each measure the vertex depth — do they agree?',
                 x=0.01, ha='left', fontsize=14, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    foot(fig, f'Only pairs with an angle difference above {DT_MIN} in both views, so each depth is '
         'measured.  A real common vertex puts pairs on the diagonal; independent tracks do not.  '
         'Copper: z_x = z_y.   ' + FOOT)
    save(fig, 'intra_test2')

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6))
    for i, arm in enumerate(ARMS):
        for j, (col, edges, lab) in enumerate((
                ('zx', np.linspace(-300, 300, 61), 'vertex depth from the x view, z_x  [mm]'),
                ('yv', np.linspace(-250, 250, 51), 'legs’ mean y at that depth  [mm]'))):
            ax = axes[i, j]
            for kind in ('real', 'mixed'):
                c, ls, lw = KIND[kind]
                g = h[(h.arm == arm) & (h.kind == kind)]
                g = g[(np.abs(g.da) > DT_MIN) & (np.abs(g.db) > DT_MIN)
                      & (np.abs(g.zx - g.zy) < 30)]
                v = g[col].to_numpy(float)
                v = v[np.isfinite(v)]
                if len(v) < 30:
                    continue
                ax.stairs(np.histogram(v, edges)[0] / len(v), edges, color=c, ls=ls, lw=lw,
                          label=f'{kind}  n={len(v):,}')
            if col == 'zx':
                ax.axvline(cap[1], color=P.COPPER, lw=1)
                plane = 234.6 if arm == 'A' else -234.6
                ax.axvline(plane, color=P.MUTED, lw=1, ls=':')
            ax.set_ylim(bottom=0)
            ax.set_xlabel(lab)
            ax.set_title(f'{arm}–{arm}, the two views agree within 30 mm', loc='left',
                         fontsize=11, color=P.INK)
            ax.legend(fontsize=8.5)
            P.strip(ax)
        axes[i, 0].set_ylabel('fraction of pairs per bin')
    fig.suptitle('Test 3: where the agreeing pairs’ vertices are', x=0.01, ha='left',
                 fontsize=14, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    foot(fig, 'Copper: the measured capsule z.  Dotted: the chamber’s strip plane.  Shapes '
         'normalised to unit area; the mixed curve is what independent capsule tracks that '
         'happen to agree look like.   ' + FOOT)
    save(fig, 'intra_test3')


# --------------------------------------------------------------------------- #
# why the tests above cannot see a vertex: track quality against multiplicity
# --------------------------------------------------------------------------- #
MULT_COLS = COLS + ['x_n_strips', 'y_n_strips', 'chi2dof_x', 'chi2dof_y']
#: Separation bins [mm] for the two-track study.  12 mm is on an edge on
#: purpose: it is `wft.seed.GAP_THRESHOLD_MM`, below which hits of two tracks in
#: one plane join one seed cluster and are fitted as one.
SEP_BINS = np.array([0.0, 4.0, 8.0, 12.0, 16.0, 24.0, 40.0, 80.0, 400.0])
SEED_GAP_MM = 12.0


def mult_run(run, subruns, src, cap):
    """Per selected track: its chamber's track multiplicity and its quality; per
    track in an exactly-two-track chamber: the partner's separation."""
    try:
        out = []
        for sub in subruns:
            out.append(pd.read_parquet(Path(src) / f'tracks_{run}_{sub}.parquet',
                                       columns=MULT_COLS).assign(subrun=sub))
        t = pd.concat(out, ignore_index=True)
        t = t[t.gated & t.angle_calibrated & t.arm.isin(ARMS)].reset_index(drop=True)
        if t.empty:
            return run, None, 'no angle-calibrated A/C tracks'
        t['key'] = t.subrun + ':' + t.event_id.astype(str)
        arms = t.arm.to_numpy()
        P = t[['p0_x', 'p0_y', 'p0_z']].to_numpy(float)
        D = t[['d_x', 'd_y', 'd_z']].to_numpy(float)
        xh, _ = Z.hot_columns(t.x_local.to_numpy(float), arms, run)
        yh, _ = Z.hot_columns(t.y_local.to_numpy(float), arms, run)
        quality = (t.x_slope_reliable.fillna(False).to_numpy(bool) & ~xh
                   & t.y_slope_reliable.fillna(False).to_numpy(bool) & ~yh)
        ec = VI.signed_miss(P, D, cap)
        with np.errstate(divide='ignore', invalid='ignore'):
            t['y_c'] = (P[:, 1] + D[:, 1] / D[:, 2] * (cap[1] - P[:, 2])).astype(np.float32)
            t['x_c'] = (P[:, 0] + D[:, 0] / D[:, 2] * (cap[1] - P[:, 2]) - cap[0]).astype(np.float32)
        t['sel'] = quality & np.isfinite(ec) & (np.abs(ec) < POINT_MM)
        # multiplicity counts every GATED track of the chamber in the trigger,
        # before the quality cuts -- the reconstruction saw all of them
        t['mult'] = t.groupby(['key', 'arm']).arm.transform('size').astype(np.int16)
        keep = ['arm', 'mult', 'n_cand_x', 'n_cand_y', 'x_n_strips', 'y_n_strips',
                'chi2dof_x', 'chi2dof_y', 'y_c', 'x_c']
        tracks = t.loc[t.sel, keep].reset_index(drop=True)
        two = t[t.mult == 2].sort_values(['key', 'arm'], kind='stable').reset_index(drop=True)
        a_, b_ = two.iloc[0::2].reset_index(drop=True), two.iloc[1::2].reset_index(drop=True)
        if len(a_) != len(b_) or not ((a_.key.to_numpy() == b_.key.to_numpy()).all()
                                      and (a_.arm.to_numpy() == b_.arm.to_numpy()).all()):
            raise AssertionError('two-track chambers did not pair up row by row')
        dxl = np.abs(a_.x_local.to_numpy(float) - b_.x_local.to_numpy(float)).astype(np.float32)
        dyl = np.abs(a_.y_local.to_numpy(float) - b_.y_local.to_numpy(float)).astype(np.float32)
        sep = pd.concat([pd.DataFrame(dict(
            arm=s_.arm.to_numpy(), sel=s_.sel.to_numpy(), dxl=dxl, dyl=dyl,
            y_c=s_.y_c.to_numpy(), x_c=s_.x_c.to_numpy(),
            y_n_strips=s_.y_n_strips.to_numpy(), x_n_strips=s_.x_n_strips.to_numpy()))
            for s_ in (a_, b_)], ignore_index=True)
        return run, (tracks, sep), ''
    except Exception:
        return run, None, traceback.format_exc(limit=3).strip().splitlines()[-1]


def mult_build(src, jobs, cap):
    rs = VL.discover(src, False)
    T, S, bad = [], [], {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(mult_run, r, s, str(src), cap): r for r, s in sorted(rs.items())}
        for f in as_completed(futs):
            run, res, err = f.result()
            if err:
                bad[run] = err
                print(f'  {run:<10} --   {err}', flush=True)
                continue
            T.append(res[0])
            S.append(res[1])
    return pd.concat(T, ignore_index=True), pd.concat(S, ignore_index=True), bad


def mult_summary(T: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        g = T[T.arm == arm]
        for lab, m in (('1 track in the chamber', g.mult == 1), ('2 tracks', g.mult == 2),
                       ('3 or more tracks', g.mult >= 3)):
            h = g[m]
            if len(h) < 30:
                continue
            rows.append(dict(
                arm=arm, sample=lab, n=len(h), frac=len(h) / len(g),
                rsig_y_at_capsule=rsig(h.y_c), rsig_x_at_capsule=rsig(h.x_c),
                med_y_strips=float(h.y_n_strips.median()), med_x_strips=float(h.x_n_strips.median()),
                med_chi2dof_y=float(h.chi2dof_y.median()), med_chi2dof_x=float(h.chi2dof_x.median()),
                f_ncand_y_gt1=float(np.mean(h.n_cand_y > 1)),
                f_ncand_x_gt1=float(np.mean(h.n_cand_x > 1))))
    return pd.DataFrame(rows)


def sep_summary(Sp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        g = Sp[(Sp.arm == arm) & Sp.sel]
        for view, col, val, strips in (('y', 'dyl', 'y_c', 'y_n_strips'),
                                       ('x', 'dxl', 'x_c', 'x_n_strips')):
            b = np.digitize(g[col].to_numpy(float), SEP_BINS) - 1
            for k in range(len(SEP_BINS) - 1):
                h = g[b == k]
                rows.append(dict(arm=arm, view=view, sep_lo=SEP_BINS[k], sep_hi=SEP_BINS[k + 1],
                                 n=len(h), rsig=rsig(h[val]) if len(h) >= 30 else np.nan,
                                 med_strips=float(h[strips].median()) if len(h) else np.nan))
    return pd.DataFrame(rows)


def mult_figure(T, M, SS, cap):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mp = os.path.join(REPO, 'mpgd26')
    if mp not in sys.path:
        sys.path.insert(0, mp)
    import plotstyle as P
    from pair_vertex_imaging.make_image_figures import foot, save
    P.use()
    MC = {'1 track in the chamber': '#1b2430', '2 tracks': '#0072B2', '3 or more tracks': '#E69F00'}
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))
    e = np.linspace(-250, 250, 51)
    for i, arm in enumerate(ARMS):
        g = T[T.arm == arm]
        ax = axes[i, 0]
        for lab, m in (('1 track in the chamber', g.mult == 1), ('2 tracks', g.mult == 2),
                       ('3 or more tracks', g.mult >= 3)):
            v = g.loc[m, 'y_c'].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) < 30:
                continue
            ax.stairs(np.histogram(v, e)[0] / len(v), e, color=MC[lab], lw=2 if lab.startswith('1') else 1.5,
                      label=f'{lab}: n={len(v):,}, robust σ {rsig(v):.0f} mm')
        ax.set_xlabel('track y at the capsule’s depth  [mm]')
        ax.set_ylabel('fraction of tracks per bin')
        ax.set_ylim(bottom=0)
        ax.set_title(f'chamber {arm}: y at the capsule, by tracks in the chamber', loc='left',
                     fontsize=11, color=P.INK)
        ax.legend(fontsize=8)
        P.strip(ax)
        ref = M[(M.arm == arm) & (M['sample'] == '1 track in the chamber')].iloc[0]
        for j, (metric, reflab, ylab) in enumerate((
                ('rsig', ('rsig_y_at_capsule', 'rsig_x_at_capsule'), 'robust σ at the capsule’s depth  [mm]'),
                ('med_strips', ('med_y_strips', 'med_x_strips'), 'median strips in the plane’s fit'))):
            ax = axes[i, j + 1]
            for view, ls, c, refcol in (('y', '-', '#0072B2', reflab[0]), ('x', (0, (4, 2.5)), '#CC79A7', reflab[1])):
                s = SS[(SS.arm == arm) & (SS.view == view)]
                mid = np.sqrt(np.maximum(s.sep_lo, 1.0) * s.sep_hi)
                ax.plot(mid, s[metric], ls=ls, marker='o', ms=4, color=c,
                        label=f'{view} view, against the partner’s separation in {view}')
                ax.axhline(float(ref[refcol]), color=c, lw=0.9, ls=':',
                           label=f'{view} view, single track in the chamber')
            ax.axvline(SEED_GAP_MM, color=P.COPPER, lw=1.2)
            ax.text(SEED_GAP_MM * 1.05, ax.get_ylim()[1] * 0.95, 'seed gap 12 mm', color=P.COPPER,
                    fontsize=8.5, va='top')
            ax.set_xscale('log')
            ax.set_xlabel('separation of the two tracks on the strip plane  [mm]')
            ax.set_ylabel(ylab)
            ax.set_ylim(bottom=0)
            ax.set_title(f'chamber {arm}: exactly two tracks in the chamber', loc='left',
                         fontsize=11, color=P.INK)
            ax.legend(fontsize=7.5)
            P.strip(ax)
    fig.suptitle('Why the intra-chamber tests cannot see a vertex: a second track in the chamber '
                 'breaks the first one', x=0.01, ha='left', fontsize=14, fontweight='bold', color=P.INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    foot(fig, 'Tracks gated, angle-calibrated, slope-measured and not noisy in x and y, pointing '
         'within 30 mm of the measured capsule.  Multiplicity counts every gated track of the '
         'chamber in the trigger.  Copper: wft.seed.GAP_THRESHOLD_MM, below which two tracks’ hits '
         'join one seed cluster.  ntof_athens_26/pair_vertex_imaging  |  33 runs of the condor '
         'full pass')
    save(fig, 'intra_multiplicity')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--src', default=str(paths.spell('out', 'stage3_fullpass')))
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=71)
    ap.add_argument('--derive-only', action='store_true')
    ap.add_argument('--no-figures', action='store_true')
    ap.add_argument('--multiplicity', action='store_true',
                    help='only the track-quality-against-multiplicity study')
    a = ap.parse_args()
    od = paths.out('pair_vertex')
    cap = VI.capsule_centre()
    if a.multiplicity:
        T, SP, bad = mult_build(paths.require(Path(a.src), 'stage-3 tracks'), a.jobs, cap)
        M = mult_summary(T)
        SS = sep_summary(SP)
        M.to_csv(od / 'intra_multiplicity.csv', index=False)
        SS.to_csv(od / 'intra_twotrack_separation.csv', index=False)
        json.dump(dict(schema=SCHEMA + '/multiplicity', capsule_xz=list(cap),
                       point_mm=POINT_MM, sep_bins=SEP_BINS.tolist(), seed_gap_mm=SEED_GAP_MM,
                       n_tracks=int(len(T)), n_twotrack_tracks=int(len(SP)), runs_failed=bad),
                  open(od / 'intra_multiplicity.meta.json', 'w'), indent=1)
        pd.set_option('display.width', 260)
        print(M.to_string(index=False, float_format=lambda x: f'{x:8.3f}'))
        print()
        print(SS.to_string(index=False, float_format=lambda x: f'{x:8.2f}'))
        if not a.no_figures:
            mult_figure(T, M, SS, cap)
        return 0
    if not a.derive_only:
        d, bad = build(paths.require(Path(a.src), 'stage-3 tracks'), a.jobs, a.seed, cap)
        d.to_parquet(od / 'pairs_intra.parquet', index=False)
        json.dump(dict(schema=SCHEMA, capsule_xz=list(cap), point_mm=POINT_MM,
                       dt_min=DT_MIN, runs_failed=bad, n=int(len(d))),
                  open(od / 'pairs_intra.meta.json', 'w'), indent=1)
    d = pd.read_parquet(od / 'pairs_intra.parquet')
    S = summary(d)
    S.to_csv(od / 'intra_summary.csv', index=False)
    pd.set_option('display.width', 260)
    print(S.to_string(index=False, float_format=lambda x: f'{x:8.3f}'))
    if not a.no_figures:
        figures(d, S, cap)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
