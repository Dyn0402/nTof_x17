#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chamber_b.py -- chamber B on its own chain, as the hit detector it is.

WHY B IS DIFFERENT.  B has no field-shaping ring chain, so its drift field is
not uniform, there is no clean time-to-depth ladder, and it will never carry an
angle.  Everywhere else in this analysis B's angle columns are null and its
efficiency is quoted on hits.  But **position needs only the amplification
stage**, and B's amplification is fine -- so B is a position-and-timing
detector, and it deserves to be characterised as one rather than as a failed
tracker.

THE MEASUREMENT USES NO ANGLE ANYWHERE.  A particle from the target crossing
the strip plane at in-plane position ``u_c`` continues in a straight line to
the scintillator wall 95.9 mm further out, arriving at

    u_w = foot + LEVER * (u_c - foot),      LEVER = (234.6 + 95.9) / 234.6

which involves the source, the geometry and the cluster position -- and no
drift velocity, no angle scale, no bundle.  So it works identically in all four
chambers, and B can be put beside A, C and D on equal terms for the first time.

TWO COORDINATES, NOT ONE.  Until 2026-09-08 the wall could only say *which of
four segments* fired, so this test existed in u alone at 100 mm granularity.
The wall is also read at both ends of every bar, and that gives position ALONG
the bar -- the vertical coordinate v (`scintillators.py`).  That estimator
works in B: its two halves agree at r = -0.72 with no Micromegas involved.  So
B is characterised here in **both** in-plane coordinates.

THE ONE TRANSFER, AND IT IS LABELLED.  Converting B's along-bar estimator into
millimetres needs a slope, and a slope has to be fitted against tracks -- which
B does not have.  The slope is therefore transferred from the mean of A and C
(same bars, same readout), and every v number for B is reported as resting on
that.  The dimensionless correlation, which needs no transfer, is reported
beside it so the two can be told apart.

    python -m sept26_prelim_analysis.chamber_b --run run_145
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from sept26_prelim_analysis import paths  # noqa: E402

SCHEMA = 'sept26_prelim/chamber_b/1'
ARMS = ('A', 'B', 'C', 'D')
D_PERP_MM = 234.6
#: strip plane -> scintillator wall, mm.  geometry: SIPM_SCINT_W0 - W_STRIP.
WALL_DEPTH_MM = 126.0 - 30.1
LEVER = (D_PERP_MM + WALL_DEPTH_MM) / D_PERP_MM


def wall_group_u(run: str) -> dict:
    """{arm: {group: u centre}} -- from the DAQ's own bar placements."""
    from ntof_tracking.reco import geometry as G
    cfg = json.loads((paths.root('runs') / run / 'run_config.json').read_text())
    trs = G.detector_transforms(cfg)
    out = {}
    for arm in ARMS:
        uh, c = G.U_HAT[arm], trs[f'mx17_{arm}'].center
        bars = {}
        for d in cfg['detectors']:
            if not d['name'].startswith(f'sipm_{arm}_'):
                continue
            p = np.array([d['det_center_coords'][k] for k in 'xyz'], float)
            bars[int(d['name'].split('_')[-1])] = float(p @ uh - c @ uh)
        g = {}
        for bar, u in bars.items():
            g.setdefault((bar - 1) // 4, []).append(u)
        out[arm] = {k: float(np.mean(v)) for k, v in g.items()}
    return out


def clusters(run: str, subruns) -> pd.DataFrame:
    """Every gated cluster's in-plane position -- no angle columns touched."""
    from ntof_tracking import run145_target_imaging as TI
    from sept26_prelim_analysis.build_tracks import IN_PLANE_SIGN_Y
    src = paths.out('stage3_fullpass')
    out = []
    for sub in subruns:
        p = paths.require(src / f'tracks_{run}_{sub}.parquet',
                          f'stage-3 tracks for {sub}')
        d = pd.read_parquet(p, columns=['event_id', 'arm', 'gated',
                                        'x_p0', 'y_p0', 'x_q_sum', 'y_q_sum',
                                        'x_n_strips', 'y_n_strips', 'x_t0'])
        out.append(d.assign(subrun=sub))
    t = pd.concat(out, ignore_index=True)
    t = t[t.gated].copy()
    t['u'] = TI.local_x(t.x_p0.to_numpy())
    t['v'] = IN_PLANE_SIGN_Y * (t.y_p0.to_numpy() - TI.STRIP_MAP_HALF)
    return t.rename(columns={'event_id': 'eventId'})


def wall_hits(run: str, subruns) -> tuple:
    """Fired wall groups per event, and the along-bar estimator for each."""
    from sept26_prelim_analysis import scintillators as SC
    slim = SC.read_slim(run, subruns)
    pairs = SC.wall_pairs(slim)
    pairs = pairs[pairs.physical].copy()
    # single-group events only: with two groups lit there is no unambiguous
    # prediction to test, and keeping them would blur the residual with a
    # combinatorial choice we have no way to make.
    n = pairs.groupby(['subrun', 'eventId', 'arm']).grp.transform('size')
    return pairs[n == 1].copy(), pairs


def cluster_shape(cl: pd.DataFrame) -> pd.DataFrame:
    """Width and charge density of every gated cluster -- no wall needed.

    The most direct statement about B there is.  Without a field-shaping ring
    chain the drift field fringes, so the charge arriving at the mesh is spread
    over more strips than a uniform field would put it on; the total charge is
    unchanged, so the signature is a WIDE, LOW cluster rather than a weak one.
    That is a prediction of the hardware fault, and it is testable against the
    three chambers that do have ring chains.
    """
    rows = []
    for arm, g in cl.groupby('arm'):
        rows.append(dict(
            arm=arm, n=int(len(g)),
            width_x=float(g.x_n_strips.median()),
            width_y=float(g.y_n_strips.median()),
            q_x=float(g.x_q_sum.median()), q_y=float(g.y_q_sum.median()),
            q_per_strip_x=float((g.x_q_sum / g.x_n_strips).median()),
            q_per_strip_y=float((g.y_q_sum / g.y_n_strips).median())))
    R = pd.DataFrame(rows)
    ref = R[R.arm == 'A']
    if len(ref):
        R['width_x_vs_A'] = R.width_x / float(ref.width_x.iloc[0])
        R['q_per_strip_x_vs_A'] = (R.q_per_strip_x
                                   / float(ref.q_per_strip_x.iloc[0]))
    return R


def characterise(run: str, subruns) -> tuple:
    """Per arm: what the cluster position predicts about the wall."""
    from sept26_prelim_analysis import scintillators as SC
    from ntof_tracking import run145_target_imaging as TI

    gu = wall_group_u(run)
    cl = clusters(run, subruns)
    single, allp = wall_hits(run, subruns)
    cal = pd.read_csv(paths.require(
        paths.out('scint') / f'wall_calibration_{run}.csv',
        'the wall calibration -- run scintillators.py first'))

    # the v scale, transferred: B has no tracks to fit a slope against
    good = cal.dropna(subset=['lr_slope'])
    slope_ac = float(good[good.arm.isin(['A', 'C'])].lr_slope.mean())

    m = cl.merge(single, on=['subrun', 'eventId', 'arm'], how='inner')
    rows, resid = [], []
    for arm, g in m.groupby('arm'):
        foot = TI.PINWHEEL[arm]
        u_pred = foot + LEVER * (g.u.to_numpy() - foot)
        u_meas = g.grp.map(gu[arm]).to_numpy()
        du = u_pred - u_meas
        # which group the cluster POINTS at, against which one fired
        cent = np.array([gu[arm][k] for k in sorted(gu[arm])])
        hit_pred = np.argmin(np.abs(u_pred[:, None] - cent[None, :]), axis=1)
        match = float((hit_pred == g.grp.to_numpy()).mean())
        # the shuffled control: the same predictions against a permuted truth
        rng = np.random.default_rng(7)
        sh = float((hit_pred == rng.permutation(g.grp.to_numpy())).mean())

        # v: the along-bar estimator, per-group offsets removed on this sample
        lr = g.log_ratio.to_numpy()
        off = g.groupby('grp').log_ratio.transform('median').to_numpy()
        own = cal[cal.arm == arm]
        slope = (float(own.lr_slope.iloc[0])
                 if len(own) and np.isfinite(own.lr_slope.iloc[0])
                 else slope_ac)
        transferred = not (len(own) and np.isfinite(own.lr_slope.iloc[0]))
        v_wall = (lr - off) / slope
        v_pred = LEVER * g.v.to_numpy()
        v_pred = v_pred - np.median(v_pred)
        r_v = (float(np.corrcoef(v_pred, v_wall)[0, 1])
               if len(g) > 50 else np.nan)
        dv = v_wall - v_pred

        rows.append(dict(
            arm=arm, n=int(len(g)),
            u_match=match, u_match_shuffled=sh, u_lift=match / max(sh, 1e-9),
            u_resid_mm=float(1.4826 * np.median(np.abs(du - np.median(du)))),
            u_bias_mm=float(np.median(du)),
            v_corr=r_v,
            v_resid_mm=float(1.4826 * np.median(np.abs(dv - np.median(dv)))),
            v_slope_transferred=transferred,
            n_strips_x=float(g.x_n_strips.median()),
            n_strips_y=float(g.y_n_strips.median()),
            q_x=float(g.x_q_sum.median()), q_y=float(g.y_q_sum.median())))
        resid.append(pd.DataFrame(dict(arm=arm, du=du, dv=dv,
                                       u_pred=u_pred, v_pred=v_pred,
                                       v_wall=v_wall, grp=g.grp.to_numpy())))
    return pd.DataFrame(rows), pd.concat(resid, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subruns',
                    default='stat090_0000,stat090_0001,stat090_0002')
    a = ap.parse_args()
    subs = [s for s in a.subruns.split(',') if s]

    R, resid = characterise(a.run, subs)
    SH = cluster_shape(clusters(a.run, subs))
    od = paths.out('chamber_b')
    R.to_csv(od / f'chamber_b_{a.run}.csv', index=False)
    SH.to_csv(od / f'chamber_b_shape_{a.run}.csv', index=False)
    resid.to_parquet(od / f'chamber_b_residuals_{a.run}.parquet', index=False)
    json.dump(dict(schema=SCHEMA, run=a.run, subruns=subs,
                   lever=LEVER, wall_depth_mm=WALL_DEPTH_MM,
                   note='no angle, no drift velocity and no bundle enters any '
                        'number here; B is on equal terms with A, C and D'),
              open(od / f'chamber_b_{a.run}.meta.json', 'w'), indent=1)

    print(f'lever strip plane -> wall: {LEVER:.3f}  '
          f'({WALL_DEPTH_MM:.1f} mm past the strips)\n')
    print('POSITION ACROSS THE WALL (u) -- which segment the cluster points at')
    print(R[['arm', 'n', 'u_match', 'u_match_shuffled', 'u_lift',
             'u_resid_mm', 'u_bias_mm']].round(3).to_string(index=False))
    print('\nPOSITION ALONG THE BAR (v) -- the two-ended wall estimator')
    print(R[['arm', 'n', 'v_corr', 'v_resid_mm',
             'v_slope_transferred']].round(3).to_string(index=False))
    print('\nCLUSTER SHAPE -- all gated clusters, no wall required')
    print(SH.round(2).to_string(index=False))
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
