#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run145_target_imaging.py — image the He-3 target with wft beam tracks, and use
the image as an in-situ angle-scale (v_drift) calibration.

Built on the PLAN_08 §7.2 idea (run79_merge_prelim [1]) but for a slim-matched
run: the n_TOF ↔ DREAM matching is already done by the slim pipeline, so the
per-event n_TOF record is read straight from the slim `ntof_hits` file and
joined on eventId — no clock fit, no matcher here.

Three products per arm:
  [1] target-pointing fit: median tan_theta vs u in position bins. The track
      fan from a point-ish source at distance L gives tan = (u - u_src)/L per
      plane; the fitted |slope| vs 1/L measures the angle scale (= v_drift
      scale, since positions don't depend on v), the sign fixes the in-plane
      convention, and the intercept locates the source.
  [2] the image: back-project every 2-plane track to its point of closest
      approach to the beam axis (the He-3 capsule axis, global Y); histogram
      (y, transverse offset). The capsule is ~80 mm long, r <= 10 mm.
  [3] the v-scale scan: repeat [2] with tan scaled by k on a grid; the k that
      minimises the median axis-miss distance is the in-situ angle scale.
      Reported as v_insitu = v_bundle / k_opt per arm and plane.

The slim join is used as a purity cut: keep events whose matched n_TOF record
has a wall (WAL) hit in the SAME arm (det code 0-3 = WALA-D).

Usage:
    python -m ntof_tracking.run145_target_imaging --run run_145 \
        --subrun stat090_0000 --arms A,C,B,D \
        --slim <.../ntof_hits_run_145_stat090_0000_224670.root> [--no-slim-cut]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

BEAM_BASE = os.environ.get('WFT_BEAM_BASE',
                           '/media/dylan/data/x17/beam_july/runs/')
ANALYSIS_BASE = os.environ.get('WFT_BEAM_ANALYSIS',
                               '/media/dylan/data/x17/beam_july/analysis/wft/')

STRIP_MAP_HALF = 199.29        # strip coords run 0..398.58 mm; centre = half
TAN_SANE = 1.0                 # drop railed angle fits

# IN-PLANE SIGN — MEASURED 2026-08-20, was the dominant defect in the image.
#
# geometry.py's ALIGNMENT CAVEAT says the sign of the strip coordinate within
# the plane is provisional. It is -1: the strip index runs along -u_hat, so
#     x_local = -(x_p0 - STRIP_MAP_HALF)
# and the stored wft tan needs NO flip (this analysis used to flip the tan
# instead, by the sign of the fitted pointing slope, and leave the position
# alone -- which is the same thing as MIRRORING the track about the plane
# CENTRE).
#
# It matters because the chambers are PINWHEELED: each plane centre sits
# ~16 mm off the beam axis (geometry.PINWHEEL), so a mirror about the plane
# centre displaces the reconstructed source by 2 x pinwheel, in opposite
# global directions for opposing arms. Measured on the scale-free zero
# crossing of the pointing band (where tan = 0, independent of the angle
# scale), run_145 both sub-runs, pointing-coincident tracks:
#
#     arm  measures   old convention   corrected      surveyed
#     A    global X      -21.8 mm       -10.9 mm         0
#     C    global X      +41.7 mm        -7.1 mm         0
#     B    global Z      -38.0 mm        +6.5 mm         0
#     D    global Z      +36.2 mm        -5.2 mm         0
#
# Opposing arms have to see the SAME source. Under the old convention A and C
# disagree by 64 mm = 2 x (P_A + P_C); corrected they agree to 4 mm and every
# arm lands within ~11 mm of the beam axis. The focus improves with it (arm A
# median axis-miss 34.6 -> 14.8 mm, r<10 mm fraction 18.5 % -> 31.4 %) and so
# does the external SiPM/plastic confirmation rate (arm A 50.6 % -> 53.4 %).
IN_PLANE_SIGN = -1.0


def local_x(x_p0):
    """Strip-map coordinate -> local x, in the corrected in-plane sign."""
    return IN_PLANE_SIGN * (np.asarray(x_p0, float) - STRIP_MAP_HALF)


def plane_geometry(tr):
    """(perpendicular distance from the target to the strip plane,
    local x of the foot of that perpendicular).

    NOT |centre|: the pinwheel makes the plane centre ~16 mm off the axis, so
    the lever arm is the perpendicular 234.6 mm and the point-source relation
    is tan = (x_local - foot_x) / d_perp, not tan = x_local / |centre|."""
    n = tr.R @ np.array([0.0, 0.0, 1.0])          # outward normal, global
    uhat = tr.R @ np.array([1.0, 0.0, 0.0])       # local +x, global
    return float(np.dot(tr.center, n)), -float(np.dot(tr.center, uhat))
N_POINT_BINS = 8
MIN_PER_BIN = 25
K_GRID = np.linspace(0.5, 2.0, 61)   # v-scale scan: tan' = tan * k

WAL_CODE = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


# --------------------------------------------------------------------- inputs
def load_tracks(run, subrun, arm):
    import pandas as pd
    p = os.path.join(ANALYSIS_BASE, run, subrun, f'mx17_{arm}',
                     'events_prelim.parquet')
    df = pd.read_parquet(p)
    meta = json.load(open(p.replace('.parquet', '.meta.json')))
    return df, meta


def apply_w0_kw(df, arm, v_bundle, meta=None):
    """Post-hoc per-plane angle constants (2026-08-12 finding).

    The bench bundles carry w0/kw for tan = (w*1e3 - w0)/(kw*v), but f9e18d2's
    CalibrationBundle dropped the field: the frozen reco computes tan = w*1e3/v
    and the beam bundles (load->save round trip) do not even carry the
    constants. Correct here, reading them from the BENCH bundle.json the arm
    was seeded from. tan_c = (tan_r - w0/v)/kw. Bench-measured constants
    applied at beam v: stated assumption, same as the rest of the transfer.

    POST-RESTORE GUARD (2026-08-13): the freeze was lifted and plane_fit now
    applies w0/kw in-reco, stamping `angle_constants.applied` in the output
    meta. A table carrying that stamp must NOT be corrected again here."""
    if meta and (meta.get('angle_constants', {}) or {}).get('applied'):
        return df, dict(note='in-reco (angle_constants.applied stamp); '
                             'post-hoc correction skipped')
    from ntof_tracking.wft_beam import BEAM_DETS
    bj = os.path.join(BEAM_DETS[arm]['bundle'], 'bundle.json')
    try:
        b = json.load(open(bj))
        w0, kw = b['w0'], b['kw']
    except (FileNotFoundError, KeyError):
        return df, None
    df = df.copy()
    for pl in ('x', 'y'):
        df[f'{pl}_tan_theta'] = ((df[f'{pl}_tan_theta'] - w0[pl] / v_bundle)
                                 / kw[pl])
    return df, dict(w0=w0, kw=kw)


def slim_wall_events(slim_path, arm):
    """eventIds whose matched n_TOF record has a WAL hit in this arm."""
    import uproot
    t = uproot.open(slim_path)['hits']
    a = t.arrays(['eventId', 'det', 'is_control'], library='np')
    keep = (a['det'] == WAL_CODE[arm]) & (a['is_control'] == 0)
    return np.unique(a['eventId'][keep])


# ---------------------------------------------------- pointing coincidence
# Wall / plastic geometry per run79_merge_prelim + RUN79_PRELIM §4: the wall
# is 16 read bars of 25 mm in 4 groups of 4 (100 mm), 96.4 mm past the strip
# plane; nTOF detn 1..8 = segment pairs (top/bottom), seg = (detn-1)//2.
# Plastics: two 200 mm bars at the per-arm depth, centred on the MM (the wall
# is centred on the STRUCTURE -- geometry.py) so only the wall carries the
# pinwheel term.
#
# READ-OUT ORDER (2026-08-20): ASCENDING -- detn pair 0 reads the group at
# most NEGATIVE structure-u, and plastic detn 1 is the negative-u bar. This is
# the same empirical mapping as before, restated in the corrected in-plane
# frame (IN_PLANE_SIGN): the order was measured by maximising the coincidence,
# so mirroring the frame flips the order with it. What does NOT cancel is the
# 2 x pinwheel offset the old frame carried, and removing it is why the
# confirmed fraction goes up (arm A 50.6 % -> 53.4 %).
STRIPS_TO_WALL = 96.4
PINWHEEL = {'D': 15.5, 'B': 15.75, 'A': 16.35, 'C': 17.3}
SIPM_BAR_W, SIPM_N_BARS, N_WALL_SEG = 25.0, 20, 4
PLASTIC_U_OFFSET, PLASTIC_HALF_U = 101.72, 100.0
STRIPS_TO_PLASTIC = {'A': 188.1, 'B': 186.1, 'C': 186.1, 'D': 190.1}
DT_WINDOW = (-100.0, 60.0)     # in-time window on slim dt_ns (peak ~[-30,-2])


def _wall_seg_u(g):
    bars = [g * 4 + 1 + i for i in range(4)]
    u = [SIPM_BAR_W * (b - (SIPM_N_BARS - 1) / 2.0) for b in bars]
    return min(u) - SIPM_BAR_W / 2, max(u) + SIPM_BAR_W / 2


def pointing_coincidence(slim_path, arm, df, sel, foot_x=None):
    """Boolean mask over df rows: the track extrapolates to a wall segment AND
    a plastic bar that BOTH have an in-time slim hit in this arm.

    Extrapolation in the corrected local frame: going OUTWARD (past the
    strips, toward the wall) by d, x_local moves by +tan*d, so
        u_wall(structure) = x_local + tan*STRIPS_TO_WALL - foot_x
        u_plastic(MM)     = x_local + tan*STRIPS_TO_PLASTIC
    with foot_x the local x of the perpendicular foot (= the pinwheel; the
    structure is centred on the beam axis, the plastics on the MM)."""
    import uproot
    t = uproot.open(slim_path)['hits']
    a = t.arrays(['eventId', 'det', 'detn', 'dt_ns', 'is_control'],
                 library='np')
    it = ((a['is_control'] == 0) & (a['dt_ns'] >= DT_WINDOW[0])
          & (a['dt_ns'] <= DT_WINDOW[1]))
    wal = a['det'] == WAL_CODE[arm]
    pss = a['det'] == WAL_CODE[arm] + 4
    # per-event sets of fired wall detn-pairs and plastic bars
    from collections import defaultdict
    wal_pairs, pss_bars = defaultdict(set), defaultdict(set)
    for eid, dn in zip(a['eventId'][it & wal], a['detn'][it & wal]):
        wal_pairs[int(eid)].add((int(dn) - 1) // 2)
    for eid, dn in zip(a['eventId'][it & pss], a['detn'][it & pss]):
        pss_bars[int(eid)].add(int(dn))

    xl = local_x(df['x_p0'].to_numpy())
    tan = df['x_tan_theta'].to_numpy()
    eids = df['event_id'].to_numpy()
    fx = PINWHEEL[arm] if foot_x is None else foot_x
    u_wall = xl + STRIPS_TO_WALL * tan - fx          # structure frame
    u_pl = xl + STRIPS_TO_PLASTIC[arm] * tan         # MM frame

    # geometric wall group of the predicted crossing (-1 if off the wall)
    geo_edges = [_wall_seg_u(g) for g in range(N_WALL_SEG)]
    seg_pred = np.full(len(df), -1)
    for g, (lo, hi) in enumerate(geo_edges):
        seg_pred[(u_wall >= lo) & (u_wall < hi)] = g   # ascending
    # plastic bar: detn 2 = positive u (joint with the wall order)
    bar_pred = np.where(u_pl >= 0.0, 2, 1)
    on_bar = np.abs(u_pl - np.where(bar_pred == 2, PLASTIC_U_OFFSET,
                                    -PLASTIC_U_OFFSET)) < PLASTIC_HALF_U

    coin = np.zeros(len(df), bool)
    n_predictable = 0
    for i in range(len(df)):
        if not sel[i] or seg_pred[i] < 0 or not on_bar[i]:
            continue
        n_predictable += 1
        e = int(eids[i])
        if (seg_pred[i] in wal_pairs.get(e, ())
                and int(bar_pred[i]) in pss_bars.get(e, ())):
            coin[i] = True
    return coin, dict(n_predictable=int(n_predictable),
                      n_coincident=int(coin.sum()),
                      dt_window=list(DT_WINDOW))


def transforms(run):
    from ntof_tracking.reco import geometry as G
    cfg = json.load(open(os.path.join(BEAM_BASE, run, 'run_config.json')))
    return G.detector_transforms(cfg), G


# ------------------------------------------------------------------- pointing
def pointing_fit(u, tan, n_bins=N_POINT_BINS, min_per_bin=MIN_PER_BIN):
    """Median tan per u bin, weighted LSQ line fit. Returns slope, intercept,
    per-bin table. Robust to the (dominant) non-track background."""
    qs = np.linspace(5, 95, n_bins + 1)
    edges = np.percentile(u, qs)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (u >= lo) & (u < hi)
        if m.sum() < min_per_bin:
            continue
        rows.append((0.5 * (lo + hi), np.median(tan[m]),
                     1.4826 * np.median(np.abs(tan[m] - np.median(tan[m])))
                     / max(np.sqrt(m.sum()), 1), int(m.sum())))
    if len(rows) < 3:
        return None
    b = np.array(rows)
    w = 1.0 / np.clip(b[:, 2], 1e-4, None) ** 2
    A = np.vstack([b[:, 0], np.ones(len(b))]).T
    W = np.diag(w)
    coef, *_ = np.linalg.lstsq(W @ A, W @ b[:, 1], rcond=None)
    return dict(slope=float(coef[0]), intercept=float(coef[1]),
                bins=b.tolist())


def _robust_line(u, t, n_iter=8):
    """Soft-redescending IRLS line fit — the band sits on a broad background
    the median-per-bin fit only partly suppresses."""
    A = np.vstack([u, np.ones_like(u)]).T
    w = np.ones_like(u)
    c = np.array([0.0, 0.0])
    for _ in range(n_iter):
        c, *_ = np.linalg.lstsq(A * w[:, None], t * w, rcond=None)
        r = t - (c[0] * u + c[1])
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        w = 1.0 / np.sqrt(1.0 + (r / (2.5 * max(s, 1e-4))) ** 2)
    return float(c[0]), float(c[1])


def _source_from_band(u, t, tr, d_perp, foot_x, n_boot=200, seed=1):
    """Zero crossing of the pointing band -> the source, scale-free."""
    slope, icept = _robust_line(u, t)
    x0 = -icept / slope
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(u), len(u))
        sl, ic = _robust_line(u[i], t[i])
        bs.append(-ic / sl)
    uhat = tr.R @ np.array([1.0, 0.0, 0.0])
    src = tr.center + x0 * uhat
    ax = 'X' if abs(uhat[0]) > 0.5 else 'Z'
    return dict(n=int(len(u)), slope=slope, intercept=icept,
                zero_crossing_x_local=float(x0),
                zero_crossing_err=float(np.std(bs)),
                foot_x_surveyed=float(foot_x),
                implied_k=float(1.0 / abs(slope * d_perp)),
                source_global=[float(v) for v in src],
                source_measured_axis=ax,
                source_measured_mm=float(src[0] if ax == 'X' else src[2]))


# -------------------------------------------------------------------- imaging
def track_lines(df, tr, sel):
    """Global-frame (P0, D) for each selected 2-plane track.

    Local frame per reco/geometry.py: x_l, y_l about the strip-plane centre,
    z_l = -w (drift depth inward). wft tan_theta = du/dw, so one depth unit
    inward (dz = -1) moves (tan_x, tan_y) in plane: D_local ∝ (tan_x, tan_y, -1).

    The in-plane sign is applied HERE and nowhere else (IN_PLANE_SIGN): to the
    POSITION, with the tan left as reconstructed. Flipping the tan instead --
    which is what this analysis did until 2026-08-20 -- mirrors the track about
    the plane CENTRE, and the plane centre is 16 mm off the beam axis.
    """
    xl = local_x(df['x_p0'].to_numpy()[sel])
    yl = df['y_p0'].to_numpy()[sel] - STRIP_MAP_HALF
    tx = df['x_tan_theta'].to_numpy()[sel]
    ty = df['y_tan_theta'].to_numpy()[sel]
    P0 = tr.local_to_global(xl, yl, np.zeros_like(xl))
    P1 = tr.local_to_global(xl - tx * 30.0, yl - ty * 30.0,
                            np.full_like(xl, 30.0))
    D = P1 - P0
    D /= np.linalg.norm(D, axis=-1, keepdims=True)
    return P0, D


def axis_approach(P0, D):
    """Closest approach of each line to the global Y axis (target axis).
    Returns r_min, y_at_min, xz_at_min."""
    # minimise |(P0 + s D) - (0, y, 0)| over s, y: project onto XZ plane
    p = P0[:, [0, 2]]
    d = D[:, [0, 2]]
    dd = np.einsum('ij,ij->i', d, d)
    s = -np.einsum('ij,ij->i', p, d) / np.clip(dd, 1e-12, None)
    c = P0 + s[:, None] * D
    r = np.hypot(c[:, 0], c[:, 2])
    return r, c[:, 1], c[:, [0, 2]]


def image_metrics(r, y):
    """Sharpness metrics for the v-scale scan.

    r_core = median of the sub-30 mm population: measures the width of the
    focal spot without the (k-independent-ish) far background tail that
    flattens the plain median."""
    core = r[r < 30.0]
    return dict(r_med=float(np.median(r)),
                r_core=float(np.median(core)) if len(core) else np.inf,
                n_core=int(len(core)),
                r_q25=float(np.percentile(r, 25)),
                frac_in_capsule=float(np.mean((r < 10.0)
                                              & (y > -30) & (y < 51))))


# ------------------------------------------------------------------------ main
def run_arm(run, subrun, arm, trs, G, slim_path, slim_cut, out_dir, plots=True):
    import pandas as pd  # noqa: F401
    df, meta = load_tracks(run, subrun, arm)
    tr = trs[f'mx17_{arm}']
    v_bundle = meta['bundle']['v_drift']
    df, w0kw = apply_w0_kw(df, arm, v_bundle, meta)
    res = dict(arm=arm, n_events=len(df), v_bundle=v_bundle,
               w0_kw_applied=w0kw)

    ok = (df['x_ok'] & df['y_ok']).to_numpy()
    sane = ((np.abs(df['x_tan_theta']) < TAN_SANE)
            & (np.abs(df['y_tan_theta']) < TAN_SANE)).to_numpy()
    # FULL COVERAGE (2026-08-13): the slope_reliable gate (|tan| >= 0.08) is a
    # hits-chain inheritance — the forward fit measures the head-on band
    # unbiased (<=0.15 deg at the same sigma68, JUNE_CONTINUITY §5b). At nTOF
    # the head-on band IS the image core: a track from the origin through the
    # detector centre has tan ≈ 0, so the gate cut a hole in the middle of the
    # acceptance and undercut counts ~40 %. The gate survives only where a
    # quantity divides by tan (the k estimators, via `inc` below, whose own
    # |tan| > 0.10 floor subsumes it).
    rel = df['x_slope_reliable'].to_numpy().astype(bool)
    sel = ok & sane
    res['n_2plane'] = int(ok.sum())
    res['n_sel'] = int(sel.sum())
    res['n_relonly'] = int((ok & sane & rel).sum())   # old gated basis
    res['n_headon'] = int((ok & sane & ~rel).sum())

    if slim_path and slim_cut:
        wal = slim_wall_events(slim_path, arm)
        inwal = df['event_id'].isin(wal).to_numpy()
        res['n_wall_matched'] = int((sel & inwal).sum())
        sel = sel & inwal
        # second step: track must POINT at a wall segment + plastic bar that
        # both carry an in-time hit (external per-track confirmation)
        coin, cinfo = pointing_coincidence(slim_path, arm, df, sel,
                                           foot_x=plane_geometry(tr)[1])
        res['pointing_coincidence'] = cinfo
        res['n_pointing_coincident'] = int(coin.sum())
    else:
        coin = None

    # [1] pointing fits per plane (x = tangent u; y = beam axis v)
    #
    # The lever arm is the PERPENDICULAR distance and the point-source relation
    # is tan = (x_local - foot_x)/d_perp. Using |centre| with the foot at zero
    # (what this did until 2026-08-20) throws away the pinwheel and puts a
    # spurious ~0.07 intercept in the fit. The in-plane sign is fixed, not
    # fitted, and it lives in the POSITION (see IN_PLANE_SIGN).
    d_perp, foot_x = plane_geometry(tr)
    L = float(np.linalg.norm(tr.center[[0, 2]]))     # centre distance, legacy
    res['L_strips_mm'] = L
    res['plane_geometry'] = dict(d_perp=float(d_perp), foot_x=float(foot_x),
                                 in_plane_sign=float(IN_PLANE_SIGN))
    for plane in ('x', 'y'):
        m = (df[f'{plane}_ok'].to_numpy()
             & (np.abs(df[f'{plane}_tan_theta']) < TAN_SANE))
        u = (local_x(df['x_p0'].to_numpy()[m]) if plane == 'x'
             else df['y_p0'].to_numpy()[m] - STRIP_MAP_HALF)
        fit = pointing_fit(u, df[f'{plane}_tan_theta'].to_numpy()[m])
        if fit:
            fit['expected_slope'] = 1.0 / d_perp if plane == 'x' else None
            if plane == 'x':
                # slope_reco/slope_true = tan_reco/tan_true = v_true/v_bundle
                # (tan ∝ 1/v). Biased toward 0 by tan≈0 background — the
                # per-track and image estimators below are the trustworthy
                # ones. The ZERO CROSSING, though, is scale-free and is the
                # measurement that fixed the in-plane sign.
                fit['implied_k'] = 1.0 / abs(fit['slope'] * d_perp)
                fit['v_insitu'] = v_bundle / fit['implied_k']
                x0 = -fit['intercept'] / fit['slope']
                uhat = tr.R @ np.array([1.0, 0.0, 0.0])
                src = tr.center + x0 * uhat
                fit['zero_crossing_x_local'] = float(x0)
                fit['foot_x_surveyed'] = float(foot_x)
                fit['source_global'] = [float(v) for v in src]
                # the coordinate this arm actually measures (X for A/C, Z for
                # B/D): the other two are degenerate along the arm's own line
                fit['source_measured_axis'] = 'X' if abs(uhat[0]) > 0.5 else 'Z'
                fit['source_measured_mm'] = float(src[0] if abs(uhat[0]) > 0.5
                                                  else src[2])
        res[f'pointing_{plane}'] = fit

    # [1b] per-track angle-scale estimator, inclined tracks only.
    # A ray from the origin crossing the strip plane at u has tan_true = u/L
    # exactly, so each track measures k_i = (u_i/L)/tan_reco_i. The tan≈0
    # population carries no scale information (and its closest-approach locus
    # is a k-invariant ridge along Z=0 through the capsule), so require a
    # minimum reconstructed inclination AND a minimum |u|.
    MIN_TAN, MIN_U, MAX_U = 0.10, 40.0, 130.0
    u_all = local_x(df['x_p0'].to_numpy())
    tx_all = df['x_tan_theta'].to_numpy()
    # lever arm measured from the PERPENDICULAR FOOT, not the plane centre
    lever = u_all - foot_x
    # MAX_U: beyond ~130 mm the per-slice tan mode collapses toward zero
    # (plane-edge acceptance + window truncation of the deepest columns), so
    # those slices measure the truncation, not the angle scale.
    inc = sel & (np.abs(tx_all) > MIN_TAN) & (np.abs(lever) > MIN_U) \
        & (np.abs(lever) < MAX_U)
    res['n_inclined'] = int(inc.sum())
    if inc.sum() >= 50:
        k_i = (lever[inc] / d_perp) / tx_all[inc]
        k_i = k_i[(k_i > 0) & (k_i < 5)]        # wrong-sign = not from target
        res['k_track'] = dict(
            n=int(len(k_i)), median=float(np.median(k_i)),
            mad=float(1.4826 * np.median(np.abs(k_i - np.median(k_i)))),
            v_insitu=float(v_bundle / np.median(k_i)))

    # [2] image at bundle calibration + [3] v-scale scan
    if sel.sum() >= 50:
        scan = []
        for k in K_GRID:
            d2 = df.copy()
            d2['x_tan_theta'] = df['x_tan_theta'] * k
            d2['y_tan_theta'] = df['y_tan_theta'] * k
            # scan scored on the INCLINED population only — the tan≈0 ridge
            # is k-invariant and floods every metric with a false optimum
            P0, D = track_lines(d2, tr, inc if inc.sum() >= 50 else sel)
            r, y, _ = axis_approach(P0, D)
            scan.append(dict(k=float(k), **image_metrics(r, y)))
        # per-track k on the pointing-coincident subset (with-cut estimate)
        if coin is not None and (inc & coin).sum() >= 50:
            kc = (lever[inc & coin] / d_perp) / tx_all[inc & coin]
            kc = kc[(kc > 0) & (kc < 5)]
            res['k_track_coincident'] = dict(
                n=int(len(kc)), median=float(np.median(kc)),
                v_insitu=float(v_bundle / np.median(kc)))

        # [1c] THE SCALE-FREE MEASUREMENT: where the pointing band crosses
        # tan = 0 is where the track is perpendicular to the plane, i.e. the
        # foot of the perpendicular from the source. It does not involve the
        # angle scale at all, so it is independent of k, of v_drift and of the
        # whole bench transfer. On the confirmed subset, restricted to the
        # range where the band is linear (beyond ~130 mm from the foot the
        # window truncation flattens it).
        if coin is not None and coin.sum() >= 200:
            mm = coin & (np.abs(lever) > 30) & (np.abs(lever) < 130) \
                & (np.abs(tx_all) > 1e-3)
            if mm.sum() >= 200:
                res['pointing_x_coincident'] = _source_from_band(
                    u_all[mm], tx_all[mm], tr, d_perp, foot_x)

        res['k_scan'] = scan
        best = min(scan, key=lambda s: s['r_core'])
        res['k_opt'] = best['k']
        # tan_true = tan_reco * k_opt and tan ∝ 1/v => v_true = v_bundle/k_opt
        res['v_insitu_image'] = dict(x=v_bundle / best['k'],
                                     note='from x-plane focus (r is transverse'
                                          ' to the beam axis, tan_y-blind)')
        res['image_at_kopt'] = best
        # The QUOTED image and the head-on comparison use the trustworthy k
        # (coincident per-track median; the naive scan rails — known). At a
        # railed k inclined tracks defocus while head-on tracks are
        # k-invariant, which would fake "head-on is sharper".
        k_use = (res.get('k_track_coincident') or res.get('k_track')
                 or {'median': best['k']})['median']
        res['k_phys'] = float(k_use)
        d2 = df.copy()
        d2['x_tan_theta'] = df['x_tan_theta'] * k_use
        d2['y_tan_theta'] = df['y_tan_theta'] * k_use
        tags = [('full', sel), ('relonly', sel & rel), ('headon', sel & ~rel)]
        if coin is not None:
            # the externally CONFIRMED sample — the one the deck figure draws
            # and the only one where the background is actually gone
            tags.insert(0, ('coincident', coin))
        for tag, m in tags:
            if m.sum() >= 20:
                P0, D = track_lines(d2, tr, m)
                r, y, _ = axis_approach(P0, D)
                res[f'image_at_kphys_{tag}'] = dict(n=int(m.sum()),
                                                    **image_metrics(r, y))

        # figures
        if plots:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            for tag, k in (('bundle', 1.0), ('kphys', k_use)):
                d2 = df.copy()
                d2['x_tan_theta'] = df['x_tan_theta'] * k
                d2['y_tan_theta'] = df['y_tan_theta'] * k
                P0, D = track_lines(d2, tr, sel)
                r, y, xz = axis_approach(P0, D)
                fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
                ax[0].hist(r, bins=80, range=(0, 200))
                ax[0].axvline(10, color='r', ls='--', label='capsule r=10')
                ax[0].set_xlabel('closest approach to beam axis [mm]')
                ax[0].legend()
                h = ax[1].hist2d(xz[:, 0], xz[:, 1], bins=80,
                                 range=[[-60, 60], [-60, 60]], cmap='viridis')
                th = np.linspace(0, 2 * np.pi, 100)
                ax[1].plot(10 * np.cos(th), 10 * np.sin(th), 'r--', lw=1)
                ax[1].set_xlabel('global X [mm]')
                ax[1].set_ylabel('global Z [mm]')
                fig.colorbar(h[3], ax=ax[1])
                m = r < 25
                ax[2].hist(y[m], bins=60, range=(-120, 120))
                ax[2].axvspan(-29.5, 50.7, color='r', alpha=0.15,
                              label='He-3 gas')
                ax[2].set_xlabel('Y at closest approach [mm] (r<25)')
                ax[2].legend()
                fig.suptitle(f'{arm} {tag} k={k:.3f} '
                             f'(v={v_bundle / k:.1f} um/ns) n={sel.sum()}')
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir,
                                         f'image_{arm}_{tag}.png'), dpi=120)
                plt.close(fig)
            # head-on vs gated comparison at the physical k
            d2 = df.copy()
            d2['x_tan_theta'] = df['x_tan_theta'] * k_use
            d2['y_tan_theta'] = df['y_tan_theta'] * k_use
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
            for tag, m, c in (('inclined (old gate)', sel & rel, 'C0'),
                              ('head-on (new)', sel & ~rel, 'C1')):
                if m.sum() < 20:
                    continue
                P0, D = track_lines(d2, tr, m)
                r, y, xz = axis_approach(P0, D)
                ax[0].hist(r, bins=60, range=(0, 200), density=True,
                           histtype='step', lw=1.6, color=c,
                           label=f'{tag} (n={m.sum()})')
                if 'head-on' in tag:
                    h = ax[1].hist2d(xz[:, 0], xz[:, 1], bins=60,
                                     range=[[-60, 60], [-60, 60]],
                                     cmap='viridis')
                    fig.colorbar(h[3], ax=ax[1])
            th = np.linspace(0, 2 * np.pi, 100)
            ax[1].plot(10 * np.cos(th), 10 * np.sin(th), 'r--', lw=1)
            ax[0].axvline(10, color='r', ls='--', lw=1)
            ax[0].set_xlabel('closest approach to beam axis [mm]')
            ax[0].set_ylabel('density')
            ax[0].legend(fontsize=9)
            ax[1].set_xlabel('global X [mm] (head-on band only)')
            ax[1].set_ylabel('global Z [mm]')
            fig.suptitle(f'{arm}: the head-on band the slope gate used to '
                         f'discard, at k={k_use:.3f}')
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'image_{arm}_headon_cmp.png'),
                        dpi=120)
            plt.close(fig)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subrun', default='stat090_0000')
    ap.add_argument('--arms', default='A,C,B,D')
    ap.add_argument('--slim', default=None,
                    help='slim ntof_hits root file for this (run, subrun)')
    ap.add_argument('--no-slim-cut', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    out = a.out or os.path.join(ANALYSIS_BASE, a.run, a.subrun, 'imaging')
    os.makedirs(out, exist_ok=True)
    trs, G = transforms(a.run)
    results = []
    for arm in a.arms.split(','):
        try:
            r = run_arm(a.run, a.subrun, arm, trs, G, a.slim,
                        not a.no_slim_cut, out)
        except FileNotFoundError as e:
            r = dict(arm=arm, error=str(e))
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != 'k_scan'},
                         indent=1, default=str))
    with open(os.path.join(out, 'imaging_summary.json'), 'w') as f:
        json.dump(dict(run=a.run, subrun=a.subrun, slim=a.slim,
                       results=results), f, indent=1, default=str)
    print('wrote', os.path.join(out, 'imaging_summary.json'))


if __name__ == '__main__':
    main()
