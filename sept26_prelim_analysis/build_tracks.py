#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_tracks.py -- stage 3: the track database.

One row per **3D track segment**: a paired (x, y) candidate in one chamber of
one trigger.  This is the artefact every later analysis reads instead of
re-running anything, so the rules are strict:

* **Every X/Y pairing the reco made is a row**, gated or not.  `wft` pairs
  candidates into tracks (`track_id >= 0`) and then gates them on
  `quality_ok & plausible` in both planes; the events table's `n_tracks` counts
  only the survivors.  On run_145 tag 000 / arm A that is 69 of 128 pairings.
  Writing only the 69 would make the gate's efficiency unmeasurable from the
  product, and the 33 that are `quality_ok` but not `plausible` are exactly the
  marginal population any later cut has to argue about.  The gate is a
  **column** (`gated`), never a filter.
* **Nothing is invented.**  Columns the inputs cannot support are written as
  null with the reason recorded in the sidecar, not filled with a plausible
  guess.  See `NOT_POPULATED`.
* **Geometry comes from the waveform fit**, never from hit times
  (`../RECONSTRUCTION_BASIS.md`).

Identity is `(run, subrun, tag, event_id, arm, track_id)`.  `event_id` is
unique within a sub-run (run_145 `stat090_0000`: 1..57 754 across its 7 tags),
but *not* across sub-runs, so never join on it alone.

Usage:
    python -m sept26_prelim_analysis.build_tracks \\
        --run run_145 --subrun stat090_0000 --reco <dir with mx17_*/events_*>
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import paths
except ImportError:                                     # run as a script
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sept26_prelim_analysis import paths

from sept26_prelim_analysis import trigger_time

from ntof_tracking.reco import geometry as G

ARMS = ('A', 'B', 'C', 'D')

#: Strip coordinates run 0..398.58 mm; the plane centre is half of that.
STRIP_MAP_HALF = 199.29

#: The strip index runs along **-u_hat**, so ``x_local = -(x_p0 - half)``.
#: Measured 2026-08-20 and applied to the POSITION only, never to the tan --
#: flipping the tan instead mirrors the track about the plane *centre*, and the
#: pinwheel puts that centre ~16 mm off the beam axis, which displaces the
#: reconstructed source by twice the pinwheel in opposite directions for
#: opposing arms.  `../ntof_tracking/run145_target_imaging.py` carries the
#: measurement that fixed it; this is the same constant, not a second opinion.
IN_PLANE_SIGN = -1.0

#: **The Y plane needs the same flip, and until 2026-09-07 it did not get it.**
#: The 2026-08-20 measurement was made on the target image, which lives in the
#: XZ projection and is blind to the y sign -- so only x was ever measured and
#: y silently kept the raw strip direction.
#:
#: Measured the same way the x sign was, on the pointing-coincident sample of
#: run_145, both sub-runs: a track from the source arriving at ``y_local`` must
#: move further from zero going outward, so ``corr(y_local, tan_y)`` has to be
#: POSITIVE.  Unflipped it is negative in all four chambers and both sub-runs
#: (A -0.63/-0.67, C -0.34/-0.39, D -0.19/-0.24, B -0.23/-0.08); flipped it is
#: positive in all eight, and the implied angle scale from the y band lands in
#: the same family as the x band's (A 1.46 vs 1.30, C 2.5 vs 1.8).  k_y runs
#: systematically higher than k_x because the He-3 capsule is ~80 mm long along
#: y (``geometry.HE3_GAS_Y``) and an extended source dilutes the band -- so k_y
#: is an upper bound, not a competing measurement.
#:
#: What it broke: every 3D direction had a mirrored vertical component.
#: ``dca_axis_mm`` is the XZ projection and barely notices, which is why this
#: survived; ``target_y_mm``, ``angle_to_beam_deg``, the LS/plastic crossings
#: and **any opening angle between two chambers** all did notice.
IN_PLANE_SIGN_Y = -1.0

#: Angle fits that railed.  |tan| > 1 is a 45 deg track in a 30 mm gap, which
#: the acceptance does not contain; they are kept as rows and flagged.
TAN_SANE = 1.0

#: Columns the current inputs cannot support.  Present in the schema (so the
#: table's shape does not change when they arrive) and null, with the reason
#: carried into the sidecar rather than left for a reader to guess.
#:
#: **Empty since 2026-09-11.**  It held ``t_since_flash_ns`` and
#: ``e_neutron_keV``, described as blocked on a slim regeneration.  They were
#: not blocked: the slim's ``events`` tree already carries the trigger's time
#: since the gamma flash as ``t_dream_ns`` (DREAM clock) and ``t_pred_ns``
#: (n_TOF base, the segment's own fitted clock).
#: :mod:`sept26_prelim_analysis.trigger_time` has the evidence and does the
#: join.  The dict stays because the next column that cannot be supported
#: should be declared the same way.
NOT_POPULATED: dict[str, str] = {}

#: Angle-derived columns.  They are populated only for an arm carrying a
#: certified ``k`` (:mod:`sept26_prelim_analysis.k_arm`); for any other arm they
#: are NaN by construction, because the tans they descend from are NaN.  The
#: positions are NOT in this list: ``p0``, ``x_local``/``y_local`` and the
#: strip-map geometry do not depend on the drift velocity at all.
ANGLE_DERIVED = (
    'tanx', 'tany', 'd_x', 'd_y', 'd_z', 'path_len_mm', 'q_per_len',
    'drift_len_mm', 'dca_axis_mm', 'target_x_mm', 'target_y_mm', 'target_z_mm',
    'in_bore', 'angle_to_beam_deg', 'pred_sipm_bar', 'pred_plastic',
    'pred_ls', 'pred_n_cross', 'pred_sipm_s_mm',
)

SCHEMA = 'sept26_prelim/tracks/1'


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def load_reco(reco_dir: Path, arm: str) -> tuple[pd.DataFrame, dict]:
    """Every tag's candidate rows for one arm, plus the merged sidecar.

    Reads the ``.candidates.parquet`` files -- the events table is a reduction
    of them (one winner per plane) and cannot express a second track.
    """
    d = Path(reco_dir) / f'mx17_{arm}'
    frames, metas = [], {}
    for p in sorted(d.glob('events_*.candidates.parquet')):
        tag = p.name[len('events_'):-len('.candidates.parquet')]
        c = pd.read_parquet(p)
        c['tag'] = tag
        frames.append(c)
        mp = d / f'events_{tag}.meta.json'
        if mp.is_file():
            metas[tag] = json.loads(mp.read_text())
    if not frames:
        raise FileNotFoundError(
            f'no events_*.candidates.parquet under {d} -- stage 2 has not run '
            f'for arm {arm}, or WFT_EMIT_CANDIDATES was off when it did')
    return pd.concat(frames, ignore_index=True), metas


def pair_rows(cand: pd.DataFrame) -> pd.DataFrame:
    """Candidate rows -> one row per (tag, event, track), x and y side by side.

    Only ``track_id >= 0`` (the reco's X/Y pairings).  Unpaired candidates are
    not tracks and are not rows; their count survives as ``n_cand_x/y``.
    """
    t = cand[cand['track_id'] >= 0]
    key = ['tag', 'event_id', 'track_id']
    x = t[t['plane'] == 'x'].set_index(key)
    y = t[t['plane'] == 'y'].set_index(key)
    common = x.index.intersection(y.index)
    lost = len(x.index.symmetric_difference(y.index))
    if lost:
        # wft pairs one x with one y by construction, so this cannot happen
        # unless the sidecar was written by a different version.
        raise AssertionError(
            f'{lost} track_id(s) are not an x/y pair -- the candidates file '
            f'does not match the pairing contract in wft/reco.py')
    x, y = x.loc[common], y.loc[common]
    out = pd.DataFrame(index=common)
    for pl, src in (('x', x), ('y', y)):
        for c in ('p0', 'w', 't0', 'tan_theta', 'theta_deg', 'chi2', 'dof',
                  'p0_err', 'tan_err', 't0_err', 'q_sum', 'q_u50', 'q_u90',
                  'q_uend', 'n_strips', 'n_seed', 'n_dropped',
                  'slope_reliable', 'quality_ok', 'plausible', 'isochronous',
                  'n_candidates', 'rank', 'dchi2', 'ftst'):
            out[f'{pl}_{c}'] = src[c].to_numpy()
    out['gated'] = x['track_gated'].to_numpy() & y['track_gated'].to_numpy()
    return out.reset_index()


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def local_and_global(df: pd.DataFrame, tr: G.DetTransform,
                     gap_mm: float = G.DRIFT_GAP,
                     k: float | None = None) -> pd.DataFrame:
    """Attach local coordinates and the global line (p0, d) to each track.

    The direction is built from two points a full drift gap apart rather than
    from the tan directly, so the local->global rotation is applied once, to
    points, and the arm's convention lives entirely in ``DetTransform``.

    ``k`` is the in-situ angle scale (:mod:`sept26_prelim_analysis.k_arm`) and
    is applied HERE, once, to the tans -- so every angle-derived quantity
    downstream (the global direction, the pointing, the scintillator
    predictions, the path length) inherits it and no two of them can disagree
    about which calibration they are on.  ``k=None`` means the arm has no
    certified scale, and then the tans are **NaN**: an uncalibrated angle in a
    column called ``angle_to_beam_deg`` is exactly the silent error this
    package exists to prevent.  Positions are untouched either way -- they do
    not depend on the drift velocity.
    """
    xl = IN_PLANE_SIGN * (df['x_p0'].to_numpy(float) - STRIP_MAP_HALF)
    yl = IN_PLANE_SIGN_Y * (df['y_p0'].to_numpy(float) - STRIP_MAP_HALF)
    scale = np.nan if k is None else float(k)
    tx = df['x_tan_theta'].to_numpy(float) * scale
    ty = df['y_tan_theta'].to_numpy(float) * scale

    P0 = tr.local_to_global(xl, yl, np.zeros_like(xl))
    P1 = tr.local_to_global(xl - tx * gap_mm, yl - ty * gap_mm,
                            np.full_like(xl, gap_mm))
    D = P1 - P0
    n = np.linalg.norm(D, axis=-1, keepdims=True)
    D = np.divide(D, n, out=np.full_like(D, np.nan), where=n > 0)

    df = df.copy()
    df['x_local'], df['y_local'] = xl, yl
    df['tanx'], df['tany'] = tx, ty
    df['tan_raw_x'] = df['x_tan_theta'].to_numpy(float)
    df['tan_raw_y'] = df['y_tan_theta'].to_numpy(float)
    df['angle_calibrated'] = k is not None
    df['tan_sane'] = (np.abs(tx) <= TAN_SANE) & (np.abs(ty) <= TAN_SANE)
    for i, k in enumerate('xyz'):
        df[f'p0_{k}'], df[f'd_{k}'] = P0[:, i], D[:, i]
    return df


#: `q_uend` is the last depth bin above 5 % of the profile peak, so it is
#: quantised to the model's depth grid and **cannot exceed its last bin**.
#: `wft_beam.make_bundle` sets `n_depth_bins = 18`, so that edge is
#: 18 x 60 = 1080 ns -- and on run_145 **50.4 % of gated tracks sit exactly on
#: it**.  For those tracks q_uend is a censoring bound, not a measurement: the
#: column lit at least that much gap and possibly more.
DEPTH_BIN_NS = 60.0


def drift_extent(df: pd.DataFrame, v_um_ns: float,
                 n_depth_bins: int | None = None) -> pd.DataFrame:
    """Depth of the measured segment, from the charge profile's time extent.

    ``q_uend * v`` is how much of the 30 mm gap the track lit.  It is a quality
    and dE/dx quantity -- the *line* comes from (p0, tan) and does not depend
    on it -- but a track that lit 4 mm of gap and one that lit 28 mm are not
    equally trustworthy and the table has to be able to say so.

    **Half of them cannot say it.**  ``q_uend`` rails at the depth grid's last
    bin (see :data:`DEPTH_BIN_NS`), and a railed value is a lower bound.  So
    ``drift_railed`` is a column, the raw time is kept as measured, and
    ``q_per_len`` -- whose denominator would then be censored, making it a
    *wrong* number rather than an uncertain one -- is **null** where it rails.
    ``q_total`` stays populated either way: it is not divided by anything.

    This also makes `wft`'s plausibility window one-sided in practice: it
    requires ``250 <= q_uend <= 1100`` and the grid cannot produce more than
    1080, so the upper bound is unreachable and only the shallow cut bites.
    """
    v_mm = v_um_ns / 1000.0
    df = df.copy()
    ue = np.nanmax(np.c_[df['x_q_uend'].to_numpy(float),
                         df['y_q_uend'].to_numpy(float)], axis=1)
    df['drift_t_end_ns'] = ue
    df['drift_len_mm'] = ue * v_mm
    edge = (n_depth_bins * DEPTH_BIN_NS) if n_depth_bins else np.nan
    df['drift_railed'] = (ue >= edge - 1e-6) if np.isfinite(edge) else False
    df['depth_grid_edge_ns'] = edge

    sec = np.sqrt(1.0 + df['tanx'].to_numpy(float) ** 2
                  + df['tany'].to_numpy(float) ** 2)
    df['path_len_mm'] = df['drift_len_mm'] * sec
    df['q_total'] = df['x_q_sum'].to_numpy(float) + df['y_q_sum'].to_numpy(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        qpl = df['q_total'] / df['path_len_mm'].replace(0, np.nan)
    df['q_per_len'] = qpl.where(~df['drift_railed'])
    return df


def pointing(df: pd.DataFrame) -> pd.DataFrame:
    """Closest approach of each track line to the beam axis (global Y).

    The He-3 capsule is the source, it lies on the beam axis, and it is
    ~80 mm long by 10 mm in radius -- so "does this track come from the
    target?" is a distance to a *line*, not to a point, and the height along
    that line is itself a measurement worth keeping.
    """
    P0 = df[['p0_x', 'p0_y', 'p0_z']].to_numpy(float)
    D = df[['d_x', 'd_y', 'd_z']].to_numpy(float)
    # minimise |(P0 + s D) - (0, y, 0)|: only the XZ projection matters
    p, d = P0[:, [0, 2]], D[:, [0, 2]]
    dd = np.einsum('ij,ij->i', d, d)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = -np.einsum('ij,ij->i', p, d) / np.where(dd > 1e-12, dd, np.nan)
    c = P0 + s[:, None] * D
    df = df.copy()
    df['dca_axis_mm'] = np.hypot(c[:, 0], c[:, 2])
    df['target_y_mm'] = c[:, 1]
    df['target_x_mm'], df['target_z_mm'] = c[:, 0], c[:, 2]
    df['in_bore'] = ((df['dca_axis_mm'] <= G.HE3_R_MAX)
                     & (df['target_y_mm'] >= float(G.HE3_GAS_Y[0]))
                     & (df['target_y_mm'] <= float(G.HE3_GAS_Y[-1])))
    df['angle_to_beam_deg'] = np.degrees(np.arccos(np.clip(np.abs(D[:, 1]), 0, 1)))
    return df


def predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Which scintillator volumes each track's line crosses, going outward.

    `geometry.split_crossings` applies the beamline-origin prior: crossings
    from the beam-axis closest approach *outward* are the plausible particle
    path; anything behind it is a geometric line extension and is NOT a claim
    the particle went there.  Only the outward set is predicted here.

    These are the columns stage 4 calibrates the scintillator positions
    against, and stage 5 uses to confirm a pair -- so a prediction that is
    absent must read as absent, not as bar 0.
    """
    cols = {k: [] for k in ('pred_sipm_bar', 'pred_plastic', 'pred_ls',
                            'pred_n_cross', 'pred_sipm_s_mm')}
    for row in df[['p0_x', 'p0_y', 'p0_z', 'd_x', 'd_y', 'd_z']].itertuples(index=False):
        p0 = np.array(row[:3], float)
        d = np.array(row[3:], float)
        if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(d)):
            for k in cols:
                cols[k].append(np.nan if k.endswith(('_bar', '_mm', '_cross'))
                               else None)
            continue
        gseg = dict(p_lo_global=p0, p_hi_global=p0 + d, dir_global=d)
        out = G.split_crossings(gseg)['outward']
        bar, plastic, ls, s_sipm = np.nan, None, False, np.nan
        for c in out:
            if c['name'].startswith('SiPM bar') and not np.isfinite(bar):
                bar = float(c['name'].split()[-1])
                s_sipm = 0.5 * (c['s_in'] + c['s_out'])
            elif c['name'].startswith('plastic') and plastic is None:
                plastic = c['name'].split()[-1]         # 'L' or 'R'
            elif c['name'] == 'LS':
                ls = True
        cols['pred_sipm_bar'].append(bar)
        cols['pred_plastic'].append(plastic)
        cols['pred_ls'].append(ls)
        cols['pred_n_cross'].append(float(len(out)))
        cols['pred_sipm_s_mm'].append(s_sipm)
    df = df.copy()
    for k, v in cols.items():
        df[k] = v
    return df


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_arm(reco_dir: Path, arm: str, tr: G.DetTransform,
              run: str, subrun: str,
              k: float | None = None) -> tuple[pd.DataFrame, dict]:
    cand, metas = load_reco(reco_dir, arm)
    df = pair_rows(cand)
    df['run'], df['subrun'], df['arm'] = run, subrun, arm

    # v_drift is per bundle and per tag; assert the tags agree rather than
    # silently averaging two calibrations into one column.
    vs = {m['bundle']['v_drift'] for m in metas.values()}
    if len(vs) > 1:
        raise ValueError(f'arm {arm}: tags disagree on v_drift {sorted(vs)} -- '
                         'a bundle is per detector AND per run condition; '
                         'these tables cannot go in one table')
    v = float(next(iter(vs))) if vs else np.nan

    nbins = {m['bundle'].get('n_depth_bins') for m in metas.values()}
    nbins = next(iter(nbins)) if len(nbins) == 1 else None
    # The bundle's v is the Magboltz PRIOR.  k is the measured correction, so
    # the velocity that actually converts a drift time to a depth is v/k -- the
    # same k the tans are scaled by, since both follow from tan = w/v.  Using
    # the prior here while correcting the angles there would put the depth and
    # the direction on two different calibrations.
    v_insitu = v / float(k) if k else np.nan
    df = local_and_global(df, tr, k=k)
    df = drift_extent(df, v_insitu, n_depth_bins=nbins)
    df = pointing(df)
    df = predictions(df)
    df['v_drift_um_ns'] = v_insitu
    df['v_drift_prior_um_ns'] = v
    df['k_arm'] = float(k) if k else np.nan
    df['n_cand_x'] = df['x_n_candidates']
    df['n_cand_y'] = df['y_n_candidates']
    for c in ('x', 'y'):
        with np.errstate(divide='ignore', invalid='ignore'):
            df[f'chi2dof_{c}'] = (df[f'{c}_chi2'].to_numpy(float)
                                  / df[f'{c}_dof'].replace(0, np.nan))
    prov = dict(
        arm=arm, v_drift_prior_um_ns=v, k_arm=(float(k) if k else None),
        v_drift_um_ns=v_insitu, angle_calibrated=bool(k), n_depth_bins=nbins,
        frac_drift_railed=round(float(df['drift_railed'].mean()), 4),
        n_tags=len(metas),
        bundles=sorted({m['calibration'] for m in metas.values()}),
        code_commit=sorted({(m['bundle']['provenance'] or {}).get('code_commit',
                                                                 'unknown')
                            for m in metas.values()}),
        angle_constants_applied=sorted(
            {bool((m.get('angle_constants') or {}).get('applied'))
             for m in metas.values()}),
        allowlist=sorted({json.dumps((m.get('allowlist') or {}).get('header', {})
                                     .get('policy', {}), sort_keys=True)
                          for m in metas.values()}),
        seeding=dict(
            n_allowed=int(sum((m.get('allowlist') or {}).get('n_allowed', 0)
                              for m in metas.values())),
            n_seeded=int(sum((m.get('allowlist') or {}).get('n_seeded', 0)
                             for m in metas.values())),
            n_missing=int(sum((m.get('allowlist') or {}).get('n_missing', 0)
                              for m in metas.values()))),
    )
    return df, prov


def attach_context(tracks: pd.DataFrame, stage1: Path | None,
                   allow: Path | None, run: str | None = None,
                   subrun: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Join the stage-1 class and n_TOF flags, the stage-2 selection reason,
    and the trigger's time since the gamma flash.

    A track with no class is a track from an event stage 1 never classified --
    it should not exist, so it is flagged rather than dropped.
    """
    if stage1 is not None and Path(stage1).is_file():
        s1 = pd.read_parquet(stage1)
        keep = (['eventId', 'cls', 'arms_lit', 'bunch', 'is_flash',
                 'n_coinc_arms']
                + [f'coinc_{a}' for a in ARMS] + [f'wall_{a}' for a in ARMS]
                + [f'plastic_{a}' for a in ARMS])
        keep = [c for c in keep if c in s1.columns]
        tracks = tracks.merge(s1[keep].rename(columns={'eventId': 'event_id',
                                                       'cls': 'event_class'}),
                              on='event_id', how='left')
        # the n_TOF coincidence for THIS arm, as one column
        if all(f'coinc_{a}' in tracks.columns for a in ARMS):
            tracks['coinc_this_arm'] = [
                int(r[f'coinc_{a}']) if pd.notna(r[f'coinc_{a}']) else -1
                for a, r in zip(tracks['arm'], tracks.to_dict('records'))]
        tracks['no_stage1_class'] = tracks['event_class'].isna()
    if allow is not None and Path(allow).is_file():
        al = pd.read_parquet(allow)[['eventId', 'arm', 'reason']]
        tracks = tracks.merge(al.rename(columns={'eventId': 'event_id',
                                                 'reason': 'select_reason'}),
                              on=['event_id', 'arm'], how='left')
    # The time base. Its product is per sub-run and keyed on event_id, which is
    # unique only within one -- so it is joined here, where the sub-run is
    # known, and never on a combined frame.
    tprov = {}
    if run and subrun:
        try:
            tracks, tprov = trigger_time.attach(tracks, run, subrun)
        except FileNotFoundError as exc:
            tprov = dict(filled=False, why=str(exc))
            print(f'  [t] no time since flash: {exc}')
    return tracks, tprov


ORDER = (
    ['run', 'subrun', 'tag', 'event_id', 'arm', 'track_id', 'event_class',
     'select_reason', 'bunch']
    + ['x_local', 'y_local', 'tanx', 'tany', 'drift_t_end_ns', 'drift_len_mm',
       'path_len_mm', 'drift_railed']
    + [f'p0_{k}' for k in 'xyz'] + [f'd_{k}' for k in 'xyz']
    + ['gated', 'x_quality_ok', 'y_quality_ok', 'x_plausible', 'y_plausible',
       'chi2dof_x', 'chi2dof_y', 'x_n_strips', 'y_n_strips',
       'x_slope_reliable', 'y_slope_reliable', 'x_isochronous',
       'y_isochronous', 'tan_sane', 'n_cand_x', 'n_cand_y', 'x_rank', 'y_rank']
    + ['q_total', 'q_per_len', 'x_q_sum', 'y_q_sum', 'x_q_u50', 'y_q_u50',
       'x_q_u90', 'y_q_u90']
    + ['x_t0', 'y_t0', 'x_ftst', 'y_ftst', 't_since_flash_ns', 'e_neutron_keV']
    + ['dca_axis_mm', 'target_x_mm', 'target_y_mm', 'target_z_mm', 'in_bore',
       'angle_to_beam_deg']
    + ['pred_sipm_bar', 'pred_plastic', 'pred_ls', 'pred_n_cross',
       'pred_sipm_s_mm']
    + ['v_drift_um_ns', 'v_drift_prior_um_ns', 'k_arm', 'angle_calibrated',
       'tan_raw_x', 'tan_raw_y']
)


def build(run: str, subrun: str, reco_dir: Path, stage1: Path | None = None,
          allow: Path | None = None, out_dir: Path | None = None,
          k_arm: dict | None = None, write: bool = True):
    base = str(paths.root('runs')) + '/'
    cfg = json.loads((Path(base) / run / 'run_config.json').read_text())
    trs = G.detector_transforms(cfg)

    k_arm = dict(k_arm or {})
    frames, prov = [], {}
    for arm in ARMS:
        k = k_arm.get(arm)
        try:
            df, p = build_arm(Path(reco_dir), arm, trs[G.DET_NAME[arm]],
                              run, subrun, k=k)
        except FileNotFoundError as exc:
            print(f'  arm {arm}: skipped -- {exc}')
            continue
        note = (f'k={k:.4f}, v {p["v_drift_prior_um_ns"]:.1f} -> '
                f'{p["v_drift_um_ns"]:.1f} um/ns' if k
                else 'NO ANGLE SCALE -- angles null')
        print(f'  arm {arm}: {len(df):,} track segments '
              f'({int(df.gated.sum()):,} gated), {note}')
        frames.append(df)
        prov[arm] = p
    if not frames:
        raise FileNotFoundError(f'no reco for any arm under {reco_dir}')

    tracks = pd.concat(frames, ignore_index=True)
    tracks, tprov = attach_context(tracks, stage1, allow, run, subrun)
    for c, _why in NOT_POPULATED.items():
        if c not in tracks.columns:
            tracks[c] = np.nan
    cols = [c for c in ORDER if c in tracks.columns]
    tracks = tracks[cols + [c for c in tracks.columns if c not in cols]]
    tracks = tracks.sort_values(['arm', 'tag', 'event_id', 'track_id'])
    tracks = tracks.reset_index(drop=True)

    meta = dict(
        schema=SCHEMA, run=run, subrun=subrun,
        built=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        reco_dir=str(reco_dir), stage1=str(stage1) if stage1 else None,
        allowlist=str(allow) if allow else None,
        n_tracks=int(len(tracks)), n_gated=int(tracks['gated'].sum()),
        n_events=int(tracks.groupby(['tag', 'event_id']).ngroups),
        by_arm={a: int((tracks['arm'] == a).sum()) for a in ARMS},
        by_class=(tracks['event_class'].value_counts().to_dict()
                  if 'event_class' in tracks else {}),
        geometry=dict(in_plane_sign=IN_PLANE_SIGN,
                      strip_map_half=STRIP_MAP_HALF,
                      drift_gap_mm=G.DRIFT_GAP,
                      pinwheel=G.PINWHEEL,
                      mm_dist_x=G.MM_DIST_X, mm_dist_z=G.MM_DIST_Z),
        not_populated=NOT_POPULATED,
        trigger_time=tprov,
        k_arm=dict(applied=k_arm,
                   uncalibrated=[a_ for a_ in prov if a_ not in k_arm],
                   angle_derived_null_for_uncalibrated=list(ANGLE_DERIVED),
                   source='sept26_prelim_analysis.k_arm',
                   rule='tan_true = k * tan_reco; v_insitu = v_prior / k. An '
                        'arm with no certified k gets NaN angles, never k=1.'),
        provenance=prov)

    if write:
        out_dir = Path(out_dir) if out_dir else paths.out('stage3')
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f'tracks_{run}_{subrun}.parquet'
        tracks.to_parquet(p, index=False)
        (out_dir / f'tracks_{run}_{subrun}.meta.json').write_text(
            json.dumps(meta, indent=1, default=str))
        print(f'\n  -> {p}  ({len(tracks):,} rows, {len(cols)} named columns)')
    return tracks, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subrun', default='stat090_0000')
    ap.add_argument('--reco', type=Path, required=True,
                    help='directory holding mx17_<arm>/events_<tag>.parquet')
    ap.add_argument('--stage1', type=Path, default=None,
                    help='stage-1 candidates parquet (default: <out>/stage1/)')
    ap.add_argument('--allow', type=Path, default=None,
                    help='stage-2 allowlist parquet (default: <out>/stage2/)')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--k-arm', default=None,
                    help='in-situ angle scale, e.g. "A=1.02,C=0.98". Overrides '
                         '--k-file; an arm named here is applied whatever the '
                         'calibration says, so use it for diagnostics only.')
    ap.add_argument('--k-file', type=Path, default=None,
                    help='k_arm_<run>.json from sept26_prelim_analysis.k_arm. '
                         'Only arms it certifies are applied; the rest get '
                         'null angles. Default: <out>/kcal/k_arm_<run>.json')
    a = ap.parse_args()

    s1 = a.stage1 or paths.out('stage1') / f'candidates_{a.run}_{a.subrun}.parquet'
    al = a.allow or paths.out('stage2') / f'allowlist_{a.run}_{a.subrun}.parquet'
    kf = a.k_file or paths.out('kcal') / f'k_arm_{a.run}.json'
    k = {}
    if Path(kf).exists():
        cal = json.loads(Path(kf).read_text())
        k = {arm: float(v) for arm, v in (cal.get('apply') or {}).items()}
        for arm in ARMS:
            v = (cal.get('arms') or {}).get(arm, {})
            if arm not in k:
                print(f'  [k] arm {arm}: {v.get("verdict", "absent")} -- '
                      f'{v.get("reason", "no calibration")}')
    else:
        print(f'  [k] no calibration at {kf}; every angle will be null '
              f'unless --k-arm is given')
    if a.k_arm:                       # explicit override, diagnostics only
        k.update({arm: float(val)
                  for arm, val in (kv.split('=') for kv in a.k_arm.split(','))})

    print(f'{a.run}/{a.subrun}')
    tracks, meta = build(a.run, a.subrun, a.reco, stage1=s1, allow=al,
                         out_dir=a.out, k_arm=k)
    print(f'\n  {meta["n_tracks"]:,} segments in {meta["n_events"]:,} '
          f'(tag, event)s; {meta["n_gated"]:,} gated')
    if meta['by_class']:
        print('  by event class:', meta['by_class'])
    if NOT_POPULATED:
        print('\n  null by construction:')
        for c, why in NOT_POPULATED.items():
            print(f'    {c:<20} {why}')
    t = meta.get('trigger_time') or {}
    if t.get('n_with_time'):
        print(f'\n  time since flash: {t["n_with_time"]:,} of {t["n_tracks"]:,} '
              f'tracks, {t["t_ms"]["min"]:.3f}-{t["t_ms"]["max"]:.3f} ms '
              f'({t["e_eV"]["max"]:.2f} eV down to {t["e_eV"]["min"] * 1e3:.2f} meV)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
