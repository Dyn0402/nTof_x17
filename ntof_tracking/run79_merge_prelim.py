#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run79_merge_prelim.py -- join the PRELIMINARY run_79 waveform tracks to the
n_TOF scintillators, and run the two end-to-end checks that do not need a
reference telescope.

This is the "does the whole system roughly work?" pass of PLAN_08 §7-8, built
on the preliminary (transferred, not in-situ-calibrated) reconstruction of
`wft_beam.py`. Numbers here are indicative, not quotable.

Two validations, in increasing strength:

  [1] target pointing (internal, no n_TOF needed)
      The tracks come from a point-ish source -- the He-3 target at the origin,
      234.6 mm from the arm-A strip plane. So the reconstructed angle must
      correlate with the reconstructed position: tan(theta) = +-u / 234.6.
      A fit of tan vs u gives a slope whose MAGNITUDE tests the angle scale
      (i.e. v_drift) and whose SIGN fixes the in-plane convention.

  [2] wall pointing (external)
      Each DREAM trigger is a wall AND plastic coincidence. Matching the DREAM
      event to the n_TOF hit that caused it tells us WHICH wall segment fired
      -- the wall is 16 bars of 25 mm summed in 4 groups of 4 (100 mm each),
      95 mm beyond the strip plane. Extrapolating the track there and looking
      at the predicted u per fired segment tests the tracking AND the matcher
      at once, with no shared failure mode.

Usage:
    python -m ntof_tracking.run79_merge_prelim --tracks <events_prelim.parquet> \
        [--arm A] [--subrun stat090_0000] [--bunches 300] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

NTOF_RUN = 224572
V12_DIR = Path('/media/dylan/data/x17/ntof_reproc/v12_liqpileup')
OUT_BASE = Path('/media/dylan/data/x17/beam_july/analysis/wft/run_79')

# The clock fit of the reference pair (ntof_dream_merge/match_window.__main__):
# t_nTOF = (1 + k) * t_DREAM + t0, k = 108.9 ppm.
CLOCK_K, CLOCK_T0 = 1.089e-4, -197.5
# Accept bands, measured at t > 40 ms where the accidental floor is 2 %.
BANDS = ((-150.0, 150.0), (250.0, 450.0))

# ---- geometry, from ntof_tracking/reco/geometry.py (ported from the Geant build)
TARGET_TO_STRIPS = {'A': 234.6, 'B': 234.1, 'C': 234.6, 'D': 234.1}
STRIPS_TO_WALL = 96.4        # (126.5 - 30.1) mm along the arm's outward normal
PINWHEEL = {'D': 15.5, 'B': 15.75, 'A': 16.35, 'C': 17.3}
STRIP_MAP_HALF = 199.29      # strip-map coordinates run 0 .. 398.58 mm
SIPM_BAR_W, SIPM_N_BARS = 25.0, 20
N_WALL_SEG = 4               # 16 read bars summed in 4 groups of 4
N_POINTING_BINS = 6          # position bins for the tan-vs-u median fit
MIN_PER_BIN = 15
TAN_SANE = 1.0               # drop railed fits from the pointing fit only

# ---- the plastics, the other half of the trigger ---------------------------
# Two PVT bars per arm, 200 mm wide in u, 300 mm in v, sitting BEHIND the SiPM
# wall: for arm A the active depth is 178.1-198.1 mm past the strip plane, i.e.
# 82-102 mm past the wall. Every DREAM trigger is a wall AND plastic
# coincidence, so the plastic footprint is where we can trigger at all. Their
# centres are in the MM frame (pinwheel-shifted), the wall bars in the
# structure frame -- both are put in the u_wall coordinate below.
#
# Taken from ntof_tracking.reco.geometry, re-synced 2026-07-31 with the current
# Geant build (20 mm PVT, per-arm SiPM->plastic gap) and cross-checked against
# run_79/run_config.json (plastic_A_R at x=+85.37, z=425.22 -> bar centre
# +85.4 mm on the structure, front face 178.1 mm past the strips).
from ntof_tracking.reco import geometry as _geo    # noqa: E402

PLASTIC_HALF_U = _geo.PLASTIC_HALF_U
PLASTIC_HALF_V = _geo.PLASTIC_HALF_V     # along the beam
PLASTIC_U_OFFSET = _geo.PLASTIC_U_OFFSET
# NB the run_config puts the wall 97.4 mm past the strips, 1.0 mm beyond the
# STRIPS_TO_WALL above; u_wall in the stored table therefore carries a ~0.3 mm
# systematic, which is far inside everything this file measures. Not corrected
# here so that the table and the checks stay one consistent set of numbers.
PLASTIC_W = {a: (_geo.plastic_span(a)[0] - _geo.W_STRIP,
                 _geo.plastic_span(a)[1] - _geo.W_STRIP) for a in _geo.ARMS}
STRIPS_TO_PLASTIC = {a: 0.5 * sum(v) for a, v in PLASTIC_W.items()}


def wall_segment_u(seg: int) -> tuple:
    """(u_lo, u_hi) of wall group `seg` (0..3) in the STRUCTURE frame, under
    the geometry.py assumption that read bars are 1..16 of 20 and groups are
    four consecutive bars. Which group carries which n_TOF channel pair is NOT
    established -- that is one of the things check [2] measures."""
    bars = [seg * 4 + 1 + i for i in range(4)]
    u = [SIPM_BAR_W * (b - (SIPM_N_BARS - 1) / 2.0) for b in bars]
    return min(u) - SIPM_BAR_W / 2, max(u) + SIPM_BAR_W / 2


def plastic_bar_u(detn: int, arm: str, mapping: str = 'descending') -> tuple:
    """(u_lo, u_hi) of plastic bar `detn` (1 or 2), in the same coordinate as
    `u_wall_*` -- i.e. the structure frame, so the MM-frame bar centres carry
    the pinwheel offset.

    Which bar is which detn is the same open joint statement as the wall order:
    mx_july_beam_qa says detn 1 = left bar seen from the back and wall groups
    1-4 run left to right, so detn 1 sits at positive u exactly when the wall
    order is 'descending'. The data agrees (see fig. 2) but does not separate
    the mapping from the sign of the in-plane axis."""
    sgn = 1.0 if ((mapping == 'descending') == (int(detn) == 1)) else -1.0
    c = sgn * PLASTIC_U_OFFSET - PINWHEEL[arm]
    return c - PLASTIC_HALF_U, c + PLASTIC_HALF_U


def plastic_at_strips(arm: str, detn: int = 0, mapping: str = 'descending'):
    """The plastic footprint mapped back onto the STRIP plane along a ray from
    the target, in the mesh coordinates the target-pointing figure uses
    (u_mm / v_mm, MM frame). `detn` 1 or 2 gives that bar's u span; `detn` 0
    gives the v half-extent, which both bars share.

    CAVEAT: this is exact only for a track that comes from the target. A track
    with some other angle has its own lever arm to the plastic, so the real
    acceptance in (u, tan) is a diagonal band, not a vertical one -- these
    lines are the acceptance for the radial population the figure is about."""
    f = TARGET_TO_STRIPS[arm] / (TARGET_TO_STRIPS[arm] + STRIPS_TO_PLASTIC[arm])
    if not detn:
        return PLASTIC_HALF_V * f
    lo, hi = plastic_bar_u(detn, arm, mapping)
    return lo * f + PINWHEEL[arm], hi * f + PINWHEEL[arm]


def plastic_u_at_wall(u, arm: str):
    """Map an in-plane coordinate at the plastic depth onto the WALL plane,
    along a ray from the target at the origin. The plastics are 82-102 mm
    further out than the wall, so their footprint shrinks by ~0.78 when it is
    drawn on a "u at the wall" axis; without this the trigger acceptance would
    look wider than it is."""
    f = ((TARGET_TO_STRIPS[arm] + STRIPS_TO_WALL) /
         (TARGET_TO_STRIPS[arm] + STRIPS_TO_PLASTIC[arm]))
    return np.asarray(u, float) * f


# --------------------------------------------------------------- n_TOF side
def point_ntof_at_v12():
    """Point the reader at our reprocessed v12 partials, with the caches
    sandboxed. Same recipe (and the same order-of-operations trap) as
    ntof_processing/dream_regression.py: build the bunch join FIRST, against
    the official staged file, then repoint."""
    import ntof_dream_merge.ntof_io as ntof_io
    import ntof_dream_merge.tflash_repair as rep
    files = sorted(V12_DIR.glob(f'run{NTOF_RUN}_[0-9]*.root'),
                   key=lambda p: int(p.stem.split('_')[-1]))
    if not files:
        raise SystemExit(f'no v12 partials in {V12_DIR}')
    ntof_io.ntof_paths = lambda r: files          # type: ignore
    ntof_io.ntof_path = lambda r: files[0]        # type: ignore
    cache = ntof_io.variant_cache(V12_DIR, files)
    rep.CACHE_DIR = ntof_io.CACHE_DIR = cache
    ntof_io._TFLASH_FIX_CACHE.clear()
    return files, cache


def match_all_arms(ev: pd.DataFrame, bunches, arms=('A', 'B', 'C', 'D')):
    """Wall match in every arm, as the control for check [2].

    The DREAM trigger is an OR over the four sector SINGLES, so an event with a
    track in chamber A may well have been triggered by arm C. Events whose wall
    match is in another arm are the null sample: the arm-A track has no reason
    to point at the arm-A wall segment that fired, so any pointing correlation
    that survives on them is a bug, not physics.
    """
    from ntof_dream_merge.ntof_io import read_bunches
    out = {}
    for arm in arms:
        h = read_bunches(NTOF_RUN, f'WAL{arm}', bunches,
                         branches=('BunchNumber',), repair_tflash=False)
        o = np.lexsort((h['t_since_flash_ns'], h['BunchNumber']))
        cb, ct = h['BunchNumber'][o], h['t_since_flash_ns'][o]
        hit = np.zeros(len(ev), bool)
        pos = {int(e): i for i, e in enumerate(ev['eventId'].to_numpy())}
        for b, g in ev.groupby('BunchNumber'):
            s, e = np.searchsorted(cb, [b, b + 1])
            if e <= s:
                continue
            tt = ct[s:e]
            et = g['t_since_flash_ns'].to_numpy().astype(float)
            pred = et + CLOCK_K * et + CLOCK_T0
            for j, eidv in enumerate(g['eventId'].to_numpy()):
                lo = np.searchsorted(tt, pred[j] - 500.0)
                hi = np.searchsorted(tt, pred[j] + 500.0)
                if hi <= lo:
                    continue
                r = tt[lo:hi] - pred[j]
                ok = np.zeros(r.size, bool)
                for blo, bhi in BANDS:
                    ok |= (r >= blo) & (r <= bhi)
                if ok.any():
                    hit[pos[int(eidv)]] = True
        out[f'wal_hit_{arm}'] = hit
    return pd.DataFrame(out, index=ev.index)


def match_events(ev: pd.DataFrame, bunches, arm: str):
    """Per DREAM event, the n_TOF wall hit that fired its trigger.

    For each event we take the nearest WAL<arm> hit to the predicted time and
    keep it if it lands in an accept band; the plastic partner is looked up the
    same way. `repair_tflash=False` -- v12's own tflash is the point of using
    v12 (ntof_processing/HANDOFF_2026-07-29).
    """
    from ntof_dream_merge.ntof_io import read_bunches
    out = {}
    for kind in ('WAL', 'PSS'):
        tree = f'{kind}{arm}'
        h = read_bunches(NTOF_RUN, tree, bunches,
                         branches=('BunchNumber', 'detn', 'amp'),
                         repair_tflash=False)
        o = np.lexsort((h['t_since_flash_ns'], h['BunchNumber']))
        cb = h['BunchNumber'][o]
        ct = h['t_since_flash_ns'][o]
        cd = h['detn'][o].astype(int)
        ca = h['amp'][o]
        res = {k: np.full(len(ev), np.nan) for k in ('dt', 'detn', 'amp')}
        eid_pos = {int(e): i for i, e in enumerate(ev['eventId'].to_numpy())}
        for b, g in ev.groupby('BunchNumber'):
            s, e = np.searchsorted(cb, [b, b + 1])
            if e <= s:
                continue
            tt, dd, aa = ct[s:e], cd[s:e], ca[s:e]
            et = g['t_since_flash_ns'].to_numpy().astype(float)
            pred = et + CLOCK_K * et + CLOCK_T0
            for j, eidv in enumerate(g['eventId'].to_numpy()):
                lo = np.searchsorted(tt, pred[j] - 20_000.0)
                hi = np.searchsorted(tt, pred[j] + 20_000.0)
                if hi <= lo:
                    continue
                r = tt[lo:hi] - pred[j]
                # the trigger hit is the earliest one inside an accept band;
                # nearest-in-time would prefer the ~330 ns delayed wall lobe
                ok = np.zeros(r.size, bool)
                for blo, bhi in BANDS:
                    ok |= (r >= blo) & (r <= bhi)
                if not ok.any():
                    continue
                i = int(np.flatnonzero(ok)[0])
                k = eid_pos[int(eidv)]
                res['dt'][k] = r[i]
                res['detn'][k] = dd[lo + i]
                res['amp'][k] = aa[lo + i]
        for k, v in res.items():
            out[f'{kind.lower()}_{k}'] = v
    return pd.DataFrame(out, index=ev.index)


# ---------------------------------------------------------------- track side
# Bench-measured drift gaps (mx_june_wft/GAP_STUDY_2026-07-30.md). Assumes the
# cathode travelled with the chamber from the June bench to n_TOF -- stated,
# not verified (PLAN_08 §11).
GAP_MM = {'A': 27.9, 'B': 30.5, 'C': 30.0, 'D': 30.0}


def column_v_estimate(d: pd.DataFrame, arm: str) -> dict:
    """A crude, reference-free drift velocity from the fitted charge column.

    A minimum-ionising track crossing the whole gap deposits charge uniformly
    in depth, so the NNLS charge-arrival profile runs from t0 to t0 + gap/v:
    its END is gap/v and its MEDIAN is half that. Two estimators, deliberately
    both reported, because each is biased in a known direction here:

      q_uend  -- the column end. Biased HIGH by the fits whose profile rails
                 into the last basis bin (K x 60 = 1080 ns), so those are cut.
      2*q_u50 -- twice the median arrival. Biased LOW by right-truncation: the
                 20-sample window clips the deep end of the column.

    This is NOT the measurement PLAN_08 §6.3 asks for (stacked profiles + an
    erfc endpoint, `mx_june_wft/bench/gap_study.py`). It is a factor-of-ten
    sanity check on the assumed v -- which is all it is used for here.
    """
    gap = GAP_MM[arm]
    out = {'gap_mm': gap}
    for plane in ('x', 'y'):
        m = d[f'{plane}_ok'] & d[f'{plane}_quality_ok']
        ue = d.loc[m, f'{plane}_q_uend'].to_numpy()
        u5 = d.loc[m, f'{plane}_q_u50'].to_numpy()
        railed = ue >= 1050.0
        t_end = float(np.nanmedian(ue[~railed])) if (~railed).sum() > 20 else np.nan
        t_med = float(np.nanmedian(u5))
        # gap [mm] / t [ns] * 1e3 -> um/ns
        out[plane] = dict(
            n=int(m.sum()), railed_frac=float(np.mean(railed)),
            t_end_ns=t_end,
            v_from_end=(gap * 1e3 / t_end) if t_end and t_end > 0 else np.nan,
            t_2u50_ns=2 * t_med,
            v_from_u50=(gap * 1e3 / (2 * t_med)) if t_med > 0 else np.nan)
    return out


# ---------------------------------------------------------------- geometry
def rescale_angles(tracks: pd.DataFrame, v_bundle: float, v_new: float) -> pd.DataFrame:
    """Re-map w -> tan for a different drift velocity, without refitting.

    v_drift is NOT a parameter of the forward model: the fit measures the
    transverse speed w [mm/ns] and the bundle only converts it,
    tan = (w*1e3 - w0) / (kw * v). So a corrected v is a pure rescaling of the
    angle columns -- worth knowing, because it means the expensive part of the
    chain does not have to be redone when v is finally measured properly.
    """
    if not (np.isfinite(v_bundle) and np.isfinite(v_new) and v_new > 0):
        raise ValueError(f'cannot rescale angles: v_bundle={v_bundle}, '
                         f'v_new={v_new}')
    d = tracks.copy()
    s = v_bundle / v_new
    for p in ('x', 'y'):
        d[f'{p}_tan_theta'] = d[f'{p}_tan_theta'] * s
        d[f'{p}_tan_err'] = d[f'{p}_tan_err'] * s
        d[f'{p}_theta_deg'] = np.degrees(np.arctan(d[f'{p}_tan_theta']))
    return d


def track_frame(tracks: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Add the geometric quantities the checks need.

    u_mm  = in-plane coordinate about the chamber centre (strip map 0..398.58)
    u_wall = the same coordinate extrapolated to the wall plane, in the wall's
             (structure) frame -- the pinwheel offset differs between the two.

    SIGN CAVEAT: the fit's `tan_theta` is d(position)/d(drift depth), and depth
    increases TOWARD the target (the cathode is the target-facing side). So the
    outward extrapolation carries a minus sign. Whether the strip-map axis and
    the fit's transverse-speed axis share a handedness is exactly what check [1]
    resolves -- until then `u_wall` is computed for both signs.
    """
    d = tracks.copy()
    d['u_mm'] = d['x_p0'] - STRIP_MAP_HALF
    d['v_mm'] = d['y_p0'] - STRIP_MAP_HALF
    for s, tag in ((-1.0, 'm'), (+1.0, 'p')):
        d[f'u_wall_{tag}'] = (d['u_mm'] + s * STRIPS_TO_WALL * d['x_tan_theta']
                              - PINWHEEL[arm])
    return d


def target_pointing(d: pd.DataFrame, arm: str) -> dict:
    """Check [1]: the tan-vs-position slope a point source at the origin
    imposes. |slope| = 1 / 234.6 mm = 0.00426 if the angle scale is right."""
    out = {}
    L = TARGET_TO_STRIPS[arm]
    for plane, u in (('x', 'u_mm'), ('y', 'v_mm')):
        m = (d[f'{plane}_ok'] & d[f'{plane}_quality_ok']
             & np.isfinite(d[u]) & np.isfinite(d[f'{plane}_tan_theta'])
             & (d[f'{plane}_tan_theta'].abs() < TAN_SANE))
        if m.sum() < 50:
            out[plane] = None
            continue
        x = d.loc[m, u].to_numpy()
        y = d.loc[m, f'{plane}_tan_theta'].to_numpy()
        # robust: median tan in position bins, then a straight line through them.
        # Medians, not a least-squares fit on the raw points: the fit's tan has
        # a heavy tail (nothing bounds it -- TAN_MAX only ranks candidates), and
        # a handful of railed events otherwise set the slope.
        bins = np.linspace(-150, 150, N_POINTING_BINS + 1)
        ib = np.digitize(x, bins) - 1
        bx, by, bn = [], [], []
        for i in range(len(bins) - 1):
            s = ib == i
            if s.sum() < MIN_PER_BIN:
                continue
            bx.append(0.5 * (bins[i] + bins[i + 1]))
            by.append(float(np.median(y[s])))
            bn.append(int(s.sum()))
        if len(bx) < 4:
            out[plane] = None
            continue
        sl, ic = np.polyfit(bx, by, 1)
        out[plane] = dict(n=int(m.sum()), slope=float(sl), intercept=float(ic),
                          expected=1.0 / L, ratio=float(abs(sl) * L),
                          bins=[dict(u=a, tan=b, n=c) for a, b, c in zip(bx, by, bn)])
    return out


def wall_pointing(d: pd.DataFrame, sign_tag: str) -> dict:
    """Check [2]: predicted u at the wall, per fired wall segment.

    Which n_TOF detn pair sits at which u is NOT established (the wall read-out
    mapping is an open item -- mx_july_beam_qa/README.md), so both orders are
    scored: 'ascending' = segment 0 on the four most-negative bars, 'descending'
    = the reverse. The data picks one.
    """
    seg = ((d['wal_detn'] - 1) // 2).astype('Int64')
    out = {'n_matched': int(np.isfinite(d['wal_detn']).sum()), 'segments': []}
    for g in range(N_WALL_SEG):
        m = (seg == g) & np.isfinite(d[f'u_wall_{sign_tag}'])
        if m.sum() < 20:
            out['segments'].append(dict(seg=g, n=int(m.sum())))
            continue
        u = d.loc[m, f'u_wall_{sign_tag}'].to_numpy()
        lo, hi = wall_segment_u(g)
        rlo, rhi = wall_segment_u(N_WALL_SEG - 1 - g)
        out['segments'].append(dict(
            seg=g, n=int(m.sum()), u_median=float(np.median(u)),
            u_p25=float(np.percentile(u, 25)), u_p75=float(np.percentile(u, 75)),
            geom_lo=lo, geom_hi=hi,
            geom_lo_rev=rlo, geom_hi_rev=rhi,
            inside_frac=float(np.mean((u >= lo) & (u <= hi))),
            inside_frac_rev=float(np.mean((u >= rlo) & (u <= rhi)))))
    ok = [s for s in out['segments'] if 'u_median' in s]
    if len(ok) >= 3:
        g = np.array([s['seg'] for s in ok], float)
        u = np.array([s['u_median'] for s in ok])
        n = np.array([s['n'] for s in ok], float)
        out['ordering_corr'] = float(np.corrcoef(g, u)[0, 1])
        out['spread_mm'] = float(u.max() - u.min())
        out['inside_ascending'] = float(
            np.sum(n * np.array([s['inside_frac'] for s in ok])) / n.sum())
        out['inside_descending'] = float(
            np.sum(n * np.array([s['inside_frac_rev'] for s in ok])) / n.sum())
        out['mapping'] = ('descending'
                          if out['inside_descending'] > out['inside_ascending']
                          else 'ascending')
    return out


# -------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks', required=True)
    ap.add_argument('--arm', default='A')
    ap.add_argument('--run', default='run_79')
    ap.add_argument('--subrun', default='stat090_0000')
    ap.add_argument('--bunches', type=int, default=0,
                    help='use only the first N bunches (0 = all the tracks cover)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--rescale-v', type=float, default=None,
                    help='re-map w -> tan for this drift velocity [um/ns] '
                         'instead of the bundle value (no refit needed)')
    a = ap.parse_args()

    tracks = pd.read_parquet(a.tracks)
    print(f'[merge] {len(tracks):,} reconstructed events from {a.tracks}')

    # The bundle's v_drift IS the angle scale, so it is not optional. It comes
    # from the reco sidecar; falling back to the bundle directory covers a
    # table checkpointed by an older driver. A missing value used to propagate
    # as a NaN rescale and silently empty every check downstream -- fail here
    # instead.
    meta_p = Path(str(a.tracks).replace('.parquet', '.meta.json'))
    v_bundle, bundle_dir = np.nan, None
    if meta_p.exists():
        mj = json.load(open(meta_p))
        v_bundle = float(mj['bundle']['v_drift'])
        bundle_dir = mj.get('calibration')
        if mj.get('partial'):
            print(f'[merge] NOTE: the track table is a PARTIAL checkpoint '
                  f'({len(mj.get("tags_done", []))} file tag(s) done)')
    if not np.isfinite(v_bundle):
        cand = Path(a.tracks).parent / 'calib_bundle_prelim' / 'bundle.json'
        if cand.exists():
            v_bundle = float(json.load(open(cand))['v_drift'])
            print(f'[merge] no reco sidecar; took v_drift from {cand}')
    if a.rescale_v and not np.isfinite(v_bundle):
        raise SystemExit('--rescale-v needs the bundle v_drift, and neither '
                         f'{meta_p} nor a calib_bundle_prelim/bundle.json was '
                         'found next to the track table')
    cv = column_v_estimate(tracks, a.arm)
    print(f'[merge] drift velocity: bundle {v_bundle:.1f} um/ns; charge column '
          f'says (gap {cv["gap_mm"]} mm)')
    for plane in ('x', 'y'):
        c = cv[plane]
        print(f'    {plane}: column end {c["t_end_ns"]:.0f} ns -> '
              f'{c["v_from_end"]:.1f} um/ns   2 x median arrival '
              f'{c["t_2u50_ns"]:.0f} ns -> {c["v_from_u50"]:.1f} um/ns   '
              f'(basis railed in {c["railed_frac"]:.0%})')
    if a.rescale_v:
        print(f'[merge] rescaling angles to v = {a.rescale_v} um/ns '
              f'(x{v_bundle / a.rescale_v:.3f})')
        tracks = rescale_angles(tracks, v_bundle, a.rescale_v)

    # ORDER MATTERS: the bunch join runs off the official file's PKUP/index
    # trees, so build it BEFORE repointing the reader at the v12 partials.
    from ntof_dream_merge.bunch_join import dream_event_to_bunch
    ev_all = dream_event_to_bunch(a.run, a.subrun, NTOF_RUN)
    print(f'[merge] bunch join: {len(ev_all):,} DREAM events, '
          f'{(ev_all["BunchNumber"] > 0).mean():.1%} carry a bunch')

    files, cache = point_ntof_at_v12()
    print(f'[merge] n_TOF: v12, {len(files)} partials, caches in {cache}')

    ev = ev_all[(ev_all['BunchNumber'] > 0) & (~ev_all['is_flash'])
                & ev_all['eventId'].isin(tracks['event_id'])].copy()
    bunches = np.sort(ev['BunchNumber'].unique())
    if a.bunches:
        bunches = bunches[:a.bunches]
        ev = ev[ev['BunchNumber'].isin(bunches)]
    print(f'[merge] {len(ev):,} reconstructed events in {len(bunches)} bunches '
          f'({bunches.min()}-{bunches.max()})')

    ev = ev.reset_index(drop=True)
    m = match_events(ev, bunches, a.arm)
    ctrl = match_all_arms(ev, bunches)
    ev = pd.concat([ev, m, ctrl], axis=1)
    print('[merge] wall match by arm: ' + '  '.join(
        f'{arm} {ctrl[f"wal_hit_{arm}"].mean():.1%}' for arm in 'ABCD'))
    wal_ok = np.isfinite(ev['wal_dt'])
    pss_ok = np.isfinite(ev['pss_dt'])
    print(f'[merge] wall match {wal_ok.mean():.1%}, plastic match {pss_ok.mean():.1%}, '
          f'both {(wal_ok & pss_ok).mean():.1%}')

    d = track_frame(tracks, a.arm).merge(ev, left_on='event_id',
                                         right_on='eventId', how='inner')
    print(f'[merge] merged table: {len(d):,} rows')

    tp = target_pointing(d, a.arm)
    print('\n[1] target pointing (expected |slope| = '
          f'{1.0 / TARGET_TO_STRIPS[a.arm]:.5f} /mm)')
    for plane, r in tp.items():
        if r is None:
            print(f'    {plane}: too few events')
            continue
        print(f'    {plane}: slope {r["slope"]:+.5f} /mm  '
              f'(|slope|/expected = {r["ratio"]:.2f})  '
              f'intercept {r["intercept"]:+.4f}  n = {r["n"]:,}')

    # Which sign carries the outward extrapolation is fixed by check [1], not
    # assumed: a target-pointing track has tan = -u / L (drift depth increases
    # toward the target), so a NEGATIVE measured slope means the fit's
    # transverse-speed axis and the strip-map axis agree, and the wall -- 96 mm
    # further OUT -- sits at u - 96.4 * tan. A positive slope means they are
    # anti-aligned and the other sign applies.
    # Dilution test. A median slope shallower than 1/L can mean the angle scale
    # (v_drift) is low, OR that a good fraction of the tracks simply do not come
    # from the target -- a median is robust to outliers but not to a large
    # non-pointing population. Events with an arm-A wall AND plastic tag are
    # through-going in this arm; if the slope steepens on them, dilution is
    # real and the pointing ratio on all events is a lower bound.
    tagged = d[np.isfinite(d['wal_dt']) & np.isfinite(d['pss_dt'])]
    tp_tag = target_pointing(tagged, a.arm) if len(tagged) > 200 else None
    if tp_tag and tp_tag.get('x'):
        print(f'    ... on the {len(tagged):,} arm-{a.arm} wall+plastic tagged '
              f'events: slope {tp_tag["x"]["slope"]:+.5f} /mm '
              f'({tp_tag["x"]["ratio"]:.2f}x expected)')

    sign_tag = 'm' if (tp.get('x') or {}).get('slope', 0.0) < 0 else 'p'
    wp = wall_pointing(d, sign_tag)
    print(f'\n[2] wall pointing (extrapolation sign "{sign_tag}", '
          f'{wp["n_matched"]:,} wall-matched)')
    for s in wp['segments']:
        if 'u_median' not in s:
            print(f'    seg {s["seg"]}: n = {s["n"]}')
            continue
        print(f'    seg {s["seg"]}: n = {s["n"]:5,}  u_pred median '
              f'{s["u_median"]:+7.1f} mm  IQR [{s["u_p25"]:+6.1f}, {s["u_p75"]:+6.1f}]'
              f'   bars asc [{s["geom_lo"]:+6.1f}, {s["geom_hi"]:+6.1f}] '
              f'{s["inside_frac"]:3.0%}'
              f'   desc [{s["geom_lo_rev"]:+6.1f}, {s["geom_hi_rev"]:+6.1f}] '
              f'{s["inside_frac_rev"]:3.0%}')
    if 'ordering_corr' in wp:
        print(f'    segment-vs-u ordering correlation {wp["ordering_corr"]:+.2f}, '
              f'spread {wp["spread_mm"]:.0f} mm')
        print(f'    detn->bar mapping: {wp["mapping"]} '
              f'(inside-band {wp["inside_ascending"]:.0%} ascending vs '
              f'{wp["inside_descending"]:.0%} descending)')

    # the null sample: events whose trigger came from another arm's wall
    null = d[~d[f'wal_hit_{a.arm}'] & d[['wal_hit_B', 'wal_hit_C', 'wal_hit_D',
                                         'wal_hit_A']].any(axis=1)]
    ctrl_tp = target_pointing(null, a.arm) if len(null) > 200 else None
    if ctrl_tp and ctrl_tp.get('x'):
        print(f'\n    control: {len(null):,} events triggered by another arm -- '
              f'their arm-{a.arm} tracks give slope '
              f'{ctrl_tp["x"]["slope"]:+.5f} /mm '
              f'({ctrl_tp["x"]["ratio"]:.2f}x expected)')

    out = Path(a.out or (OUT_BASE / a.subrun / f'mx17_{a.arm}'))
    out.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out / 'merged_prelim.parquet', index=False)
    with open(out / 'merged_prelim.summary.json', 'w') as f:
        json.dump(dict(status='PRELIMINARY', tracks=a.tracks, ntof_run=NTOF_RUN,
                       ntof_variant='v12_liqpileup', arm=a.arm,
                       run=a.run, sub_run=a.subrun,
                       n_tracks=int(len(tracks)), n_merged=int(len(d)),
                       n_bunches=int(len(bunches)),
                       match=dict(wall=float(wal_ok.mean()),
                                  plastic=float(pss_ok.mean()),
                                  both=float((wal_ok & pss_ok).mean())),
                       wall_match_by_arm={arm: float(ev[f'wal_hit_{arm}'].mean())
                                          for arm in 'ABCD'},
                       v_bundle=float(v_bundle), v_rescaled=a.rescale_v,
                       column_v=cv,
                       target_pointing=tp, target_pointing_tagged=tp_tag,
                       wall_pointing=wp, control_other_arm=ctrl_tp,
                       extrapolation_sign=sign_tag), f, indent=1, default=str)
    print(f'\n[merge] wrote {out}/merged_prelim.parquet')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
