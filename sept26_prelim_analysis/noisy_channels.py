#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
noisy_channels.py -- per-strip dead/hot/noisy classifier, from raw hits.

WHY THIS EXISTS.  ``HANDOFF_D_NOISY_CHANNELS.md`` found that D's hot cells --
22.4 % of its pointing-coincident sample, and the only sample cut that moves
its angle scale -- are whole x COLUMNS sitting on the 49.8 mm connector
boundaries, not 2D blobs.  That finding came from ``k_robustness.hot_cells``,
a 10 mm 2D grid built on the FITTED cluster position (``events_prelim.x_p0``),
which is post-fit and therefore partly circular for anything that is about to
change how the fit works.  This module rebuilds the same classification at
**channel granularity on raw ``combined_hits``** -- the honest, pre-fit input
-- so the column/connector structure can be confirmed independently, and so a
channel that never survives to a fitted cluster still shows up.

THREE CLASSES, ONE SCALE EACH.

  * ``dead``  -- occupancy far under its local neighbourhood.  Already known
    from ``source_imaging.dead_ranges`` (also x_p0-based); this reproduces it
    from hits.
  * ``hot``   -- occupancy far over its local neighbourhood.  D's dominant
    fault (HANDOFF_D_NOISY_CHANNELS.md Sec. 2).
  * ``noisy`` -- ordinary occupancy, but the hits-level spatial clusters this
    channel joins are wider and less dense than its neighbours' (Sec. 2.1: hot
    cells carry the SAME median charge as normal cells (0.94x) but 1.7x more
    strips -- wide, dilute deposits).  Uses ``wft.seed`` clustering -- the
    production significance-floor + spatial-gap clustering -- restricted to
    ``ntof_tracking.reco.noise``-cleaned hits of track-like events only (see
    ``shape_stats``/``track_like_event_keys``) -- so this is hits only, no
    waveform fit, no NNLS. **Weaker evidence than ``hot``**: it is sensitive
    to its own threshold (NOISY_WIDTH_FACTOR/NOISY_DENSITY_FACTOR -- see the
    comment there) and, on D-x, the plane Sec. 2.1's evidence is actually
    about, it currently flags nothing at all -- ``hot`` already accounts for
    the shape-anomalous channels there.

**Local, not plane-wide, throughout**: the plane median is dragged by the
trigger's own two-lobe illumination (``plastic_acceptance.py``), which is real
physics and must not be flagged.  Every median here is taken over a
``WINDOW_STRIPS``-wide neighbourhood (one connector), and over OCCUPIED /
GOOD channels only -- letting a dead or already-flagged channel into the
median drags the threshold down and calls ordinary channels hot (the same
trap ``k_robustness.hot_cells`` and ``source_imaging.dead_ranges`` guard).

**Two bugs found and fixed while validating the shape pass (2026-09-08),
both upstream of this module**: (1) the DAQ's ``eventId`` resets every
subrun, so concatenating subruns and grouping on ``eventId`` alone silently
merges unrelated events -- fixed by carrying a ``subrun`` column and grouping
on ``(subrun, eventId)`` everywhere. (2) raw ``combined_hits`` is dominated by
a documented ~1.2-1.3 MHz coherent-band noise residual (``noise.py``) that
``wft.seed``'s own significance floor does not remove (it is relative to the
event's own max, and a coherent band shares one amplitude scale across
hundreds of strips) -- without both fixes the median hits-level cluster
"width" on the clean A-x plane was 45-110 strips (nearly a quarter of the
plane); with them it is 18. ``load_hits`` now runs
``ntof_tracking.reco.noise.flag_noise`` per subrun before the shape pass uses
it; ``occupancy()`` is untouched by either fix -- it always used every raw hit.

Order of work (HANDOFF_D_NOISY_CHANNELS.md Sec. 4): build standalone and look
at the map for all four chambers BEFORE touching the reconstruction. This
module does that and stops there -- it does not yet wire wildcards into
``wft/reco.py`` (Sec. 3.2), which needs its own re-run and comparison against
the frozen products.

``noise.flag_noise`` is a per-event Python pass (isolated-hit removal in
particular is O(n) per plane-event, not vectorized) -- the default 3 subruns
take **~15 minutes**, almost all of it in ``load_hits``.

    python -m sept26_prelim_analysis.noisy_channels --run run_145
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

SCHEMA = 'sept26_prelim/noisy_channels/1'
ARMS = ('A', 'B', 'C', 'D')
PLANES = ('x', 'y')
N_STRIPS = 512
STRIP_MM = 398.58 / N_STRIPS               # 0.7784 mm/strip
CONNECTOR_STRIPS = 64                      # 512 / 8 -- 49.8 mm/connector

#: Local-median window, in strips -- one connector, matching the fault's own
#: scale (Sec. 2.3) and well under the illumination lobe's ~100+ mm scale.
WINDOW_STRIPS = 64
#: A channel is dead below, hot above, this multiple of its local median
#: occupancy.  DEAD_THRESH matches source_imaging.DEAD_THRESH (same
#: convention, same 0.20).  HOT_FACTOR matches k_robustness.HOT_FACTOR (same
#: 5.0) -- checked empirically 2026-09-08 that this ALSO reproduces the right
#: scale at per-channel granularity: A/B/C land at 0-3 % of channels flagged
#: (k_robustness's 2D-cell hot_cell_frac: A 2.1 %, B 1.5 %, C 1.9 %) and D at
#: ~8-12 % per plane (k_robustness: 10.75 %). A lower factor (e.g. 3.0) floods
#: every chamber, A included, with single-/double-channel flags that are
#: ordinary strip-to-strip gain scatter, not the connector fault.
DEAD_THRESH = 0.20
HOT_FACTOR = 5.0
#: A window needs at least this many occupied (resp. good) neighbours before
#: its local median is trusted; below that the global (occupied / good-only)
#: median is used instead -- the same fallback both source guards use.
MIN_NEIGHBOURS = 4

#: Shape (``noisy``) thresholds -- Sec. 2.1's ratios (0.94 charge, 1.7x width)
#: relative to a channel's own local reference, with margin so the noise in a
#: median-of-medians does not itself get flagged.
#:
#: FRAGILE -- checked 2026-09-08, after the two shape-pipeline fixes below:
#: A-x's flagged count swings 27 -> 2 -> 0 as (width_factor, density_factor)
#: tightens from (1.2, 0.85) to (1.3, 0.75) to (1.5, 0.6), and D-x -- the
#: plane the handoff's own shape evidence (Sec. 2.1) is about -- comes out
#: EMPTY at every setting tried: its shape-anomalous channels are already
#: caught by occupancy (``hot``), leaving nothing in the 'good' pool to flag.
#: Read ``noisy`` as "a channel worth a second look", not as validated the
#: way HOT_FACTOR is (matched against k_robustness's independent 2D-cell
#: measurement to within ~1 point on every chamber -- see the module
#: docstring). The values below are the more conservative end of what was
#: tried.
NOISY_WIDTH_FACTOR = 1.3
NOISY_DENSITY_FACTOR = 0.75
MIN_SHAPE_CLUSTERS = 20                    # channels seen in fewer clusters are 'unmeasured'


# --------------------------------------------------------------------------- #
# Loading -- raw hits, all of them, before any selection
# --------------------------------------------------------------------------- #

def load_hits(run: str, subruns) -> pd.DataFrame:
    """Every real combined hit across ``subruns``, mapped to (det, plane,
    channel, pos_mm), with amplitude and significance for the shape pass.

    Carries its own ``subrun`` column and nothing groups on ``eventId`` alone
    anywhere downstream: the DAQ's event counter **resets every subrun**
    (checked directly -- stat090_0000 and _0001 share 57 741 of 57 742
    eventIds), so a bare ``groupby('eventId')`` after concatenating subruns
    silently merges two unrelated events' hits into one fake supercluster.

    Also carries a ``clean`` column from ``ntof_tracking.reco.noise.flag_noise``
    -- the coherent-band + isolated-hit filter that is the production answer to
    exactly the problem this module hit first (see ``shape_stats``): raw
    ``combined_hits`` is dominated by a ~1.2-1.3 MHz whole-plane residual that
    lights up 50-400 strips at once, at ordinary amplitude, in one narrow time
    slice (``noise.py``'s own docstring). It is applied per SUBRUN (its
    ``groupby('eventId')`` is not collision-safe across subruns -- see below)
    before ``subrun`` is even attached, so it sees exactly what production
    sees.

    Occupancy itself (``occupancy()``) still uses every row, ``clean`` or not
    -- the honest, fully unselected count ``source_imaging.dead_ranges`` and
    ``k_robustness.hot_cells`` could not compute (both start from a fitted
    position). ``clean`` exists for the shape pass, which needs it.
    """
    import uproot
    from ntof_tracking.reco import io as rio
    from ntof_tracking.reco import noise

    cfg = rio.load_run_config(run)
    lut = rio.build_channel_lut(cfg)
    cols = ['eventId', 'feu', 'channel', 'amplitude', 'significance', 'time']
    rows = []
    for sub in subruns:
        d = paths.root('runs') / run / sub / 'combined_hits_root'
        files = [str(p) for p in sorted(d.glob('*.root')) if '_datrun_' in p.name]
        if not files:
            continue
        df = uproot.concatenate([f'{f}:hits' for f in files],
                                expressions=cols, library='pd')
        df = df.drop_duplicates(subset=['eventId', 'feu', 'channel', 'time'])
        df = rio.drop_unphysical(df, tag=f'{run}/{sub}', verbose=False)
        df = df.merge(lut, on=['feu', 'channel'], how='inner')
        df = noise.flag_noise(df)
        df['subrun'] = sub
        rows.append(df[['subrun', 'eventId', 'feu', 'det', 'plane', 'channel',
                        'pos_mm', 'amplitude', 'significance', 'clean']])
    if not rows:
        raise FileNotFoundError(
            f'no combined_hits for {run} under any of {list(subruns)}')
    return pd.concat(rows, ignore_index=True)


def track_like_event_keys(run: str, subruns, arm: str, plane: str) -> set:
    """``{(subrun, eventId), ...}`` where THIS arm's THIS plane already has at
    least one track-like hits-level cluster, per ``candidate_filter``'s own
    per-arm/per-plane segment count (``{arm}_trk_x``/``{arm}_trk_y``).

    Why this and not the coarser ``cls`` column: ``cls`` is a whole-event,
    whole-detector verdict (78 % of triggers are ``NONE`` on run_145) and
    mixes all four arms, so filtering shape stats on it would still let a
    busy OTHER arm's triggers leak in. This is per (arm, plane) instead --
    the same axis ``shape_stats`` is computed on -- and it is exactly the
    thing `wft.seed` itself would find, already computed and on disk, so
    reusing it needs no extra clustering pass.
    """
    keys = set()
    for sub in subruns:
        p = paths.out('stage1') / f'candidates_{run}_{sub}.parquet'
        if not p.exists():
            continue
        col = f'{arm}_trk_{plane}'
        d = pd.read_parquet(p, columns=['eventId', col])
        ev = d.loc[d[col] >= 1, 'eventId'].to_numpy()
        keys.update((sub, int(e)) for e in ev)
    return keys


# --------------------------------------------------------------------------- #
# Occupancy -> dead / hot
# --------------------------------------------------------------------------- #

def occupancy(hits: pd.DataFrame, det: str, plane: str) -> np.ndarray:
    """Raw hit count per channel (0..511), no selection."""
    g = hits[(hits.det == det) & (hits.plane == plane)]
    return np.bincount(g.channel.to_numpy(dtype=int),
                       minlength=N_STRIPS)[:N_STRIPS].astype(float)


def local_median_masked(values: np.ndarray, mask: np.ndarray,
                        window: int = WINDOW_STRIPS,
                        min_n: int = MIN_NEIGHBOURS) -> np.ndarray:
    """Median of ``values[mask]`` in a +-window/2 neighbourhood of each index.

    Falls back to the global median over ``mask`` when a window has fewer
    than ``min_n`` valid neighbours (a desert wider than one window, or a
    plane with almost nothing flagged 'good' yet) -- the same trap
    ``k_robustness.hot_cells`` guards against for its 2D median.
    """
    n = len(values)
    half = window // 2
    valid = mask & np.isfinite(values)
    global_med = float(np.median(values[valid])) if valid.any() else np.nan
    out = np.full(n, global_med)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        vm = valid[lo:hi]
        if vm.sum() >= min_n:
            out[i] = float(np.median(values[lo:hi][vm]))
    return out


def classify_occupancy(occ: np.ndarray, window: int = WINDOW_STRIPS,
                       hot_factor: float = HOT_FACTOR,
                       dead_thresh: float = DEAD_THRESH) -> tuple:
    """('dead'/'hot'/'good' per channel, the local-median reference used)."""
    med = local_median_masked(occ, occ > 0, window)
    cls = np.full(len(occ), 'good', dtype=object)
    cls[occ == 0] = 'dead'
    finite = np.isfinite(med)
    cls[finite & (occ > 0) & (occ < dead_thresh * med)] = 'dead'
    cls[finite & (occ > hot_factor * med)] = 'hot'
    return cls, med


# --------------------------------------------------------------------------- #
# Shape -> noisy (ordinary rate, wrong shape)
# --------------------------------------------------------------------------- #

def shape_stats(hits: pd.DataFrame, det: str, plane: str,
                event_keys: set | None = None) -> tuple:
    """Per-channel median (cluster width, charge density) it joins, and how
    many clusters that is based on -- from the hits-level largest spatial
    cluster of each event (``wft.seed``, production significance floor +
    12 mm gap clustering, largest cluster kept). No waveform fit anywhere.

    ``event_keys`` restricts to a given ``{(subrun, eventId), ...}`` set
    (see ``track_like_event_keys``). ``hits.clean`` (``noise.flag_noise``)
    restricts to non-coherent-band, non-isolated hits. **Both matter, and
    neither alone is enough**: measured on A/x here, without them the median
    hits-level cluster "width" is 45-110 strips (nearly a quarter of the
    plane) even after ``wseed``'s own significance floor, because that floor
    is RELATIVE to the event's own max and does nothing against a coherent
    band, where hundreds of strips share the same modest amplitude -- one
    inspected event lit up strips over the full 57-509 channel range at
    amplitude 10-200 throughout, textbook coherent-band noise
    (``noise.py``'s docstring), and the 12 mm gap threshold merged all of it
    into one supercluster. That inflated baseline would bury any real shape
    anomaly a genuinely noisy channel adds on top of it.
    """
    from wft import seed as wseed
    g = hits[(hits.det == det) & (hits.plane == plane) & hits.clean]
    if event_keys is not None:
        keep = pd.Series(list(zip(g.subrun, g.eventId)), index=g.index).isin(event_keys)
        g = g[keep]
    g = wseed.apply_significance_floor(g, wseed.SIG_REL_FLOOR)

    widths = [[] for _ in range(N_STRIPS)]
    dens = [[] for _ in range(N_STRIPS)]
    for _, ev in g.groupby(['subrun', 'eventId'], sort=False):
        s = wseed.seed_candidates(ev.pos_mm.to_numpy(),
                                  ev.channel.to_numpy(dtype=int),
                                  ev.amplitude.to_numpy(), n_candidates=1)
        if not s:
            continue
        cl = s[0]
        d = cl.amp_sum / cl.n_strips
        for ch in cl.channels:
            widths[ch].append(cl.n_strips)
            dens[ch].append(d)
    med_w = np.array([np.median(w) if w else np.nan for w in widths])
    med_d = np.array([np.median(d) if d else np.nan for d in dens])
    n_cl = np.array([len(w) for w in widths], dtype=int)
    return med_w, med_d, n_cl


def classify_shape(cls_occ: np.ndarray, med_w: np.ndarray, med_d: np.ndarray,
                   n_cl: np.ndarray, window: int = WINDOW_STRIPS,
                   width_factor: float = NOISY_WIDTH_FACTOR,
                   density_factor: float = NOISY_DENSITY_FACTOR,
                   min_clusters: int = MIN_SHAPE_CLUSTERS) -> np.ndarray:
    """Which currently-'good' channels have an anomalous cluster shape."""
    good = (cls_occ == 'good') & (n_cl >= min_clusters)
    w_ref = local_median_masked(med_w, good, window)
    d_ref = local_median_masked(med_d, good, window)
    noisy = (good & np.isfinite(med_w) & np.isfinite(med_d)
            & np.isfinite(w_ref) & np.isfinite(d_ref)
            & (med_w > width_factor * w_ref)
            & (med_d < density_factor * d_ref))
    return noisy


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def classify_run(run: str, subruns) -> pd.DataFrame:
    hits = load_hits(run, subruns)
    rows = []
    for arm in ARMS:
        det = f'mx17_{arm}'
        for plane in PLANES:
            occ = occupancy(hits, det, plane)
            cls, occ_ref = classify_occupancy(occ)
            keys = track_like_event_keys(run, subruns, arm, plane)
            med_w, med_d, n_cl = shape_stats(hits, det, plane, event_keys=keys)
            noisy = classify_shape(cls, med_w, med_d, n_cl)
            cls = cls.copy()
            cls[noisy] = 'noisy'
            for ch in range(N_STRIPS):
                rows.append(dict(
                    arm=arm, plane=plane, channel=ch,
                    pos_mm=float(ch * STRIP_MM - 398.58 / 2.0 + STRIP_MM / 2.0),
                    occupancy=float(occ[ch]), local_median_occ=float(occ_ref[ch]),
                    med_width=float(med_w[ch]) if np.isfinite(med_w[ch]) else np.nan,
                    med_density=float(med_d[ch]) if np.isfinite(med_d[ch]) else np.nan,
                    n_shape_clusters=int(n_cl[ch]),
                    cls=cls[ch]))
    return pd.DataFrame(rows)


def _bands(flagged_channels: np.ndarray) -> list:
    """Contiguous channel ranges -> [(ch_lo, ch_hi, mm_lo, mm_hi), ...]."""
    out, i = [], 0
    idx = np.where(flagged_channels)[0]
    if idx.size == 0:
        return out
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    for r in runs:
        lo, hi = int(r[0]), int(r[-1]) + 1
        out.append((lo, hi, lo * STRIP_MM, hi * STRIP_MM))
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, plane), g in df.groupby(['arm', 'plane']):
        n = len(g)
        rows.append(dict(
            arm=arm, plane=plane,
            frac_dead=float((g.cls == 'dead').mean()),
            frac_hot=float((g.cls == 'hot').mean()),
            frac_noisy=float((g.cls == 'noisy').mean()),
            n_hot_bands=len(_bands((g.cls == 'hot').to_numpy())),
            hits_in_hot=float(g.loc[g.cls == 'hot', 'occupancy'].sum()
                              / max(g.occupancy.sum(), 1))))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subruns',
                    default='stat090_0000,stat090_0001,stat090_0002')
    a = ap.parse_args()
    subs = [s for s in a.subruns.split(',') if s]

    df = classify_run(a.run, subs)
    S = summarise(df)
    od = paths.out('noisy_channels')
    df.to_csv(od / f'noisy_channels_{a.run}.csv', index=False)
    S.to_csv(od / f'noisy_channels_summary_{a.run}.csv', index=False)
    json.dump(dict(schema=SCHEMA, run=a.run, subruns=subs,
                   window_strips=WINDOW_STRIPS, hot_factor=HOT_FACTOR,
                   dead_thresh=DEAD_THRESH, noisy_width_factor=NOISY_WIDTH_FACTOR,
                   noisy_density_factor=NOISY_DENSITY_FACTOR,
                   min_shape_clusters=MIN_SHAPE_CLUSTERS,
                   connector_strips=CONNECTOR_STRIPS, strip_mm=STRIP_MM),
              open(od / f'noisy_channels_{a.run}.meta.json', 'w'), indent=1)

    print('SUMMARY -- fraction of the 512 channels flagged, per arm/plane')
    print(S.round(4).to_string(index=False))

    print('\nHOT BANDS -- contiguous flagged channel ranges (connector = 64 ch = 49.8 mm)')
    for (arm, plane), g in df.groupby(['arm', 'plane']):
        bands = _bands((g.cls == 'hot').to_numpy())
        if not bands:
            continue
        print(f'  {arm}-{plane}:')
        for lo, hi, mlo, mhi in bands:
            on_conn = (lo % CONNECTOR_STRIPS < 4) or (hi % CONNECTOR_STRIPS < 4)
            print(f'    ch [{lo:3d},{hi:3d})  u [{mlo:6.1f},{mhi:6.1f}) mm'
                 f'  {"<- connector boundary" if on_conn else ""}')

    print('\nNOISY (shape-only) BANDS')
    for (arm, plane), g in df.groupby(['arm', 'plane']):
        bands = _bands((g.cls == 'noisy').to_numpy())
        if not bands:
            continue
        print(f'  {arm}-{plane}:')
        for lo, hi, mlo, mhi in bands:
            print(f'    ch [{lo:3d},{hi:3d})  u [{mlo:6.1f},{mhi:6.1f}) mm')

    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
