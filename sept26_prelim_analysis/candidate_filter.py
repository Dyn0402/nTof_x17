#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 -- the candidate filter.  ``PLAN.md`` §3, stage 1.

Assigns **one terminal class to every DREAM trigger**, from `combined_hits` and
the n_TOF slim alone -- no waveform fitting.  That asymmetry is the whole design
of the week: reconstructing 25.6 M triggers × 4 arms blind is of order 10⁵
core-hours, and this is what decides which small fraction is worth it.

    python -m sept26_prelim_analysis.candidate_filter --run run_145 \\
        --subrun stat090_0000 [--tag 260805_14H06_000] [--bench]

Output: ``candidates.parquet``, one row per trigger.  It is a **partition** --
every trigger gets exactly one class -- which is what makes it an honest
denominator for every efficiency downstream.

| class | definition |
|---|---|
| ``INTER``   | track-like activity in exactly 2 arms -- the signal topology |
| ``INTRA``   | ≥ 2 *separated* track-like clusters in 1 arm -- the IPC control |
| ``IMPLIED`` | track-like activity in 1 arm, n_TOF arm coincidence in ≥ 2 |
| ``SINGLE``  | track-like activity in exactly 1 arm, nothing else |
| ``BUSY``    | ≥ 3 arms track-like, or > 120 clean strips in ≥ 3 arms |
| ``NONE``    | nothing track-like |

**Precedence is BUSY > INTER > INTRA > IMPLIED > SINGLE > NONE**, and it has to
be stated because the definitions are not disjoint on their face: a 2-arm event
where one arm also holds two separated clusters is ``INTER``, not ``INTRA``.
INTER and INTRA are then separated purely by how many arms are lit, so the only
real ordering decisions are BUSY first (pile-up is vetoed before anything is
believed) and IMPLIED before SINGLE (the n_TOF evidence upgrades a lone track).

What this stage is and is not
-----------------------------
It is **candidate finding**, which is what `combined_hits` is legitimately for
(``../RECONSTRUCTION_BASIS.md``).  It does **no** geometry: no X↔Y pairing, no
3D segments, no angles, no drift depth.  The per-strip hit time is an aggregate
of neighbouring strips' charge and compresses the drift ladder by 20-30 %, so
any position or angle built from it is wrong in a way no threshold fixes.  All
this stage asks of a cluster is *does it look like a track*, which the existing
taxonomy in ``reco/segments.py`` already answers.

``reco.search.sift_events`` is the near relative of this module and deliberately
**not** reused: it produces a continuous ranking score rather than a partition,
and it does do 3D pairing and global geometry per event, which is both the wrong
basis here and far too slow at campaign scale.

Known unmeasured: **this filter's own efficiency.**  How many real pairs
``NONE`` and ``SINGLE`` swallow is not known, and cannot be known from this
stage.  The prescaled control sample in stage 2 is what measures it (D13).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ntof_tracking.reco import io as rio, noise, segments as segmod   # noqa: E402

try:
    from . import paths
except ImportError:                                    # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paths                                       # type: ignore

ARMS = ('A', 'B', 'C', 'D')

#: The n_TOF slim's ``det`` code is the index into this tuple.  Single source:
#: ``ntof_processing/slim_pipeline/config.py::SCINT_TREES``.
SCINT_TREES = ('WALA', 'WALB', 'WALC', 'WALD',
               'PSSA', 'PSSB', 'PSSC', 'PSSD',
               'LIQA', 'LIQB', 'LIQC', 'LIQD')
DET_CODE = {t: i for i, t in enumerate(SCINT_TREES)}

#: Coincidence window on the slim's ``dt_ns``.  The same 25 ns the pulse ledger
#: and the DAQ's own N1081B use -- ``slim_pipeline/config.py::ACCEPT_NS``.
ACCEPT_NS = 25.0

#: Two track-like clusters in one plane count as resolved when the GAP between
#: their strip intervals exceeds this.  Below it they merge into one bad-chi2
#: cluster that the reconstruction cannot separate either (D1), so this is the
#: same number in both places on purpose.
SEP_MM = 12.0

#: Clean strips in one arm above which the arm is 'busy' -- a discharge or dense
#: pile-up, where any line through the hits is accidental.  From
#: ``reco/search.py::BUSY_DET_STRIPS``.
BUSY_ARM_STRIPS = 120
BUSY_MIN_ARMS = 3


# --------------------------------------------------------------------------- #
# The n_TOF side -- read straight off the slim, no clock fit needed
# --------------------------------------------------------------------------- #
def arm_flags(slim_root: Path, accept_ns: float = ACCEPT_NS) -> pd.DataFrame:
    """Per-trigger n_TOF evidence: one row per ``eventId`` in the slim.

    An arm is in **coincidence** when the same trigger has a wall hit *and* a
    plastic hit from that arm inside ``±accept_ns``.  That is the physical
    question the DAQ trigger asks, and it is deliberately not "did the offline
    N1081B emulation rebuild it" -- 99.5 % of the triggers the emulator calls
    unmatched do have both legs inside the window, so requiring the emulation
    would import several per cent of inefficiency for nothing
    (``slim_pipeline/coincidence_arbiter.py::pulse_coincidence``).

    The weaker tiers are kept beside it: ``wall_<arm>`` and ``plastic_<arm>``
    alone, and ``liq_<arm>``, because a wall-only arm is still evidence and
    ``IMPLIED`` needs to be able to say which tier it rests on.
    """
    import uproot

    with uproot.open(slim_root) as f:
        ev = f['events'].arrays(
            ['eventId', 'bunch', 't_dream_ns', 'is_flash', 'matched', 'arm'],
            library='pd')
        hi = f['hits'].arrays(['eventId', 'det', 'dt_ns'], library='pd')

    near = hi[np.abs(hi['dt_ns'].to_numpy()) <= accept_ns]
    out = ev.set_index('eventId')

    for fam, col in (('WAL', 'wall'), ('PSS', 'plastic'), ('LIQ', 'liq')):
        for a in ARMS:
            ids = np.unique(near.loc[near['det'] == DET_CODE[f'{fam}{a}'],
                                     'eventId'].to_numpy())
            out[f'{col}_{a}'] = out.index.isin(ids)

    for a in ARMS:
        out[f'coinc_{a}'] = out[f'wall_{a}'] & out[f'plastic_{a}']
    out['n_coinc_arms'] = sum(out[f'coinc_{a}'].astype(int) for a in ARMS)
    out['n_wall_arms'] = sum(out[f'wall_{a}'].astype(int) for a in ARMS)
    return out.reset_index()


_ARM_FLAG_CACHE: dict[str, pd.DataFrame | None] = {}


def _arm_flags_cached(ntof_dir: Path):
    """:func:`arm_flags` memoised per sub-run.

    The slim is one ~55 MB ROOT file per sub-run and its per-trigger flags are
    the same for every file tag in it, so parsing it once per tag was seven
    reads of the same file across a :func:`run_tags` sweep.
    """
    key = str(ntof_dir)
    if key not in _ARM_FLAG_CACHE:
        slim = sorted(Path(ntof_dir).glob('*.root'))
        _ARM_FLAG_CACHE[key] = arm_flags(slim[0]) if slim else None
    return _ARM_FLAG_CACHE[key]


# --------------------------------------------------------------------------- #
# The hot-channel mask -- PLAN.md §3 stage 1 step 1
# --------------------------------------------------------------------------- #
#: A channel is hot when it fires in more than this fraction of triggers *and*
#: more than HOT_K times its own plane's median.  Both, not either: the floor
#: stops a very quiet plane from masking its own ordinary strips, and the
#: relative term stops a busy one from keeping genuinely pathological channels.
HOT_FLOOR = 0.08
HOT_K = 8.0
#: Events sampled to measure the occupancy.  A few thousand is plenty -- the
#: channels this catches fire in tens of per cent of triggers.
HOT_SAMPLE_EVENTS = 1500


def hot_channel_mask(hits: pd.DataFrame, floor: float = HOT_FLOOR,
                     k: float = HOT_K,
                     sample_events: int = HOT_SAMPLE_EVENTS) -> pd.DataFrame:
    """Channels firing far too often to be physics, measured from the data.

    A track illuminates any given strip in a small fraction of triggers, so a
    plane's per-channel occupancy is a tight distribution — measured on
    run_145, the per-plane **median is 1.1-2.5 % on all four arms**. What
    separates the arms is the tail: arm D has 26 channels above 20 % occupancy
    *on each plane*, against 0-4 for A, B and C, clustered at connector
    boundaries (x 92-127, x 448-450). An excess concentrated on individual
    channels while the plane median is unremarkable is instrumental; physics
    spreads across the plane.

    Returns one row per masked channel with the occupancy that condemned it, so
    the mask is auditable rather than a bare list.

    Two honest caveats:

    * **It is measured on the same data it is applied to.** A dedicated
      pedestal/noise-run mask is D10 and does not exist yet. The circularity is
      mild — the threshold is far above anything a track produces — but it is
      real, and the mask is written out so a later D10 map can be diffed
      against it.
    * **It is per run condition**, like everything else calibrated here, and
      must be re-measured rather than carried across the 23-July noise
      boundary.
    """
    ev = hits['eventId'].unique()[:sample_events]
    sub = hits[hits['eventId'].isin(ev)]
    sub = sub[sub['clean']] if 'clean' in sub.columns else sub
    n = max(len(ev), 1)

    occ = (sub.groupby(['det', 'plane', 'channel']).eventId.nunique() / n)
    occ = occ.rename('occ').reset_index()
    out = []
    for (det, plane), g in occ.groupby(['det', 'plane']):
        med = float(g.occ.median())
        thr = max(floor, k * med)
        hot = g[g.occ > thr].copy()
        hot['plane_median'] = round(med, 5)
        hot['threshold'] = round(thr, 5)
        out.append(hot)
    mask = (pd.concat(out, ignore_index=True) if out else
            pd.DataFrame(columns=['det', 'plane', 'channel', 'occ',
                                  'plane_median', 'threshold']))
    return mask.sort_values(['det', 'plane', 'channel']).reset_index(drop=True)


def apply_hot_mask(hits: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Clear ``clean`` on masked channels.

    Nothing is dropped — the hits stay in the table with ``clean=False``, the
    same convention ``reco.noise`` uses, so a display can still show what was
    masked and why.
    """
    if not len(mask):
        return hits
    key = set(zip(mask.det, mask.plane, mask.channel))
    hot = pd.Series(list(zip(hits.det, hits.plane, hits.channel)),
                    index=hits.index).isin(key)
    hits = hits.copy()
    hits['hot_masked'] = hot
    hits['clean'] = hits['clean'] & ~hot
    return hits


# --------------------------------------------------------------------------- #
# The Micromegas side
# --------------------------------------------------------------------------- #
def _n_separated(segs, sep_mm: float = SEP_MM) -> int:
    """How many of these track-like clusters are mutually resolvable.

    Greedy over strip position: sort by lower edge and take a cluster whenever
    the GAP from the last accepted one exceeds ``sep_mm``.  The gap between
    intervals, not the centroid distance, because two long overlapping clusters
    have well-separated centroids and are still one unresolvable smear.
    """
    if len(segs) <= 1:
        return len(segs)
    iv = sorted((s['pos_lo_mm'], s['pos_hi_mm']) for s in segs)
    n, last_hi = 1, iv[0][1]
    for lo, hi in iv[1:]:
        if lo - last_hi > sep_mm:
            n += 1
            last_hi = max(last_hi, hi)
        else:
            last_hi = max(last_hi, hi)
    return n


def event_arm_summary(g_ev: pd.DataFrame, sep_mm: float = SEP_MM) -> dict:
    """Per-arm cluster census for one event's noise-flagged hits.

    Returns, per arm: clean strips, per-plane track counts, per-plane separated
    track counts, and the other three cluster classes.  Nothing here decides
    anything -- :func:`classify` does, from these numbers.
    """
    out = {}
    clean = g_ev[g_ev['clean']]
    for det, gd in clean.groupby('det', sort=False):
        arm = det[-1]
        rec = dict(n_clean=len(gd),
                   n_strips=int(gd[['plane', 'channel']].drop_duplicates().shape[0]))
        for plane in ('x', 'y'):
            gp = gd[gd['plane'] == plane]
            segs = (segmod.find_segments(gp, det, plane,
                                         int(g_ev['eventId'].iloc[0]),
                                         measure=False)
                    if len(gp) else [])
            trk = [s for s in segs if s['cls'] == 'track']
            rec[f'n_track_{plane}'] = len(trk)
            rec[f'n_sep_{plane}'] = _n_separated(trk, sep_mm)
            rec[f'n_point_{plane}'] = sum(s['cls'] == 'point' for s in segs)
            rec[f'n_blob_{plane}'] = sum(s['cls'] == 'blob' for s in segs)
            rec[f'n_bandfrag_{plane}'] = sum(
                s['cls'] == 'band_fragment' for s in segs)
            rec[f'q_{plane}'] = float(sum(s['q_sum'] for s in trk))
        out[arm] = rec
    return out


def classify(arm_rec: dict, n_coinc_arms: int, strict: bool = True) -> dict:
    """The one terminal class for a trigger, plus the numbers behind it.

    ``strict`` decides what "track-like activity in an arm" means:

    * ``True``  -- a track cluster in **both** planes.  A 3D track needs an X
      and a Y projection, so this is the definition that matches what stage 2
      can actually reconstruct, and it is the default.
    * ``False`` -- a track cluster in **either** plane.  Looser, and the right
      definition for pure tagging.

    Both counts are always recorded (``n_arms_strict`` / ``n_arms_loose``), so
    the choice can be revisited on the written table without re-running the
    stage.
    """
    lit_strict, lit_loose, busy = [], [], []
    for arm, r in arm_rec.items():
        if r['n_track_x'] >= 1 and r['n_track_y'] >= 1:
            lit_strict.append(arm)
        if r['n_track_x'] >= 1 or r['n_track_y'] >= 1:
            lit_loose.append(arm)
        if r['n_strips'] > BUSY_ARM_STRIPS:
            busy.append(arm)

    lit = lit_strict if strict else lit_loose

    # Separated track clusters, counted **inside the lit arm only**.
    #
    # It has to be the lit arm and not the best arm anywhere: INTRA means "two
    # resolvable tracks in THIS chamber", so a max over every arm would let a
    # second, unlit chamber's doubles reclassify a genuine single track as
    # INTRA -- attributing one chamber's pair to another, and inflating INTRA
    # at SINGLE's expense. Per plane and then the better of the two, because a
    # pair need only be resolvable in one projection to be countable at all.
    n_sep_best = max((max(r['n_sep_x'], r['n_sep_y'])
                      for a, r in arm_rec.items() if a in lit), default=0)
    # Kept for QA: the same quantity over every arm, so the size of what the
    # line above excludes stays measurable rather than assumed.
    n_sep_any_arm = max((max(r['n_sep_x'], r['n_sep_y'])
                         for r in arm_rec.values()), default=0)

    if len(busy) >= BUSY_MIN_ARMS or len(lit) >= 3:
        cls = 'BUSY'
    elif len(lit) == 2:
        cls = 'INTER'
    elif len(lit) == 1:
        if n_sep_best >= 2:
            cls = 'INTRA'
        elif n_coinc_arms >= 2:
            cls = 'IMPLIED'
        else:
            cls = 'SINGLE'
    else:
        cls = 'NONE'

    return dict(cls=cls,
                n_arms_strict=len(lit_strict), n_arms_loose=len(lit_loose),
                arms_lit=''.join(sorted(lit)),
                n_busy_arms=len(busy), busy_arms=''.join(sorted(busy)),
                n_sep_best=n_sep_best, n_sep_any_arm=n_sep_any_arm)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_subrun(run: str, subrun: str, tag: str | None = None,
               strict: bool = True, limit: int | None = None,
               no_mask: bool = False, mask: pd.DataFrame | None = None,
               verbose_every: int = 2000) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Classify every trigger of one sub-run (or one file tag of it).

    Returns ``(candidates, benchmark, hot_mask)``. The mask is returned rather
    than only applied so it can be written beside the candidates -- it is a
    per-run-condition calibration and must travel with the product it shaped.

    Pass an existing ``mask`` to reuse one rather than measure a new one. That
    is what :func:`run_tags` does across a sub-run's file tags: **the mask is a
    property of the run condition, not of a file**, so measuring it per tag
    would give the same detector a slightly different mask in each one and make
    the tags non-comparable -- the exact failure mode ``../CLAUDE.md`` warns
    about for every other calibrated quantity.
    """
    base = str(paths.root('runs')) + '/'
    cfg = rio.load_run_config(run, base_path=base)
    lut = rio.build_channel_lut(cfg)

    t0 = time.time()
    if tag:
        # combined_hits carries no tag column -- the split exists only in the
        # filenames, so one tag means reading exactly that one file. Loading
        # the whole sub-run first and discarding it would be ~8x the I/O across
        # a run_tags() sweep, which is what this used to do.
        d = Path(base) / run / subrun / 'combined_hits_root'
        one = [p for p in sorted(d.glob('*.root')) if tag in p.name]
        if not one:
            raise FileNotFoundError(f'no combined_hits file for tag {tag} in {d}')
        import uproot
        hits = uproot.concatenate([f'{p}:hits' for p in one],
                                  rio.HIT_COLUMNS, library='pd')
        hits = hits.drop_duplicates(subset=['eventId', 'feu', 'channel', 'time'])
        hits = rio.drop_unphysical(hits, tag=f'{run}/{subrun}/{tag}', verbose=False)
        hits = hits.merge(lut, on=['feu', 'channel'], how='inner')
        hits = hits.sort_values(['eventId', 'det', 'plane', 'time'])
        hits = hits.reset_index(drop=True)
    else:
        hits = rio.load_subrun_hits(run, subrun, lut, base_path=base)
        if hits is None:
            raise FileNotFoundError(
                f'no combined_hits for {run}/{subrun} under {base}')
    t_load = time.time() - t0

    flags = _arm_flags_cached(Path(base) / run / subrun / 'ntof_hits')
    if flags is None:
        print(f'  !! no slim for {run}/{subrun} -- n_TOF columns will be empty, '
              f'so IMPLIED can never fire and its count is not a measurement')
    coinc = (flags.set_index('eventId')['n_coinc_arms'].to_dict()
             if flags is not None else {})

    # Limit BEFORE the noise pass, not after: flagging is ~20 % of the per-event
    # cost, and limiting after it would quietly leave that cost out of the
    # benchmark this stage exists to produce.
    ev_ids = hits['eventId'].unique()
    if limit:
        ev_ids = ev_ids[:limit]
        hits = hits[hits['eventId'].isin(ev_ids)].reset_index(drop=True)

    t0 = time.time()
    hits = noise.flag_noise(hits)
    if no_mask:
        mask = pd.DataFrame()
    elif mask is None:
        mask = hot_channel_mask(hits)
    hits = apply_hot_mask(hits, mask)
    t_noise = time.time() - t0

    t0 = time.time()
    rows = []
    for i, (ev, g) in enumerate(hits.groupby('eventId', sort=True)):
        if verbose_every and i and i % verbose_every == 0:
            el = time.time() - t0
            print(f'    {i}/{len(ev_ids)}  {i / el:.0f} ev/s')
        rec = event_arm_summary(g)
        r = classify(rec, coinc.get(int(ev), 0), strict=strict)
        r.update(eventId=int(ev), n_hits=len(g), n_clean=int(g['clean'].sum()))
        for arm in ARMS:
            a = rec.get(arm)
            r[f'{arm}_strips'] = a['n_strips'] if a else 0
            r[f'{arm}_trk_x'] = a['n_track_x'] if a else 0
            r[f'{arm}_trk_y'] = a['n_track_y'] if a else 0
            r[f'{arm}_sep_x'] = a['n_sep_x'] if a else 0
            r[f'{arm}_sep_y'] = a['n_sep_y'] if a else 0
            r[f'{arm}_pnt'] = (a['n_point_x'] + a['n_point_y']) if a else 0
            r[f'{arm}_blob'] = (a['n_blob_x'] + a['n_blob_y']) if a else 0
        rows.append(r)
    t_sift = time.time() - t0

    df = pd.DataFrame(rows)
    if flags is not None:
        keep = (['eventId', 'bunch', 'is_flash', 'matched', 'n_coinc_arms',
                 'n_wall_arms']
                + [f'coinc_{a}' for a in ARMS] + [f'wall_{a}' for a in ARMS]
                + [f'plastic_{a}' for a in ARMS] + [f'liq_{a}' for a in ARMS])
        df = df.merge(flags[keep], on='eventId', how='left')

    df['run'], df['subrun'], df['tag'] = run, subrun, (tag or '')
    bench = dict(run=run, subrun=subrun, tag=tag or '',
                 n_triggers=len(df), n_hits=int(len(hits)),
                 t_load_s=round(t_load, 1), t_noise_s=round(t_noise, 1),
                 t_sift_s=round(t_sift, 1),
                 # Throughput counts noise + sift, the two per-event costs.
                 # Loading is per FILE and amortises away at campaign scale.
                 ev_per_s=round(len(df) / max(t_noise + t_sift, 1e-9), 1),
                 core_hours_per_1e6_triggers=round(
                     1e6 * (t_noise + t_sift) / max(len(df), 1) / 3600.0, 1),
                 strict=strict, sep_mm=SEP_MM, accept_ns=ACCEPT_NS,
                 has_slim=flags is not None,
                 hot_channels_masked=int(len(mask)),
                 hot_floor=HOT_FLOOR, hot_k=HOT_K)
    return df, bench, mask


def list_tags(run: str, subrun: str) -> list[str]:
    """File tags of one sub-run, from the combined_hits filenames."""
    d = Path(str(paths.root('runs'))) / run / subrun / 'combined_hits_root'
    tags = set()
    for p in sorted(d.glob('*.root')):
        # Mx17_<subrun>_datrun_<TAG>_feu-combined_hits.root
        stem = p.name
        if '_datrun_' in stem:
            tags.add(stem.split('_datrun_')[1].split('_feu-')[0])
    return sorted(tags)


def run_tags(run: str, subrun: str, tags: list[str] | None = None,
             strict: bool = True, no_mask: bool = False,
             mask: pd.DataFrame | None = None,
             verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Classify every tag of a sub-run, one at a time, under ONE hot mask.

    Tag-at-a-time rather than whole-sub-run because a sub-run is ~13 M hits and
    loading it in one frame is several GB for no benefit -- nothing in stage 1
    is correlated across tags.

    The mask is measured **once**, on the first tag, and reused for the rest
    (see :func:`run_subrun`). It is measured from data either way; what this
    guarantees is that every tag of the sub-run is filtered by the *same*
    calibration, so the per-tag censuses are comparable to each other.

    Returns ``(candidates, per_tag_bench, mask)``.
    """
    tags = tags or list_tags(run, subrun)
    if not tags:
        raise FileNotFoundError(f'no combined_hits tags for {run}/{subrun}')
    frames, benches = [], []
    for i, t in enumerate(tags):
        if verbose:
            print(f'  [{i + 1}/{len(tags)}] {t}')
        df, bench, m = run_subrun(run, subrun, t, strict=strict,
                                  no_mask=no_mask, mask=mask, verbose_every=0)
        if mask is None and not no_mask:
            mask = m                      # first tag defines it for the rest
            bench['mask_measured_on'] = t
        else:
            bench['mask_measured_on'] = tags[0]
        frames.append(df)
        benches.append(bench)
        if verbose:
            c = census(df)
            got = dict(zip(c.cls, c.n))
            print(f'      {bench["n_triggers"]} triggers, '
                  f'INTER {got.get("INTER", 0)} INTRA {got.get("INTRA", 0)} '
                  f'IMPLIED {got.get("IMPLIED", 0)}, '
                  f'{bench["ev_per_s"]:.0f} ev/s')
    return (pd.concat(frames, ignore_index=True),
            pd.DataFrame(benches),
            mask if mask is not None else pd.DataFrame())


def census(df: pd.DataFrame) -> pd.DataFrame:
    """The class census -- what fraction of the sample is in each class."""
    order = ['INTER', 'INTRA', 'IMPLIED', 'SINGLE', 'BUSY', 'NONE']
    n = len(df)
    rows = []
    for c in order:
        sel = df[df.cls == c]
        rows.append(dict(cls=c, n=len(sel), frac=round(len(sel) / n, 6),
                         median_clean=int(sel.n_clean.median()) if len(sel) else 0))
    rows.append(dict(cls='(total)', n=n, frac=1.0,
                     median_clean=int(df.n_clean.median())))
    return pd.DataFrame(rows)


def arm_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-arm track rate -- the check C3 says must precede believing the census.

    Arm D seeded ~80 % of run_145's triggers in the waveform reconstruction
    against 37-39 % for A, B and C. If that asymmetry is also present here it
    will drive the ``INTER`` count, because ``INTER`` is "exactly 2 arms lit".
    """
    n = len(df)
    rows = []
    for a in ARMS:
        both = ((df[f'{a}_trk_x'] >= 1) & (df[f'{a}_trk_y'] >= 1)).sum()
        either = ((df[f'{a}_trk_x'] >= 1) | (df[f'{a}_trk_y'] >= 1)).sum()
        rows.append(dict(
            arm=a,
            frac_track_both=round(both / n, 4),
            frac_track_either=round(either / n, 4),
            frac_track_x=round((df[f'{a}_trk_x'] >= 1).sum() / n, 4),
            frac_track_y=round((df[f'{a}_trk_y'] >= 1).sum() / n, 4),
            median_strips=int(df[f'{a}_strips'].median()),
            frac_busy=round((df[f'{a}_strips'] > BUSY_ARM_STRIPS).sum() / n, 4),
            frac_coinc=(round(df[f'coinc_{a}'].fillna(False).sum() / n, 4)
                        if f'coinc_{a}' in df else None),
        ))
    return pd.DataFrame(rows)


def event_mixing_background(df: pd.DataFrame, n_rep: int = 20,
                            seed: int = 0, strict: bool = True) -> pd.DataFrame:
    """**DEFECTIVE -- DO NOT USE.** Symmetric event mixing, kept only as a record.

    This permutes *every* arm's block independently across triggers, which is
    the wrong null for this dataset and gives an answer that looks like a
    physics result and is not one.

    **Why it is wrong.** The DAQ trigger is a wall+plastic coincidence in *any
    one arm*, so **every recorded event already contains one guaranteed
    particle**. The four arms are therefore anti-correlated by construction --
    whichever arm the trigger particle went into, it did not go into the other
    three -- and the per-arm "lit" probabilities measured on this sample are
    dominated by "this is the arm that fired the trigger", not by an
    independent per-arm rate.

    Mixing all four arms destroys that constraint: mixed events can end up with
    zero trigger particles or two, at rates set by the marginals. The estimator
    then effectively predicts the second track at **p squared**, which it is
    not. The first track is guaranteed by the trigger; the second is the pair
    partner from the same interaction, whose probability per triggered event is
    a physics constant.

    Run on run_145 this produced an "accidental" 2-arm rate of 1.45 % against
    1.03 % observed, and the 29 % "deficit" was read as a physics statement. It
    was the trigger constraint, and nothing else.

    **What a correct version does:** identify the trigger arm event by event
    (the arm holding the wall+plastic coincidence -- 93.7 % of triggers have
    exactly one), hold that arm's content fixed, and mix only the *other* arms
    across events. That measures the thing actually wanted: given the trigger
    particle, how often does an unrelated track appear elsewhere. See
    :func:`accidental_second_track`.
    """
    raise NotImplementedError(
        'symmetric event mixing is the wrong null for a triggered sample -- '
        'see the docstring, and use accidental_second_track() instead')


def _mix_symmetric_UNUSED(df: pd.DataFrame, n_rep: int = 20,
                          seed: int = 0, strict: bool = True) -> pd.DataFrame:
    """The body of the defective estimator, retained for reproducibility only."""
    rng = np.random.default_rng(seed)
    cols = {a: [f'{a}_trk_x', f'{a}_trk_y', f'{a}_sep_x', f'{a}_sep_y',
                f'{a}_strips'] for a in ARMS}
    out = []
    for rep in range(n_rep):
        parts = []
        for _, g in df.groupby('tag', sort=False):
            m = g.copy()
            for a in ARMS:
                idx = rng.permutation(len(m))
                m[cols[a]] = g[cols[a]].to_numpy()[idx]
            parts.append(m)
        mixed = pd.concat(parts, ignore_index=True)

        rec = {}
        for a in ARMS:
            rec[a] = dict(zip(('n_track_x', 'n_track_y', 'n_sep_x', 'n_sep_y',
                               'n_strips'),
                              (mixed[c].to_numpy() for c in cols[a])))
        # classify() is per event; vectorise its logic here rather than loop
        # 190 k x n_rep times. Kept deliberately parallel to classify() -- if
        # one changes the other must.
        lit = {a: (rec[a]['n_track_x'] >= 1) & (rec[a]['n_track_y'] >= 1)
               if strict else
               (rec[a]['n_track_x'] >= 1) | (rec[a]['n_track_y'] >= 1)
               for a in ARMS}
        nlit = sum(lit[a].astype(int) for a in ARMS)
        nbusy = sum((rec[a]['n_strips'] > BUSY_ARM_STRIPS).astype(int)
                    for a in ARMS)
        sep_lit = np.zeros(len(mixed), dtype=int)
        for a in ARMS:
            s_a = np.maximum(rec[a]['n_sep_x'], rec[a]['n_sep_y'])
            sep_lit = np.maximum(sep_lit, np.where(lit[a], s_a, 0))
        ncoinc = (mixed['n_coinc_arms'].fillna(0).to_numpy()
                  if 'n_coinc_arms' in mixed else np.zeros(len(mixed)))

        cls = np.full(len(mixed), 'NONE', dtype=object)
        one = nlit == 1
        cls[one & (ncoinc >= 2)] = 'IMPLIED'
        cls[one & (ncoinc < 2)] = 'SINGLE'
        cls[one & (sep_lit >= 2)] = 'INTRA'
        cls[nlit == 2] = 'INTER'
        cls[(nbusy >= BUSY_MIN_ARMS) | (nlit >= 3)] = 'BUSY'

        n = len(mixed)
        out.append({c: float((cls == c).sum()) / n for c in
                    ('INTER', 'INTRA', 'IMPLIED', 'SINGLE', 'BUSY', 'NONE')})
    return pd.DataFrame(out)


def accidental_second_track(df: pd.DataFrame, n_rep: int = 20,
                            seed: int = 0) -> pd.DataFrame:
    """Second-arm CLUMPING, trigger arm held fixed. **Not** an accidental rate.

    The question this answers is the only one event mixing can answer on a
    triggered sample: **given the particle that fired the trigger, how often
    does a second, uncorrelated track show up in another arm?** That is the
    accidental background under `INTER`; the rest of `INTER` is the pair
    partner from the same interaction, which mixing must not be allowed to
    manufacture or destroy.

    Method, per repetition:

    * the **trigger arm** is the arm holding the wall+plastic coincidence.
      93.7 % of run_145 triggers have exactly one; events with zero or more
      than one are dropped, and the fraction dropped is reported, because for
      those the trigger arm is not identifiable and guessing it would bias the
      answer in an uncontrolled direction.
    * that arm's content is **held fixed** -- it is the real particle.
    * every *other* arm's block is permuted across events **within a file
      tag**, so each non-trigger arm keeps its own singles rate and the tag's
      conditions, and loses only its correlation with this event.

    **What it actually measures, and why that is not the accidental fraction.**
    Permuting an arm's flag among the events where it is *not* the trigger arm
    **conserves that arm's total count exactly** -- mixing only reassigns which
    event each second track lands in. So the mixed P(>=1 second arm) can differ
    from the observed one only through *clumping*, never through content.

    Measured on run_145: 11 459 non-trigger lit arms spread over 11 077 events;
    observed P(>=1) = 6.2335 %, mixed 6.3154 +- 0.0082 %. The 1.3 % gap is
    entirely the fact that real events clump two second-arms together slightly
    more often than chance (372 events with >= 2). **It is not an accidental
    rate.**

    The conclusion is structural, not a matter of tuning: **the stage-1 ledger
    holds only counts, and no permutation of counts can separate a pair partner
    from an unrelated track.** What distinguishes them is that the pair is
    *simultaneous* and points back at the target -- timing and geometry, i.e.
    the per-arm ``t0`` and direction from stage 2/3. ``PLAN.md`` §5 puts event
    mixing at stage 5 for exactly this reason; at stage 1 it is vacuous.

    Use it for what it does measure (second-arm clumping) and nothing else.

    Returns one row per repetition: ``p_second_observed`` and
    ``p_second_accidental`` -- the latter named for the quantity it was built
    to estimate, which it does not.
    """
    rng = np.random.default_rng(seed)
    need = [f'coinc_{a}' for a in ARMS]
    if any(c not in df.columns for c in need):
        raise ValueError('no n_TOF coincidence columns -- the trigger arm '
                         'cannot be identified without the slim join')

    co = np.column_stack([df[f'coinc_{a}'].fillna(False).astype(bool).to_numpy()
                          for a in ARMS])
    one = co.sum(axis=1) == 1
    d = df[one].reset_index(drop=True)
    trig = np.argmax(co[one], axis=1)          # index into ARMS
    print(f'  trigger arm identified for {one.sum():,} of {len(df):,} triggers '
          f'({100*one.mean():.1f} %); the rest have 0 or >1 arm coincidences '
          f'and are dropped')

    cols = {a: [f'{a}_trk_x', f'{a}_trk_y'] for a in ARMS}
    lit_real = np.column_stack(
        [((d[f'{a}_trk_x'] >= 1) & (d[f'{a}_trk_y'] >= 1)).to_numpy()
         for a in ARMS])
    # observed: a lit arm that is NOT the trigger arm
    notrig = np.ones_like(lit_real, dtype=bool)
    notrig[np.arange(len(d)), trig] = False
    obs = float((lit_real & notrig).any(axis=1).mean())

    out = []
    tags = d['tag'].to_numpy()
    for rep in range(n_rep):
        mixed = lit_real.copy()
        for ai, a in enumerate(ARMS):
            for t in np.unique(tags):
                m = np.flatnonzero(tags == t)
                # permute this arm's lit flag among events where it is NOT the
                # trigger arm; a trigger arm's own content is never moved.
                sel = m[trig[m] != ai]
                mixed[sel, ai] = lit_real[rng.permutation(sel), ai]
        acc = float((mixed & notrig).any(axis=1).mean())
        out.append(dict(p_second_observed=obs, p_second_accidental=acc))
    return pd.DataFrame(out)


def figure_arm_rates(masked: pd.DataFrame, unmasked: pd.DataFrame,
                     out_dir: Path, stem: str = ''):
    """What the hot-channel mask does, per arm -- the C3 figure.

    The message is that the mask is targeted: it leaves B and C untouched and
    only pulls D back into line. A figure that just showed the masked rates
    would not make that point, so both are plotted together.
    """
    try:
        from . import figstyle
    except ImportError:
        import figstyle                                  # type: ignore

    figstyle.use()
    fig, ax = figstyle.slide(figsize=figstyle.WIDE)

    x = np.arange(len(ARMS))
    w = 0.36
    um = [float(unmasked.loc[unmasked.arm == a, 'frac_track_both'].iloc[0]) * 100
          for a in ARMS]
    ms = [float(masked.loc[masked.arm == a, 'frac_track_both'].iloc[0]) * 100
          for a in ARMS]

    ax.bar(x - w / 2, um, w, color=figstyle.LINE, label='before the hot-channel mask')
    for i, a in enumerate(ARMS):
        ax.bar(x[i] + w / 2, ms[i], w, color=figstyle.DET_COLOR[a])
    # One legend entry for the four coloured bars -- they are one series.
    ax.bar([np.nan], [np.nan], w, color=figstyle.DET_COLOR['A'],
           label='after the mask')

    for i, (u, m) in enumerate(zip(um, ms)):
        ax.text(x[i] - w / 2, u + 0.15, f'{u:.1f}', ha='center', va='bottom',
                fontsize=figstyle.BASE_PT * 0.8, color=figstyle.MUTED)
        ax.text(x[i] + w / 2, m + 0.15, f'{m:.1f}', ha='center', va='bottom',
                fontsize=figstyle.BASE_PT * 0.8, color=figstyle.INK,
                fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'chamber {a}' for a in ARMS])
    ax.set_ylabel('triggers with a track\nin both planes  [%]')
    # Headroom for a two-column legend ABOVE the tallest value label: at 1.25x
    # the legend landed on chamber C's "8.8".
    ax.set_ylim(0, max(um + ms) * 1.55)
    ax.legend(loc='upper center', ncol=2)
    figstyle.title(
        ax, 'The hot-channel mask only moves chamber D',
        'B and C are unchanged to the digit; D falls by a third and its median '
        'occupancy by 4x')
    figstyle.preliminary(ax, 'lower right')
    figstyle.note(fig, 'sept26_prelim_analysis/candidate_filter.py - '
                       f'run_145 {stem} - mask measured on the same data (D10 pending)')
    tab = pd.DataFrame(dict(arm=list(ARMS), before_pct=um, after_pct=ms))
    return figstyle.save(fig, out_dir / f'arm_rates_mask{"_" + stem if stem else ""}',
                         data=tab)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subrun', default='stat090_0000')
    ap.add_argument('--tag', default=None,
                    help='one combined_hits file tag, e.g. 260805_14H06_000')
    ap.add_argument('--all-tags', action='store_true',
                    help='every tag of the sub-run, one at a time, under one '
                         'hot mask measured on the first')
    ap.add_argument('--limit', type=int, default=None,
                    help='first N triggers only -- for benchmarking')
    ap.add_argument('--loose', action='store_true',
                    help='an arm is lit on a track in EITHER plane (default: both)')
    ap.add_argument('--no-mask', action='store_true',
                    help='skip the hot-channel mask -- for measuring its effect')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    if args.all_tags:
        df, benches, mask = run_tags(args.run, args.subrun,
                                     strict=not args.loose,
                                     no_mask=args.no_mask)
        bench = dict(run=args.run, subrun=args.subrun, tag='ALL',
                     n_tags=len(benches),
                     n_triggers=int(benches.n_triggers.sum()),
                     ev_per_s=round(float(benches.n_triggers.sum() /
                                          (benches.t_noise_s + benches.t_sift_s).sum()), 1),
                     core_hours_per_1e6_triggers=round(
                         1e6 * float((benches.t_noise_s + benches.t_sift_s).sum())
                         / max(int(benches.n_triggers.sum()), 1) / 3600.0, 2),
                     hot_channels_masked=int(len(mask)),
                     mask_measured_on=str(benches.mask_measured_on.iloc[0]),
                     strict=not args.loose, sep_mm=SEP_MM, accept_ns=ACCEPT_NS)
    else:
        df, bench, mask = run_subrun(args.run, args.subrun, args.tag,
                                     strict=not args.loose, limit=args.limit,
                                     no_mask=args.no_mask)
        benches = None
    if len(mask):
        print('\nhot-channel mask (per arm: channels, strips/event removed)\n')
        g = mask.groupby('det').agg(channels=('channel', 'size'),
                                    strips_per_event=('occ', 'sum'))
        print(g.round(1).to_string())

    print('\nclass census\n')
    print(census(df).to_string(index=False))
    print('\nper-arm track rate\n')
    print(arm_rates(df).to_string(index=False))
    print('\nbenchmark\n')
    for k, v in bench.items():
        print(f'  {k:<32} {v}')

    out = args.out or paths.out('stage1')
    out.mkdir(parents=True, exist_ok=True)
    stem = f'{args.run}_{args.subrun}' + (f'_{args.tag}' if args.tag else '')
    if benches is not None:
        benches.to_csv(out / f'bench_per_tag_{stem}.csv', index=False)
        # Per-tag census: the spread across tags is the only handle stage 1 has
        # on whether the class fractions are stable within a sub-run.
        per = []
        for t, g in df.groupby('tag'):
            c = census(g).set_index('cls').frac
            per.append(dict(tag=t, n=len(g), **{k: round(float(c[k]), 5) for k in
                                                ('INTER','INTRA','IMPLIED',
                                                 'SINGLE','BUSY','NONE')}))
        pd.DataFrame(per).to_csv(out / f'census_per_tag_{stem}.csv', index=False)
        print('\nper-tag census (class fractions)\n')
        print(pd.DataFrame(per).to_string(index=False))
    df.to_parquet(out / f'candidates_{stem}.parquet', index=False)
    census(df).to_csv(out / f'census_{stem}.csv', index=False)
    arm_rates(df).to_csv(out / f'arm_rates_{stem}.csv', index=False)
    (out / f'bench_{stem}.json').write_text(json.dumps(bench, indent=1))
    mask.to_csv(out / f'hot_mask_{stem}.csv', index=False)
    print(f'\n  -> {out / f"candidates_{stem}.parquet"}  ({len(df)} rows)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
