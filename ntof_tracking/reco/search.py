#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
search.py — event sifting: find the rare events that carry genuine track
candidates among thousands of noise-dominated triggers.

Why not hit counts: the residual ~1.3 MHz common-mode bands give EVERY event
hundreds of hits; empty-but-noisy and track-bearing events overlap completely
in multiplicity. Instead each event is scored on reconstruction evidence:

    per plane:  noise-flag -> cluster -> robust line fit -> class
    per event:  best 3D pair score, else best single-plane track score

score = 0                              nothing track-like
        2 + r2                         single-plane 'track' segment (best)
        10 + 5*iou + r2x + r2y         paired X/Y 3D segment (best pair)
plus log10(q_sum)/2 of the best track (charge significance).

Physics priors (what run_48 doubles data should contain, 2026-07-17):
  * mostly SINGLE charged particles radiating OUTWARD FROM THE BEAMLINE
    (thermal region; the doubles trigger itself is largely combinatorial) —
    a 3D pair whose extrapolation passes near the beam axis (small DCA to
    the global Y axis) earns a pointing bonus, +more if it crosses the He-3
    gas volume itself;
  * tracks in >2 Micromegas at once are NOT expected (2 already extremely
    rare) — events with pairs in >=3 chambers are down-ranked as pile-up
    ('multi_det'); 2-chamber events keep full score and are flagged
    (n_pair_dets column) as rare-golden for manual inspection;
  * odd-angle tracks (cosmics, rare scatters) stay findable — the pointing
    bonus biases the RANKING, it never vetoes.

Output: one row per event with the score, its best-detector breakdown,
pointing metrics and segment counts — rank by 'score', display from the top.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import noise, segments as segmod, pairing
from . import geometry as geo


BUSY_DET_STRIPS = 120        # clean strips in one detector => busy (discharge
                             # or dense pile-up; accidental diagonals abound)
BURST_MIN_DETS = 3           # >= this many busy detectors = prompt burst
                             # (flash/shower pile-up across the setup)


def busy_detectors(hits_ev: pd.DataFrame) -> List[str]:
    """Detectors whose CLEAN hits still cover an implausible strip count —
    saturated prompt bursts / discharges, where line fits are accidental."""
    g = hits_ev[hits_ev['clean']]
    n = (g.groupby('det')
          .apply(lambda d: d[['plane', 'channel']].drop_duplicates().shape[0],
                 include_groups=False))
    return sorted(n[n > BUSY_DET_STRIPS].index.tolist())


POINTING_BONUS_MAX = 4.0     # extrapolation through the beam axis region
POINTING_DCA_SCALE = 400.0   # bonus tapers linearly to 0 at this DCA [mm]
HE3_CROSS_BONUS = 2.0        # extrapolation crosses the He-3 gas itself
MULTI_DET_FACTOR = 0.2       # >=3 chambers with pairs = pile-up, down-rank


def pointing_bonus(gseg: dict) -> float:
    """Beamline-radiating prior for one GLOBAL 3D segment."""
    b = POINTING_BONUS_MAX * max(
        0.0, 1.0 - gseg['dca_beam_axis_mm'] / POINTING_DCA_SCALE)
    if any(c['arm'] == 'target'
           for c in geo.line_crossings(gseg['p_lo_global'],
                                       gseg['dir_global'])):
        b += HE3_CROSS_BONUS
    return b


def score_event(segs: List[dict], gsegs: List[dict]) -> Dict:
    """Event-level score from per-plane segments + GLOBAL 3D pairs."""
    tracks = [s for s in segs if s['cls'] == 'track']
    best = 0.0
    best_det = ''
    kind = 'none'
    best_g = None
    if gsegs:
        for p in gsegs:
            s = 10.0 + 5.0 * p['iou'] + p['r2_x'] + p['r2_y'] \
                + 0.5 * np.log10(max(p['q_x'] + p['q_y'], 10.0)) \
                + pointing_bonus(p)
            if s > best:
                best, best_det, kind, best_g = s, p['det'], '3d_pair', p
    elif tracks:
        for t in tracks:
            s = 2.0 + t.get('r2', 0.0) + 0.5 * np.log10(max(t['q_sum'], 10.0))
            if s > best:
                best, best_det, kind = s, f"{t['det']}/{t['plane']}", '1plane'
    pair_dets = {p['det'] for p in gsegs}
    if len(pair_dets) >= 3:
        best *= MULTI_DET_FACTOR
        kind = 'multi_det'
    n_cls = {c: sum(1 for s in segs if s['cls'] == c)
             for c in ('track', 'point', 'band_fragment', 'blob')}
    return dict(score=best, best_det=best_det, kind=kind,
                n_track=n_cls['track'], n_point=n_cls['point'],
                n_bandfrag=n_cls['band_fragment'], n_blob=n_cls['blob'],
                n_pairs=len(gsegs), n_pair_dets=len(pair_dets),
                dca_beam_mm=(best_g['dca_beam_axis_mm'] if best_g else np.nan),
                vert_deg=(best_g['angle_to_vertical_deg'] if best_g else np.nan),
                he3_cross=bool(best_g and any(
                    c['arm'] == 'target' for c in geo.line_crossings(
                        best_g['p_lo_global'], best_g['dir_global']))))


def _null_row(ev, g, kind, busy=()) -> Dict:
    return dict(eventId=int(ev), score=0.0, best_det='', kind=kind,
                n_track=0, n_point=0, n_bandfrag=0, n_blob=0, n_pairs=0,
                n_pair_dets=0, dca_beam_mm=np.nan, vert_deg=np.nan,
                he3_cross=False, n_hits=len(g),
                n_clean=int(g['clean'].sum()),
                busy_dets=','.join(d[-1] for d in busy))


def sift_events(hits: pd.DataFrame, drift: geo.DriftModel,
                transforms: Dict[str, geo.DetTransform],
                min_clean_hits: int = 4,
                verbose_every: int = 200):
    """Run the reco front-end on every event. Returns (candidates, tracks):
      candidates — one row per event, descending score;
      tracks     — one row per 3D X/Y pair (all non-burst events), with the
                   global direction and beam-axis projection (dca_beam_mm,
                   beam_y_mm) for source-profile studies.
    `hits` must be the mapped output of io.load_subrun_hits (noise flags
    added here if missing)."""
    if 'clean' not in hits.columns:
        hits = noise.flag_noise(hits)
    rows = []
    trows = []
    ev_ids = hits['eventId'].unique()
    for i, (ev, g) in enumerate(hits.groupby('eventId', sort=True)):
        if verbose_every and i % verbose_every == 0:
            print(f'    sift {i}/{len(ev_ids)}')
        if g['clean'].sum() < min_clean_hits:
            rows.append(_null_row(ev, g, 'none'))
            continue
        busy = busy_detectors(g)
        if len(busy) >= BURST_MIN_DETS:
            # burst events: don't reconstruct (accidental diagonals), just tag
            rows.append(_null_row(ev, g, 'burst', busy))
            continue
        segs = segmod.segments_for_event(g)
        # drop segments (and later pairs) living in a busy detector
        segs = [s for s in segs if s['det'] not in busy]
        pairs = pairing.pair_xy_3d(segs, drift)
        gsegs = [geo.segment_to_global(p, transforms[p['det']])
                 for p in pairs]
        r = score_event(segs, gsegs)
        r.update(eventId=int(ev), n_hits=len(g),
                 n_clean=int(g['clean'].sum()),
                 busy_dets=','.join(d[-1] for d in busy))
        rows.append(r)
        for p in gsegs:
            d = p['dir_global']
            trows.append(dict(
                eventId=int(ev), det=p['det'], iou=p['iou'],
                bal_pull=p['bal_pull'], r2_x=p['r2_x'], r2_y=p['r2_y'],
                n_strips_x=p['n_strips_x'], n_strips_y=p['n_strips_y'],
                q_x=p['q_x'], q_y=p['q_y'], tan_theta=p['tan_theta'],
                dir_x=d[0], dir_y=d[1], dir_z=d[2],
                dca_origin_mm=p['dca_origin_mm'],
                dca_beam_mm=p['dca_beam_axis_mm'],
                beam_y_mm=p['beam_y_mm'],
                vert_deg=p['angle_to_vertical_deg'],
                n_pair_dets=r['n_pair_dets'],
            ))
    df = pd.DataFrame(rows).sort_values('score', ascending=False)
    return df.reset_index(drop=True), pd.DataFrame(trows)
