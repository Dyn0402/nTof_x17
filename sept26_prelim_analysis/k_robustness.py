#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k_robustness.py -- is the angle scale a property of the chamber, or of the
sample it was measured on?

WHY THIS EXISTS.  `k_arm.py` certifies a chamber when its three estimators
agree and reproduce between sub-runs.  Neither of those catches a bias that is
*common* to all three, and chamber D has three candidates for exactly that:

  * **a dead surface** -- ~130 of 512 x channels, 23 % of the plane, all on one
    side (`source_imaging.dead_ranges`);
  * **hot cells** -- regions firing far above the plane's own median, which are
    noise rather than tracks and whose fitted angles are meaningless;
  * **an outer ring** -- clusters within 20 mm of the plane edge, where the
    charge column is truncated by the edge and the fit has less to work with.
    A large fraction of D's clusters sit there.

Each of those changes *which tracks* the estimators see.  If k moves when they
are removed, k was a property of the contamination.  If it does not, the
certification survives -- and that is a result, not a formality.

THE VARIANTS ARE NESTED ON PURPOSE.  ``clean`` applies all three at once, so
the comparison ``baseline`` vs ``clean`` is the whole question in one line; the
single-cut rows say which of the three, if any, did the moving.

WHAT WOULD FALSIFY A CERTIFICATION.  Stated before the numbers, so it cannot be
chosen afterwards: **k moving by more than the focus plateau half-width** under
any variant.  The plateau is already the range the data cannot separate, so a
shift inside it is not a measurement of anything; a shift outside it means the
sample, not the chamber, was setting k.

    python -m sept26_prelim_analysis.k_robustness --run run_145
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
from sept26_prelim_analysis import k_arm as K  # noqa: E402

SCHEMA = 'sept26_prelim/k_robustness/1'
ARMS = ('A', 'B', 'C', 'D')

#: Half-width of the strip map, mm.  Beyond ``EDGE_MM`` of it is "the ring".
STRIP_HALF = 199.29
EDGE_MM = 20.0
#: A cell is hot if its occupancy exceeds this multiple of the plane median.
HOT_FACTOR = 5.0
HOT_CELL_MM = 10.0

VARIANTS = ('baseline', 'no_ring', 'no_dead', 'no_hot', 'no_hotstrip', 'clean')

#: ``no_hotstrip`` is ``no_hot`` done at the right granularity, and it is here
#: to be compared against it rather than to replace it.
#:
#: ``no_hot`` is a 2D cell mask on the FITTED position, so it is post-fit and
#: partly circular -- a fit displaced by a bad channel is judged by where it
#: landed. ``no_hotstrip`` cuts on the hot content of the event's cluster in
#: the raw hits (`hot_seed_strata.py`), before any fit: the trigger is dropped
#: when the largest cluster the seeder would have taken is made ENTIRELY of
#: channels the per-strip classifier flagged. On D that is 29 % of triggers,
#: sitting on 42 channels that run ~35x their neighbours' occupancy.
#:
#: It is deliberately NOT folded into ``clean``: that variant's numbers are
#: already reported, and changing what it means silently would make the two
#: incomparable.
#:
#: **This variant is now the production default** (``k_arm.coincident_tracks``
#: applies it), which is exactly why ``measure()`` below asks that function for
#: the UNCUT sample: a baseline that already had the cut in it could not
#: measure the cut.
#:
#: Measured on run_145, 2026-09-08 -- it does most of ``no_hot``'s work for a
#: sixth of the sample, and unlike ``no_hot`` it is a no-op where it should be:
#:
#:   arm  variant      removed        k shift   spread          repro
#:   D    no_hot        22.4 %        3.76 %    0.103 -> 0.056  0.059 -> 0.029
#:   D    no_hotstrip    3.6 %        0.55 %    0.103 -> 0.087  0.059 -> 0.029
#:   A    no_hot         1.2 %        0.02 %    0.089 -> 0.103  0.014 -> 0.043
#:   A    no_hotstrip    0.0 %        0.00 %    unchanged       unchanged
#:   C    no_hot         4.1 %        0.94 %    0.128 -> 0.136  0.048 -> 0.039
#:   C    no_hotstrip    0.0 %        0.00 %    unchanged       unchanged
#:
#: HANDOFF_D_NOISY_CHANNELS.md Sec. 3.3 set "A and C do not move" as a success
#: criterion before any of this was measured. ``no_hot`` does not meet it --
#: it shifts C by ~1 % and makes A's estimator spread and reproducibility
#: WORSE -- because a 10 mm post-fit cell mask on a plane whose illumination
#: is genuinely two-lobed removes real tracks. Cutting on the raw per-strip
#: classification instead touches nothing at all in A, B or C: those chambers
#: have no all-hot triggers to drop (B and C have literally zero in either
#: plane; see hot_seed_strata.py).
#:
#: What the cut IS lives in ``hot_seed_strata`` (``DROP_STRATA``,
#: ``DROP_PLANES``, ``dropped_events``) and is not restated here -- this module
#: measures it, so a second spelling of it here could drift from the one
#: production actually applies and the measurement would quietly stop being
#: about the cut.


def hot_cells(x_p0: np.ndarray, y_p0: np.ndarray) -> tuple:
    """2D cells firing far above the plane's own median.

    Returned as (edges_u, edges_v, mask) in RAW strip coordinates, so the
    caller converts the same way it converts positions.  The median is taken
    over OCCUPIED cells only: an empty cell is a dead one, and letting it into
    the median would drag the threshold down and call ordinary cells hot.
    """
    e = np.arange(0.0, 398.58 + HOT_CELL_MM, HOT_CELL_MM)
    H, _, _ = np.histogram2d(x_p0, y_p0, bins=[e, e])
    occ = H[H > 0]
    if occ.size == 0:
        return e, e, np.zeros_like(H, bool)
    return e, e, H > HOT_FACTOR * np.median(occ)


class Masks:
    """The three exclusions for one arm, in the arm's own local frame."""

    def __init__(self, run: str, subruns, arm: str, merged_dir: str):
        from sept26_prelim_analysis import source_imaging as SI
        from ntof_tracking import run145_target_imaging as TI
        xs, ys = [], []
        for sub in subruns:
            p = paths.require(os.path.join(merged_dir, sub, f'mx17_{arm}',
                                           'events_prelim.parquet'),
                              f'full pass for {arm}/{sub}')
            d = pd.read_parquet(p, columns=['x_p0', 'y_p0'])
            xs.append(d.x_p0.to_numpy())
            ys.append(d.y_p0.to_numpy())
        x, y = np.concatenate(xs), np.concatenate(ys)
        ok = np.isfinite(x) & np.isfinite(y)
        self.eu, self.ev, self.hot = hot_cells(x[ok], y[ok])
        self.dead = SI.dead_ranges(x[ok])
        self.sign = TI.IN_PLANE_SIGN
        self.half = TI.STRIP_MAP_HALF
        self.hot_frac = float(self.hot.sum()) / max(self.hot.size, 1)
        self.dead_mm = float(sum(b - a for a, b in self.dead))
        self.hotstrip = self._hotstrip_events(run, arm)

    @staticmethod
    def _hotstrip_events(run: str, arm: str) -> dict:
        """The production cut's own definition -- ``{subrun: set(event_id)}``.

        Empty (and the variant a no-op) when the strata table has not been
        built for this arm; ``no_hotstrip`` then reports ``removed = 0``
        rather than silently masking nothing under a name that implies it did.
        """
        from sept26_prelim_analysis.hot_seed_strata import dropped_events
        return dropped_events(run, arm)

    def _raw(self, xl, yl):
        return self.sign * xl + self.half, self.sign * yl + self.half

    def keep(self, xl, yl, variant: str, event_id=None,
             subrun: str = '') -> np.ndarray:
        m = np.ones(len(xl), bool)
        rx, ry = self._raw(xl, yl)
        if variant == 'no_hotstrip':
            drop = self.hotstrip.get(subrun)
            if event_id is None or not drop:
                return m
            return ~np.isin(np.asarray(event_id), list(drop))
        if variant in ('no_ring', 'clean'):
            m &= (np.abs(xl) < STRIP_HALF - EDGE_MM) & \
                 (np.abs(yl) < STRIP_HALF - EDGE_MM)
        if variant in ('no_dead', 'clean'):
            for lo, hi in self.dead:
                m &= ~((rx >= lo) & (rx < hi))
        if variant in ('no_hot', 'clean'):
            i = np.clip(np.digitize(rx, self.eu) - 1, 0, self.hot.shape[0] - 1)
            j = np.clip(np.digitize(ry, self.ev) - 1, 0, self.hot.shape[1] - 1)
            m &= ~self.hot[i, j]
        return m


def _subset(S: dict, m: np.ndarray) -> dict:
    out = dict(S)
    for k in ('xl', 'yl', 'tx', 'ty', 'q', 'event_id'):
        if k in S:
            out[k] = S[k][m]
    return out


def measure(run: str, subruns, merged_dir: str) -> tuple:
    """k under every variant, per arm, plus what each variant removed."""
    from ntof_tracking.reco import geometry as G
    cfg = json.loads((paths.root('runs') / run / 'run_config.json').read_text())
    trs = G.detector_transforms(cfg)

    masks = {a: Masks(run, subruns, a, merged_dir) for a in ARMS}
    per_sub = {(a, v): {} for a in ARMS for v in VARIANTS}
    kept = []
    for sub in subruns:
        for a in ARMS:
            try:
                # the UNCUT sample on purpose -- see no_hotstrip's note above
                S = K.coincident_tracks(run, sub, a, merged_dir,
                                        drop_hotstrip=False)
            except FileNotFoundError:
                continue
            for v in VARIANTS:
                m = masks[a].keep(S['xl'], S['yl'], v, S.get('event_id'), sub)
                kept.append(dict(subrun=sub, arm=a, variant=v,
                                 n_in=int(len(m)), n_out=int(m.sum()),
                                 removed=float(1 - m.mean())))
                if m.sum() < 200:
                    continue
                Sv = _subset(S, m)
                sc = K.focus_scan(Sv, trs[f'mx17_{a}'])
                per_sub[(a, v)][sub] = K.estimators(Sv, sc)

    rows = []
    for a in ARMS:
        base = None
        for v in VARIANTS:
            ps = per_sub[(a, v)]
            if not ps:
                rows.append(dict(arm=a, variant=v, k=np.nan,
                                 verdict='NO DATA'))
                continue
            r = K.combine(ps)
            if v == 'baseline':
                base = r
            pl = None
            # the falsification threshold, from the BASELINE plateau -- fixed
            # before any variant is looked at
            if base and base.get('k'):
                halfw = 0.5 * (max(base['per_estimator'].values())
                               - min(base['per_estimator'].values()))
            else:
                halfw = np.nan
            rows.append(dict(
                arm=a, variant=v, n_subruns=len(ps),
                k=r.get('k'),
                # NOT the published verdict: k_arm downgrades a chamber to
                # PROVISIONAL on the FOCUS PLATEAU width, which is applied in
                # k_arm.build and not in combine().  A, C and D are all
                # PROVISIONAL there and stay so; this column is the estimator
                # agreement alone, which is what a sample cut can move.
                verdict_estimators=r.get('verdict'),
                spread=r.get('spread'), repro=r.get('repro'),
                v_insitu=r.get('v_insitu'),
                **{f'k_{e}': (r.get('per_estimator') or {}).get(e)
                   for e in K.ESTIMATORS},
                shift=(abs(r['k'] - base['k']) if base and base.get('k')
                       and r.get('k') else np.nan),
                shift_pct=(100 * abs(r['k'] / base['k'] - 1)
                           if base and base.get('k') and r.get('k')
                           else np.nan),
                tolerance=halfw,
                # meaningless for an arm with no certification to survive
                survives=(bool(abs(r['k'] - base['k']) <= halfw)
                          if (base and base.get('k') and r.get('k')
                              and np.isfinite(halfw)
                              and base.get('verdict') != 'NOT CALIBRATED')
                          else None)))
    surf = pd.DataFrame([dict(arm=a, dead_mm=masks[a].dead_mm,
                              dead_frac_plane=masks[a].dead_mm / 398.58,
                              hot_cell_frac=masks[a].hot_frac)
                         for a in ARMS])
    return pd.DataFrame(rows), pd.DataFrame(kept), surf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subruns',
                    default='stat090_0000,stat090_0001,stat090_0002')
    a = ap.parse_args()
    subs = [s for s in a.subruns.split(',') if s]
    merged = str(paths.out('fullpass') / a.run)

    R, K_, S = measure(a.run, subs, merged)
    od = paths.out('kcal')
    R.to_csv(od / f'k_robustness_{a.run}.csv', index=False)
    K_.to_csv(od / f'k_robustness_samples_{a.run}.csv', index=False)
    S.to_csv(od / f'k_robustness_surface_{a.run}.csv', index=False)
    json.dump(dict(schema=SCHEMA, run=a.run, subruns=subs,
                   edge_mm=EDGE_MM, hot_factor=HOT_FACTOR,
                   hot_cell_mm=HOT_CELL_MM,
                   criterion='k must move by less than the baseline '
                             'estimator half-spread under every variant'),
              open(od / f'k_robustness_{a.run}.meta.json', 'w'), indent=1)

    print('SURFACE -- what each variant has to work with')
    print(S.to_string(index=False))
    print('\nWHAT EACH VARIANT REMOVES FROM THE COINCIDENT SAMPLE (mean over sub-runs)')
    print(K_.groupby(['arm', 'variant']).removed.mean().unstack()
          .reindex(columns=VARIANTS).round(3).to_string())
    print('\nk UNDER EACH VARIANT')
    print(R[['arm', 'variant', 'k', 'shift_pct', 'tolerance', 'survives',
             'spread', 'repro']].to_string(index=False))
    print('\n(`survives` is blank for B: it has no certification to survive.)')
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
