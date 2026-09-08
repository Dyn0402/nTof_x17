#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acceptance.py -- what fraction of pairs of a given opening angle we could see.

A straight-line toy through the as-built geometry.  It is deliberately thrown
FLAT IN OPENING ANGLE so that what comes out is an acceptance and not a
prediction: the physics is multiplied in afterwards (`pair_physics.py`), never
folded in here.

WHAT IS IN IT, in the order the chain applies it:

  1. the VERTEX, drawn by volume from the real He-3 gas polycone
     (`geometry.HE3_GAS_Y/R`), displaced by the offset the pointing crossing
     measured (`source_imaging.py`: X = -8.4, Z = -4.1 mm).  That coupling is
     the reason S2 comes before S4.
  2. the PAIR: one isotropic direction, the second at the thrown opening angle
     with a random azimuth about it.  No polarisation, no correlation with the
     capsule -- both would be physics.
  3. the CHAMBER: the strip plane's active area, in the arm's own (u, v).
  4. DEAD CHANNELS, as the u ranges `source_imaging.dead_ranges` finds in the
     data -- chamber D loses 23 % of its x plane this way and it is not
     optional.
  5. the TRIGGER: the production trigger is a wall AND plastic coincidence, and
     the two plastic bars have a gap between them, so a particle threading the
     gap makes no trigger.  Ray-traced, not assumed -- it is the same geometry
     that explains the two-lobe structure in every measured hit map.
  6. the RECONSTRUCTION EFFICIENCY, per chamber, from the scintillator-tagged
     measurement (`efficiency.py`), applied in u bins where the map has them.

WHAT IS NOT IN IT, and each of these makes the acceptance an over-estimate:
multiple scattering, energy loss, the leptons' own energies (a 1 MeV electron
does not reach the plastic the way a 10 MeV one does), pile-up, and the
double-track finding efficiency at small separations (PLAN D1/D2).  So the
acceptance-corrected spectrum is a preliminary shape and never a rate.

CHAMBER B IS A TAGGING CHAMBER.  It has no drift field, so it produces no
angle; a pair with a leg in B cannot contribute a measured opening angle.  B is
therefore counted in the trigger and excluded from every measurable topology,
which is exactly how the data treats it.

    python -m sept26_prelim_analysis.acceptance --run run_145
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

SCHEMA = 'sept26_prelim/acceptance/1'
ARMS = ('A', 'B', 'C', 'D')
#: Chambers whose angles are published.  B never is (no field cage).
ANGLE_ARMS = ('A', 'C', 'D')
#: Which arms face each other, from the geometry: A/C on +-Z, B/D on +-X.
OPPOSING = {frozenset('AC'), frozenset('BD')}
THETA_BINS = np.arange(0.0, 181.0, 5.0)


def topology(a1: str, a2: str) -> str:
    if a1 == a2:
        return 'intra'
    return 'opposing' if frozenset((a1, a2)) in OPPOSING else 'perpendicular'


def _isotropic(rng, n):
    c = rng.uniform(-1, 1, n)
    s = np.sqrt(1 - c ** 2)
    p = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([s * np.cos(p), c, s * np.sin(p)])


def _rotate_by(rng, d, theta_rad):
    """A unit vector at angle ``theta`` from ``d``, azimuth random about it."""
    n = len(d)
    # any vector not parallel to d
    t = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    bad = np.abs(d[:, 2]) > 0.9
    t[bad] = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(d, t)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(d, e1)
    phi = rng.uniform(0, 2 * np.pi, n)
    st, ct = np.sin(theta_rad), np.cos(theta_rad)
    return (ct[:, None] * d
            + st[:, None] * (np.cos(phi)[:, None] * e1
                             + np.sin(phi)[:, None] * e2))


def _vertices(rng, n, offset):
    from ntof_tracking.reco import geometry as G
    ys, rs = G.HE3_GAS_Y, G.HE3_GAS_R
    out = []
    got = 0
    while got < n:
        m = int((n - got) * 3.5) + 100
        yy = rng.uniform(ys.min(), ys.max(), m)
        rr = G.HE3_R_MAX * np.sqrt(rng.uniform(0, 1, m))
        ok = rr <= np.interp(yy, ys, rs)
        yy, rr = yy[ok], rr[ok]
        ph = rng.uniform(0, 2 * np.pi, len(yy))
        out.append(np.column_stack([rr * np.cos(ph), yy, rr * np.sin(ph)]))
        got += len(yy)
    return np.vstack(out)[:n] + np.asarray(offset, float)


class Chambers:
    """Everything about one run's four arms the toy needs, precomputed."""

    def __init__(self, run: str, subruns, merged_dir: str, eff_map=None,
                 eff_headline=None):
        from ntof_tracking.reco import geometry as G
        from ntof_tracking import run145_target_imaging as TI
        from sept26_prelim_analysis import source_imaging as SI
        self.G = G
        self.tr = SI.transforms(run)
        self.dead = {}
        for a in ARMS:
            x = SI.plane_occupancy(run, subruns, a, merged_dir)
            # dead ranges arrive in raw strip coordinates; the toy works in
            # the arm's own u, so convert once, here, the same way positions do
            self.dead[a] = [tuple(sorted(
                TI.IN_PLANE_SIGN * (np.array([lo, hi]) - TI.STRIP_MAP_HALF)))
                for lo, hi in SI.dead_ranges(x)]
        self.eff_map = eff_map
        self.eff = eff_headline or {}

    def cross(self, P, D, arm):
        """(u, v, hit_plane, hit_plastic) for each ray on one arm."""
        G = self.G
        wh, uh, vh = G.W_HAT[arm], G.U_HAT[arm], G.V_HAT
        centre = self.tr[f'mx17_{arm}'].center
        dw = D @ wh
        with np.errstate(divide='ignore', invalid='ignore'):
            s0 = (centre @ wh - P @ wh) / dw
        X0 = P + s0[:, None] * D
        u0, v0 = X0 @ uh - centre @ uh, X0 @ vh
        dp = G.PLASTIC_W0[arm] - G.W_STRIP + G.PLASTIC_THICK / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            sp = (centre @ wh + dp - P @ wh) / dw
        Xp = P + sp[:, None] * D
        up, vp = Xp @ uh - centre @ uh, Xp @ vh
        fwd = (s0 > 0) & np.isfinite(s0)
        plane = (fwd & (np.abs(u0) < G.MM_SIZE_U / 2)
                 & (np.abs(v0) < G.MM_SIZE_V / 2))
        for lo, hi in self.dead[arm]:
            plane &= ~((u0 >= lo) & (u0 <= hi))
        plastic = (fwd & (np.abs(vp) < G.PLASTIC_HALF_V)
                   & (np.abs(np.abs(up) - G.PLASTIC_U_OFFSET)
                      < G.PLASTIC_HALF_U))
        return u0, v0, plane, plastic

    def efficiency(self, arm, u):
        """Per-track reconstruction efficiency at in-plane position ``u``."""
        base = float(self.eff.get(arm, 0.0))
        if self.eff_map is None:
            return np.full(len(u), base)
        g = self.eff_map[self.eff_map.arm == arm]
        if g.empty:
            return np.full(len(u), base)
        # The map is P(track | seed) in u bins; the headline is the absolute
        # scale.  Using the map's SHAPE times the headline keeps both: the
        # absolute number stays the measured one and the variation across the
        # plane is not thrown away.
        e = np.interp(u, g.u_mid.to_numpy(), g.eff_track_given_seed.to_numpy(),
                      left=np.nan, right=np.nan)
        shape = e / np.nanmean(g.eff_track_given_seed.to_numpy())
        shape = np.where(np.isfinite(shape), shape, 0.0)
        return np.clip(base * shape, 0.0, 1.0)


def throw(ch: Chambers, n: int = 2_000_000, offset=(0.0, 0.0, 0.0),
          seed: int = 17, use_eff: bool = True) -> pd.DataFrame:
    """One row per thrown pair: its opening angle and where each leg landed."""
    rng = np.random.default_rng(seed)
    P = _vertices(rng, n, offset)
    d1 = _isotropic(rng, n)
    # FLAT IN THETA, not in cos(theta): the output is an acceptance per angle
    # bin, so every bin has to be thrown equally or the statistics follow the
    # solid angle instead of the question.
    th = rng.uniform(0.0, np.pi, n)
    d2 = _rotate_by(rng, d1, th)

    hit = {}
    for leg, D in ((1, d1), (2, d2)):
        for a in ARMS:
            u, v, plane, plastic = ch.cross(P, D, a)
            keep = plane
            if use_eff:
                keep = keep & (rng.uniform(0, 1, n) < ch.efficiency(a, u))
            hit[(leg, a)] = dict(plane=plane, plastic=plastic, reco=keep, u=u)

    # the trigger: at least one leg makes a wall+plastic coincidence in an arm
    # it also crossed.  The wall sits between the chamber and the plastic, so a
    # leg that reaches the plastic has crossed the wall.
    trig = np.zeros(n, bool)
    for leg in (1, 2):
        for a in ARMS:
            trig |= hit[(leg, a)]['plane'] & hit[(leg, a)]['plastic']

    rows = []
    for a1 in ANGLE_ARMS:
        for a2 in ANGLE_ARMS:
            if a2 < a1:
                continue
            both = (hit[(1, a1)]['reco'] & hit[(2, a2)]['reco'] & trig)
            if a1 != a2:
                both |= (hit[(1, a2)]['reco'] & hit[(2, a1)]['reco'] & trig)
            if not both.any():
                continue
            rows.append(pd.DataFrame(dict(
                theta=np.degrees(th[both]), arm1=a1, arm2=a2,
                topology=topology(a1, a2))))
    thrown = pd.DataFrame(dict(theta=np.degrees(th)))
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), thrown


def acceptance_curves(acc: pd.DataFrame, thrown: pd.DataFrame,
                      bins=THETA_BINS) -> pd.DataFrame:
    """A(theta) per topology and per arm pair, with binomial errors."""
    n0, _ = np.histogram(thrown.theta, bins=bins)
    mid = 0.5 * (bins[:-1] + bins[1:])
    rows = []
    groups = [('all', acc)]
    groups += [(t, g) for t, g in acc.groupby('topology')]
    groups += [(f'{r.arm1}-{r.arm2}', g) for r, g in
               [(g.iloc[0], g) for _, g in acc.groupby(['arm1', 'arm2'])]]
    for name, g in groups:
        k, _ = np.histogram(g.theta, bins=bins)
        with np.errstate(divide='ignore', invalid='ignore'):
            a = np.where(n0 > 0, k / n0, np.nan)
            e = np.where(n0 > 0, np.sqrt(np.clip(k, 1, None)) / n0, np.nan)
        rows.append(pd.DataFrame(dict(group=name, theta=mid, n_acc=k,
                                      n_thrown=n0, acc=a, err=e)))
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--subruns',
                    default='stat090_0000,stat090_0001,stat090_0002')
    ap.add_argument('--n', type=int, default=2_000_000)
    a = ap.parse_args()
    subs = [s for s in a.subruns.split(',') if s]
    merged = str(paths.out('fullpass') / a.run)

    im = paths.out('imaging')
    meta = json.load(open(paths.require(im / f'imaging_{a.run}.meta.json',
                                        'the source position -- run '
                                        'source_imaging.py first')))
    v = {d['axis']: d['source_mm'] for d in meta['verdict']}
    offset = (v.get('X', 0.0), 0.0, v.get('Z', 0.0))

    ed = paths.out('efficiency')
    hl = pd.read_csv(paths.require(ed / f'efficiency_headline_{a.run}.csv',
                                   'the efficiency headline'))
    emap = pd.read_csv(paths.require(ed / f'efficiency_map_{a.run}.csv',
                                     'the efficiency map'))
    # B's number is a HIT efficiency and is not a track efficiency; it must not
    # be used as one.  B is excluded from the measurable topologies anyway.
    eff = {r.arm: (r.efficiency if r.basis == 'tracks' else 0.0)
           for r in hl.itertuples()}

    ch = Chambers(a.run, subs, merged, eff_map=emap, eff_headline=eff)
    acc, thrown = throw(ch, a.n, offset=offset)
    curves = acceptance_curves(acc, thrown)

    od = paths.out('angle')
    curves.to_csv(od / f'acceptance_{a.run}.csv', index=False)
    json.dump(dict(schema=SCHEMA, run=a.run, subruns=subs, n_thrown=a.n,
                   source_offset_mm=list(offset),
                   efficiency=eff,
                   dead_u_ranges={k: [[round(x, 1) for x in r] for r in v]
                                  for k, v in ch.dead.items()},
                   angle_arms=list(ANGLE_ARMS)),
              open(od / f'acceptance_{a.run}.meta.json', 'w'), indent=1)

    print(f'thrown {a.n:,} pairs from ({offset[0]:+.1f}, 0, {offset[2]:+.1f}) mm')
    print(f'accepted {len(acc):,} pair-legs in measurable topologies\n')
    tot = curves[curves.group == 'all']
    print('overall acceptance vs opening angle')
    for lo in range(0, 180, 20):
        m = (tot.theta >= lo) & (tot.theta < lo + 20)
        print(f'  {lo:3d}-{lo + 20:3d} deg   {100 * tot.acc[m].mean():7.4f} %')
    print('\nby topology (all angles)')
    print(acc.topology.value_counts().to_string())
    print('\nby arm pair')
    print(acc.groupby(['arm1', 'arm2']).size().to_string())
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
