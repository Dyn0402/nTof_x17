#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pair_physics.py -- the opening-angle distribution an e+e- pair is born with.

SUPERSEDED FOR IPC, 2026-09-09 -- USE :mod:`ipc_born` INSTEAD.  Everything
below about the IPC continuum is an ansatz carried as a four-variant band, and
that band spans a factor of 38 in the fraction above 109 deg.  None of that
spread is irreducible: at Z = 2 and 20.6 MeV the Born calculation is essentially
exact (alpha*Z = 0.015) and gives one closed-form curve per multipole --
4.6 % above 109 deg for M1, 9.4 % for E1, 11.4 % for E0, against the 11.8 % the
`geant` variant here produces for neither reason.  `ipc_born` is validated
against Wilkinson's published E0 law; this module is validated against a Geant4
generator that makes the same two guesses it does, which is not the same thing.
See ``sept26_prelim/ipc/report.html`` and <https://dylan-neff.web.cern.ch/x17/ipc-continuum/>.

The X17 half of this module is NOT superseded -- a 16.8 MeV boson from a
20.58 MeV transition is exact two-body kinematics and :func:`x17_angles` is
right.  What is still open is wiring the two-channel (E0 + M1) IPC model into
`opening_angle.py` in place of :data:`VARIANTS`.

WHY THIS EXISTS SEPARATELY FROM THE ACCEPTANCE.  The measured spectrum is
``physics(theta) x acceptance(theta)``.  The acceptance is ours -- our geometry,
our dead channels, our efficiency -- and we can measure it.  The physics is not
ours, it is nuclear structure, and the honest thing is to carry it as a BAND
rather than a curve.  This module is where that band is generated and where its
assumptions are written down.

THE KINEMATICS.  4He* de-excites at E = 20.58 MeV.  Two channels:

  X17   4He* -> 4He + X17 (m = 16.8 MeV), X17 -> e+e-.  gamma = E/m = 1.225,
        beta = 0.577, and the opening angle is minimised at symmetric sharing:
        cos(theta_min) = 1 - 2 m^2/E^2 = -0.333 -> 109 deg.  The signal piles up
        at 110-140 deg because a slow parent cannot collimate its daughters.
  IPC   4He* -> 4He + gamma*, gamma* -> e+e- with invariant mass M_ee anywhere
        from 2 m_e up to E.  A light gamma* is fast (gamma = E/M_ee is large),
        so its daughters are collimated -- which is why the IPC continuum falls
        steeply from small angles and the X17 peak sits on its tail.

WHAT IS ASSUMED, AND WHAT THE BAND IS.  The Geant4 pair generator
(`MX17_Full_Geant/src/X17PrimaryGenerator.cc`) makes two choices for IPC that
are ansatz rather than matrix element:

  1. ``dN/dM_ee ~ 1/M_ee`` (log-uniform between 2 m_e and E)
  2. the decay is ISOTROPIC in the gamma* rest frame

Neither is the internal-pair-creation matrix element.  In particular the
20.21 MeV state of 4He is **0+ -> 0+**, which cannot emit a real photon at all:
the transition is E0 and proceeds *only* through a virtual photon, which is
**longitudinally polarised**.  A longitudinal virtual photon decays as
``sin^2(theta*)``, not isotropically.  So :func:`opening_angles` takes the decay
distribution as a parameter, and the difference between the two is quoted as
the dominant modelling systematic instead of being hidden.

VALIDATION IS THE POINT.  :func:`validate_against_geant` reproduces the full
Geant4 truth distribution using Geant's OWN assumptions.  If the toy cannot do
that, nothing it says about the alternatives is worth anything.

    python -m sept26_prelim_analysis.pair_physics
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

SCHEMA = 'sept26_prelim/pair_physics/1'
M_E = 0.51099895            # MeV
E_TRANSITION = 20.58        # MeV, SimConfig.hh
M_X17 = 16.8                # MeV

#: The Geant4 pair sample, if the simulation repo is on this machine.  Its
#: truth is what the toy is validated against; nothing downstream requires it.
GEANT_NPZ = os.path.expanduser(
    '~/CLionProjects/MX17_Full_Geant/analysis/al_pair/signal_reco.npz')


# --------------------------------------------------------------------------- #
# kinematics
# --------------------------------------------------------------------------- #
def opening_angles(m_parent, e_parent: float = E_TRANSITION,
                   decay: str = 'isotropic', seed: int = 11) -> np.ndarray:
    """Lab opening angle [deg] for parents of mass ``m_parent``.

    ``m_parent`` may be a scalar (X17) or an array (one sampled M_ee per IPC
    event).  In the parent rest frame the two leptons are back to back with
    energy m/2; ``decay`` picks cos(theta*) relative to the boost axis:

      isotropic   flat in cos(theta*)      -- what the Geant generator assumes
      long        ~ sin^2(theta*)          -- a longitudinal virtual photon,
                                              which is what an E0 transition
                                              (0+ -> 0+, no real photon) emits
      trans       ~ 1 + cos^2(theta*)      -- a transverse virtual photon

    The opening angle does not depend on the boost DIRECTION, only on
    theta* and beta, so the parent's own isotropy never has to be sampled.
    """
    rng = np.random.default_rng(seed)
    m = np.atleast_1d(np.asarray(m_parent, float))
    n = len(m)
    if not np.all(m > 2 * M_E):
        raise ValueError('parent mass below the e+e- threshold')

    if decay == 'isotropic':
        c = rng.uniform(-1, 1, n)
    elif decay == 'long':
        # sin^2 -- rejection against its maximum, cheap and exact
        c = np.empty(n)
        todo = np.ones(n, bool)
        while todo.any():
            t = rng.uniform(-1, 1, todo.sum())
            acc = rng.uniform(0, 1, todo.sum()) < (1 - t ** 2)
            idx = np.where(todo)[0][acc]
            c[idx] = t[acc]
            todo[idx] = False
    elif decay == 'trans':
        c = np.empty(n)
        todo = np.ones(n, bool)
        while todo.any():
            t = rng.uniform(-1, 1, todo.sum())
            acc = rng.uniform(0, 1, todo.sum()) < (1 + t ** 2) / 2.0
            idx = np.where(todo)[0][acc]
            c[idx] = t[acc]
            todo[idx] = False
    else:
        raise ValueError(f'unknown decay distribution {decay!r}')
    s = np.sqrt(np.clip(1 - c ** 2, 0, None))

    # parent boost
    p_par = np.sqrt(np.clip(e_parent ** 2 - m ** 2, 0, None))
    beta = p_par / e_parent
    gamma = e_parent / m
    # daughters in the parent frame, along +-(s, 0, c)
    e_star = m / 2.0
    p_star = np.sqrt(np.clip(e_star ** 2 - M_E ** 2, 0, None))

    def boost(cz, sx):
        pz = gamma * (p_star * cz + beta * e_star)
        px = p_star * sx
        return px, pz

    px1, pz1 = boost(c, s)
    px2, pz2 = boost(-c, -s)
    dot = px1 * px2 + pz1 * pz2
    n1 = np.hypot(px1, pz1)
    n2 = np.hypot(px2, pz2)
    return np.degrees(np.arccos(np.clip(dot / (n1 * n2), -1, 1)))


def sample_mee(n: int, spectrum: str = 'inv', e_parent: float = E_TRANSITION,
               seed: int = 13) -> np.ndarray:
    """Virtual-photon invariant masses.

    ``inv``   dN/dM ~ 1/M   -- log-uniform, the Geant generator's ansatz
    ``inv3``  dN/dM ~ 1/M^3 -- the steeper falloff a real IPC matrix element
                               has once the phase-space factor is included;
                               used only to bracket the shape, not as a claim.
    """
    rng = np.random.default_rng(seed)
    lo, hi = 2 * M_E, e_parent
    u = rng.uniform(0, 1, n)
    if spectrum == 'inv':
        return lo * (hi / lo) ** u
    if spectrum == 'inv3':
        a, b = lo ** -2, hi ** -2
        return (a + u * (b - a)) ** -0.5
    raise ValueError(f'unknown spectrum {spectrum!r}')


def x17_angles(n: int = 400_000, **kw) -> np.ndarray:
    return opening_angles(np.full(n, M_X17), **kw)


def ipc_angles(n: int = 400_000, spectrum: str = 'inv',
               decay: str = 'isotropic', seed: int = 13) -> np.ndarray:
    return opening_angles(sample_mee(n, spectrum, seed=seed), decay=decay,
                          seed=seed + 1)


#: The variants that make the band.  The first is the simulation's own
#: assumption, so it is the one the toy is validated against.
VARIANTS = {
    'geant (1/M, isotropic)':  dict(spectrum='inv', decay='isotropic'),
    'E0 (1/M, longitudinal)':  dict(spectrum='inv', decay='long'),
    'transverse (1/M)':        dict(spectrum='inv', decay='trans'),
    'steeper mass (1/M³)':     dict(spectrum='inv3', decay='isotropic'),
}


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def geant_truth() -> dict:
    """X17 and IPC truth opening angles from the full Geant4 pair sample."""
    if not os.path.exists(GEANT_NPZ):
        return {}
    z = np.load(GEANT_NPZ)
    t, th = z['type'], z['theta_truth']
    return {'X17': th[t == 0], 'IPC': th[t == 1]}


def validate_against_geant(nbin: int = 60) -> pd.DataFrame:
    """Does the toy reproduce the full simulation under ITS assumptions?

    Compared on shape only (both normalised), because the toy has no
    cross-section and makes no claim about the relative rate.  The statistic is
    the largest absolute difference of the cumulative distributions -- a KS
    distance -- which is scale-free and does not need matched binning.
    """
    g = geant_truth()
    if not g:
        return pd.DataFrame()
    rows = []
    grid = np.linspace(0, 180, 721)
    for name, toy in (('X17', x17_angles(400_000, decay='isotropic')),
                      ('IPC', ipc_angles(400_000, **VARIANTS['geant (1/M, isotropic)']))):
        ref = g[name]
        c1 = np.searchsorted(np.sort(ref), grid) / len(ref)
        c2 = np.searchsorted(np.sort(toy), grid) / len(toy)
        rows.append(dict(channel=name, n_geant=len(ref), n_toy=len(toy),
                         ks=float(np.max(np.abs(c1 - c2))),
                         median_geant=float(np.median(ref)),
                         median_toy=float(np.median(toy)),
                         frac_gt110_geant=float((ref > 110).mean()),
                         frac_gt110_toy=float((toy > 110).mean())))
    return pd.DataFrame(rows)


def shapes(n: int = 400_000, bins=None) -> tuple:
    """Normalised dN/dtheta for X17 and every IPC variant, on one grid."""
    if bins is None:
        bins = np.arange(0, 181, 3.0)
    mid = 0.5 * (bins[:-1] + bins[1:])
    out = {'X17': np.histogram(x17_angles(n), bins=bins, density=True)[0]}
    for name, kw in VARIANTS.items():
        out[f'IPC · {name}'] = np.histogram(ipc_angles(n, **kw), bins=bins,
                                            density=True)[0]
    return mid, pd.DataFrame(out, index=pd.Index(mid, name='theta_deg'))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--n', type=int, default=400_000)
    a = ap.parse_args()

    V = validate_against_geant()
    mid, S = shapes(a.n)
    od = paths.out('angle')
    S.to_csv(od / 'physics_shapes.csv')
    V.to_csv(od / 'physics_validation.csv', index=False)
    g = geant_truth()
    if g:
        np.savez_compressed(od / 'geant_truth.npz', **g)
    json.dump(dict(schema=SCHEMA, m_x17=M_X17, e_transition=E_TRANSITION,
                   variants={k: v for k, v in VARIANTS.items()},
                   geant_source=GEANT_NPZ if g else None),
              open(od / 'physics.meta.json', 'w'), indent=1)

    print('VALIDATION -- the toy against the full Geant4 sample, '
          'using Geant\'s own assumptions')
    print(V.to_string(index=False) if len(V) else '  (no Geant sample here)')
    print('\nSHAPE SUMMARY -- fraction above the X17 threshold of 109 deg')
    for c in S.columns:
        w = S[c].to_numpy()
        frac = w[mid > 109].sum() / w.sum()
        print(f'  {c:<34} median {mid[np.searchsorted(np.cumsum(w), w.sum() / 2)]:6.1f} deg'
              f'   above 109 deg: {frac:5.1%}')
    print(f'\nwrote -> {od}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
