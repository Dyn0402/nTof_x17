#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
same_chamber_physics.py -- what the common-vertex same-chamber pairs are.

`det_a_intra` finds that on the is2_v2 re-pass, same-chamber pairs whose two
lines meet on the beam axis are ~2x more common than event mixing allows (A
x2.14, C x1.67).  This module asks what that excess is, with the three things
the data can say without a pair energy or a charge sign:

1.  WHERE the vertices are along the beam (:func:`vertex_y`): the per-bin
    excess of real vertexed pairs over the event-mixed expectation, against
    the measured capsule extent (`capsule_y`: centre +33 +- 3 mm, the CAD gas
    profile -29.5..+50.7 shifted by it).

2.  WHAT opening angles the excess has (:func:`excess_spectrum`), computed per
    angle bin as  k_real(theta) - n_real(theta) * r_mixed(theta)  so the chance
    vertex rate, which itself depends on angle, is taken out bin by bin.

3.  WHICH source shape that matches (:func:`fold`): a toy that throws pairs
    from the capsule wall into one chamber and applies everything the data
    went through -- active area, the |true tan| < 0.6 cut, the measured
    per-view two-track efficiency, the measured angular resolution (cosmic
    A-C truth, `ntof_cosmics/results/tracking/pooled/resolution.csv`) and the
    same vertex test as `det_a_intra` (lines within 30 mm, midpoint within
    20 mm of the axis).  Models:

    * ``Al conversion``: 27Al(n,g) gammas converting in the capsule, the
      opening angle at the chambers after multiple scattering, from Geant4
      (MX17_Full_Geant analysis/al_pair/convpair_reco.npz, theta_r, all pairs
      with both leptons in a drift gap).  85 % of the capsule pairs.
    * ``Al internal``: 27Al+12C internal pairs after the wall
      (`ipc_aluminium.shape_comparison`, capsule_after_wall).  15 %.
    * ``two leptons, independent``: two electrons from one capture with
      unrelated directions -- the Compton electrons of two cascade gammas,
      which Geant4 makes 4x more often than pairs (thermal accounting F3).
    * ``Al conversion, no scattering``: the Geant4 truth angle, to show what
      the multiple scattering does.

Shapes only.  The absolute rate needs the trigger and readout livetime per
pair, which the simulation's accounting does not carry.

    python -m sept26_prelim_analysis.same_chamber_physics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sept26_prelim_analysis import paths

SRC = Path.home() / 'x17_local' / 'same_chamber'
G4_CONV = Path.home() / 'CLionProjects/MX17_Full_Geant/analysis/al_pair/convpair_reco.npz'
IPC_SHAPE = paths.out('ipc') / 'al_shape_comparison.csv'
RESOLUTION = (Path(__file__).resolve().parent.parent / 'ntof_cosmics' / 'results'
              / 'tracking' / 'pooled' / 'resolution.csv')
#: strip plane distance from the beam axis, mm (geometry.MM_DIST_Z + W_STRIP)
D_PLANE = 234.6
#: active half-sizes, mm (geometry.MM_SIZE_U/V)
HALF_U, HALF_V = 190.0, 170.0
#: the axis foot in the chamber's own u (geometry.PINWHEEL)
U_AXIS = {'A': 16.35, 'C': 17.3}
#: capsule extent along the beam in the data frame: the CAD gas profile
#: (-29.5..+50.7) moved by the measured height (+32 relative to CAD's +0.8)
CAPSULE_V = (-29.5 + 32.2, 50.7 + 32.2)
CAPSULE_R = 10.0
TAN_MAX = 0.6
VERTEX_R_MM, VERTEX_DCA_MM = 20.0, 30.0
OPEN_BINS = np.arange(0.0, 90.1, 10.0)
Y_BINS = np.arange(-150.0, 151.0, 25.0)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _pairs(chain: str, arm: str) -> pd.DataFrame:
    return pd.read_parquet(SRC / f'{chain}_{arm}' / 'pairs.parquet',
                           columns=['mixed', 'vertex', 'open_deg', 'vy_mm', 'run'])


def excess_spectrum(chain: str, arm: str, bins=OPEN_BINS) -> pd.DataFrame:
    """Vertexed real pairs minus the event-mixed expectation, per angle bin."""
    P = _pairs(chain, arm)
    R, M = P[~P.mixed], P[P.mixed]
    nr, _ = np.histogram(R.open_deg, bins)
    kr, _ = np.histogram(R[R.vertex].open_deg, bins)
    nm, _ = np.histogram(M.open_deg, bins)
    km, _ = np.histogram(M[M.vertex].open_deg, bins)
    rm = np.where(nm > 0, km / np.maximum(nm, 1), 0.0)
    exp = nr * rm
    var = kr + nr ** 2 * np.where(nm > 0, np.maximum(km, 1) / np.maximum(nm, 1) ** 2, 0)
    return pd.DataFrame(dict(chain=chain, arm=arm, lo=bins[:-1], hi=bins[1:],
                             mid=0.5 * (bins[:-1] + bins[1:]), n_real=nr,
                             k_real=kr, n_mixed=nm, k_mixed=km, expected=exp,
                             excess=kr - exp, err=np.sqrt(var)))


def vertex_y(chain: str, arm: str, bins=Y_BINS) -> pd.DataFrame:
    """Where along the beam the vertexed pairs are: real, mixed-expected, excess."""
    P = _pairs(chain, arm)
    R, M = P[~P.mixed], P[P.mixed]
    s = len(R) / max(len(M), 1)
    hr, _ = np.histogram(R[R.vertex].vy_mm, bins)
    hm, _ = np.histogram(M[M.vertex].vy_mm, bins)
    return pd.DataFrame(dict(chain=chain, arm=arm, lo=bins[:-1], hi=bins[1:],
                             mid=0.5 * (bins[:-1] + bins[1:]), real=hr,
                             expected=hm * s, excess=hr - hm * s,
                             err=np.sqrt(hr + hm * s * s)))


# --------------------------------------------------------------------------- #
# the toy
# --------------------------------------------------------------------------- #
def _resolution(arm: str):
    """sigma(|tan|) per view, and the tail fraction, from the cosmic A-C truth."""
    R = pd.read_csv(RESOLUTION)
    R = R[R.arm == arm]
    out = {}
    for v in ('x', 'y'):
        g = R[R.axis == v].sort_values('lo')
        out[v] = (g.hi.to_numpy(), g.sigma.to_numpy(), g.tail_frac.to_numpy())
    return out


def _smear(t, view, res, rng):
    hi, sig, tail = res[view]
    i = np.minimum(np.searchsorted(hi, np.abs(t)), len(hi) - 1)
    s = sig[i]
    # the table's sigma is the core; its tail fraction goes out at 4x
    wide = rng.random(len(t)) < np.minimum(tail[i], 0.5) * 0.5
    return t + rng.normal(0.0, 1.0, len(t)) * np.where(wide, 4.0 * s, s)


def _sep_eff(arm: str):
    E = pd.read_csv(SRC / f'cf_{arm}' / 'two_track_efficiency_axis.csv')
    f = {}
    for ax in ('u', 'v'):
        g = E[E.axis == ax]
        f[ax] = (g.sep_mid.to_numpy(), g.eff.to_numpy())
    return lambda du, dv: (np.interp(np.abs(du), *f['u'], left=0.0, right=1.0)
                           * np.interp(np.abs(dv), *f['v'], left=0.0, right=1.0))


def _iso(n, rng):
    z = rng.uniform(-1, 1, n)
    ph = rng.uniform(0, 2 * np.pi, n)
    s = np.sqrt(1 - z * z)
    return np.c_[s * np.cos(ph), s * np.sin(ph), z]


def _rotate_by(d1, theta, rng):
    """A direction at angle theta from each d1, uniform in azimuth."""
    a = np.where(np.abs(d1[:, [0]]) < 0.9, [[1, 0, 0]], [[0, 1, 0]])
    e1 = np.cross(d1, a)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(d1, e1)
    ps = rng.uniform(0, 2 * np.pi, len(d1))[:, None]
    t = theta[:, None]
    return np.cos(t) * d1 + np.sin(t) * (np.cos(ps) * e1 + np.sin(ps) * e2)


def fold(arm: str, model: str, n: int = 4_000_000, seed: int = 7,
         theta_sampler=None) -> pd.DataFrame:
    """Throw common-vertex pairs from the capsule wall into chamber ``arm``.

    Frame: (u, v, w) of the chamber, w along its outward normal, the beam axis
    the line (U_AXIS, v, 0).  A leg is reconstructed as its strip-plane hit,
    smeared by 1 mm per view, with its tans smeared by the measured resolution.
    Returns, per reconstructed opening-angle bin, the accepted and the
    vertexed weight.
    """
    rng = np.random.default_rng(seed)
    res = _resolution(arm)
    sep = _sep_eff(arm)
    ph = rng.uniform(0, 2 * np.pi, n)
    V = np.c_[U_AXIS[arm] + CAPSULE_R * np.cos(ph),
              rng.uniform(*CAPSULE_V, n), CAPSULE_R * np.sin(ph)]
    # leg 1 aimed into the chamber's solid angle (uniform in the tans inside
    # 0.7, which brackets |tan| < 0.6 after smearing); weighted back to
    # isotropic by the solid-angle Jacobian
    tx = rng.uniform(-0.7, 0.7, n)
    ty = rng.uniform(-0.7, 0.7, n)
    d1 = np.c_[tx, ty, np.ones(n)]
    d1 /= np.linalg.norm(d1, axis=1, keepdims=True)
    w1 = d1[:, 2] ** 3                       # dOmega = cos^3 dtx dty
    if model == 'independent':
        d2 = _iso(n, rng)
    else:
        d2 = _rotate_by(d1, np.radians(theta_sampler(n, rng)), rng)
    out = {}
    for k, d in (('1', d1), ('2', d2)):
        ok = d[:, 2] > 0
        s = np.where(ok, (D_PLANE - V[:, 2]) / np.where(ok, d[:, 2], 1), np.nan)
        hit = V + d * s[:, None]
        out[k] = dict(ok=ok & (np.abs(hit[:, 0]) < HALF_U) & (np.abs(hit[:, 1]) < HALF_V),
                      u=hit[:, 0] + rng.normal(0, 1.0, n),
                      v=hit[:, 1] + rng.normal(0, 1.0, n),
                      tx=_smear(d[:, 0] / np.where(ok, d[:, 2], 1), 'x', res, rng),
                      ty=_smear(d[:, 1] / np.where(ok, d[:, 2], 1), 'y', res, rng))
    a, b = out['1'], out['2']
    acc = (a['ok'] & b['ok'] & (np.abs(a['tx']) < TAN_MAX) & (np.abs(a['ty']) < TAN_MAX)
           & (np.abs(b['tx']) < TAN_MAX) & (np.abs(b['ty']) < TAN_MAX))
    w = np.where(acc, w1 * np.nan_to_num(sep(a['u'] - b['u'], a['v'] - b['v'])), 0.0)
    # reconstructed lines through (u, v, D) with direction (tx, ty, 1)
    p1 = np.c_[a['u'], a['v'], np.full(n, D_PLANE)]
    p2 = np.c_[b['u'], b['v'], np.full(n, D_PLANE)]
    r1 = np.c_[a['tx'], a['ty'], np.ones(n)]
    r2 = np.c_[b['tx'], b['ty'], np.ones(n)]
    w0 = p1 - p2
    aa, bb, cc = (r1 * r1).sum(1), (r1 * r2).sum(1), (r2 * r2).sum(1)
    dd, ee = (r1 * w0).sum(1), (r2 * w0).sum(1)
    den = aa * cc - bb * bb
    sc = np.where(den > 1e-12, (bb * ee - cc * dd) / np.where(den > 1e-12, den, 1), 0)
    tc = np.where(den > 1e-12, (aa * ee - bb * dd) / np.where(den > 1e-12, den, 1), 0)
    c1, c2 = p1 + sc[:, None] * r1, p2 + tc[:, None] * r2
    dca = np.linalg.norm(c1 - c2, axis=1)
    mid = 0.5 * (c1 + c2)
    vr = np.hypot(mid[:, 0] - U_AXIS[arm], mid[:, 2])
    vtx = acc & (dca < VERTEX_DCA_MM) & (vr < VERTEX_R_MM)
    th_ok = np.isfinite(dca)
    cosr = (r1 * r2).sum(1) / np.sqrt(aa * cc)
    th = np.where(th_ok & acc, np.degrees(np.arccos(np.clip(np.nan_to_num(cosr), -1, 1))), -1.0)
    ha, _ = np.histogram(th, OPEN_BINS, weights=w)
    hv, _ = np.histogram(th, OPEN_BINS, weights=w * vtx)
    return pd.DataFrame(dict(arm=arm, model=model, lo=OPEN_BINS[:-1], hi=OPEN_BINS[1:],
                             mid=0.5 * (OPEN_BINS[:-1] + OPEN_BINS[1:]),
                             accepted=ha, vertexed=hv,
                             eff_vertex=np.where(ha > 0, hv / np.maximum(ha, 1e-300), np.nan)))


def samplers() -> dict:
    """Opening-angle samplers, degrees, for the correlated models."""
    g4 = np.load(G4_CONV)
    tr, tt = g4['theta_r'], g4['theta_t']
    S = pd.read_csv(IPC_SHAPE)
    pdf = S.capsule_after_wall.to_numpy()
    cdf = np.cumsum(pdf) / pdf.sum()
    lo = S.theta_mid.to_numpy() - 0.5

    def ipc(n, rng):
        i = np.searchsorted(cdf, rng.random(n))
        return lo[np.minimum(i, len(lo) - 1)] + rng.random(n)
    return {'Al conversion': lambda n, rng: rng.choice(tr, n),
            'Al conversion, no scattering': lambda n, rng: rng.choice(tt, n),
            'Al internal': ipc}


def compare(E: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    """Shape chi2 of the data excess against each folded model's vertexed spectrum."""
    rows = []
    for (arm, chain), e in E.groupby(['arm', 'chain']):
        live = e.k_real.to_numpy() > 0
        for model, f in F[F.arm == arm].groupby('model'):
            m = f.vertexed.to_numpy()
            if m.sum() <= 0:
                continue
            k = e.excess.to_numpy()
            s = e.err.to_numpy()
            pred = m / m.sum() * k.sum()
            chi2 = float((((k - pred) / np.where(s > 0, s, 1)) ** 2)[live].sum())
            ndf = int(live.sum()) - 1
            mid = f.mid.to_numpy()
            rows.append(dict(arm=arm, chain=chain, model=model, excess=float(k.sum()),
                             chi2=chi2, ndf=ndf,
                             median_model=float(np.interp(0.5, np.cumsum(m) / m.sum(), mid)),
                             frac_model_vertexed=float(m.sum() / f.accepted.sum())))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--n', type=int, default=4_000_000)
    ap.add_argument('--out', default=str(paths.out('same_chamber')))
    a = ap.parse_args()
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    E = pd.concat([excess_spectrum(c, arm) for arm in ('A', 'C') for c in ('cf', 'is2')],
                  ignore_index=True)
    Y = pd.concat([vertex_y(c, arm) for arm in ('A', 'C') for c in ('cf', 'is2')],
                  ignore_index=True)
    S = samplers()
    F = pd.concat([fold(arm, m, a.n, theta_sampler=S.get(m))
                   for arm in ('A', 'C')
                   for m in ('Al conversion', 'Al internal', 'independent',
                             'Al conversion, no scattering')], ignore_index=True)
    C = compare(E, F)
    E.to_csv(od / 'physics_excess_spectrum.csv', index=False)
    Y.to_csv(od / 'physics_vertex_y.csv', index=False)
    F.to_csv(od / 'physics_folded_models.csv', index=False)
    C.to_csv(od / 'physics_compare.csv', index=False)
    json.dump(dict(capsule_v=CAPSULE_V, capsule_r=CAPSULE_R, d_plane=D_PLANE,
                   tan_max=TAN_MAX, n_throw=a.n, g4_conv=str(G4_CONV),
                   ipc_shape=str(IPC_SHAPE), resolution=str(RESOLUTION)),
              open(od / 'physics.meta.json', 'w'), indent=1)
    pd.set_option('display.width', 200)
    print(C.to_string(index=False))
    for arm in ('A', 'C'):
        e = E[(E.arm == arm) & (E.chain == 'cf')]
        print(f'\n{arm} excess by angle:', ' '.join(f'{m:.0f}:{x:+.0f}±{s:.0f}'
                                                 for m, x, s in zip(e.mid, e.excess, e.err)))
        for model, f in F[F.arm == arm].groupby('model'):
            v = f.vertexed.to_numpy()
            print(f'   {model:30s}', ' '.join(f'{x:.2f}' for x in v / v.sum()),
                  f' vertex eff {f.vertexed.sum() / f.accepted.sum():.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
