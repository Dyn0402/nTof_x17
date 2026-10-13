#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capsule_pose.py -- where is the 3He capsule relative to the neutron beam and to
the chambers?  A forward model of where beam neutrons are captured on the
capsule's aluminium, pushed through the measured track response and the data's
own pointing estimators, compared with the data
(HANDOFF_CAPSULE_POSE_FIT.md).

Model
-----
* Geometry: the STEP polycones of MX17_Full_Geant DetectorConstruction.cc (gas,
  Al vessel, CFRP wrap), nose-first: local axial s runs from the tip (s = -35)
  to the valve (s = +51) and points along +y when untilted.
* Pose: capsule origin (x_c, y_c, z_c) in the chamber frame, tilt theta from +y
  toward azimuth phi (phi = 0 is +x, 90 deg is +z).  Beam along +y centred at
  (x_b, z_b), radial profile from Lambda2D (lamda2DvsEn_EAR2.root) weighted by
  flux x 1/v over the window's energies.
* Captures: each beam ray is walked through the posed capsule; Al (and CFRP)
  capture per step Sigma ds (the Al is thin: 0.8 % per 5.5 mm nose), the gas is
  a black absorber (the ray stops at gas entry).  Optional sources: an
  isotropic thermal flux (captures in the Al volume weighted by the unshadowed
  solid angle) and a broad parallel flux from below.
* Tracks: the Geant4 nose-first neutron run (g4_angle/neutrons_nose, the
  digitised A/C subset ntof_cosmics/g4_digi/g4_<arm>_is2_full through the real
  reconstruction).  Each Geant4 Al-capture track is TRANSLATED from its own
  capture point (beam vertex in x, z; Al-path midpoint in y) to a toy capture
  point.  So the pair/Compton kinematics, scattering in the wall and air
  (the line misses its vertex by ~40 mm) and the reconstruction response are
  Geant4's; only where the capture happens changes.
* Estimators: exactly capsule_tof's -- the robust band crossing for x (A, C)
  on all tracks and per height bin, and the y band crossing on the tracks that
  point within R_FOCUS of the source in x.

The data are mirrored in y on load (the stage-3 y mirror,
HANDOFF_CAPSULE_POSE_FIT.md section 2): p0_y -> -p0_y, d_y -> -d_y.

Steps (CSV to <out>/capsule_pose/):

  beam      beam-on-capsule fractions and the Al capture split against offset
  closure   the toy at the Geant4 pose against Geant4 through the same estimators
  data      the mirrored data's estimators per ToF window (A, C)
  shear     x crossing against height: the data, and the toy's point-like nose
            source with its Geant4-library bootstrap spread
  tof       background-subtracted y crossing against time since the flash
  scan      toy grid: beam-capsule offset d and tilt (theta, phi), per window
  fit       the capsule height each shape needs to match the data; the bound
            on d for a given prior on that height

    python -m sept26_prelim_analysis.capsule_pose [beam|closure|data|scan|fit|all]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from sept26_prelim_analysis import paths
from sept26_prelim_analysis import capsule_tof as CT

G4ROOT = Path.home() / 'CLionProjects/MX17_Full_Geant'
LAMBDA = G4ROOT / 'data/lamda2DvsEn_EAR2.root'
NOSE = Path('/media/dylan/data/x17/ntof_cosmics/g4_angle/neutrons_nose')
G4DIGI = CT.G4DIGI
FLIGHT_M = CT.FLIGHT_M
HALF = CT.HALF
D_PLANE = CT.D_PLANE
R_FOCUS = CT.R_FOCUS
#: sim arm ids in the reduced tables (run_digi --g4-arm: 2 = A, 3 = C)
SIM_ARM = {'A': 2, 'C': 3}
#: sim local u against global x: u = +x on A (w = +z), u = -x on C (w = -z),
#: v = +y on both (a right-handed (u, v, w) with w outward)
U_SIGN = {'A': 1.0, 'C': -1.0}
#: the axis foot in each sim arm's own u (capsule_tof.FOOT_SIM)
FOOT = CT.FOOT_SIM
#: reco active window on the strip plane (x_p0, y_p0 in 0 .. 2 HALF)
P0_MAX = 2 * HALF

# ---- STEP polycones (DetectorConstruction.cc; mm, local axial s) ----------
S_GAS = np.array([-29.5, -28, -26, -24, -22, -20, -15, -5, 5, 15, 20, 22, 24, 26, 28, 30, 32, 34,
                  36, 38, 40, 44, 50.7])
R_GAS = np.array([0.001, 6.000, 8.000, 9.165, 9.798, 10.000, 10.000, 10.000, 10.000, 10.000, 10.000,
                  9.798, 9.165, 8.000, 6.299, 4.842, 3.660, 2.711, 1.967, 1.410, 1.026, 0.750, 0.750])
S_AL = np.array([-35, -34, -33, -31, -29, -27, -25, -23, -21, -20, -15, -5, 5, 15, 20, 21, 23, 25, 27,
                 29, 31, 33, 35, 37, 39, 40, 45, 50, 51], float)
R_AL = np.array([0.000, 3.803, 5.287, 7.206, 8.480, 9.375, 9.994, 10.386, 10.600, 10.600, 10.600,
                 10.600, 10.600, 10.600, 10.600, 10.600, 10.386, 9.994, 9.375, 8.480, 7.206, 5.747,
                 4.708, 4.015, 3.621, 3.500, 3.500, 3.500, 3.500])
R_CF = R_AL + 0.9
R_CF[0] = 0.0
#: thermal capture macroscopic cross sections at 25.3 meV (1/mm); 1/v scaled.
#: Al: 0.231 b x 6.03e22 /cm3.  CFRP set so that the nominal pose gives the
#: Geant4 CFRP/Al capture ratio (2 348 / 148 635 = 1.6 %).
SIG_AL = 0.231e-24 * 6.03e22 / 10.0
SIG_CF = 0.21 * SIG_AL
#: scattering (1/mm), which removes a neutron from its ray: Al 1.4 b; CFRP's
#: hydrogen and carbon (H 2.07 % by mass, DetectorConstruction.cc) ~0.8 /cm.
#: They matter only for rays that graze the barrel (~40 mm of Al after a long
#: oblique path through the dome's CFRP).  SCAT_CF is tuned on the Geant4
#: radius distribution of Al captures (closure step).
SCAT_AL = 1.40e-24 * 6.03e22 / 10.0
SCAT_CF = 0.15
#: regions of the Al for the capture split (local s)
REGION_EDGES = (-35.0, -20.0, 20.0, 51.0)
REGIONS = ('nose', 'barrel', 'shoulder+neck')


def _rad(s, S, R):
    """Polycone radius at axial s (0 outside the solid's axial range)."""
    r = np.interp(s, S, R, left=0.0, right=0.0)
    return np.where((s < S[0]) | (s > S[-1]), 0.0, r)


def axis_of(theta_deg: float, phi_deg: float) -> np.ndarray:
    t, p = np.radians(theta_deg), np.radians(phi_deg)
    return np.array([np.sin(t) * np.cos(p), np.cos(t), np.sin(t) * np.sin(p)])


def material(P: np.ndarray, centre, axis) -> np.ndarray:
    """0 outside, 1 CFRP, 2 Al, 3 gas, for points P (..., 3) in the chamber frame."""
    q = P - np.asarray(centre, float)
    s = q @ axis
    rho = np.sqrt(np.maximum((q * q).sum(-1) - s * s, 0.0))
    m = np.zeros(s.shape, np.int8)
    m[rho < _rad(s, S_AL, R_CF)] = 1
    m[rho < _rad(s, S_AL, R_AL)] = 2
    m[rho < _rad(s, S_GAS, R_GAS)] = 3
    return m, s


# ---- beam ---------------------------------------------------------------
_PROFILE = {}


def window_energy(lo_ms: float, hi_ms: float):
    """Energy range (eV) of a ToF window at the EAR2 flight path."""
    def e(t_ms):
        v = FLIGHT_M / (t_ms * 1e-3)
        return 0.5 * 1.674927e-27 * v * v / 1.602177e-19
    return e(hi_ms), e(lo_ms)


def radial_profile(window=(20.0, 75.0)):
    """(r edges mm, pdf in r) of the beam: Lambda2D rows inside the window's
    energies, weighted by the EAR2 flux x 1/v (captures in a thin absorber)."""
    key = tuple(window)
    if key in _PROFILE:
        return _PROFILE[key]
    import uproot
    v, ex, ey = uproot.open(LAMBDA)['Lambda2D'].to_numpy()
    E = np.sqrt(ex[:-1] * ex[1:])
    lo, hi = window_energy(*window)
    fv, fe = uproot.open(CT.FLUX)['flux_n_pulse_NOisolet_100bpd'].to_numpy()
    fE = np.sqrt(fe[:-1] * fe[1:])
    flux = np.interp(np.log(E), np.log(fE), fv, left=fv[0], right=0.0)  # per bin, smooth
    m = (E >= lo) & (E <= hi) & (v.sum(1) > 0)
    rows = v[m] / v[m].sum(1, keepdims=True)
    w = flux[m] * np.sqrt(0.0253 / E[m])
    w = w / w.sum() if w.sum() > 0 else np.ones(m.sum()) / m.sum()
    pdf = (rows * w[:, None]).sum(0)
    _PROFILE[key] = (ey * 10.0, pdf / pdf.sum())
    return _PROFILE[key]


def sample_beam(n: int, rng, window, centre=(0.0, 0.0), width_scale=1.0):
    """n beam rays (x, z) from the radial profile."""
    edges, pdf = radial_profile(window)
    i = rng.choice(len(pdf), n, p=pdf)
    r = (edges[i] + rng.random(n) * (edges[i + 1] - edges[i])) * width_scale
    a = rng.random(n) * 2 * np.pi
    return centre[0] + r * np.cos(a), centre[1] + r * np.sin(a), r


# ---- capture points ------------------------------------------------------
DY = 0.1


def trace_rays(x, z, centre, axis, e_scale=1.0, chunk=2000):
    """Walk rays along +y.  Per ray: total capture probability in Al and CFRP
    before the gas, whether it reaches the gas, and one capture point (and its
    local s and material) drawn along the path by weight."""
    st = float(np.hypot(axis[0], axis[2]))
    ys = np.arange(centre[1] - 37 - 60 * st, centre[1] + 53 + 60 * st, DY) + DY / 2
    n = len(x)
    w_tot = np.zeros(n)
    w_al = np.zeros(n)
    hit_gas = np.zeros(n, bool)
    yc = np.full(n, np.nan)
    sc = np.full(n, np.nan)
    mc = np.zeros(n, np.int8)
    rng = np.random.default_rng(12345)
    # restrict to rays that can touch the capsule's bounding cylinder
    # the axis segment (tip to valve) projected on the transverse plane; a ray
    # can touch the capsule only within the wrap radius of it
    a0 = centre + S_AL[0] * axis
    a1 = centre + S_AL[-1] * axis
    p0, p1 = np.array([a0[0], a0[2]]), np.array([a1[0], a1[2]])
    seg = p1 - p0
    L2 = max(seg @ seg, 1e-12)
    q = np.stack([x, z], -1) - p0
    tt = np.clip(q @ seg / L2, 0, 1)
    dist = np.linalg.norm(q - tt[:, None] * seg, axis=1)
    near = np.where(dist < 11.6 / max(abs(axis[1]), 0.3))[0]
    for k in range(0, len(near), chunk):
        idx = near[k:k + chunk]
        P = np.empty((len(idx), len(ys), 3))
        P[..., 0] = x[idx, None]
        P[..., 1] = ys[None, :]
        P[..., 2] = z[idx, None]
        m, s = material(P, centre, axis)
        gas = m == 3
        first = np.where(gas.any(1), gas.argmax(1), len(ys))
        before = np.arange(len(ys))[None, :] < first[:, None]
        sig = np.where(m == 2, SIG_AL, np.where(m == 1, SIG_CF, 0.0)) * e_scale
        sig = sig * before
        att = np.where(m == 2, SCAT_AL, np.where(m == 1, SCAT_CF, 0.0)) + sig
        surv = np.exp(-(np.cumsum(att, 1) - att) * DY)
        w = sig * DY * surv
        tot = w.sum(1)
        w_tot[idx] = tot
        w_al[idx] = (w * (m == 2)).sum(1)
        hit_gas[idx] = first < len(ys)
        ok = tot > 0
        c = np.cumsum(w[ok], 1)
        u = rng.random(ok.sum()) * c[:, -1]
        j = (c < u[:, None]).sum(1)
        ii = idx[ok]
        yc[ii] = ys[j]
        sc[ii] = s[ok, j]
        mc[ii] = m[ok, j]
    return dict(w=w_tot, w_al=w_al, gas=hit_gas, y=yc, s=sc, mat=mc)


def beam_captures(pose: dict, n: int = 200_000, window=(20.0, 75.0), seed=0, width_scale=1.0):
    """Capture points (x, y, z, weight, s, mat) of beam rays on a posed capsule,
    plus the beam fractions."""
    rng = np.random.default_rng(seed)
    centre = np.array([pose['x_c'], pose['y_c'], pose['z_c']], float)
    axis = axis_of(pose.get('theta', 0.0), pose.get('phi', 0.0))
    x, z, _ = sample_beam(n, rng, window, (pose['x_b'], pose['z_b']), width_scale)
    t = trace_rays(x, z, centre, axis)
    ok = t['w'] > 0
    C = pd.DataFrame(dict(x=x[ok], y=t['y'][ok], z=z[ok], w=t['w'][ok], s=t['s'][ok], mat=t['mat'][ok]))
    frac = dict(on_capsule=float(((t['w'] > 0) | t['gas']).mean()), gas=float(t['gas'].mean()),
                al_per_ray=float(t['w_al'].mean()), cap_per_ray=float(t['w'].mean()))
    al = C[C.mat == 2]
    for name, lo, hi in zip(REGIONS, REGION_EDGES[:-1], REGION_EDGES[1:]):
        frac[f'al_{name}'] = float(al.w[(al.s >= lo) & (al.s < hi)].sum() / max(al.w.sum(), 1e-30))
    frac['cfrp_over_al'] = float(C.w[C.mat == 1].sum() / max(al.w.sum(), 1e-30))
    return C, frac


def isotropic_captures(pose: dict, n: int = 60_000, n_dir: int = 48, seed=1):
    """Captures under an isotropic thermal flux: points uniform in the Al (and
    CFRP) volume, weighted by Sigma x the fraction of directions from which a
    neutron arrives without crossing the gas (the gas is black)."""
    rng = np.random.default_rng(seed)
    centre = np.array([pose['x_c'], pose['y_c'], pose['z_c']], float)
    axis = axis_of(pose.get('theta', 0.0), pose.get('phi', 0.0))
    # sample in the capsule's own frame (axis = +y), then rotate
    s = rng.uniform(-35, 51, 4 * n)
    rr = 11.5 * np.sqrt(rng.random(4 * n))
    a = rng.random(4 * n) * 2 * np.pi
    L = np.stack([rr * np.cos(a), s, rr * np.sin(a)], -1)
    ral, rcf, rg = _rad(s, S_AL, R_AL), _rad(s, S_AL, R_CF), _rad(s, S_GAS, R_GAS)
    mat = np.where(rr < rg, 3, np.where(rr < ral, 2, np.where(rr < rcf, 1, 0)))
    keep = (mat == 1) | (mat == 2)
    L, s, mat = L[keep][:n], s[keep][:n], mat[keep][:n]
    # rotation taking +y to axis
    R = _rot_y_to(axis)
    P = L @ R.T + centre
    vis = np.zeros(len(P))
    steps = np.arange(0.25, 100.0, 0.5)
    for k in range(0, len(P), 2000):
        p = P[k:k + 2000]
        dirs = rng.normal(size=(len(p), n_dir, 3))
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        blocked = np.zeros((len(p), n_dir), bool)
        for st in np.array_split(steps, 8):
            Q = p[:, None, None, :] + dirs[:, :, None, :] * st[None, None, :, None]
            m, _ = material(Q, centre, axis)
            blocked |= (m == 3).any(-1)
        vis[k:k + 2000] = 1 - blocked.mean(1)
    w = np.where(mat == 2, SIG_AL, SIG_CF) * vis
    C = pd.DataFrame(dict(x=P[:, 0], y=P[:, 1], z=P[:, 2], w=w, s=s, mat=mat))
    return C


def _rot_y_to(axis):
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(y, axis)
    c = float(y @ axis)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


NOMINAL = dict(x_c=0.0, y_c=0.0, z_c=0.0, theta=0.0, phi=0.0, x_b=0.0, z_b=0.0)


# ---- the Geant4 track library ---------------------------------------------
_LIB = {}


def library(arm: str) -> pd.DataFrame:
    """Geant4 nose-first tracks of one arm, reco (g4_digi) joined to the truth
    capture volume and beam vertex (g4_angle/neutrons_nose).  Columns in the sim
    frame: ur, vr (reco line at the mesh), tur, tvr (reco du/dw, dv/dw, signs
    resolved), ut, vt, tut, tvt (truth gap line), and the capture point
    estimate cx, cy, cz in the chamber frame."""
    if arm in _LIB:
        return _LIB[arm]
    parts = sorted(glob.glob(str(G4DIGI / f'g4_{arm}_is2_full' / '*.parquet')))
    R = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    for c in ('x_quality_ok', 'y_quality_ok'):
        R[c] = R[c].astype('boolean').fillna(False).astype(bool)
    T = []
    for f in sorted(glob.glob(str(NOSE / 'r*.parquet'))):
        t = pd.read_parquet(f, columns=['eventID', 'arm', 'vtx_x', 'vtx_z', 'capture_vol', 'neutron_E_eV'])
        t = t[t.arm == SIM_ARM[arm]].drop(columns='arm')
        t['file'] = 'neutron_bg_job' + Path(f).stem[1:]
        T.append(t)
    T = pd.concat(T).drop_duplicates(['file', 'eventID'])
    R = R.merge(T, on=['file', 'eventID'], how='left', validate='many_to_one')
    q = R.x_quality_ok & R.y_quality_ok & (R.x_tan_theta.abs() < 0.6) & (R.y_tan_theta.abs() < 0.6)
    R = R[q].copy()
    ur = -(R.x_p0.to_numpy(float) - HALF)
    vr = -(R.y_p0.to_numpy(float) - HALF)
    # reco tan signs: the pair that focuses (as capsule_tof.g4)
    best = None
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            c = np.corrcoef(ur, sx * R.x_tan_theta)[0, 1] + np.corrcoef(vr, sy * R.y_tan_theta)[0, 1]
            if best is None or c > best[0]:
                best = (c, sx, sy)
    _, sx, sy = best
    L = pd.DataFrame(dict(ur=ur, vr=vr, tur=sx * R.x_tan_theta.to_numpy(float),
                          tvr=sy * R.y_tan_theta.to_numpy(float),
                          ut=R.u_mesh.to_numpy(float), vt=R.v_mesh.to_numpy(float),
                          tut=R.tan_u.to_numpy(float), tvt=R.tan_v.to_numpy(float),
                          vol=R.capture_vol.fillna('?').to_numpy(), vx=R.vtx_x.to_numpy(float),
                          vz=R.vtx_z.to_numpy(float)))
    # capture point: the beam vertex in x, z (thermal neutrons barely scatter
    # in 5 mm of Al), the Al-path midpoint along y at that radius
    r = np.hypot(L.vx, L.vz)
    L['capsule'] = L.vol.isin(['He3Cap_Al', 'He3Cap_CFRP'])
    L['cx'], L['cz'] = L.vx, L.vz
    L['cy'] = _al_mid_y(r)
    L.attrs.update(sx=sx, sy=sy)
    _LIB[arm] = L
    return L


def _al_mid_y(r):
    """Mid-height of the Al traversed by an axial ray at radius r before the gas
    (nominal pose)."""
    s = np.arange(-35, 51, 0.05)
    ral, rg = _rad(s, S_AL, R_AL), _rad(s, S_GAS, R_GAS)
    out = np.full(len(r), np.nan)
    for i, rv in enumerate(np.asarray(r)):
        inal = (rv < ral) & ~(rv < rg)
        g = np.where(rv < rg)[0]
        if g.size:
            inal &= np.arange(len(s)) < g[0]
        out[i] = s[inal].mean() if inal.any() else np.nan
    return out


def toy_tracks(arm: str, C: pd.DataFrame, n: int = 60_000, seed=7, truth=False,
               keep_non_capsule=0.0, s_y=1.0, alpha_deg=0.0) -> pd.DataFrame:
    """Track sample of one arm for capture points C: each toy capture takes a
    random Geant4 capsule track (nose captures, r < 8 mm) translated from its
    own capture point to the toy's.  Returns pos/disp/hgt in the chamber frame
    (the data's capture_tof columns)."""
    L = library(arm)
    lib = L[L.capsule & (np.hypot(L.vx, L.vz) < 8.0) & np.isfinite(L.cy)]
    rng = np.random.default_rng(seed)
    p = C.w.to_numpy() / C.w.sum()
    ic = rng.choice(len(C), n, p=p)
    il = rng.integers(0, len(lib), n)
    c = C.iloc[ic]
    g = lib.iloc[il]
    su = U_SIGN[arm]
    du = su * (c.x.to_numpy() - g.cx.to_numpy())
    dv = c.y.to_numpy() - g.cy.to_numpy()
    dw = su * (c.z.to_numpy() - g.cz.to_numpy())  # outward normal: +z on A, -z on C
    if truth:
        u, v, tu, tv = g.ut.to_numpy(), g.vt.to_numpy(), g.tut.to_numpy(), g.tvt.to_numpy()
        lim = 179.0
        u_off = 0.0
    else:
        u, v, tu, tv = g.ur.to_numpy(), g.vr.to_numpy(), g.tur.to_numpy(), g.tvr.to_numpy()
        lim = HALF
        u_off = 0.0
    u2 = u + du - tu * dw
    v2 = v + dv - tv * dw
    ok = (np.abs(u2 - u_off) < lim) & (np.abs(v2) < lim)
    u2, v2, tu, tv = u2[ok], v2[ok], tu[ok], s_y * tv[ok]
    if alpha_deg:
        # a common in-plane rotation of the chamber frame about the axis foot
        # (A and C turn the same way in the global frame: u = +-x)
        a = np.radians(alpha_deg) * U_SIGN[arm]
        ca, sa = np.cos(a), np.sin(a)
        uc = u2 - FOOT[arm]
        u2, v2 = FOOT[arm] + ca * uc - sa * v2, sa * uc + ca * v2
        tu, tv = ca * tu - sa * tv, sa * tu + ca * tv
    T = _frame(arm, u2, v2, tu, tv)
    if keep_non_capsule > 0:
        o = L[~L.capsule]
        m = rng.random(len(o)) < keep_non_capsule
        o = o[m]
        T = pd.concat([T, _frame(arm, o.ur.to_numpy(), o.vr.to_numpy(), o.tur.to_numpy(),
                                 o.tvr.to_numpy())], ignore_index=True)
    return T


def _frame(arm, u, v, tu, tv):
    """Sim (u, v at the mesh, du/dw, dv/dw) -> the data's crossing columns in
    the chamber frame: pos (global x on the plane), disp (to the axis plane),
    p0_y, hgt (global y at the axis plane)."""
    su = U_SIGN[arm]
    pos = su * (u - FOOT[arm])
    disp = -su * tu * D_PLANE
    return pd.DataFrame(dict(pos=pos, disp=disp, p0_y=v, hgt=v - tv * D_PLANE))


# ---- estimators (shared by data and toy) -----------------------------------
H_EDGES = np.arange(-110.0, 71.0, 20.0)


def estimators(T: pd.DataFrame, per_height=True) -> dict:
    """x crossing (all tracks), y crossing (tracks within R_FOCUS in x), the
    height spread of focused tracks, and x crossings per height bin.  T has
    pos, disp, p0_y, hgt in the TRUE (mirrored) frame."""
    xs, xe, n = CT.crossing(T.pos, T.disp)
    foc = T[np.abs(T.pos + T.disp - xs) < R_FOCUS]
    y0, ye, ny = CT.crossing(foc.p0_y, foc.hgt - foc.p0_y)
    h = foc.hgt.to_numpy()
    q16, q50, q84 = np.percentile(h, [16, 50, 84]) if len(h) > 50 else (np.nan,) * 3
    out = dict(n=len(T), x_cross=xs, x_err=xe, y_cross=y0, y_err=ye, n_focus=len(foc),
               f_focus=len(foc) / max(len(T), 1), h_med=q50, h_w68=(q84 - q16) / 2,
               h_skew=((q84 - q50) - (q50 - q16)) / max(q84 - q16, 1e-9))
    # band slope of y (scale-carrying: 1 for a point source and a true y scale)
    if len(foc) > 50:
        a = np.polyfit(foc.p0_y, foc.hgt - foc.p0_y, 1)[0]
        out['y_band_slope'] = -a
    if per_height:
        rows = []
        for lo, hi in zip(H_EDGES[:-1], H_EDGES[1:]):
            s = T[(T.hgt >= lo) & (T.hgt < hi)]
            c, e, k = CT.crossing(s.pos, s.disp)
            rows.append((0.5 * (lo + hi), c, e, k))
        H = np.array(rows, float)
        out['shear'] = H
        good = np.isfinite(H[:, 1]) & (H[:, 3] > 300) & (np.abs(H[:, 0] - y0) < 50)
        if good.sum() >= 3:
            w = 1 / np.maximum(H[good, 2], 0.05) ** 2
            p = np.polyfit(H[good, 0], H[good, 1], 1, w=np.sqrt(w))
            out['shear_slope'] = p[0]
    return out


# ---- data -------------------------------------------------------------------
WINDOWS = {'late': (20.0, 75.0), 'mid': (12.0, 20.0), 'beam': (4.0, 12.0)}


def load_data(src=CT.CF, arms=('A', 'C')) -> pd.DataFrame:
    """capsule_tof.load with the stage-3 y mirror undone: true y = -stored y."""
    d = CT.load(src, arms)
    d['p0_y'] = -d.p0_y
    d['d_y'] = -d.d_y
    d['tany'] = -d.tany
    d['hgt'] = -d.hgt
    return d


def data_step(out: Path) -> pd.DataFrame:
    d = load_data()
    rows, shear = [], []
    for arm, g in d.groupby('arm'):
        for name, (lo, hi) in WINDOWS.items():
            s = g[(g.ms >= lo) & (g.ms < hi)]
            e = estimators(s)
            H = e.pop('shear')
            rows.append(dict(arm=arm, window=name, lo=lo, hi=hi, **e))
            for hc, c, ce, k in H:
                shear.append(dict(arm=arm, window=name, h=hc, x_cross=c, x_err=ce, n=k))
        # early window, background-subtracted y crossing: the in-focus sample of
        # the beam window minus the off-focus control rescaled to the null
        rows.append(dict(arm=arm, window='beam_bgsub', lo=4.0, hi=12.0,
                         **bg_subtracted(g[(g.ms >= 4) & (g.ms < 12)], g[g.ms >= 20])))
    R = pd.DataFrame(rows)
    R.to_csv(out / 'data.csv', index=False)
    pd.DataFrame(shear).to_csv(out / 'data_shear.csv', index=False)
    return R


def bg_subtracted(s: pd.DataFrame, late: pd.DataFrame) -> dict:
    """y crossing of the capsule component of a low-purity window.  The
    focused sample = signal + background; the background's (p0_y, dy) is
    taken from the same window's tracks that miss the source in x by
    30-60 mm (lines that are not from the capsule), normalised to the
    shuffled-angle null of the focus fraction.  A weighted least-squares band
    line on focus (+1) and control (-k) gives the signal's crossing."""
    xs = CT.crossing(late.pos, late.disp)[0]
    miss = np.abs(s.pos + s.disp - xs)
    rng = np.random.default_rng(1)
    f, f0 = CT.focus_excess(s.pos.to_numpy(), s.disp.to_numpy(), xs, rng)
    foc = s[miss < R_FOCUS]
    ctl = s[(miss > 30) & (miss < 60)]
    # expected background in focus: the null fraction x all tracks
    k = f0 * len(s) / max(len(ctl), 1)
    v = np.r_[foc.p0_y, ctl.p0_y]
    dy = np.r_[foc.hgt - foc.p0_y, ctl.hgt - ctl.p0_y]
    w = np.r_[np.ones(len(foc)), -k * np.ones(len(ctl))]
    keep = np.ones(len(v), bool)
    for _ in range(4):
        A = np.vstack([v[keep], np.ones(keep.sum())]).T
        W = w[keep]
        a, b = np.linalg.solve(A.T @ (A * W[:, None]), A.T @ (W * dy[keep]))
        r = dy - (a * v + b)
        sig = 1.4826 * np.median(np.abs(r[(w > 0) & keep]))
        keep = np.abs(r) < 3 * sig
    y0 = -b / a
    # signal-only control check: the same weighted fit with the late window's
    # focused tracks as the signal template gives that window's crossing
    return dict(n=len(s), n_focus=len(foc), f_focus=f, y_cross=float(y0), x_cross=float(xs),
                y_band_slope=float(-a), purity=float((f - f0) / f) if f > 0 else np.nan,
                bg_scale=float(k))


# ---- steps -----------------------------------------------------------------
def beam_step(out: Path) -> pd.DataFrame:
    rows = []
    for win in ('beam', 'late'):
        for d in (0, 2, 4, 6, 8, 9, 10, 11, 12, 14):
            pose = dict(NOMINAL, x_c=-float(d))
            C, fr = beam_captures(pose, n=200_000, window=WINDOWS[win])
            al = C[C.mat == 2]
            rows.append(dict(window=win, d=d, **fr,
                             y_mean_al=float(np.average(al.y, weights=al.w)),
                             x_mean_al=float(np.average(al.x, weights=al.w))))
    R = pd.DataFrame(rows)
    R.to_csv(out / 'beam.csv', index=False)
    return R


def closure_step(out: Path) -> pd.DataFrame:
    """The toy at the Geant4 pose against Geant4 itself (truth and reco), with
    the same estimators.  The toy keeps Geant4's tracks but draws the capture
    point from the ray trace, so this tests the ray trace and the translation."""
    rows = []
    C, fr = beam_captures(NOMINAL, n=200_000, window=(1.0, 75.0))
    for arm in ('A', 'C'):
        L = library(arm)
        for kind in ('reco', 'truth'):
            if kind == 'reco':
                G = _frame(arm, L.ur.to_numpy(), L.vr.to_numpy(), L.tur.to_numpy(), L.tvr.to_numpy())
            else:
                G = _frame(arm, L.ut.to_numpy(), L.vt.to_numpy(), L.tut.to_numpy(), L.tvt.to_numpy())
            Gc = G[L.capsule.to_numpy()]
            e_g = estimators(G, per_height=True)
            e_gc = estimators(Gc, per_height=False)
            T = toy_tracks(arm, C, n=len(Gc) * 4, truth=(kind == 'truth'))
            e_t = estimators(T)
            for lab, e in (('geant4 all', e_g), ('geant4 capsule', e_gc), ('toy nominal', e_t)):
                rows.append(dict(arm=arm, kind=kind, sample=lab,
                                 **{k: v for k, v in e.items() if k != 'shear'}))
            # translation sanity: shift the toy capture by +5 mm in x and +10 in y
            C2 = C.assign(x=C.x + 5.0, y=C.y + 10.0)
            e2 = estimators(toy_tracks(arm, C2, n=len(Gc) * 4, truth=(kind == 'truth')), per_height=False)
            rows.append(dict(arm=arm, kind=kind, sample='toy shifted +5 x, +10 y', **e2))
    R = pd.DataFrame(rows)
    R.to_csv(out / 'closure.csv', index=False)
    return R


SCAN_D = (0, 2, 4, 6, 7, 8, 9, 10, 11, 12, 14)
SCAN_THETA = (0, 5, 10, 20)
#: phi = 0 tilts the top of the capsule toward the beam side (+x when the
#: capsule sits at -x of the beam), i.e. the nose AWAY from the beam: Dylan's
#: hypothesis.  phi = 180 is the opposite sense.
SCAN_PHI = (0, 180)


def scan_step(out: Path) -> pd.DataFrame:
    """Toy grid in the shape parameters.  Everything is translation-equivariant
    (shifting capsule and beam together shifts every crossing by the same
    amount: closure step), so only the beam-capsule offset d and the tilt are
    scanned; the absolute position is solved against the data in fit."""
    rows = []
    for win in ('late', 'beam'):
        for d in SCAN_D:
            for th in SCAN_THETA:
                for ph in (SCAN_PHI if th else (0,)):
                    if win == 'beam' and th not in (0, 10):
                        continue
                    pose = dict(NOMINAL, x_c=-float(d), theta=float(th), phi=float(ph))
                    C, fr = beam_captures(pose, n=80_000, window=WINDOWS[win])
                    r = dict(window=win, d=d, theta=th, phi=ph, **fr)
                    for arm in ('A', 'C'):
                        e = estimators(toy_tracks(arm, C, n=50_000), per_height=False)
                        r.update({f'{k}_{arm}': e[k] for k in ('x_cross', 'y_cross', 'h_w68', 'h_skew')})
                    rows.append(r)
                    print(win, d, th, ph, round(r['y_cross_A'], 2), round(r['al_barrel'], 3), flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out / 'scan.csv', index=False)
    return R


#: z crossings of B and D (campaign medians, imaging_campaign/per_arm.csv; D
#: on its hot-channel-cleaned sample).  No B/D Geant4 reco exists to correct
#: them by, so they are used as the beam's z directly.
Z_BD = {'B': (-4.12, 1.6), 'D': (-3.45, 0.8)}


def fit_step(out: Path) -> pd.DataFrame:
    """For every scanned shape: where the capsule must sit to give the data's
    late crossings (x, y per arm), and what that pose implies.  The capsule's
    height in the chamber frame is not measured by anything else, so the
    offset d is constrained only through the height it requires."""
    S = pd.read_csv(out / 'scan.csv')
    D = pd.read_csv(out / 'data.csv')
    late = D[D.window == 'late'].set_index('arm')
    S = S[S.window == 'late'].copy()
    for arm in ('A', 'C'):
        S[f'dy_c_{arm}'] = late.loc[arm, 'y_cross'] - S[f'y_cross_{arm}']
        S[f'x_b_{arm}'] = late.loc[arm, 'x_cross'] - S[f'x_cross_{arm}']
    S['dy_c'] = 0.5 * (S.dy_c_A + S.dy_c_C)
    S['x_beam'] = 0.5 * (S.x_b_A + S.x_b_C)
    z = np.array([v for v, _ in Z_BD.values()])
    w = 1 / np.array([e for _, e in Z_BD.values()]) ** 2
    S['z_beam'] = float((z * w).sum() / w.sum())
    # what the nominal (CAD) tip height would have to be wrong by
    S['tip_y'] = -35.0 + S.dy_c
    # transverse offset of the nose tip from the beam centre: a tilt can carry
    # the nose back into the beam however far the capsule centre is off it
    t, ph = np.radians(S.theta), np.radians(S.phi)
    S['d_tip'] = np.hypot(-S.d + S_AL[0] * np.sin(t) * np.cos(ph), S_AL[0] * np.sin(t) * np.sin(ph))
    S.to_csv(out / 'fit.csv', index=False)
    # what survives for a given prior on the capsule height: the geometry
    # (d, tilt) is degenerate, the consequences are not
    half_ac = 0.5 * abs(late.loc['A', 'y_cross'] - late.loc['C', 'y_cross'])
    rows = []
    for prior in (2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30):
        ok = S[S.dy_c.abs() <= prior + half_ac]
        rows.append(dict(height_prior_mm=prior, n_poses=len(ok),
                         d_max=ok.d.max() if len(ok) else np.nan,
                         d_tip_max=ok.d_tip.max() if len(ok) else np.nan,
                         gas_min=ok.gas.min() if len(ok) else np.nan,
                         barrel_max=ok.al_barrel.max() if len(ok) else np.nan,
                         al_rel_min=ok.al_per_ray.min() / S.al_per_ray[(S.d == 0) & (S.theta == 0)].iloc[0]
                         if len(ok) else np.nan))
    B = pd.DataFrame(rows)
    B.to_csv(out / 'bound.csv', index=False)
    return B


def shear_step(out: Path, n_boot: int = 20) -> pd.DataFrame:
    """x crossing per height bin for the toy at the Geant4 shape (point-like
    nose source), aligned to the data's late crossings, with the spread over
    bootstrap replicas of the Geant4 track library (the toy's limit: ~10 k
    tracks per arm).  Beside it, the data's own late curve."""
    D = pd.read_csv(out / 'data.csv').set_index(['arm', 'window'])
    S = pd.read_csv(out / 'data_shear.csv')
    C, _ = beam_captures(NOMINAL, n=150_000, window=WINDOWS['late'])
    rows = []
    for arm in ('A', 'C'):
        L0 = library(arm)
        d = D.loc[(arm, 'late')]
        curves = []
        for b in range(n_boot):
            rng = np.random.default_rng(100 + b)
            _LIB[arm] = L0.iloc[rng.integers(0, len(L0), len(L0))].reset_index(drop=True)
            T = toy_tracks(arm, C, n=60_000, seed=b)
            e = estimators(T, per_height=False)
            T = T.assign(pos=T.pos + d.x_cross - e['x_cross'], p0_y=T.p0_y + d.y_cross - e['y_cross'],
                         hgt=T.hgt + d.y_cross - e['y_cross'])
            curves.append(estimators(T)['shear'][:, 1])
        _LIB[arm] = L0
        c = np.array(curves)
        sd = S[(S.arm == arm) & (S.window == 'late')].set_index('h')
        for k, h in enumerate(H_EDGES[:-1] + 10):
            rows.append(dict(arm=arm, h=h, toy=np.nanmean(c[:, k]), toy_sd=np.nanstd(c[:, k]),
                             data=sd.loc[h, 'x_cross'], data_err=sd.loc[h, 'x_err'], n_data=sd.loc[h, 'n']))
    R = pd.DataFrame(rows)
    # the step: mean of the bins 20-60 mm above the source minus 10-30 below
    for arm in ('A', 'C'):
        y0 = D.loc[(arm, 'late'), 'y_cross']
        r = R[R.arm == arm]
        up = r[(r.h - y0 > 15) & (r.h - y0 < 75)]
        lo = r[(r.h - y0 > -35) & (r.h - y0 < 5)]
        st_d = up.data.mean() - lo.data.mean()
        st_t = up.toy.mean() - lo.toy.mean()
        err = np.sqrt((up.toy_sd ** 2).sum() / len(up) ** 2 + (lo.toy_sd ** 2).sum() / len(lo) ** 2)
        R.loc[R.arm == arm, 'step_data'] = st_d
        R.loc[R.arm == arm, 'step_toy'] = st_t
        R.loc[R.arm == arm, 'step_toy_err'] = err
    R.to_csv(out / 'shear.csv', index=False)
    return R


def tof_step(out: Path) -> pd.DataFrame:
    """Background-subtracted y crossing against ToF (A, C)."""
    d = load_data()
    rows = []
    for arm, g in d.groupby('arm'):
        late = g[g.ms >= 20]
        for lo, hi in ((4, 6), (6, 8), (8, 10), (10, 12), (12, 15), (15, 20), (20, 30), (30, 45), (45, 75)):
            s = g[(g.ms >= lo) & (g.ms < hi)]
            r = bg_subtracted(s, late)
            e = estimators(s, per_height=False)
            rows.append(dict(arm=arm, lo=lo, hi=hi, n=len(s), y_raw=e['y_cross'], y_bgsub=r['y_cross'],
                             purity=r['purity'], x_cross=e['x_cross'], x_err=e['x_err']))
    R = pd.DataFrame(rows)
    R.to_csv(out / 'tof.csv', index=False)
    return R


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('step', nargs='?', default='all',
                    choices=['beam', 'closure', 'data', 'tof', 'shear', 'scan', 'fit', 'all'])
    a = ap.parse_args()
    out = paths.out('capsule_pose')
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    steps = (('beam', beam_step), ('closure', closure_step), ('data', data_step), ('tof', tof_step),
             ('shear', shear_step), ('scan', scan_step), ('fit', fit_step))
    for name, fn in steps:
        if a.step in (name, 'all'):
            print(f'\n== {name}\n{fn(out).round(3).to_string(index=False)}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
