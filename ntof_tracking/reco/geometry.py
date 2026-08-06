#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geometry.py — MX17 global geometry for track extrapolation & display.

Frame (matches MX17_Full_Geant, GEOMETRY_COORDINATE_CONVENTION.md):
  right-handed, origin at the He-3 target centre, beam = +Y (floor->ceiling
  at nTOF EAR2's vertical flight path), arms: A=+Z, B=-X, C=-Z, D=+X.
  Top-down view = screen x = Z (East), screen y = X (North).

Detector transforms are read from run_config.json (det_center_coords = strip
-plane centre [mm], det_orientation = rotation about Y). Verified against the
Geant build: centre = front_face + 30.1 mm * outward, pinwheel included.

Local frame per detector (consistent with those Y-rotations):
  x_local = strip-x coordinate (runs along the arm tangent u_hat)
  y_local = strip-y coordinate (runs along global +Y, the beam axis)
  z_local = outward normal; the 30 mm drift gap spans z_local in [-30, 0]
            with the strip/anode plane at 0 and the mylar cathode at -30.
  A hit at drift depth w (mm from the strips, inward) sits at z_local = -w.

ALIGNMENT CAVEAT: the sign/offset of the strip coordinates within the plane
(which end of the plane is strip 0, in the mounted orientation) is taken
from the mapping convention and has NOT been survey-verified. Treat in-plane
signs as provisional until inter-chamber links (TRACK_PLAN_04) pin them.

Surrounding elements (SiPM wall, plastics, LS box, He-3 capsule) are ported
from MX17_Full_Geant/scripts/plot_geometry.py (2026-07-15 stack) for display.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# constants ported from MX17_Full_Geant (cm -> mm)
# ---------------------------------------------------------------------------
MM_DIST_X = 204.0        # mylar front-face distance, +-X arms (D, B)
MM_DIST_Z = 204.5        # +-Z arms (A, C)
PINWHEEL = {'D': 15.5, 'B': 15.75, 'A': 16.35, 'C': 17.3}   # mm, along -u_hat
MM_SIZE_U = 380.0
MM_SIZE_V = 340.0
DRIFT_GAP = 30.0
W_STRIP = 30.1           # mylar front -> strip plane (40um mylar + cath + 30mm)

# ---- ACTIVE volumes only (SimConfig.hh / DetectorConstruction.cc) ----------
# Re-synced 2026-07-31 against the current Geant build (SimConfig.hh via
# scripts/plot_geometry.py) and cross-checked against the DAQ
# run_79/run_config.json, which places every scintillator in the global frame.
# Depths below are from the arm's mylar front face; the strip plane is at
# W_STRIP, so subtract 30.1 mm for "past the strips".
#
# SiPM wall: 20 bars x 25 mm on the STRUCTURE; 16 instrumented, window
# shifted 1 bar toward the MM = toward -u (DetectorConstruction.cc:620-632,
# bars 1..16 -> u in [-225, +175] mm). The run_config agrees: sipm_A_01..16
# sit at u = -212.5 .. +162.5 mm. NB plot_geometry.py in the Geant repo has
# the shift sign flipped (bars 3..18, u in [-175, +225]) — a bug in that
# DRAWING script only; the sim and the DAQ config agree with the table here.
SIPM_BAR_W = 25.0
SIPM_N_BARS = 20
SIPM_READ_BARS = list(range(1, 17))       # instrumented bar indices 0..19
SIPM_SCINT_W0, SIPM_SCINT_W1 = 126.0, 129.0   # 3 mm active scint, container mid
SIPM_CONT_W0, SIPM_CONT_W1 = 110.0, 145.0     # container (mostly empty)
SIPM_HALF_V = 250.0
# Plastics: two PVT bars 200x300x20 mm, centred on the MM, side by side in u.
# 20 mm, not 25: corrected in the sim on 2026-07-20. The SiPM-back -> plastic
# gap was surveyed per arm on 2026-07-17, so the depth is per arm too.
PLASTIC_W0 = {'D': 210.2, 'B': 206.2, 'A': 208.2, 'C': 206.2}   # active PVT
PLASTIC_THICK = 20.0
PLASTIC_HALF_U = 100.0                    # one bar
PLASTIC_HALF_V = 150.0
PLASTIC_U_OFFSET = 101.72                 # bar centre offset (wrap + 3 mm gap)
# LS: one LAB slab, 451x450x21.2 mm + a 12.5 mm bulge dome on each face,
# surveyed per arm 2026-07-17/18 and centred on the STRUCTURE (not the MM).
LS_W0 = {'D': 268.0, 'B': 272.0, 'A': 268.0, 'C': 272.0}
LS_THICK = 21.2
LS_BULGE = 12.5
LS_HALF_U = 225.6
LS_HALF_V = 225.3
LS_OFFSET_U = {'D': 8.3, 'B': -13.4, 'A': 6.3, 'C': -17.4}      # on structure
LS_OFFSET_V = {'D': -0.4, 'B': 0.3, 'A': 0.6, 'C': -0.7}


def plastic_span(arm: str) -> tuple:
    """(w0, w1) of the active PVT from the mylar front face."""
    return PLASTIC_W0[arm], PLASTIC_W0[arm] + PLASTIC_THICK


def sipm_bar_u(bar: int) -> tuple:
    """(u_lo, u_hi) of SiPM bar index `bar` (0..19) on the STRUCTURE."""
    c = SIPM_BAR_W * (bar - (SIPM_N_BARS - 1) / 2.0)
    return c - SIPM_BAR_W / 2, c + SIPM_BAR_W / 2

# He-3 target capsule ACTIVE gas: STEP-derived polycone profile, axis = +Y
# (plot_geometry.py Z_GAS/RO_GAS, cm -> mm)
HE3_GAS_Y = np.array([-29.50, -28.00, -26.00, -24.00, -22.00, -20.00, -15.00,
                      -5.00, 5.00, 15.00, 20.00, 22.00, 24.00, 26.00, 28.00,
                      30.00, 32.00, 34.00, 36.00, 38.00, 40.00, 44.00, 50.70])
HE3_GAS_R = np.array([0.001, 6.000, 8.000, 9.165, 9.798, 10.000, 10.000,
                      10.000, 10.000, 10.000, 10.000, 9.798, 9.165, 8.000,
                      6.299, 4.842, 3.660, 2.711, 1.967, 1.410, 1.026,
                      0.750, 0.750])
HE3_R_MAX = 10.0

ARMS = ['A', 'B', 'C', 'D']
DET_NAME = {a: f'mx17_{a}' for a in ARMS}
# outward normal (w_hat) and tangent (u_hat = local +x) per arm
W_HAT = {'A': np.array([0., 0., 1.]), 'B': np.array([-1., 0., 0.]),
         'C': np.array([0., 0., -1.]), 'D': np.array([1., 0., 0.])}
U_HAT = {'A': np.array([1., 0., 0.]), 'B': np.array([0., 0., 1.]),
         'C': np.array([-1., 0., 0.]), 'D': np.array([0., 0., -1.])}
V_HAT = np.array([0., 1., 0.])   # beam / vertical


def _rot_y(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


@dataclass
class DetTransform:
    name: str
    arm: str
    center: np.ndarray       # strip-plane centre, global mm
    R: np.ndarray            # local -> global rotation

    def local_to_global(self, xl, yl, w):
        """(x_local, y_local, drift depth w>=0 from strips) -> global mm.
        Accepts scalars or arrays; returns (..., 3)."""
        l = np.stack(np.broadcast_arrays(np.asarray(xl, float),
                                         np.asarray(yl, float),
                                         -np.asarray(w, float)), axis=-1)
        return self.center + l @ self.R.T


def detector_transforms(cfg: dict) -> Dict[str, DetTransform]:
    """Build transforms from run_config.json detector entries."""
    out = {}
    for det_cfg in cfg.get('detectors', []):
        name = det_cfg['name']
        if not name.startswith('mx17_'):
            continue
        arm = name.split('_')[1]
        c = det_cfg['det_center_coords']
        o = det_cfg.get('det_orientation', {}) or {}
        out[name] = DetTransform(
            name=name, arm=arm,
            center=np.array([c['x'], c.get('y', 0.0), c['z']], float),
            R=_rot_y(float(o.get('y', 0.0))),
        )
    return out


# ---------------------------------------------------------------------------
# drift model
# ---------------------------------------------------------------------------
# bench Ar/iso 95/5 (+H2O) drift curve, E [V/cm] -> v [um/ns] (det3, M3 v2)
_BENCH_E = np.array([233.3, 300.0, 333.3, 366.7])
_BENCH_V = np.array([23.31, 29.67, 34.30, 35.53])


@dataclass
class DriftModel:
    """v_drift + DAQ t0 for time -> depth conversion.

    t0_ns is the hit time at which charge born AT the strip plane arrives —
    a DAQ latency constant, initially estimated from the earliest edges of
    real tracks (run_48 evt 1107: ~450 ns); calibrate in situ (PLAN_05).
    """
    v_um_ns: float
    t0_ns: float = 450.0

    @property
    def v_mm_ns(self) -> float:
        return self.v_um_ns / 1000.0

    def depth_mm(self, t_ns):
        return (np.asarray(t_ns, float) - self.t0_ns) * self.v_mm_ns

    @classmethod
    def from_drift_hv(cls, drift_hv: float, t0_ns: float = 450.0,
                      gap_mm: float = DRIFT_GAP) -> 'DriftModel':
        e = drift_hv / (gap_mm / 10.0)   # V/cm
        return cls(v_um_ns=float(np.interp(e, _BENCH_E, _BENCH_V)), t0_ns=t0_ns)


# ---------------------------------------------------------------------------
# 3D segments -> global lines, pointing metrics
# ---------------------------------------------------------------------------
def segment_to_global(seg3d: dict, tr: DetTransform) -> dict:
    """Attach global-frame endpoints/direction to a pairing.pair_xy_3d dict."""
    p_lo = tr.local_to_global(*seg3d['p_lo_local'][:2], seg3d['p_lo_local'][2])
    p_hi = tr.local_to_global(*seg3d['p_hi_local'][:2], seg3d['p_hi_local'][2])
    d = p_hi - p_lo
    n = np.linalg.norm(d)
    d = d / n if n > 0 else d
    seg3d = dict(seg3d)
    seg3d.update(p_lo_global=p_lo, p_hi_global=p_hi, dir_global=d)
    # pointing diagnostics
    seg3d['dca_origin_mm'] = point_line_dist(np.zeros(3), p_lo, d)
    seg3d['dca_beam_axis_mm'] = line_line_dist(
        np.zeros(3), V_HAT, p_lo, d)
    seg3d['angle_to_vertical_deg'] = float(
        np.degrees(np.arccos(min(1.0, abs(float(d @ V_HAT))))))
    # beam-axis projection: Y of the point on the beam axis closest to the
    # track line (the track's apparent emission height along the beamline)
    b = float(d @ V_HAT)
    den = 1.0 - b * b
    if den > 1e-9:
        d0 = float(p_lo @ d)
        e0 = float(p_lo @ V_HAT)
        seg3d['beam_y_mm'] = float((e0 - b * d0) / den)
    else:
        seg3d['beam_y_mm'] = np.nan   # track parallel to the beam axis
    return seg3d


def point_line_dist(pt, p0, d) -> float:
    v = np.asarray(pt, float) - p0
    return float(np.linalg.norm(v - (v @ d) * d))


def line_line_dist(p1, d1, p2, d2) -> float:
    n = np.cross(d1, d2)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return point_line_dist(p2, p1, d1 / np.linalg.norm(d1))
    return float(abs((np.asarray(p2, float) - p1) @ n) / nn)


# ---------------------------------------------------------------------------
# active-volume boxes (display + crossing tests)
# ---------------------------------------------------------------------------
def arm_front_face(arm: str, on: str = 'mm') -> np.ndarray:
    d = MM_DIST_X if arm in ('B', 'D') else MM_DIST_Z
    ff = W_HAT[arm] * d
    if on == 'mm':
        ff = ff - U_HAT[arm] * PINWHEEL[arm]
    return ff


def arm_active_volumes(arm: str) -> List[dict]:
    """ACTIVE volumes of one arm: {name, kind, on, u_lo, u_hi, w0, w1,
    half_v} in the arm's (u, w) frame. 'on' = 'mm' (pinwheel-shifted with the
    MM) or 'struct' (mechanical structure centre). Passive material (PCB,
    wraps, containers, CFRP) is intentionally absent."""
    els = [dict(name='MM drift gas', kind='mm', on='mm',
                u_lo=-MM_SIZE_U / 2, u_hi=MM_SIZE_U / 2,
                w0=W_STRIP - DRIFT_GAP, w1=W_STRIP, half_v=MM_SIZE_V / 2)]
    for i in SIPM_READ_BARS:
        u_c = SIPM_BAR_W * (i - (SIPM_N_BARS - 1) / 2.0)
        els.append(dict(name=f'SiPM bar {i:02d}', kind='sipm', on='struct',
                        u_lo=u_c - SIPM_BAR_W / 2, u_hi=u_c + SIPM_BAR_W / 2,
                        w0=SIPM_SCINT_W0, w1=SIPM_SCINT_W1,
                        half_v=SIPM_HALF_V))
    w0, w1 = plastic_span(arm)
    for side, tag in ((-1.0, 'L'), (+1.0, 'R')):
        u_c = side * PLASTIC_U_OFFSET
        els.append(dict(name=f'plastic {tag}', kind='plastic', on='mm',
                        u_lo=u_c - PLASTIC_HALF_U, u_hi=u_c + PLASTIC_HALF_U,
                        w0=w0, w1=w1, half_v=PLASTIC_HALF_V))
    u_c = LS_OFFSET_U[arm]
    els.append(dict(name='LS', kind='ls', on='struct',
                    u_lo=u_c - LS_HALF_U, u_hi=u_c + LS_HALF_U,
                    w0=LS_W0[arm] - LS_BULGE,
                    w1=LS_W0[arm] + LS_THICK + LS_BULGE,
                    half_v=LS_HALF_V))
    return els


def element_aabbs() -> List[dict]:
    """All active volumes as global axis-aligned boxes {arm, name, lo, hi}
    [mm]. Every arm's axes coincide with global axes, so AABBs are exact.
    Includes the He-3 gas (bounding box of the polycone) as arm='target'."""
    out = []
    for arm in ARMS:
        for el in arm_active_volumes(arm):
            ff = arm_front_face(arm, el['on'])
            u, w = U_HAT[arm], W_HAT[arm]
            c0 = ff + w * el['w0'] + u * el['u_lo']
            c1 = ff + w * el['w1'] + u * el['u_hi']
            half_v = np.abs(V_HAT) * el['half_v']
            lo = np.minimum(c0, c1) - half_v
            hi = np.maximum(c0, c1) + half_v
            out.append(dict(arm=arm, name=el['name'], lo=lo, hi=hi))
    out.append(dict(
        arm='target', name='He3 gas',
        lo=np.array([-HE3_R_MAX, HE3_GAS_Y[0], -HE3_R_MAX]),
        hi=np.array([HE3_R_MAX, HE3_GAS_Y[-1], HE3_R_MAX])))
    return out


def orient_outward(gseg: dict):
    """Orient a global 3D segment's direction to point OUTWARD from the beam
    axis (the physics prior: particles radiate from the beamline). Returns
    (p_mid, d_out, s_beam) with s_beam = signed distance along d_out from
    p_mid to the closest approach to the beam axis (always <= 0)."""
    p_mid = 0.5 * (gseg['p_lo_global'] + gseg['p_hi_global'])
    d = gseg['dir_global']
    den = d[0] ** 2 + d[2] ** 2
    s_beam = (-(p_mid[0] * d[0] + p_mid[2] * d[2]) / den) if den > 1e-12 else 0.0
    if s_beam > 0:          # beam axis lies ahead -> flip so it lies behind
        d = -d
        s_beam = -s_beam
    return p_mid, d, s_beam


def split_crossings(gseg: dict) -> dict:
    """Crossings of a segment's line, split by the beamline-origin prior:
      'outward'  — from the beam-axis closest approach onward through the
                   measured chamber and beyond (the plausible particle path);
      'backward' — behind the beam axis (the opposite side of the setup):
                   geometric line extension only, NOT a claim the particle
                   went there (a real through-goer must show hits there).
    """
    p_mid, d_out, s_beam = orient_outward(gseg)
    out, back = [], []
    for c in line_crossings(p_mid, d_out):
        (out if 0.5 * (c['s_in'] + c['s_out']) >= s_beam else back).append(c)
    return dict(outward=out, backward=back)


def line_crossings(p0: np.ndarray, d: np.ndarray) -> List[dict]:
    """Elements the INFINITE line p0 + s*d crosses (slab ray-box test, both
    directions — use split_crossings for the physical-path split). Returns
    [{arm, name, s_in, s_out, p_in, p_out}] sorted by |s at entry|."""
    hits = []
    for box in element_aabbs():
        s_lo, s_hi = -np.inf, np.inf
        ok = True
        for k in range(3):
            if abs(d[k]) < 1e-12:
                if not (box['lo'][k] <= p0[k] <= box['hi'][k]):
                    ok = False
                    break
                continue
            a = (box['lo'][k] - p0[k]) / d[k]
            b = (box['hi'][k] - p0[k]) / d[k]
            s_lo = max(s_lo, min(a, b))
            s_hi = min(s_hi, max(a, b))
        if ok and s_lo <= s_hi:
            hits.append(dict(arm=box['arm'], name=box['name'],
                             s_in=float(s_lo), s_out=float(s_hi),
                             p_in=p0 + s_lo * d, p_out=p0 + s_hi * d))
    return sorted(hits, key=lambda h: abs(h['s_in']))
