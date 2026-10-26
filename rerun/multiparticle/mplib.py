"""Shared inputs and geometry for the multi-particle study (rerun/multiparticle).

Frames. Every det3 quantity here is in the RAW strip frame of the chamber (x, y = strip coordinates, mm), with the
M3 ray already carried into it (rerun/effbudget/budget.py: mesh_* = the ray at the mesh plane, tan_* = its raw-frame
tangents). The vertical coordinate Z is the M3 one (mm, up; M3 planes at 24/144 below and 1185/1302 above, det3's mesh
at Z0 = 715, the alignment's z_x). A line is pos(Z) = p0 + tan * (Z - Z0). The M3 tangent is d pos / d Z
(M3RefTracking.get_xy_angles: arctan((x_down - x_up) / (z_down - z_up))), and the wft fit's ladder-scaled tangent tl =
(1e3 w - c) / s is the same quantity (the ladder is fitted against it), so the two lines compose directly.

Geometry comes from the forward waveform fit (wft final_phys candidates); combined_hits times are never used.
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
SAT = os.path.join(REPO, 'rerun', 'satstudy'); OLD = os.path.join(REPO, 'rerun', 'outliers')
sys.path[:0] = [HERE, SAT, OLD]
import satfit as S  # noqa: E402  (also puts the repo, qa_config and cosmic_bench_analysis on the path)
import multitrack as MT  # noqa: E402
from olib import load  # noqa: E402

OUT = os.path.expanduser('~/x17/cosmic_bench/satstudy/multiparticle')
OLB = os.path.expanduser('~/x17/cosmic_bench/outliers')
ALIGN = os.path.join(OLB, 'final/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3/wft/'
                          'final_phys/alignment/alignment.json')
Z0 = 715.0
ZM3 = (1302.0, 1185.0, 144.0, 24.0)
GAP = 30.0
os.makedirs(OUT, exist_ok=True)


def ladder():
    return load(S.DET).attrs['ladder']


def tl(w, view, L):
    s, c = L[view]
    return (np.asarray(w, float) * 1e3 - c) / s


def rays_all():
    return pd.read_parquet(MT.RAYS).set_index('event_id')


def rays():
    return MT.rays()


def candidates(L=None):
    """Production candidates (rank >= 0) with time extent (multitrack.feat), particle flag and ladder tangent."""
    L = L or ladder()
    C = pd.read_parquet(MT.CAND); C = C[C['rank'] >= 0]
    C = C.merge(pd.read_parquet(MT.FEAT), on=['event_id', 'plane', 'rank'], how='left')
    C['particle'] = C.ext.fillna(99) <= MT.EXT_MAX
    C['tl'] = np.where(C.plane == 'x', tl(C.w, 'x', L), tl(C.w, 'y', L))
    C['mid'] = C.p0 + 0.5 * GAP * C.tl
    return C


def m3feat():
    return pd.read_parquet(os.path.join(OLB, 'm3feat_det3.parquet')).set_index('event_id')


def align():
    return json.load(open(ALIGN))


def raw_to_m3(px, py, a=None):
    """Raw det frame point at the DUT -> M3 frame (inverse of m3feat.to_raw / budget.rays_table)."""
    a = a or align(); th = np.deg2rad(a['theta_deg']); ct, st = np.cos(th), np.sin(th)
    dx, dy = np.asarray(px) - a['centre_x'], np.asarray(py) - a['centre_y']
    u = ct * dx - st * dy; w = st * dx + ct * dy
    return a['ref_x_sign'] * (u + a['centre_x'] + a['x_offset']), w + a['centre_y'] + a['y_offset']


def tan_raw_to_m3(tx, ty, a=None):
    a = a or align(); th = np.deg2rad(a['theta_deg']); ct, st = np.cos(th), np.sin(th)
    return a['ref_x_sign'] * (ct * np.asarray(tx) - st * np.asarray(ty)), st * np.asarray(tx) + ct * np.asarray(ty)


def m3_boxes(RA, a=None):
    """Per M3 plane Z, the region every recipe ray crossed (0.1 / 99.9 % of ray positions, M3 frame)."""
    a = a or align()
    xm, ym = raw_to_m3(RA.mesh_x.to_numpy(), RA.mesh_y.to_numpy(), a)
    txm, tym = tan_raw_to_m3(RA.tan_x.to_numpy(), RA.tan_y.to_numpy(), a)
    out = {}
    for z in ZM3:
        X = xm + txm * (z - Z0); Y = ym + tym * (z - Z0); ok = np.isfinite(X) & np.isfinite(Y)
        out[z] = (np.percentile(X[ok], [0.1, 99.9]), np.percentile(Y[ok], [0.1, 99.9]))
    return out


def dca(p1, d1, p2, d2):
    """Closest approach of two 3-D lines p + t d (arrays (...,3)). Returns distance, Z of the midpoint of the
    closest-approach segment, and the 3-D opening angle [deg]."""
    p1, d1, p2, d2 = map(lambda v: np.asarray(v, float), (p1, d1, p2, d2))
    w0 = p1 - p2
    a = (d1 * d1).sum(-1); b = (d1 * d2).sum(-1); c = (d2 * d2).sum(-1); d = (d1 * w0).sum(-1); e = (d2 * w0).sum(-1)
    den = a * c - b * b
    with np.errstate(invalid='ignore', divide='ignore'):
        s = (b * e - c * d) / den; t = (a * e - b * d) / den
    q1 = p1 + s[..., None] * d1; q2 = p2 + t[..., None] * d2
    cosang = np.clip(b / np.sqrt(a * c), -1, 1)
    return np.linalg.norm(q1 - q2, axis=-1), 0.5 * (q1[..., 2] + q2[..., 2]), np.degrees(np.arccos(cosang))


def line(p0x, p0y, tx, ty):
    p = np.stack([np.asarray(p0x, float), np.asarray(p0y, float), np.full(np.shape(p0x), Z0)], -1)
    d = np.stack([np.asarray(tx, float), np.asarray(ty, float), np.ones(np.shape(p0x))], -1)
    return p, d


def rsig(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float('nan')
