"""common.py -- shared loaders and fits for the chamber-bow follow-up (FB, model audit 2026-10-27).

Frame and signs (pinned to external facts, see REPORT.md "Sign"):
  * M3 z points UP: the M3 ray files carry Z_Up = 1302 mm (top plane) and Z_Down = 24 mm (bottom plane), and the
    bench's upper DUT slot (P2) is at z = 702 mm, the lower (P1) at z = 232 mm.
  * tan_ref = d(position)/dz_up in each view's strip frame (M3 angle = atan(dx/dz); the alignment maps positions and
    tangents with the same rotation and the same ref_x_sign, build_table.py).
  * The chambers sit with the readout board DOWN and the cathode UP (the micro-TPC slope w has the SAME sign as tan_ref
    on every chamber, kw > 0: charge arriving later comes from higher up), so electrons drift DOWN onto the mesh.
  * h(u, w) = height of the mesh surface along z_up relative to the plane the alignment assumes: the position residual
    r = p0 - ref_mesh of a track with slope tan is r = tan * h  ->  h = dr/dtan.  h smaller at the centre = the mesh
    centre sits LOWER than its rim = further from the cathode.
  * A drift field normal to a mesh surface with local slope dh/du sends charge from height d to u + d*dh/du, so the
    micro-TPC transverse speed gains w0 = +v * r_f * dh/du, with r_f = (1 + f)/2 when the cathode follows a fraction f of
    the mesh bow (f = 0: flat cathode, the field tilt falls linearly from the mesh to the cathode; f = 1: parallel bow).
Coordinates: u = track x at mid-gap (strip frame of the X view), w = track y at mid-gap; U = (u - 200)/100, W likewise.
"""
import json
import os
import numpy as np
import pandas as pd

ROOT = os.path.expanduser('~/x17/cosmic_bench/model_audit')
FB = os.path.join(ROOT, 'fb')
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
RES = os.path.join(HERE, 'results')
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
COL = {'det2': '#1f77b4', 'det3': '#2ca02c', 'det4': '#d62728', 'det6': '#9467bd', 'det7': '#ff7f0e'}
GAP = 30.0
MID = 15.0

# dataset name -> (chamber, parquet, slot, note).  Golden rows: ws_v_w0 tables (finalize reco + M3 + alignment).
DATASETS = {
    'det2': ('det2', f'{ROOT}/ws_v_w0/det2.parquet', 'top', '6-22 longer, 495 V mesh, 1000 V drift'),
    'det3': ('det3', f'{ROOT}/ws_v_w0/det3.parquet', 'top', '6-27 sat long, 490 / 1000 V'),
    'det4': ('det4', f'{ROOT}/ws_v_w0/det4.parquet', 'top', '6-24 day long, 495 / 600 V'),
    'det6': ('det6', f'{ROOT}/ws_v_w0/det6.parquet', 'bottom', '6-26 night long, 495 / 700 V'),
    'det7': ('det7', f'{ROOT}/ws_v_w0/det7.parquet', 'top', '6-26 night long, 495 / 700 V'),
    'det3_day625': ('det3', f'{ROOT}/ws_v_w0/det3_day625.parquet', 'bottom', '6-25 day, BOTTOM slot, 495 / 500 V'),
    'det3_d700': ('det3', f'{ROOT}/ws_v_w0/det3_d700.parquet', 'top', '6-27 drift scan 700 V (small)'),
    'det3_d500': ('det3', f'{ROOT}/ws_v_w0/det3_d500.parquet', 'top', '6-27 drift scan 500 V (small)'),
}
# consistency rows reconstructed for FB (make_fb_dag.py), tables built by build_fb_tables.py
FB_ROWS = {
    'det2_o22long': ('det2', 'top', '6-22 long (same night, other subrun)'),
    'det2_o17': ('det2', 'top', '6-17 night (remounted since)'),
    'det3_p2': ('det3', 'top', '6-28 night (same mount)'),
    'det3_o23': ('det3', 'bottom', '6-23 night, BOTTOM slot, 600 V drift'),
    'det4_o23': ('det4', 'top', '6-23 night'),
    'det6_longer': ('det6', 'bottom', '6-26 longer (same night)'),
    'det7_longer': ('det7', 'top', '6-26 longer (same night)'),
    'det6_hv505': ('det6', 'bottom', 'mesh 505 V'),
    'det7_hv480': ('det7', 'top', 'mesh 480 V'),
    'det6_hv530': ('det6', 'bottom', 'mesh 530 V'),
    'det7_hv505': ('det7', 'top', 'mesh 505 V'),
}
for _n, (_d, _s, _note) in FB_ROWS.items():
    _p = f'{FB}/tables/{_n}.parquet'
    if os.path.exists(_p):
        DATASETS[_n] = (_d, _p, _s, _note)


def bundle(det):
    return json.load(open(f'{ROOT}/{det}/calib_bundle/bundle.json'))


def load(name):
    """Per-event table with the bow observables of both views (u, w at mid-gap from M3)."""
    det, path, slot, note = DATASETS[name]
    return load_path(path, bundle(det), det, slot, note, name)


def load_path(path, b, det, slot='', note='', name=''):
    """load() for any joined table (ws_v_w0/build_table layout) and bundle dict b (its v_drift, kw, w0)."""
    d = pd.read_parquet(path)
    if 'event_id' in d.columns:
        d = d.set_index('event_id')
    out = pd.DataFrame(index=d.index)
    out['u'] = d['ref_mesh_x'] + MID * d['tan_ref_x']
    out['w'] = d['ref_mesh_y'] + MID * d['tan_ref_y']
    out['u_mesh'], out['w_mesh'] = d['ref_mesh_x'], d['ref_mesh_y']
    out['u_cath'] = d['ref_mesh_x'] + GAP * d['tan_ref_x']
    out['w_cath'] = d['ref_mesh_y'] + GAP * d['tan_ref_y']
    for p in 'xy':
        out[f'{p}_t'] = d[f'tan_ref_{p}']
        out[f'{p}_r'] = d[f'{p}_p0'] - d[f'ref_mesh_{p}']
        out[f'{p}_wv'] = d[f'{p}_w'] * 1e3
        for c in ('p0', 't0', 'q_u50', 'q_u90', 'q_uend', 'q_sum', 'n_seed', 'n_strips', 'chi2', 'dof', 'ftst', 'ok',
                  'n_flagged_strips'):
            if f'{p}_{c}' in d.columns:
                out[f'{p}_{c}'] = d[f'{p}_{c}']
    out.attrs.update(det=det, slot=slot, note=note, v=b['v_drift'], kw=b['kw'], w0=b['w0'], name=name)
    return out


def view_mask(T, p, tmax=0.45, rcut=3.0):
    t, r = T[f'{p}_t'].to_numpy(), T[f'{p}_r'].to_numpy()
    ok = np.isfinite(t) & np.isfinite(r) & np.isfinite(T.u.to_numpy()) & np.isfinite(T.w.to_numpy())
    ok &= np.abs(t) < tmax
    if f'{p}_ok' in T:
        ok &= T[f'{p}_ok'].fillna(False).astype(bool).to_numpy()
    med = np.nanmedian(r[ok])
    ok &= np.abs(r - med) < rcut
    return ok


def huber(X, y, scale, it=15):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(it):
        r = np.abs(X @ beta - y)
        sw = np.sqrt(np.where(r <= scale, 1.0, scale / np.maximum(r, 1e-12)))
        beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    return beta


def boot(fn, n, nboot=30, seed=1):
    rng = np.random.default_rng(seed)
    return np.array([fn(rng.integers(0, n, n)) for _ in range(nboot)])


def surf_terms(U, W):
    """h(U, W) basis: [1, U, W, U^2/2, W^2/2, U*W] (U, W in units of 100 mm)."""
    return np.c_[np.ones_like(U), U, W, 0.5 * U * U, 0.5 * W * W, U * W]


SURF = ['h0', 'gu', 'gw', 'kuu', 'kww', 'kuw']   # kXX in mm per (100 mm)^2;  x 100 -> 1/km
