"""common.py -- GB shared loaders: M3 rays (v2, per single-track event), the FD measured angle basis per chamber
(final_physmask per_event.npz + calib_bundle w0/kw/v, exactly as fd_floor_realism/measured_chi2.py), the linear map
from the M3 frame to each chamber's strip frame (fitted on the matched events, so no alignment file is needed), and
the FD s68 convention.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
AUDIT = os.path.expanduser('~/x17/cosmic_bench/model_audit')
BENCH = os.path.expanduser('~/x17/cosmic_bench')
M3_Z = np.array([1302.0, 1185.0, 144.0, 24.0])           # L0..L3, run_config.json (muons go down: L0 is the top)
DUT_Z = {'det2': 702.0, 'det3': 702.0, 'det4': 702.0, 'det6': 232.0, 'det7': 702.0}
RAYS = {
    'det6_det7': f'{BENCH}/det6_det7/mx17_det6_det7_overnight_6-26-26/long_run/m3_tracking_root_v2',
    'det2': f'{BENCH}/det2_det3/mx17_det2_det3_overnight_6-22-26/longer_run/m3_tracking_root_v2',
    'det3': f'{BENCH}/det3/mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/m3_tracking_root_v2',
    'det4': f'{BENCH}/det4_day/mx17_det4_day_6-24-26/long_run/m3_tracking_root_v2',
}
RAYS['det6'] = RAYS['det7'] = RAYS['det6_det7']


def s68(a):
    """FD convention: clip |a| <= 5 deg, 68th percentile of |a - median|."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a) & (np.abs(a) <= 5)]
    return float(np.percentile(np.abs(a - np.median(a)), 68)) if len(a) > 20 else np.nan


def load_rays(path, stop=None):
    """Single-track M3 events -> DataFrame(evn, tX, tY [dx/dz in the M3 frame], chi2X/Y, nX/nY)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(path, '*.root')))[:stop]:
        t = uproot.open(f)
        tr = t[sorted(t.keys(), key=lambda k: int(k.split(';')[1]))[-1]]
        br = [b for b in ('evn', 'rayN', 'Z_Up', 'Z_Down', 'X_Up', 'Y_Up', 'X_Down', 'Y_Down', 'Chi2X', 'Chi2Y',
                          'NClusX', 'NClusY') if b in tr.keys()]
        a = tr.arrays(br, library='ak')
        import awkward as ak
        one = ak.num(a['Chi2X']) == 1
        a = a[one]
        d = {'evn': ak.to_numpy(a['evn']).astype(np.int64)}
        for k in br[2:]:
            d[k] = ak.to_numpy(ak.firsts(a[k])) if a[k].ndim > 1 else ak.to_numpy(a[k])
        rows.append(pd.DataFrame(d))
    r = pd.concat(rows, ignore_index=True)
    dz = r['Z_Down'] - r['Z_Up']
    r['tX'] = (r['X_Down'] - r['X_Up']) / dz
    r['tY'] = (r['Y_Down'] - r['Y_Up']) / dz
    r = r.rename(columns={'Chi2X': 'chi2X', 'Chi2Y': 'chi2Y', 'NClusX': 'nX', 'NClusY': 'nY'})
    r = r.drop_duplicates('evn', keep=False)
    return r.set_index('evn')


def load_dut(det, view):
    """FD measured basis: eid, tan_ref (production, chi2<1 & NClus 4), tan_fit, the events row."""
    z = np.load(os.path.join(AUDIT, det, 'final_physmask', 'angles', 'per_event.npz'))
    ev = pd.read_parquet(os.path.join(AUDIT, det, 'final_physmask', 'events.parquet'))
    ev = ev[~ev['event_id'].duplicated(keep=False)].set_index('event_id')
    b = json.load(open(os.path.join(AUDIT, det, 'calib_bundle', 'bundle.json')))
    eid = z[f'{view}_eid']
    tr = z[f'{view}_tan_ref']
    tf = (z[f'{view}_w'] * 1e3 - b['w0'][view]) / (b['kw'][view] * float(b['v_drift']))
    return pd.DataFrame({'tan_ref': tr, 'tan_fit': tf}, index=pd.Index(eid, name='eid')), ev, b


def tan_fit_all(ev, b, view):
    """tan_fit for every fitted event of the table (not only the reference-matched ones)."""
    return (ev[f'{view}_w'] * 1e3 - b['w0'][view]) / (b['kw'][view] * float(b['v_drift']))


def fit_map(tan_ref, rays_sub):
    """tan_ref (strip frame) = a*tX + b*tY + c, least squares on matched events; returns coeffs, rms residual."""
    A = np.c_[rays_sub['tX'].to_numpy(), rays_sub['tY'].to_numpy(), np.ones(len(rays_sub))]
    c, *_ = np.linalg.lstsq(A, np.asarray(tan_ref), rcond=None)
    return c, float(np.std(np.asarray(tan_ref) - A @ c))


def apply_map(c, tX, tY):
    return c[0] * np.asarray(tX) + c[1] * np.asarray(tY) + c[2]


def deg(t):
    return np.degrees(np.arctan(np.asarray(t, float)))
