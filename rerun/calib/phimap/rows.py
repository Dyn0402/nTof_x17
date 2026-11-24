"""rows.py -- registry rows, their pass-1 gate tables, bundles and alignments (bench re-finalization step 3).

Layout of the staged inputs (read-only): $PASS1/<run>/<subrun>/<det>/{bow_gate_table.parquet, events.parquet,
alignment/alignment.json, calib_bundle/ (calibration rows only)}.  A row's bundle is the one of its ``calib_key`` row, its
alignment that of ``align_key`` (or calib_key).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)
from common import mx17_active_area as AA  # noqa: E402

JOB = os.environ.get('PHIMAP_JOB', '/media/ucla/work/claude-jobs/d006ab29')
PASS1 = f'{JOB}/pass1_all'
OUTDIR = os.environ.get('PHIMAP_OUT', f'{JOB}/phimap')
MID = 14.0            # mm, the mid-gap depth of the reference position (as d_lib / gapmap)
MARGIN = 30.0         # mm, interior margin of the selection (as d_lib)
SAMPLE_NS = 60.0


def _full():
    r = pd.read_csv(os.path.join(REPO, 'rerun', 'pass_registry.csv'))
    r['calib_key'] = r['calib_key'].fillna(r['key'])
    r['align_key'] = r['align_key'].where(r['align_key'].notna(), r['calib_key'])
    r['short'] = r.run.str.replace('mx17_', '', regex=False).str.replace('-26', '', regex=False) + '/' + r.subrun
    return r.set_index('key', drop=False)


_FULL = None


def full():
    global _FULL
    if _FULL is None:
        _FULL = _full()
    return _FULL


def reg():
    """The 156 non-excluded rows."""
    f = full()
    return f[f.cls != 'excluded']


def rowdir(key):
    r = full().loc[key]
    return f'{PASS1}/{r.run}/{r.subrun}/{r.det}'


def table_path(key):
    return rowdir(key) + '/bow_gate_table.parquet'


def bundle_path(key):
    return rowdir(full().loc[key].calib_key) + '/calib_bundle/bundle.json'


def bundle(key):
    return json.load(open(bundle_path(key)))


def alignment(key):
    return json.load(open(rowdir(full().loc[key].align_key) + '/alignment/alignment.json'))


def active(det, margin=MARGIN):
    a = AA.TRUE_ACTIVE_BY_DET[det]
    return (a['x'][0] + margin, a['x'][1] - margin), (a['y'][0] + margin, a['y'][1] - margin)


COLS = ['n_tracks', 'x_ok', 'y_ok', 'x_quality_ok', 'y_quality_ok', 'spark', 'Chi2X', 'Chi2Y', 'NClusX', 'NClusY',
        'x_w', 'y_w', 'x_p0', 'y_p0', 'x_tan_theta', 'y_tan_theta', 'tan_ref_x', 'tan_ref_y', 'ref_mesh_x', 'ref_mesh_y',
        'x_q_uend', 'y_q_uend', 'x_q_u90', 'y_q_u90', 'x_q_sum', 'y_q_sum', 'x_n_sat_strips', 'y_n_sat_strips',
        'timestamp', 'acq_tag']
NB = 5   # time blocks per row (outer cross-validation)


def load_full(key):
    """The whole gate table of a row (columns COLS), with ``ridx`` (table index), ``blk`` (time block 0..NB-1 of the row, from
    the rank in (acq_tag, timestamp) over ALL table events), ``sel`` (the gate selection of d_lib.load: single track, both
    views ok and quality_ok, no spark, M3 chi2 < 1, >= 4 clusters per view, reference mid-gap position inside the chamber's
    30 mm interior; plus |tan_ref| <= 0.6, never binding), ``cens_x`` / ``cens_y`` (that view's q_uend reaches the end of
    the fit's depth grid, K x 60 ns).  Tables whose n_tracks is 0 for every event (det3 on 6-23 and the 100 V drift scan:
    the column is not filled there) are selected without the n_tracks == 1 condition; ``ntracks_unavailable`` in attrs."""
    t = pd.read_parquet(table_path(key), columns=COLS)
    t['ridx'] = np.arange(len(t))
    o = np.lexsort((t.timestamp.values.astype(float), t.acq_tag.values.astype(str)))
    rank = np.empty(len(t), int)
    rank[o] = np.arange(len(t))
    t['blk'] = np.minimum((rank * NB) // len(t), NB - 1)
    t['trank'] = rank
    nt_ok = bool((t.n_tracks == 1).any())
    ok1 = (t.n_tracks == 1) if nt_ok else True
    s = (ok1 & t.x_ok & t.y_ok & t.x_quality_ok & t.y_quality_ok & ~t.spark
         & (t.Chi2X < 1) & (t.Chi2Y < 1) & (t.NClusX >= 4) & (t.NClusY >= 4))
    t['xm'] = t.ref_mesh_x + MID * t.tan_ref_x
    t['ym'] = t.ref_mesh_y + MID * t.tan_ref_y
    t['gate'] = (s & (np.abs(t.tan_ref_x) <= 0.6) & (np.abs(t.tan_ref_y) <= 0.6)).values   # the gate without the interior condition
    ix, iy = active(full().loc[key].det)
    s &= (t.xm > ix[0]) & (t.xm < ix[1]) & (t.ym > iy[0]) & (t.ym < iy[1])
    s &= (np.abs(t.tan_ref_x) <= 0.6) & (np.abs(t.tan_ref_y) <= 0.6)
    t['sel'] = s.values
    ge = grid_end(key)
    t['cens_x'] = (t.x_q_uend >= ge - 1).values
    t['cens_y'] = (t.y_q_uend >= ge - 1).values
    t.attrs['ntracks_unavailable'] = not nt_ok
    return t


def load(key, xhalf=False):
    """Selected events of a row in time order (see load_full).  xhalf: reference x >= 200 mm only."""
    t = load_full(key)
    d = t[t.sel]
    if xhalf:
        d = d[d.xm >= 200.0]
    d = d.sort_values('trank').reset_index(drop=True)
    d.attrs.update(t.attrs)
    return d


def grid_end(key):
    """End of the fit's depth grid [ns] = K x 60 ns, K from the registry (24, or 30 at <= 500 V)."""
    return float(full().loc[key].K) * SAMPLE_NS
