#!/usr/bin/env python
"""angles_cv.py -- the map1 angles of a registry row, by event id, for requoting numbers on another selection (step 5, Paper I).

Paper I quotes angle resolutions on its own selection (effbudget 03_angles ``interior_good``, one per_event.npz per row with
``<v>_eid``, ``<v>_tan_ref``, ``<v>_tan_fit``).  This module gives, for the same events, the angle of the new tables:

  p1    pass-1 tangent (x_tan_pass1)
  cv    THE QUOTED ANGLE: step 3's five-time-block cross-validated value (``<v>_tan_final_cv``: map and row constants
        trained without the event's time block, railed-track rescale of C) where the map is used for the event
        (``map_used_<v>``: inside the fit domain, the 30 mm interior), pass 1 elsewhere
  prod  the shipped table column (production map, in sample for the training events)
  used, railed, excl  flags (excl = railed track of chamber D or E, left out of the quoted numbers by decision 11-24)

Every event of the row's table has a CV value (the folds are time blocks of the whole table), so events of the paper's selection
that the step-3 gate selection did not contain are out of sample as well.

Inputs (read only): $PHIMAP_JOB (default /media/ucla/work/claude-jobs/d006ab29) /map1/<run>/<subrun>/<det>/events.parquet,
/phimap/final/<key>.parquet, /pass1_all/.../bow_gate_table.parquet (event order).

    from angles_cv import row_angles, restat
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import rows as R  # noqa: E402

JOB = R.JOB
MAP1 = os.environ.get('MAP1_OUT', f'{JOB}/map1')
FINAL = f'{JOB}/phimap/final'
P1RES = f'{JOB}/p1res'      # per_event.npz of the paper's selection, fetched from EOS (paper1_eff/angles)
EOS_ANGLES = '/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_eff/angles'
XRD_PY = os.path.expanduser('~/.venvs/xrd/bin/python')
EXCL_CHAMBERS = ('D', 'E')


def split_key(key):
    r = R.full().loc[key]
    return r.run, r.subrun, r.det


def row_angles(key):
    """DataFrame indexed by event_id (unique): per view p1, cv, prod, used, railed, excl, resc."""
    run, sub, det = split_key(key)
    od = f'{MAP1}/{run}/{sub}/{det}'
    e = pd.read_parquet(f'{od}/events.parquet')
    t = pd.read_parquet(R.table_path(key), columns=['acq_tag'])
    f = pd.read_parquet(f'{FINAL}/{key}.parquet')
    assert len(f) == len(t) == len(e), (key, len(f), len(t), len(e))
    assert (t.index.values == e.event_id.values).all(), key
    assert t.index.is_unique, key
    chamber = pd.read_csv(f'{R.OUTDIR}/mountings.csv').set_index('key').loc[key, 'chamber']
    out = pd.DataFrame(index=pd.Index(e.event_id.values, name='event_id'))
    for v in 'xy':
        used = e[f'map_used_{v}'].values.astype(bool)
        p1 = e[f'{v}_tan_pass1'].values
        out[f'{v}_p1'] = p1
        out[f'{v}_cv'] = np.where(used, f[f'{v}_tan_final_cv'].values, p1)
        out[f'{v}_prod'] = e[f'{v}_tan_theta'].values
        out[f'{v}_used'] = used
        out[f'{v}_railed'] = e[f'{v}_railed'].values.astype(bool)
        out[f'{v}_excl'] = out[f'{v}_railed'] & (chamber in EXCL_CHAMBERS)
    out.attrs['chamber'] = chamber
    return out


def fetch_npz(key, dest=None):
    """per_event.npz of the row's paper selection (pass 1, paper1_eff/angles) from EOS, cached under $PHIMAP_JOB/p1res."""
    run, sub, det = split_key(key)
    dest = dest or f'{P1RES}/{run}/{sub}/{det}/per_event.npz'
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    src = f'{EOS_ANGLES}/{run}/{sub}/{det}/per_event.npz'
    cp = ("from XRootD import client\n"
          "p = client.CopyProcess()\n"
          f"p.add_job('root://eospublic.cern.ch/{src}', '{dest}', force=True)\n"
          "p.prepare()\nst, _ = p.run()\nprint(st.message)\nraise SystemExit(0 if st.ok else 1)\n")
    subprocess.run([XRD_PY, '-c', cp], check=True)
    return dest


def s68(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return float(np.percentile(np.abs(d - np.median(d)), 68)) if len(d) else np.nan


def boot_s68(d, nb=300, seed=17):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(seed)
    return float(np.std([s68(rng.choice(d, len(d))) for _ in range(nb)]))


def scale(x, y):
    """Robust straight-line slope of y against x (|x| > 0.05, iterated 3 MAD trim) and its error (hv_metrics._scale)."""
    k = np.abs(x) > 0.05
    A = np.vstack([x[k], np.ones(k.sum())]).T
    yk = y[k]
    m = np.ones(k.sum(), bool)
    for _ in range(4):
        c, *_ = np.linalg.lstsq(A[m], yk[m], rcond=None)
        r = yk - A @ c
        m = np.abs(r) < 3 * 1.4826 * np.median(np.abs(r[m]))
    cov = np.linalg.inv(A[m].T @ A[m]) * np.var(r[m])
    return float(c[0]), float(np.sqrt(cov[0, 0]))


def paper_events(key, npz=None, variant='cv', exclude_railed=True):
    """The paper's selection of the row with the angle of ``variant`` ('p1', 'cv', 'prod'):
    {view: DataFrame(eid, ref, fit_npz, fit, used)}; tan_fit of the npz is the pass(-1) value the paper quoted.
    ``fit`` is the new angle where the event is in the map1 table and its pass-1 tangent equals the npz one (1e-6), else the npz
    value (0.1 % of events differ: the two-track re-pass of A re-fitted them)."""
    npz = npz or fetch_npz(key)
    z = np.load(npz)
    A = row_angles(key)
    out = {}
    for v in 'xy':
        eid = z[f'{v}_eid']
        d = pd.DataFrame(dict(eid=eid, ref=z[f'{v}_tan_ref'], fit_npz=z[f'{v}_tan_fit'], w=z[f'{v}_w']))
        j = A.reindex(eid)
        d['in_table'] = j[f'{v}_p1'].notna().values
        same = d.in_table.values & (np.abs(j[f'{v}_p1'].values - d.fit_npz.values) < 1e-6)
        d['same'] = same
        d['fit'] = np.where(same, j[f'{v}_{variant}'].values, d.fit_npz.values)
        d['used'] = np.where(same, j[f'{v}_used'].values, False)
        d['excl'] = np.where(same, j[f'{v}_excl'].values, False)
        d['railed'] = np.where(same, j[f'{v}_railed'].values, False)
        if exclude_railed:
            d = d[~d.excl]
        out[v] = d.reset_index(drop=True)
    return out


def restat(key, npz=None, variant='cv', exclude_railed=True, boot=True):
    """s68 [deg], its bootstrap error, n, median and robust angle scale per view, on the paper's selection."""
    P = paper_events(key, npz, variant, exclude_railed)
    res = {}
    for v, d in P.items():
        d = d[np.isfinite(d.fit) & np.isfinite(d.ref)]
        dd = np.degrees(np.arctan(d.fit.values)) - np.degrees(np.arctan(d.ref.values))
        sc = scale(d.ref.values, d.fit.values)
        res[v] = dict(n=int(len(d)), s68=s68(dd), s68_err=boot_s68(dd) if boot else np.nan, median=float(np.median(dd)),
                      scale=sc[0], scale_err=sc[1], frac_used=float(d.used.mean()), n_excl_railed=None)
    return res


if __name__ == '__main__':
    k = sys.argv[1]
    for var in ('p1', 'cv'):
        print(var, json.dumps(restat(k, variant=var, boot=False), indent=0))
