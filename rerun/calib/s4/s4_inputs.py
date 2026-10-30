#!/usr/bin/env python3
"""s4_inputs.py <data dir> [--out results/s4_inputs.json] -- S4 (PASS_PLAN §0): the physics inputs re-measured on pass 1.

Existing estimators only, applied to the five golden rows of Analysis_pass1:
  T        FA's model-free column duration (colsum.py, run with the PASS bundle: its template and dead masks; condor
           s4_row.sh), clean unsaturated triggers (vgas.clean), T = 20 % trimmed mean of (T_x + T_y) / 2 (vgas.main)
  v_T      gap / T, the timing v's own definition (gap / column duration), with the input gap of physics_inputs.json
  ladder   the pass's angle-implied v: Huber slope of w [um/ns] on the M3 tan_ref (vgas.ladder) on angles/per_event.npz
  gap_C19  ladder x T (the 'reco-ladder v x stacked drift time' route of C19 / gaps.py for det6)
For reference the same colsum estimator on the audit bundles (FA, 10-27) is read too, so a shift between the pre-pass and
the pass templates is separated from the estimator's own offset against the stack T the inputs came from.

<data dir> holds <golden row>/wft/{angles/per_event.npz, calib_bundle/bundle.json} (pulled from Analysis_pass1),
<golden row>/fa/colsum.npz (S4, pass bundle) and fa_audit/<golden row>/fa/colsum.npz (FA, audit bundle).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FA_DIR = os.path.abspath(os.path.join(HERE, '..', '..', 'model_audit', 'followups', 'fa_gas_v_tilt'))
sys.path.insert(0, FA_DIR)
from fa_common import GOLD_ROW, load_colsum, tmean  # noqa: E402
from vgas import clean, ladder  # noqa: E402

PHYS = json.load(open(os.path.join(HERE, '..', 'physics_inputs.json')))
# stated errors of the timing v (rerun/vladder/ladder_summary.py TIMING; det6 scaled x 28/30 with its v)
V_ERR = dict(det2=1.2, det3=0.6, det4=0.7, det6=0.8 * 28.0 / 30.0, det7=1.1)
GAP_ERR = dict(det6=1.0)            # det6's adopted 28.0 +- 1.0 (gaps.py); the others are in physics_inputs.json


def colsum_T(path):
    """(T [ns], its error, n) of the clean unsaturated triggers, as vgas.main does for det3."""
    df, _, _ = load_colsum(path)
    m = clean(df)
    T = 0.5 * (df.x_T.to_numpy() + df.y_T.to_numpy())[m]
    err = float(1.2533 * 1.4826 * np.median(np.abs(T - np.median(T))) / np.sqrt(len(T)))
    return tmean(T), err, int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 's4_inputs.json'))
    a = ap.parse_args()
    out = {}
    for det, row in GOLD_ROW.items():
        P = PHYS[det]
        gap, gap_err = P['gap_mm'], P['gap_err_mm'] if P['gap_err_mm'] is not None else GAP_ERR[det]
        v_in, v_err = P['v_timing_um_ns'], V_ERR[det]
        b = json.load(open(os.path.join(a.data, row, 'wft', 'calib_bundle', 'bundle.json')))
        r = dict(gap_in=gap, gap_err=gap_err, v_in=v_in, v_err=v_err, v_bundle=b['v_drift'],
                 Dp_bundle=b['hyper'].get('Dp'), sigma_p0_bundle=b['hyper'].get('sigma_p0'))
        z = np.load(os.path.join(a.data, row, 'wft', 'angles', 'per_event.npz'))
        for p in 'xy':
            bb, eb, n = ladder(z[f'{p}_w'] * 1e3, z[f'{p}_tan_ref'])
            r[f'ladder_{p}'], r[f'ladder_{p}_err'], r[f'ladder_{p}_n'] = float(bb[0]), float(eb[0]), n
        r['ladder'] = 0.5 * (r['ladder_x'] + r['ladder_y'])
        for tag, sub in (('pass', os.path.join(a.data, row, 'fa', 'colsum.npz')),
                         ('audit', os.path.join(a.data, 'fa_audit', row, 'fa', 'colsum.npz'))):
            if not os.path.exists(sub):
                continue
            T, Te, n = colsum_T(sub)
            r[f'T_{tag}'], r[f'T_{tag}_err'], r[f'T_{tag}_n'] = T, Te, n
            r[f'vT_{tag}'] = gap / T * 1e3
            r[f'vT_{tag}_err'] = r[f'vT_{tag}'] * np.hypot(gap_err / gap, Te / T)
        if 'T_pass' in r:
            T = r['T_pass']
            r['vT_pull'] = (r['vT_pass'] - v_in) / v_err
            r['gap_C19'] = r['ladder'] * T * 1e-3
            r['gap_C19_pull'] = (r['gap_C19'] - gap) / gap_err
            r['T_implied_by_input'] = gap / v_in * 1e3
        out[det] = r
        print(det, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print('->', a.out)


if __name__ == '__main__':
    main()
