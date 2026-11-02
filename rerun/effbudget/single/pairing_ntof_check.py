#!/usr/bin/env python3
"""pairing_ntof_check.py TABLE.candidates.parquet BUNDLE_DIR [...] -- what the 'plausible' gate and the 'cost' pairing
mode do on n_TOF candidate tables (PAIRING_FIX_2026-11-02.md §4).

Per table: how often candidates and winners are implausible and why (raw |tan| >= tan_max, column end below / above
the u window), how many events the gate leaves with one pair while a second coincident x-y pair exists among the
other candidates, and a replay of select_tracks in 'gate' (must reproduce the file) and 'cost' mode: events whose
pairs change, n_tracks transitions.  No truth on beam: this measures the size of a change, not whether it is right.
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from wft import reco  # noqa: E402
from wft.calib import CalibrationBundle  # noqa: E402
from rerun.effbudget.single.pairing_replay import fits_of, ftst_diff_of, tracks  # noqa: E402


def check(path, bdir):
    cal = CalibrationBundle.load(bdir)
    reco.wm.use_calibration(cal)
    pairing = cal.xy_pairing
    reco.TC_TOL_NS = reco.tc_tol(pairing)
    C = pd.read_parquet(path)
    C = C[C['rank'] >= 0]
    key = ['acq_tag', 'event_id'] if 'acq_tag' in C else ['event_id']
    lo, hi = reco.u_window(cal)
    tm = reco.tan_max(cal)
    imp = C[~C.plausible.astype(bool)]
    r0 = C[C['rank'] == 0]
    out = dict(table=path, bundle=bdir, tan_max=tm, u_window=[lo, hi], n_cand=int(len(C)),
               implausible=float(1 - C.plausible.mean()), winner_implausible=float(1 - r0.plausible.mean()),
               implausible_why=dict(tan=float((imp.tan_theta.abs() >= tm).mean()),
                                    uend_lo=float((imp.q_uend < lo).mean()), uend_hi=float((imp.q_uend > hi).mean())))
    n = dict(events=0, reproduced=0, changed=0, ntr={}, gate_one_pair_with_second_coincident=0)
    for _k, g in C.groupby(key, sort=False):
        fits = fits_of(g)
        fd = ftst_diff_of(g)
        prod = {(r.plane, int(r.rank)): int(r.track_id) for r in g.itertuples() if r.track_id >= 0}
        old = reco.select_tracks(fits, fd, cal, max_tracks=3, pairing=pairing, mode='gate')
        new = reco.select_tracks(fits, fd, cal, max_tracks=3, pairing=pairing, mode='cost')
        n['events'] += 1
        n['reproduced'] += int(tracks(fits, old) == prod)
        if len(old) == 1:
            ux = [i for i in range(len(fits['x'])) if i != old[0][0]]
            uy = [j for j in range(len(fits['y'])) if j != old[0][1]]
            n['gate_one_pair_with_second_coincident'] += int(any(
                reco._coincident(fits['x'][i], fits['y'][j], fd) for i in ux for j in uy))
        if [p[:2] for p in old] != [p[:2] for p in new] or [p[2] for p in old] != [p[2] for p in new]:
            n['changed'] += 1
        t = f'{sum(g_ for *_a, g_ in old)}->{sum(g_ for *_a, g_ in new)}'
        n['ntr'][t] = n['ntr'].get(t, 0) + 1
    n['ntr'] = dict(sorted(n['ntr'].items()))
    out.update(n)
    return out


def main():
    a = sys.argv[1:]
    res = [check(a[i], a[i + 1]) for i in range(0, len(a), 2)]
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
