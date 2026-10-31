#!/usr/bin/env python3
"""matched_rows.py <jobs.txt> --out results/rows_matched.csv [--nboot 30]
-- the timing widths of every bench row on its M3-matched single-track triggers (Paper I §7 standard sample).

Per row: timing_row.py's per-trigger times (<stem>.npz, clean triggers) joined on (acquisition tag, eventId) with the
row's pass-1 events table (Analysis_pass1/<run>/<subrun>/<det>/wft/events.parquet): both views fitted ok, one track.
The pass table holds the triggers with an M3 reference ray, so this keeps one cosmic muon through the chamber and drops
non-track triggers (showers, noise, faulty-channel clusters: det7's all-trigger width is 3.6x its matched width).
Widths as timing_row.py (s68): sigma_xy, sigma_x, sigma_y, sigma_evt, sigma_common; trig_cov = cov(start, end) of the
column (central 90 %); bootstrap errors.  Rows with fewer than 150 matched triggers are listed with n only.
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd

A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
COLS = ['key', 'det', 'run', 'subrun', 'resist_V', 'drift_V', 'cls', 'n', 'T_x', 'T_y', 'sigma_xy', 'sigma_xy_err', 'sigma_x',
        'sigma_y', 'sigma_evt', 'sigma_evt_err', 'sigma_common', 'sigma_common_err', 'trig_cov_ns2', 'trig_cov_err_ns2']


def s68(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 30:
        return np.nan
    q = np.percentile(a, [16, 84])
    return float((q[1] - q[0]) / 2)


def tcov(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    qa, qb = np.percentile(a, [5, 95]), np.percentile(b, [5, 95])
    c = (a > qa[0]) & (a < qa[1]) & (b > qb[0]) & (b < qb[1])
    return float(np.cov(a[c], b[c])[0, 1])


def widths(tx, ty, ex, ey):
    D, M, En = tx - ty, (tx + ty) / 2, (ex + ey) / 2
    sxy, sev = s68(D) / np.sqrt(2), s68(M)
    return dict(sigma_xy=sxy, sigma_x=s68(tx), sigma_y=s68(ty), sigma_evt=sev,
                sigma_common=float(np.sqrt(max(sev ** 2 - sxy ** 2 / 2, 0))), trig_cov_ns2=tcov(M, En))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('jobs'); ap.add_argument('--out', required=True)
    ap.add_argument('--nboot', type=int, default=30)
    a = ap.parse_args()
    rng = np.random.default_rng(2)
    out = []
    for line in open(a.jobs):
        colsum, bundle, stem, *meta = line.split()
        md = dict(kv.split('=', 1) for kv in meta)
        r = {k: md.get(k, '') for k in ('key', 'det', 'run', 'subrun', 'resist_V', 'drift_V', 'cls')}
        f = stem + '.npz'
        ev = os.path.join(A, md['run'], md['subrun'], md['det'], 'wft', 'events.parquet')
        if not (os.path.exists(f) and os.path.exists(ev)):
            r['n'] = 0; out.append(r); continue
        z = np.load(f); tags = list(z['acq_tags'])
        T = pd.DataFrame(dict(acq_tag=[tags[i] for i in z['acq']], event_id=z['eid'], tx=z['tx'], ty=z['ty'],
                              ex=z['ex'], ey=z['ey']))
        E = pd.read_parquet(ev, columns=['acq_tag', 'event_id', 'x_ok', 'y_ok', 'n_tracks'])
        m = T.merge(E, on=['acq_tag', 'event_id'])
        m = m[m.x_ok & m.y_ok & (m.n_tracks == 1)]
        r['n'] = int(len(m))
        if len(m) >= 150:
            arr = [m[c].to_numpy(float) for c in ('tx', 'ty', 'ex', 'ey')]
            w = widths(*arr)
            bs = [widths(*[x[ii] for x in arr]) for ii in (rng.integers(0, len(m), len(m)) for _ in range(a.nboot))]
            r.update(w)
            for k in ('sigma_xy', 'sigma_evt', 'sigma_common'):
                r[k + '_err'] = float(np.nanstd([b[k] for b in bs]))
            r['trig_cov_err_ns2'] = float(np.nanstd([b['trig_cov_ns2'] for b in bs]))
        js = stem + '.json'
        if os.path.exists(js):
            import json
            j = json.load(open(js))
            r['T_x'], r['T_y'] = j.get('T_median_x', ''), j.get('T_median_y', '')
        out.append(r)
        print(r['key'], r['n'], round(r.get('sigma_evt', np.nan), 1), flush=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, COLS); w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, '') for k in COLS})


if __name__ == '__main__':
    main()
