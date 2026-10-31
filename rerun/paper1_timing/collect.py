#!/usr/bin/env python3
"""collect.py <rows dir> --out results/rows.csv -- one line per bench row from timing_row.py's JSONs."""
import argparse
import csv
import glob
import json
import os

COLS = ['key', 'det', 'run', 'subrun', 'resist_V', 'drift_V', 'cls', 'source', 'calib_key', 'n_clean', 'v_timing_um_ns',
        'sigma_xy', 'sigma_xy_err', 'sigma_x', 'sigma_y', 'sigma_evt', 'sigma_evt_err', 'sigma_common', 'sigma_common_err',
        'bias_xy', 'trig_cov_ns2', 'trig_cov_err_ns2', 'startstart_cov_ns2', 'sigma_evt_mm', 'sigma_xy_mm',
        'walk_span_ns', 'box_sigma_evt', 'fixT_sigma_evt', 'cf10_sigma_evt', 'edge15_sigma_evt', 'edge40_sigma_evt',
        'box_sigma_xy', 'cf10_sigma_xy', 'skip']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('rows'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = []
    for f in sorted(glob.glob(os.path.join(a.rows, '*.json'))):
        j = json.load(open(f))
        r = {k: j.get(k, '') for k in ('key', 'det', 'run', 'subrun', 'resist_V', 'drift_V', 'cls', 'source', 'calib_key',
                                        'n_clean', 'v_timing_um_ns', 'trig_cov_ns2', 'trig_cov_err_ns2', 'startstart_cov_ns2', 'skip')}
        if 'estimators' in j:
            e = j['estimators'][j['primary']]
            for k in ('sigma_xy', 'sigma_x', 'sigma_y', 'sigma_evt', 'sigma_common', 'bias_xy'):
                r[k] = e[k]
            for k in ('sigma_xy', 'sigma_evt', 'sigma_common'):
                r[k + '_err'] = e['err'][k]
            r['sigma_evt_mm'] = j.get('mm', {}).get('sigma_evt', '')
            r['sigma_xy_mm'] = j.get('mm', {}).get('sigma_xy', '')
            w = j['walk_median_evt_by_amp_quartile']
            r['walk_span_ns'] = max(w) - min(w)
            for k in ('box', 'fixT', 'cf10', 'edge15', 'edge40'):
                r[f'{k}_sigma_evt'] = j['estimators'][k]['sigma_evt']
            for k in ('box', 'cf10'):
                r[f'{k}_sigma_xy'] = j['estimators'][k]['sigma_xy']
        out.append(r)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, COLS); w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, '') for k in COLS})
    print(f'{len(out)} rows -> {a.out}')


if __name__ == '__main__':
    main()
