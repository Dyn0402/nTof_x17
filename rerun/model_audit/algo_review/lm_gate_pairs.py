#!/usr/bin/env python3
"""lm_gate_pairs.py --root <gate_tree> [--a fe_nm] [--b fe_lm] -- per-event paired comparison of two gate arms' tables
(<root>/<row>/wft/<arm>/events.parquet, pulled by fe_unify2/pull_fe.sh): per chamber and view, on events fitted in both,
the fraction with the same w (1 um/ns), |dtheta| quantiles, t0 moved > 30 ns, |dp0| > 0.3 mm, the chi2 change, the
quality-flag and ok agreement, and the error ratios.  The gate table proper (efficiency, far, s68 vs M3) is
gate_compare.py --arms <b> --base <a>.  Writes results/lm_gate_pairs.json and prints a markdown table."""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gate_compare import ROWS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--a', default='fe_nm')
    ap.add_argument('--b', default='fe_lm')
    a = ap.parse_args()
    out = {}
    print(f'| chamber·view | events in both | ok A / B / both | same w | dθ p50 / p95 / >1° | t0 moved > 30 ns | '
          f'p0 moved > 0.3 mm | χ² B/A p50 | w_err B/A | tan_err B/A |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for det, row in ROWS.items():
        pa = os.path.join(a.root, row, 'wft', a.a, 'events.parquet')
        pb = os.path.join(a.root, row, 'wft', a.b, 'events.parquet')
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        A, B = pd.read_parquet(pa), pd.read_parquet(pb)
        m = A.merge(B, on='event_id', suffixes=('_a', '_b'))
        for v in 'xy':
            oka, okb = m[f'{v}_ok_a'].astype(bool), m[f'{v}_ok_b'].astype(bool)
            s = m[oka & okb]
            g = lambda k, arm: s[f'{v}_{k}_{arm}'].to_numpy(float)
            dth = np.abs(g('theta_deg', 'b') - g('theta_deg', 'a'))
            r = dict(n=len(m), ok_a=float(oka.mean()), ok_b=float(okb.mean()), both=int(len(s)),
                     same_w=float(np.mean(np.abs(g('w', 'b') - g('w', 'a')) * 1e3 < 1.0)),
                     dth_p50=float(np.nanmedian(dth)), dth_p95=float(np.nanpercentile(dth, 95)),
                     dth_gt1=float(np.nanmean(dth > 1.0)),
                     t0_moved=float(np.mean(np.abs(g('t0', 'b') - g('t0', 'a')) > 30)),
                     p0_moved=float(np.mean(np.abs(g('p0', 'b') - g('p0', 'a')) > 0.3)),
                     chi2_ratio=float(np.nanmedian(g('chi2', 'b') / g('chi2', 'a'))),
                     w_err_ratio=float(np.nanmedian(g('w_err', 'b') / g('w_err', 'a'))),
                     tan_err_ratio=float(np.nanmedian(g('tan_err', 'b') / g('tan_err', 'a'))),
                     quality_agree=float(np.mean(s[f'{v}_quality_ok_a'].to_numpy() == s[f'{v}_quality_ok_b'].to_numpy())))
            out[f'{det}_{v}'] = r
            print(f"| {det} {v.upper()} | {r['n']} | {r['ok_a']:.1%} / {r['ok_b']:.1%} / {r['both']} | {r['same_w']:.1%} | "
                  f"{r['dth_p50']:.3f} / {r['dth_p95']:.3f} / {r['dth_gt1']:.1%} | {r['t0_moved']:.1%} | {r['p0_moved']:.1%} | "
                  f"{r['chi2_ratio']:.4f} | {r['w_err_ratio']:.2f} | {r['tan_err_ratio']:.3f} |")
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'lm_gate_pairs.json')
    json.dump(out, open(fn, 'w'), indent=1)


if __name__ == '__main__':
    main()
