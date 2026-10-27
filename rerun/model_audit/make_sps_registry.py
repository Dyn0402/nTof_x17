#!/usr/bin/env python3
"""make_sps_registry.py [--out rerun/sps_registry.csv] -- the SPS det4 rows of the pass (RERUN_PLAN §12 B3), from
sps_beam_test_26/analysis/datasets.py (the single source of truth for conditions).  One row per (dataset, sub-run,
plateau): run, subrun, files, mount, gas, drift/resist V, field over the 28.8 mm gap, n_samples, raw/ZS, the
span-route v(E) where measured (EXTRACTION_2026-08-05 §4; CF4 mixture only), the depth grid K that covers the column
at 64 samples, and the class (golden_quoted = the inclined operating point and the flat operating point; consistency =
ladders and scans; skeptical = the flush transient, truncated runs).
"""
import argparse, csv, os, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'sps_beam_test_26', 'analysis'))
import datasets  # noqa: E402

GAP_MM = 28.8
V_SPAN = {235: 14.0, 217: 13.2, 200: 12.4, 182: 11.8, 165: 11.6, 148: 11.6, 113: 11.5}      # CF4 mixture, span route
CLASS = {'run63_rot25': 'consistency', 'run63_flat': 'golden_quoted', 'run71_raw': 'golden_quoted', 'run61_op25': 'golden_quoted',
         'run60_flush': 'skeptical', 'run59_co2': 'consistency', 'run66_flat_resist': 'consistency', 'run56_m70V': 'consistency'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(REPO, 'rerun', 'sps_registry.csv'))
    a = ap.parse_args(); rows = []
    for name, D in sorted(datasets.DATASETS.items()):
        run = os.path.basename(D['stage'].rstrip('/')).replace('_m70V', '')
        for sub, stem, t0, idx in D['subruns']:
            for lab, lo, hi, dv, rv in D['plateaus']:
                field = dv / (GAP_MM / 10.0)
                vs = min(V_SPAN.items(), key=lambda kv: abs(kv[0] - field)) if 'CF4' in D['gas'] and 'TRANSITION' not in D['gas'] else None
                v = vs[1] if vs and abs(vs[0] - field) < 12 else ''
                k = int(np.ceil(GAP_MM * 1e3 / (v if v else 10.0) / 60.0)) + 5 if v else ''
                rows.append(dict(key=f'{name}__{sub}__{lab}', dataset=name, run=run, subrun=sub, stem=stem, files=' '.join(idx),
                                 plateau=lab, t_lo_s=round(lo, 1), t_hi_s=round(hi, 1), mount=D['mount'], gas=D['gas'], drift_V=dv,
                                 resist_V=rv, field_Vcm=round(field, 1), n_samples=D['n_samples'], raw=int(bool(D.get('raw'))),
                                 zs_sigma=D['zs_sigma'], v_span_um_ns=v, K=k, cls=CLASS.get(name, 'consistency'),
                                 bundle='bench det4 calib_bundle, dead list dropped (FEU 03 cabling), v = v_span, WFT_ZS_CENSOR unless raw'))
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'wrote {a.out}: {len(rows)} rows, {len({r["dataset"] for r in rows})} datasets')
    import collections; print(collections.Counter(r['cls'] for r in rows))


if __name__ == '__main__':
    main()
