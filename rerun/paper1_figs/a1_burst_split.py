#!/usr/bin/env python3
"""a1_burst_split.py --a1 A1_ROOT --sparks OPS_SPARK_RAYS [--out results/a1_burst_split.json] -- the A1 spark classes of
the golden rows split into true discharges and non-discharge bursts (decision H, PAPER1_NUMBERS), with the §4 session's
definition (rerun/paper1_ops/spark_timeline.py): a vetoed trigger (> 50 strips over the floor) is a discharge when a chip
has >= 8 strips over 3000 ADC (band_census.py's test), else a burst.  Joins a1_recat.py's per-ray classes
(<A1_ROOT>/<rel>/recat.parquet) to spark_timeline's per-ray flags (<OPS>/<key>.npz: in_discharge, in_burst) on
event_id, over interior_good (region interior, good_x).  Labelling only: the pass's veto is unchanged."""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = {'det2': ('o22_long_det2', 'mx17_det2_det3_overnight_6-22-26/longer_run/mx17_2'),
        'det3': ('sat_det3', 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3'),
        'det4': ('g_det4', 'mx17_det4_day_6-24-26/long_run/mx17_4'),
        'det6': ('g_det6_long', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_6'),
        'det7': ('g_det7_long', 'mx17_det6_det7_overnight_6-26-26/long_run/mx17_7')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a1', required=True); ap.add_argument('--sparks', required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'a1_burst_split.json'))
    a = ap.parse_args()
    out = {}
    for det, (key, rel) in ROWS.items():
        C = pd.read_parquet(os.path.join(a.a1, rel, 'recat.parquet'))
        C = C.reset_index() if 'event_id' not in C.columns else C
        C = C[C.good_x.astype(bool)]
        Z = np.load(os.path.join(a.sparks, key + '.npz'))
        F = pd.DataFrame(dict(event_id=Z['event_id'], dis=Z['in_discharge'], burst=Z['in_burst']))
        M = C.merge(F, on='event_id', how='left')
        n = len(M)
        sp = M.recat.str.startswith('spark:')
        r = dict(key=key, n=n, unmatched_spark=int((sp & M.dis.isna()).sum()),
                 spark_pct=100 * sp.mean(),
                 discharge_pct=100 * (sp & (M.dis == True)).mean(),          # noqa: E712
                 burst_pct=100 * (sp & (M.burst == True)).mean())            # noqa: E712
        for c in ('spark:discharge', 'spark:muon+other', 'spark:shower'):
            s = M.recat == c
            r[c] = dict(pct=100 * s.mean(), as_discharge_pct=100 * (s & (M.dis == True)).mean(),   # noqa: E712
                        as_burst_pct=100 * (s & (M.burst == True)).mean())                         # noqa: E712
        out[det] = r
        print(f"{det} {key}: n {n}  spark {r['spark_pct']:.2f} % = discharge {r['discharge_pct']:.2f} + burst "
              f"{r['burst_pct']:.2f}  (unmatched {r['unmatched_spark']})  | " +
              '  '.join(f"{c.split(':')[1]} {r[c]['pct']:.2f}={r[c]['as_discharge_pct']:.2f}d+{r[c]['as_burst_pct']:.2f}b"
                        for c in ('spark:discharge', 'spark:muon+other', 'spark:shower')))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
