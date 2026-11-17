#!/usr/bin/env python3
"""make_vt_ab.py -- A/B test: does the frozen v_drift shape the pass beyond tan = w / (kw v)?  (2026-11-17)

v does not enter the waveform fit (gc_gas_time/REPORT.md: +-10 % v gives bit-identical w on det4), so v(t) is meant to
be a table-level correction.  Three places in the reco do use v: the raw |tan| < tan_max "plausible" test that ranks
candidates and gates two-track splits, the seeding (tan -> w), and the column-end u window (30 mm / v).  With the
frozen v 14-18 % below the measured v (C at its fastest), the raw-tan edge 0.6 sits at true tan ~0.51.

B = the frozen bundle with only `v_drift` (and its provenance) set to the sub-run's measured column-duration v
(vt_table.py, `vt_<arm>_subrun.parquet`).  A = the main pass output of the same acquisition.  The comparison
(`ab_compare.py`) checks: same events found, w / p0 / t0 identical, tan_B == tan_A * v_A / v_B (bow aside), and
which tracks change between the two near the acceptance edge.

    make_vt_ab.py --pick PICK.csv --frozen DIR --vt DIR --out DIR

PICK.csv columns: arm, run, subrun, tag, bundle (frozen bundle name).  Writes DIR/bundles/<arm>_<run>_<subrun>/ and
DIR/jobs.txt (ntof.sub format).  Upload the bundles to EOS_BASE/bundles/.
"""
import argparse, json, os, shutil

import pandas as pd

EOS_BASE = '/eos/experiment/ntof/data/x17/july_beam/Analysis_pass1/tests/vt_ab'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pick', required=True)
    ap.add_argument('--frozen', required=True, help='directory holding the frozen bundles (<arm>_E2_beam/ ...)')
    ap.add_argument('--vt', required=True, help='vt_table.py output dir')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    P = pd.read_csv(a.pick)
    S = {arm: pd.read_parquet(f'{a.vt}/vt_{arm}_subrun.parquet').set_index(['run', 'subrun']) for arm in P.arm.unique()}
    jobs, meta = [], {}
    for r in P.itertuples():
        name = f'{r.arm}_{r.run}_{r.subrun}'
        out = f'{a.out}/bundles/{name}'
        if name not in meta:
            src = f'{a.frozen}/{r.bundle}'
            v = round(float(S[r.arm].loc[(r.run, r.subrun), 'v']), 3)
            os.makedirs(out, exist_ok=True)
            shutil.copy2(f'{src}/arrays.npz', f'{out}/arrays.npz')
            b = json.load(open(f'{src}/bundle.json'))
            v0 = b['v_drift']
            b['v_drift'] = v
            b['provenance']['ntof_bundle']['v_drift'] = (f'{v} um/ns: vt_table.py sub-run column-duration v of run_{r.run} '
                                                          f'{r.subrun} (make_vt_ab.py test bundle; frozen {r.bundle} '
                                                          f'had {v0})')
            json.dump(b, open(f'{out}/bundle.json', 'w'), indent=1)
            meta[name] = dict(src=r.bundle, v_frozen=v0, v_test=v)
        jobs.append(f'{r.run} {r.subrun} {r.tag} {r.arm} {EOS_BASE}/bundles/{name} '
                    f'{EOS_BASE}/B/run_{r.run}/{r.subrun}/{r.arm}')
    open(f'{a.out}/jobs.txt', 'w').write('\n'.join(jobs) + '\n')
    json.dump(meta, open(f'{a.out}/ab.json', 'w'), indent=1)
    print(f'{len(jobs)} jobs, {len(meta)} bundles')
    for k, m in meta.items():
        print(k, m)


if __name__ == '__main__':
    main()
