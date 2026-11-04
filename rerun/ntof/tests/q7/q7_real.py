#!/usr/bin/env python3
"""q7_real.py -- reconstruct real n_TOF triggers once, in every Q7 configuration (q7lib.fit_multi).

    q7_real.py --run 145 --subrun stat090_0000 --arm A --tags 260805_14H06_000 ... [--limit 1200] [--events FILE]
               [--stem r145_A] [--jobs 5]

Seeding and windows are ntof_reco.py's (rr1 hits, min strips 3, dead padding); bundle v1k_<arm> (t0 free).  Output:
$Q7_DATA/real/<stem>.{events,candidates,splits}.parquet with a 'cfg' column and an 'acq_tag' column.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q7lib as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=int, required=True)
    ap.add_argument('--subrun', required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--tags', nargs='*', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--events', default=None)
    ap.add_argument('--stem', required=True)
    ap.add_argument('--jobs', type=int, default=5)
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--ac-seeded', action='store_true', help='only events seeded in both views of both A and C')
    a = ap.parse_args()
    import pandas as pd
    from wft.calib import CalibrationBundle
    bundle = a.bundle or os.path.join(L.BUNDLES, f'v1k_{a.arm}')
    cal = CalibrationBundle.load(bundle)
    cfg = L.beam_cfg(a.run, a.subrun, a.arm)
    tags = a.tags or L.subrun_tags(cfg)
    ev = {int(x) for x in open(a.events).read().split()} if a.events else None
    configs = L.all_configs()
    out_dir = os.path.join(L.DATA, 'real')
    allres = {L.cfg_name(*c): ([], [], []) for c in configs}
    t0 = time.time()
    n_tot = 0
    acs = L.ac_seeded_events(a.run, a.subrun, tags) if a.ac_seeded else None
    for tag in tags:
        e_tag = ev if acs is None else (acs[tag] if ev is None else acs[tag] & ev)
        payloads, n = L.tag_payloads(cfg, tag, cal, events=e_tag, limit=a.limit)
        t = time.time()
        res = L.run_payloads(bundle, payloads, configs, a.jobs)
        for k, (E, C, S) in res.items():
            for lst, src in zip(allres[k], (E, C, S)):
                lst.extend(dict(r, acq_tag=tag) for r in src)
        n_tot += len(payloads)
        print(f'[q7real] {a.stem} {tag}: {len(payloads)} events in {time.time() - t:.0f} s '
              f'({(time.time() - t) * a.jobs / max(len(payloads), 1):.2f} cpu-s/event, all {len(configs)} configs)',
              flush=True)
        L.write_tables(allres, out_dir, a.stem)
    meta = dict(ac_seeded=a.ac_seeded, run=a.run, subrun=a.subrun, arm=a.arm, tags=tags, limit=a.limit, events=a.events, bundle=bundle,
                configs=[L.cfg_name(*c) for c in configs], n_events=n_tot, minutes=(time.time() - t0) / 60,
                hits_dir=L.HITS_DIR, min_strips=L.MIN_STRIPS)
    json.dump(meta, open(os.path.join(out_dir, f'{a.stem}.meta.json'), 'w'), indent=1)
    print(f'[q7real] wrote {out_dir}/{a.stem}.*  ({n_tot} events, {meta["minutes"]:.1f} min)')


if __name__ == '__main__':
    main()
