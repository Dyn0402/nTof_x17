#!/usr/bin/env python3
"""
collect_results.py — unpack the campaign's reco_r*.tar.gz into a staging tree,
NEVER over the live Analysis outputs.

    rsync -av lxplus:~/wft_campaign/reco_r*.tar.gz \
        /home/dylan/x17/cosmic_bench/condor_campaign/back/
    ../../.venv/bin/python mx_june_wft/condor/collect_results.py
        [--back .../condor_campaign/back] [--results .../condor_campaign/results]
        [--promote]

Staging layout: <results>/<key>[__tag]/events(.candidates).parquet + meta +
job_row.json. `--promote` copies each PROD-tagged result whose row passed its
gate into the Analysis tree at <OUT_BASE>/wft/, first moving any existing
events.parquet (+sidecars) to wft/pre_campaign_backup/. Promotion of gate arms
(t0p/offcond tags) is intentionally not supported — they are inputs to a
decision, not production outputs; offcond recos live in the staging tree and
are consumed from there with their off_conditions flag.
"""
import argparse
import glob
import json
import os
import shutil
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = '/home/dylan/x17/cosmic_bench'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--back', default=f'{BENCH}/condor_campaign/back')
    ap.add_argument('--results', default=f'{BENCH}/condor_campaign/results')
    ap.add_argument('--promote', action='store_true')
    args = ap.parse_args()

    tars = sorted(glob.glob(os.path.join(args.back, 'reco_r*.tar.gz')))
    print(f'{len(tars)} result tarballs')
    os.makedirs(args.results, exist_ok=True)
    for t in tars:
        with tarfile.open(t) as tf:
            tf.extractall(args.back + '/_tmp')
        for d in glob.glob(os.path.join(args.back, '_tmp', 'out', '*')):
            dest = os.path.join(args.results, os.path.basename(d))
            shutil.rmtree(dest, ignore_errors=True)
            shutil.move(d, dest)
        shutil.rmtree(args.back + '/_tmp', ignore_errors=True)

    done = sorted(os.listdir(args.results))
    print(f'{len(done)} results staged in {args.results}')

    if not args.promote:
        return
    for name in done:
        d = os.path.join(args.results, name)
        rowf = os.path.join(d, 'job_row.json')
        if not os.path.exists(rowf):
            continue
        row = json.load(open(rowf))
        if row.get('off_conditions'):
            continue
        # The promotable artifact is the dir named EXACTLY after the row's key;
        # arms carry a suffix (<key>__t0p, <key>__prod_noprior). Testing for
        # '__' anywhere in the name instead silently skipped all 17 tier-A rows
        # whose synthesized keys are c26__<run>__<subrun>__<det>, and testing
        # out_tag alone would promote the parked no-prior arms (their out_tag is
        # empty — they WERE the prod arm before the gate adopted the prior).
        if name != row['key']:
            continue
        out_base = os.path.join(BENCH, 'Analysis', row['run'], row['subrun'],
                                row['det'], 'wft')
        os.makedirs(out_base, exist_ok=True)
        bak = os.path.join(out_base, 'pre_campaign_backup')
        for f in ('events.parquet', 'events.candidates.parquet',
                  'events.meta.json'):
            live = os.path.join(out_base, f)
            if os.path.exists(live):
                os.makedirs(bak, exist_ok=True)
                # NEVER overwrite an existing backup: the first one is the true
                # pre-campaign state. Without this guard a second --promote
                # parks the ALREADY-PROMOTED campaign file on top of it and the
                # original is gone (this destroyed the 7-31 parquets of the five
                # golden keys on 2026-08-12 — they are not reproducible, the
                # code that made them is not in this repo).
                if os.path.exists(os.path.join(bak, f)):
                    os.remove(live)
                else:
                    shutil.move(live, os.path.join(bak, f))
            src = os.path.join(d, f)
            if os.path.exists(src):
                shutil.copy2(src, live)
        print('promoted', name, '->', out_base)


if __name__ == '__main__':
    main()
