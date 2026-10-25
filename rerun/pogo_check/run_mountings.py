#!/usr/bin/env python3
"""run_mountings.py [--only RUN] -- the pogo check on every bench mounting (one chamber on one night), not only the
golden rows (Dylan 10-25: make sure the det7 pogo check has been done on the other detectors).

A mounting = (run, det) in rerun/pass_registry.csv (excluded rows dropped); all its sub-runs are pooled. FEUs and
tree come from mx_june_wft/condor/campaign_manifest.csv, so c26__ keys need no qa_config entry. Output names are
m_<run>__<det>_<plane>.npz in out/, then pogo_check.py writes results/.  Reads local hits_root (QA only); four nights
are on the pre-fa80c06 decode there (0.1-0.4 % fused events, irrelevant for pair statistics).
"""
import argparse, csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]
import numpy as np
import run_bench as RB

BENCH = '/home/dylan/x17/cosmic_bench'


def mountings():
    reg = [r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'pass_registry.csv'))) if r['cls'] != 'excluded']
    man = {(r['run'], r['subrun'], r['det']): r
           for r in csv.DictReader(open(os.path.join(REPO, 'mx_june_wft', 'condor', 'campaign_manifest.csv')))}
    m = {}
    for r in reg:
        row = man.get((r['run'], r['subrun'], r['det']))
        if row is None:
            print('not in manifest:', r['key']); continue
        m.setdefault((r['run'], r['det']), []).append(row)
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--only', nargs='*'); ap.add_argument('--out-dir', default=os.path.join(HERE, 'out'))
    a = ap.parse_args()
    from qa_config import _Config, setup_paths
    setup_paths()
    from wft.io import strip_position_map
    for (run, det), rows in sorted(mountings().items()):
        if a.only and run not in a.only:
            continue
        r0 = rows[0]
        feus = [int(r0['feu_x']), int(r0['feu_y'])]
        if any([int(r['feu_x']), int(r['feu_y'])] != feus for r in rows):
            print('FEU change inside', run, det); continue
        cfg = _Config(r0['key'], run, r0['subrun'], feus=feus, det_z=float(r0['det_z']), det_name=det,
                      base_path=os.path.join(BENCH, r0['tree']) + '/')
        pm = strip_position_map(cfg)
        subs = sorted({r['subrun'] for r in rows})
        tag = f'm_{run}__{det}'
        for p, feu in (('x', feus[0]), ('y', feus[1])):
            out = os.path.join(a.out_dir, f'{tag}_{p}.npz')
            if os.path.exists(out):
                print('exists', out); continue
            try:
                ps, rms = RB.plane(cfg, subs, feu, np.asarray(pm[feu], float))
            except SystemExit as e:
                print('SKIP', tag, p, e); continue
            ps.save(out, rms=rms, pos=np.asarray(pm[feu], float), run_key=tag, detector=det, plane=p, feu=feu,
                    subs=np.array(subs))
            print(tag, p, 'FEU', feu, 'subs', len(subs), 'events', ps.nev, '->', out, flush=True)


if __name__ == '__main__':
    main()
