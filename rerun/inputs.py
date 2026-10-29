#!/usr/bin/env python3
"""rerun/inputs.py -- the re-run's input registry and EOS stager (RERUN_PLAN W10, 2026-10-23).

One place says which generation of each input the pass reads.  Analysis code is not edited: it keeps reading the
canonical directory names (``decoded_root``, ``combined_hits_root``, ``m3_tracking_root_v2``) under
``<bench>/<area>/<run>/<subrun>/``, and a condor job stages the chosen generation from EOS into that layout in its
scratch dir, then points ``qa_config`` at it with ``X17_BENCH_ROOT`` (see qa_config._rebase).

Generations
-----------
``rr1``  the re-run inputs: ``decoded_root_fa80c06`` (C11), ``combined_hits_root_rr1_cm0`` (C12/C13: plain-CM hits),
         ``m3_tracking_root_v2`` (C13: production rays stand).
``prod`` the 7-24/8-13 production inputs (old decoder), for comparisons only.

What a staging leaves out (written into ``INPUTS.json`` beside the data, with sizes):
  * acquisitions absent from the M3 directory (no reference, and false starts restart their event ids);
  * EXCLUDED acquisitions (the run registry's known false starts);
and it writes ``incomplete_events.json``: per acquisition and FEU, the eventIds the decoder audit found incomplete
(FEU packet loss).  Efficiency counts those as not measured, never as misses.

    python rerun/inputs.py stage <manifest key> <bench_root> [--gen rr1] [--parts run_config,m3,decoded,hits]
    python rerun/inputs.py show <manifest key>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MANIFEST = os.path.join(REPO, 'mx_june_wft', 'condor', 'campaign_manifest.csv')
AUDIT_DIR = '/eos/user/d/dneff/x17rerun/decode_audit'
#: The re-run's own Analysis tree (outputs of every stage, synced in and out by stage_job.py).
EOS_ANALYSIS_ROOT = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'   # the from-scratch pass (10-29); rr1 products stay in Analysis_rr1

GENERATIONS = {
    'rr1': {'decoded_root': 'decoded_root_fa80c06',
            'combined_hits_root': 'combined_hits_root_rr1_cm0',
            'm3_tracking_root_v2': 'm3_tracking_root_v2'},
    'prod': {'decoded_root': 'decoded_root',
             'combined_hits_root': 'combined_hits_root',
             'm3_tracking_root_v2': 'm3_tracking_root_v2'},
}

#: Acquisitions that are not part of their run (W8 registry).  (run, subrun) -> date_time stems.
EXCLUDED = {
    # 1277-event false start whose eventIds collide with 01H34's (moved aside locally 7-06; still on EOS)
    ('mx17_det3_p2_det1_overnight_6-27-26', 'long_run_p2_det1_sanity_check'): {'260628_01H29'},
}

STEM = re.compile(r'_datrun_(\d{6}_\d\dH\d\d)_(\d{3})')


def rows():
    with open(MANIFEST) as f:
        return list(csv.DictReader(f))


def row_for(key):
    """Manifest row for a manifest key, or for a run registered in qa_config.RUNS (e.g. g_det3_wknd, whose manifest
    key is synthesized), matched on run, subrun and detector.  The returned row carries that key."""
    hit = [r for r in rows() if r['key'] == key]
    if not hit:
        sys.path[:0] = [os.path.join(REPO, 'mx_june_cosmic_qa')]
        import qa_config
        if key in qa_config.RUNS:
            c = qa_config.RUNS[key]
            hit = [dict(r, key=key) for r in rows()
                   if (r['run'], r['subrun'], r['det']) == (c.RUN, c.SUB_RUN, c.DET_NAME)]
    if len(hit) != 1:
        raise KeyError(f'{key!r}: {len(hit)} manifest rows')
    return hit[0]


def acq(name):
    """'..._datrun_260628_01H34_003_07.root' -> ('260628_01H34', '003')."""
    m = STEM.search(name)
    return (m.group(1), m.group(2)) if m else None


def _copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for k in range(4):
        try:
            shutil.copyfile(src, dst + '.part')
            os.replace(dst + '.part', dst)
            if os.path.getsize(dst) == os.path.getsize(src):
                return os.path.getsize(dst)
        except OSError:
            pass
        import time
        time.sleep(3 + 5 * k)
    raise OSError(f'copy failed: {src}')


def plan(row, gen='rr1', feus=None):
    """What a staging of this row would copy: {canonical_dir: [(src, name)]}, plus the left-out acquisitions."""
    g = GENERATIONS[gen]
    sub = os.path.join(row['eos_run_dir'], row['subrun'])
    m3d = os.path.join(sub, g['m3_tracking_root_v2'])
    excl = EXCLUDED.get((row['run'], row['subrun']), set())
    m3 = sorted(n for n in os.listdir(m3d) if n.endswith('_rays.root') and acq(n) and acq(n)[0] not in excl)
    covered = {acq(n) for n in m3}
    feus = feus or [int(row['feu_x']), int(row['feu_y'])]
    pats = tuple(f'_{f:02d}.root' for f in feus)
    out = {'m3_tracking_root_v2': [(os.path.join(m3d, n), n) for n in m3]}
    left = {}
    for canon in ('decoded_root', 'combined_hits_root'):
        d = os.path.join(sub, g[canon])
        names = sorted(n for n in os.listdir(d) if n.endswith('.root') and acq(n))
        if canon == 'decoded_root':
            names = [n for n in names if n.endswith(pats) and '_datrun_' in n]
        keep = [n for n in names if acq(n) in covered]
        left[canon] = sorted({'_'.join(acq(n)) for n in names if acq(n) not in covered})
        out[canon] = [(os.path.join(d, n), n) for n in keep]
    return out, left, sorted(excl), feus


def incomplete_events(row, feus):
    """{'<stem>_<fnum>': {feu: [eventIds incomplete]}} from the decoder audit (C11)."""
    import numpy as np
    d = os.path.join(AUDIT_DIR, row['run'], row['subrun'])
    out = {}
    if not os.path.isdir(d):
        return out
    for n in sorted(os.listdir(d)):
        if not n.endswith('.npz') or '_datrun_' not in n:
            continue
        a = acq(n)
        feu = int(n[-6:-4])
        if a is None or feu not in feus:
            continue
        z = np.load(os.path.join(d, n))
        ns, nw = z['nsamp'], z['nwords']
        mns, mnw = (np.bincount(ns).argmax(), np.bincount(nw).argmax()) if len(ns) else (0, 0)
        comp = (ns == mns) & (z['smax'] == mns - 1) & (nw == mnw)
        bad = z['eventId'][~comp].astype(int).tolist()
        if bad:
            out.setdefault('_'.join(a), {})[feu] = bad
    return out


def stage(row, bench_root, gen='rr1', parts=('run_config', 'm3', 'decoded', 'hits'), feus=None):
    """Copy this row's inputs from EOS into <bench_root>/<area>/<run>/<subrun>/ under the canonical names."""
    files, left, excl, feus = plan(row, gen, feus)
    run_dir = os.path.join(bench_root, row['tree'], row['run'])
    sub_dir = os.path.join(run_dir, row['subrun'])
    want = {'m3': 'm3_tracking_root_v2', 'decoded': 'decoded_root', 'hits': 'combined_hits_root'}
    rec = dict(generation=gen, source_dirs={k: v for k, v in GENERATIONS[gen].items()},
               eos_subrun=os.path.join(row['eos_run_dir'], row['subrun']), feus=feus,
               excluded_acquisitions=excl, acquisitions_not_in_m3=left, files={})
    if 'run_config' in parts:
        _copy(os.path.join(row['eos_run_dir'], 'run_config.json'), os.path.join(run_dir, 'run_config.json'))
    for p in parts:
        if p not in want:
            continue
        canon = want[p]
        rec['files'][canon] = {}
        for src, name in files[canon]:
            rec['files'][canon][name] = _copy(src, os.path.join(sub_dir, canon, name))
    inc = incomplete_events(row, feus)
    os.makedirs(sub_dir, exist_ok=True)
    json.dump(inc, open(os.path.join(sub_dir, 'incomplete_events.json'), 'w'))
    rec['incomplete_events'] = {k: {f: len(v) for f, v in d.items()} for k, d in inc.items()}
    json.dump(rec, open(os.path.join(sub_dir, 'INPUTS.json'), 'w'), indent=1)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['stage', 'show'])
    ap.add_argument('key')
    ap.add_argument('bench_root', nargs='?')
    ap.add_argument('--gen', default='rr1')
    ap.add_argument('--parts', default='run_config,m3,decoded,hits')
    a = ap.parse_args(argv)
    row = row_for(a.key)
    if a.cmd == 'show':
        files, left, excl, feus = plan(row, a.gen)
        print(json.dumps(dict(row=row, n_files={k: len(v) for k, v in files.items()},
                              acquisitions_not_in_m3=left, excluded=excl, feus=feus), indent=1))
        return 0
    rec = stage(row, a.bench_root, a.gen, tuple(a.parts.split(',')))
    print(json.dumps({k: (len(v) if isinstance(v, dict) else v) for k, v in rec['files'].items()}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
