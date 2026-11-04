#!/usr/bin/env python3
"""census.py -- the n_TOF channel census, one acquisition at a time, read from EOS over XRootD (P2).

    # 1. the sample (first and last acquisition of every kept sub-run of the registry)
    census.py sample  [--per-subrun 2]
    # 2. probe each sampled acquisition (needs XRootD + uproot: a throwaway venv, e.g. /tmp/xrdvenv_p2/bin/python -I)
    census.py probe   [--workers 3] [--limit N]

Reads only eventId, feu, channel, amplitude, significance, saturated, sample of the acquisition's
``combined_hits_root`` file (field-identical to the rr1 recipe on run_145: NTOF_PASS_PLAN §6 Q4), straight from
root://eospublic.cern.ch (Kerberos; nothing is copied).  Hits are QA input here, never geometry.

Per acquisition and FEU (one view of one chamber) it writes to $WORK_DATA/ntof_pass1/channel_layer/acq/<key>.npz:
  occ, n1, f1, n2, f2       flank / silent accumulators (rules.flank_accumulate), summable over an era
  bseed, bsize              seed counts per 1000-event block (rules.block_seed_counts), for the hot probe
  on300                     events with the channel > 300 ADC
  S, N                      short accumulators (first sampled acquisition of each run, and every E1 acquisition)
and to <key>.json the per-acquisition decisions: silent, flank-dead, hot (seed probe), near oscillators, coupled
channels and the oscillators' ON fractions, per-connector occupancy.
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
import rules  # noqa: E402

WORK = os.path.join(os.environ.get('WORK_DATA', '/media/ucla/work'), 'ntof_pass1', 'channel_layer')
REGISTRY = os.path.join(REPO, 'rerun', 'ntof', 'ntof_registry.csv')
XRD = 'root://eospublic.cern.ch/'
COLS = ['eventId', 'feu', 'channel', 'amplitude', 'significance', 'saturated', 'sample']
SAMPLE_SANITY_MAX = 1000.0          # ntof_tracking/reco/io.py: diverged pulse-time fits
FEU_ARM = {3: ('A', 'x'), 4: ('A', 'y'), 5: ('B', 'x'), 6: ('B', 'y'), 7: ('C', 'x'), 8: ('C', 'y'),
           1: ('D', 'x'), 2: ('D', 'y')}
#: every n_TOF channel reads the strip at the bench strip-map position (cabling.py: 77/77 run_configs, all 4096
#: channels); this is that position
C = np.arange(512)
POS = np.round(49.92 * (C // 64) + (63 - C % 64) * 0.78, 2).astype(float)
BENCH_DET = {'A': 'det3', 'B': 'det2', 'C': 'det6', 'D': 'det7'}


def chamber_side_prior():
    """{arm: {view: bool[512]}}: the F12 chamber-side faults (f12_table.json class 'detector': flagged in >= 2 datasets
    read through different electronics) at their n_TOF channel (same position frame)."""
    T = json.load(open(os.path.join(REPO, 'rerun', 'model_audit', 'ws_dead_charge', 'f12_table.json')))
    out = {}
    for arm, det in BENCH_DET.items():
        out[arm] = {}
        for v in 'xy':
            m = np.zeros(512, bool)
            for r in T[f'{det}_{v}']['positions']:
                if r['cls'] == 'detector':
                    j = int(np.argmin(np.abs(POS - r['pos'])))
                    if abs(POS[j] - r['pos']) < 0.3:
                        m[j] = True
            out[arm][v] = m
    return out


# ------------------------------------------------------------------------------------------------------ sample
def make_sample(per_subrun=2):
    listing = json.load(open(os.path.join(WORK, 'hits_listing.json')))
    rows = [r for r in csv.DictReader(open(REGISTRY)) if r['excluded'] == 'False']
    by = {}
    for r in rows:
        by.setdefault((r['run'], r['subrun']), []).append(r)
    out, first_of_run = [], set()
    for (run, sub), rs in by.items():
        rs = sorted(rs, key=lambda r: r['tag'])
        pick = [rs[0]] if per_subrun == 1 or len(rs) == 1 else [rs[0], rs[-1]]
        files = listing[f'{run}/{sub}']['old']
        for r in pick:
            f = [x for x in files if f"_datrun_{r['tag']}_" in x]
            if not f:
                print('no hits file for', run, sub, r['tag']); continue
            # short accumulators: the first sampled acquisition of each run, and every acquisition of the short E1 era
            # (3 runs only; make_masks votes shorts per acquisition)
            corr = run not in first_of_run or r['access_epoch'] == 'pre_access_27jul'
            first_of_run.add(run)
            out.append(dict(key=f"run_{run}__{sub}__{r['tag']}", run=run, subrun=sub, tag=r['tag'], rclass=r['rclass'],
                            t_local=r['t_local'], access_epoch=r['access_epoch'],
                            path=r['eos_dir'] + '/combined_hits_root/' + f[0], corr=int(corr)))
    p = os.path.join(WORK, 'sample.csv')
    with open(p, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f'{len(out)} acquisitions from {len(by)} sub-runs -> {p}')


# ------------------------------------------------------------------------------------------------------ probe
def probe(row, prior):
    import uproot
    key = row['key']
    npz, js = os.path.join(WORK, 'acq', key + '.npz'), os.path.join(WORK, 'acq', key + '.json')
    if os.path.exists(js):
        return key, 'cached'
    t0 = time.time()
    with uproot.open(XRD + row['path']) as f:
        t = f['hits']
        a = t.arrays([c for c in COLS if c in t.keys()], library='np')
    n_h = len(a['eventId'])
    # early-July hits (pre-production) carry no significance / saturated (flank_ntof.py): every hit counts as significant
    a.setdefault('significance', np.full(n_h, np.inf, np.float32))
    a.setdefault('saturated', np.zeros(n_h, bool))
    a.setdefault('sample', np.zeros(n_h, np.float32))
    t_read = time.time() - t0
    s = a['sample'].astype(float)
    ok = np.isfinite(s) & (np.abs(s) <= SAMPLE_SANITY_MAX) & np.isfinite(a['amplitude'])
    a = {k: v[ok] for k, v in a.items()}
    ue, inv = np.unique(a['eventId'].astype(np.int64), return_inverse=True)
    n_ev = len(ue)
    rec = dict(key=key, run=row['run'], subrun=row['subrun'], tag=row['tag'], rclass=row['rclass'],
               t_local=row['t_local'], n_events=int(n_ev), eventid_span=int(ue[-1] - ue[0] + 1) if n_ev else 0,
               n_hits=int(len(inv)), dropped_unphysical=int((~ok).sum()), feu={})
    arrs = {}
    for feu, (arm, view) in FEU_ARM.items():
        m = a['feu'] == feu
        ch = a['channel'][m].astype(int); amp = a['amplitude'][m].astype(np.float32)
        sig = a['significance'][m]; sat = a['saturated'][m]; ii = inv[m]
        good = (sig > rules.FC.SIG) | (amp > rules.FC.BIG)
        A = np.zeros((n_ev, 512), np.float32)
        np.maximum.at(A, (ii[good], ch[good]), amp[good])
        nh = np.bincount(ii[good], minlength=n_ev)
        acc = rules.flank_accumulate(A, A, POS, nh)
        dec = rules.flank_decide(acc, POS)
        skip = prior[arm][view] | dec['silent'] | dec['dead']
        sd = rules.seeds(A, nh)
        bseed, bsize = rules.block_seed_counts(sd)
        hot = rules.hot_in_blocks(bseed, bsize, POS, skip)
        near = rules.near_oscillators(A, hot, POS, skip | hot)
        cpl, onf = rules.coupled_channels(A, hot | near, POS, skip)
        conn_occ = [round(float(dec['occ'][64 * k:64 * k + 64].mean()), 5) for k in range(8)]
        conn_sil = [int(dec['silent'][64 * k:64 * k + 64].sum()) for k in range(8)]
        L = lambda b: [int(c) for c in np.flatnonzero(b)]
        rec['feu'][str(feu)] = dict(arm=arm, view=view, silent=L(dec['silent']), flank_dead=L(dec['dead']),
                                    hot=L(hot), near_osc=L(near), coupled={str(k): v for k, v in cpl.items()},
                                    osc_on_frac={str(k): v for k, v in onf.items()},
                                    hot_seed_frac={str(c): round(float(bseed[:, c].max() / bsize.max()), 4) for c in L(hot)},
                                    conn_occ=conn_occ, conn_silent=conn_sil,
                                    median_occ=round(float(np.median(dec['occ'])), 5),
                                    n_flank_events=acc['n_flank_events'], busy_frac=round(float((nh > rules.SEED_BUSY).mean()), 4))
        for k in ('occ', 'n1', 'f1', 'n2', 'f2'):
            arrs[f'{k}_{feu}'] = acc[k].astype(np.int32)
        arrs[f'bseed_{feu}'] = bseed; arrs[f'bsize_{feu}'] = bsize
        arrs[f'on300_{feu}'] = (A > rules.COUPLE_ON_ADC).sum(0).astype(np.int32)
        if int(row['corr']):
            S, N = rules.short_accumulate(ii, ch, amp, sig, sat)
            arrs[f'S_{feu}'] = S.astype(np.float32); arrs[f'N_{feu}'] = N
    rec['t_read_s'] = round(t_read, 1); rec['t_total_s'] = round(time.time() - t0, 1)
    np.savez_compressed(npz, n_events=n_ev, **arrs)
    json.dump(rec, open(js, 'w'))
    return key, f'{n_ev} ev, read {t_read:.1f}s, total {rec["t_total_s"]}s'


def _probe(args):
    row, prior = args
    try:
        return probe(row, prior)
    except Exception:
        return row['key'], 'ERROR ' + traceback.format_exc().splitlines()[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['sample', 'probe'])
    ap.add_argument('--per-subrun', type=int, default=2)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--only', default=None, help='substring of the acquisition key')
    ap.add_argument('--sample', default=os.path.join(WORK, 'sample.csv'), help='sample csv (default: the registry sample)')
    a = ap.parse_args()
    os.makedirs(os.path.join(WORK, 'acq'), exist_ok=True)
    if a.cmd == 'sample':
        return make_sample(a.per_subrun)
    rows = list(csv.DictReader(open(a.sample)))
    if a.only:
        rows = [r for r in rows if a.only in r['key']]
    rows = rows[:a.limit] if a.limit else rows
    prior = chamber_side_prior()
    t0 = time.time()
    if a.workers <= 1:
        it = map(_probe, [(r, prior) for r in rows])
    else:
        from multiprocessing import Pool
        pool = Pool(a.workers)
        it = pool.imap_unordered(_probe, [(r, prior) for r in rows])
    for i, (k, msg) in enumerate(it):
        print(f'[{i + 1}/{len(rows)} {time.time() - t0:.0f}s] {k}: {msg}', flush=True)


if __name__ == '__main__':
    main()
