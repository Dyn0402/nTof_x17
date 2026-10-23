#!/usr/bin/env python3
"""summarize_audit.py <fdfs.txt> <out.json> -- roll up the W9 re-decode audit (job.sh / audit_decode.py).

Per file: did the C++ decode agree with the independent decoder, how many events are complete, how many eventIDs
are missing from the file's span.  Per acquisition (one file per FEU sharing a date_time_idx tag):

  * cross-FEU: does every FEU, the M3 FEU (01) included, hold the same eventIDs, or do some triggers exist on one
    FEU and not on another (an FEU-specific loss reads as detector inefficiency downstream);
  * M3 v2 rays against FEU 01: rays whose eventID FEU 01 does not hold, or holds incomplete;
  * rays whose eventID a detector FEU does not hold, or holds incomplete -- these must leave the efficiency
    denominator ("not measured"), not count as misses;
  * old EOS decoded_root and combined_hits: entries whose eventID the new decode has incomplete (fused or partial).
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import uproot

AUD = '/eos/user/d/dneff/x17rerun/decode_audit'
def retry(fn, *a, n=5):
    """EOS fuse throws transient EIO under load: retry, then give up with None."""
    import time
    for k in range(n):
        try:
            return fn(*a)
        except OSError:
            time.sleep(2 + 3 * k)
    return None


TAG = re.compile(r'_(\d{6}_\d\dH\d\d_\d{3})_(\d\d)\.fdf$')


def main():
    fdfs = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    out = dict(files={}, acq={}, missing_audit=[], not_identical=[])
    groups = defaultdict(dict)
    for f in fdfs:
        sub = os.path.dirname(os.path.dirname(f))
        run, s = os.path.basename(os.path.dirname(sub)), os.path.basename(sub)
        b = os.path.basename(f)[:-4]
        j = os.path.join(AUD, run, s, b + '.json')
        if not os.path.exists(j):
            out['missing_audit'].append(f)
            continue
        r = retry(lambda p: json.load(open(p)), j)
        if r is None:
            out['missing_audit'].append(f + ' (unreadable)')
            continue
        ds = r.get('decode_stats') or {}
        rec = dict(identical=r['identical'], n=r['n_indep'], complete=r['complete_events'],
                   missing_in_span=r['eventid_missing_in_span'], closed_eventid=ds.get('closed_eventid'),
                   samples=r['samples_per_event'], indep=dict(orphan_nonzero=r['indep']['orphan_runs_nonzero'],
                   stray=r['indep']['stray_non_data_words'], hdr=r['indep']['feu_header_len'],
                   dlen=r['indep']['data_run_len'], vep=r['indep']['vep_word_class']),
                   frames_rep=r['indep']['frames_repeating_sample_in_event'], not8x64=r['indep']['frames_not_8x64'])
        out['files'][f] = rec
        if not r['identical']:
            out['not_identical'].append(f)
        m = TAG.search(os.path.basename(f))
        if m and '_datrun_' in os.path.basename(f):
            groups[(run, s, sub, m.group(1))][m.group(2)] = os.path.join(AUD, run, s, b + '.npz')
    for (run, s, sub, tag), feus in sorted(groups.items()):
        E = {}
        for feu, p in feus.items():
            if os.path.exists(p):
                z = retry(lambda q: dict(np.load(q)), p)
                if z is not None:
                    # complete = the file's modal sample count and modal word count (a FEU with fewer than 8 DREAM
                    # chips populated, e.g. the P2 board on FEU 06 of 6-27, is complete at 4 x 64 per sample)
                    ns, nw = z['nsamp'], z['nwords']
                    mns = np.bincount(ns).argmax() if len(ns) else 0
                    mnw = np.bincount(nw).argmax() if len(nw) else 0
                    comp = (ns == mns) & (z['smax'] == mns - 1) & (nw == mnw)
                    E[feu] = (z['eventId'].astype(np.int64), comp)
        if not E:
            continue
        allids = np.unique(np.concatenate([e for e, _ in E.values()]))
        a = dict(run=run, sub=s, tag=tag, n_union=int(len(allids)), feu={})
        for feu, (e, c) in E.items():
            a['feu'][feu] = dict(n=int(len(e)), complete=int(c.sum()), missing_vs_union=int(len(allids) - len(e)))
        # M3 rays
        rays = glob.glob(os.path.join(sub, 'm3_tracking_root_v2', f'*_{tag}_rays.root'))
        if rays:
            try:
                ev = uproot.open(rays[0])['T'].arrays(['evn'], library='np')['evn'].astype(np.int64)
                a['rays'] = int(len(ev))
                for feu, (e, c) in E.items():
                    have = np.isin(ev, e)
                    comp = np.isin(ev, e[c])
                    a['feu'][feu]['rays_absent'] = int((~have).sum())
                    a['feu'][feu]['rays_incomplete'] = int((have & ~comp).sum())
            except Exception as x:
                a['rays_error'] = str(x)
        # old decode and combined hits
        for feu, (e, c) in E.items():
            old = glob.glob(os.path.join(sub, 'decoded_root', f'*_{tag}_{feu}.root'))
            if old:
                try:
                    oe = uproot.open(old[0])['nt'].arrays(['eventId'], library='np')['eventId'].astype(np.int64)
                    a['feu'][feu]['old_n'] = int(len(oe))
                    a['feu'][feu]['old_ids_incomplete_now'] = int(np.isin(oe, e[~c]).sum())
                    a['feu'][feu]['old_ids_absent_now'] = int((~np.isin(oe, e)).sum())
                except Exception as x:
                    a['feu'][feu]['old_error'] = str(x)
        ch = glob.glob(os.path.join(sub, 'combined_hits_root', f'*_{tag}_feu-combined_hits.root'))
        if ch:
            try:
                he = np.unique(uproot.open(ch[0])['hits'].arrays(['eventId'], library='np')['eventId'].astype(np.int64))
                a['hits_events'] = int(len(he))
                inc = np.unique(np.concatenate([e[~c] for feu, (e, c) in E.items() if feu != '01'] or [np.zeros(0)]))
                a['hits_events_incomplete_now'] = int(np.isin(he, inc).sum())
            except Exception as x:
                a['hits_error'] = str(x)
        out['acq'][f'{run}/{s}/{tag}'] = a
    json.dump(out, open(sys.argv[2], 'w'), indent=1)
    nf = len(out['files'])
    print(f"files audited {nf}, missing {len(out['missing_audit'])}, NOT identical {len(out['not_identical'])}")


if __name__ == '__main__':
    main()
