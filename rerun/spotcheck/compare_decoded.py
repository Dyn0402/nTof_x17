#!/usr/bin/env python3
"""compare_decoded.py <new.root> <eos.root> <out.json> -- branch-by-branch equality of two decoded_root files.

The rerun keeps the EOS decoded_root (Dylan, 2026-10-23); this proves a fresh decode of the same .fdf with the
pinned decoder gives the same events.  Compares the newest cycle of tree ``nt``: entry count, then every branch
exactly (event-aligned).  Extra trees in the new file (decode_stats) are reported, not compared.
"""
import json
import sys

import awkward as ak
import numpy as np
import uproot

new, old, out = sys.argv[1:4]
fn, fo = uproot.open(new), uproot.open(old)
tn, to = fn['nt'], fo['nt']
r = dict(new=new, old=old, n_new=int(tn.num_entries), n_old=int(to.num_entries),
         trees_new=sorted(set(k.split(';')[0] for k in fn.keys())), trees_old=sorted(set(k.split(';')[0] for k in fo.keys())),
         branches={})
n = min(r['n_new'], r['n_old'])
for b in to.keys():
    if b not in tn.keys():
        r['branches'][b] = 'missing in new'
        continue
    a, c = tn[b].array(entry_stop=n), to[b].array(entry_stop=n)
    if a.ndim == 1 and not isinstance(a.layout, ak.contents.ListOffsetArray):
        eq = np.asarray(a) == np.asarray(c)
        r['branches'][b] = dict(n_diff_entries=int((~eq).sum()))
    else:
        ln, lo = ak.num(a), ak.num(c)
        same_len = np.asarray(ln == lo)
        nd = int((~same_len).sum())
        ok = same_len.copy()
        idx = np.where(same_len)[0]
        if len(idx):
            fa, fc = ak.flatten(a[idx]), ak.flatten(c[idx])
            eqv = ak.all(a[idx] == c[idx], axis=1)
            ok[idx] = np.asarray(eqv)
        r['branches'][b] = dict(n_len_diff=nd, n_diff_entries=int((~ok).sum()))
r['identical'] = (r['n_new'] == r['n_old'] and
                  all(isinstance(v, dict) and v['n_diff_entries'] == 0 for v in r['branches'].values()))
json.dump(r, open(out, 'w'), indent=1)
print(json.dumps(dict(identical=r['identical'], n_new=r['n_new'], n_old=r['n_old'])))
