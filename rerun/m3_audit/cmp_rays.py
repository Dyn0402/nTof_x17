#!/usr/bin/env python3
"""cmp_rays.py <a_rays.root> <b_rays.root> [label] -- compare two M3 rays files event by event.

Counts, equality of the shared branches, and the quality-cut yield (NClusX>=3 & NClusY>=3 & chi2<5 on both
coordinates, the production recipe) with the events gained/lost between the two.
"""
import sys

import awkward as ak
import numpy as np
import uproot


def good(t):
    """Per event: has at least one ray passing the production recipe."""
    c = (t['NClusX'] >= 3) & (t['NClusY'] >= 3) & (t['Chi2X'] < 5) & (t['Chi2Y'] < 5) & (t['Chi2X'] > 0) & (t['Chi2Y'] > 0)
    return ak.to_numpy(ak.any(c, axis=1))


def main():
    a, b = uproot.open(sys.argv[1])['T'], uproot.open(sys.argv[2])['T']
    lab = sys.argv[3] if len(sys.argv) > 3 else ''
    ka, kb = set(a.keys()), set(b.keys())
    common = sorted(ka & kb)
    A, B = a.arrays(common), b.arrays(common)
    # the tracking runs multi-threaded (config thread_number): entry order is not reproducible, so key on evn
    A = A[np.argsort(ak.to_numpy(A['evn']), kind='stable')]
    B = B[np.argsort(ak.to_numpy(B['evn']), kind='stable')]
    print(f'{lab} events {a.num_entries} vs {b.num_entries}; branches only-a {sorted(ka - kb)[:6]} only-b {sorted(kb - ka)[:6]}')
    if a.num_entries == b.num_entries:
        diff = {}
        for k in common:
            x, y = A[k], B[k]
            if x.ndim == 1:
                n = int(ak.sum(x != y))
            else:
                same_len = ak.to_numpy(ak.num(x) == ak.num(y))
                n = int((~same_len).sum())
                if same_len.any():
                    fx, fy = ak.to_numpy(ak.flatten(x[same_len], axis=None)), ak.to_numpy(ak.flatten(y[same_len], axis=None))
                    bad = ~((fx == fy) | (np.isnan(fx) & np.isnan(fy)))
                    n += int(bad.sum() > 0) * int(bad.sum())
            if n:
                diff[k] = n
        print(f'  branches that differ (entries/values): {diff if diff else "none (identical)"}')
    ea, eb = ak.to_numpy(A['evn']), ak.to_numpy(B['evn'])
    ga, gb = good(A), good(B)
    sa, sb = set(ea[ga]), set(eb[gb])
    nra = int(ak.sum(ak.num(A['Chi2X']))); nrb = int(ak.sum(ak.num(B['Chi2X'])))
    print(f'  rays {nra} vs {nrb}; events with a good ray {len(sa)} vs {len(sb)} ({100*(len(sb)-len(sa))/max(len(sa),1):+.2f} %);'
          f' gained {len(sb - sa)}, lost {len(sa - sb)}')


if __name__ == '__main__':
    main()
