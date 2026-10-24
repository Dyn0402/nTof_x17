"""Pick the Mec8-pin -> DREAM-channel convention: which choice makes the measured
short groups spatially compact on the pogo grid."""
import json, itertools, numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

POGO = json.load(open('m1_pogo_to_mec8.json'))
CL = json.load(open('/home/dylan/PycharmProjects/nTof_x17/rerun/channel_layer/det7_g_det7_long.json'))


def sig_index(pin):
    return pin - 1 if pin < 17 else pin - 2    # grounds at pins 0, 17, 34


def feu_positions(conv, perm, card_conns=(0, 1, 2, 3)):
    """conv: (row_first, row_swap, invert); perm: mec8 connector -> DREAM connector slot."""
    row_swap, invert, pin_rev = conv
    pos = {}
    for o in POGO:
        if len(o['mec']) != 1: continue
        m = o['mec'][0]
        s = sig_index(m['pin'])
        if pin_rev: s = 31 - s
        r = m['row'] ^ row_swap
        loc = 2 * s + r
        if invert: loc = 63 - loc
        feu = 64 * perm[m['conn']] + loc
        pos[feu] = (o['col'], o['row'])
    return pos


def score(pos, groups):
    good = tot = 0; L = []
    for g in groups:
        p = np.array([pos[c] for c in g if c in pos], float)
        if len(p) < 2: continue
        d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
        t = minimum_spanning_tree(d).toarray(); e = t[t > 0]
        good += (e <= 1.01).sum(); tot += len(e); L += list(e)
    return good, tot, np.mean(L)


if __name__ == '__main__':
    groups = CL['x']['short_groups']
    res = []
    for conv in itertools.product([0, 1], repeat=3):
        for perm in itertools.permutations(range(4)):
            pos = feu_positions(conv, perm)
            g, t, m = score(pos, groups)
            res.append((g, -m, conv, perm, t))
    res.sort(reverse=True)
    for r in res[:12]: print(r)
    print('worst', res[-1])
