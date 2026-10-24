"""Trace the M1 card copper: which pogo pad (bottom, 10x30 grid on 1 mm) is wired
to which Mec8 pin (top, 4 connectors x 2 rows x 35). Pure geometric connectivity
(net attributes are stripped near the pogo field)."""
import json, sys, numpy as np, collections
import gerber_flash as g

G = '/home/dylan/CLionProjects/MX17_Geant/design/gerbers/M1/DFS3498AM1_'
LAYERS = ['top', 'L2', 'L3', 'L4', 'L5', 'bot']


class UF:
    def __init__(s, n): s.p = list(range(n))
    def f(s, a):
        while s.p[a] != a:
            s.p[a] = s.p[s.p[a]]; a = s.p[a]
        return a
    def u(s, a, b): s.p[s.f(a)] = s.f(b)


def ap_radius(ap):
    t, p = ap
    if t == 'C': return p[0] / 2
    if t in ('R', 'O'): return min(p[0], p[1]) / 2
    if t == 'BOXR': return min(p[0], p[1]) / 2
    return 0.1


def seg_pt_dist(a, b, p):
    a, b, p = map(np.asarray, (a, b, p))
    ab = b - a; L2 = ab @ ab
    t = 0 if L2 == 0 else np.clip((p - a) @ ab / L2, 0, 1)
    return np.hypot(*(a + t * ab - p))


def build():
    items = []   # (layer, kind, data)
    for L in LAYERS:
        aps, fl, dr = g.parse(G + L + '.gbr', keep_draws=True)
        for q in fl:
            if q.polarity != 'D': continue
            items.append((L, 'F', (q.x, q.y, ap_radius(aps[q.ap]), aps[q.ap], q.ap)))
        for (x0, y0, x1, y1, ap, net, pol) in dr:
            if pol != 'D': continue
            items.append((L, 'S', (x0, y0, x1, y1, ap_radius(aps[ap]))))
    aps, holes, _ = g.parse(G + 'pla.gbr')
    uf = UF(len(items) + len(holes))
    by_layer = collections.defaultdict(list)
    for i, it in enumerate(items): by_layer[it[0]].append(i)
    for L, idx in by_layer.items():
        F = [i for i in idx if items[i][1] == 'F']
        S = [i for i in idx if items[i][1] == 'S']
        fx = np.array([items[i][2][:3] for i in F])
        fhw = np.array([(items[i][2][3][1][0] / 2, items[i][2][3][1][1] / 2) if items[i][2][3][0] == 'R'
                        else (0, 0) for i in F])
        # segment endpoints vs flashes and segments
        ends = []
        for i in S:
            x0, y0, x1, y1, r = items[i][2]
            ends += [(x0, y0, r, i), (x1, y1, r, i)]
        ends = np.array(ends)
        for (x, y, r, i) in ends:
            i = int(i)
            d = np.hypot(fx[:, 0] - x, fx[:, 1] - y)
            hit = d < fx[:, 2] + r * 0.5
            hit |= (np.abs(fx[:, 0] - x) <= fhw[:, 0]) & (np.abs(fx[:, 1] - y) <= fhw[:, 1])
            for j in np.nonzero(hit)[0]:
                uf.u(i, F[j])
            d = np.hypot(ends[:, 0] - x, ends[:, 1] - y)
            for j in np.nonzero(d < 1e-3)[0]:
                uf.u(i, int(ends[j, 3]))
        # T junctions: endpoint lying on another segment
        segs = np.array([items[i][2] for i in S])
        for (x, y, r, i) in ends:
            a = segs[:, :2]; b = segs[:, 2:4]; ab = b - a
            L2 = (ab ** 2).sum(1); L2[L2 == 0] = 1e-12
            t = np.clip(((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1]) / L2, 0, 1)
            d = np.hypot(a[:, 0] + t * ab[:, 0] - x, a[:, 1] + t * ab[:, 1] - y)
            for j in np.nonzero(d < (segs[:, 4] + r) * 0.5)[0]:
                uf.u(int(i), S[j])
        # flash-flash overlap (pad + via ring)
        for a_ in range(len(F)):
            d = np.hypot(fx[:, 0] - fx[a_, 0], fx[:, 1] - fx[a_, 1])
            for j in np.nonzero(d < (fx[:, 2] + fx[a_, 2]) * 0.9)[0]:
                uf.u(F[a_], F[j])
    # plated holes join every layer's flash at that position
    for k, h in enumerate(holes):
        hid = len(items) + k
        for L, idx in by_layer.items():
            for i in idx:
                if items[i][1] == 'F':
                    x, y, r = items[i][2][:3]
                    if np.hypot(x - h.x, y - h.y) < 0.02:
                        uf.u(hid, i)
    return items, uf


def main():
    items, uf = build()
    pogo = [(i, it) for i, it in enumerate(items) if it[0] == 'bot' and it[1] == 'F' and it[2][3][0] == 'BOXR']
    mec = [(i, it) for i, it in enumerate(items) if it[0] == 'top' and it[1] == 'F' and it[2][3][0] == 'R' and it[2][3][1] == [0.5, 2.0]]
    # Mec8 indexing: connector by row pair (sorted y), row 0/1, pin 0..34 along x
    rows = sorted(set(round(it[2][1], 2) for _, it in mec))
    mec_info = {}
    for i, it in mec:
        y = round(it[2][1], 2); r = rows.index(y)
        xs = sorted(round(q[2][0], 3) for _, q in mec if round(q[2][1], 2) == y)
        mec_info[i] = dict(conn=r // 2, row=r % 2, pin=xs.index(round(it[2][0], 3)), x=it[2][0], y=y)
    root2mec = collections.defaultdict(list)
    for i in mec_info: root2mec[uf.f(i)].append(i)
    out = []
    for i, it in pogo:
        x, y = it[2][0], it[2][1]
        col = int(round(x - 228.5)); rowp = int(round(y - 85.34))
        ms = root2mec.get(uf.f(i), [])
        out.append(dict(x=x, y=y, col=col, row=rowp, mec=[mec_info[m] for m in ms]))
    n1 = sum(len(o['mec']) == 1 for o in out); n0 = sum(len(o['mec']) == 0 for o in out)
    print('pogo pads', len(out), 'to exactly one Mec8 pin', n1, 'to none', n0, 'to many', len(out) - n1 - n0)
    json.dump(out, open('m1_pogo_to_mec8.json', 'w'), indent=0)


if __name__ == '__main__':
    main()
