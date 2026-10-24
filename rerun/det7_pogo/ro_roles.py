"""Readout board (DFS3498A) pogo pad field: for each of the 4 fields of 10 x 30 pads, which
pads have copper attached on any inner layer (a trace or plane touching the pad's via).
Compares the roles with the M1 card's (signal / GND / NC). Region-limited parse."""
import sys, numpy as np, collections, json
import gerber_flash as g
D = '/home/dylan/CLionProjects/MX17_Geant/design/gerbers/readout_pcb/'
aps, fl, _ = g.parse(D + 'DFS3498A_top-cu.gbr')
pads45 = [(q.x, q.y) for q in fl if aps[q.ap][0] == 'BOXR' and aps[q.ap][1][3] == 45.0]
pads135 = [(q.x, q.y) for q in fl if aps[q.ap][0] == 'BOXR' and aps[q.ap][1][3] == 135.0]
vias = np.array([(q.x, q.y) for q in fl if q.ap == 17])
print('pads', len(pads45), len(pads135), 'vias', len(vias))
P = np.array(pads45 + pads135)
# via of each pad
vi = [np.argmin(np.hypot(*(vias - p).T)) for p in P]
off = vias[vi] - P
print('pad->via offsets', collections.Counter(tuple(np.round(o, 3)) for o in off))
res = {}
for L in ['DFS3498A_top-cu', 'DFS3498A_L2-pads', 'DFS3498A_L3-TrackY', 'DFS3498A_L4-TrackX', 'DFS3498A_L5']:
    aps, fl, dr = g.parse(D + L + '.gbr', keep_draws=True)
    E = np.array([(x0, y0) for (x0, y0, x1, y1, a, n, p) in dr] + [(x1, y1) for (x0, y0, x1, y1, a, n, p) in dr]) if dr else np.zeros((0, 2))
    F = np.array([(q.x, q.y) for q in fl])
    near = (np.abs(E[:, 0]) > 220) | (np.abs(E[:, 1]) > 220)
    E = E[near]
    hit = []
    for v in vias[vi]:
        d = np.hypot(*(E - v).T) if len(E) else np.array([9.])
        hit.append(int((d < 0.05).sum()))
    res[L] = hit
    print(L, 'draws', len(dr), 'pads with a trace end at their via:', sum(h > 0 for h in hit))
json.dump(dict(pads=P.tolist(), traces=res), open('ro_pad_traces.json', 'w'))
