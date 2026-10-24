"""Non-rigid diagnostic: for every 3 x 3 window of the pogo grid, the uniform pin displacement
(dx, dy) that best reproduces the measured neighbour shorts inside that window. If the
connector were displaced rigidly (shift + small rotation), the window displacements would
form a smooth linear field. Writes local_fit.json."""
import json, numpy as np
import model as M

C = np.load('corr_feu6.npz')['C']
MEAS = C[M._PA, M._PB] > 0.9
D = np.round(np.arange(-0.5, 0.5001, 0.025), 4)
RP = [0.13, 0.2, 0.3]
POSA = np.array([M.POS[a] for a in M._PA]); POSB = np.array([M.POS[b] for b in M._PB])

if __name__ == '__main__':
    pred = {}
    for rp in RP:
        for dx in D:
            for dy in D:
                l = M.fast_labels(dx, dy, 0.0, rp)
                pred[(rp, dx, dy)] = (l[0][M._PA] == l[0][M._PB]) & ~l[1][M._PA]
    out = []
    for r0 in range(0, 28):
        for c0 in range(0, 8):
            inw = lambda P: (P[:, 0] >= c0) & (P[:, 0] < c0 + 3) & (P[:, 1] >= r0) & (P[:, 1] < r0 + 3)
            w = inw(POSA) & inw(POSB)
            if w.sum() < 6:
                continue
            m = MEAS[w]
            best = None
            for k, p in pred.items():
                err = int((p[w] != m).sum())
                if best is None or err < best[0]:
                    best = (err, k, [kk for kk, pp in pred.items() if int((pp[w] != m).sum()) == err])
            err, k, ties = best
            out.append(dict(c0=c0, r0=r0, n_pairs=int(w.sum()), n_short=int(m.sum()), err=err,
                            n_ties=len(ties), dx_range=[min(t[1] for t in ties), max(t[1] for t in ties)],
                            dy_range=[min(t[2] for t in ties), max(t[2] for t in ties)]))
    json.dump(out, open('local_fit.json', 'w'), indent=0)
    for o in out:
        if o['n_short']:
            print(o)
