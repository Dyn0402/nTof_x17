"""Fine scan around the best basins of fit_scan.py, both contact modes. Writes refine.json."""
import json, numpy as np, multiprocessing as mp
import model as M
C = np.load('corr_feu6.npz')['C']; MEAS = C[M._PA, M._PB] > 0.9
BASINS = [(-0.7, 0.35, -0.55), (1.9, -0.45, -0.6), (-0.8, 0.45, 0.45), (0.6, -0.45, 0.55)]


def work(args):
    th, mode, (t0, x0, y0) = args
    best = []
    for rp in np.arange(0.15, 0.3501, 0.025):
        for dx in np.arange(x0 - 0.1, x0 + 0.1001, 0.01):
            for dy in np.arange(y0 - 0.1, y0 + 0.1001, 0.01):
                l = M.fast_labels(dx, dy, th, rp, mode)
                tp, fp, fn = M.fast_score(l[0], l[1], MEAS)
                best.append((2 * tp / (2 * tp + fp + fn), tp, fp, fn, float(th), float(rp), float(dx), float(dy), mode))
    best.sort(reverse=True)
    return best[:3]


if __name__ == '__main__':
    jobs = [(t0 + dt, mode, b) for b in BASINS for mode in ('readout', 'carrier') for dt in np.arange(-0.5, 0.501, 0.1) for t0 in [b[0]]]
    with mp.Pool(8) as p:
        res = [r for rr in p.map(work, jobs) for r in rr]
    res.sort(reverse=True)
    out = [dict(f1=r[0], tp=r[1], fp=r[2], fn=r[3], th=r[4], rp=r[5], dx=r[6], dy=r[7], mode=r[8]) for r in res[:40]]
    json.dump(out, open('refine.json', 'w'), indent=1)
    for o in out[:10]: print(o)
    for mode in ('readout', 'carrier'):
        print(mode, max((o for o in out if o['mode'] == mode), key=lambda o: o['f1'], default=None))
