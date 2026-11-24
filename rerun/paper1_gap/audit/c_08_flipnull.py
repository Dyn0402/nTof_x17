import sys, json
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from c_lib import *
class Flip:
    def __init__(s, mp, fu, fw): s.mp, s.fu, s.fw = mp, fu, fw
    def ev(s, U, W): return s.mp.ev(-U if s.fu else U, -W if s.fw else W)
D = {k: load(rel) for k, rel in ROWS.items()}
out = {}
for p in 'xy':
    T = view(D['r625'], p)
    G = {q: np.concatenate([view(D[k], p)[q] for k in ('run1', 'run2')]) for q in ('U', 'W', 'w', 'tr')}
    for n in (4, 6):
        mp = Maps(G['U'], G['W'], G['w'], G['tr'], n)
        # fine column dropped: reference RSS with plane only
        base = None
        for fu, fw in ((0, 0), (1, 0), (0, 1), (1, 1)):
            r = kfit(T, Flip(mp, fu, fw))
            # RSS with fine column removed: refit without
            ap, af, s, _ = Flip(mp, fu, fw).ev(T['U'], T['W'])
            X = np.vstack([np.ones(len(ap)), T['tr'], T['tr'] * s, ap]).T
            c, keep, rss0, _ = lsq_clip(X, T['w'])
            Xf = np.vstack([np.ones(len(ap)), T['tr'], T['tr'] * s, ap, af]).T
            cf = np.linalg.lstsq(Xf[keep], T['w'][keep], rcond=None)[0]
            rss1 = float(np.sum((T['w'] - Xf @ cf)[keep] ** 2)); rss0k = float(np.sum((T['w'] - X @ c)[keep] ** 2))
            out[f'{p}{n} fU{fu} fW{fw}'] = dict(kf=r['kf'], dRSS_pct=100 * (rss0k - rss1) / rss0k, nkeep=int(keep.sum()))
            print(p, n, fu, fw, {a: round(b, 3) for a, b in out[f'{p}{n} fU{fu} fW{fw}'].items()}, flush=True)
json.dump(out, open(f'{OUT}/flipnull.json', 'w'), indent=1)
