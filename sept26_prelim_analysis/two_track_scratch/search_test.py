"""Is production's two-track miss a search failure? Same window, same parent."""
import os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sept26_prelim_analysis import two_track_limit as tl

CELLS = [('A', 0.0, 2.0), ('A', 0.0, 3.0), ('C', 0.3, 4.0), ('C', 0.3, 6.0), ('A', 0.3, 10.0)]
N = 24


def job(args):
    from wft import model as wm, reco as wr
    arm, tan, d, seed = args
    tl._init(arm)
    wm.DEAD, wm.HOT = {}, {}
    rng = np.random.default_rng(seed)
    w = tan * tl.ts._CAL.v_drift * 1e-3
    p0a = float(100.0 + rng.uniform(0.0, 0.78))
    tr = [(p0a, w, 0.0), (p0a + d, w, 0.0)]
    P = tl.synth_window('x', [t + (tl.QTOT,) for t in tr], rng)
    live = np.flatnonzero(P.W.max(axis=1) / tl.NOISE > tl.ts.SIG_SEED)
    lo, hi = max(0, live.min() - 3), min(len(P.pos) - 1, live.max() + 3)
    sl = slice(lo, hi + 1)
    win = dict(W=P.W[sl], pos=P.pos[sl], noise=P.noise[sl],
               ch=np.round(P.pos[sl] / wm.PITCH).astype(int))
    f = wr.fit_plane(win, 'x', tl.ts._CAL)
    probe = wr.two_track_probe(win, 'x', f, tl.ts._CAL.hyper)
    Wp, npr, pos, sat = probe['W'], probe['noise'], probe['pos'], probe['sat']
    Q = tl.Plane('x', Wp, pos, npr, sat)
    out = dict(arm=arm, tan=tan, d=d, seed=seed, parent_tan=f.w / 42.6e-3,
               parent_t0=f.t0, chi_one=probe['chi_one'])
    # (a) production search
    r = wr.fit_plane_two(win, 'x', tl.ts._CAL, f, probe=probe)
    kids = [(c.p0, c.w, c.t0) for c in r['children']]
    out['a_found'] = tl.match(kids, tr)[0]
    out['a_chi2'] = r['chi2_two']
    # (b) oracle-style search, no truth
    c2, x2 = tl.fit_two(Q, (f.p0, f.w, f.t0), None)
    out['b_found'] = tl.match([(x2[0], x2[1], x2[4]), (x2[2], x2[3], x2[4])], tr)[0]
    out['b_chi2'] = c2
    # (c) production's own starts, oracle optimiser (all refined, restarted, no barrier)
    parent_r = dict(p0=f.p0, w=f.w, t0=f.t0, q=probe['q_one'])
    starts = wr._two_track_starts(Wp, npr, pos, sat, 'x', parent_r, tl.ts._CAL.hyper)
    best = (np.inf, None)
    for pa, pb, _tie in starts:
        v0 = (pa[0], pa[1], pb[0], pb[1], 0.5 * (pa[2] + pb[2]))
        b = tl._nm(Q.two, v0, [0.3, 2e-3, 0.3, 2e-3, 20.0])
        if b[0] < best[0]:
            best = b
    x = best[1]
    out['c_found'] = tl.match([(x[0], x[1], x[4]), (x[2], x[3], x[4])], tr)[0]
    out['c_chi2'] = best[0]
    # (d) global grid of parallel line pairs, tied t0, then NM from the best 3
    v = tl.ts._CAL.v_drift * 1e-3
    grid = []
    ps = pos[np.argsort(pos)]
    for tq in np.arange(-0.4, 0.401, 0.1):
        for dt in (-60.0, 0.0, 60.0):
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    if ps[j] - ps[i] > 14.0:
                        break
                    grid.append((Q.two((ps[i], tq * v, ps[j], tq * v, f.t0 + dt)),
                                 (ps[i], tq * v, ps[j], tq * v, f.t0 + dt)))
    grid.sort(key=lambda g: g[0])
    best = (np.inf, None)
    for _c, v0 in grid[:3]:
        b = tl._nm(Q.two, v0, [0.3, 2e-3, 0.3, 2e-3, 20.0])
        if b[0] < best[0]:
            best = b
    x = best[1]
    out['d_found'] = tl.match([(x[0], x[1], x[4]), (x[2], x[3], x[4])], tr)[0]
    out['d_chi2'] = best[0]
    out['d_ngrid'] = len(grid)
    # truth-seeded reference on the same window
    ct = tl._nm(Q.two, (tr[0][0], w, tr[1][0], w, 0.0), [0.3, 2e-3, 0.3, 2e-3, 20.0])
    out['truth_chi2'] = ct[0]
    return out


if __name__ == '__main__':
    J = [(a, t, d, 777000 + 100 * i + k) for i, (a, t, d) in enumerate(CELLS) for k in range(N)]
    with ProcessPoolExecutor(14) as ex:
        R = pd.DataFrame(list(ex.map(job, J)))
    R.to_parquet(__file__.replace('.py', '.parquet'))
    R['a_gap'] = R.a_chi2 - R.truth_chi2
    R['c_gap'] = R.c_chi2 - R.truth_chi2
    R['d_gap'] = R.d_chi2 - R.truth_chi2
    print(R.groupby(['arm', 'tan', 'd'])[['a_found', 'b_found', 'c_found', 'd_found']].mean().round(2))
    print(R.groupby(['arm', 'tan', 'd'])[['a_gap', 'c_gap', 'd_gap', 'parent_tan', 'd_ngrid']].median().round(1))
