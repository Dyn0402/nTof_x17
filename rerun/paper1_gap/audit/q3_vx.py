"""q3_vx.py -- item 3: the X-view velocity map varies across chamber A more than the gap explains.  Does saturation change
the angle scale (ladder slope) and, through the saturated fraction that varies with position, add structure to the v map?
Golden mounting, wx/wy normalised per row (r3_lib).  -> printed; numbers copied into q_RESULT.md."""
import numpy as np
from r3_lib import load_mounting
import track_test as TT

D, info, _ = load_mounting('A628')
sat = (D.satx > 0) | (D.saty > 0); satx = D.satx > 0; saty = D.saty > 0
rng = np.random.default_rng(3)


def v(m, p):
    tr = D.trx if p == 'x' else D.trY; w = D.wx if p == 'x' else D.wy
    s = m & (np.abs(tr) <= 0.6)
    return TT.robust_const(tr[s], w[s], 0.0, 1.0)[1]


for p, sp in (('x', satx), ('y', saty)):
    a, b = v(sp, p), v(~sp, p)
    bs = []
    for _ in range(100):
        idx = rng.integers(0, D.n, D.n); S = sp[idx]
        tr = (D.trx if p == 'x' else D.trY)[idx]; w = (D.wx if p == 'x' else D.wy)[idx]; s = np.abs(tr) <= 0.6
        bs.append(TT.robust_const(tr[s & S], w[s & S], 0.0, 1.0)[1] / TT.robust_const(tr[s & ~S], w[s & ~S], 0.0, 1.0)[1])
    print(f'view {p}: angle scale saturated / unsaturated (this view) = {a / b:.4f} +- {np.std(bs):.4f}')

N = 6
eu, ew = np.linspace(-1.75, 1.75, N + 1), np.linspace(-1.55, 1.55, N + 1)
ci, cj = np.searchsorted(eu, D.U) - 1, np.searchsorted(ew, D.W) - 1
inside = (ci >= 0) & (ci < N) & (cj >= 0) & (cj < N)


def cmap(fn, r, extra=None):
    M = np.full((N, N), np.nan)
    for a in range(N):
        for b in range(N):
            m = inside & (ci == a) & (cj == b) & (D.row == r)
            if extra is not None:
                m &= extra
            if m.sum() > 60:
                M[a, b] = fn(m)
    return M


def cc(A, B):
    ok = np.isfinite(A) & np.isfinite(B); a = A[ok] - A[ok].mean(); b = B[ok] - B[ok].mean()
    return a @ b / np.sqrt(a @ a * b @ b)


M = {}
for r in (0, 1):
    M[r] = dict(vx=cmap(lambda m: np.log(v(m, 'x')), r), vy=cmap(lambda m: np.log(v(m, 'y')), r),
                vxu=cmap(lambda m: np.log(v(m, 'x')), r, ~satx), vyu=cmap(lambda m: np.log(v(m, 'y')), r, ~saty),
                sf=cmap(lambda m: sat[m].mean(), r))
for k in ('vx', 'vxu', 'vy', 'vyu'):
    signal = np.sqrt(max(np.nanmean((M[0][k] - np.nanmean(M[0][k])) * (M[1][k] - np.nanmean(M[1][k]))), 0))
    print(f'{k:4s}: cross-run r {cc(M[0][k], M[1][k]):+.2f}, reproducible rms {signal:.4f}; '
          f'r with the other run\'s saturated fraction {0.5 * (cc(M[0][k], M[1]["sf"]) + cc(M[1][k], M[0]["sf"])):+.2f}')
print('r(vx, vy) cross-run', round(0.5 * (cc(M[0]['vx'], M[1]['vy']) + cc(M[1]['vx'], M[0]['vy'])), 2),
      ' unsat-only', round(0.5 * (cc(M[0]['vxu'], M[1]['vyu']) + cc(M[1]['vxu'], M[0]['vyu'])), 2))
