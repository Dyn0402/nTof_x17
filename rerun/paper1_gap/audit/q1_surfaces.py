"""q1_surfaces.py -- item 1: about 70 % of the reproducible fine tilt is not described by the gradient of the gap.  Is the
rest a fine structure of the mesh?  Two-surface model: tilt potential P = h + a g (mm), drift-time gap g.  So the fine mesh
height predicted from timing and tilt alone is h_pred = P - a g, and the fine cathode c_pred = P + (1 - a) g.  The position
route measures h directly: per cell, the slope of (p0 - ref_mesh) against tan_ref.  We compare fine parts (quadratic
removed) on 6 x 6 cells of the golden mounting.

P: the audit's spline map (results/a2_final_map_det3.json; Phi in um/ns x (100 mm), P = 100 Phi / v, Y scaled by kappa).
g: cell mean of walk-corrected ln q_u90 x G0 / (1 + beta), beta 0.69.  h_pos: per cell, per run (cross-run products).
-> printed; numbers copied into q_RESULT.md"""
import json
import numpy as np
from r3_lib import load_mounting, DL, MOUNT
import a2_lib as A2

D, info, bundle = load_mounting('A628')
M = json.load(open('results/a2_final_map_det3.json'))
N = 6
eu, ew = np.linspace(-1.75, 1.75, N + 1), np.linspace(-1.55, 1.55, N + 1)
Uc, Wc = 0.5 * (eu[1:] + eu[:-1]), 0.5 * (ew[1:] + ew[:-1])
UU, WW = np.meshgrid(Uc, Wc, indexing='ij')
Qm = np.c_[np.ones(N * N), UU.ravel(), WW.ravel(), UU.ravel() ** 2, WW.ravel() ** 2, UU.ravel() * WW.ravel()]


def fine(v):
    v = v.ravel(); ok = np.isfinite(v); r = np.full_like(v, np.nan)
    r[ok] = v[ok] - Qm[ok] @ np.linalg.lstsq(Qm[ok], v[ok], rcond=None)[0]
    return r


# P on cells: average of Phi over a fine sub-grid of each cell
K = M['K']; phi = np.array(M['phi'])
sub = np.linspace(0, 1, 7)[1:-1]
P = np.zeros((N, N))
for a in range(N):
    for b in range(N):
        uu = eu[a] + (eu[a + 1] - eu[a]) * sub; ww = ew[b] + (ew[b + 1] - ew[b]) * sub
        U_, W_ = [x.ravel() for x in np.meshgrid(uu, ww, indexing='ij')]
        B = A2.tensor(A2.basis(U_, K, *M['U_domain']), A2.basis(W_, K, *M['W_domain']))
        P[a, b] = np.mean(B @ phi)
v = 0.5 * (M['v_used']['x'] + M['v_used']['y'])
P = 100 * P / v                       # mm

ci, cj = np.searchsorted(eu, D.U) - 1, np.searchsorted(ew, D.W) - 1
inside = (ci >= 0) & (ci < N) & (cj >= 0) & (cj < N)
# position route needs p0 - ref_mesh: reload the raw columns in load_mounting order
ref = {p: np.concatenate([DL.load(k)[f'ref_mesh_{p}'].values for k in MOUNT['A628']['rows']]) for p in 'xy'}
p0 = {'x': D.p0x, 'y': D.p0y}
trr = {'x': D.trx, 'y': D.trY}


def maps(r):
    g = np.full((N, N), np.nan); h = {p: np.full((N, N), np.nan) for p in 'xy'}
    for a in range(N):
        for b in range(N):
            m = inside & (ci == a) & (cj == b) & (D.row == r)
            ok = m & np.isfinite(D.lnq)
            if ok.sum() > 50:
                g[a, b] = 28.2 / 1.69 * np.mean(D.lnq[ok])
            for p in 'xy':
                s = m & (np.abs(trr[p]) <= 0.6)
                if s.sum() > 50:
                    y = p0[p][s] - ref[p][s]; t = trr[p][s]
                    keep = np.ones(s.sum(), bool)
                    for _ in range(3):
                        c = np.polyfit(t[keep], y[keep], 1)
                        res = y - np.polyval(c, t)
                        keep = np.abs(res) < 3.5 * 1.4826 * np.median(np.abs(res))
                    h[p][a, b] = c[0]
    return g, h


G, H = {}, {}
for r in (0, 1):
    G[r], H[r] = maps(r)


def xc(A0, A1, B0, B1):
    """cross-run correlation of fine maps: A from one run with B from the other"""
    def c(a, b):
        a, b = fine(a), fine(b); ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
        return a @ b / np.sqrt(a @ a * b @ b), (a @ b) / (a @ a)
    return [round(float(x), 2) for x in np.mean([c(A0, B1), c(A1, B0)], axis=0)]


hpos = {r: 0.5 * (H[r]['x'] + H[r]['y']) for r in (0, 1)}
print('position-route h: x vs y view (cross-run r, slope):', xc(H[0]['x'], H[1]['x'], H[0]['y'], H[1]['y']))
print('position-route h: run 1 vs run 2 (fine):', xc(hpos[0], hpos[1], hpos[0], hpos[1]))
print('fine rms: P %.3f mm, g %.3f mm, h_pos %.3f mm' % (np.nanstd(fine(P)), np.nanstd(fine(0.5 * (G[0] + G[1]))),
                                                     np.nanstd(fine(0.5 * (hpos[0] + hpos[1])))))
for a in (0.0, 0.2, 0.4, 0.6):
    hp = {r: P - a * G[r] for r in (0, 1)}
    print(f'a = {a:.1f}: r(h_pred = P - a g, h_pos) fine, cross-run [r, slope h_pos on h_pred]:', xc(hp[0], hp[1], hpos[0], hpos[1]))
print('r(g, h_pos) fine:', xc(G[0], G[1], hpos[0], hpos[1]))


# ---- cross-run version: the tilt potential refitted per run (tilt-only spline, R3's fit_generic), so P of one run is
# compared with h_pos and g of the other run, and no event enters both sides.
from r3_lib import Spec, fit_generic
import scipy.sparse as sp_
Pr = {}
for r in (0, 1):
    Dr = D.sub(D.row == r)
    m = fit_generic(Spec('phi', 12, 0, 1.0), Dr, 0.1)
    c = m['c']; sp = m['spec']; a0, a1 = sp.off['phi']
    Pc = np.zeros((N, N))
    for a in range(N):
        for b in range(N):
            uu = eu[a] + (eu[a + 1] - eu[a]) * sub; ww = ew[b] + (ew[b + 1] - ew[b]) * sub
            U_, W_ = [x.ravel() for x in np.meshgrid(uu, ww, indexing='ij')]
            B = A2.tensor(A2.basis(U_, 12, *A2.UD), A2.basis(W_, 12, *A2.WD))
            Pc[a, b] = np.mean(B @ c[a0:a1])
    Pr[r] = 100 * Pc        # wx, wy are in tan units (normalised per row), so Phi is already in 100 mm x tan
print('P run 1 vs run 2 (fine):', xc(Pr[0], Pr[1], Pr[0], Pr[1]), ' P per run vs a2 union map:', xc(Pr[0], Pr[1], P, P))
for a in (0.0, 0.2, 0.4, 0.6):
    hp = {r: Pr[r] - a * G[r] for r in (0, 1)}
    print(f'cross-run, a = {a:.1f}: r(h_pred(run i), h_pos(run j)) [r, slope]:', xc(hp[0], hp[1], hpos[0], hpos[1]))
