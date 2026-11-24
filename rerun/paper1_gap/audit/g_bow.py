"""g_bow.py -- quadratic level (audit test G, step 5): does the n_TOF residual map beyond the bench bow block say that the block
over- or under-corrects at n_TOF?  Also a registration / sign scan of the bench shape in the held-out sigma (audit test G, step 4 cross-check).

    r_ntof(x_local, y_local) = (f - 1) * SGN * a_bow(U, W)  (+ constant)   ->  f = 1: the block is right at n_TOF, f = 0: it should not be applied
"""
import itertools
import json
import os
import pickle
import sys

os.environ.setdefault('X17_TQ_OUT', '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/g/tq')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import g_common as G
import g_direct as GD
from rerun.ntof.improve import angle_map as AM
from wft import reco as R

BUND = '/media/ucla/work/ntof_pass1/bench_bundles'


def bow_maps(arm):
    b = json.load(open(f'{BUND}/{arm}/bundle.json'))['bow']
    X, Y = np.meshgrid(G.CENTRES, G.CENTRES, indexing='ij')       # [ix, iy] = (x_local, y_local)
    U = (-(X) - 0.71) / 100.0          # strip_x = 199.29 - x_local ;  U = (strip_x - 200)/100
    W = (Y - 0.71) / 100.0             # strip_y = 199.29 + y_local
    out = {}
    for p in 'xy':
        a = R._bow_poly(b[f'a_{p}'], U, W)
        sc = b.get(f's_{p}') or {}
        s = sc.get('su', 0.0) * U + sc.get('sw', 0.0) * W
        out[p] = dict(a=a, s=s, U=U, W=W, block=b[f'a_{p}'], sblock=sc)
    return out


def plane(Z, w, X, Y):
    ok = np.isfinite(Z) & (w > 0)
    A = np.c_[np.ones(ok.sum()), X[ok], Y[ok]]
    W_ = np.sqrt(w[ok])[:, None]
    coef = np.linalg.lstsq(A * W_, Z[ok] * W_[:, 0], rcond=None)[0]
    return coef          # c, d/dx_local, d/dy_local (per mm)


def boot_slope(u, v, w=None, nb=2000, seed=3):
    ok = np.isfinite(u) & np.isfinite(v)
    u, v = u[ok], v[ok]
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(nb):
        q = rng.integers(0, len(u), len(u))
        if np.std(u[q]) > 0:
            b.append(np.polyfit(u[q], v[q], 1)[0])
    return float(np.polyfit(u, v, 1)[0]), float(np.std(b))


def bow_level():
    X, Y = np.meshgrid(G.CENTRES, G.CENTRES, indexing='ij')
    rows = []
    for arm in 'AC':
        bm = bow_maps(arm)
        for p in 'xy':
            sg = G.SGN[(arm, p)]
            res = {}
            for which in ('r2', 'r1'):
                nt = G.ntof_map(arm, p, which)
                good = nt['n'] >= G.N_MIN_NTOF
                at = np.where(good, nt['a'], np.nan)
                st = np.where(good, nt['s'], np.nan)
                abow = sg * bm[p]['a']
                sbow = bm[p]['s']
                # plane fits (weights = number of lines)
                pa = plane(at, nt['n'].astype(float), X, Y)
                pb = plane(abow, good.astype(float), X, Y)
                f_a, e_a = boot_slope(abow[good], at[good])
                f_s, e_s = boot_slope(sbow[good], st[good])
                # the same regression with the bench's own residual map, and the n_TOF - bench difference (isolates what changed with the move)
                Bm = pickle.load(open(f'{G.OUT}/bench_maps.pkl', 'rb'))[0.0][arm]['all'][p]
                ab_res = sg * Bm['a']
                both = good & np.isfinite(ab_res)
                f_b, e_b = boot_slope(abow[both], ab_res[both])
                f_d, e_d = boot_slope(abow[both], (at - ab_res)[both])
                sb_res = Bm['s']
                sboth = good & np.isfinite(sb_res)
                fs_b, es_b = boot_slope(sbow[sboth], sb_res[sboth])
                fs_d, es_d = boot_slope(sbow[sboth], (st - sb_res)[sboth])
                # gradient ratio over the SAME cells
                rows.append(dict(arm=arm, view=p, map=which, grad_resid_x=pa[1] * 100, grad_resid_y=pa[2] * 100,
                                 grad_bow_x=pb[1] * 100, grad_bow_y=pb[2] * 100,
                                 ratio_x=pa[1] / pb[1] if abs(pb[1]) > 1e-9 else np.nan, ratio_y=pa[2] / pb[2] if abs(pb[2]) > 1e-9 else np.nan,
                                 fminus1_a=f_a, fminus1_a_err=e_a, fminus1_s=f_s, fminus1_s_err=e_s, bench_fm1_a=f_b, bench_fm1_a_err=e_b, diff_fm1_a=f_d, diff_fm1_a_err=e_d, bench_fm1_s=fs_b, bench_fm1_s_err=es_b, diff_fm1_s=fs_d, diff_fm1_s_err=es_d, n_both=int(both.sum()),
                                 rms_bow_a=float(np.sqrt(np.nanmean(abow[good] ** 2))), rms_resid_a=float(np.sqrt(np.nanmean(at ** 2))),
                                 corr_a=G.pearson(at, abow), corr_s=G.pearson(st, sbow),
                                 cu=bm[p]['block'].get('cu'), cw=bm[p]['block'].get('cw'), auu=bm[p]['block'].get('auu'),
                                 aww=bm[p]['block'].get('aww'), auw=bm[p]['block'].get('auw'), su=bm[p]['sblock'].get('su'), sw=bm[p]['sblock'].get('sw')))
    T = pd.DataFrame(rows)
    T.to_csv(f'{G.OUT}/bow_level.csv', index=False)
    pd.set_option('display.width', 250, 'display.max_columns', 40)
    print(T.round(4).to_string())
    return T


def flips2(M, fx, fy, dx=0, dy=0):
    M = M[::-1, :] if fx else M
    M = M[:, ::-1] if fy else M
    if dx:
        M = np.roll(M, dx, axis=0)
    if dy:
        M = np.roll(M, dy, axis=1)
    return M


def scan(margin=0.0):
    """held-out sigma of the 'bench_1' model (trained global + unit-amplitude bench shape) for the 8 sign/mirror combinations and for
    shifts of the bench shape by whole cells along x_local / y_local.  Mirrors are applied to the bench map as a whole (a and s)."""
    M = pd.read_parquet(GD.LINES)
    rn = M.run.str.replace('run_', '').astype(int)
    rows = []
    for s, arm in (('a', 'A'), ('c', 'C')):
        for p in 'xy':
            Gc = AM.clean_lines(M, s, p)
            parc = Gc.run.str.replace('run_', '').astype(int) % 2
            a_b, s_b, a_sh, s_sh, _, _ = GD.shape(arm, p, margin)
            derived = G.SGN[(arm, p)]
            for tr in (0, 1):
                Gt = Gc[parc == tr]
                te = (rn % 2) != tr
                Gs = Gc[parc != tr]
                slope_c = Gs.r.to_numpy() + Gs.j.to_numpy()
                a_g, s_g, _ = AM.robust_line(Gt.j.to_numpy(), Gt.r.to_numpy())

                def sig(a_map, s_map):
                    mp = dict(a=a_g + a_map, s=s_g + s_map, centres=G.CENTRES, a0=a_g, s0=s_g)
                    cc = AM.correct(slope_c, Gs.x.to_numpy(), Gs.y.to_numpy(), mp)
                    return G.mad(cc - Gs.j.to_numpy())
                base = sig(0 * a_sh, 0 * s_sh)
                for fx, fy, fa in itertools.product((0, 1), repeat=3):
                    rows.append(dict(arm=arm, view=p, train=tr, kind='mirror', key=f'fx{fx}fy{fy}fa{fa}', dx=0, dy=0,
                                     derived=int(fx == 0 and fy == 0 and fa == 0),
                                     sig=sig(flips2(a_sh, fx, fy) * (-1 if fa else 1), flips2(s_sh, fx, fy)), base=base))
                for dx, dy in [(i, k) for i in range(-3, 4) for k in range(-3, 4)]:
                    # shifts without wrap: zero the wrapped part
                    def sh(Z):
                        Zs = np.roll(np.roll(Z, dx, 0), dy, 1)
                        if dx > 0: Zs[:dx, :] = 0
                        if dx < 0: Zs[dx:, :] = 0
                        if dy > 0: Zs[:, :dy] = 0
                        if dy < 0: Zs[:, dy:] = 0
                        return Zs
                    rows.append(dict(arm=arm, view=p, train=tr, kind='shift', key=f'{dx},{dy}', dx=dx, dy=dy, derived=int(dx == 0 and dy == 0),
                                     sig=sig(sh(a_sh), sh(s_sh)), base=base))
    T = pd.DataFrame(rows)
    T.to_csv(f'{G.OUT}/scan_m{margin:.0f}.csv', index=False)
    g = T[T.kind == 'mirror'].groupby(['arm', 'view', 'key']).agg(sig=('sig', 'mean'), base=('base', 'mean'), derived=('derived', 'max')).reset_index()
    g['gain'] = 1 - g.sig / g.base
    for (arm, p), q in g.groupby(['arm', 'view']):
        q = q.sort_values('gain', ascending=False)
        print(arm, p, 'mirror combos by held-out gain (derived marked *):')
        print(q.head(4).round(4).to_string(index=False), '\n   derived:', q[q.derived == 1].round(4).to_string(index=False, header=False))
    sh_ = T[(T.kind == 'shift')].groupby(['arm', 'view', 'dx', 'dy']).agg(sig=('sig', 'mean'), base=('base', 'mean')).reset_index()
    sh_['gain'] = 1 - sh_.sig / sh_.base
    for (arm, p), q in sh_.groupby(['arm', 'view']):
        b = q.sort_values('gain', ascending=False).iloc[0]
        z = q[(q.dx == 0) & (q.dy == 0)].iloc[0]
        print(arm, p, f'shift scan: best ({int(b.dx)},{int(b.dy)}) cells gain {b.gain:+.3f}; unshifted gain {z.gain:+.3f}')
    return T


if __name__ == '__main__':
    bow_level()
    scan(0.0)
