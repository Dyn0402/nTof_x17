"""c_lib: loader for the pass1_all layout, with GM.load_table's selection (audit C)."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap')
import gapmap as GM, cellobs as CO, track_test as TT
WORK = '/media/ucla/work/claude-jobs/e1a7a26e/gap'
OUT = f'{WORK}/audit/c'
ROWS = {'run1': 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V',
        'run2': 'mx17_det3_p2_det1_overnight_6-27-26/long_run_p2_det1_sanity_check',
        'r625': 'mx17_det3_day_6-25-26/long_run'}
UL, WL = (-1.7, 1.7), (-1.51, 1.51)

def load(rel):
    d = pd.read_parquet(f'{WORK}/pass1_all/{rel}/mx17_3/bow_gate_table.parquet')
    d = d[(d.n_tracks == 1) & d.x_ok & d.y_ok & d.x_quality_ok & d.y_quality_ok & ~d.spark
          & (d.Chi2X < 1) & (d.Chi2Y < 1) & (d.NClusX >= 4) & (d.NClusY >= 4)].copy()
    d['xm'] = d.ref_mesh_x + GM.MID * d.tan_ref_x
    d['ym'] = d.ref_mesh_y + GM.MID * d.tan_ref_y
    d = d[(d.xm > GM.IN_X[0]) & (d.xm < GM.IN_X[1]) & (d.ym > GM.IN_Y[0]) & (d.ym < GM.IN_Y[1])]
    d = d.reset_index(drop=True)
    d['U'] = (d.x_p0 + 14 * d.x_tan_theta - 200) / 100   # reconstructed mid-gap position (as track_grid.py)
    d['W'] = (d.y_p0 + 14 * d.y_tan_theta - 200) / 100
    return d

def view(d, p):
    """arrays for |tan_ref|<=0.6"""
    s = (np.abs(d[f'tan_ref_{p}'].values) <= GM.TAN_LADDER)
    return dict(U=d.U.values[s], W=d.W.values[s], w=d[f'{p}_w'].values[s] * 1e3, tr=d[f'tan_ref_{p}'].values[s],
                tp=d[f'{p}_tan_theta'].values[s], idx=np.where(s)[0])

# ---------------------------------------------------------------- cell maps
def cellmap(U, W, w, tr, n):
    eu, ew = np.linspace(*UL, n + 1), np.linspace(*WL, n + 1)
    i, j = np.searchsorted(eu, U) - 1, np.searchsorted(ew, W) - 1
    M = {k: np.full((n, n), np.nan) for k in ('Uc', 'Wc', 'w0', 'v', 'w0e', 've', 'N')}
    for a in range(n):
        for b in range(n):
            m = (i == a) & (j == b)
            w0, v, w0e, ve, nn = CO.line(tr[m], w[m])
            for k, x in zip(('Uc', 'Wc', 'w0', 'v', 'w0e', 've', 'N'),
                            (0.5 * (eu[a] + eu[a + 1]), 0.5 * (ew[b] + ew[b + 1]), w0, v, w0e, ve, nn)):
                M[k][a, b] = x
    return M


def plane_fit(M, A, Ae):
    """weighted plane a + b U + c W of A over valid cells; floor on the error = half the median error."""
    ok = np.isfinite(A) & np.isfinite(Ae)
    fl = 0.5 * np.median(Ae[ok])
    wt = 1 / (Ae[ok] ** 2 + fl ** 2)
    X = np.vstack([np.ones(ok.sum()), M['Uc'][ok], M['Wc'][ok]]).T
    c = np.linalg.solve((X * wt[:, None]).T @ X, (X * wt[:, None]).T @ A[ok])
    return c


class Maps:
    """maps of one dataset (events of one view): A (tan units), S, plane + fine split."""
    def __init__(self, U, W, w, tr, n):
        self.n = n
        self.M = M = cellmap(U, W, w, tr, n)
        self.vg = CO.line(tr, w)[1]
        self.A = (M['w0'] - np.nanmean(M['w0'])) / self.vg
        self.Ae = M['w0e'] / self.vg
        self.S = M['v'] / np.nanmean(M['v']) - 1
        self.Se = M['ve'] / np.nanmean(M['v'])
        self.pc = plane_fit(M, self.A, self.Ae)
        self.Afine = self.A - (self.pc[0] + self.pc[1] * M['Uc'] + self.pc[2] * M['Wc'])
        self.Aplane = self.A - self.Afine

    def _cells(self, key):
        M = self.M
        out = []
        for a in range(self.n):
            for b in range(self.n):
                out.append(dict(U=M['Uc'][a, b], W=M['Wc'][a, b], k=getattr(self, key)[a, b]))
        return out

    def ev(self, U, W):
        """(A_plane, A_fine, S) at tracks; plane analytic, fine and S bilinear (nan cells -> mean)."""
        ap = self.pc[1] * U + self.pc[2] * W
        af = TT.bilinear(self._cells('Afine'), 'k', U, W)
        s = TT.bilinear(self._cells('S'), 'k', U, W)
        return ap, af, s, TT.bilinear(self._cells('A'), 'k', U, W)


def lsq_clip(X, y, clip=3.5, passes=3):
    keep = np.ones(len(y), bool)
    for _ in range(passes):
        c = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        r = y - X @ c
        rs = 1.4826 * np.median(np.abs(r[keep]))
        keep = np.abs(r) < clip * rs
    c = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
    r = y - X @ c
    return c, keep, float(np.sum(r[keep] ** 2)), rs


def kfit(V, mp, boot=0, seed=1, keep_cols=('s', 'p', 'f')):
    """1e3 w = W0 + v' [tr (1 + ks S) + kp Ap + kf Af]: linear in (W0, a, a ks, a kp, a kf)."""
    ap, af, s, _ = mp.ev(V['U'], V['W'])
    X = np.vstack([np.ones(len(ap)), V['tr'], V['tr'] * s, ap, af]).T
    y = V['w']

    def one(idx):
        c, keep, rss, rs = lsq_clip(X[idx], y[idx])
        return np.array([c[1], c[2] / c[1], c[3] / c[1], c[4] / c[1], c[0]]), rss, int(keep.sum())
    c0, rss, nk = one(np.arange(len(y)))
    res = dict(vp=c0[0], ks=c0[1], kp=c0[2], kf=c0[3], W0=c0[4], rss=rss, nkeep=nk, n=len(y))
    if boot:
        rng = np.random.default_rng(seed)
        B = np.array([one(rng.integers(0, len(y), len(y)))[0] for _ in range(boot)])
        for k, col in zip(('vp', 'ks', 'kp', 'kf'), range(4)):
            res[k + '_e'] = float(B[:, col].std())
    return res
