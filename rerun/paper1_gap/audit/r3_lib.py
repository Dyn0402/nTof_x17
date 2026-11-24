"""r3_lib.py -- audit R3: per-mounting angle maps on all 14 gate rows.  Reuses a2_lib (spline maps), r1_lib (gap field, mesh
gradient), d_lib (row loader).  Everything per row is expressed in tan units: w_n = (w - c0_row)/c1_row (robust ladder constants
of the row), so maps and the reference-free tilt are in relative units and shared between rows of a mounting.
"""
import sys, json, time
import numpy as np
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
import d_lib as DL
import a2_lib as A2
import r1_lib as R1
from a2_lib import Data, Spec, fit_map, eval_map, s68, UD, WD
import track_test as TT
import c_lib as CL

OUT = '/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/r3'
import os
os.makedirs(OUT, exist_ok=True)

MOUNT = {
    'B':    dict(chamber='B (det2)', rows=['det2:long', 'det2:longer', 'det2:short'], qsrc='xy'),
    'A625': dict(chamber='A (det3)', rows=['det3:d625'], qsrc='xy'),
    'A628': dict(chamber='A (det3)', rows=['det3:sat', 'det3:p2'], qsrc='xy'),
    'E623': dict(chamber='E (det4)', rows=['det4:623'], qsrc='xy'),
    'E624': dict(chamber='E (det4)', rows=['det4:624'], qsrc='xy'),
    'C':    dict(chamber='C (det6)', rows=['det6:long', 'det6:longer', 'det6:short'], qsrc='xy'),
    'D':    dict(chamber='D (det7)', rows=['det7:long', 'det7:longer', 'det7:short'], qsrc='y'),
    'Dx200': dict(chamber='D (det7), x>=200 mm only', rows=['det7:long', 'det7:longer', 'det7:short'], qsrc='xy', xhalf=True),
}
BETA0 = 0.68
G0 = R1.G0


def qamp_row(d, src):
    """ln-ready q_u90 with the amplitude walk removed (one global slope against ln q_sum), as r1_lib.qwalk, optionally y only."""
    if src == 'xy':
        q = 0.5 * (d.x_q_u90.values + d.y_q_u90.values); lq = np.log(0.5 * (d.x_q_sum.values + d.y_q_sum.values))
    else:
        q = d.y_q_u90.values.copy(); lq = np.log(d.y_q_sum.values)
    ok = np.isfinite(q) & np.isfinite(lq) & (q > 0)
    keep = ok.copy()
    for _ in range(3):
        A = np.vstack([np.ones(keep.sum()), lq[keep] - np.median(lq[ok])]).T
        c = np.linalg.lstsq(A, q[keep], rcond=None)[0]
        r = q - c[0] - c[1] * (lq - np.median(lq[ok]))
        rs = 1.4826 * np.median(np.abs(r[keep]))
        keep = ok & (np.abs(r) < 4 * rs)
    qa = q - c[1] * (lq - np.median(lq[ok]))
    return qa, ok & keep & (qa > 0)


def load_mounting(name):
    M = MOUNT[name]
    parts = []
    info = []
    for ri, key in enumerate(M['rows']):
        d = DL.load(key, xhalf=M.get('xhalf', False))
        # ---- normalisation per row
        wx, wy = d.x_w.values * 1e3, d.y_w.values * 1e3
        trx, tr_y = d.tan_ref_x.values, d.tan_ref_y.values
        cx, cy = TT.robust_const(trx, wx, 0.0, 1.0), TT.robust_const(tr_y, wy, 0.0, 1.0)
        qa, qok = qamp_row(d, M['qsrc'])
        lnq = np.full(len(d), np.nan); lnq[qok] = np.log(qa[qok]) - np.median(np.log(qa[qok]))
        order = np.lexsort((d.timestamp.values.astype(float), d.acq_tag.values))
        parts.append(dict(
            U=d.Urec.values, W=d.Wrec.values, wx=(wx - cx[0]) / cx[1], wy=(wy - cy[0]) / cy[1], trx=trx, trY=tr_y,
            tpx=d.x_tan_theta.values, tpy=d.y_tan_theta.values, row=np.full(len(d), ri),
            xm=d.xm.values, ym=d.ym.values, lnq=lnq,
            satx=d.x_n_sat_strips.values, saty=d.y_n_sat_strips.values, acq=d.acq_tag.values.astype(str),
            ts=d.timestamp.values.astype(float), p0x=d.x_p0.values, p0y=d.y_p0.values))
        al = DL.align(key); b = DL.bundle(key)
        info.append(dict(key=key, N=len(d), v_x=float(cx[1]), v_y=float(cy[1]), w0_x=float(cx[0]), w0_y=float(cy[0]),
                         z_x=al['z_x'], z_y=al['z_y'], theta=al['theta_deg'], x_off=al['x_offset'], y_off=al['y_offset'],
                         sat_x=float((d.x_n_sat_strips > 0).mean()), sat_y=float((d.y_n_sat_strips > 0).mean()),
                         sat_any=float(((d.x_n_sat_strips > 0) | (d.y_n_sat_strips > 0)).mean()),
                         first_acq=str(np.min(d.acq_tag.values)), volt=DL.VOLT[key]))
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    # global time rank: acq tag (chronological string) then timestamp
    o = np.lexsort((cat['ts'], cat['acq'])); rank = np.empty(len(o), int); rank[o] = np.arange(len(o))
    n = len(cat['U'])
    D = Data(U=cat['U'], W=cat['W'], wx=cat['wx'], wy=cat['wy'], trx=cat['trx'], trY=cat['trY'], tpx=cat['tpx'], tpy=cat['tpy'],
             rank=rank, row=cat['row'], xm=cat['xm'], ym=cat['ym'], lnq=cat['lnq'], satx=cat['satx'], saty=cat['saty'],
             p0x=cat['p0x'], p0y=cat['p0y'],
             bax=np.zeros(n), bay=np.zeros(n), bsx=np.ones(n), bsy=np.ones(n), n=n)
    bundle = DL.bundle(M['rows'][0])['bow']
    return D, info, bundle


# ---------------------------------------------------------------- gap field and mesh gradient (reference-free)
def gap_field(D, K=None, lam=0.1, only=None):
    """P-spline of delta ln q_u90 on the mid-gap position of the events (reference-free).  -> (Spl, coef, ok mask)."""
    ok = np.isfinite(D.lnq)
    if only is not None:
        ok &= only
    K = K or (12 if ok.sum() >= 4000 else 8)
    spl = R1.Spl(K)
    c, keep = spl.fit(D.U[ok], D.W[ok], D.lnq[ok] - np.mean(D.lnq[ok]), lam)
    return spl, c, ok


def attach_field(D, bundle, spl, c):
    """per-event: lf (delta ln q field), glx, gly (d lf / dx, dy in 1/mm), ghx, ghy (per-view mesh gradient, tan)."""
    D.lf = spl.basis(D.U, D.W) @ c
    D.glx = (spl.basis(D.U, D.W, 1, 0) @ c) / 100
    D.gly = (spl.basis(D.U, D.W, 0, 1) @ c) / 100
    D.ghx = R1.hgrad(D.U, D.W, 'perview', bundle)[0]
    D.ghy = R1.hgrad(D.U, D.W, 'perview', bundle)[1]


def rf_base(D, b, beta, vel):
    """reference-free base: tilt = gh + b * G0/(1+beta) * grad lf ; vrel = (g/<g>)^(-beta) with g = G0 (1 + lf/(1+beta))."""
    fac = G0 / (1 + beta)
    ax = D.ghx + b * fac * D.glx
    ay = D.ghy + b * fac * D.gly
    if vel:
        g = G0 + fac * D.lf
        vr = (g / g.mean()) ** (-beta)
    else:
        vr = np.ones(D.n)
    return ax, ay, vr, vr


# ---------------------------------------------------------------- generic fit / predict with a base model
def with_base(D, ax, ay, sx, sy):
    D2 = Data(**{k: v for k, v in D.__dict__.items() if k != 'n'}, n=D.n)
    D2.bax, D2.bay, D2.bsx, D2.bsy = ax, ay, sx, sy
    return D2


def fit_generic(spec, D, lam):
    """remainder map on top of the base: target w - bax, regressor tr * bsx."""
    D2 = Data(**{k: v for k, v in D.__dict__.items() if k != 'n'}, n=D.n)
    D2.wx = D.wx - D.bax; D2.wy = D.wy - D.bay
    D2.trx = D.trx * D.bsx; D2.trY = D.trY * D.bsy
    return fit_map(spec, D2, lam)


def tans(D, m):
    """predicted tan per event: constants refit per row; a = bax + a', s = bsx * s'."""
    if m is None:
        ax, ay, sx, sy = D.bax, D.bay, D.bsx, D.bsy
    else:
        a1, a2, s1, s2 = eval_map(m, D.U, D.W)
        ax, ay, sx, sy = D.bax + a1, D.bay + a2, D.bsx * s1, D.bsy * s2
    tx, ty = np.zeros(D.n), np.zeros(D.n)
    for r in np.unique(D.row):
        i = D.row == r
        for (a, s, w, tr, out) in ((ax, sx, D.wx, D.trx, tx), (ay, sy, D.wy, D.trY, ty)):
            dw = a[i] - np.mean(a[i]); vr = s[i] / np.mean(s[i])
            c = TT.robust_const(tr[i], w[i], dw, vr)
            out[i] = (w[i] - c[0] - dw) / (c[1] * vr)
    return tx, ty


def dth(t, tr):
    return np.degrees(np.arctan(t) - np.arctan(tr))


def cv_inner(spec, D, lam, nf=4):
    f = np.minimum((D.rank.argsort().argsort() * nf) // D.n, nf - 1)
    ex, ey = [], []
    for k in range(nf):
        tr_, te = D.sub(f != k), D.sub(f == k)
        m = fit_map(spec, tr_, lam) if False else fit_generic(spec, tr_, lam)
        tx, ty = tans(te, m)
        ex.append(dth(tx, te.trx)); ey.append(dth(ty, te.trY))
    return s68(np.concatenate(ex)), s68(np.concatenate(ey))


KGRID = (6, 9, 12, 16)
LAMS = (0.01, 0.1, 1.0, 10.0)


def select(D, variant, kgrid=KGRID, ksgrid=(0, 4, 8), lams=LAMS, nf=4):
    """inner CV on D (training part only).  variant 'S': phi + scale maps; 'T': phi only (S = const).  Restrict the number of
    coefficients to N/15.  -> (K, KS, lam, cv score)"""
    best = None
    ks = ksgrid if variant == 'S' else (0,)
    ok = [(K, KS) for K in kgrid for KS in ks if K * K + 2 * KS * KS <= D.n / 15.0]
    if not ok:
        ok = [(5, 0)]
    for K, KS in ok:
        spec = Spec('phi', K, KS, 1.0)
        for lam in lams:
            sx, sy = cv_inner(spec, D, lam, nf)
            if best is None or sx + sy < best[3]:
                best = (K, KS, lam, sx + sy)
    return best


def fit_b(D, rows=None, clip=3.5):
    """event-level free b (a = 1): w_n - tr - gh = c + b * fac1 * grad lf, both views stacked; fac1 = G0/(1+0.68) (b is then
    quoted at beta = 0.68, i.e. the amplitude of the gap tilt relative to the nominal one).  -> b, per view b."""
    fac = G0 / (1 + BETA0)
    out = {}
    for p in 'xy':
        w = D.wx if p == 'x' else D.wy; tr = D.trx if p == 'x' else D.trY
        gh = D.ghx if p == 'x' else D.ghy; gl = D.glx if p == 'x' else D.gly
        y = w - tr - gh
        X = np.vstack([np.ones(D.n), fac * gl]).T
        c = CL.lsq_clip(X, y, clip)[0]
        out[p] = float(c[1])
    # both views together
    y = np.concatenate([D.wx - D.trx - D.ghx, D.wy - D.trY - D.ghy])
    X = np.zeros((2 * D.n, 3)); X[:D.n, 0] = 1; X[D.n:, 1] = 1; X[:D.n, 2] = fac * D.glx; X[D.n:, 2] = fac * D.gly
    c = CL.lsq_clip(X, y, clip)[0]
    out['xy'] = float(c[2])
    return out


def paired_boot(darr, ref='pass1', nb=400, seed=3):
    rng = np.random.default_rng(seed)
    names = list(darr); n = len(darr[ref]); M = np.vstack([darr[k] for k in names])
    S = np.empty((nb, len(names)))
    for b in range(nb):
        i = rng.integers(0, n, n)
        q = np.percentile(M[:, i], [16, 84], axis=1)
        S[b] = (q[1] - q[0]) / 2
    j = names.index(ref)
    return ({k: (s68(darr[k]), float(S[:, i].std())) for i, k in enumerate(names)},
            {k: (s68(darr[k]) - s68(darr[ref]), float((S[:, i] - S[:, j]).std())) for i, k in enumerate(names)})
