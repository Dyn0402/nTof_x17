"""r1_lib2: field evaluation from the stored spline coefficients, event-level tilt prediction, event-level (a, b) fit."""
from r1_lib import *
K, LAM = 12, 0.1
SPL = Spl(K)
FLD = pickle.load(open(f'{OUT}/fields.pkl', 'rb'))


class Fld:
    """gap field from stored coefficients (central or bootstrap replicate)"""
    def __init__(self, name, rep=None, beta=None):
        f = FLD[name]
        self.c = f['c'] if rep is None else f['cb'][rep]
        self.beta = f['beta'] if beta is None else beta
        self.fac = G0 / (1 + self.beta)

    def dg(self, U, W):
        return self.fac * (SPL.basis(U, W) @ self.c)

    def grad(self, U, W):
        return self.fac * (SPL.basis(U, W, 1, 0) @ self.c) / 100, self.fac * (SPL.basis(U, W, 0, 1) @ self.c) / 100

    def vrel(self, U, W):
        g = G0 + self.dg(U, W)
        return (g / g.mean()) ** (-self.beta)


def regressors(v, p, field, hw='h', bundle=None):
    """per-event mesh and gap tilt regressors (dimensionless): gh = dh/dx_p, gg = d dg/dx_p"""
    i = 0 if p == 'x' else 1
    gh = hgrad(v['U'], v['W'], hw, bundle)[i]
    gg = field.grad(v['U'], v['W'])[i]
    return gh, gg


def plane_removed(U, W, x):
    X = np.vstack([np.ones(len(U)), U, W]).T
    c = np.linalg.lstsq(X, x, rcond=None)[0]
    return x - X @ c


def abfit(v, gh, gg, fine=False, clip=3.5):
    """event-level robust LS of  w = c0 + c1 tr + c1 (a gh + b gg)  [+ free plane c4 U + c5 W when fine=True, with the
    plane removed from the regressors].  Returns dict(a, b, v, c0)."""
    ghf, ggf = (plane_removed(v['U'], v['W'], gh), plane_removed(v['U'], v['W'], gg)) if fine else (gh, gg)
    cols = [np.ones(len(gh)), v['tr'], ghf, ggf]
    if fine:
        cols += [v['U'], v['W']]
    X = np.vstack(cols).T
    c, keep, rss, rs = CL.lsq_clip(X, v['w'], clip)
    return dict(a=c[2] / c[1], b=c[3] / c[1], v=c[1], c0=c[0])
