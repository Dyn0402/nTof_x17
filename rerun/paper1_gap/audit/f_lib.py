"""f_lib: vectorised, extensible re-implementation of joint.Model (audit F). Reproduces joint.fit for the default options."""
import json, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap')
import joint as JT

WORK = '/media/ucla/work/claude-jobs/e1a7a26e/gap'
OBS = json.load(open(f'{WORK}/cellobs.json'))
FLOORS = json.load(open('/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/results/joint.json'))['floors']
COMMITTED = json.load(open('/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/results/joint.json'))['arms']
CLASSES = JT.CLASSES
ARMS = JT.ARMS
KINDS = ['hp_x', 'hp_y', 'w0_x', 'w0_y', 'dX', 'dY', 'G', 'GY', 'T', 'v_x', 'v_y']
KCLS = {'hp_x': 'hp', 'hp_y': 'hp', 'w0_x': 'w0', 'w0_y': 'w0', 'dX': 'd', 'dY': 'd', 'G': 'G', 'GY': 'GY', 'T': 'T', 'v_x': 'v', 'v_y': 'v'}
SURF = JT.SURF
# extra basis terms: name -> (f(U,W), dfdU, dfdW)
EXTRA = {
 'u3': (lambda U, W: U**3, lambda U, W: 3*U**2, lambda U, W: 0*U),
 'u2w': (lambda U, W: U**2*W, lambda U, W: 2*U*W, lambda U, W: U**2),
 'uw2': (lambda U, W: U*W**2, lambda U, W: W**2, lambda U, W: 2*U*W),
 'w3': (lambda U, W: W**3, lambda U, W: 0*U, lambda U, W: 3*W**2),
 'u4': (lambda U, W: U**4, lambda U, W: 4*U**3, lambda U, W: 0*U),
 'w4': (lambda U, W: W**4, lambda U, W: 0*U, lambda U, W: 4*W**3),
 'r4': (lambda U, W: (U*U+W*W)**2, lambda U, W: 4*U*(U*U+W*W), lambda U, W: 4*W*(U*U+W*W)),
}
CUBIC = ['u3', 'u2w', 'uw2', 'w3']
QUARTIC_SEP = ['u4', 'w4']
QUARTIC_RAD = ['r4']
V_DRIFT, G_REF = JT.V_DRIFT, JT.G_REF


class Fitter:
    """extra_h / extra_c: lists of EXTRA names. a: None (=0.5 fixed), 'free', 'free_xy', or float. tie: dict free-name -> {theta-name: coef}."""
    def __init__(self, classes, floors, cells=None, weights=None, extra_h=(), extra_c=(), a=0.5, beta=None, m3=False,
                 tie=None, zero=(), runs=None, cov_pair=None, cellfilter=None, obs=None):
        self.obs = obs if obs is not None else OBS
        self.classes, self.floors = classes, dict(floors)
        self.extra_h, self.extra_c = list(extra_h), list(extra_c)
        self.runs = runs or sorted(self.obs)
        names = ['h_' + s for s in SURF] + ['c_' + s for s in SURF] + ['h_' + e for e in self.extra_h] + ['c_' + e for e in self.extra_c]
        names += ['g0', 'beta', 'oGY', 'ax', 'ay']
        for r in self.runs:
            names += [f'ox_{r}', f'oy_{r}', f'Wx_{r}', f'Wy_{r}', f'lnT_{r}', f'lnvx_{r}', f'lnvy_{r}']
        self.names = names
        self.a, self.beta_fixed, self.zero, self.tie = a, beta, set(zero), tie or {}
        # observation arrays
        rows = []
        for ri, r in enumerate(self.runs):
            for ci, o in enumerate(self.obs[r]):
                if cells is not None and (o['i'], o['j']) not in cells:
                    continue
                if cellfilter is not None and not cellfilter(o):
                    continue
                for ki, k in enumerate(KINDS):
                    cls = KCLS[k]
                    if cls not in classes:
                        continue
                    y, e = o.get(k), o.get(k + '_err')
                    if y is None or not np.isfinite(y) or not np.isfinite(e) or e <= 0:
                        continue
                    if cls in ('T', 'v'):
                        y, e = np.log(y), e / y
                    w = 1.0 if weights is None else weights.get((o['i'], o['j']), 1.0)
                    if w == 0:
                        continue
                    rows.append((ri, ki, o['U'], o['W'], y, e, w, o['i'], o['j']))
        A = np.array(rows) if rows else np.zeros((0, 9))
        self.ri, self.ki = A[:, 0].astype(int), A[:, 1].astype(int)
        self.U, self.W, self.y, self.e0, self.w = A[:, 2], A[:, 3], A[:, 4], A[:, 5], A[:, 6]
        self.cellij = A[:, 7:9].astype(int)
        self.cls = np.array([KCLS[KINDS[k]] for k in self.ki])
        fl = np.array([self.floors.get(c, 0.0) for c in self.cls])
        self.e = np.hypot(self.e0, fl)
        self.idx = {n: i for i, n in enumerate(names)}
        self.cov_pair = cov_pair  # dict (run,i,j,view)->cov(w0,v) in raw units
        self.pairs = []
        if cov_pair:
            key = {}
            for n in range(len(self.y)):
                key[(self.ri[n], self.cellij[n, 0], self.cellij[n, 1], KINDS[self.ki[n]])] = n
            for (r, i, j, view), c in cov_pair.items():
                if r not in self.runs: continue
                ri = self.runs.index(r)
                a, b = key.get((ri, i, j, 'w0_' + view)), key.get((ri, i, j, 'v_' + view))
                if a is None or b is None: continue
                vv = np.exp(self.y[b])
                ew, ev = self.e[a], self.e[b]
                cc = c / vv * np.sqrt(self.e0[a]**2 / 1.0) * 0  # placeholder
                cw = c / vv  # cov(w0, ln v)
                M = np.array([[ew**2, cw], [cw, ev**2]])
                L = np.linalg.cholesky(M)
                self.pairs.append((a, b, np.linalg.inv(L)))
        self._free()

    def _free(self):
        used = set()
        for c in self.classes:
            used |= {'hp': {'ox', 'oy'}, 'w0': {'Wx', 'Wy'}, 'GY': {'oGY'}, 'T': {'lnT'}, 'v': {'lnvx', 'lnvy'}}.get(c, set())
        grp = {'ox', 'oy', 'Wx', 'Wy', 'oGY', 'lnT', 'lnvx', 'lnvy'}
        free = [n for n in self.names if n.split('_')[0] not in grp or n.split('_')[0] in used]
        if not ({'T', 'v'} & set(self.classes)) or self.beta_fixed is not None:
            free.remove('beta') if 'beta' in free else None
        if not ({'G', 'GY', 'T', 'v'} & set(self.classes)):
            free.remove('g0') if 'g0' in free else None
        # a: free only if w0 used
        if self.a == 'free':
            free.remove('ay'); 
        elif self.a == 'free_xy':
            pass
        else:
            free = [n for n in free if n not in ('ax', 'ay')]
        if 'w0' not in self.classes:
            free = [n for n in free if n not in ('ax', 'ay')]
        free = [n for n in free if n not in self.zero]
        self.free = free
        # theta: tie handling
        theta = [n for n in free if n not in self.tie]
        extra_theta = []
        for n, ex in self.tie.items():
            for t in ex:
                if t not in theta and t not in extra_theta and t not in free:
                    extra_theta.append(t)
        self.theta = theta + extra_theta
        self.theta_names_free = free

    def base(self):
        P = dict.fromkeys(self.names, 0.0)
        P['g0'] = 28.2; P['beta'] = 0.68 if self.beta_fixed is None else self.beta_fixed
        P['ax'] = P['ay'] = 0.5 if isinstance(self.a, str) else self.a
        for r in self.runs:
            Ts = [o['T'] for o in self.obs[r] if np.isfinite(o['T'])]
            P[f'lnT_{r}'] = np.log(np.median(Ts))
            P[f'lnvx_{r}'] = np.log(np.nanmedian([o['v_x'] for o in self.obs[r]]))
            P[f'lnvy_{r}'] = np.log(np.nanmedian([o['v_y'] for o in self.obs[r]]))
        return P

    def full(self, th):
        P = self.base()
        T = dict(zip(self.theta, th))
        for n in self.theta:
            P[n] = T[n]
        for n, ex in self.tie.items():
            P[n] = sum(c * (T[t] if t in T else P[t]) for t, c in ex.items())
        if self.a == 'free':
            P['ay'] = P['ax']
        return P

    def surf(self, P, pre, U, W):
        s = [P[pre + n] for n in SURF]
        f = s[0]*U + s[1]*W + s[2]*U*U/2 + s[3]*W*W/2 + s[4]*U*W
        du = s[0] + s[2]*U + s[4]*W
        dw = s[1] + s[3]*W + s[4]*U
        for e in (self.extra_h if pre == 'h_' else self.extra_c):
            F, dU, dW = EXTRA[e]
            c = P[pre + e]
            f = f + c*F(U, W); du = du + c*dU(U, W); dw = dw + c*dW(U, W)
        return f, du, dw

    def pred(self, P):
        U, W = self.U, self.W
        h, hu, hw = self.surf(P, 'h_', U, W)
        c, cu, cw = self.surf(P, 'c_', U, W)
        g = P['g0'] + c - h
        lg = np.log(np.maximum(g, 1.0) / G_REF)
        ri = self.ri
        def run(pref):
            return np.array([P[f'{pref}_{r}'] for r in self.runs])[ri]
        ax, ay = P['ax'], P['ay']
        tu, tw = (hu + cu) / 200.0, (hw + cw) / 200.0
        mu = V_DRIFT * ((1 - ax) * hu + ax * cu) / 100.0
        mw = V_DRIFT * ((1 - ay) * hw + ay * cw) / 100.0
        out = np.zeros(len(U))
        k = self.ki
        K = {n: i for i, n in enumerate(KINDS)}
        out = np.where(k == K['hp_x'], h + run('ox'), out)
        out = np.where(k == K['hp_y'], h + run('oy'), out)
        out = np.where(k == K['w0_x'], run('Wx') + mu, out)
        out = np.where(k == K['w0_y'], run('Wy') + mw, out)
        out = np.where(k == K['dX'], tu, out)
        out = np.where(k == K['dY'], tw, out)
        out = np.where(k == K['G'], g, out)
        out = np.where(k == K['GY'], g + P['oGY'], out)
        out = np.where(k == K['T'], run('lnT') + (1 + P['beta']) * lg, out)
        out = np.where(k == K['v_x'], run('lnvx') - P['beta'] * lg, out)
        out = np.where(k == K['v_y'], run('lnvy') - P['beta'] * lg, out)
        return out

    def resid(self, th):
        P = self.full(th)
        r = (self.y - self.pred(P)) / self.e
        for a, b, Li in self.pairs:
            ra, rb = r[a] * self.e[a], r[b] * self.e[b]
            z = Li @ np.array([ra, rb]); r[a], r[b] = z
        return np.sqrt(self.w) * r

    def fit(self):
        P0 = self.base()
        th0 = np.array([P0[n] for n in self.theta])
        r = least_squares(self.resid, th0, x_scale='jac')
        J = r.jac
        cov = np.linalg.pinv(J.T @ J)
        th = r.x
        P = self.full(th)
        res = self.resid(th)
        # jacobian of all named params wrt theta, to propagate covariance
        eps = 1e-6
        Pf = np.array([P[n] for n in self.names])
        D = np.zeros((len(self.names), len(th)))
        for j in range(len(th)):
            t2 = th.copy(); t2[j] += eps
            P2 = self.full(t2)
            D[:, j] = (np.array([P2[n] for n in self.names]) - Pf) / eps
        covP = D @ cov @ D.T
        per = {c: (float(np.sum(res[self.cls == c] ** 2)), int(np.sum(self.cls == c))) for c in self.classes}
        return Result(self.names, Pf, covP, float(np.sum(res**2)), len(res), len(th), per, self)


class Result:
    def __init__(self, names, p, cov, chi2, n, npar, per, fitter):
        self.names, self.p, self.cov, self.chi2, self.n, self.npar, self.per, self.fitter = names, p, cov, chi2, n, npar, per, fitter
        self.i = {n: k for k, n in enumerate(names)}
    def __getitem__(self, n): return self.p[self.i[n]]
    def err(self, n): return float(np.sqrt(max(self.cov[self.i[n], self.i[n]], 0)))
    def lin(self, comb):
        v = np.zeros(len(self.names))
        for n, c in comb.items(): v[self.i[n]] = c
        return float(v @ self.p), float(np.sqrt(max(v @ self.cov @ v, 0)))
    def dof(self): return self.n - self.npar


# the headline quantities as linear combinations
QUANT = {
 'h_kuu': {'h_kuu': 1}, 'h_kww': {'h_kww': 1}, 'c_kuu': {'c_kuu': 1}, 'c_kww': {'c_kww': 1},
 'gap_kuu': {'c_kuu': 1, 'h_kuu': -1}, 'gap_kww': {'c_kww': 1, 'h_kww': -1},
 'gtilt_U': {'c_lu': 1, 'h_lu': -1}, 'gtilt_W': {'c_lw': 1, 'h_lw': -1}, 'beta': {'beta': 1},
}

def quants(R, extra=None):
    out = {}
    for k, comb in QUANT.items():
        try: out[k] = R.lin(comb)
        except KeyError: out[k] = (np.nan, np.nan)
    return out


def retune(classes, base_floors, start=None, **kw):
    """joint.main's floor iteration, with this fitter."""
    floors = dict(start) if start else {c: 0.0 for c in CLASSES}
    scale = {'hp': 0.1, 'w0': 0.02, 'd': 0.002, 'G': 0.2, 'GY': 0.2, 'T': 0.005, 'v': 0.005}
    for it in range(12):
        R = Fitter(CLASSES, floors, **kw).fit()
        new = {}
        for c, (chi, n) in R.per.items():
            red = chi / max(n, 1)
            new[c] = floors[c] if red <= 1.05 else floors[c] + scale[c] * np.sqrt(red - 1) * (0.5 if it else 1)
        if all(abs(new[c] - floors[c]) < 1e-6 for c in CLASSES): break
        floors = new
    return floors, R


def depth_comb(pre, axis, extras, U=1.5):
    """combination for the even part of surface pre ('h'/'c') along axis ('u'/'w') at |U|=1.5 (150 mm): (s(+)+s(-))/2 - s(0)."""
    k = {'u': 'kuu', 'w': 'kww'}[axis]
    d = {f'{pre}_{k}': U * U / 2}
    for e in extras:
        if (axis == 'u' and e in ('u4',)) or (axis == 'w' and e in ('w4',)):
            d[f'{pre}_{e}'] = U**4
        if e == 'r4':
            d[f'{pre}_{e}'] = U**4
    return d

def five(R, eh=(), ec=()):
    """mesh x, mesh y, cathode x, cathode y depth at 150 mm and gap tilt x (mm per 100 mm): (value, err)."""
    out = {'mesh_x': R.lin(depth_comb('h', 'u', eh)), 'mesh_y': R.lin(depth_comb('h', 'w', eh)),
           'cath_x': R.lin(depth_comb('c', 'u', ec)), 'cath_y': R.lin(depth_comb('c', 'w', ec)),
           'gtilt_U': R.lin({'c_lu': 1, 'h_lu': -1})}
    return out
