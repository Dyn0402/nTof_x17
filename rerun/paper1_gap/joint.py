#!/usr/bin/env python3
"""joint.py <work> -- one mesh surface and one cathode surface for chamber A, fitted to every observable of cellobs.py.

Model (mm, M3 z up, U, W in 100 mm from (200, 200); quadratic surfaces q(U, W) = l_u U + l_w W + k_uu U^2/2 + k_ww W^2/2
+ k_uw U W):
  mesh      h = h0 + q_h                    cathode   c = h0 + g0 + q_c          gap  g = c - h = g0 + q_c - q_h
  hp_x, hp_y = h + o_x, o_y                     (position route; one offset per view and run: the alignment z)
  w0_x = W_x + v (dh/dU + dc/dU) / 200           (drift lines leave the mesh normal to it and end normal to the cathode;
  w0_y = W_y + v (dh/dW + dc/dW) / 200            over the column the mean tilt is the average of the two slopes)
  dX   = (dh/dU + dc/dU) / 200,  dY likewise     (the same drift tilt, timing-free, from the +/- tan gap asymmetry)
  G    = g,  GY = g + o_GY                       (geometric gap, X / Y strips)
  ln T = ln T_run + (1 + beta) ln(g / 28.2)      (drift time at fixed voltage, v ~ E^beta, E = V / g)
  ln v_p = ln v_{p,run} - beta ln(g / 28.2)      (angle-scale velocity)
Arms (which classes enter): all; tilt (hp, w0, d: the cathode from the drift tilt alone); gap (hp, G, GY, T, v);
geo (hp, G, GY, d: no timing at all); timing (hp, T, v, w0).  Each class gets an error floor added in quadrature so that
its reduced chi2 in the 'all' fit is 1 (iterated): the floor is the size of what a quadratic model does not describe.
-> results/joint.json
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
V_DRIFT = 37.0
G_REF = 28.2
CLASSES = ('hp', 'w0', 'd', 'G', 'GY', 'T', 'v')
ARMS = {'all': CLASSES, 'tilt': ('hp', 'w0', 'd'), 'gap': ('hp', 'G', 'GY', 'T', 'v'), 'geo': ('hp', 'G', 'GY', 'd'),
        'timing': ('hp', 'w0', 'T', 'v'), 'geo_gap': ('hp', 'G', 'GY'),
        'no_hp': ('w0', 'd', 'G', 'GY', 'T', 'v'), 'm3dish': CLASSES, 'beta068': CLASSES, 'beta087': CLASSES}
#: arm options: m3dish adds a dish of the reference telescope to the position route only (hp = h + m3); betaXXX fixes beta
OPTS = {'m3dish': dict(m3=True), 'beta068': dict(beta=0.68), 'beta087': dict(beta=0.87)}
SURF = ['lu', 'lw', 'kuu', 'kww', 'kuw']


def q(p, U, W):
    return p[0] * U + p[1] * W + p[2] * U * U / 2 + p[3] * W * W / 2 + p[4] * U * W


def dq(p, U, W):
    return p[0] + p[2] * U + p[4] * W, p[1] + p[3] * W + p[4] * U


class Model:
    def __init__(self, obs, classes, floors, m3=False):
        self.runs = sorted(obs)
        self.classes, self.floors = classes, floors
        names = ['h_' + s for s in SURF] + ['c_' + s for s in SURF] + ['g0', 'beta', 'oGY', 'm3_kuu', 'm3_kww']
        self.m3 = m3
        for r in self.runs:
            names += [f'ox_{r}', f'oy_{r}', f'Wx_{r}', f'Wy_{r}', f'lnT_{r}', f'lnvx_{r}', f'lnvy_{r}']
        self.names = names
        self.obs = obs

    def unpack(self, p):
        return dict(zip(self.names, p))

    def resid(self, p, detail=False):
        P = self.unpack(p)
        qh = [P['h_' + s] for s in SURF]; qc = [P['c_' + s] for s in SURF]
        out, lab = [], []
        for r in self.runs:
            for o in self.obs[r]:
                U, W = o['U'], o['W']
                h = q(qh, U, W)
                g = P['g0'] + q(qc, U, W) - h
                hu, hw = dq(qh, U, W); cu, cw = dq(qc, U, W)
                tu, tw = (hu + cu) / 200.0, (hw + cw) / 200.0
                m3 = (P['m3_kuu'] * U * U + P['m3_kww'] * W * W) / 2
                pred = {
                    'hp_x': h + m3 + P[f'ox_{r}'], 'hp_y': h + m3 + P[f'oy_{r}'],
                    'w0_x': P[f'Wx_{r}'] + V_DRIFT * tu, 'w0_y': P[f'Wy_{r}'] + V_DRIFT * tw,
                    'dX': tu, 'dY': tw, 'G': g, 'GY': g + P['oGY'],
                    'T': P[f'lnT_{r}'] + (1 + P['beta']) * np.log(g / G_REF),
                    'v_x': P[f'lnvx_{r}'] - P['beta'] * np.log(g / G_REF),
                    'v_y': P[f'lnvy_{r}'] - P['beta'] * np.log(g / G_REF)}
                for k, pr in pred.items():
                    cls = {'hp_x': 'hp', 'hp_y': 'hp', 'w0_x': 'w0', 'w0_y': 'w0', 'dX': 'd', 'dY': 'd', 'G': 'G', 'GY': 'GY',
                           'T': 'T', 'v_x': 'v', 'v_y': 'v'}[k]
                    if cls not in self.classes:
                        continue
                    y, e = o.get(k), o.get(k + '_err')
                    if y is None or not np.isfinite(y) or not np.isfinite(e) or e <= 0:
                        continue
                    if cls in ('T', 'v'):
                        y, e = np.log(y), e / y
                    e = np.hypot(e, self.floors.get(cls, 0.0))
                    out.append((y - pr) / e)
                    lab.append(cls)
        return (np.array(out), lab) if detail else np.array(out)


def fit(obs, classes, floors, m3=False, beta=None):
    M = Model(obs, classes, floors, m3)
    p0 = np.zeros(len(M.names))
    P = dict(zip(M.names, p0)); P['g0'] = 28.2; P['beta'] = 0.68 if beta is None else beta
    for r in M.runs:
        P[f'lnT_{r}'] = np.log(np.nanmedian([o['T'] for o in obs[r]]))
        P[f'lnvx_{r}'] = np.log(np.nanmedian([o['v_x'] for o in obs[r]]))
        P[f'lnvy_{r}'] = np.log(np.nanmedian([o['v_y'] for o in obs[r]]))
    p0 = np.array([P[n] for n in M.names])
    used = set()
    for c in classes:
        used |= {'hp': {'ox', 'oy'}, 'w0': {'Wx', 'Wy'}, 'GY': {'oGY'}, 'T': {'lnT'}, 'v': {'lnvx', 'lnvy'}}.get(c, set())
    free = [i for i, n in enumerate(M.names) if n.split('_')[0] not in {'ox', 'oy', 'Wx', 'Wy', 'oGY', 'lnT', 'lnvx', 'lnvy'}
            or n.split('_')[0] in used]
    if not ({'T', 'v'} & set(classes)) or beta is not None:
        free = [i for i in free if M.names[i] != 'beta']
    if not m3:
        free = [i for i in free if not M.names[i].startswith('m3_')]
    if not ({'G', 'GY', 'T', 'v'} & set(classes)):
        free = [i for i in free if M.names[i] != 'g0']

    def f(x):
        p = p0.copy(); p[free] = x
        return M.resid(p)
    r = least_squares(f, p0[free], x_scale='jac')
    p = p0.copy(); p[free] = r.x
    J = r.jac
    cov = np.linalg.pinv(J.T @ J)
    err = np.full(len(p), np.nan); err[free] = np.sqrt(np.diag(cov))
    res, lab = M.resid(p, detail=True)
    per = {c: dict(chi2=float(np.sum(res[np.array(lab) == c] ** 2)), n=int(np.sum(np.array(lab) == c))) for c in classes}
    return dict(params={n: float(v) for n, v in zip(M.names, p)}, errors={n: float(v) for n, v in zip(M.names, err)},
                chi2=float(np.sum(res ** 2)), n=len(res), nfree=len(free), per_class=per)


def derived(F):
    P, E = F['params'], F['errors']
    d = {}
    for s in ('kuu', 'kww'):
        h, c = P['h_' + s], P['c_' + s]
        d[f'cathode/mesh {s}'] = c / h if h else np.nan
        d[f'gap {s}'] = c - h
        d[f'r_f {s}'] = (1 + c / h) / 2 if h else np.nan
        d[f'gap {s} err'] = float(np.hypot(E['h_' + s], E['c_' + s]))
    d['gap tilt U (mm/100mm)'] = P['c_lu'] - P['h_lu']
    d['gap tilt W (mm/100mm)'] = P['c_lw'] - P['h_lw']
    d['mesh dish at 150 mm, U / W'] = (P['h_kuu'] * 1.125, P['h_kww'] * 1.125)
    return d


def main():
    work = sys.argv[1]
    obs = json.load(open(f'{work}/cellobs.json'))
    floors = {c: 0.0 for c in CLASSES}
    for it in range(8):
        F = fit(obs, CLASSES, floors)
        new = {}
        for c, v in F['per_class'].items():
            red = v['chi2'] / max(v['n'], 1)
            scale = {'hp': 0.1, 'w0': 0.02, 'd': 0.002, 'G': 0.2, 'GY': 0.2, 'T': 0.005, 'v': 0.005}[c]
            new[c] = floors[c] if red <= 1.05 else floors[c] + scale * np.sqrt(red - 1) * (0.5 if it else 1)
        if all(abs(new[c] - floors[c]) < 1e-6 for c in CLASSES):
            break
        floors = new
    print('floors', {c: round(v, 4) for c, v in floors.items()})
    out = dict(floors=floors, arms={})
    for arm, cl in ARMS.items():
        F = fit(obs, cl, floors, **OPTS.get(arm, {}))
        F['derived'] = derived(F)
        out['arms'][arm] = F
        P, E = F['params'], F['errors']
        print(f'== {arm:8s} chi2 {F["chi2"]:.1f}/{F["n"] - F["nfree"]}  per class',
              {c: f'{v["chi2"]:.0f}/{v["n"]}' for c, v in F['per_class'].items()})
        print('   mesh  ', ' '.join(f'{s} {P["h_" + s]:+.2f}±{E["h_" + s]:.2f}' for s in SURF))
        print('   cathode', ' '.join(f'{s} {P["c_" + s]:+.2f}±{E["c_" + s]:.2f}' for s in SURF),
              f' g0 {P["g0"]:.2f} beta {P["beta"]:.2f}±{E["beta"]:.2f}'
              + (f'  M3 dish kuu {P["m3_kuu"]:+.2f}±{E["m3_kuu"]:.2f} kww {P["m3_kww"]:+.2f}±{E["m3_kww"]:.2f}' if arm == 'm3dish' else ''))
        print('   derived', {k: (round(v, 3) if isinstance(v, float) else tuple(round(x, 2) for x in v)) for k, v in F['derived'].items()})
    os.makedirs(f'{HERE}/results', exist_ok=True)
    json.dump(out, open(f'{HERE}/results/joint.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
