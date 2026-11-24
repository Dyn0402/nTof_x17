#!/usr/bin/env python3
"""q5_physics.py <model> -- the fitted potential and common scale in physical terms (k = 1 first guess).

Conventions (t6lib): r_loc = dphi/du + s j_loc with u = x_local/200; the drift-line deflection delta = d(local p)/dw
= -r_loc, and in the perpendicular-field picture delta = grad (h + c)/2 with h, c the mesh and cathode heights measured
along w (strips -> cathode).  So (h + c)/2 = -200 phi  [mm].  Gap: 1 + s = v_true/v_assumed, v ~ g^(-beta) =>
dg/g = -s/beta (beta = 0.68) or -s (d ln v/d ln g = -1).  g0 = 28.2 mm (bench A).  Errors: Gaussian from the fit
covariance (2000 draws) and, separately, the spread of the two run-parity halves.
-> physics_<model>.json, phys_<model>_<arm>.png
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from t6lib import OUT, mono, Poly, rows_ac, rows_d

G0 = 28.2
rng = np.random.default_rng(3)


def phi_val(beta, names, u, v):
    out = 0.0
    for b, n in zip(beta, names):
        if n.startswith('phi_'):
            i, k = int(n[4]), int(n[5])
            out = out + b * u ** i * v ** k
    return -200.0 * out                      # (h + c)/2 in mm


def s_val(beta, names, u, v, view=None):
    out = 0.0
    for b, n in zip(beta, names):
        if n.startswith('s_') and n != 's_off_y':
            i, k = int(n[2]), int(n[3])
            out = out + b * u ** i * v ** k
        elif view is not None and n.startswith(f's{view}_'):
            i, k = int(n[3]), int(n[4])
            out = out + b * u ** i * v ** k
    return out


def quantities(beta, names):
    q = {}
    p = lambda u, v: phi_val(beta, names, u, v)  # noqa: E731
    q['phi_dish_x150_mm'] = 0.5 * (p(0.75, 0) + p(-0.75, 0)) - p(0, 0)
    q['phi_dish_y150_mm'] = 0.5 * (p(0, 0.75) + p(0, -0.75)) - p(0, 0)
    q['phi_tilt_x_mm_per_100'] = (p(0.25, 0) - p(-0.25, 0))     # over 100 mm
    q['phi_tilt_y_mm_per_100'] = (p(0, 0.25) - p(0, -0.25))
    s = lambda u, v: s_val(beta, names, u, v)  # noqa: E731
    q['s_centre'] = s(0, 0)
    q['s_grad_x_per_100'] = s(0.25, 0) - s(-0.25, 0)
    q['s_grad_y_per_100'] = s(0, 0.25) - s(0, -0.25)
    q['s_dome_x150'] = 0.5 * (s(0.75, 0) + s(-0.75, 0)) - s(0, 0)
    q['s_dome_y150'] = 0.5 * (s(0, 0.75) + s(0, -0.75)) - s(0, 0)
    for lab, fac in (('b068', 1 / 0.68), ('dlnv_m1', 1.0)):
        g = lambda u, v: -G0 * fac * s(u, v)  # noqa: E731,B023
        q[f'gap_{lab}_grad_x_mm_per_100'] = g(0.25, 0) - g(-0.25, 0)
        q[f'gap_{lab}_grad_y_mm_per_100'] = g(0, 0.25) - g(0, -0.25)
        q[f'gap_{lab}_dome_x150_mm'] = 0.5 * (g(0.75, 0) + g(-0.75, 0)) - g(0, 0)
        q[f'gap_{lab}_dome_y150_mm'] = 0.5 * (g(0, 0.75) + g(0, -0.75)) - g(0, 0)
        # mesh and cathode shapes: h = phi - g/2, c = phi + g/2 (curvature only is meaningful)
        q[f'mesh_{lab}_dish_x150_mm'] = q['phi_dish_x150_mm'] - 0.5 * q[f'gap_{lab}_dome_x150_mm']
        q[f'mesh_{lab}_dish_y150_mm'] = q['phi_dish_y150_mm'] - 0.5 * q[f'gap_{lab}_dome_y150_mm']
        q[f'cath_{lab}_dish_x150_mm'] = q['phi_dish_x150_mm'] + 0.5 * q[f'gap_{lab}_dome_x150_mm']
        q[f'cath_{lab}_dish_y150_mm'] = q['phi_dish_y150_mm'] + 0.5 * q[f'gap_{lab}_dome_y150_mm']
    return q


def main(model, arms=('A', 'C', 'D')):
    co = json.load(open(OUT / 'coeffs.json'))
    out = {}
    for arm in arms:
        if model not in co.get(arm, {}):
            continue
        c = co[arm][model]
        b, names = np.array(c['beta']), c['names']
        cov = np.array(c['cov'])
        q0 = quantities(b, names)
        draws = rng.multivariate_normal(b, cov, 2000)
        Q = [quantities(d, names) for d in draws]
        err = {k: float(np.std([x[k] for x in Q])) for k in q0}
        # parity halves
        D = rows_d() if arm == 'D' else rows_ac(arm)
        halves = []
        for tr in (0, 1):
            m = Poly(**{k: (np.asarray(v) if k == 'cellA' and v is not None else v) for k, v in c['spec'].items()})
            m.fit(D[(D.part == tr) & D.clean])
            halves.append(quantities(m.beta, m.names))
        out[arm] = {k: dict(value=float(q0[k]), err_fit=err[k], half0=float(halves[0][k]), half1=float(halves[1][k]))
                    for k in q0}
        # figure: (h+c)/2 and gap (beta 0.68) maps
        uu = np.linspace(-1, 1, 81)
        U, V = np.meshgrid(uu, uu, indexing='ij')
        P = phi_val(b, names, U, V)
        lin = np.polyfit  # remove the plane (alignment-degenerate) for display
        A_ = np.column_stack([np.ones(U.size), U.ravel(), V.ravel()])
        Pc = P.ravel() - A_ @ np.linalg.lstsq(A_, P.ravel(), rcond=None)[0]
        Sg = -G0 / 0.68 * s_val(b, names, U, V)
        fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
        for ax, Z, t in ((axs[0], Pc.reshape(U.shape), f'{arm}: (h+c)/2 minus plane (mm), k=1'),
                         (axs[1], Sg - Sg[40, 40], f'{arm}: gap change (mm), beta=0.68, g0=28.2')):
            vm = np.nanmax(np.abs(Z))
            im = ax.imshow(Z.T, origin='lower', extent=(-200, 200, -200, 200), cmap='RdBu_r', vmin=-vm, vmax=vm)
            ax.contour(U * 200, V * 200, Z, 10, colors='k', linewidths=0.4)
            ax.set_title(t, fontsize=9); ax.set_xlabel('x_local (mm)'); ax.set_ylabel('y_local (mm)')
            fig.colorbar(im, ax=ax, shrink=0.85)
        fig.savefig(OUT / f'phys_{model}_{arm}.png', dpi=110)
        plt.close(fig)
    json.dump(out, open(OUT / f'physics_{model}.json', 'w'), indent=1)
    for arm, d in out.items():
        print(f'== {arm} {model}')
        for k, v in d.items():
            print(f'  {k:34s} {v["value"]:+8.3f} +- {v["err_fit"]:.3f}   halves {v["half0"]:+8.3f} {v["half1"]:+8.3f}')


if __name__ == '__main__':
    main(sys.argv[1])
