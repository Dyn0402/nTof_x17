#!/usr/bin/env python3
"""bench_attach.py -- head-on bench stacks with the physics written out:

  electronics h  = the X-plane template of the chamber's bundle (X has no
                   resistive spreading, so its template is the DREAM response;
                   Y's carries an extra undershoot from charge leaving along
                   the resistive strip and is NOT used),
  arrival I(u)   = exp(-u / lambda) over the drift time T = gap / v
                   (attachment; lambda free),
  footprint      = sigma^2 = sigma_0^2 + 2 Dd u  (+ 2 D_rc s on Y),
                   Dd pinned from Magboltz at the chamber's measured v
                   (bench_pinned.py's water interpolation), or free,

    W_o(t) = A_view * sum_u I(u) sum_s h(t - t_x - u - s) dF_o(u, s)

Fitted per chamber over both views: sigma_0, D_rc, lambda, per-view amplitude
and time offset; then the same with Dd free as the test of the Magboltz value.

    bench_attach.py
"""
import json, os, pickle, sys
import numpy as np
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from width_vs_time import SOURCES
from rc_diffusion import stack, F, GRID, OFFS, STEP

BUNDLES = '/tmp/claude-1000/-home-dylan-PycharmProjects-nTof-x17/9071ba25-9a97-4e04-b80b-71e5850e28a6/scratchpad/prodbundles/bundles'
BUNDLE = {'det2': 'mx17_2/calib_bundle_r06', 'det3': 'mx17_3/calib_bundle_r06',
          'det4': 'mx17_4/calib_bundle_lp', 'det6': 'mx17_6/calib_bundle_lp',
          'det7': 'mx17_7/calib_bundle_r06'}
GAP = 30.0
# uniform charge along the track (no attachment); the attenuated form is kept
# only to show it was fitting the stacking artefact (FINDINGS §10)
UNIFORM = os.environ.get('CB_ATTACH', '0') != '1'


def template(det):
    z = np.load(f'{BUNDLES}/{BUNDLE[det]}/arrays.npz')
    g, h = z['grid'], z['tmpl_x']
    tg = np.arange(0, g[-1] - g[0], STEP)
    hh = np.interp(tg + g[0], g, h)
    return hh / hh.max(), g[0]


def predict(h, sig0, Dd, Drc, lam, T, A, t_shift, n):
    nu = max(int(round(T / STEP)), 1)
    u = (np.arange(nu) + 0.5) * STEP
    I = np.exp(-u / lam)
    s = np.arange(n) * STEP
    out = []
    for o in OFFS:
        acc = np.zeros(n)
        for k in range(nu):
            sig = np.sqrt(sig0 ** 2 + 2 * Dd * u[k] + 2 * Drc * s)
            dF = np.diff(np.r_[0.0, F(sig, o)])
            resp = np.convolve(h, dF)[:n]
            acc[k:] += I[k] * resp[:n - k]
        out.append(acc)
    P = A * np.array(out)
    t = np.arange(n) * STEP
    return np.array([np.interp(t - t_shift, t, p, left=0.0, right=p[-1]) for p in P])


def main():
    pin = json.load(open(f'{HERE}/results/bench_pinned.json'))
    out = {}
    for name in ('det2', 'det3', 'det4', 'det6', 'det7'):
        h, g0 = template(name)
        v = pin[name]['v']; T = GAP / (v * 1e-3)
        Dd_mb = pin[name]['magboltz']['Dd']
        ev = pickle.load(open(SOURCES[name], 'rb'))
        Ms = {p: stack(ev, p)[1] for p in ('x', 'y')}
        sels = {p: np.all(np.isfinite(M), axis=0) for p, M in Ms.items()}
        n = len(GRID)
        # data grid starts at GRID[0] = -300 relative to t50; model t starts at 0
        out[name] = dict(v=v, T=T, Dd_magboltz=Dd_mb)
        for lab in ('Dd_magboltz', 'Dd_free', 'Dd_zero'):
            def unpack(q):
                if lab == 'Dd_free':
                    sig0, Drc, lam, ax, ay, dx, dy, Dd = q
                else:
                    sig0, Drc, lam, ax, ay, dx, dy = q
                    Dd = Dd_mb if lab == 'Dd_magboltz' else 0.0
                return sig0, Drc, lam, ax, ay, dx, dy, Dd

            def res(q):
                sig0, Drc, lam, ax, ay, dx, dy, Dd = unpack(q)
                r = []
                for p, A, dt in (('x', ax, dx), ('y', ay, dy)):
                    P = predict(h, sig0, Dd, Drc if p == 'y' else 0.0, lam, T, A, dt, n)
                    r.append((P[:, sels[p]] - Ms[p][:, sels[p]]).ravel())
                return np.concatenate(r)
            x0 = [0.4, 4e-4, 800.0, 0.02, 0.02, 0.0, 0.0] + ([2e-4] if lab == 'Dd_free' else [])
            lo = [0.02, 0, 50, 1e-4, 1e-4, -600, -600] + ([0] if lab == 'Dd_free' else [])
            hi = [1.5, 0.01, 1e5, 10, 10, 600, 600] + ([0.005] if lab == 'Dd_free' else [])
            if UNIFORM:                     # no attenuation: lambda pinned at 1e6 ns
                x0[2], lo[2], hi[2] = 1e6 - 1, 1e6 - 2, 1e6
            # coarse start for the time offset: the model's 50 % rise onto the data's (-300 grid start)
            best = None
            for d0 in (100.0, 200.0, 300.0, 400.0):
                x0[5] = x0[6] = d0
                r = least_squares(res, x0, bounds=(lo, hi), diff_step=1e-3,
                                  x_scale=[0.1, 1e-4, 200, 0.01, 0.01, 30, 30] + ([1e-4] if lab == 'Dd_free' else []))
                if best is None or r.cost < best.cost:
                    best = r
            r = best
            J = r.jac
            cov = np.linalg.pinv(J.T @ J) * (r.fun ** 2).sum() / max(len(r.fun) - len(r.x), 1)
            e = np.sqrt(np.diag(cov))
            sig0, Drc, lam, ax, ay, dx, dy, Dd = unpack(r.x)
            out[name][lab] = dict(sig0=sig0, sig0_err=float(e[0]), Drc=Drc, lam=lam, Dd=Dd,
                                  Dd_err=float(e[7]) if lab == 'Dd_free' else None,
                                  rms=float(np.sqrt((r.fun ** 2).mean())))
            q = out[name][lab]
            print(f'{name} {lab:12s}: sig0 {sig0:.3f}±{e[0]:.3f}  Dd {Dd:.2e}'
                  f'{"±%.1e" % e[7] if lab == "Dd_free" else "":9s} Drc {Drc:.2e}  lambda {lam:.0f} ns '
                  f'(= {lam * v * 1e-3:.0f} mm)  rms {q["rms"]:.4f}', flush=True)
    tag = 'uniform' if UNIFORM else 'attach'
    json.dump(out, open(f'{HERE}/results/bench_{tag}.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
