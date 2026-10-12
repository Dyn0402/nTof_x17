#!/usr/bin/env python3
"""bench_pinrc.py -- the bench head-on fit with D_rc pinned at the beam value, and the arms it implies.

FINDINGS §27: det4's bench D_rc comes out above its beam value (2.2e-4 mm^2/ns, loss-corrected,
rc_drift.py), and on the bench D_rc is degenerate with the drift diffusion Dd.  Here the beam D_rc
is imposed on the unbiased bench head-on stack with bench_attach's model (X-template electronics,
uniform arrival over T = gap / v, sigma^2 = sigma_0^2 + 2 Dd u + 2 D_rc s) and the rest refitted --
sigma_0, per-view amplitude and offset, and Dd free / wet Magboltz / dry Magboltz -- so the stack
says what the pin costs and plane_bench says whether reconstruction cares.

    bench_pinrc.py --profile [--dets det2,det3,det4,det6,det7]
        -> results/bench_pinrc_profile_<det>.json: rms vs pinned D_rc (Dd free) and vs pinned Dd (D_rc free),
           the shape of the D_rc-Dd degeneracy per chamber

    bench_pinrc.py [--det det4] [--drc 2.2e-4] [--arms-out DIR]
        -> results/bench_pinrc_<det>.json; arms <det>_rcf (all free: reproduces bench_uniform
           Dd_free = rcm), <det>_rcp (beam D_rc, Dd free), _rcpw (wet Dd), _rcpd (dry Dd)
"""
import argparse, json, os, pickle, sys
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from width_vs_time import SOURCES
from rc_diffusion import stack, GRID
from bench_attach import predict, template, bundle_dir, GAP

LAM = 1e6          # no attenuation on the bench (FINDINGS §10, §21)


def fit(h, T, Ms, sels, drc=None, dd=None):
    """Free: sigma_0, A_x, A_y, dt_x, dt_y, plus Dd and D_rc unless given."""
    n = len(GRID)

    def unpack(q):
        sig0, ax, ay, dx, dy, *rest = q
        Dd = dd if dd is not None else rest.pop(0)
        Drc = drc if drc is not None else rest.pop(0)
        return sig0, Dd, Drc, ax, ay, dx, dy

    def res(q):
        sig0, Dd, Drc, ax, ay, dx, dy = unpack(q)
        r = []
        for p, A, dt in (('x', ax, dx), ('y', ay, dy)):
            P = predict(h, sig0, Dd, Drc if p == 'y' else 0.0, LAM, T, A, dt, n)
            r.append((P[:, sels[p]] - Ms[p][:, sels[p]]).ravel())
        return np.concatenate(r)

    x0, lo, hi = [0.4, 0.02, 0.02, 0.0, 0.0], [0.02, 1e-4, 1e-4, -600, -600], [1.5, 10, 10, 600, 600]
    sc = [0.1, 0.01, 0.01, 30, 30]
    for given, start in ((dd, 2e-4), (drc, 4e-4)):
        if given is None:
            x0.append(start); lo.append(0.0); hi.append(0.005 if given is dd else 0.01); sc.append(1e-4)
    best = None
    for d0 in (100.0, 200.0, 300.0, 400.0):
        x0[3] = x0[4] = d0
        r = least_squares(res, x0, bounds=(lo, hi), diff_step=1e-3, x_scale=sc)
        if best is None or r.cost < best.cost:
            best = r
    J = best.jac
    cov = np.linalg.pinv(J.T @ J) * (best.fun ** 2).sum() / max(len(best.fun) - len(best.x), 1)
    e = np.sqrt(np.diag(cov))
    sig0, Dd, Drc, ax, ay, dx, dy = unpack(best.x)
    nx = int(sels['x'].sum()) * Ms['x'].shape[0]
    rx, ry = best.fun[:nx], best.fun[nx:]
    return dict(sig0=sig0, sig0_err=float(e[0]), Dd=Dd, Drc=Drc, A=[ax, ay], dt=[dx, dy],
                rms=float(np.sqrt((best.fun ** 2).mean())), rms_x=float(np.sqrt((rx ** 2).mean())),
                rms_y=float(np.sqrt((ry ** 2).mean())), n_free=len(best.x))


DRC_GRID = [1.0e-4, 1.5e-4, 2.2e-4, 3.0e-4, 4.0e-4, 5.5e-4, 7.5e-4]
DD_GRID = [0.0, 1.0e-4, 1.95e-4, 3.0e-4, 4.25e-4, 6.0e-4]


def load(det):
    pin = json.load(open(f'{HERE}/results/bench_pinned.json'))[det]
    T = GAP / (pin['v'] * 1e-3)
    h, _ = template(det)
    ev = pickle.load(open(SOURCES[det], 'rb'))
    Ms = {p: stack(ev, p)[1] for p in ('x', 'y')}
    sels = {p: np.all(np.isfinite(M), axis=0) for p, M in Ms.items()}
    return pin, T, h, Ms, sels


def profile(dets):
    out = {}
    for det in dets:
        pin, T, h, Ms, sels = load(det)
        free = fit(h, T, Ms, sels)
        rows_r = [dict(Drc=d, **{k: fit(h, T, Ms, sels, drc=d)[k] for k in ('Dd', 'sig0', 'rms')}) for d in DRC_GRID]
        rows_d = [dict(Dd=d, **{k: fit(h, T, Ms, sels, dd=d)[k] for k in ('Drc', 'sig0', 'rms')}) for d in DD_GRID]
        out[det] = dict(free=free, vs_drc=rows_r, vs_dd=rows_d, Dd_wet=pin['magboltz']['Dd'], Dd_dry=pin['dry']['Dd'])
        print(f'{det} free: sig0 {free["sig0"]:.3f} Dd {free["Dd"]:.2e} Drc {free["Drc"]:.2e} rms {free["rms"]:.5f}')
        for r in rows_r:
            print(f'   Drc {r["Drc"]:.1e} -> Dd {r["Dd"]:.2e} sig0 {r["sig0"]:.3f} rms/free {r["rms"] / free["rms"]:.4f}')
        for r in rows_d:
            print(f'   Dd  {r["Dd"]:.2e} -> Drc {r["Drc"]:.2e} sig0 {r["sig0"]:.3f} rms/free {r["rms"] / free["rms"]:.4f}',
                  flush=True)
        json.dump(out[det], open(f'{HERE}/results/bench_pinrc_profile_{det}.json', 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--dets', default='det2,det3,det4,det6,det7')
    ap.add_argument('--det', default='det4')
    ap.add_argument('--drc', type=float, default=2.2e-4, help='beam D_rc, loss-corrected (FINDINGS §27)')
    ap.add_argument('--arms-out', default='/home/dylan/x17/cosmic_bench/cloud_basics/arms')
    a = ap.parse_args()
    if a.profile:
        return profile(a.dets.split(','))
    det = a.det
    pin, T, h, Ms, sels = load(det)
    cases = {'rcf': dict(),
             'rcp': dict(drc=a.drc),
             'rcpw': dict(drc=a.drc, dd=pin['magboltz']['Dd']),
             'rcpd': dict(drc=a.drc, dd=pin['dry']['Dd'])}
    out = dict(det=det, drc_beam=a.drc, T=T, bundle=bundle_dir(det), fits={})
    base = json.load(open(f'{bundle_dir(det)}/bundle.json'))['hyper']
    base = {k: float(v) for k, v in base.items() if k != 'kTauY'}
    for tag, kw in cases.items():
        r = fit(h, T, Ms, sels, **kw)
        out['fits'][tag] = r
        print(f'{det} {tag:5s}: sig0 {r["sig0"]:.3f}±{r["sig0_err"]:.3f}  Dd {r["Dd"]:.2e}  Drc {r["Drc"]:.2e}  '
              f'rms {r["rms"]:.5f} (x {r["rms_x"]:.5f} y {r["rms_y"]:.5f})  [{r["n_free"]} free]', flush=True)
        hy = dict(base, sigma_p0=r['sig0'], Dp=float(np.sqrt(2 * r['Dd'])), rc_D_y=r['Drc'], rc_D_x=0.0,
                  rc_tmpl=1.0)
        name = f'{det}_{tag}'
        json.dump(dict(arm=name, hyper=hy, model_frac=0.0,
                       source=f'bench_pinrc.py {tag} {kw}: head-on bench stack, no chi2 refit'),
                  open(f'{a.arms_out}/arm_{name}.json', 'w'), indent=1)
    json.dump(out, open(f'{HERE}/results/bench_pinrc_{det}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
