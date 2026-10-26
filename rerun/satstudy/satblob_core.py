#!/usr/bin/env python3
"""satblob_core.py -- the fitting half of satblob.py, standalone (imports only wft + numpy/scipy/pandas) so it runs on
lxplus condor from a packed payload:

    python satblob_core.py <payload.pkl> <out.parquet> --bundle <bundle dir> [--masks prod railbot]

Payload: list of dict(event_id, view, group, P (window: W, pos, noise, ch), prior (t0, sigma) or None, top, bot
(per-sample rails of the window, pedestal + CM subtracted like W), starts ((p0, w, t0) rows), arms (track-only arms)).

Per view: (1) the production fit (wft.reco.fit_plane, satfit.py's way) under each track-only censor arm
  rail      top censor at the channel's own rail (4095 - ped - CM) instead of SAT = 3550
  bot       SAT = 3550 + the bottom rail (raw ADC 0) censored
  railbot   both
(2) the track-only and track + blob fits of the own optimiser (satblob.py docstring) under each mask in --masks,
from every start (the view's satfit arms + the fits of step 1).  'mask@f' adds a fractional model-error term f in
quadrature to the noise (sigma^2 = noise^2 + (f W)^2), as wft.model.MODEL_FRAC.
Censored samples carry production's one-sided penalty: top -- model below the rail; bottom -- model above it.
"""
import argparse, os, pickle, sys, time
import numpy as np
import pandas as pd

SAT = 3550.0
RAIL_TOL = 8.0          # ADC: within this of the rail = clipped (raw maxima sit exactly at 4095; float16 W is +-2)
K_BLOB = 5              # pb, kb, three bin amplitudes
BLOB_HALF = 1           # blob = depth bins kb-1 .. kb+1
CAL = None


def masks(Wraw, top, bot, arm):
    """(top-censor, bottom-censor) masks of a window under an arm."""
    hi = (Wraw >= SAT) if arm in ('prod', 'bot') else (Wraw >= top - RAIL_TOL)
    lo = (Wraw <= bot + RAIL_TOL) if arm in ('bot', 'railbot') else np.zeros_like(hi)
    return hi, lo & ~hi


def init(bundle):
    global CAL
    from wft.calib import CalibrationBundle
    from wft import model as wm
    CAL = CalibrationBundle.load(bundle); wm.use_calibration(CAL); wm.set_nsamp(32)


# ------------------------------------------------------------------------------------------------ production fit
def track_fit(J, arm):
    from wft import model as wm, reco
    P, v = J['P'], J['view']
    hi, lo = masks(np.asarray(P['W'], float), J['top'], J['bot'], arm)
    real_prep = wm.prep_plane

    def prep(PP, plane, _m=hi | lo):
        W, noise, pos, sat = real_prep(PP, plane)
        dead = sat & ~(np.asarray(PP['W'], float) >= wm.SAT)
        return W, noise, pos, _m | dead
    wm.prep_plane = prep
    try:
        f = reco.fit_plane(P, v, CAL, t0_prior=J['prior'])
    except Exception as e:      # noqa: BLE001 -- printed, not swallowed
        print('fit error', J['event_id'], v, arm, repr(e), flush=True); f = None
    finally:
        wm.prep_plane = real_prep
    return dict(event_id=J['event_id'], view=v, group=J['group'], arm=arm, ok=f is not None,
                n_top=int(hi.sum()), n_bot=int(lo.sum()), n_top_prod=int((np.asarray(P['W'], float) >= SAT).sum()),
                p0=f.p0 if f else np.nan, w=f.w if f else np.nan, t0=f.t0 if f else np.nan,
                chi2=f.chi2 if f else np.nan, dof=f.dof if f else np.nan)


# ------------------------------------------------------------------------------------------------ track + blob
class Plane:
    """One window, gain-corrected, with the arm's censor masks; chi2 of any block of columns."""

    def __init__(self, P, v, hi, lo, prior, frac=0.0):
        from wft import model as wm
        self.wm = wm; self.v = v
        W, noise, pos, sat = wm.prep_plane(P, v)                  # sat = production clip | dead
        dead = sat & ~(np.asarray(P['W'], float) >= wm.SAT)
        self.W, self.noise, self.pos = W, noise, pos
        self.hi, self.lo = hi & ~dead, lo & ~dead
        self.ok = ~(hi | lo | dead).reshape(-1)
        self.n = int(self.ok.sum())
        # frac > 0: a fractional model-error term in quadrature (wft.model.MODEL_FRAC's form), so percent-level
        # template mismatch on the bright unclipped edges of a saturated pulse cannot outweigh the faint track
        self.nz = np.sqrt(noise[:, None] ** 2 + (frac * np.maximum(W, 0.0)) ** 2)
        self.wt = (1.0 / self.nz).reshape(-1)
        self.y = (W / self.nz).reshape(-1)
        self.prior = prior
        self.nfev = 0

    def solve(self, M, t0):
        from scipy.optimize import nnls
        self.nfev += 1
        A = (M * self.wt[:, None])[self.ok]
        try:
            q, rn = nnls(A, self.y[self.ok], maxiter=50 * M.shape[1])
        except Exception:
            return np.inf, None
        chi = rn * rn
        if self.hi.any() or self.lo.any():
            model = (M @ q).reshape(self.W.shape)
            chi += float(((np.maximum(0, self.W - model)[self.hi] / self.nz[self.hi]) ** 2).sum())
            chi += float(((np.maximum(0, model - self.W)[self.lo] / self.nz[self.lo]) ** 2).sum())
        if self.prior is not None:
            chi += ((t0 - self.prior[0]) / self.prior[1]) ** 2
        return chi, q

    def Mt(self, p0, w, t0, h):
        return self.wm.build_matrix(self.v, self.pos, p0, w, t0, h)

    def Mb(self, pb, kb, t0, h):
        K = self.wm.K
        a, b = max(0, kb - BLOB_HALF), min(K, kb + BLOB_HALF + 1)
        return self.wm.build_matrix(self.v, self.pos, pb, 0.0, t0, h)[:, a:b]

    def chi_t(self, x, h):
        return self.solve(self.Mt(x[0], x[1], x[2], h), x[2])[0]

    def chi_tb(self, x, kb, h):
        return self.solve(np.hstack([self.Mt(x[0], x[1], x[2], h), self.Mb(x[3], kb, x[2], h)]), x[2])[0]


def _nm(f, x0, steps, maxiter):
    from scipy.optimize import minimize
    x0 = np.asarray(x0, float)
    simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * steps[i] for i in range(len(x0))])
    r = minimize(f, x0, method='Nelder-Mead', options=dict(xatol=1e-3, fatol=0.3, maxiter=maxiter,
                                                         initial_simplex=simplex))
    return float(r.fun), r.x


def blob_fit(pl, starts, hypers):
    """Track-only best (T) and track + blob best (TB) on one Plane."""
    best_t = (np.inf, None, None)
    for h in hypers:
        for s in starts:
            c, x = _nm(lambda x: pl.chi_t(x, h), s, (0.4, 1.5e-3, 20.0), 140)
            if c < best_t[0]:
                best_t = (c, x, h)
    cT, xT, h = best_t
    K = pl.wm.K
    MT = pl.Mt(*xT, h)
    grid = []
    for pb in pl.pos:                                       # blob cells at the T track: every strip x depth bin
        Mfull = pl.wm.build_matrix(pl.v, pl.pos, pb, 0.0, xT[2], h)
        for kb in range(K):
            a, b = max(0, kb - BLOB_HALF), min(K, kb + BLOB_HALF + 1)
            grid.append((pl.solve(np.hstack([MT, Mfull[:, a:b]]), xT[2])[0], pb, kb))
    grid.sort(key=lambda g: g[0])
    seen, cells = [], []
    for c, pb, kb in grid:                                  # the 3 best distinct cells
        if any(abs(pb - p) < 1.6 and abs(kb - k) <= 2 for p, k in seen):
            continue
        seen.append((pb, kb)); cells.append((pb, kb))
        if len(cells) == 3:
            break
    best_tb = (np.inf, None, None)
    for pb, kb in cells:
        c, x = _nm(lambda x: pl.chi_tb(x, kb, h), np.r_[xT, pb], (0.4, 1.5e-3, 20.0, 0.4), 220)
        for dk in (-1, 1):
            k2 = kb + dk
            if 0 <= k2 < K:
                c2, x2 = _nm(lambda x: pl.chi_tb(x, k2, h), x, (0.2, 0.8e-3, 10.0, 0.2), 120)
                if c2 < c:
                    c, x, kb = c2, x2, k2
        if c < best_tb[0]:
            best_tb = (c, x, kb)
    cTB, xTB, kbB = best_tb
    _, q = pl.solve(np.hstack([pl.Mt(*xTB[:3], h), pl.Mb(xTB[3], kbB, xTB[2], h)]), xTB[2])
    n = pl.n
    dbic = n * np.log(cT / cTB) - K_BLOB * np.log(n) if np.isfinite(cTB) and cTB > 0 else -np.inf
    return dict(chi2_T=cT, p0_T=xT[0], w_T=xT[1], t0_T=xT[2], chi2_TB=cTB, p0_TB=xTB[0], w_TB=xTB[1],
                t0_TB=xTB[2], pb=xTB[3], kb=int(kbB), q_track=float(q[:K].sum()), q_blob=float(q[K:].sum()),
                n=n, dbic=float(dbic), sig_p0=float(h.get(f'sigma_p0_{pl.v}', h.get('sigma_p0'))), nfev=pl.nfev)


def blob_view(J, mk, starts):
    v = J['view']; h0 = CAL.hyper; hypers = [h0]
    for s in h0.get(f'sigma_p0_prof_{v}') or ():
        h2 = dict(h0); h2[f'sigma_p0_{v}'] = float(s); hypers.append(h2)
    base, _, f = mk.partition('@')               # e.g. 'railbot@0.05' = railbot mask, 5 % model-error term
    hi, lo = masks(np.asarray(J['P']['W'], float), J['top'], J['bot'], base)
    pl = Plane(J['P'], v, hi, lo, J['prior'], float(f or 0.0))
    r = blob_fit(pl, starts, hypers)
    return dict(event_id=J['event_id'], view=v, group=J['group'], arm=f'blob_{mk}', mask=mk, frac=float(f or 0.0),
                n_top=int(hi.sum()), n_bot=int(lo.sum()), **r)


def pinned_view(J, mk):
    """Diagnostic: the same model with the track PINNED to M3 (p0, w from the reference; t0, the profile and the blob
    free).  chi2_Tref / chi2_TBref against the free optimum say whether a failure is the optimiser missing the right
    basin (pinned chi2 lower) or the model preferring the wrong line (pinned chi2 higher)."""
    v = J['view']; h = CAL.hyper
    base, _, f = mk.partition('@')
    hi, lo = masks(np.asarray(J['P']['W'], float), J['top'], J['bot'], base)
    pl = Plane(J['P'], v, hi, lo, J['prior'], float(f or 0.0))
    p0r, wr = J['ref']
    K = pl.wm.K
    best_t = (np.inf, None)
    for t0s in np.arange(100.0, 461.0, 30.0):
        c, x = _nm(lambda x: pl.chi_t((p0r, wr, x[0]), h), [t0s], (20.0,), 60)
        if c < best_t[0]:
            best_t = (c, x[0])
    cT, t0 = best_t
    MT = pl.Mt(p0r, wr, t0, h); grid = []
    for pb in pl.pos:
        Mfull = pl.wm.build_matrix(v, pl.pos, pb, 0.0, t0, h)
        for kb in range(K):
            a, b = max(0, kb - BLOB_HALF), min(K, kb + BLOB_HALF + 1)
            grid.append((pl.solve(np.hstack([MT, Mfull[:, a:b]]), t0)[0], pb, kb))
    grid.sort(key=lambda g: g[0])
    best = (np.inf, None, None)
    for c0, pb, kb in grid[:3]:
        c, x = _nm(lambda x: pl.chi_tb((p0r, wr, x[0], x[1]), kb, h), [t0, pb], (20.0, 0.4), 120)
        if c < best[0]:
            best = (c, x, kb)
    return dict(event_id=J['event_id'], view=v, group=J['group'], mask=mk, chi2_Tref=cT, t0_Tref=t0,
                chi2_TBref=best[0], pb_ref=best[1][1], kb_ref=int(best[2]), n=pl.n)


def run(payload, out, masks_, no_track=False, pinned=False):
    Js = pickle.load(open(payload, 'rb'))
    if pinned:
        rows = []
        for J in Js:
            if J.get('ref') is None:
                continue
            for mk in masks_:
                rows.append(pinned_view(J, mk))
            print(J['event_id'], J['view'], flush=True)
        pd.DataFrame(rows).to_parquet(out.replace('.parquet', '_pinned.parquet'))
        return
    T, B = [], []
    for i, J in enumerate(Js):
        t = time.time()
        starts = [tuple(s) for s in J['starts']]
        if not no_track:
            for arm in J['arms']:
                r = track_fit(J, arm); T.append(r)
                if r['ok']:
                    starts.append((r['p0'], r['w'], r['t0']))
        starts = np.unique(np.round(np.array(starts, float), 4), axis=0)
        for mk in masks_:
            try:
                B.append(blob_view(J, mk, starts))
            except Exception as e:      # noqa: BLE001
                print('blob error', J['event_id'], J['view'], mk, repr(e), flush=True)
        print(i, J['event_id'], J['view'], J['group'], f'{time.time() - t:.1f}s', flush=True)
    pd.DataFrame(T).to_parquet(out.replace('.parquet', '_track.parquet'))
    pd.DataFrame(B).to_parquet(out.replace('.parquet', '_blob.parquet'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('payload'); ap.add_argument('out')
    ap.add_argument('--bundle', required=True); ap.add_argument('--masks', nargs='+', default=['prod', 'railbot', 'railbot@0.05'])
    ap.add_argument('--no-track', action='store_true'); ap.add_argument('--pinned', action='store_true')
    a = ap.parse_args()
    init(a.bundle)
    run(a.payload, a.out, a.masks, a.no_track, a.pinned)


if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
    main()
