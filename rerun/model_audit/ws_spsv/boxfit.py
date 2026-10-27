#!/usr/bin/env python3
"""boxfit.py --bundle B --out o.json CACHE.pkl [CACHE.pkl ...] [--n 300] [--sig-end 30] [--tmpl x]

Drift-column duration T from the STRIP-SUM waveform, with no track model.  Summed over the strips of a view, every
lateral term of the forward model drops out (strip fractions sum to one at every time since landing, the RC increments
telescope to zero), so for any track angle

    S(t) = sum_k q(u_k) h(t - t0 - u_k)        h = the bench electronics template (tmpl_x by default, the clean
                                                  response the RC kernel drives both views with; same DREAM peaking and
                                                  gain registers at H4: 0xD023 / 0xAAAA)

and a track crossing the whole gap is a box in u: q(u) = a + b u/T for 0 <= u < T, its end smeared by a Gaussian of
width SIG_END (longitudinal diffusion, last-cluster depth).  Per event and view: grid over (t0, T) (20/60 ns, then
5/10 ns), (a, b) by linear least squares; samples missing on every strip of the window (RAW dropped sample group, or
ZS) are left out.  Also fits the stacked (per-event peak-normalised, per-sample median) strip sum.  Prints T, writes JSON.
"""
import argparse, json, os, pickle, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, REPO)
from wft.calib import CalibrationBundle          # noqa: E402

NS, DTS = 64, 60.0
TS = np.arange(NS) * DTS


class Kern:
    """Box integrals in closed form through the template's running integrals: with G(tau) = int_-inf^tau h and
    J(tau) = int_-inf^tau G, the flat box is G(tau) - G(tau - T) and the gradient term (1/T) int_0^T u h(tau - u) du
    = [J(tau) - J(tau - T)]/T - G(tau - T), tau = t - t0.  The end terms use G, J smeared by sig_end."""

    def __init__(self, grid, tmpl, sig_end):
        g = np.arange(grid[0], grid[-1] + 6000.0, 1.0)
        h = np.interp(g, grid, tmpl, left=0.0, right=0.0)
        G = np.cumsum(h); J = np.cumsum(G)
        self.g, self.G, self.J = g, G, J
        if sig_end > 0:
            x = np.arange(-int(4 * sig_end), int(4 * sig_end) + 1)
            k = np.exp(-0.5 * (x / sig_end) ** 2); k /= k.sum(); pad = len(x) // 2
            self.Gs = np.convolve(np.r_[np.zeros(pad), G, np.full(pad, G[-1])], k, 'valid')
            self.Js = np.convolve(np.r_[np.zeros(pad), J, J[-1] + G[-1] * np.arange(1, pad + 1)], k, 'valid')
        else:
            self.Gs, self.Js = G, J

    def __call__(self, t0, T):
        tau = TS - t0
        G0 = np.interp(tau, self.g, self.G, left=0.0); J0 = np.interp(tau, self.g, self.J, left=0.0)
        G1 = np.interp(tau - T, self.g, self.Gs, left=0.0); J1 = np.interp(tau - T, self.g, self.Js, left=0.0)
        return np.stack([G0 - G1, (J0 - J1) / max(T, 1.0) - G1], 1)


def fit_one(S, ok, kern, t0s, Ts, grad=True):
    best = (np.inf, None); y = S[ok]
    for t0 in t0s:
        for T in Ts:
            X = kern(t0, T)[ok]
            if not grad:
                X = X[:, :1]
            c, *_ = np.linalg.lstsq(X, y, rcond=None)
            c = np.r_[c, 0.0] if not grad else c
            if c[0] <= 0 or c[0] + c[1] <= 0:          # charge stays positive across the box
                continue
            r = y - X @ c[:X.shape[1]]; chi = float(r @ r)
            if chi < best[0]:
                best = (chi, (float(t0), float(T), float(c[0]), float(c[1])))
    return best


def fit_two_stage(S, ok, kern, grad=True, t0_lo=200, t0_hi=1000, T_lo=600, T_hi=3600):
    chi, p = fit_one(S, ok, kern, np.arange(t0_lo, t0_hi, 20.0), np.arange(T_lo, T_hi, 60.0), grad)
    if p is None:
        return chi, p
    return fit_one(S, ok, kern, np.arange(p[0] - 20, p[0] + 21, 5.0), np.arange(p[1] - 60, p[1] + 61, 10.0), grad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('caches', nargs='+'); ap.add_argument('--bundle', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=300); ap.add_argument('--sig-end', type=float, default=30.0)
    ap.add_argument('--views', default='xy'); ap.add_argument('--tmpl', default='x'); ap.add_argument('--no-grad', action='store_true')
    ap.add_argument('--stack-only', action='store_true', help='skip the per-event fits')
    a = ap.parse_args()
    cal = CalibrationBundle.load(a.bundle)
    grid = np.asarray(cal.grid, float); tmpl = np.asarray(cal.tmpl[a.tmpl], float); tmpl = tmpl / tmpl.max()
    kern = Kern(grid, tmpl, a.sig_end); grad = not a.no_grad
    res = {}
    for c in a.caches:
        C = pickle.load(open(c, 'rb')); key = os.path.basename(c).replace('.pkl', ''); res[key] = {}
        for view in a.views:
            stack, per = [], []
            for eid in sorted(C)[:a.n]:
                ev = C[eid]
                if view not in ev:
                    continue
                W = np.asarray(ev[view]['W'], float); S = W.sum(0); ok = ~(W == 0).all(0)
                if ok.sum() < 30:
                    continue
                S = S / S[ok].max()
                stack.append(np.where(ok, S, np.nan))
                chi, p = (np.inf, None) if a.stack_only else fit_two_stage(S, ok, kern, grad)
                if p is not None:
                    per.append((eid, *p, chi))
            per = np.array(per, float); st = np.nanmedian(np.array(stack), 0); okst = np.isfinite(st)
            chi, p = fit_two_stage(st, okst, kern, grad)
            per = per if len(per) else np.full((1, 6), np.nan); T = per[:, 2]
            r = dict(n=int(len(per)), T_med=float(np.median(T)), T_p16=float(np.percentile(T, 16)), T_p84=float(np.percentile(T, 84)),
                     T_err=float(1.2533 * 1.4826 * np.median(np.abs(T - np.median(T))) / np.sqrt(len(T))),
                     t0_med=float(np.median(per[:, 1])), grad_med=float(np.median(per[:, 4] / np.maximum(per[:, 3], 1e-9))),
                     stack_T=p[1], stack_t0=p[0], stack_grad=p[3] / p[2],
                     stack=[None if not np.isfinite(x) else float(x) for x in st],
                     stack_model=[float(x) for x in kern(p[0], p[1]) @ np.array(p[2:4])], per=per.tolist())
            res[key][view] = r
            print(f'{key} {view}: n {r["n"]}  T med {r["T_med"]:.0f} ± {r["T_err"]:.0f} (p16-84 {r["T_p16"]:.0f}-{r["T_p84"]:.0f})  '
                  f't0 {r["t0_med"]:.0f}  grad {r["grad_med"]:+.2f} | stack T {r["stack_T"]:.0f} t0 {r["stack_t0"]:.0f} '
                  f'grad {r["stack_grad"]:+.2f}', flush=True)
    json.dump(res, open(a.out, 'w'))


if __name__ == '__main__':
    main()
