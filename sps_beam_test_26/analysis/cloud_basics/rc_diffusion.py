#!/usr/bin/env python3
"""rc_diffusion.py -- does a continuous RC spread along the resistive strip
describe the head-on neighbour waveforms?

Physics.  Charge arriving at the anode at time u lands with a prompt lateral
footprint sigma_0.  On the view ACROSS the resistive strips (X) it stays put.
On the view ALONG them (Y) it then spreads as 1-D RC diffusion,
sigma^2(s) = sigma_0^2 + 2 D_rc s (s = time since arrival), and may drain with
a time constant tau_d.  The readout sees the change of the charge above each
strip, through the electronics.  With F_o(s) the fraction above strip offset
o (averaged over the track's position inside the centre strip):

    W_o(t) = sum_j S(t - s_j) [F_o(s_j) - F_o(s_{j-1})],     S = sum_o W_o

S, the all-strip sum, is (electronics x arrival profile) and is taken from the
data, so neither the impulse template nor the drift profile (attachment, gap,
v) is modelled.  Drift diffusion is NOT modelled: on the beam (run_71, wet,
front-loaded charge) it is small; on the bench it is not, so bench fits are
reported with that caveat and with drift diffusion folded in as a second
prompt term (sigma_0 there is "prompt + mean drift diffusion").

Fitted per view: sigma_0, D_rc, tau_d (Y);  sigma_0 only (X, D_rc = 0) and,
as a check, X with D_rc free.

    rc_diffusion.py
"""
import json, os, pickle, sys
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
from scipy.stats import trim_mean

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from width_vs_time import SOURCES, SAMPLE_NS, PITCH

STEP = 30.0
GRID = np.arange(-300.0, 3600.0, STEP)
OFFS = np.arange(-4, 5)
U = np.linspace(-PITCH / 2, PITCH / 2, 41)


THR_ADC = 60.0


def stack(ev, plane, tan_max=0.03, unbiased=None):
    """Head-on (offset, time) stack.  Default (unbiased): per event NOT
    normalised, aligned on the first crossing of THR_ADC by the 9-strip sum,
    plain mean -- a uniform track stacks to a flat top.  unbiased=False: the
    original stacking (normalised to the event's peak, aligned on 50 % of it,
    trimmed mean), which leans every event on its largest ionisation cluster
    and makes a uniform track look front-loaded (retracted 'attachment',
    FINDINGS §10).  The linear relations fitted here hold per event, so the
    RC fits survive either stacking; anything that assumes a charge profile
    needs the unbiased one."""
    if unbiased is None:
        unbiased = os.environ.get('CB_STACK', 'unbiased') == 'unbiased'
    rows = []
    for e in ev.values():
        if plane not in e or abs(e[f'tan_{plane}']) > tan_max:
            continue
        W = np.asarray(e[plane]['W'], float)
        ic = int(np.argmax(W.sum(1)))
        if ic < 4 or ic > len(W) - 5:
            continue
        s = W[ic - 4:ic + 5].sum(0)
        if unbiased:
            k = next((k for k in range(1, len(s)) if s[k - 1] < THR_ADC <= s[k]), None)
            if k is None or k < 3:
                continue
            t = (np.arange(len(s)) - (k - 1 + (THR_ADC - s[k - 1]) / (s[k] - s[k - 1]))) * SAMPLE_NS
            norm = 1.0
        else:
            ipk = int(np.argmax(s)); half = 0.5 * s[ipk]
            k = next((k for k in range(ipk, 0, -1) if s[k - 1] < half <= s[k]), None)
            if k is None or s[ipk] <= 0:
                continue
            t = (np.arange(len(s)) - (k - 1 + (half - s[k - 1]) / (s[k] - s[k - 1]))) * SAMPLE_NS
            norm = s[ipk]
        rows.append([np.interp(GRID, t, W[ic + o] / norm, left=np.nan, right=np.nan)
                     for o in OFFS])
    R = np.array(rows)
    M = np.full((len(OFFS), len(GRID)), np.nan)
    for i in range(len(OFFS)):
        for j in range(len(GRID)):
            col = R[:, i, j]; col = col[np.isfinite(col)]
            if len(col) > 0.6 * len(R):
                M[i, j] = col.mean() if unbiased else trim_mean(col, 0.2)
    if unbiased:
        M /= np.nanmax(np.nansum(M, axis=0))
    return len(R), M


def F(sig, o):
    """fraction above strip offset o for a Gaussian of width sig, averaged
    over the track position U inside the centre strip.  sig: (n,)"""
    z = 1.0 / (np.sqrt(2) * np.maximum(sig, 1e-3))[:, None]
    c = o * PITCH - U[None, :]
    return (0.5 * (erf((c + PITCH / 2) * z) - erf((c - PITCH / 2) * z))).mean(1)


def predict(p, S, n_lag, mode):
    sig0, D, taud = p
    s = np.arange(n_lag) * STEP
    sig = np.sqrt(sig0 ** 2 + 2 * D * s)
    drain = np.exp(-s / taud) if taud < 1e5 else np.ones_like(s)
    out = []
    for o in OFFS:
        Fo = F(sig, o) * drain
        dF = np.diff(np.r_[0.0, Fo])
        out.append(np.convolve(S, dF)[:len(S)])
    return np.array(out)


def fit(M, mode):
    ok_t = np.all(np.isfinite(M), axis=0)
    last = np.where(ok_t)[0].max()
    S = np.nan_to_num(M).sum(0)
    S[~ok_t] = 0
    sel = ok_t.copy()
    n_lag = len(GRID)

    def res(q):
        if mode == 'x0':
            p = (q[0], 0.0, 1e9)
        elif mode == 'x':
            p = (q[0], q[1], 1e9)
        else:
            p = (q[0], q[1], q[2])
        P = predict(p, S, n_lag, mode)
        return (P[:, sel] - M[:, sel]).ravel()

    x0 = {'x0': [0.35], 'x': [0.35, 1e-4], 'y': [0.35, 5e-4, 3000.0]}[mode]
    lo = {'x0': [0.02], 'x': [0.02, 0.0], 'y': [0.02, 0.0, 200.0]}[mode]
    hi = {'x0': [1.5], 'x': [1.5, 0.05], 'y': [1.5, 0.05, 1e6]}[mode]
    r = least_squares(res, x0, bounds=(lo, hi), x_scale='jac')
    J = r.jac
    cov = np.linalg.pinv(J.T @ J) * (r.fun ** 2).sum() / max(len(r.fun) - len(r.x), 1)
    return r.x, np.sqrt(np.diag(cov)), float(np.sqrt((r.fun ** 2).mean())), S, sel


def main():
    out = {}
    print('D_rc in mm^2/ns; the spread after 1 us is sqrt(2 D 1000) mm')
    for name, path in SOURCES.items():
        ev = pickle.load(open(path, 'rb'))
        out[name] = {}
        for plane in ('x', 'y'):
            n, M = stack(ev, plane)
            rows = {}
            for mode in (('x0', 'x') if plane == 'x' else ('x0', 'y')):
                x, e, rms, S, sel = fit(M, mode)
                rows[mode] = dict(p=x.tolist(), err=e.tolist(), rms=rms)
            out[name][plane] = dict(n=n, fits=rows, M=np.nan_to_num(M, nan=-99).tolist())
            fx = rows['x0']; fy = rows['x' if plane == 'x' else 'y']
            line = (f'{name:7s}{plane} n={n:4d} | prompt-only: sig0 {fx["p"][0]:.3f} rms {fx["rms"]:.4f} | '
                    f'+RC: sig0 {fy["p"][0]:.3f}±{fy["err"][0]:.3f} D {fy["p"][1]:.2e}')
            if plane == 'y':
                D = fy['p'][1]
                line += f' (1us spread {np.sqrt(2 * D * 1000):.2f} mm) taud {fy["p"][2]:.0f}'
            line += f' rms {fy["rms"]:.4f}'
            print(line, flush=True)
    json.dump(out, open(f'{HERE}/results/rc_diffusion.json', 'w'))


if __name__ == '__main__':
    main()
