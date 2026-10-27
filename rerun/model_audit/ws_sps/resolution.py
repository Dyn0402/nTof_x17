#!/usr/bin/env python3
"""resolution.py [--dir DIR] -- det4 position resolution of the waveform fit at 0 deg and 25.64 deg against the uRWELLs.

Inputs: res_<set>_<plateau>.pkl (resolution_cache.py: events with the raw front/back uRWELL clusters) and the
heldout_arm.py fits of them, res_<set>_<plateau>_<arm>.npz (per rows: eid, view, ..., p0 at column 9).
Per view: p0 regressed on (fx, fy, bx, by, 1) -- the interpolation weights and the alignment in one robust linear fit
(3x MAD trimming, iterated); the residual's core width from a FIXED-window Gaussian fit (+-1.2 mm; an adaptive clip
latches onto the mis-association tail, sps-spatial-resolution trap) and its robust sigma.  The same for the
amplitude centroid of the time-integrated charge (+-2 strips around the brightest) on the same events.
The residual contains the uRWELL pointing (~0.13-0.37 mm by zone and view, not subtracted here), so the absolute
numbers are upper bounds on det4; the 0 deg -> 25.64 deg change on the same reference is the angle cost.
"""
import argparse, glob, json, os, pickle
import numpy as np
from scipy.optimize import curve_fit


def rsig(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def core(res, win=1.2):
    r = res[np.abs(res) < win]
    h, b = np.histogram(r, bins=48, range=(-win, win)); c = 0.5 * (b[1:] + b[:-1])
    try:
        p, _ = curve_fit(lambda x, a, m, s, k: a * np.exp(-0.5 * ((x - m) / s) ** 2) + k, c, h, p0=[h.max(), 0, 0.2, 0])
        return abs(float(p[2]))
    except Exception:
        return np.nan


def align(y, U):
    A = np.c_[U, np.ones(len(U))]; ok = np.isfinite(y) & np.isfinite(U).all(1)
    for _ in range(6):
        c = np.linalg.lstsq(A[ok], y[ok], rcond=None)[0]
        r = y - A @ c
        s = rsig(r[ok]); ok = np.isfinite(y) & (np.abs(r - np.median(r[ok])) < 3 * max(s, 0.05))
    return y - A @ c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', default=os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_sps'))
    a = ap.parse_args(); out = {}
    for f in sorted(glob.glob(os.path.join(a.dir, 'res_*_*_*.npz'))):
        name = os.path.basename(f)[:-4]; cname = name.rsplit('_', 1)[0]
        E = pickle.load(open(os.path.join(a.dir, cname + '.pkl'), 'rb'))
        per = np.load(f, allow_pickle=True)['per']
        out[name] = {}
        for vi, v in enumerate('xy'):
            R = per[per[:, 1] == vi]
            eids = R[:, 0].astype(int); p0 = R[:, 9]
            U = np.array([E[e]['urw'] for e in eids])
            cen = []
            for e in eids:
                W = np.asarray(E[e][v]['W'], float); pos = np.asarray(E[e][v]['pos'], float)
                q = np.clip(W, 0, None).sum(1); k = int(np.argmax(q)); s = slice(max(k - 2, 0), k + 3)
                cen.append(float((q[s] * pos[s]).sum() / q[s].sum()))
            cen = np.array(cen)
            rw = align(p0, U); rc = align(cen, U)
            out[name][v] = dict(n=len(eids), wft_core_um=core(rw) * 1e3, wft_rsig_um=rsig(rw[np.abs(rw) < 3]) * 1e3,
                                cen_core_um=core(rc) * 1e3, cen_rsig_um=rsig(rc[np.abs(rc) < 3]) * 1e3,
                                wft_tail_gt1mm=float(np.mean(np.abs(rw) > 1.0)), cen_tail_gt1mm=float(np.mean(np.abs(rc) > 1.0)))
            o = out[name][v]
            print(f'{name:28s} {v}: n {o["n"]:5d}  wft core {o["wft_core_um"]:4.0f} rsig {o["wft_rsig_um"]:4.0f} um (>1mm {o["wft_tail_gt1mm"]:.3f}) | '
                  f'centroid core {o["cen_core_um"]:4.0f} rsig {o["cen_rsig_um"]:4.0f} um (>1mm {o["cen_tail_gt1mm"]:.3f})')
            np.save(os.path.join(a.dir, f'resid_{name}_{v}.npy'), np.c_[rw, rc])
    json.dump(out, open(os.path.join(a.dir, 'resolution.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
