#!/usr/bin/env python3
"""gate_artefact.py — the F4 'implied v falls with |tan|' numbers came from 03_angles.py: median(w / tan_ref) per |tan_ref|
bin over the slope_reliable events only (|tan_FIT| >= 0.08).  In the 0.08-0.14 reference bin that gate drops the fits
that came out shallower than 0.08, i.e. a cut on the measured quantity, which inflates the shallow bin.  This script
recomputes the implied v (a) as F4 did, (b) without the gate, (c) without the gate and with the bundle w0 removed, signed
bins averaged (= S·v, the symmetric scale), with bootstrap errors.  Writes ws_widthtilt/gate_artefact.json."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_split import load, ROOT

BINS = [(0.08, 0.14), (0.14, 0.20), (0.20, 0.28), (0.28, 0.45)]
RNG = np.random.default_rng(7)


def boot(f, *arrs, n=300):
    vals = []
    L = len(arrs[0])
    for _ in range(n):
        i = RNG.integers(0, L, L)
        vals.append(f(*[a[i] for a in arrs]))
    return float(np.nanstd(vals))


def sym(t, wc):
    p, n_ = t > 0, t < 0
    if p.sum() < 5 or n_.sum() < 5:
        return np.nan
    return 0.5 * (np.median(wc[p] / t[p]) + np.median(wc[n_] / t[n_]))


def main():
    out = {'bins': BINS}
    for det in ['det2', 'det3', 'det4', 'det6', 'det7']:
        b, T = load(det); v = b['v_drift']
        for p in 'xy':
            df = T[p]; t = df.t.to_numpy(); w = df.w.to_numpy(); a = np.abs(t)
            tf = (w - b['w0'][p]) / (b['kw'][p] * v)
            gated, ungated, symm, symm_e = [], [], [], []
            for lo, hi in BINS:
                m = (a >= lo) & (a < hi)
                g = m & (np.abs(tf) >= 0.08)
                gated.append(float(np.median(w[g] / t[g])))
                ungated.append(float(np.median(w[m] / t[m])))
                wc = w[m] - b['w0'][p]
                symm.append(float(sym(t[m], wc))); symm_e.append(boot(sym, t[m], wc))
            out[f'{det}_{p}'] = dict(v=v, kw=b['kw'][p], gated=gated, ungated=ungated, sym=symm, sym_err=symm_e,
                                     n=[int(((a >= lo) & (a < hi)).sum()) for lo, hi in BINS])
            f = lambda xs: ' '.join(f'{x:5.2f}' for x in xs)
            print(f'{det} {p} v={v:5.2f}  F4-gated {f(gated)} | ungated {f(ungated)} | sym,w0-free {f(symm)} ± {f(symm_e)}')
    json.dump(out, open(f'{ROOT}/ws_widthtilt/gate_artefact.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
