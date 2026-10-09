#!/usr/bin/env python3
"""neighbour_vs_time.py -- robust head-on neighbour ratios against arrival time.

Per event (view normal to the track): centre strip = max integrated charge,
time aligned on the 50 % rise of the summed waveform, resampled to a 30 ns
grid.  Per (offset, time bin) the 20 %-trimmed mean over events of W(offset,
t) / max_t W(0, t) (absent = 0).  Reported: r1(t) = mean(+-1)/centre,
r2(t) = mean(+-2)/centre, and sigma_eq(t), the Gaussian width that gives the
measured r1 for a track uniformly placed in the centre strip (the prompt-cloud
equivalent; it absorbs any neighbour charge, resistive or not).

    neighbour_vs_time.py [--tan-max 0.03]
"""
import argparse, json, os, pickle, sys
import numpy as np
from scipy.special import erf
from scipy.stats import trim_mean
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from width_vs_time import SOURCES, SAMPLE_NS, PITCH

GRID = np.arange(-300.0, 3600.0, 30.0)


def r1_of_sigma(sig):
    u = np.linspace(-PITCH / 2, PITCH / 2, 201)          # track offset in the centre strip
    def frac(c):
        z = 1 / (np.sqrt(2) * sig)
        return 0.5 * (erf((c + PITCH / 2 - u) * z) - erf((c - PITCH / 2 - u) * z))
    f0, fp, fm = frac(0), frac(PITCH), frac(-PITCH)
    # centre strip chosen as the max: the stacked ratio is mean(nbr)/mean(centre)
    return 0.5 * (fp.mean() + fm.mean()) / f0.mean()


SIG = np.linspace(0.02, 1.5, 300)
R1 = np.array([r1_of_sigma(s) for s in SIG])


def sigma_eq(r):
    return float(np.interp(r, R1, SIG, left=np.nan, right=np.nan))


def profiles(events, plane, tan_max):
    rows = []
    for ev in events.values():
        if plane not in ev or abs(ev[f'tan_{plane}']) > tan_max:
            continue
        W = np.asarray(ev[plane]['W'], float)
        q = W.sum(axis=1)
        ic = int(np.argmax(q))
        if ic < 2 or ic > len(q) - 3:
            continue
        s = W[ic - 2:ic + 3].sum(axis=0)
        ipk = int(np.argmax(s))
        half = 0.5 * s[ipk]
        k = next((k for k in range(ipk, 0, -1) if s[k - 1] < half <= s[k]), None)
        if k is None:
            continue
        t50 = (k - 1 + (half - s[k - 1]) / (s[k] - s[k - 1])) * SAMPLE_NS
        t = np.arange(W.shape[1]) * SAMPLE_NS - t50
        peak = W[ic].max()
        if peak <= 0:
            continue
        rows.append(np.array([np.interp(GRID, t, W[ic + o], left=np.nan, right=np.nan) / peak
                              for o in (-2, -1, 0, 1, 2)]))
    return np.array(rows)          # (n, 5, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tan-max', type=float, default=0.03)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  'results', 'neighbour_vs_time.json'))
    a = ap.parse_args()
    out = {}
    for name, path in SOURCES.items():
        ev = pickle.load(open(path, 'rb'))
        out[name] = {}
        for plane in ('x', 'y'):
            R = profiles(ev, plane, a.tan_max)
            m = np.full((5, len(GRID)), np.nan)
            for o in range(5):
                for j in range(len(GRID)):
                    col = R[:, o, j]
                    col = col[np.isfinite(col)]
                    if len(col) > 0.5 * len(R):
                        m[o, j] = trim_mean(col, 0.2)
            c = m[2]
            ok = c > 0.1
            r1 = np.where(ok, 0.5 * (m[1] + m[3]) / c, np.nan)
            r2 = np.where(ok, 0.5 * (m[0] + m[4]) / c, np.nan)
            se = [sigma_eq(x) if np.isfinite(x) else np.nan for x in r1]
            out[name][plane] = dict(n=len(R), t=GRID.tolist(), centre=c.tolist(),
                                    m=m.tolist(), r1=r1.tolist(), r2=r2.tolist(), sigma_eq=se)
            ipk = int(np.nanargmax(c))
            jl = int(np.argmax(c > 0.3))          # leading edge, 30 % of the centre peak
            print(f'{name:7s} {plane} n={len(R):4d} | lead(30%) t={GRID[jl]:+5.0f} r1 {r1[jl]:.3f} '
                  f'r2 {r2[jl]:.3f} sig_eq {se[jl]:.3f} | peak t={GRID[ipk]:+4.0f} r1 {r1[ipk]:.3f} '
                  f'r2 {r2[ipk]:.3f} sig_eq {se[ipk]:.3f}')
    json.dump(out, open(a.out, 'w'), default=lambda x: None if x != x else x)


if __name__ == '__main__':
    main()
