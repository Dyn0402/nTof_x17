#!/usr/bin/env python3
"""ladder_profile.py -- template-free charge vs drift depth on the 25.64 deg beam ladder.

run_63 rotated mount (Ar/CF4/iso 88/10/2, ZS 4 sigma, resist 769.8 V): the Y view
carries the drift ladder (kernel_lib: -198 ns/mm), so each Y strip collects the
charge of one depth slice: 30 mm of gap spans 30 * tan(25.64 deg) = 14.4 mm =
18.4 strips.  Attachment removes a fraction exp(-z / lambda) of the charge that
starts at depth z, so the per-strip charge falls from the mesh end of the ladder
to the cathode end.  No attachment = flat between the two ends.

No template, no time alignment, nothing from hit times.  Per event, each Y strip
is placed at u = position - pY (the uRWELL prediction, one offset per event) and
contributes
  A    = its peak sample (censoring-immune whenever the strip fired at all;
         longitudinal diffusion over 3 cm is ~30 ns against 180 ns shaping, so it
         does not lower the peak with depth),
  Q    = the sum of its samples (ZS-censored: low samples are absent),
  tpk  = its peak time (used ONLY to orient the ladder -- the mesh end is the
         early end -- and to drop strips whose pulse runs off the window).
The stack is the median of each per bin of u, and the fraction of events with
the strip present.  Strips with peak sample >= LAST_OK are flagged truncated.

    ladder_profile.py [--arms rot_d425 rot_d325 rot_d225]
Output: results/ladder_profile.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'angled_kernel'))
import kernel_lib as K                                       # noqa: E402

UBINS = np.arange(-20.0, 20.01, 0.78)
LAST_OK = 60          # a peak at sample >= 60 (3.6 us) may be cut by the 64-sample window


def profile(name):
    d = K.load_arm(name)
    pY = d['pY']
    U, A, Q, T, EV = [], [], [], [], []
    for eid, (pos, W, _ch) in d['y'].items():
        if not np.isfinite(pY[eid]):
            continue
        U.append(pos - pY[eid]); A.append(W.max(1)); Q.append(W.sum(1))
        T.append(W.argmax(1)); EV.append(np.full(len(pos), eid))
    U, A, Q, T, EV = map(np.concatenate, (U, A, Q, T, EV))
    evs = np.unique(EV)
    nev = len(evs)
    rows = []
    for lo, hi in zip(UBINS[:-1], UBINS[1:]):
        m = (U >= lo) & (U < hi)
        ok = m & (T < LAST_OK)
        # censoring-proof: one value per event, a strip that did not fire counts 0
        # (two strips of one event in a bin: keep the larger -- bins are one pitch wide)
        full = np.zeros(nev)
        ei = np.searchsorted(evs, EV[m])
        np.maximum.at(full, ei, np.where(T[m] < LAST_OK, A[m], 0.0))
        qs = {f'A_q{q}': float(np.quantile(full, q / 100)) for q in (50, 60, 70, 80, 90)}
        rows.append(dict(u=float(0.5 * (lo + hi)), n=int(m.sum()), occ=float(m.sum() / nev),
                         A_med=float(np.median(A[ok])) if ok.sum() > 50 else np.nan,
                         A_mean=float(np.mean(A[ok])) if ok.sum() > 50 else np.nan,
                         Q_med=float(np.median(Q[ok])) if ok.sum() > 50 else np.nan,
                         A_meanall=float(full.mean()),
                         t_med=float(np.median(T[m]) * K.SNS) if m.sum() > 50 else np.nan,
                         trunc=float((T[m] >= LAST_OK).mean()) if m.sum() else np.nan, **qs))
    # interior: from the strip at the profile maximum (the first full depth slice past
    # the mesh edge) down to the last bin with < 5 % window truncation
    u = np.array([r['u'] for r in rows]); ma = np.array([r['A_meanall'] for r in rows])
    tr = np.array([np.nan_to_num(r['trunc'], nan=1.0) for r in rows])
    tm = np.array([r['t_med'] for r in rows])
    ipk = int(np.nanargmax(ma))
    lo = ipk
    while lo - 1 >= 0 and tr[lo - 1] < 0.05 and ma[lo - 1] > 0.3 * ma[ipk]:
        lo -= 1
    sel = np.arange(lo, ipk + 1)
    # per-event charge on the interior grid (strip not fired = 0) for the bootstrap
    G = np.zeros((nev, len(sel)))
    for j, b in enumerate(sel):
        m = (U >= UBINS[b]) & (U < UBINS[b + 1]) & (T < LAST_OK)
        np.maximum.at(G[:, j], np.searchsorted(evs, EV[m]), A[m])
    tz = 1.0 / np.tan(np.radians(K.TILT_DEG))
    rng = np.random.default_rng(5)
    fit = {}
    for key, x in (('per_mm', -u[sel] * tz), ('per_ns', tm[sel])):
        f = lambda g: -np.polyfit(x, np.log(g.mean(0)), 1)[0]
        bs = [f(G[rng.integers(0, nev, nev)]) for _ in range(300)]
        fit[key] = [float(f(G)), float(np.std(bs))]
    fit['u_range'] = [float(u[sel[0]]), float(u[sel[-1]])]
    fit['t_range'] = [float(tm[sel[-1]]), float(tm[sel[0]])]
    fit['depth_span_mm'] = float((u[sel[-1]] - u[sel[0]]) * tz)
    fit['survival_over_span'] = float(G[:, 0].mean() / G[:, -1].mean())
    return dict(arm=name, drift_V=d['meta']['drift_V'], n_events=nev, rows=rows, fit=fit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=['rot_d425', 'rot_d325', 'rot_d225'])
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    out = {}
    for name in a.arms:
        r = profile(name)
        out[name] = r
        f = r['fit']
        print(f'== {name}: interior u {f["u_range"]}, t {f["t_range"]} ns, depth span '
              f'{f["depth_span_mm"]:.1f} mm, survival {f["survival_over_span"]:.3f};  loss '
              f'{f["per_mm"][0]:.4f}±{f["per_mm"][1]:.4f} /mm, '
              f'{f["per_ns"][0] * 1e4:.2f}±{f["per_ns"][1] * 1e4:.2f} e-4/ns')
        if not a.verbose:
            continue
        print(f'== {name}  drift {r["drift_V"]} V  events {r["n_events"]}')
        print('   u[mm]   occ   A_med  Q_med  t_med[ns] trunc')
        for row in r['rows']:
            if row['occ'] > 0.05:
                print(f'  {row["u"]:6.1f}  {row["occ"]:.2f}  {row["A_med"]:6.0f} {row["Q_med"]:6.0f}'
                      f'  {row["t_med"]:6.0f}   {row["trunc"]:.2f}')
    json.dump(out, open(os.path.join(HERE, 'results', 'ladder_profile.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
