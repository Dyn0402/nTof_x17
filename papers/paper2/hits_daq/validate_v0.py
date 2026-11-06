#!/usr/bin/env python3
"""validate_v0.py <xw window pickles...> [--out results/v0_validation.json] -- does V0 reproduce our production hits?

V0 (emulate.dream) is the emulation of WaveformAnalyzer.cpp, the hit finder that made combined_hits.  On the same
strips of the same events (extract_windows.py stores the production hits of each window's channels) this compares:
  presence   strips hit in production vs in V0 (both / production only / V0 only), scanning the gate-noise factor
  time       V0 time - production 'time' (30 % leading edge) and V0 peak time - production 'time_of_max', per strip
             after removing each plane's median offset (the production times carry the ftst clock phase)
  amplitude  V0 / production 'amplitude'
and the plane's ladder slope from production times against V0 times (the quantity every tier is built on).
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emulate as E  # noqa: E402
from tiers import ladder_u  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickles', nargs='+')
    ap.add_argument('--out')
    ap.add_argument('--nmax', type=int, default=3000)
    a = ap.parse_args()
    planes = []
    for f in a.pickles:
        planes += pickle.load(open(f, 'rb'))['planes']
    planes = [p for p in planes if p.get('hits') is not None and len(p['hits'].get('time', []))][:a.nmax]
    # production's gate sigma per channel: rms_gate = (5-sample boxcar peak) / significance, from the EVEN planes;
    # every count below is scored on the ODD planes only, so the per-channel gate is not tuned on the planes it scores
    calib, planes = planes[0::2], planes[1::2]
    perch = {}
    for p in calib:
        idx = {int(c): i for i, c in enumerate(p['ch'])}
        for c, s in zip(p['hits']['channel'], p['hits']['significance']):
            i = idx.get(int(c))
            if i is None or s <= 0 or p['noise'][i] <= 0:
                continue
            g = np.convolve(p['W'][i].astype(float), np.ones(5) / 5, mode='same').max()
            perch.setdefault((p['view'], int(c)), []).append(g / s / p['noise'][i])
    gmed = {k: float(np.median(v)) for k, v in perch.items()}
    gall = float(np.median(np.concatenate([v for v in perch.values()]))) if perch else E.GATE_K

    def gate_of(p, gk):
        if gk != 'per_channel':
            return gk
        return np.array([gmed.get((p['view'], int(c)), gall) for c in p['ch']])

    res = dict(n_planes=len(planes), n_calib_planes=len(calib), gate_k_measured=gall,
               n_channels_gate=len(gmed), gate=[])
    for gk in (0.447, 0.55, 0.62, gall, 0.75, 'per_channel'):
        both = ponly = vonly = 0
        for p in planes:
            h = E.dream(p['W'].astype(float), p['noise'].astype(float), gate_k=gate_of(p, gk))
            prod = set(int(c) for c in p['hits']['channel'])
            emu = set(int(c) for c, t in zip(p['ch'], h['t']) if np.isfinite(t))
            both += len(prod & emu); ponly += len(prod - emu); vonly += len(emu - prod)
        res['gate'].append(dict(gate_k=gk, both=both, prod_only=ponly, v0_only=vonly,
                                agree=both / max(1, both + ponly + vonly)))
        print(f'gate_k {gk}: both {both}, production only {ponly}, V0 only {vonly}')
    gk = 'per_channel'  # the faithful emulation; the scalar scan shows how much the per-channel detail matters
    dt, dtm, ra, ur = [], [], [], []
    for p in planes:
        h = E.dream(p['W'].astype(float), p['noise'].astype(float), gate_k=gate_of(p, gk))
        idx = {int(c): i for i, c in enumerate(p['ch'])}
        H = p['hits']
        # one production hit per channel: the largest
        sel = {}
        for j, c in enumerate(H['channel']):
            c = int(c)
            if c in idx and (c not in sel or H['amplitude'][j] > H['amplitude'][sel[c]]):
                sel[c] = j
        d1, d2, rr = [], [], []
        tp = np.full(len(p['ch']), np.nan)
        ap_ = np.full(len(p['ch']), np.nan)
        for c, j in sel.items():
            i = idx[c]
            tp[i], ap_[i] = H['time'][j], H['amplitude'][j]
            if np.isfinite(h['t'][i]):
                d1.append(h['t'][i] - H['time'][j])
                rr.append(h['a'][i] / H['amplitude'][j] if H['amplitude'][j] > 0 else np.nan)
        if len(d1) >= 3:
            d1 = np.array(d1)
            dt += list(d1 - np.median(d1))
            ra += rr
        pos = p['pos'].astype(float)
        u_prod, _ = ladder_u(pos, tp, ap_)
        u_emu, _ = ladder_u(pos, h['t'], h['a'])
        if np.isfinite(u_prod) and np.isfinite(u_emu) and abs(p['w']) > 0.002:
            ur.append(u_emu / u_prod)
    dt, ra, ur = np.array(dt), np.array(ra), np.array(ur)
    mad = lambda x: float(1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))))  # noqa: E731
    res.update(scored_gate_k=gk, time_resid_ns=dict(median=float(np.median(dt)), mad=mad(dt), n=len(dt)),
               amp_ratio=dict(median=float(np.nanmedian(ra)), mad=mad(ra)),
               ladder_ratio_emu_over_prod=dict(median=float(np.median(ur)), mad=mad(ur), n=len(ur)))
    print(json.dumps({k: v for k, v in res.items() if k != 'gate'}, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
