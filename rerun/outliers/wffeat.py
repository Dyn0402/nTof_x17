#!/usr/bin/env python3
"""wffeat.py <det> -- waveform-level features for the events cached by display.extract (QA, not geometry).

Per event and view, on the full plane (pedestal + plain CM, strip max over the window, in noise units):
  w<v>_snr_ref    largest strip SNR inside the M3 track's footprint: strips within +-BAND mm of the segment
                  [ref_mesh, ref_mesh + tan_ref * GAP] (where a straight track through the gap leaves charge)
  w<v>_qfrac_ref  share of the plane's above-threshold charge (sum of strip maxima, SNR > THR) inside that footprint
  w<v>_ncl        clusters of above-threshold strips (gap > 2 pitches splits) holding > 15 % of the largest one
  w<v>_ncl_far    of those, how many sit > 5 mm outside the footprint
  w<v>_wid1       width [mm] of the largest cluster; w<v>_q1 its charge [ADC]; w<v>_dfoot1 its distance to the footprint
  w<v>_nsatstrip  strips that reach the ADC rail (> 3500)
  w<v>_pos1       charge-weighted centre of the largest cluster [mm]; w<v>_cls JSON [q, lo, hi, centre] of the top 8
"""
import argparse, json, os, pickle, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from olib import BASE, load  # noqa: E402
import display as D  # noqa: E402

THR, BAND, GAP = 6.0, 2.5, 30.0


def feats(W, noise, pos, pref, tref):
    ok = np.isfinite(pos); o = np.argsort(np.where(ok, pos, np.inf))[:ok.sum()]
    P = pos[o]; M = W[o].astype(np.float32).max(1); N = np.maximum(noise[o].astype(np.float32), 3.0)
    snr = M / N; on = snr > THR
    lo, hi = sorted((pref, pref + tref * GAP)); lo -= BAND; hi += BAND
    inb = (P >= lo) & (P <= hi)
    f = dict(snr_ref=float(snr[inb].max()) if inb.any() else 0.0,
             qfrac_ref=float(M[on & inb].sum() / max(M[on].sum(), 1e-9)) if on.any() else 0.0,
             nsatstrip=int((M > 3500).sum()))
    if not on.any():
        f.update(ncl=0, ncl_far=0, wid1=0.0, q1=0.0, dfoot1=np.nan, pos1=np.nan, cls='[]'); return f
    idx = np.flatnonzero(on); pitch = np.median(np.diff(P))
    cl = np.split(idx, np.flatnonzero(np.diff(P[idx]) > 2.5 * pitch) + 1)
    info = []
    for c in cl:
        a, b = P[c].min(), P[c].max(); q = M[c].sum()
        dfoot = 0.0 if (b >= lo and a <= hi) else min(abs(a - hi), abs(b - lo))
        cen = float((P[c] * M[c]).sum() / max(M[c].sum(), 1e-9))
        info.append((q, a, b, dfoot, len(c), cen))
    info.sort(key=lambda t: -t[0])
    big = [t for t in info if t[0] > 0.15 * info[0][0] and t[4] >= 2]
    f.update(ncl=len(big), ncl_far=sum(t[3] > 5 for t in big), wid1=float(info[0][2] - info[0][1]),
             q1=float(info[0][0]), dfoot1=float(info[0][3]), pos1=info[0][5],
             cls=json.dumps([[round(float(t[0]), 1), round(float(t[1]), 2), round(float(t[2]), 2), round(t[5], 2)]
                             for t in info[:8]]))
    return f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('det'); a = ap.parse_args()
    d = load(a.det).set_index('event_id')
    wf = pickle.load(open(os.path.join(BASE, f'wf_{a.det}.pkl'), 'rb'))
    pm = D.posmap(a.det); rows = []
    for e, ev in wf.items():
        if e not in d.index:
            continue
        r = d.loc[e]; out = {'event_id': e}
        for v in 'xy':
            if v not in ev or not np.isfinite(r[f'ref_mesh_{v}']):
                continue
            for k, val in feats(ev[v], ev[f'noise_{v}'], pm[v], r[f'ref_mesh_{v}'], r[f'ref_tan_{v}']).items():
                out[f'w{v}_{k}'] = val
        rows.append(out)
    F = pd.DataFrame(rows); F.to_parquet(os.path.join(BASE, f'wffeat_{a.det}.parquet')); print(a.det, len(F))


if __name__ == '__main__':
    main()
