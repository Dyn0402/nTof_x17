#!/usr/bin/env python3
"""railstats.py -- where the ADC rail actually sits in the det3 golden-row waveforms that satfit.py fits (pedestal +
plain CM subtracted, wft.io), against the fit's fixed censor level wft.model.SAT = 3550:
  strip maxima of every strip > 3000 ADC (the pile-up IS the rail: 4095 - pedestal - CM, per channel)
  per clipped strip: plateau length, plateau sag (max - min over the plateau: the CM moving under a clipped sample),
  samples INSIDE a plateau (both time neighbours >= 3550) that sit below 3550 -> fitted as real measurements
  post-clip undershoot: the strip's minimum after its plateau, in ADC
-> ~/x17/cosmic_bench/satstudy/railstats.json"""
import json, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import satfit as S  # noqa: E402

SAT = 3550.0


def main():
    D = pickle.load(open(os.path.join(S.OUT, 'wf_det3.pkl'), 'rb'))
    mx, plen, sag, under, nsand, nplat, under_ref = [], [], [], [], 0, 0, []
    for e, ev in D['wf'].items():
        for v in 'xy':
            if v not in ev:
                continue
            W = ev[v].astype(np.float32); m = W.max(1)
            mx += list(m[m > 3000])
            for i in np.flatnonzero(m > 3600):
                w = W[i]; hi = np.flatnonzero(w >= SAT)
                a, b = hi.min(), hi.max()
                seg = w[a:b + 1]
                plen.append(int(b - a + 1)); sag.append(float(seg.max() - seg.min())); nplat += int(b - a + 1)
                nsand += int((seg < SAT).sum())
                if b + 1 < len(w):
                    under.append(float(w[b + 1:].min()))
            for i in np.flatnonzero((m > 2000) & (m < 3400)):
                w = W[i]; k = int(np.argmax(w))
                if k + 1 < len(w):
                    under_ref.append(float(w[k + 1:].min() / m[i]))
    q = lambda a: {str(p): round(float(np.percentile(a, p)), 3) for p in (10, 25, 50, 75, 90)}
    h, ed = np.histogram(mx, bins=np.arange(3000, 4201, 25))
    out = dict(n_events=len(D['wf']), sat_level=SAT, hist_edges=ed.tolist(), hist_counts=h.tolist(),
               n_clipped_strips=len(plen), plateau_len_q=q(plen), plateau_sag_q=q(sag),
               n_plateau_samples=nplat, n_inside_plateau_below_sat=nsand,
               undershoot_after_clip_q=q(under), undershoot_ratio_unsat_q=q(under_ref),
               )
    json.dump(out, open(os.path.join(S.OUT, 'railstats.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith('hist')}, indent=1))


if __name__ == '__main__':
    main()
