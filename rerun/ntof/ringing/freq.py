#!/usr/bin/env python3
"""freq.py -- is each chamber's ringing frequency a fixed property?  Dominant frequency of the block common mode per
FEU, ToF bin and run (char_all.py caches), and the residual noise after the correction against the pedestal noise
that sets the 5-sigma thresholds.

    python -m rerun.ntof.ringing.freq
Out: $WORK_DATA/ntof_pass1/ana/ringing/freq.csv, noise_after.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from rerun.ntof.ringing import deflect as DF                            # noqa: E402
from rerun.ntof.ringing.char_all import load, FEU_OF, OUT, CACHE         # noqa: E402

BINS = [1, 2, 3, 5, 8, 12, 20]
NB = [0, 1, 2, 3, 5, 8, 12, 20, 30, 80]


def main():
    rows, nrows = [], []
    for f in sorted(CACHE.glob('*_000_feu*.npz')):
        run = '_'.join(f.stem.split('_')[:2])
        sub = '_'.join(f.stem.split('_')[2:4])
        feu = int(f.stem.split('feu')[1])
        A, ms, ev, nz = load(run, sub, '000', feu)
        N, ns, _ = A.shape
        c = np.median(A.reshape(N, ns, 8, 64), axis=3).mean(axis=2)
        c = c - c.mean(axis=1, keepdims=True)
        sp = np.abs(np.fft.rfft(c * np.hanning(ns), n=400, axis=1)) ** 2
        fr = np.fft.rfftfreq(400, d=60e-9) / 1e6
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = (ms >= lo) & (ms < hi) & (c.std(axis=1) > 10)
            if m.sum() < 10:
                continue
            S = sp[m].mean(0)
            rows.append(dict(run=run, sub=sub, feu=feu, view=FEU_OF[feu], lo=lo, hi=hi, n=int(m.sum()),
                             f_peak_MHz=float(fr[3:][np.argmax(S[3:])]),
                             f_event_med=float(np.median(fr[3:][np.argmax(sp[m][:, 3:], axis=1)])),
                             cm_rms=float(np.median(c[m].std(axis=1)))))
        if run == 'run_145':
            late = ms > 30
            P = np.stack([DF.plain_cm(a) for a in A[late][:300]])
            good = (P.std(axis=(0, 1)) / nz) < 2.5
            for lo, hi in zip(NB[:-1], NB[1:]):
                idx = np.flatnonzero((ms >= lo) & (ms < hi))[:150]
                if len(idx) < 10:
                    continue
                r_p, r_c = [], []
                for i in idx:
                    for W, acc in ((DF.plain_cm(A[i]), r_p), (DF.deflect(A[i], nz), r_c)):
                        z = W[:, good] / nz[good]
                        # robust per-event noise: MAD over all samples and good channels, pulses excluded by the MAD
                        acc.append(1.4826 * np.median(np.abs(z - np.median(z))))
                nrows.append(dict(view=FEU_OF[feu], lo=lo, hi=hi, n=len(idx), rel_noise_plain=float(np.median(r_p)),
                                  rel_noise_plain_p90=float(np.percentile(r_p, 90)),
                                  rel_noise_v1=float(np.median(r_c)), rel_noise_v1_p90=float(np.percentile(r_c, 90))))
    F = pd.DataFrame(rows)
    F.to_csv(OUT / 'freq.csv', index=False)
    Nn = pd.DataFrame(nrows)
    Nn.to_csv(OUT / 'noise_after.csv', index=False)
    pd.set_option('display.width', 250)
    print(F.pivot_table(index=['view', 'run'], columns='lo', values='f_peak_MHz').round(2).to_string())
    print(F.pivot_table(index=['view', 'run'], columns='lo', values='cm_rms').round(0).to_string())
    print('residual noise / pedestal noise (robust, per event, good channels): plain -> v1 (median, p90)')
    Nn['s'] = [f'{a:.2f}({b:.2f})->{c:.2f}({d:.2f})' for a, b, c, d in
               zip(Nn.rel_noise_plain, Nn.rel_noise_plain_p90, Nn.rel_noise_v1, Nn.rel_noise_v1_p90)]
    print(Nn.pivot_table(index='view', columns='lo', values='s', aggfunc='first').to_string())


if __name__ == '__main__':
    main()
