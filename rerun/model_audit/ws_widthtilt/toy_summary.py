#!/usr/bin/env python3
"""toy_summary.py [dir] — closure table of the toy runs: median w_fit / w_true per |tan| bin, symmetric over signs,
plus the even term (µm/ns) and the fraction of fits that picked the wide profile footprint.  Writes toy_summary.json."""
import glob, json, os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_widthtilt/eos')
BINS = [(0.10, 0.14), (0.14, 0.20), (0.20, 0.28), (0.28, 0.45)]


def main():
    out = {}
    for f in sorted(glob.glob(f'{D}/toy_*.npz')):
        tag = os.path.basename(f)[4:-4]
        R = np.load(f)['per']
        res = {}
        for vi, p in enumerate('xy'):
            m = R[:, 1] == vi
            t, wt, wf, sig = R[m, 2], R[m, 3], R[m, 4], R[m, 7]
            S, E, nar, N = [], [], [], []
            for lo, hi in BINS:
                a = np.abs(t); b = (a >= lo) & (a < hi)
                bp, bn = b & (t > 0), b & (t < 0)
                if bp.sum() < 5 or bn.sum() < 5:
                    S.append(np.nan); E.append(np.nan); nar.append(np.nan); N.append(int(b.sum())); continue
                rp, rn = np.median(wf[bp] / wt[bp]), np.median(wf[bn] / wt[bn])
                S.append(float(0.5 * (rp + rn))); E.append(float(0.5 * (np.median(wf[bp] - wt[bp]) + np.median(wf[bn] - wt[bn]))))
                nar.append(float(np.mean(sig[b] > 0.40))); N.append(int(b.sum()))
            # all-angle robust scale + per-event spread
            r = wf / wt
            res[p] = dict(S=S, E=E, wide_frac=nar, n=N, scale_all=float(np.median(r)),
                          rsig_deg=float(1.4826 * np.median(np.abs(np.degrees(np.arctan(wf / wt * t) - np.arctan(t))))))
            fmt = lambda xs: ' '.join(f'{x:6.3f}' for x in xs)
            print(f'{tag:14s} {p}  S {fmt(S)}   E {fmt(E)}   wide {fmt(nar)}  n {N}')
        out[tag] = res
    json.dump(out, open(f'{D}/../toy_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
