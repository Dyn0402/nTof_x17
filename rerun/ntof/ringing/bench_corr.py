#!/usr/bin/env python3
"""bench_corr.py -- every waveform correction variant on every FEU, on the same events (open items, 2026-11-23).

    python -m rerun.ntof.ringing.bench_corr [--run run_145 --sub stat090_0000 --tag 000] [--per-bin 200]

Uses the char_all.py cache.  Per FEU and ToF bin, for each variant: the median (90th percentile) number of channels
below -5 sigma and above +5 sigma (5 sigma = the pedestal-run noise), the residual rms, and two signal tests:
  keep_late   late (>= 30 ms) events, channels with a clean pulse (> 8 sigma, never < -3 sigma) that are well behaved
              (late rms / noise < 2.5): corrected positive charge / plain-CM positive charge.
  inject      a late event's pulse channels (their plain-CM waveform) are added to an event of this ToF bin; the
              recovered charge is sum(corr(event + pulse) - corr(event)) over the pulse's positive samples, divided by
              the injected charge.  This measures the charge a real track keeps when it sits on top of the ringing.
Out: $WORK_DATA/ntof_pass1/ana/ringing/bench_<run>_<sub>_<tag>.csv
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from rerun.ntof.ringing import deflect as DF          # noqa: E402
from rerun.ntof.ringing.char_all import load, FEU_OF, OUT, cache_path   # noqa: E402

BINS = [0, 1, 1.25, 1.5, 2, 3, 5, 8, 12, 20, 30, 80]
VARIANTS = {
    'plain (pass 1)': lambda A, nz: DF.plain_cm(A),
    'deflect v1': lambda A, nz: DF.deflect(A, nz),
    'v2 rank1': lambda A, nz: DF.deflect2(A, nz, rank=1),
    'v2 rank1+dc': lambda A, nz: DF.deflect2(A, nz, rank=1, dc=True),
    'v2 rank2': lambda A, nz: DF.deflect2(A, nz, rank=2),
    'v2 rank2+dc': lambda A, nz: DF.deflect2(A, nz, rank=2, dc=True),
    'v4': lambda A, nz: DF.deflect4(A, nz),
    'v1+group_dc': lambda A, nz: DF.group_dc(A, nz),
    'v4+2nd sv2': lambda A, nz: DF.deflect4(A, nz, second=True, sv_nsig=2.0),
    'v4+2nd sv3': lambda A, nz: DF.deflect4(A, nz, second=True, sv_nsig=3.0),
    'v4 pm4': lambda A, nz: DF.deflect4(A, nz, pmask=4.0),
    'v3 sv2': lambda A, nz: DF.deflect3(A, nz, sv_nsig=2.0),
    'v3 sv3': lambda A, nz: DF.deflect3(A, nz, sv_nsig=3.0),
    'v3 sv1.5': lambda A, nz: DF.deflect3(A, nz, sv_nsig=1.5),
}


def pulses(Rl, nz, good, min_adc=100.0, min_run=3):
    """list of (channel indices, (ns, k) pulse waveforms) from late plain-CM events: track-like clusters, i.e. runs of
    >= min_run adjacent well-behaved channels with a clean pulse (> 8 sigma and > min_adc, never < -3 sigma), +-1
    channel of the cluster's neighbours included (their shared charge)."""
    out = []
    for e in Rl:
        z = e / nz
        hit = (z.max(0) > 8) & (e.max(0) > min_adc) & (z.min(0) > -3) & good
        k = np.flatnonzero(hit)
        if not len(k):
            continue
        runs = np.split(k, np.flatnonzero(np.diff(k) > 1) + 1)
        for r in runs:
            if min_run <= len(r) <= 40 and r[0] // 64 == r[-1] // 64:
                ch = np.arange(max(r[0] - 1, (r[0] // 64) * 64), min(r[-1] + 2, (r[0] // 64 + 1) * 64))
                out.append((ch, e[:, ch].copy()))
    return out


def run(run_, sub, tag, per_bin, variants):
    rng = np.random.default_rng(1)
    rows = []
    for feu, name in FEU_OF.items():
        if not cache_path(run_, sub, tag, feu).exists():
            continue
        A, ms, ev, nz = load(run_, sub, tag, feu)
        N, ns, _ = A.shape
        P = np.stack([DF.plain_cm(a) for a in A])
        lt = np.flatnonzero(ms > 30)
        good = (P[lt].std(axis=(0, 1)) / nz) < 2.5
        PL = pulses(P[lt], nz, good)
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            idx = np.flatnonzero((ms >= lo) & (ms < hi))
            if len(idx) < 10:
                continue
            idx = rng.choice(idx, min(per_bin, len(idx)), replace=False)
            inj = [PL[k] for k in rng.integers(0, len(PL), len(idx))]
            for vn in variants:
                f = VARIANTS[vn]
                nneg, npos, rms, rec, qi, qr = [], [], [], [], [], []
                for i, (ch, pw) in zip(idx, inj):
                    W = f(A[i], nz)
                    z = W / nz
                    nneg.append(int((z.min(0) < -5).sum()))
                    npos.append(int((z.max(0) > 5).sum()))
                    rms.append(float(np.sqrt((W[:, good] ** 2).mean())))
                    A2 = A[i].copy()
                    A2[:, ch] += pw
                    W2 = f(A2, nz)
                    pos = pw > 0
                    qr.append(float(((W2[:, ch] - W[:, ch]) * pos).sum()))
                    qi.append(float((pw * pos).sum()))
                    rec.append(qr[-1] / max(qi[-1], 1.0))
                rows.append(dict(feu=feu, view=name, lo=lo, hi=hi, variant=vn, n=len(idx),
                                 neg5=np.median(nneg), neg5_p90=np.percentile(nneg, 90), pos5=np.median(npos),
                                 rms_good=np.median(rms), inject_med=np.median(rec), inject_p10=np.percentile(rec, 10),
                                 inject_p02=np.percentile(rec, 2), inject_sum=np.sum(qr) / np.sum(qi)))
        # late signal keep on real late pulses
        sel = lt[:600]
        z0 = P[sel] / nz
        sig = (z0.max(1) > 8) & (z0.min(1) > -3) & good[None, :]
        q0 = np.clip(P[sel], 0, None).sum(1)[sig]
        for vn in variants:
            W = np.stack([VARIANTS[vn](A[i], nz) for i in sel])
            q1 = np.clip(W, 0, None).sum(1)[sig]
            rows.append(dict(feu=feu, view=name, lo=30, hi=80, variant=vn, n=int(sig.sum()), keep_late=np.median(q1 / q0),
                             keep_late_p10=np.percentile(q1 / q0, 10)))
        print(f'{name} done', flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / f'bench_{run_}_{sub}_{tag}.csv', index=False)
    return T


def show(T):
    pd.set_option('display.width', 250)
    for v in ('neg5', 'pos5', 'inject_sum', 'inject_med', 'inject_p10', 'inject_p02'):
        print(f'== {v}')
        print(T.dropna(subset=[v]).pivot_table(index=['view', 'variant'], columns='lo', values=v, sort=False).round(2).to_string())
    print('== keep_late (median / p10)')
    k = T.dropna(subset=['keep_late'])
    print(k[['view', 'variant', 'n', 'keep_late', 'keep_late_p10']].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='run_145')
    ap.add_argument('--sub', default='stat090_0000')
    ap.add_argument('--tag', default='000')
    ap.add_argument('--per-bin', type=int, default=150)
    ap.add_argument('--variants', default=','.join(VARIANTS))
    a = ap.parse_args()
    show(run(a.run, a.sub, a.tag, a.per_bin, a.variants.split(',')))


if __name__ == '__main__':
    main()
