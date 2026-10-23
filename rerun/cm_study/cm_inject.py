#!/usr/bin/env python3
"""cm_inject.py -- error each common-mode estimator puts on a track's strips, by injection into real noise (W11).

Signal: real track events from the file, cleaned with a generous masked common mode (±10 strips) and zeroed outside
that mask, so each injected pattern is a real cluster with its real tails and undershoot.  Noise + common mode: real
events of the same file with no signal on the FEU.  For each estimator E the signal-induced error is

    Δ_E(t, block) = E(quiet + signal) − E(quiet)

i.e. what the signal itself does to the baseline (the estimator's own noise cancels).  Δ is applied to every strip of
the block, so it is read against the signal it distorts: the window-integrated charge of the cluster's centre strip
and of its ±1, ±2, ±3 neighbours (the shape the RC kernel's σ0 and D_rc, and the D_T stacks, are fitted to).

    python cm_inject.py <decoded.root> <label> [--n 3000] [--pairs 1500]
"""
import argparse
import json
import os
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cm_structure import load, signal_blocks, NB, BW  # noqa: E402

THR = 5.0


def masked_cm(X, noise, M, min_keep=16, fallback=None):
    """Per-sample block CM of one event X (S, 512) with channels within M of a cluster removed.
    fallback: None = keep the plain median when too few channels remain; 'feu' = median of the other blocks' masked
    CM plus that block's mean offset to them over the masked samples (not used when the block is fully covered)."""
    S = X.shape[0]
    blk = X.reshape(S, NB, BW)
    cm = np.median(blk, axis=2)
    if M < 0:
        return cm
    U = X - np.repeat(cm, BW, axis=1)
    above = U > THR * noise[None, :]
    pers = (above[1:] & above[:-1]).any(0)
    sig = np.zeros(512, bool)
    idx = np.flatnonzero(pers)
    if len(idx) >= 2:
        for g in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(g) >= 2:
                sig[max(0, g[0] - M):g[-1] + M + 1] = True
    if not sig.any():
        return cm
    keep = ~sig.reshape(NB, BW)
    short = []
    for b in range(NB):
        if keep[b].all():
            continue
        if keep[b].sum() < min_keep:
            short.append(b)
            continue
        cm[:, b] = np.median(blk[:, b, keep[b]], axis=1)
    if fallback == 'feu' and short:
        good = [b for b in range(NB) if b not in short]
        if good:
            ref = np.median(cm[:, good], axis=1)
            for b in short:
                cm[:, b] = ref
    return cm


ESTIMATORS = {
    'plain': dict(M=-1),
    'mask0': dict(M=0),
    'mask2': dict(M=2),
    'mask3': dict(M=3),
    'mask5': dict(M=5),
    'mask8': dict(M=8),
    'mask5_feu': dict(M=5, fallback='feu'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('f'); ap.add_argument('label')
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--pairs', type=int, default=1500)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    _, X = load(a.f, a.n)
    ped = np.median(X[:300], axis=(0, 1))
    X -= ped[None, None, :]
    cm = np.median(X.reshape(*X.shape[:2], NB, BW), axis=3)
    noise = 1.4826 * np.median(np.abs((X - np.repeat(cm, BW, axis=2))[:300]), axis=(0, 1))
    sig = signal_blocks(X, noise)
    quiet = np.flatnonzero(~sig.any(1))
    track = np.flatnonzero(sig.any(1) & (sig.sum(1) <= 3))
    dead = noise < 0.3 * np.median(noise)
    rows = []
    for _ in range(a.pairs):
        ts, qs = int(rng.choice(track)), int(rng.choice(quiet))
        Xs = X[ts]
        c10 = masked_cm(Xs, noise, 10)
        Ssig = Xs - np.repeat(c10, BW, axis=1)
        U = Xs - np.repeat(cm[ts], BW, axis=1)
        above = (U > THR * noise[None, :])
        pers = (above[1:] & above[:-1]).any(0) & ~dead
        idx = np.flatnonzero(pers)
        if len(idx) < 2:
            continue
        mask = np.zeros(512, bool)
        for g in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(g) >= 2:
                mask[max(0, g[0] - 10):g[-1] + 11] = True
        Ssig[:, ~mask] = 0.0
        Ssig[:, dead] = 0.0
        Q = Ssig.sum(0)
        c = int(np.argmax(np.where(mask, Q, -np.inf)))
        b = c // BW
        nsig_blk = int(pers[b * BW:(b + 1) * BW].sum())
        Xq = X[qs]
        rec = dict(width=nsig_blk, q0=float(Q[c]))
        for k in (1, 2, 3, 5):
            qk = [Q[c + d] for d in (-k, k) if 0 <= c + d < 512 and (c + d) // BW == b]
            rec[f'q{k}'] = float(np.mean(qk)) if qk else np.nan
        pk = int(np.argmax(Ssig[:, c]))
        for name, kw in ESTIMATORS.items():
            e1 = masked_cm(Xq + Ssig, noise, **kw)[:, b]
            e0 = masked_cm(Xq, noise, **kw)[:, b]
            d = e1 - e0
            rec[name] = dict(dQ=float(d.sum()), dpk=float(d[pk]), dmax=float(np.abs(d).max()))
        rows.append(rec)
    # summarise by cluster width in the block
    bins = [(2, 8), (8, 14), (14, 20), (20, 30), (30, 65)]
    out = dict(label=a.label, file=os.path.basename(a.f), n=len(rows), noise_median=float(np.median(noise)),
               by_width={})
    for lo, hi in bins:
        R = [r for r in rows if lo <= r['width'] < hi]
        if len(R) < 10:
            continue
        s = dict(n=len(R), q0=float(np.median([r['q0'] for r in R])),
                 q1=float(np.nanmedian([r['q1'] for r in R])), q2=float(np.nanmedian([r['q2'] for r in R])),
                 q3=float(np.nanmedian([r['q3'] for r in R])), q5=float(np.nanmedian([r['q5'] for r in R])))
        for name in ESTIMATORS:
            dQ = np.array([r[name]['dQ'] for r in R])
            s[name] = dict(dQ_med=float(np.median(dQ)), dQ_p90abs=float(np.percentile(np.abs(dQ), 90)),
                           dpk_med=float(np.median([r[name]['dpk'] for r in R])),
                           dmax_med=float(np.median([r[name]['dmax'] for r in R])),
                           rel_q1=float(np.median(-dQ / np.array([r['q1'] for r in R]))),
                           rel_q2=float(np.nanmedian(-dQ / np.array([r['q2'] for r in R]))),
                           rel_q3=float(np.nanmedian(-dQ / np.array([r['q3'] for r in R]))),
                           rel_q5=float(np.nanmedian(-dQ / np.array([r['q5'] for r in R]))),
                           rel_q3_p90=float(np.nanpercentile(np.abs(dQ / np.array([r['q3'] for r in R])), 90)))
        out['by_width'][f'{lo}-{hi}'] = s
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f'inject_{a.label}.json'), 'w'), indent=1)
    print(f"{a.label}: {len(rows)} injections, noise {out['noise_median']:.1f} ADC")
    for wb, s in out['by_width'].items():
        print(f"  width {wb:6s} n={s['n']:4d}  Q0 {s['q0']:7.0f} Q±1 {s['q1']:6.0f} Q±2 {s['q2']:6.0f} Q±3 {s['q3']:6.0f}")
        for name in [x for x in ESTIMATORS if x in ('plain', 'mask0', 'mask3', 'mask5', 'mask5_feu')]:
            e = s[name]
            print(f"     {name:10s} dQ {e['dQ_med']:7.1f} (|p90| {e['dQ_p90abs']:6.1f})  at peak {e['dpk_med']:5.1f}"
                  f"  max {e['dmax_med']:5.1f}   bias on Q±1 {100*e['rel_q1']:+6.1f}%  Q±2 {100*e['rel_q2']:+6.1f}%"
                  f"  Q±3 {100*e['rel_q3']:+6.1f}%  Q±5 {100*e['rel_q5']:+6.1f}%  |Q±3| p90 {100*e['rel_q3_p90']:5.1f}%")


if __name__ == '__main__':
    main()
