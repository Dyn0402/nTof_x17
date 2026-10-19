#!/usr/bin/env python3
"""xasym_tests.py -- is the X-view +-1 sharing asymmetry a property of the board, or a drift walk?  (FINDINGS §42)

The claim on record (plane_ratio §5a, EXTRACTION 08-05, wft c1_asym): X's +-1 neighbours are unequal, -1 > +1,
on det3 and on det4 in H4 ("internal to the board, same in both mounts").  Two physical mechanisms give
unequal neighbours; they predict different things, and every test here is model-free (raw waveform windows,
no template, no kernel):

  coupling   a fixed lateral offset between the resistive strips / pads and the readout lines, or crosstalk
             in the routing: the neighbours differ in AMOUNT, with the SAME pulse shape and timing; constant
             in the track angle; constant along x if the pitches match, oscillating with the beat
             1/|1/0.780 - 1/0.800| = 31.2 mm if the resistive pitch is 800 um against the 780 um pads.
  walk       the charge drifts sideways as it drifts down (a field tilt, a reference/chamber angle offset):
             the neighbours differ in TIMING (one gets the early charge, the other the late) and the amount
             asymmetry grows with the track angle and crosses zero at tan = -delta, not at 0.

Per event (rows sorted by position; unsaturated, no flagged strip within +-2 of the charge maximum c):
  aQ = (Q[c-1] - Q[c+1]) / (Q[c-1] + Q[c+1])      Q = window integral
  dt = t[c-1] - t[c+1]                              t = charge-weighted mean sample time (positive samples)
and against the reference tan_x (head-on and inclined caches), position x (head-on), and chamber.

    xasym_tests.py  -> results/xasym_tests.json
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x_width as xw                                         # noqa: E402
from strip_gain_audit import bad_set                         # noqa: E402

AUD = json.load(open(os.path.join(HERE, 'results', 'strip_gain_audit.json')))
GAPC = os.path.expanduser('~/x17/cosmic_bench/cloud_basics/gap')
SETS = {'det2': ('o22_long_det2', 'o22_long_det2'), 'det3': ('sat_det3__long_run_resist_490V_drift_1000V', 'sat_det3'),
        'det4': ('g_det4', 'g_det4'), 'det6': ('g_det6_long', 'g_det6_long'), 'det7': ('g_det7_long', 'g_det7_long')}
STEP = 60.0                     # ns per sample (bench)
SAT = 3500.0
BEAT = 1.0 / abs(1.0 / 0.780 - 1.0 / 0.800)
TBINS = np.array([-0.8, -0.45, -0.3, -0.2, -0.12, -0.06, -0.03, 0.0, 0.03, 0.06, 0.12, 0.2, 0.3, 0.45, 0.8])


def per_event(e, bad):
    P = e['x']
    W = np.asarray(P['W'], float)
    if W.max() >= SAT:
        return None
    c = int(np.argmax(W.sum(1)))
    if c < 2 or c > len(W) - 3:
        return None
    ch = [int(q) for q in P['ch'][c - 2:c + 3]]
    if any(q in bad for q in ch):
        return None
    Q = W.sum(1)
    if Q[c - 1] <= 0 or Q[c + 1] <= 0 or W[c].max() < 300:
        return None
    t = np.arange(W.shape[1]) * STEP

    def tm(r):
        w = np.clip(W[r], 0, None)
        return float((w * t).sum() / max(w.sum(), 1e-9))
    return dict(aQ=float((Q[c - 1] - Q[c + 1]) / (Q[c - 1] + Q[c + 1])), dt=tm(c - 1) - tm(c + 1),
                r1=float((Q[c - 1] + Q[c + 1]) / (2 * Q[c])), tan=float(e['tan_x']), x=float(P['pos'][c]),
                pk=float(W[c].max()), t0c=tm(c))


def shapes(hk, ik, bad, tan_c=-0.005, tan_w=0.02):
    """Stacked +-1 and centre waveforms (each event normalised to its centre's integral, aligned on the centre's
    peak sample) at the time-symmetric angle, and the -1 minus +1 difference fitted as a x centre + b x d(centre)/dt:
    a coupling gives b = 0, a delayed copy b/a = -(delay in samples), a pure walk a = 0."""
    Wm = Wp = W0 = 0.0
    n = 0
    seen = set()
    for path in (f'{xw.CACHES}/headon_{hk}.pkl', f'{GAPC}/incl_{ik}.pkl'):
        if not os.path.exists(path):
            continue
        for e in pickle.load(open(path, 'rb')).values():
            if 'x' not in e or abs(e['tan_x'] - tan_c) >= tan_w:
                continue
            k = (round(e['tan_x'], 5), round(float(e['x']['pos'][0]), 2))
            if k in seen:
                continue
            seen.add(k)
            W = np.asarray(e['x']['W'], float)
            if W.max() >= SAT:
                continue
            c = int(np.argmax(W.sum(1)))
            if c < 2 or c > len(W) - 3 or any(int(q) in bad for q in e['x']['ch'][c - 2:c + 3]):
                continue
            t = int(np.argmax(W[c]))
            if t < 6 or t > W.shape[1] - 14:
                continue
            sl = slice(t - 6, t + 14)
            q = W[c, sl].sum()
            Wm = Wm + W[c - 1, sl] / q; Wp = Wp + W[c + 1, sl] / q; W0 = W0 + W[c, sl] / q; n += 1
    if n < 20:
        return None
    Wm, Wp, W0 = Wm / n, Wp / n, W0 / n
    D, dW = Wm - Wp, np.gradient(W0)
    A = np.vstack([W0, dW]).T
    coef, *_ = np.linalg.lstsq(A, D, rcond=None)
    fit = A @ coef
    return dict(n=n, centre=W0.tolist(), m1=Wm.tolist(), p1=Wp.tolist(), a=float(coef[0]), b=float(coef[1]),
                delay_ns=float(-coef[1] / coef[0] * STEP) if coef[0] else float('nan'),
                explained=float(1 - ((D - fit) ** 2).sum() / (D ** 2).sum()))


def boot_mean(v, n=400, seed=3):
    v = np.asarray(v, float)
    if len(v) < 5:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    b = [np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return float(np.mean(v)), float(np.std(b))


def main():
    out = {'beat_mm': BEAT, 'tbins': TBINS.tolist()}
    for det, (hk, ik) in SETS.items():
        bad = bad_set(AUD[hk], 'x')
        evs = []
        for path in (f'{xw.CACHES}/headon_{hk}.pkl', f'{GAPC}/incl_{ik}.pkl'):
            if os.path.exists(path):
                for e in pickle.load(open(path, 'rb')).values():
                    if 'x' in e:
                        r = per_event(e, bad)
                        if r:
                            evs.append(r)
        # de-duplicate events present in both caches (same tan, x, pk)
        seen, E = set(), []
        for r in evs:
            k = (round(r['tan'], 5), round(r['x'], 2), round(r['pk']))
            if k not in seen:
                seen.add(k); E.append(r)
        tan = np.array([r['tan'] for r in E]); aQ = np.array([r['aQ'] for r in E]); dt = np.array([r['dt'] for r in E])
        head = np.abs(tan) < 0.03
        res = dict(n=len(E), n_head=int(head.sum()))
        res['head_aQ'] = boot_mean(aQ[head]); res['head_dt'] = boot_mean(dt[head])
        # by tan
        rows = []
        for lo, hi in zip(TBINS[:-1], TBINS[1:]):
            m = (tan >= lo) & (tan < hi)
            if m.sum() >= 15:
                rows.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()), tan=float(np.median(tan[m])),
                                 aQ=boot_mean(aQ[m]), dt=boot_mean(dt[m])))
        res['by_tan'] = rows
        # straight-line fits in |tan| < 0.3: aQ = a0 + a1 tan, dt = b0 + b1 tan  (zero crossing = -a0/a1)
        m = np.abs(tan) < 0.3
        if m.sum() > 30:
            A = np.vstack([np.ones(m.sum()), tan[m]]).T
            for nm, y in (('aQ', aQ[m]), ('dt', dt[m])):
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                rr = y - A @ coef
                cov = np.linalg.inv(A.T @ A) * (rr @ rr) / max(len(y) - 2, 1)
                res[f'fit_{nm}'] = dict(c0=float(coef[0]), c0_err=float(np.sqrt(cov[0, 0])), c1=float(coef[1]),
                                        c1_err=float(np.sqrt(cov[1, 1])),
                                        zero=float(-coef[0] / coef[1]) if coef[1] else float('nan'))
        # position: head-on, binned in x; and the Moire phase (x mod BEAT)
        xs = np.array([r['x'] for r in E])[head]
        ah = aQ[head]
        if head.sum() > 60:
            xb = np.arange(0, 401, 25.0)
            res['by_x'] = [dict(x=float(0.5 * (a + b)), n=int(((xs >= a) & (xs < b)).sum()), aQ=boot_mean(ah[(xs >= a) & (xs < b)]))
                           for a, b in zip(xb[:-1], xb[1:]) if ((xs >= a) & (xs < b)).sum() >= 15]
            ph = 2 * np.pi * (xs % BEAT) / BEAT
            A = np.vstack([np.ones(len(ph)), np.cos(ph), np.sin(ph)]).T
            coef, *_ = np.linalg.lstsq(A, ah, rcond=None)
            rr = ah - A @ coef
            cov = np.linalg.inv(A.T @ A) * (rr @ rr) / max(len(ah) - 3, 1)
            amp = float(np.hypot(coef[1], coef[2]))
            amp_err = float(np.sqrt((coef[1] ** 2 * cov[1, 1] + coef[2] ** 2 * cov[2, 2]) / max(amp ** 2, 1e-12)))
            res['moire'] = dict(const=float(coef[0]), amp=amp, amp_err=amp_err, phase=float(np.arctan2(coef[2], coef[1])))
            pb = np.linspace(0, 2 * np.pi, 9)
            res['moire_bins'] = [dict(ph=float(0.5 * (a + b)), aQ=boot_mean(ah[(ph >= a) & (ph < b)])) for a, b in zip(pb[:-1], pb[1:])]
        res['shape'] = shapes(hk, ik, bad)
        out[det] = res
        f = res.get('fit_aQ', {}); g = res.get('fit_dt', {}); mo = res.get('moire', {})
        print(f'{det}: n {res["n"]} (head-on {res["n_head"]})  head-on aQ {res["head_aQ"][0]:+.3f} ± {res["head_aQ"][1]:.3f}  '
              f'dt {res["head_dt"][0]:+.1f} ± {res["head_dt"][1]:.1f} ns | aQ = {f.get("c0", np.nan):+.3f}±{f.get("c0_err", np.nan):.3f} '
              f'+ {f.get("c1", np.nan):+.2f}±{f.get("c1_err", np.nan):.2f} tan (zero at {f.get("zero", np.nan):+.3f}) | '
              f'dt = {g.get("c0", np.nan):+.1f}±{g.get("c0_err", np.nan):.1f} + {g.get("c1", np.nan):+.0f} tan | '
              f'Moire {BEAT:.1f} mm amp {mo.get("amp", np.nan):.3f} ± {mo.get("amp_err", np.nan):.3f}', flush=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'xasym_tests.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
