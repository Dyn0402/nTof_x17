#!/usr/bin/env python3
"""trade_analysis.py [--dir DIR] -- item 6: what is left of the steep-track compression and the bright-track shallow
reading once t0 is pinned, sigma0 is one fixed value, the chamber bow is mapped, and (separately) the charge grid is 20 ns.

Inputs: fc_arm.py outputs <dir>/<det>_<arm>.npz (same 1320 held-out events per chamber, make_fc_dag.py), plus the
audit's earlier arms on the same events for the 'before' state: heldout/<det>_t0i.npz (production: free t0, sigma0
0.238/0.343 + per-event wide choice) and ws_t0/heldout/<det>_lat.npz (t0 pinned, production sigma0 + choice).

Per chamber / view / arm:
  * the bow map (position-dependent w0 and angle scale, ws_widthtilt/set_angle_map.measure: 12 bins of mid-gap
    position, robust line w = w0(p) + s(p) * vk * tan_ref) measured on one half of the events (eid parity) and applied
    to the other, both ways (cross-validated, so the map cannot learn the angle response of the events it corrects);
    vk = the arm's own ladder (median (w - c) / tan_ref over 0.1 < |tan| < 0.45);
  * the symmetric response S(|t|) = (median r+ + median r-)/2, r = tan_corr / tan_ref, in |tan_ref| bins (w0-free);
    compression C = S(0.22-0.45) / S(0.08-0.16) - 1  (negative = steep tracks read shallow; the 1500-event caches
    hold only ~20 tracks per view above |tan| 0.28, so the steep bin starts at 0.22);
  * brightness: the same symmetric scale over 0.10 < |t| < 0.40 in charge (q_sum) terciles; B = S(bright)/S(faint) - 1;
    and split by saturation (any sample >= SAT);
  * angle s68 (deg, |d| < 5 deg) and far fraction after the map; chi2/dof median.
Event-bootstrap errors (200).  Writes results/fc_trade.json.
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'ws_widthtilt'))
from set_angle_map import measure  # noqa: E402

DIR = os.path.expanduser('~/x17/cosmic_bench/model_audit/fc/trade')
OLD = {'prod': os.path.expanduser('~/x17/cosmic_bench/model_audit/heldout/{det}_t0i.npz'),
       'lat': os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_t0/heldout/{det}_lat.npz')}
DETS = ['det2', 'det3', 'det4', 'det6', 'det7']
ARMS = ['prod', 'lat', 's31free', 's24', 's31', 's38', 's45', 'split', 's31g20']
SIG = {'s24': 0.24, 's31': 0.31, 's38': 0.38, 's45': 0.45}
BINS = [(0.03, 0.08), (0.08, 0.14), (0.14, 0.20), (0.20, 0.28), (0.28, 0.45)]
SAT = 3550.0


def load(det, arm, d):
    if arm in OLD:
        f = OLD[arm].format(det=det)
        return np.load(f)['per'] if os.path.exists(f) else None
    f = os.path.join(d, f'{det}_{arm}.npz')
    if os.path.exists(f):
        return np.load(f)['per']
    fs = sorted(glob.glob(os.path.join(d, f'{det}_{arm}_c*.npz')))
    return np.concatenate([np.load(x)['per'] for x in fs]) if fs else None


def ladder(t, w):
    m = (np.abs(t) > 0.1) & (np.abs(t) < 0.45)
    c = np.median(w[np.abs(t) < 0.05]) if (np.abs(t) < 0.05).sum() > 20 else 0.0
    return float(np.median((w[m] - c) / t[m])), float(c)


def apply_map(mp, p, w, vk):
    k, w0, s = mp
    return (w - np.interp(p, k, w0)) / (np.interp(p, k, s) * vk)


def corrected(R, use_map=True):
    """tan after the cross-validated bow map (or the constant mapping): (tan_corr, tan_ref, q, nsat_flag, valid)."""
    t, w, p0, q, wmax = R[:, 2], R[:, 3], R[:, 9], R[:, 15], R[:, 6]
    vk, c = ladder(t, w)
    tc0 = (w - c) / vk
    pos = p0 + 15.0 * tc0
    out = tc0.copy()
    if use_map:
        par = (R[:, 0].astype(int) % 2)
        for h in (0, 1):
            tr, te = par == h, par != h
            ok = tr & np.isfinite(pos) & np.isfinite(t) & np.isfinite(w)
            k, a, s, n = measure(pos[ok], t[ok], w[ok], vk)
            if len(k) >= 4:
                out[te] = apply_map((k, a, s), pos[te], w[te], vk)
    return out, t, q, wmax >= SAT, np.isfinite(out) & np.isfinite(t), vk


def sym_scale(tc, t, lo, hi):
    mp = (t >= lo) & (t < hi); mn = (t <= -lo) & (t > -hi)
    if mp.sum() < 5 or mn.sum() < 5:
        return np.nan
    return 0.5 * (np.median(tc[mp] / t[mp]) + np.median(tc[mn] / t[mn]))


def metrics(tc, t, q, sat, R):
    S = [sym_scale(tc, t, lo, hi) for lo, hi in BINS]
    s_sh = sym_scale(tc, t, 0.08, 0.16); s_st = sym_scale(tc, t, 0.22, 0.45)
    inc = (np.abs(t) > 0.10) & (np.abs(t) < 0.40)
    qe = np.nanpercentile(q[inc], [33.3, 66.7])
    sq = []
    for m in (q <= qe[0], (q > qe[0]) & (q <= qe[1]), q > qe[1]):
        sq.append(sym_scale(tc[m], t[m], 0.10, 0.40))
    ssat = [sym_scale(tc[~sat], t[~sat], 0.10, 0.40), sym_scale(tc[sat], t[sat], 0.10, 0.40)]
    d = np.degrees(np.arctan(tc) - np.arctan(t)); far = np.abs(d) > 5; k = ~far
    s68 = float(np.percentile(np.abs(d[k] - np.median(d[k])), 68))
    chi = float(np.median(R[:, 4] / np.maximum(R[:, 5], 1)))
    return dict(S=S, C=s_st / s_sh - 1, B=sq[2] / sq[0] - 1, Sq=sq, Bsat=ssat[1] / ssat[0] - 1 if np.isfinite(ssat[1]) else np.nan,
                sat_frac=float(sat[inc].mean()), s68=s68, far=float(100 * far.mean()), chi2dof=chi)


def analyse(R, use_map=True, nboot=200):
    tc, t, q, sat, ok, vk = corrected(R, use_map)
    tc, t, q, sat, R = tc[ok], t[ok], q[ok], sat[ok], R[ok]
    M = metrics(tc, t, q, sat, R)
    rng = np.random.default_rng(3); B = []
    for _ in range(nboot):
        i = rng.integers(0, len(t), len(t)); m = metrics(tc[i], t[i], q[i], sat[i], R[i])
        B.append([m['C'], m['B'], m['Bsat'], m['s68']] + m['S'])
    B = np.array(B)
    M['err'] = dict(C=float(np.nanstd(B[:, 0])), B=float(np.nanstd(B[:, 1])), Bsat=float(np.nanstd(B[:, 2])),
                    s68=float(np.nanstd(B[:, 3])), S=np.nanstd(B[:, 4:], 0).tolist())
    M['ladder'] = vk; M['n'] = int(len(t))
    return M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', default=DIR); ap.add_argument('--nboot', type=int, default=200)
    a = ap.parse_args(); out = {}
    for det in DETS:
        for arm in ARMS:
            P = load(det, arm, a.dir)
            if P is None:
                continue
            for vi, v in enumerate('xy'):
                R = P[P[:, 1] == vi]
                if len(R) < 200:
                    continue
                for mp in (False, True):
                    M = analyse(R, mp, a.nboot)
                    key = f'{det}|{v}|{arm}|{"map" if mp else "nomap"}'
                    out[key] = M
                    if mp or arm in ('prod', 's31'):
                        print(f'{key:28s} n {M["n"]:4d} ladder {M["ladder"]:.2f}  C {100*M["C"]:+5.1f}±{100*M["err"]["C"]:.1f}%  '
                              f'B {100*M["B"]:+5.1f}±{100*M["err"]["B"]:.1f}%  Bsat {100*M["Bsat"]:+5.1f}% (sat {100*M["sat_frac"]:.0f}%)  '
                              f's68 {M["s68"]:.3f}±{M["err"]["s68"]:.3f}  far {M["far"]:.1f}%  chi2/dof {M["chi2dof"]:.0f}', flush=True)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'results', 'fc_trade.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
