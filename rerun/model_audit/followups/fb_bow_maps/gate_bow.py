#!/usr/bin/env python3
"""gate_bow.py -- held-out gate of the bow correction (FB, 2026-10-27).

The angle map acts AFTER the waveform fit (wft.reco.tan_of converts the fitted w and p0 into tan), so it can be gated
exactly on the per-event tables: fit the map on one half of the events (event_id even), apply it to the other half
(odd), and swap.  Coordinates are the production ones: the track at mid-gap from the fit itself, u = x_p0 + 15 tan_x,
w = y_p0 + 15 tan_y with the constant-mapping tangents.  Reference: M3 tan_ref, positions ref_mesh.

Arms (angle):
  const   tan = (w - w0) / (kw v), w0 and kw refit on the training half          [today's bundle content]
  map1d   ws_widthtilt angle_map: w0(p), s(p) in 12 bins of the view's own coordinate  [in wft/, in no bundle]
  bow2d   w = A(U, W) + B tan (1 + su U + sw W), A quadratic in (U, W)           [static surface + scale]
  phys    A(U, W) = c0 + v r_f dh/dp with h(U, W) from the POSITION route on the training half: one shape from
          positions, two free numbers (tilt c0, r_f) from the angles               [the physical model as a corrector]
Transfer (--transfer <golden> <row>): the static shape of A (minus its constant) measured on the golden row, applied
to another run of the same chamber with only the rigid tilt c0 and kw re-measured on that run (the proposed bundle
split: per-chamber bow + per-run tilt).
Position: residual p0 - ref_mesh before / after the height correction p0 - tan * h(U, W) (h from the training half).
Metrics: s68 of the angle residual [deg], and |bias| in |tan| bins; s68 of the position residual [mm].
Writes results/gate_bow.json (and gate_transfer.json).
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fbcommon import DATASETS, DETS, RES, load, view_mask, huber, surf_terms

EDGES12 = np.linspace(0, 400, 13)


def s68(x):
    x = x[np.isfinite(x)]
    return float(0.5 * np.diff(np.percentile(x, [16, 84]))[0]) if len(x) > 20 else np.nan


def prep(T, p):
    m = view_mask(T, p)
    v, kw, w0 = T.attrs['v'], T.attrs['kw'][p], T.attrs['w0'][p]
    t = T[f'{p}_t'].to_numpy(); wv = T[f'{p}_wv'].to_numpy()
    m &= np.isfinite(wv) & (np.abs(wv - v * t) < 0.15 * v)
    # production coordinates: the fit's own mid-gap position, both views
    tx = (T.x_wv - T.attrs['w0']['x']).to_numpy() / (T.attrs['kw']['x'] * v)
    ty = (T.y_wv - T.attrs['w0']['y']).to_numpy() / (T.attrs['kw']['y'] * v)
    u = T.x_p0.to_numpy() + 15 * tx; w = T.y_p0.to_numpy() + 15 * ty
    m &= np.isfinite(u) & np.isfinite(w)
    m &= (np.abs(u - 200) <= 220) & (np.abs(w - 200) <= 220)     # both views' points on the plate (pass-1 review 10-30)
    eid = np.asarray(T.index)
    return dict(m=m, t=t, wv=wv, u=u, w=w, U=(u - 200) / 100, W=(w - 200) / 100, r=T[f'{p}_r'].to_numpy(),
                eid=eid, v=v, own=(u if p == 'x' else w))


def fit_const(D, k):
    X = np.c_[np.ones(k.sum()), D['t'][k]]
    return huber(X, D['wv'][k], 0.03 * D['v'])


def tan_const(D, b, k):
    return (D['wv'][k] - b[0]) / b[1]


def fit_map1d(D, k):
    out = []
    for lo, hi in zip(EDGES12[:-1], EDGES12[1:]):
        q = k & (D['own'] >= lo) & (D['own'] < hi)
        if q.sum() < 40 or (D['t'][q] > 0.04).sum() < 10 or (D['t'][q] < -0.04).sum() < 10:
            continue
        b = huber(np.c_[np.ones(q.sum()), D['t'][q]], D['wv'][q], 0.03 * D['v'])
        out.append((0.5 * (lo + hi), b[0], b[1]))
    return np.array(out)


def tan_map1d(D, M, k):
    pp = D['own'][k]
    return (D['wv'][k] - np.interp(pp, M[:, 0], M[:, 1])) / np.interp(pp, M[:, 0], M[:, 2])


def design2d(D, k):
    U, W, t = D['U'][k], D['W'][k], D['t'][k]
    return np.c_[surf_terms(U, W), t, t * U, t * W]


def fit_bow2d(D, k):
    return huber(design2d(D, k), D['wv'][k], 0.03 * D['v'])


def tan_bow2d(D, b, k):
    U, W = D['U'][k], D['W'][k]
    A = surf_terms(U, W) @ b[:6]
    return (D['wv'][k] - A) / (b[6] + b[7] * U + b[8] * W)


def fit_h(D, k):
    """position route on the training half: r = a + bu U + bw W + t h(U, W)."""
    U, W, t = D['U'][k], D['W'][k], D['t'][k]
    X = np.c_[np.ones_like(U), U, W, t[:, None] * surf_terms(U, W)]
    return huber(X, D['r'][k], 0.5)


def dh(hb, U, W, p):
    """dh/dp [mm/mm] of the fitted surface, p = own axis (x -> U, y -> W)."""
    h = hb[3:]
    return ((h[1] + h[3] * U + h[5] * W) if p == 'x' else (h[2] + h[4] * W + h[5] * U)) / 100.0


def fit_phys(D, hb, p, k):
    g = D['v'] * dh(hb, D['U'][k], D['W'][k], p)
    X = np.c_[np.ones(k.sum()), g, D['t'][k]]
    return huber(X, D['wv'][k], 0.03 * D['v'])


def tan_phys(D, hb, b, p, k):
    g = D['v'] * dh(hb, D['U'][k], D['W'][k], p)
    return (D['wv'][k] - b[0] - b[1] * g) / b[2]


def metrics(res_tan, t):
    out = dict(s68_deg=s68(np.degrees(np.arctan(t + res_tan)) - np.degrees(np.arctan(t))))
    for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.45)):
        q = (np.abs(t) >= lo) & (np.abs(t) < hi)
        out[f'bias_{lo:.1f}'] = float(np.degrees(np.median(res_tan[q] * np.sign(t[q])))) if q.sum() > 30 else np.nan
    return out


def gate(name):
    T = load(name)
    res = {}
    for p in 'xy':
        D = prep(T, p)
        par = (D['eid'] % 2 == 0)
        acc = {a: [] for a in ('const', 'map1d', 'bow2d', 'phys')}
        pos = {'align': [], 'height': []}
        tt = []
        for tr in (par, ~par):
            ktr, kte = D['m'] & tr, D['m'] & ~tr
            t = D['t'][kte]; tt.append(t)
            b = fit_const(D, ktr); acc['const'].append(tan_const(D, b, kte) - t)
            M = fit_map1d(D, ktr); acc['map1d'].append(tan_map1d(D, M, kte) - t)
            b2 = fit_bow2d(D, ktr); acc['bow2d'].append(tan_bow2d(D, b2, kte) - t)
            hb = fit_h(D, ktr)
            bp = fit_phys(D, hb, p, ktr); acc['phys'].append(tan_phys(D, hb, bp, p, kte) - t)
            r = D['r'][kte] - (hb[0] + hb[1] * D['U'][kte] + hb[2] * D['W'][kte])
            hh = surf_terms(D['U'][kte], D['W'][kte]) @ hb[3:]
            pos['align'].append(r - t * hb[3]); pos['height'].append(r - t * hh)
        t = np.concatenate(tt)
        res[p] = {a: metrics(np.concatenate(v), t) for a, v in acc.items()}
        res[p]['pos_s68_mm'] = {a: s68(np.concatenate(v)) for a, v in pos.items()}
        res[p]['n'] = int(len(t))
        print(f'{name} {p}: s68 deg ' + ' '.join(f'{a} {res[p][a]["s68_deg"]:.3f}' for a in acc) +
              f' | pos s68 {res[p]["pos_s68_mm"]["align"]:.3f} -> {res[p]["pos_s68_mm"]["height"]:.3f} mm', flush=True)
    return res


def transfer(gold, row):
    """static shape from the golden row, per-run tilt c0 + kw (+ scale terms) refit on the row."""
    G, R = load(gold), load(row)
    out = {}
    for p in 'xy':
        Dg, Dr = prep(G, p), prep(R, p)
        bg = fit_bow2d(Dg, Dg['m'])
        k = Dr['m']
        A_shape = surf_terms(Dr['U'][k], Dr['W'][k])[:, 1:] @ bg[1:6]
        # on the row: w - A_shape = c0 + B t   (tilt and scale re-measured, shape fixed), 2-fold to stay held out
        par = (Dr['eid'] % 2 == 0)[k]
        t, wv = Dr['t'][k], Dr['wv'][k]
        rc, rf, rconst, rmap = [], [], [], []
        for tr in (par, ~par):
            X = np.c_[np.ones(tr.sum()), t[tr]]
            b = huber(X, wv[tr] - A_shape[tr], 0.03 * Dr['v'])
            rc.append((wv[~tr] - A_shape[~tr] - b[0]) / b[1] - t[~tr])
            b0 = huber(X, wv[tr], 0.03 * Dr['v'])
            rconst.append((wv[~tr] - b0[0]) / b0[1] - t[~tr])
            kk = np.where(k)[0]
            ktr = np.zeros_like(k); ktr[kk[tr]] = True
            kte = np.zeros_like(k); kte[kk[~tr]] = True
            b2 = fit_bow2d(Dr, ktr)
            rf.append(tan_bow2d(Dr, b2, kte) - Dr['t'][kte])
        tt = np.concatenate([t[~par], t[par]])
        out[p] = dict(const=metrics(np.concatenate(rconst), tt), static_plus_tilt=metrics(np.concatenate(rc), tt),
                      refit_full=metrics(np.concatenate(rf), tt), n=int(k.sum()))
        print(f'transfer {gold}->{row} {p}: const {out[p]["const"]["s68_deg"]:.3f}  static+tilt '
              f'{out[p]["static_plus_tilt"]["s68_deg"]:.3f}  own full map {out[p]["refit_full"]["s68_deg"]:.3f}', flush=True)
    return out


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--transfer':
        pairs = [(DATASETS[n][0], n) for n in DATASETS if n not in DETS]
        out = {f'{g}->{r}': transfer(g, r) for g, r in pairs}
        json.dump(out, open(f'{RES}/gate_transfer.json', 'w'), indent=1)
    else:
        out = {n: gate(n) for n in (sys.argv[1:] or DETS)}
        json.dump(out, open(f'{RES}/gate_bow.json', 'w'), indent=1)
