#!/usr/bin/env python3
"""tiers.py <window pickles...> --ref REF --out OUT.json [--families ...] [--nsig 5] -- hits tiers against W, same planes.

Input: windows from extract_windows.py (or rerun/paper1_floor/boot_extract.py; those carry no raw samples, so the
analog families run on the CM-subtracted window and are flagged 'no_raw').  REF: the M3 reference per event and view
(a frames parquet with event_id, view, tan_ref, ig; or the bootstrap ids json {view: {event_id, tan_ref}}).

Per plane and family (emulate.py) the hits give a ladder t(x); u = dx/dt [mm/ns] from an amplitude-weighted line
(binary: unweighted), optionally with iterative 3-sigma trimming of strips ('r', the role of the Hough step in the
NSW / APV25 chains) and the first/last-strip drop below 50 % ('e', NSW/VMM) or 25 % ('q', APV25) of the neighbour.  Tiers (HITS_DAQ_PLAN §4):

  H-naive   tan = u / v, v the bundle's calibrated drift velocity (the best external knowledge of v).  No reference.
  H0        tan = a + k u, (a, k) fitted against tan_ref on the other half of the events (event_id parity), the
            hits_arm.py convention: a global slope remap
  H2a       tan = f(u), f the binned-median map of tan_ref against u (20 quantile bins, linear between), trained on
            the other half: the best any slope-only correction can do
  W         the pass waveform fit: tan = (1e3 w - w0) / (kw v) (wft.calib, the bundle's angle constants; no bow map)

Scores per tier: d = atan(tan) - atan(tan_ref) [deg]; bias = median d, steep = median d sign(tan_ref), and s68 = 68th percentile of |d - median| (over
|d| <= 5 deg, floorlib convention), overall and in |tan_ref| bins; implied v = 1 / (slope of u against tan_ref).
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import emulate as E  # noqa: E402

TBINS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)


def drop_ends(pos, t, a, frac):
    """the first and last hit strip (in position) are dropped when their amplitude is below frac x their inner
    neighbour's (NSW/VMM 0.5, APV25 0.25: DAQ_SURVEY.md); returns the reduced t"""
    k = np.flatnonzero(np.isfinite(t) & np.isfinite(a))
    if len(k) < 5:
        return t
    k = k[np.argsort(pos[k])]
    t = t.copy()
    if a[k[0]] < frac * a[k[1]]:
        t[k[0]] = np.nan
    if a[k[-1]] < frac * a[k[-2]]:
        t[k[-1]] = np.nan
    return t


def ladder_u(pos, t, a, weighted=True, robust=False, min_hits=3, ends=0.0):
    if ends > 0:
        t = drop_ends(pos, t, a, ends)
    k = np.isfinite(t) & np.isfinite(a) & ((a > 0) if weighted else True)
    if k.sum() < min_hits:
        return np.nan, int(k.sum())
    x, y = pos[k].astype(float), t[k].astype(float)
    wt = a[k].astype(float) if weighted else np.ones(k.sum())
    for it in range(4 if robust else 1):
        xm, ym = np.average(x, weights=wt), np.average(y, weights=wt)
        sxx = np.sum(wt * (x - xm) ** 2)
        if sxx <= 0:
            return np.nan, len(x)
        s = np.sum(wt * (x - xm) * (y - ym)) / sxx
        if not robust or len(x) <= min_hits:
            break
        r = y - (ym + s * (x - xm))
        sig = 1.4826 * np.median(np.abs(r - np.median(r)))
        keep = np.abs(r) <= 3 * max(sig, 10.0)
        if keep.all() or keep.sum() < min_hits:
            break
        x, y, wt = x[keep], y[keep], wt[keep]
    return (1.0 / s if abs(s) > 1e-9 else np.nan), len(x)


def load_ref(path):
    if path.endswith('.json'):
        J = json.load(open(path))
        rows = [(int(e), v, float(t)) for v in ('x', 'y') for e, t in zip(J[v]['event_id'], J[v]['tan_ref'])]
        return {(e, v): t for e, v, t in rows}
    F = pd.read_parquet(path, columns=['event_id', 'view', 'tan_ref', 'ig'])
    F = F[F.ig & np.isfinite(F.tan_ref)]
    F = F[~F.duplicated(['event_id', 'view'], keep=False)]
    return {(int(e), v): float(t) for e, v, t in zip(F.event_id, F.view, F.tan_ref)}


def run_families(p, fams, nsig, ideal):
    W = p['W'].astype(float)
    nz = p['noise'].astype(float)
    if p.get('raw') is not None and p.get('ped') is not None:
        R = p['raw'].astype(float) - p['ped'].astype(float)[:, None]
        # analog noise: the per-strip sigma of raw - ped includes the common mode; estimated from the pre-signal samples
        nzR = np.maximum(np.std(R[:, :4], axis=1), nz)
    else:
        R, nzR = W, nz
    out = {}
    for f in fams:
        if f == 'dream':
            out[f] = E.dream(W, nz, nsig=nsig)
        elif f == 'fd':
            out[f] = E.fd(W, nz, nsig=nsig)
        elif f == 'cw':
            out[f] = E.cw(W, nz, nsig=nsig)
        elif f == 'ideal':
            out[f] = ideal(W, nz, nsig=nsig)
        elif f == 'binary':
            out[f] = E.binary(R, nzR, nsig=nsig, rng=np.random.default_rng(p['event_id']))
        else:
            out[f] = getattr(E, f)(R, nzR, nsig=nsig)
    return out


def cv_fit(u, tr, eid, kind):
    """tan predicted from u, trained on the other parity half."""
    pred = np.full(len(u), np.nan)
    ok = np.isfinite(u) & np.isfinite(tr)
    for h in (0, 1):
        tr_m = ok & (eid % 2 != h)
        te_m = ok & (eid % 2 == h)
        if tr_m.sum() < 50:
            continue
        nb = int(min(20, max(4, tr_m.sum() // 15)))
        if kind == 'H0':
            # binned-median line of tan_ref against u (hits_arm: robust to the huge u of near-head-on planes)
            q = np.quantile(tr[tr_m], np.linspace(0, 1, nb + 1))
            xs, ys = [], []
            for lo, hi in zip(q[:-1], q[1:]):
                m = tr_m & (tr >= lo) & (tr < hi)
                if m.sum() > 3:
                    xs.append(np.median(u[m]))
                    ys.append(np.median(tr[m]))
            if len(xs) < 2:
                continue
            # u = v' tan  ->  tan = u / v'  : fit u against tan on the binned medians, invert
            c = np.polyfit(ys, xs, 1)
            pred[te_m] = (u[te_m] - c[1]) / c[0]
        else:
            q = np.quantile(u[tr_m], np.linspace(0, 1, nb + 1))
            xs, ys = [], []
            for lo, hi in zip(q[:-1], q[1:]):
                m = tr_m & (u >= lo) & (u < hi)
                if m.sum() > 3:
                    xs.append(np.median(u[m]))
                    ys.append(np.median(tr[m]))
            if len(xs) < 2:
                continue
            pred[te_m] = np.interp(u[te_m], xs, ys)
    return pred


def s68(d):
    d = d[np.isfinite(d) & (np.abs(d) <= 5)]
    if len(d) < 30:
        return None, None
    m = np.median(d)
    return float(np.percentile(np.abs(d - m), 68)), float(m)


def steep(d, tr):
    """median of d * sign(tan_ref) [deg]: > 0 = reads too steep (the compression), independent of the angle sign"""
    k = np.isfinite(d) & (np.abs(d) <= 5) & (np.abs(tr) > 0.02)
    return float(np.median(d[k] * np.sign(tr[k]))) if k.sum() >= 30 else None


def score(tan, tr):
    d = np.degrees(np.arctan(tan) - np.arctan(tr))
    o = {}
    s, b = s68(d)
    fin = d[np.isfinite(d)]
    o['all'] = dict(n=int(np.isfinite(d).sum()), s68=s, bias=b, steep=steep(d, tr),
                    sig_mad=float(1.4826 * np.median(np.abs(fin - np.median(fin)))) if len(fin) > 30 else None,
                    s68_full=float(np.percentile(np.abs(fin - np.median(fin)), 68)) if len(fin) > 30 else None,
                    frac_in5=float(np.mean(np.abs(d[np.isfinite(tr)]) <= 5)) if np.isfinite(tr).any() else None)
    for lo, hi in zip(TBINS[:-1], TBINS[1:]):
        m = (np.abs(tr) >= lo) & (np.abs(tr) < hi)
        s, b = s68(d[m])
        o[f'{lo}-{hi}'] = dict(n=int((m & np.isfinite(d)).sum()), s68=s, bias=b, steep=steep(d[m], tr[m]))
    return o


def implied_v(u, tr):
    k = np.isfinite(u) & np.isfinite(tr) & (np.abs(tr) > 0.05)
    if k.sum() < 50:
        return None
    q = np.quantile(tr[k], np.linspace(0, 1, 21))
    xs, ys = [], []
    for lo, hi in zip(q[:-1], q[1:]):
        m = k & (tr >= lo) & (tr < hi)
        if m.sum() > 5:
            xs.append(np.median(tr[m]))
            ys.append(np.median(u[m]))
    return float(np.polyfit(xs, ys, 1)[0] * 1000.0)        # um/ns


def build_table(a, fams):
    ref = load_ref(a.ref) if a.ref else None
    planes, meta = [], None
    for f in a.pickles:
        D = pickle.load(open(f, 'rb'))
        meta = meta or D['meta']
        planes += D['planes']
    if a.nmax:
        planes = planes[:a.nmax]
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))
    from wft import model as wm
    from wft.calib import CalibrationBundle
    cal = CalibrationBundle.load(E.bundle_path(meta, a.bundle))
    wm.use_calibration(cal)
    ideal = {v: E.Ideal(wm.TGRID, wm.TMPL[v], meta['nsamp']) for v in ('x', 'y')}
    v = meta['v_drift'] / 1000.0
    rows = []
    for p in planes:
        tr = ref.get((p['event_id'], p['view'])) if ref is not None else np.nan
        if tr is None:
            continue
        H = run_families(p, fams, a.nsig, ideal[p['view']])
        pos = p['pos'].astype(float)
        r = dict(event_id=p['event_id'], view=p['view'], tan_ref=tr,
                 tan_W=(p['w'] * 1e3 - meta['w0'][p['view']]) / (meta['kw'][p['view']] * meta['v_drift']),
                 no_raw=p.get('raw') is None, w_W=p['w'], v_drift=meta['v_drift'])
        for f, h in H.items():
            wtd = f != 'binary'
            r[f'u_{f}'], r[f'n_{f}'] = ladder_u(pos, h['t'], h['a'], weighted=wtd)
            r[f'ur_{f}'], _ = ladder_u(pos, h['t'], h['a'], weighted=wtd, robust=True)
            r[f'ue_{f}'], _ = ladder_u(pos, h['t'], h['a'], weighted=wtd, robust=True, ends=0.5)
            r[f'uq_{f}'], _ = ladder_u(pos, h['t'], h['a'], weighted=wtd, robust=True, ends=0.25)
        rows.append(r)
    return pd.DataFrame(rows), meta, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pickles', nargs='*')
    ap.add_argument('--ref', help='M3 reference; omitted (SPS): planes kept with tan_ref NaN')
    ap.add_argument('--table-only', action='store_true', help='write the per-plane table to --out (condor parts)')
    ap.add_argument('--from-tables', nargs='*', help='score these per-plane tables (merged parts) instead of pickles')
    ap.add_argument('--bundle', help='calib_bundle for the V6 template (default: the pickle meta)')
    ap.add_argument('--out')
    ap.add_argument('--families', default=','.join(E.FAMILIES))
    ap.add_argument('--nsig', type=float, default=5.0)
    ap.add_argument('--nmax', type=int, default=0)
    ap.add_argument('--table', help='also write the per-plane table (parquet)')
    a = ap.parse_args()
    fams = a.families.split(',')
    if a.from_tables:
        T = pd.concat([pd.read_parquet(f) for f in a.from_tables], ignore_index=True)
        T = T[~T.duplicated(['event_id', 'view'], keep=False)]
        meta = dict(key=os.path.basename(a.from_tables[0]).split('_p')[0], v_drift=float(T.attrs.get('v_drift', 0)) or None)
        v = None
    else:
        T, meta, v = build_table(a, fams)
    if a.table_only:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        T.to_parquet(a.out)
        print(f'[tiers] wrote table {a.out}: {len(T)} planes')
        return
    if a.table:
        T.to_parquet(a.table)
    if v is None:
        v = float(T.v_drift.iloc[0]) / 1000.0
        meta['v_drift'] = float(T.v_drift.iloc[0])
    res = dict(meta=dict(key=meta['key'], v_drift=meta['v_drift'], nsig=a.nsig, n_planes=len(T),
                         no_raw=bool(T.no_raw.any())), views={})
    for view in ('x', 'y'):
        g = T[T.view == view].reset_index(drop=True)
        tr, eid = g.tan_ref.to_numpy(), g.event_id.to_numpy()
        V = dict(n=len(g), W=dict(score=score(g.tan_W.to_numpy(), tr)), tiers={})
        for f in fams:
            for suf, col in (('', f'u_{f}'), ('r', f'ur_{f}'), ('e', f'ue_{f}'), ('q', f'uq_{f}')):
                u = g[col].to_numpy()
                name = f'{E.CODES[f]}{suf}_{f}'
                V['tiers'][name] = dict(
                    implied_v=implied_v(u, tr), coverage=float(np.isfinite(u).mean()),
                    naive=score(u / v, tr), H0=score(cv_fit(u, tr, eid, 'H0'), tr), H2a=score(cv_fit(u, tr, eid, 'H2a'), tr))
        res['views'][view] = V
        print(f'[{view}] {len(g)} planes; W s68 {V["W"]["score"]["all"]["s68"]}')
        for name, t in V['tiers'].items():
            print(f'   {name:14s} cov {t["coverage"]:.2f}  v {t["implied_v"] or float("nan"):5.1f}  '
                  f'naive s68 {t["naive"]["all"]["s68"] or float("nan"):.2f} steep {t["naive"]["all"]["steep"] or float("nan"):+.2f}  '
                  f'H0 {t["H0"]["all"]["s68"] or float("nan"):.2f}  H2a {t["H2a"]["all"]["s68"] or float("nan"):.2f}')
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
