#!/usr/bin/env python3
"""vt_apply.py [data root] [--out DIR] -- v(t) models, their held-out test, the corrected tables and the consequences.

Models of r(t) = v(t) / v_row (the factor the angle mapping's v is multiplied by at the event's clock time):
  const   pass 1: r = 1
  blocks  piecewise linear through the row's own blocks (vt_measure.py), clamped at the ends (wft.reco.v_rel_t form)
  linear  a straight line through the row's own blocks
  night   one curve per chamber over the whole 6-26 night, v(t) = v_inf - D exp(-t / tau), fitted to the blocks of every
          row of the chamber; r = v(t) / (its mean over the row's kw-window events)
Held-out: every model is built from the even event_id half (results/blocks.json 'even') and scored on the odd half:
the residual scale of the odd half's own time blocks after the correction (rms, chi2/ndf), and the angle residual s68
(fitted - M3, all matched events).  The model chosen is then rebuilt on all events and applied to the tables.

Writes results/models.json, results/heldout.json, results/consequences.json, and with --out one corrected table per
row: <out>/<run>/<subrun>/<det>/events_vt.parquet (event_id, acq_tag, t_h, v_rel, x/y_tan_vt, x/y_theta_deg_vt,
x/y_mid_vt, z_end_vt) + vt_map.json (the wft v_map['t'] block: clock, t_h, v_rel).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from vt_common import (RES, rows, load_row, ladder, kw_window, blocks_equal_count, tan_map, s68, data_root, NAME,
                       T_REF)
from vt_measure import N_PER

GAP = {'mx17_6': 28.0, 'mx17_7': 29.44}            # physics_inputs.json gap_mm
OVERNIGHT = ('short_run', 'longer_run', 'long_run')
SYS_FLOOR = 0.01                                     # 1 % per-block floor in the night fit (HV / view scatter)


def vlad(b):
    """a block's ladder v (mean of the two views where both exist) and its error."""
    v = [b[p]['v_lad'] for p in ('x', 'y') if np.isfinite(b[p].get('v_lad', np.nan))]
    e = [b[p]['v_lad_err'] for p in ('x', 'y') if np.isfinite(b[p].get('v_lad', np.nan))]
    if not v:
        return np.nan, np.nan
    return float(np.mean(v)), float(np.sqrt(np.sum(np.square(e))) / len(e))


def expo(t, vinf, D, tau):
    return vinf - D * np.exp(-t / tau)


def fit_night(blocks_rows, det, half):
    t, v, e = [], [], []
    for r in blocks_rows:
        if r['det'] != det:
            continue
        for b in r[half]['blocks']:
            vv, ee = vlad(b)
            if np.isfinite(vv) and np.isfinite(b['t']):
                t.append(b['t']); v.append(vv); e.append(np.hypot(ee, SYS_FLOOR * vv))
    t, v, e = map(np.asarray, (t, v, e))
    p, C = curve_fit(expo, t, v, p0=(v.max(), v.max() - v.min(), 4.0), sigma=e, absolute_sigma=True, maxfev=20000)
    chi2 = float(np.sum(((v - expo(t, *p)) / e) ** 2))
    return dict(vinf=p[0], D=p[1], tau=p[2], err=np.sqrt(np.diag(C)).tolist(), chi2=chi2, ndf=len(t) - 3,
                n=len(t), t=t.tolist(), v=v.tolist(), e=e.tolist())


def row_model(name, r, half, night, t_ref_events):
    """r(t) as a function for one row, built from `half` ('even', 'all')."""
    bl = [b for b in r[half]['blocks'] if np.isfinite(vlad(b)[0]) and np.isfinite(b['t'])]
    vr = vlad(dict(x=r[half]['row']['x'], y=r[half]['row']['y']))[0]
    if name == 'const' or len(bl) < 2 and name in ('blocks', 'linear'):
        return lambda t: np.ones_like(np.asarray(t, float))
    tb = np.array([b['t'] for b in bl]); rb = np.array([vlad(b)[0] for b in bl]) / vr
    if name == 'blocks':
        return lambda t: np.interp(t, tb, rb)
    if name == 'linear':
        eb = np.array([vlad(b)[1] for b in bl]) / vr
        c = np.polyfit(tb, rb, 1, w=1 / eb)
        return lambda t: np.polyval(c, t)
    if name == 'night':
        f = night[r['det']]
        norm = float(np.mean(expo(t_ref_events, f['vinf'], f['D'], f['tau'])))
        return lambda t: expo(np.asarray(t, float), f['vinf'], f['D'], f['tau']) / norm
    raise ValueError(name)


def views_half(R, half):
    out = {}
    for p in ('x', 'y'):
        v = R['views'][p]
        k = np.ones(len(v['eid']), bool) if half is None else (v['eid'] % 2 == half)
        out[p] = {q: np.asarray(a)[k] for q, a in v.items()}
        out[p]['tp'] = (out[p]['tan_ref'] + out[p]['a']) * out[p]['S']
    return out


def score(R, V, rfun):
    """residual scale of the blocks of V after r(t), and the angle residual s68 [deg] per view."""
    cal = R['cal']
    pb = max(('x', 'y'), key=lambda p: kw_window(V[p]['tp']).sum())
    edges = blocks_equal_count(V[pb]['t_h'], kw_window(V[pb]['tp']), N_PER)
    res = {}
    for p in ('x', 'y'):
        rr = rfun(V[p]['t_h'])
        w0 = cal.w0[p]
        wc = w0 + (V[p]['w'] - w0) / rr                   # the corrected mapping, as a corrected w on the constant one
        _, kw_row, _, _ = ladder(cal, wc, V[p]['tp'])
        eps, err = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            k = (V[p]['t_h'] >= lo) & (V[p]['t_h'] < hi)
            _, kw, n, e = ladder(cal, wc[k], V[p]['tp'][k])
            if np.isfinite(kw):
                eps.append(kw / kw_row - 1); err.append(e / kw_row)
        eps, err = np.array(eps), np.array(err)
        tan = tan_map(cal, p, V[p]['w'], V[p]['a'], V[p]['S'], rr)
        dth = np.degrees(np.arctan(tan) - np.arctan(V[p]['tan_ref']))
        res[p] = dict(n_blocks=len(eps), eps_rms=float(np.sqrt(np.mean(eps ** 2))) if len(eps) else np.nan,
                      eps_max=float(np.max(np.abs(eps))) if len(eps) else np.nan,
                      chi2=float(np.sum((eps / err) ** 2)) if len(eps) else np.nan,
                      s68_deg=s68(dth - np.nanmedian(dth)), n=int(np.isfinite(dth).sum()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default=None)
    ap.add_argument('--out', default=None, help='write the corrected tables here (never into Analysis_pass1)')
    ap.add_argument('--model', default=None, help='force the applied model (default: the held-out winner)')
    a = ap.parse_args()
    root = a.root or data_root()
    if a.out and 'Analysis_pass1' in os.path.abspath(a.out):
        ap.error('--out must not be inside Analysis_pass1 (read-only)')
    B = json.load(open(os.path.join(RES, 'blocks.json')))
    Bk = {(r['run'], r['subrun'], r['det']): r for r in B}
    night = {h: {d: fit_night(B, d, h) for d in ('mx17_6', 'mx17_7')} for h in ('even', 'all')}
    for d in ('mx17_6', 'mx17_7'):
        f = night['all'][d]
        print(f"{NAME[d]} night: v_inf {f['vinf']:.2f} D {f['D']:.2f} tau {f['tau']:.2f} +- {f['err'][2]:.2f} h  "
              f"chi2/ndf {f['chi2']:.1f}/{f['ndf']}")
    json.dump({h: {d: {k: v for k, v in f.items() if k not in ('t', 'v', 'e')} | dict(t=f['t'], v=f['v'], e=f['e'])
                   for d, f in night[h].items()} for h in night},
              open(os.path.join(RES, 'models.json'), 'w'), indent=1, default=float)
    models = ('const', 'linear', 'blocks', 'night')
    held, Rs = [], {}
    for run, sub, det, wdir in rows(root):
        R = load_row(wdir, root); Rs[(run, sub, det)] = R
        r = Bk[(run, sub, det)]
        Ve, Vo = views_half(R, 0), views_half(R, 1)
        tref = np.concatenate([Ve[p]['t_h'][kw_window(Ve[p]['tp'])] for p in ('x', 'y')])
        h = dict(run=run, subrun=sub, det=det, t_start=r['t_start'], t_end=r['t_end'])
        for m in models:
            h[m] = score(R, Vo, row_model(m, r, 'even', night['even'], tref))
        held.append(h)
        if sub in OVERNIGHT:
            print(f"{NAME[det]} {sub:10s} held-out eps_rms % / s68 deg:  " + '  '.join(
                f"{m} " + '/'.join(f"{100 * h[m][p]['eps_rms']:.1f}:{h[m][p]['s68_deg']:.3f}" for p in 'xy')
                for m in models))
    json.dump(held, open(os.path.join(RES, 'heldout.json'), 'w'), indent=1, default=float)
    # the winner: lowest summed held-out chi2 over the overnight rows with >= 3 blocks, both chambers
    tot = {m: sum(h[m][p]['chi2'] for h in held for p in 'xy'
                  if h['subrun'] in OVERNIGHT and np.isfinite(h[m][p]['chi2'])) for m in models}
    ndf = sum(h['const'][p]['n_blocks'] for h in held for p in 'xy' if h['subrun'] in OVERNIGHT)
    win = a.model or min(models, key=lambda m: tot[m])
    print('held-out chi2 (overnight rows, ndf ~', ndf, '):', {m: round(v, 1) for m, v in tot.items()}, '-> applied:', win)
    cons = dict(model=win, heldout_chi2=tot, heldout_nblocks=ndf, rows=[])
    for (run, sub, det), R in Rs.items():
        r, cal, df = Bk[(run, sub, det)], R['cal'], R['df']
        Va = views_half(R, None)
        tref = np.concatenate([Va[p]['t_h'][kw_window(Va[p]['tp'])] for p in ('x', 'y')])
        rf = row_model(win, r, 'all', night['all'], tref)
        rr = rf(df.t_h.to_numpy())
        out = pd.DataFrame(dict(event_id=df.index.to_numpy(), acq_tag=df.acq_tag.to_numpy(), t_h=df.t_h.to_numpy(),
                                v_rel=rr))
        c = dict(run=run, subrun=sub, det=det, t_start=r['t_start'], t_end=r['t_end'], v_drift=cal.v_drift,
                 v_rel_min=float(np.nanmin(rr)), v_rel_max=float(np.nanmax(rr)))
        for p in ('x', 'y'):
            ok = df[f'{p}_ok'].to_numpy()
            w = df[f'{p}_w'].to_numpy() * 1e3
            t0 = tan_map(cal, p, w, df[f'{p}_a'].to_numpy(), df[f'{p}_S'].to_numpy())
            t1 = tan_map(cal, p, w, df[f'{p}_a'].to_numpy(), df[f'{p}_S'].to_numpy(), rr)
            t1 = np.where(ok, t1, np.nan); t0 = np.where(ok, t0, np.nan)
            out[f'{p}_tan_vt'] = t1
            out[f'{p}_theta_deg_vt'] = np.degrees(np.arctan(t1))
            out[f'{p}_mid_vt'] = df[f'{p}_p0'].to_numpy() + 15.0 * t1
            sl = ok & np.isfinite(t0) & (np.abs(t0) < 1.0)
            dth = np.degrees(np.arctan(t1[sl]) - np.arctan(t0[sl]))
            dmid = 15.0 * (t1[sl] - t0[sl])
            c[p] = dict(dtheta_rms_deg=float(np.sqrt(np.mean(dth ** 2))), dtheta_p95_deg=float(np.percentile(np.abs(dth), 95)),
                        dmid_rms_mm=float(np.sqrt(np.mean(dmid ** 2))), dmid_p95_mm=float(np.percentile(np.abs(dmid), 95)))
            # held-out (odd half) and in-sample angle residuals vs M3, before / after
            for nm, V in (('all', Va),):
                for lab, rfun in (('before', lambda t: np.ones_like(t)), ('after', rf)):
                    tt = tan_map(cal, p, V[p]['w'], V[p]['a'], V[p]['S'], rfun(V[p]['t_h']))
                    d = np.degrees(np.arctan(tt) - np.arctan(V[p]['tan_ref']))
                    c[p][f's68_{lab}_deg'] = s68(d - np.nanmedian(d))
        # column depth: v_drift x mean kw x r x q_uend (the column duration), both views' mean, single tracks
        km = 0.5 * (cal.kw['x'] + cal.kw['y'])
        uend = df[['x_q_uend', 'y_q_uend']].where(df[['x_ok', 'y_ok']].to_numpy()).mean(axis=1).to_numpy()
        single = (df.n_tracks <= 1).to_numpy() if 'n_tracks' in df else np.ones(len(df), bool)
        z0 = cal.v_drift * km * uend * 1e-3
        out['z_end_vt'] = z0 * rr
        sel = single & np.isfinite(uend)
        edges = blocks_equal_count(df.t_h.to_numpy(), sel, 400)
        zb = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            k = sel & (df.t_h.to_numpy() >= lo) & (df.t_h.to_numpy() < hi)
            if k.sum() >= 50:
                zb.append(dict(t=float(np.median(df.t_h.to_numpy()[k])), z_before=float(np.median(z0[k])),
                               z_after=float(np.median((z0 * rr)[k])), n=int(k.sum())))
        c['z_end_blocks'] = zb
        c['gap_mm'] = GAP[det]
        # gas signature: charge-arrival shape (u50 / uend, v-independent) and total charge per block
        f50 = (df.x_q_u50 / df.x_q_uend).to_numpy()
        qs = (df.x_q_sum + df.y_q_sum).to_numpy()
        gb = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            k = sel & (df.t_h.to_numpy() >= lo) & (df.t_h.to_numpy() < hi) & np.isfinite(f50) & np.isfinite(qs)
            if k.sum() >= 50:
                gb.append(dict(t=float(np.median(df.t_h.to_numpy()[k])), u50_uend=float(np.median(f50[k])),
                               q_sum=float(np.median(qs[k])), n=int(k.sum())))
        c['gas_blocks'] = gb
        cons['rows'].append(c)
        if a.out:
            d = os.path.join(a.out, run, sub, det)
            os.makedirs(d, exist_ok=True)
            out.to_parquet(os.path.join(d, 'events_vt.parquet'), index=False)
            tg = np.linspace(r['t_start'], r['t_end'], 25)
            json.dump(dict(model=win, clock=f'hours since {T_REF:%Y-%m-%d %H:%M} CERN local: acq_tag stem + DREAM '
                                            'timestamp x 10 ns', t_h=tg.tolist(), v_rel=rf(tg).tolist(),
                           v_drift=cal.v_drift, source='rerun/vt_within_rows/vt_apply.py', pass1_table=R['wdir']),
                      open(os.path.join(d, 'vt_map.json'), 'w'), indent=1)
        if sub in OVERNIGHT:
            print(f"{NAME[det]} {sub:10s} r {c['v_rel_min']:.3f}-{c['v_rel_max']:.3f}  " + '  '.join(
                f"{p}: dth rms {c[p]['dtheta_rms_deg']:.2f} p95 {c[p]['dtheta_p95_deg']:.2f} deg, dmid p95 "
                f"{c[p]['dmid_p95_mm']:.2f} mm, s68 {c[p]['s68_before_deg']:.3f}->{c[p]['s68_after_deg']:.3f}"
                for p in 'xy'))
    json.dump(cons, open(os.path.join(RES, 'consequences.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
