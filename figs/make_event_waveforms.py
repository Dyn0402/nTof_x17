#!/usr/bin/env python3
"""Paper IIa, figure event_waveforms (section 3, the motivating display): one cosmic track on chamber A, both views.

Each panel is the strip x time map of one view's window (ADC, pedestal and common mode subtracted), with
  - the per-strip leading-edge hit times (V0 of hits_daq/emulate.py: the DREAM hit finder, 30 % leading edge) and the
    amplitude-weighted line through them (hits_daq/tiers.ladder_u, the production-style hit ladder),
  - the track of the waveform forward-model fit (p0, w, t0 of the pass), drawn at the time its charge crosses the
    30 % leading edge of the pulse response (arrival time + the template's 30 % rise time),
  - the reference track's slope (w_ref = kw v tan_ref + w0, the angle scale of universal.py), through the fit's
    mid-column point (the tables hold the reference angle, not its position in this frame).
The hit ladder is shown on purpose: the figure's point is that it reads steeper than the track.

Event rule (no picking by eye): among events of Runs 1 and 2 with both views in the tiers tables, 0.14 <= |tan_ref| < 0.28
in both views, >= 5 hit strips and no censored (saturated) strip in both, take the event that minimises
  sum over views of |r - med(r)| / MAD(r) + |d - med(d)| / MAD(d),
r = hit-ladder slope / reference slope, d = tan_W - tan_ref, medians and MADs per view over that same set: the most
typical event in both the hit bias and the waveform-fit residual.

    .venv/bin/python papers/paper2/figs/make_event_waveforms.py
Inputs: $WORK_DATA/paper2_hits/{windows/*.pkl, eos_tables/tiers_*.parquet, bundles/det3/calib_bundle}.
Writes papers/paper2/figs/out/event_waveforms.{pdf,png,json}.
"""
import glob
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'papers'), os.path.join(REPO, 'papers', 'paper2', 'hits_daq'), REPO]
import paper_style as ps  # noqa: E402
import emulate as E  # noqa: E402
from tiers import ladder_u  # noqa: E402

WD = os.path.join(os.environ.get('WORK_DATA', os.path.expanduser('~/work_data')), 'paper2_hits')
OUT = os.path.join(HERE, 'out')
L_GAP = 28.08
TAN_LO, TAN_HI = 0.14, 0.28
RUNS = (('Run 1', 'sat_det3'), ('Run 2', 'p2_det3'))
C_HIT, C_FIT, C_REF = '#D55E00', '#000000', '#1b9e77'


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def choose(meta):
    kw, w0, v = meta['kw'], meta['w0'], meta['v_drift'] / 1000.0
    cand = []
    for run, key in RUNS:
        T = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{WD}/eos_tables/tiers_{key}_p*.parquet'))])
        T = T[np.isfinite(T.tan_ref) & np.isfinite(T.u_dream)]
        T = T[~T.duplicated(['event_id', 'view'], keep=False)]
        T['w_ref'] = [kw[vw] * v * t + w0[vw] * 1e-3 for vw, t in zip(T.view, T.tan_ref)]
        T['r'] = T.u_dream / T.w_ref
        T['d'] = T.tan_W - T.tan_ref
        T['run'] = run
        cand.append(T)
    T = pd.concat(cand)
    T = T[(T.tan_ref.abs() >= TAN_LO) & (T.tan_ref.abs() < TAN_HI) & (T.n_dream >= 5)]
    X, Y = T[T.view == 'x'].set_index(['run', 'event_id']), T[T.view == 'y'].set_index(['run', 'event_id'])
    both = X.index.intersection(Y.index)
    X, Y = X.loc[both], Y.loc[both]
    stat = {}
    score = np.zeros(len(both))
    for vw, G in (('x', X), ('y', Y)):
        stat[vw] = dict(med_r=float(np.median(G.r)), mad_r=float(mad(G.r)), med_d=float(np.median(G.d)),
                        mad_d=float(mad(G.d)), n=int(len(G)))
        score += np.abs(G.r.to_numpy() - stat[vw]['med_r']) / stat[vw]['mad_r'] \
            + np.abs(G.d.to_numpy() - stat[vw]['med_d']) / stat[vw]['mad_d']
    order = np.argsort(score)
    return [(both[i], float(score[i])) for i in order], X, Y, stat


def find_planes(key, eid):
    out = {}
    for f in sorted(glob.glob(f'{WD}/windows/{key}_p*.pkl')):
        for p in pickle.load(open(f, 'rb'))['planes']:
            if p['event_id'] == eid:
                out.setdefault(p['view'], []).append(p)
    return out


def main():
    from wft import model as wm
    from wft.calib import CalibrationBundle
    meta = pickle.load(open(sorted(glob.glob(f'{WD}/windows/sat_det3_p0of4.pkl'))[0], 'rb'))['meta']
    cal = CalibrationBundle.load(os.path.join(WD, 'bundles', 'det3', 'calib_bundle'))
    wm.use_calibration(cal)
    v = meta['v_drift'] / 1000.0
    ranked, X, Y, stat = choose(meta)
    key_of = dict(RUNS)
    chosen = None
    for (run, eid), sc in ranked[:50]:
        P = find_planes(key_of[run], int(eid))
        if all(len(P.get(vw, [])) == 1 and not np.any(P[vw][0]['censor']) for vw in 'xy'):
            chosen = (run, int(eid), sc, P)
            break
    run, eid, sc, P = chosen
    ps.use()
    fig, axs = plt.subplots(1, 2, figsize=(6.1, 3.5), sharey=True, gridspec_kw=dict(wspace=0.06))
    vmax = max(np.percentile(P[vw][0]['W'], 99.7) for vw in 'xy')
    t_lo = min(P[vw][0]['t0'] for vw in 'xy')
    num = dict(rule=__doc__.split('Event rule')[1].split('\n\n')[0].strip(), run=run, event_id=eid, score=sc,
               n_candidates=int(len(ranked)), selection_stats=stat, views={})
    for ax, vw, lab in zip(axs, 'xy', ('(a)', '(b)')):
        p = P[vw][0]
        W, pos = p['W'].astype(float), p['pos'].astype(float)
        o = np.argsort(pos)
        W, pos, nz = W[o], pos[o], p['noise'].astype(float)[o]
        h = E.dream(W, nz, nsig=5.0)
        u, nh = ladder_u(pos, h['t'], h['a'])
        row = (X if vw == 'x' else Y).loc[(run, eid)]
        tmpl = wm.TMPL['x' if cal.hyper.get('rc_tmpl', 'x') in ('x', 1, 1.0) else vw]
        kpk = int(np.argmax(tmpl))
        t_rise = float(np.interp(0.3 * tmpl[kpk], tmpl[:kpk + 1], wm.TGRID[:kpk + 1]))
        ts = np.arange(W.shape[1]) * E.SNS
        x0 = pos.mean()
        pe = np.concatenate([[pos[0] - 0.39], 0.5 * (pos[1:] + pos[:-1]), [pos[-1] + 0.39]]) - x0
        te = np.concatenate([ts - 30.0, [ts[-1] + 30.0]])
        im = ax.pcolormesh(pe, te, W.T, cmap='Blues', vmin=0, vmax=vmax, rasterized=True)
        k = np.isfinite(h['t'])
        ax.plot(pos[k] - x0, h['t'][k], 'o', ms=4.2, mfc='white', mec=C_HIT, mew=1.2, zorder=6, label='hit time per strip')
        # the amplitude-weighted hit line (through the weighted centroid, slope 1/u)
        xm, tm = np.average(pos[k], weights=h['a'][k]), np.average(h['t'][k], weights=h['a'][k])
        xx = np.array([pos[k].min() - 0.5, pos[k].max() + 0.5])
        ax.plot(xx - x0, tm + (xx - xm) / u, '-', c=C_HIT, lw=1.6, zorder=5, label='line through the hits')
        # waveform fit: charge of drift time s lands at p0 + w s at t0 + s
        s = np.array([0.0, L_GAP / v])
        ax.plot(p['p0'] + p['w'] * s - x0, p['t0'] + s + t_rise, '-', c=C_FIT, lw=1.6, zorder=7, label='waveform fit')
        pm, tmid = p['p0'] + p['w'] * s.mean(), p['t0'] + s.mean() + t_rise
        ax.plot(pm + row.w_ref * (s - s.mean()) - x0, tmid + (s - s.mean()), '--', c=C_REF, lw=1.6, zorder=8,
                label='reference track')
        ax.set_xlim(pe[0], pe[-1]); ax.set_ylim(t_lo - 150, t_lo + L_GAP / v + 640)
        ax.grid(False)
        ax.set_xlabel(f'{vw} strip position [mm]')
        if vw == 'x':
            ax.set_ylabel('time [ns]')
        ax.text(0.03, 0.97, f'{lab} {vw.upper()} view', transform=ax.transAxes, fontweight='bold', va='top')
        ang = lambda w_: float(np.degrees(np.arctan((w_ * 1e3 - meta['w0'][vw]) / (meta['kw'][vw] * meta['v_drift']))))
        num['views'][vw] = dict(tan_ref=float(row.tan_ref), angle_ref_deg=float(np.degrees(np.arctan(row.tan_ref))),
                                angle_fit_deg=ang(p['w']), angle_hits_deg=ang(u), u_hits=float(u), u_table=float(row.u_dream),
                                w_fit=float(p['w']), w_ref=float(row.w_ref), ratio_hits_ref=float(u / row.w_ref),
                                ratio_fit_ref=float(p['w'] / row.w_ref), n_hits=int(nh), n_strips=int(len(pos)),
                                t_rise_ns=t_rise, t0=float(p['t0']), chi2=float(p['chi2']), dof=int(p['dof']))
        a = num['views'][vw]
        ax.text(0.97, 0.03, f"reference {abs(a['angle_ref_deg']):.1f}°\nwaveform fit {abs(a['angle_fit_deg']):.1f}°\n"
                f"hits {abs(a['angle_hits_deg']):.1f}°", transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5,
                bbox=dict(fc='white', ec='none', alpha=0.85, pad=2))
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=7.8, columnspacing=1.4)
    cb = fig.colorbar(im, ax=axs, fraction=0.035, pad=0.02)
    cb.set_label('amplitude [ADC]')
    ps.save(fig, os.path.join(OUT, 'event_waveforms'))
    json.dump(num, open(os.path.join(OUT, 'event_waveforms.json'), 'w'), indent=1)
    print(json.dumps(num, indent=1))


if __name__ == '__main__':
    main()
