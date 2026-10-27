#!/usr/bin/env python3
"""summary.py -- the headline numbers of the bow study from the results/*.json, in one place (results/summary.json).
Used by the slide notes, REPORT.md and the paper section, so every quoted number has one source."""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fbcommon import RES, DETS, DATASETS

RIM = 1.5     # 150 mm from the centre, in units of 100 mm


def J(n):
    p = f'{RES}/{n}'
    return json.load(open(p)) if os.path.exists(p) else None


def ds_summary(name):
    X = J(f'xcheck_{name}.json')
    if X is None:
        return None
    k4 = np.array([X['pos_x']['kuu'], X['pos_x']['kww'], X['pos_y']['kuu'], X['pos_y']['kww']])
    e4 = np.array([X['pos_x']['kuu_err'], X['pos_x']['kww_err'], X['pos_y']['kuu_err'], X['pos_y']['kww_err']])
    kp = np.array([X['pospin_x'].get('kuu', np.nan), X['pospin_x'].get('kww', np.nan),
                   X['pospin_y'].get('kuu', np.nan), X['pospin_y'].get('kww', np.nan)])
    w = 1 / e4 ** 2
    kbar = float(np.sum(w * k4) / np.sum(w)); kbar_e = float(1 / np.sqrt(np.sum(w)))
    # scatter-inflated error (the four terms are not one number if the dish is not round)
    chi2 = float(np.sum(w * (k4 - kbar) ** 2)); kbar_e2 = kbar_e * max(1.0, np.sqrt(chi2 / 3))
    P = J(f'profiles_{name}.json')
    out = dict(name=name, det=X['det'], slot=X['slot'], note=X['note'], n=X['n_events'],
               k4=k4.tolist(), k4_err=e4.tolist(), kbar=kbar, kbar_err=kbar_e2,
               depth_mm=0.5 * kbar * RIM ** 2, depth_err=0.5 * kbar_e2 * RIM ** 2,
               k4_pinned=kp.tolist(), kbar_pinned=float(np.nanmean(kp)),
               xy_pull=[(X['pos_x'][k] - X['pos_y'][k]) / np.hypot(X['pos_x'][k + '_err'], X['pos_y'][k + '_err'])
                        for k in ('kuu', 'kww', 'kuw')],
               scale_bu=X['pos_x']['bu'], scale_bw=X['pos_y']['bw'],
               ang_cu_x=X['ang_x']['cu'], ang_cw_y=X['ang_y']['cw'], B_x=X['ang_x']['B'], B_y=X['ang_y']['B'],
               c0_x=X['ang_x']['c0'], c0_y=X['ang_y']['c0'])
    # r_f from the 2-D fit with the local ladder slope as v
    out['r_f2d'] = [X['ang_x']['cu'] * 100 / (X['ang_x']['B'] * X['pos_x']['kuu']),
                    X['ang_y']['cw'] * 100 / (X['ang_y']['B'] * X['pos_y']['kww'])]
    if P:
        out['r_f1d'] = [P[q]['r_f'] if P.get(q) else np.nan for q in 'xy']
        out['w0_slope'] = [P[q]['w0_slope'] if P.get(q) else np.nan for q in 'xy']
    # width route: slope of t*(p) against the angle-route prediction -w0(p)/b
    ws = []
    for p in 'xy':
        Wr = X.get(f'width_{p}')
        if not Wr or P is None or not P.get(p):
            ws.append(None); continue
        R = np.array(Wr, float); g = np.isfinite(R[:, 1])
        if g.sum() < 3:
            ws.append(None); continue
        sl = np.polyfit(R[g, 0], R[g, 1], 1, w=1 / np.maximum(R[g, 2], 0.003))[0]
        pred = -P[p]['w0_slope'] / P[p]['b_med']
        ws.append(dict(slope_per_m=float(sl * 1e3), pred_per_m=float(pred * 1e3)))
    out['width'] = ws
    G = J(f'gap_{name}.json')
    if G:
        out['gap'] = {p: {k: dict(kuu=G[p][k]['gap_fit'][3], kww=G[p][k]['gap_fit'][4],
                                  kuu_err=G[p][k]['gap_fit_err'][3], kww_err=G[p][k]['gap_fit_err'][4])
                          for k in ('uend', 'uend_q', 'u90m50', 'u90m50_q')} for p in 'xy'}
    return out


def main():
    S = {n: ds_summary(n) for n in DATASETS}
    S = {k: v for k, v in S.items() if v}
    gate = J('gate_bow.json'); tr = J('gate_transfer.json')
    S['_gate'] = gate; S['_transfer'] = tr; S['_phys'] = J('physmodel.json')
    json.dump(S, open(f'{RES}/summary.json', 'w'), indent=1, default=float)
    for n, s in S.items():
        if n.startswith('_'):
            continue
        print(f'{n:14s} [{s["slot"]:6s}] depth {s["depth_mm"]:.2f}±{s["depth_err"]:.2f} mm  kbar {s["kbar"]:.2f}  pinned '
              f'{s["kbar_pinned"]:.2f}  r_f1d {s.get("r_f1d")}  xy pulls ' + ' '.join(f'{x:+.1f}' for x in s['xy_pull']))


if __name__ == '__main__':
    main()
