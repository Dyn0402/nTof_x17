#!/usr/bin/env python3
"""bow_pass1.py build --out-dir D | fig --in-dir D -- the Paper I bow figures (bow_profiles, bow_closure) re-made on
the pass-1 golden rows, with and without the in-row v(t) (PASS1_RESOLUTION_2026-10-31.md §6).

build  (lxplus; READ-ONLY on Analysis_pass1) per golden row, the per-event table in the layout of
       rerun/model_audit/ws_v_w0/build_table (what fb_bow_maps/fbcommon.load_path reads): the M3 ray at the mesh and its
       tangents in the chamber's raw strip frame (the effbudget rays: the row's own waveform alignment), joined to the
       pass table; plus v_rel per event (vt_correction, rows of vt_corrected_rows.csv) and the row's angle constants
       -> D/<det>.parquet, D/<det>.bundle.json
fig    (anywhere) fb_bow_maps' profiles (profiles.run's estimator, unchanged) on each table in two arms:
         pass  w as reconstructed
         vt    w -> w0 + (w - w0) / r(t): the transverse speed a constant v would have given (r = v_rel; the w that
               maps to the same angle under the row's mean v), so the ladder slope b and w0(p) are read at one v
       -> results/bow/profiles_<det>_<arm>.json, figures/bow_profiles.pdf, bow_closure.pdf (the vt arm: the pass arm
       is identical on det2/det3, which carry no v(t) map), and figures/bow_profiles_pass.pdf for the comparison.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('RES_CODE') or os.path.abspath(os.path.join(HERE, '..', '..'))
P1 = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
EFF = '/eos/user/d/dneff/x17rerun/paper1_eff'
GOLD = {'det2': 'o22_long_det2', 'det3': 'sat_det3', 'det4': 'g_det4', 'det6': 'g_det6_long', 'det7': 'g_det7_long'}
COLS = ('p0', 'w', 't0', 'q_u50', 'q_u90', 'q_uend', 'q_sum', 'n_seed', 'n_strips', 'chi2', 'dof', 'ftst', 'ok',
        'n_flagged_strips')


def rel(r):
    return f"{r['run']}/{r['subrun']}/{r['det']}"


def build(out_dir):
    reg = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'pass_registry.csv')))}
    vt = {r['key']: r for r in csv.DictReader(open(os.path.join(REPO, 'rerun', 'vt_all', 'results', 'vt_corrected_rows.csv')))}
    os.makedirs(out_dir, exist_ok=True)
    for d, key in GOLD.items():
        r = reg[key]
        wdir = os.path.join(P1, rel(r), 'wft')
        m = json.load(open(os.path.join(wdir, 'events.meta.json')))
        rays = pd.read_parquet(os.path.join(EFF, 'budget', rel(r), f'rays_{d}.parquet')).drop_duplicates('event_id', keep=False)
        df = pd.read_parquet(os.path.join(wdir, 'events.parquet')).drop_duplicates('event_id', keep=False)
        keep = ['event_id'] + [f'{p}_{c}' for p in 'xy' for c in COLS if f'{p}_{c}' in df.columns]
        T = rays[['event_id', 'mesh_x', 'mesh_y', 'tan_x', 'tan_y']].rename(
            columns=dict(mesh_x='ref_mesh_x', mesh_y='ref_mesh_y', tan_x='tan_ref_x', tan_y='tan_ref_y'))
        T = T.merge(df[keep], on='event_id', how='inner')
        T['v_rel'] = 1.0
        if key in vt:
            v = pd.read_parquet(vt[key]['table']).drop_duplicates('event_id').set_index('event_id')['v_rel']
            T['v_rel'] = v.reindex(T.event_id).fillna(1.0).to_numpy()
        T.to_parquet(os.path.join(out_dir, f'{d}.parquet'))
        ac = m['angle_constants']
        json.dump(dict(key=key, v_drift=float(m['bundle']['v_drift']), kw=ac['kw'], w0=ac['w0'], vt=key in vt,
                       bow=m.get('bow')), open(os.path.join(out_dir, f'{d}.bundle.json'), 'w'), indent=1)
        print(d, key, len(T), 'vt' if key in vt else '')


def fig(in_dir):
    fb = os.path.join(REPO, 'rerun', 'model_audit', 'followups', 'fb_bow_maps')
    sys.path.insert(0, fb)
    import fbcommon as F
    import profiles as PR
    res = os.path.join(HERE, 'results', 'bow')
    figd = os.path.join(HERE, 'figures')
    os.makedirs(res, exist_ok=True); os.makedirs(figd, exist_ok=True)
    out = {}
    for d in F.DETS:
        b = json.load(open(os.path.join(in_dir, f'{d}.bundle.json')))
        for arm in ('pass', 'vt'):
            T = F.load_path(os.path.join(in_dir, f'{d}.parquet'), b, d, name=f'{d}_{arm}')
            if arm == 'vt':
                vr = pd.read_parquet(os.path.join(in_dir, f'{d}.parquet')).set_index('event_id')['v_rel'].reindex(T.index).to_numpy()
                for p in 'xy':
                    w0 = b['w0'][p]
                    T[f'{p}_wv'] = w0 + (T[f'{p}_wv'].to_numpy() - w0) / vr
            # profiles.run on a table already loaded: the same estimator, its output redirected
            PR.load = lambda name, T=T: T
            PR.DATASETS = {f'{d}_{arm}': None}
            PR.RES = res
            PR.run(f'{d}_{arm}')
            out[(d, arm)] = json.load(open(os.path.join(res, f'profiles_{d}_{arm}.json')))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    COL = {'det2': '#1f5fa8', 'det3': '#2f8a5b', 'det4': '#c63d4f', 'det6': '#7a52b8', 'det7': '#c99318'}
    for arm, suffix in (('vt', ''), ('pass', '_pass')):
        f1, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for ax, p in zip(axs, 'xy'):
            for d in F.DETS:
                pr = out[(d, arm)].get(p)
                if not pr:
                    continue
                pos, h, he = np.array(pr['pos']), np.array(pr['h']), np.array(pr['h_err'])
                c = np.polyval(pr['h_par'], 0.0)
                ok = np.abs(h - c) < 4
                ax.errorbar(pos[ok], (h - c)[ok], yerr=he[ok], fmt='o-', ms=3, lw=1.2, capsize=1.5, color=COL[d], label=d)
            ax.axhline(0, color='grey', lw=0.6, ls=':')
            ax.set_xlabel(f'track {p} at mid-gap [mm]'); ax.set_title(f'{p.upper()} strips', fontsize=10)
        axs[0].set_ylabel(r'mesh height $h - h_{\rm centre}$ [mm]')
        axs[0].legend(fontsize=7, ncol=2, frameon=False)
        f1.tight_layout()
        for ext in ('pdf', 'png'):
            f1.savefig(os.path.join(figd, f'bow_profiles{suffix}.{ext}'), dpi=200)
        plt.close(f1)
        f2, ax = plt.subplots(figsize=(4.4, 4.0))
        xs = np.linspace(0, 10, 10)
        ax.plot(xs, xs, 'k--', lw=1, label=r'$r_f = 1$ (cathode follows)')
        ax.plot(xs, 0.5 * xs, 'k:', lw=1, label=r'$r_f = 1/2$ (flat cathode)')
        for d in F.DETS:
            for p, mk in (('x', 'o'), ('y', 'D')):
                P = out[(d, arm)].get(p)
                if not P:
                    continue
                ax.plot(P['b_med'] * P['h_curv'] * 1e3, P['w0_slope'] * 1e3, mk, color=COL[d], ms=6,
                        label=(d + (' (circle X, diamond Y)' if d == 'det2' else '')) if p == 'x' else None)
        ax.set_xlim(0, 10); ax.set_ylim(0, 8)
        ax.set_xlabel(r'$v\,h^{\prime\prime}$ from positions [µm/ns per m]')
        ax.set_ylabel(r'measured $dw_0/dp$ [µm/ns per m]')
        ax.legend(fontsize=7, frameon=False, loc='upper left')
        f2.tight_layout()
        for ext in ('pdf', 'png'):
            f2.savefig(os.path.join(figd, f'bow_closure{suffix}.{ext}'), dpi=200)
        plt.close(f2)
    S = {f'{d}_{arm}': {p: (dict(h_curv_per_km=v[p]['h_curv'] * 1e6, w0_slope=v[p]['w0_slope'], r_f=v[p]['r_f'],
                                 b_med=v[p]['b_med']) if v.get(p) else None) for p in 'xy'}
         for (d, arm), v in out.items()}
    json.dump(S, open(os.path.join(res, 'summary.json'), 'w'), indent=1, default=float)
    print(json.dumps(S, indent=1, default=float))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('build', 'fig'))
    ap.add_argument('--out-dir'); ap.add_argument('--in-dir')
    a = ap.parse_args()
    build(a.out_dir) if a.mode == 'build' else fig(a.in_dir)
