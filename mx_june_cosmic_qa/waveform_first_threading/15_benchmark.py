#!/usr/bin/env python3
"""Benchmark harness: score any freefit-format pkl on the standard metrics.

Usage: wf15_benchmark.py file1.pkl[:label] file2.pkl[:label] ...
Metrics per plane: angle med/sig vs M3 (at each file's calibrated v), mesh
med/sig, census <0.5/1.0/1.5 mm over 29 mm, v-flatness (implied-v spread over
angle bins), median chi2/dof. Prints a table; saves benchmark.csv.
"""
import os, sys, pickle, json
import numpy as np
import pandas as pd

BASE = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
        'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
GAP = 29.0

def robust_sigma(a):
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return np.nan
    return 1.4826 * np.median(np.abs(a - np.median(a)))

def score(path, label, v):
    res = pickle.load(open(os.path.join(BASE, path), 'rb'))
    rows = []
    for p in ('x', 'y'):
        tan, w, p0, p0r, chi, dof = [], [], [], [], [], []
        for r in res:
            if p not in r or 'error' in r or 'error' in r.get(p, {}):
                continue
            d = r[p]
            tan.append(d['tan_ref']); w.append(d['w'])
            p0.append(d['p0']); p0r.append(d['p0_ref'])
            chi.append(d['chi2']); dof.append(d['dof'])
        tan, w, p0, p0r = map(np.array, (tan, w, p0, p0r))
        chi, dof = np.array(chi), np.array(dof)
        tanf = w * 1e3 / v
        dth = np.degrees(np.arctan(tanf)) - np.degrees(np.arctan(tan))
        dp = p0 - p0r
        dev0 = np.abs(dp)
        dev1 = np.abs(p0 + tanf * GAP - (p0r + tan * GAP))
        worst = np.maximum(dev0, dev1)
        # v-flatness: spread of median implied v across angle bins
        at = np.abs(tan)
        vb = []
        for a, b in [(0.08, 0.14), (0.14, 0.20), (0.20, 0.28), (0.28, 0.45)]:
            m = (at >= a) & (at < b)
            if m.sum() > 30:
                vb.append(np.nanmedian(w[m] * 1e3 / tan[m]))
        rows.append(dict(
            variant=label, plane=p, n=len(tan),
            ang_med=np.nanmedian(dth), ang_sig=robust_sigma(dth),
            mesh_med=np.nanmedian(dp), mesh_sig=robust_sigma(dp),
            c05=np.nanmean(worst < 0.5), c10=np.nanmean(worst < 1.0),
            c15=np.nanmean(worst < 1.5),
            v_flat=(max(vb) - min(vb)) if len(vb) >= 3 else np.nan,
            v_med=np.nanmedian((w * 1e3 / tan)[at > 0.08]),
            chi2dof=np.nanmedian(chi / dof)))
    return rows

if __name__ == '__main__':
    args = sys.argv[1:]
    all_rows = []
    for a in args:
        if ':' in a:
            path, label = a.split(':', 1)
        else:
            path, label = a, a.replace('.pkl', '')
        # pick v: v2 files use hyper_v2, v1 uses hyper_ref
        hyp = 'hyper_v2.json' if 'freefit2' in path or 'v2' in label else 'hyper_ref.json'
        try:
            v = json.load(open(os.path.join(BASE, hyp)))['v']
        except FileNotFoundError:
            v = 36.65
        all_rows += score(path, label, v)
    df = pd.DataFrame(all_rows)
    pd.set_option('display.width', 200)
    for c in ('ang_med', 'ang_sig', 'mesh_med', 'mesh_sig', 'v_flat', 'v_med', 'chi2dof'):
        df[c] = df[c].round(3)
    for c in ('c05', 'c10', 'c15'):
        df[c] = (df[c] * 100).round(1)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(BASE, 'benchmark.csv'), index=False)
