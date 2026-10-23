#!/usr/bin/env python3
"""why_cm5.py -- events matched with the plain-CM hits and not with the masked-CM hits: what changed."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge as J
from common.Mx17StripMap import RunConfig
import glob, uproot

rc_path, detn, sub, tags, raysdir = sys.argv[1:6]
tags = tags.split(',')
rc = RunConfig(rc_path, os.path.join(J.REPO, 'mx17_m1_map.csv'))
det = rc.get_detector(detn)
z = [d for d in json.load(open(rc_path))['detectors'] if d['name'] == detn][0]['det_center_coords']['z']
tab = J.pos_table(det)
H = {g: J.det_clusters([glob.glob(os.path.join(sub, f'combined_hits_root_rr1_{g}', f'*_{t}_feu-combined_hits.root'))[0]
                        for t in tags], tab) for g in ('cm0', 'cm5')}
run, s = os.path.basename(os.path.dirname(sub)), os.path.basename(sub)
rays = J.clean_rays([os.path.join(raysdir, f"{run}__{s}__{t.split('_')[-1]}.root") for t in tags], z)
coef = J.fit_align(rays, H['cm0'])
ev = np.array(sorted(rays)); R = np.array([rays[e] for e in ev])
P = np.column_stack([J.design(R) @ coef[0], J.design(R) @ coef[1]])
res = {}
for g in H:
    D = np.array([H[g].get(int(e), [np.nan, np.nan]) for e in ev]); res[g] = D - P
m0 = np.all(np.abs(res['cm0']) < 5, 1); m5 = np.all(np.abs(res['cm5']) < 5, 1)
lost = m0 & ~m5
print('matched cm0', m0.sum(), 'cm5', m5.sum(), 'lost', lost.sum(), 'gained', (m5 & ~m0).sum())
for v in (0, 1):
    r = res['cm5'][lost, v]
    print(f' view {"xy"[v]}: lost events, cm5 residual |r|<5: {np.mean(np.abs(r)<5):.2f}, 5-20: {np.mean((np.abs(r)>=5)&(np.abs(r)<20)):.2f}, '
          f'>=20: {np.mean(np.abs(r)>=20):.2f}, nan: {np.mean(~np.isfinite(r)):.2f}')
# hit multiplicity per event per view and cluster count, cm0 vs cm5, for the lost events
for g in ('cm0', 'cm5'):
    p = glob.glob(os.path.join(sub, f'combined_hits_root_rr1_{g}', f'*_{tags[0]}_feu-combined_hits.root'))[0]
    h = uproot.open(p)['hits'].arrays(['eventId', 'feu', 'channel', 'amplitude'], library='np')
    le = set(ev[lost].tolist())
    sel = np.isin(h['eventId'], list(le))
    print(g, 'hits/event in lost events', sel.sum() / max(len(le), 1), ' amplitude pctl', np.percentile(h['amplitude'][sel], [10, 50, 90]).round(0))
