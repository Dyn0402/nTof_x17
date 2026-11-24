import json, sys
import numpy as np
for r in ('run1', 'run2'):
    try: d = json.load(open(f'/media/ucla/work/claude-jobs/e1a7a26e/gap/audit/a2/part1_{r}.json'))
    except Exception: continue
    print('train', r, 'kappa', d['kappa'])
    g = d['grid']
    for var in ('indep', 'phi', 'phiplus'):
        print(var)
        for K in (5, 7, 9, 12, 16, 20, 26):
            rows = [(g[f'{var}|{K}|{KS}'], KS) for KS in (0, 4, 6, 8)]
            b, KSb = min(rows, key=lambda t: t[0]['cv_x'] + t[0]['cv_y'])
            print(f'  K={K:2d} CVbest KS={KSb} lam={b["lam"]:.3g} cv=({b["cv_x"]:.3f},{b["cv_y"]:.3f}) test=({b["test"]["x"]:.3f},{b["test"]["y"]:.3f}) edge={b["edge"]}   by KS:',
                  ' '.join(f'{ks}:{t["test"]["x"]:.3f}/{t["test"]["y"]:.3f}' for t, ks in rows))
