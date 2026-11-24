import json, numpy as np
from a2_lib import *
ref = json.load(open(f'{REPO}/results/track_grid.json'))
D = {r: load(r) for r in ('run1', 'run2')}
oth = {'run1': 'run2', 'run2': 'run1'}
mx = 0
for r in D:
    arms = baseline_arms(D[r], D[oth[r]])
    for p in 'xy':
        tr = D[r].trx if p == 'x' else D[r].trY
        for a, key in (('pass1', 'pass1'), ('cells4', '4x4 w0+v'), ('cells6', '6x6 w0+v'), ('cells8', '8x8 w0+v'), ('cells10', '10x10 w0+v')):
            m = TT.metrics(arms[a][p], tr, D[r].U, D[r].W)['s68']
            diff = abs(m - ref[r][p][key]['s68']); mx = max(mx, diff)
            print(r, p, a, round(m, 4), round(ref[r][p][key]['s68'], 4))
        print(r, p, 'const', round(TT.metrics(arms['const'][p], tr, D[r].U, D[r].W)['s68'], 4), 'pass1c', round(TT.metrics(arms['pass1c'][p], tr, D[r].U, D[r].W)['s68'], 4))
print('max diff', mx)
