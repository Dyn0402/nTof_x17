"""Interactive pad-field page (published as dylan-neff.web.cern.ch/notes/det7-pogo-shorts.html): fills
page_template.html with the pads, neighbour pairs, nets and the best rigid fit, and writes det7-pogo-shorts.html.
Publish with: python3 ~/PycharmProjects/dylan-cern-site/scripts/add-note.py det7-pogo-shorts.html --force --deploy"""
import json
import numpy as np
import model as M

C = np.load('corr_feu6.npz')['C']
cl = M.Cv.CL['x']
best = json.load(open('refine.json'))[0]
l = M.fast_labels(best['dx'], best['dy'], best['th'], best['rp'], best['mode'])
pred = (l[0][M._PA] == l[0][M._PB]) & ~l[1][M._PA]
mm = json.load(open('det7_x_strip_map_measured.json'))
pads = [dict(c=c, r=r, role=M.ROLE[(c, r)], ch=M.CH_AT.get((c, r))) for (c, r) in M.GRID]
pairs = [dict(a=int(a), b=int(b), k=k, c1=round(float(C[a, b]), 3), c2=round(float(C[a + 256, b + 256]), 3), p=bool(p))
         for (a, b, k), p in zip(M.PAIRS, pred)]
f = lambda k: [int(v) for v in cl[k]]
data = dict(pads=pads, pairs=pairs, nets=mm['nets'], noisy=f('noisy'), dead=f('dead'), weak=f('weak'),
            rms=cl['rms'], fire=cl['fire'], best=best, possible=[ch['possible_net_0p7'] for ch in mm['channels']])
page = open('page_template.html').read().replace('__DATA__', json.dumps(data, separators=(',', ':')))
open('det7-pogo-shorts.html', 'w').write(page)
print('det7-pogo-shorts.html')
