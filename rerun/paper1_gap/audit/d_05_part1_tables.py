"""D: tables and figure for part 1 -> tables_part1.md, fig_part1.png"""
import json, sys
sys.path.insert(0, '.')
from d_lib import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
P1 = json.load(open(f'{OUT}/part1.json')); ROW = json.load(open(f'{OUT}/rows.json'))
ref = json.load(open(f'{REPO}/rerun/paper1_gap/results/track_grid.json'))
NS = (4, 6, 8, 10)
def f(a, b=None, e=None):
    return f'{a:.3f}' if b is None else f'{a:.3f} ({b:+.3f}±{e:.3f})'
lines = []
for arm in ('w0+v', 'w0'):
    lines.append(f'\n### Part 1, arm {arm}: s68 (deg) and change against the pass-1 angle (paired bootstrap error)\n')
    lines.append('| train > test | view | N train | N test | pass 1 | const | n=4 | n=6 | n=8 | n=10 |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for k, o in P1.items():
        pair, var = k.split('|'); tr_, te = pair.split('>')
        for p, r in o.items():
            a = r['arms']; cells_ = []
            for n in NS:
                key = f'{n} {arm}'
                cells_.append(f(a[key]['s68'], a[key]['d_pass1'], a[key]['d_pass1_err']) if key in a else 'n/a')
            lab = f'{tr_} > {te}' + (' (det7 x>=200 mm)' if var == 'xhalf' else '')
            lines.append(f'| {lab} | {p} | {r["ntrain"]} | {r["ntest"]} | {f(a["pass1"]["s68"])}±{a["pass1"]["err"]:.3f} | {f(a["const"]["s68"], a["const"]["d_pass1"], a["const"]["d_pass1_err"])} | ' + ' | '.join(cells_) + ' |')
open(f'{OUT}/tables_part1.md', 'w').write('\n'.join(lines))

# reproduction table
rep = ['| run (test) | view | arm | this loader | committed track_grid.json |', '|---|---|---|---|---|']
for run, (te, tr_) in {'run1': ('det3:sat', 'det3:p2'), 'run2': ('det3:p2', 'det3:sat')}.items():
    o = P1[f'{tr_}>{te}|full']
    for p in 'xy':
        for key in ('pass1', '4 w0+v', '6 w0+v', '8 w0+v', '10 w0+v', '8 w0'):
            rep.append(f'| {run} ({te}) | {p} | {key} | {o[p]["arms"][key]["s68"]:.4f} | {ref[run][p][key if key=="pass1" else key.replace(" ", "x"+key.split()[0]+" ",1) if False else (key if key=="pass1" else key.split()[0]+"x"+key.split()[0]+" "+key.split()[1])]["s68"]:.4f} |')
open(f'{OUT}/tables_repro.md', 'w').write('\n'.join(rep))

# summary per chamber: groups of pairs
groups = {
 'B (det2) long -> longer/short': [('det2:long', 'det2:longer'), ('det2:long', 'det2:short')],
 'B (det2) longer -> long (2.2k train)': [('det2:longer', 'det2:long')],
 'A (det3) p2 -> sat (same slot, same HV)': [('det3:p2', 'det3:sat')],
 'A (det3) sat -> p2 (same slot, same HV)': [('det3:sat', 'det3:p2')],
 'A (det3) p2/sat -> 6-25 (other slot, drift 500 V)': [('det3:p2', 'det3:d625'), ('det3:sat', 'det3:d625')],
 'A (det3) 6-25 -> sat/p2 (other slot, drift 500 V)': [('det3:d625', 'det3:sat'), ('det3:d625', 'det3:p2')],
 'E (det4) 6-24 -> 6-23': [('det4:624', 'det4:623')],
 'E (det4) 6-23 -> 6-24': [('det4:623', 'det4:624')],
 'C (det6) long -> longer/short': [('det6:long', 'det6:longer'), ('det6:long', 'det6:short')],
 'C (det6) longer -> long': [('det6:longer', 'det6:long')],
 'D (det7) long -> longer/short': [('det7:long', 'det7:longer'), ('det7:long', 'det7:short')],
 'D (det7) long -> longer/short, x >= 200 mm': [('det7:long', 'det7:longer', 'xhalf'), ('det7:long', 'det7:short', 'xhalf')],
}
S = ['| group | view | N train | pass 1 | const | ' + ' | '.join(f'n={n} w0+v' for n in NS) + ' | ' + ' | '.join(f'n={n} w0' for n in NS) + ' |', '|' + '---|' * (5 + 2 * len(NS))]
summ = {}
for g, prs in groups.items():
    for p in 'xy':
        if len(prs[0]) == 3 and p == 'y':
            continue
        vals = {}
        for key in ['pass1', 'const'] + [f'{n} {a}' for a in ('w0+v', 'w0') for n in NS]:
            d, e, s = [], [], []
            for pr in prs:
                var = pr[2] if len(pr) == 3 else 'full'
                r = P1[f'{pr[0]}>{pr[1]}|{var}'][p]['arms']
                if key in r:
                    s.append(r[key]['s68'])
                    if key != 'pass1':
                        d.append(r[key]['d_pass1']); e.append(r[key]['d_pass1_err'])
            if not s:
                vals[key] = None; continue
            vals[key] = (np.mean(s), np.mean(d) if d else 0.0, np.sqrt(np.sum(np.square(e))) / len(e) if e else 0.0)
        summ[(g, p)] = vals
        ntr = ', '.join(str(P1[f'{pr[0]}>{pr[1]}|{pr[2] if len(pr)==3 else "full"}'][p]['ntrain']) for pr in prs)
        fmt = lambda v: 'n/a' if v is None else f'{v[0]:.3f} ({v[1]:+.3f}±{v[2]:.3f})'
        S.append(f'| {g} | {p} | {ntr} | {vals["pass1"][0]:.3f} | {fmt(vals["const"])} | ' + ' | '.join(fmt(vals[f"{n} w0+v"]) for n in NS) + ' | ' + ' | '.join(fmt(vals[f"{n} w0"]) for n in NS) + ' |')
open(f'{OUT}/tables_part1_summary.md', 'w').write('\n'.join(S))
json.dump({f'{g}|{p}': {k: v for k, v in vv.items()} for (g, p), vv in summ.items()}, open(f'{OUT}/part1_summary.json', 'w'), default=float, indent=1)

# figure
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=True)
sel = ['B (det2) long -> longer/short', 'A (det3) p2 -> sat (same slot, same HV)', 'A (det3) 6-25 -> sat/p2 (other slot, drift 500 V)',
       'E (det4) 6-24 -> 6-23', 'C (det6) long -> longer/short', 'D (det7) long -> longer/short, x >= 200 mm']
cols = plt.cm.tab10(np.arange(10))
for a, p in zip(ax, 'xy'):
    for i, g in enumerate(sel):
        if (g, p) not in summ:
            g2 = 'D (det7) long -> longer/short'
            if g.startswith('D'):
                g = g2
            else:
                continue
        v = summ[(g, p)]
        xs = [n for n in NS if v.get(f'{n} w0+v')]
        a.errorbar(np.array(xs) + 0.12 * (i - 2.5), [v[f'{n} w0+v'][1] for n in xs], [v[f'{n} w0+v'][2] for n in xs], marker='o', ms=4, lw=1, capsize=2, color=cols[i], label=g.replace(' -> ', '>'))
        if g.startswith('D (det7) long -> longer/short') and p == 'x' and False:
            pass
    a.axhline(0, color='k', lw=0.6); a.set_xlabel('map cells per side n'); a.set_title(f'{p.upper()} view'); a.set_xticks(NS)
ax[0].set_ylabel('s68 change, w0+v map vs pass 1 (deg)')
ax[1].legend(fontsize=6, loc='lower right')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_part1.png', dpi=160)
print(open(f'{OUT}/tables_part1_summary.md').read())
