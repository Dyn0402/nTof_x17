"""r3_report.py -- tables (paired bootstrap) from the cv_*.npz / lc_*.json / gapstats.json.  -> $OUT/tables.md, summary.json"""
import json, os, glob
import numpy as np
from r3_lib import paired_boot, s68, OUT, MOUNT

ARMS = ['pass1', 'const', 'dir_phiS', 'dir_tilt', 'rf_mesh', 'rf_tilt_nom', 'rf_tilt_fit', 'rf_tv_nom', 'rf_tv_fit', 'both_phiS', 'both_tilt']
LAB = {'pass1': 'pass 1', 'const': 'constants only', 'dir_phiS': 'direct phi + S', 'dir_tilt': 'direct phi (S = 0)',
       'rf_mesh': 'ref-free: mesh only', 'rf_tilt_nom': 'ref-free tilt, b = 0.5', 'rf_tilt_fit': 'ref-free tilt, b fitted',
       'rf_tv_nom': 'ref-free tilt + v, b = 0.5', 'rf_tv_fit': 'ref-free tilt + v, b fitted',
       'both_phiS': 'ref-free (tilt+v) then direct phi + S', 'both_tilt': 'ref-free tilt then direct phi (S = 0)'}
NB = 400
L = []
SUM = {}
gap = json.load(open(f'{OUT}/gapstats.json'))
order = ['B', 'A625', 'A628', 'E623', 'E624', 'C', 'D', 'Dx200']
fmt = lambda v: f'{v[0]:.3f} +- {v[1]:.3f}'
fmtd = lambda v: f'{v[0]:+.3f} +- {v[1]:.3f}'


def stats(z, sch, view, mask=None):
    arr = {a: z[f'{sch}|{a}|{view}'].astype(float) for a in ARMS}
    if mask is not None:
        arr = {a: v[mask] for a, v in arr.items()}
    return paired_boot(arr, 'pass1', NB)


for nm in order:
    if not os.path.exists(f'{OUT}/cv_{nm}.npz'):
        continue
    z = np.load(f'{OUT}/cv_{nm}.npz'); J = json.load(open(f'{OUT}/cv_{nm}.json'))
    L.append(f'\n## Mounting {nm}: {MOUNT[nm]["chamber"]}, rows {[r["key"] for r in J["rows"]]}, N = {J["N"]}\n')
    beta = J.get('beta', {})
    L.append(f'beta: {beta}\n')
    for sch in [s for s in ('time5', 'loro') if s in J]:
        L.append(f'\n### {nm}, scheme {sch}: fitted b = {J[sch]["b_mean"]:.2f} +- {J[sch]["b_err"]:.2f} (x {J[sch]["b_x_mean"]:.2f} +- {J[sch]["b_x_err"]:.2f}, y {J[sch]["b_y_mean"]:.2f} +- {J[sch]["b_y_err"]:.2f}); selections (K, K_S, lambda, inner CV sum):')
        sel = J[sch]['sel']
        L.append('  ' + '; '.join(f'{k}: {[round(v, 3) if isinstance(v, float) else v for v in sel[k]]}' for k in sel if k.startswith('dir_phiS') or k.startswith('both_phiS')) + '\n')
        views = [('x', None), ('y', None)]
        if nm == 'D':
            views.append(('x', z['xm'] >= 200))
        for view, mask in views:
            s, d = stats(z, sch, view, mask)
            tag = f'{view.upper()} view' + (' (events x >= 200 mm only)' if mask is not None else '')
            L.append(f'\n**{nm} {sch} {tag}** (n = {len(next(iter(s))) if False else (int(mask.sum()) if mask is not None else J["N"])})\n')
            L.append('| arm | s68 (deg) | difference to pass 1 | gain |\n|---|---|---|---|')
            for a in ARMS:
                L.append(f'| {LAB[a]} | {fmt(s[a])} | {fmtd(d[a])} | {-100 * d[a][0] / s["pass1"][0]:+.1f} % |')
            if sch == 'time5':
                key = f'{nm}|{view}|{"half" if mask is not None else "all"}'
                SUM[key] = dict(s=s, d=d)
    L.append('')

# ------------------------------------------------------------------ summary table
lc = {}
for p in glob.glob(f'{OUT}/lc_*.json'):
    q = json.load(open(p)); lc[q['name']] = q['asym']
S = ['| mounting | view | N | pass 1 | direct map CV (phi + S) | direct, S = 0 | reference-free (tilt + v, b fitted) | reference-free (tilt, b = 0.5) | both | asymptote | gain direct | gain ref-free | ref-free / direct |',
     '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for nm in order:
    for view in 'xy':
        for kind in ('all', 'half'):
            k = f'{nm}|{view}|{kind}'
            if k not in SUM:
                continue
            s, d = SUM[k]['s'], SUM[k]['d']
            J = json.load(open(f'{OUT}/cv_{nm}.json'))
            n = J['N'] if kind == 'all' else int((np.load(f'{OUT}/cv_{nm}.npz')['xm'] >= 200).sum())
            gd = -d['dir_phiS'][0] / s['pass1'][0] * 100
            gr = -d['rf_tv_fit'][0] / s['pass1'][0] * 100
            asy = lc.get(nm, {}).get(view)
            asy_s = f'{asy["s_inf"]:.3f} +- {asy["err"]:.3f}' if asy else 'n/a'
            ratio = f'{gr / gd:.2f}' if gd > 3 else 'n/a'
            lab = view.upper() + (' (x>=200 events of D)' if kind == 'half' else '')
            S.append(f'| {nm} | {lab} | {n} | {fmt(s["pass1"])} | {fmt(s["dir_phiS"])} | {fmt(s["dir_tilt"])} | {fmt(s["rf_tv_fit"])} | {fmt(s["rf_tilt_nom"])} | {fmt(s["both_phiS"])} | {asy_s} | {gd:+.1f} % (diff {d["dir_phiS"][0]:+.3f} +- {d["dir_phiS"][1]:.3f}) | {gr:+.1f} % (diff {d["rf_tv_fit"][0]:+.3f} +- {d["rf_tv_fit"][1]:.3f}) | {ratio} |')
open(f'{OUT}/tables.md', 'w').write('# R3 tables (time-block CV: 5 blocks; loro = leave one row out)\n\n## Summary (time5)\n\n' + '\n'.join(S) + '\n' + '\n'.join(L))
json.dump({k: dict(s=v['s'], d=v['d']) for k, v in SUM.items()}, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
print('\n'.join(S))
