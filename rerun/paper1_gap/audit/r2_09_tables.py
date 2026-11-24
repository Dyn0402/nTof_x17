"""r2_09_tables.py -- sigma tables (markdown) and fig3 from direct_{late,cosmic}.csv and bd_direct_{late,cosmic}.csv."""
from r2_lib import *
import json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
MODELS = ['global', 'insitu', 'tim_nom_k0', 'tim_nom_k1', 'tim_fit_k0', 'tim_fit_k1', 'tim_fit_kb', 'tim_nom_k0_bench', 'tim_nom_k1_bench', 'tim_fit_kb_bench']
lines = []
res = {}
for fld in ('late', 'cosmic'):
    R = pd.read_csv(f'{OUT}/direct_{fld}.csv')
    T = R.groupby(['arm', 'view', 'model'], sort=False)[['sig_all', 'err_all', 'sig_clean', 'err_clean']].mean().reset_index()
    lines.append(f'\n### A-C lines, field from {fld} tracks (sigma_all x 1000, MAD of corrected - joined slope, |j|<0.45, parity-averaged held-out; error = bootstrap)\n')
    lines.append('| chamber view | global | in-situ | tim nom k=0 | tim nom k=1 | tim fit k=0 | tim fit k=1 | tim fit k,b | +bench nom k=0 | +bench nom k=1 | +bench fit k,b |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for arm in 'AC':
        for v in 'xy':
            g = T[(T.arm == arm) & (T.view == v)].set_index('model')
            cells = []
            for m in MODELS:
                s, e = g.loc[m, 'sig_all'] * 1e3, g.loc[m, 'err_all'] * 1e3
                gain = 100 * (g.loc['global', 'sig_all'] - g.loc[m, 'sig_all']) / g.loc['global', 'sig_all']
                cells.append(f'{s:.1f} ± {e:.1f}' + ('' if m == 'global' else f' ({gain:+.1f} %)'))
                res[(fld, arm, v, m)] = (s, e, gain)
            lines.append(f'| {arm} {v} | ' + ' | '.join(cells) + ' |')
    lines.append('\nShare of the in-situ gain recovered (gain_model / gain_insitu, %):\n')
    lines.append('| chamber view | tim nom k=0 | tim nom k=1 | tim fit k=1 | tim fit k,b | +bench nom k=0 | +bench fit k,b |')
    lines.append('|---|---|---|---|---|---|---|')
    for arm in 'AC':
        for v in 'xy':
            gi = res[(fld, arm, v, 'insitu')][2]
            lines.append(f'| {arm} {v} | ' + ' | '.join(f'{100 * res[(fld, arm, v, m)][2] / gi:.0f}' for m in ['tim_nom_k0', 'tim_nom_k1', 'tim_fit_k1', 'tim_fit_kb', 'tim_nom_k0_bench', 'tim_fit_kb_bench']) + ' |')
    # fitted coefficients
    am = json.load(open(f'{OUT}/direct_amps_{fld}.json'))
    lines.append(f"\nEvent-level fitted coefficients, {fld} field (b' = fitted / nominal, nominal = 0.5 G0/(1+0.68); so b = 0.5 b'; k = bench dish amplitude); mean of the two parities, difference in brackets:\n")
    lines.append("| chamber view | model | k | b' | b = 0.5 b' |")
    lines.append('|---|---|---|---|---|')
    for arm in 'AC':
        for v in 'xy':
            for mdl in ('tim_fit_k0', 'tim_fit_k1', 'tim_fit_kb'):
                rr = [a for a in am if a['arm'] == arm and a['view'] == v and a['model'] == mdl]
                co = np.array([a['coef'] for a in rr])
                if mdl == 'tim_fit_kb':
                    k, b = co[:, 2], co[:, 3]
                else:
                    k, b = np.full(2, 0.0 if mdl.endswith('k0') else 1.0), co[:, 2]
                lines.append(f"| {arm} {v} | {mdl} | {k.mean():.2f} ({abs(k[0] - k[1]):.2f}) | {b.mean():.2f} ({abs(b[0] - b[1]):.2f}) | {0.5 * b.mean():.2f} |")
# D
for fld in ('late', 'cosmic'):
    R = pd.read_csv(f'{OUT}/bd_direct_{fld}.csv')
    S = R[~R.model.str.startswith('coef')].groupby(['view', 'model'], sort=False)[['sig', 'err', 'n']].mean().reset_index()
    lines.append(f'\n### Chamber D on B-D lines, field from {fld} tracks (sigma x 1000, event-parity held-out, ~1,200 test lines per view)\n')
    lines.append('| D view | global | in-situ 100 mm map | tim nom k=0 | tim nom k=1 | tim fit k=1 | tim fit k,b |')
    lines.append('|---|---|---|---|---|---|---|')
    for lv in 'zy':
        g = S[S.view == lv].set_index('model')
        cells = []
        for m in ['global', 'insitu 100mm B-D map', 'tim_nom_k0', 'tim_nom_k1', 'tim_fit_k1', 'tim_fit_kb']:
            gain = 100 * (g.loc['global', 'sig'] - g.loc[m, 'sig']) / g.loc['global', 'sig']
            cells.append(f"{g.loc[m, 'sig'] * 1e3:.1f} ± {g.loc[m, 'err'] * 1e3:.1f}" + ('' if m == 'global' else f' ({gain:+.1f} %)'))
            res[('D' + fld, lv, m)] = (g.loc[m, 'sig'] * 1e3, g.loc[m, 'err'] * 1e3, gain)
        lines.append(f'| {lv} | ' + ' | '.join(cells) + ' |')
    C = R[R.model.str.startswith('coef')].groupby(['view', 'model']).sig.mean().unstack()
    lines.append("\nD fitted coefficients (mean of parities): " + '; '.join(f"{lv}: k_fit_kb {C.loc[lv, 'coef_k']:.2f}, b' {C.loc[lv, 'coef_b']:.2f} (k=1 fixed: b' {C.loc[lv, 'coef_tim_fit_k1']:.2f})" for lv in 'zy'))
open(f'{OUT}/tables.md', 'w').write('\n'.join(lines))
print('\n'.join(lines))
# fig3
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for a, fld in zip(ax, ('late', 'cosmic')):
    names = ['A x', 'A y', 'C x', 'C y', 'D z', 'D y']
    mod = [('global', 'global'), ('insitu', 'in-situ map'), ('tim_nom_k1', 'timing nominal, k=1'), ('tim_fit_kb', 'timing fitted k,b'), ('tim_nom_k0_bench', 'timing k=0 + bench fine')]
    w = 0.16
    for q, (m, lab) in enumerate(mod):
        vals, errs = [], []
        for nm in names:
            if nm[0] in 'AC':
                s, e, g = res[(fld, nm[0], nm[2], m)]
            else:
                mm = {'global': 'global', 'insitu': 'insitu 100mm B-D map', 'tim_nom_k1': 'tim_nom_k1', 'tim_fit_kb': 'tim_fit_kb', 'tim_nom_k0_bench': None}[m]
                if mm is None: vals.append(np.nan); errs.append(0); continue
                s, e, g = res[('D' + fld, nm[2], mm)]
            base = res[(fld, nm[0], nm[2], 'global')][0] if nm[0] in 'AC' else res[('D' + fld, nm[2], 'global')][0]
            vals.append(100 * (s / base - 1)); errs.append(100 * e / base)
        a.bar(np.arange(len(names)) + (q - 2) * w, vals, w, yerr=errs, label=lab)
    a.set_xticks(np.arange(len(names))); a.set_xticklabels(names); a.axhline(0, color='k', lw=0.6)
    a.set_ylabel('change of held-out sigma against global (%)'); a.set_title(f'drift-time field from {fld} tracks', fontsize=10)
a = ax[0]; a.legend(fontsize=7)
fig.savefig(f'{OUT}/fig3_sigma.png', dpi=110)
