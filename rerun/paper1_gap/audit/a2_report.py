"""a2_report.py -- tables and figures of audit test a2 from the json / npz files written by a2_part1..3 and a2_final."""
import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from a2_lib import OUT, s68, paired_boot
plt.rcParams.update({'font.size': 10, 'figure.dpi': 130})
md = []
def P(s=''): md.append(s)
Ks = (5, 7, 9, 12, 16, 20, 26); KSl = (0, 4, 6, 8); VAR = ('indep', 'phi', 'phiplus')
# ---------------------------------------------------------------- Part 1
fig, ax = plt.subplots(2, 4, figsize=(13, 6), sharex=True)
P('## Part 1: unbinned spline maps, cross-run (maps chosen by inner CV in the train run only)\n')
sel = {}
for ci, tr in enumerate(('run1', 'run2')):
    J = json.load(open(f'{OUT}/part1_{tr}.json')); Z = np.load(f'{OUT}/part1_{tr}_dth.npz'); g = J['grid']; te = J['test']
    for pi, p in enumerate('xy'):
        a = ax[pi, 2 * ci + 0]; a2 = ax[pi, 2 * ci + 1]
        names = {}
        for var in VAR:
            ys = []; eb = []
            for K in Ks:
                KSb = min(KSl, key=lambda k: g[f'{var}|{K}|{k}']['cv_x'] + g[f'{var}|{K}|{k}']['cv_y'])
                key = f'{var}|{K}|{KSb}'; sel[(tr, var, K)] = key; ys.append(g[key]['test'][p])
            a.plot(Ks, ys, 'o-', label=var)
        for n, st in zip((4, 6, 8, 10), ('-', '--', '-.', ':')):
            a.axhline(s68(Z[f'cells{n}|{p}']), color='k', ls=st, lw=0.8, label=f'cells {n}x{n}')
        a.axhline(s68(Z[f'pass1|{p}']), color='r', lw=1, label='pass 1')
        a.set_title(f'train {tr} -> test {te}, {p} view'); a.set_xlabel('K (basis functions per axis)'); a.set_ylabel('s68 (deg)')
        # S dependence for phi
        for KS in KSl:
            a2.plot(Ks, [g[f'phi|{K}|{KS}']['test'][p] for K in Ks], 'o-', label=f'phi, K_S={KS}')
        a2.set_title(f'phi, K_S dependence ({p})'); a2.set_xlabel('K')
ax[0, 0].legend(fontsize=7); ax[0, 1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_s68_vs_K.png'); plt.close(fig)
for tr in ('run1', 'run2'):
    J = json.load(open(f'{OUT}/part1_{tr}.json')); Z = np.load(f'{OUT}/part1_{tr}_dth.npz'); g = J['grid']; te = J['test']
    P(f'### train {tr} -> test {te}  (n_test = {len(Z["pass1|x"])})\n')
    for p in 'xy':
        arms = ['pass1', 'const', 'pass1c', 'cells4', 'cells6', 'cells8', 'cells10']
        ds = {a: Z[f'{a}|{p}'] for a in arms}
        for var in VAR:
            for K in Ks: ds[f'{var} K={K}'] = Z[sel[(tr, var, K)] + f'|{p}']
        S, D1 = paired_boot(ds, 'pass1', nb=400); _, D8 = paired_boot(ds, 'cells8', nb=400)
        P(f'**{p} view**; the spline rows use K_S and lambda chosen by inner CV (K_S, lambda in the last columns); "d vs pass1" and "d vs cells8" are s68 differences with paired-bootstrap errors (400 resamples of test events; the two constants are not refit inside the bootstrap).\n')
        P('| arm | s68 (deg) | d vs pass1 | d vs cells8 | CV s68 (train) | K_S | lambda | lambda at grid edge |'); P('|---|---|---|---|---|---|---|---|')
        for a in ds:
            row = f'| {a} | {S[a][0]:.3f} +- {S[a][1]:.3f} | {D1[a][0]:+.3f} +- {D1[a][1]:.3f} | {D8[a][0]:+.3f} +- {D8[a][1]:.3f} |'
            if a.split(' ')[0] in VAR:
                var, K = a.split(' K='); k = sel[(tr, var, int(K))]; e = g[k]
                row += f' {e["cv_" + p]:.3f} | {k.split("|")[2]} | {e["lam"]:.3g} | {"yes" if e["edge"] else "no"} |'
            else: row += ' | | | |'
            P(row)
        P()
        # headline comparisons among splines at inner-CV-best K
        bestK = {var: min(Ks, key=lambda K: g[sel[(tr, var, K)]]['cv_x'] + g[sel[(tr, var, K)]]['cv_y']) for var in VAR}
        P(f'Inner-CV-best K per variant (x+y): {bestK}.\n')
        d2 = {var: Z[sel[(tr, var, bestK[var])] + f'|{p}'] for var in VAR}; d2['cells10'] = ds['cells10']; d2['cells8'] = ds['cells8']
        _, Dp = paired_boot(d2, 'phi')
        P('Paired differences w.r.t. `phi` at its CV-best K: ' + ', '.join(f'{k}: {v[0]:+.3f} +- {v[1]:.3f}' for k, v in Dp.items() if k != 'phi') + '\n')
# ---------------------------------------------------------------- Part 2
P('## Part 2: learning curve of the `phi` map (K and lambda re-chosen by inner CV at every N; K_S = 6)\n')
fig, ax = plt.subplots(1, 2, figsize=(10, 4)); fits = {}
for pi, p in enumerate('xy'):
    allres = []
    for w, col in (('A', 'C0'), ('B', 'C1'), ('U', 'C2')):
        try: R = json.load(open(f'{OUT}/part2_{w}.json'))
        except Exception: continue
        for seed in range(4):
            rr = sorted([r for r in R if r['seed'] == seed], key=lambda r: r['N'])
            if len(rr) < 4: continue
            N = np.array([r['N'] for r in rr], float); s = np.array([r[p] for r in rr])
            ax[pi].plot(1 / N, s ** 2, 'o-', color=col, alpha=0.5, ms=3, label=w if seed == 0 else None)
            for form, sl in (('all N', slice(0, None)), ('N>=N_max/4', slice(2, None)), ('3 largest', slice(-3, None))):
                A = np.vstack([np.ones(len(N[sl])), 1 / N[sl]]).T; c = np.linalg.lstsq(A, (s ** 2)[sl], rcond=None)[0]
                allres.append((w, seed, form, np.sqrt(max(c[0], 0)), c[1], s[-1]))
    fits[p] = allres
    ax[pi].set_xlabel('1 / N_train'); ax[pi].set_ylabel('s68$^2$ (deg$^2$)'); ax[pi].set_title(f'{p} view'); ax[pi].legend(); ax[pi].set_xlim(left=0); 
fig.tight_layout(); fig.savefig(f'{OUT}/fig_learning_curve.png'); plt.close(fig)
for w in 'ABU':
    try: R = json.load(open(f'{OUT}/part2_{w}.json'))
    except Exception: continue
    desc = {'A': 'train = run 2 + random half of run 1, test = other half of run 1', 'B': 'train = run 1 + random half of run 2, test = other half of run 2', 'U': 'union of both runs, 5-fold (a different held-out fold per seed), train subsampled'}[w]
    P(f'**{w}: {desc}.** Mean over 4 random splits (s68 in deg, scatter between splits in brackets); cells 8x8 trained on the same N for comparison (nan where cells have < 40 events).\n')
    P('| N_train | n_test | K chosen | phi x | phi y | cells8 x | cells8 y |'); P('|---|---|---|---|---|---|---|')
    for N in sorted(set(r['N'] for r in R)):
        rr = [r for r in R if r['N'] == N]
        P(f'| {N} | {rr[0]["ntest"]} | {sorted(set(r["K"] for r in rr))} | {np.mean([r["x"] for r in rr]):.3f} ({np.std([r["x"] for r in rr]):.3f}) | {np.mean([r["y"] for r in rr]):.3f} ({np.std([r["y"] for r in rr]):.3f}) | {np.nanmean([r["base"]["cells8"]["x"] for r in rr]):.3f} | {np.nanmean([r["base"]["cells8"]["y"] for r in rr]):.3f} |')
    P()
P('Asymptote s68_inf from s68^2 = s68_inf^2 + c/N (deg), per split and fit window; mean over splits (+- scatter), and the spread between windows:\n')
P('| set | view | fit window | s68_inf | scatter between splits | c (deg^2) | s68 at largest N |'); P('|---|---|---|---|---|---|---|')
summ = {}
for p in 'xy':
    for w in 'ABU':
        for form in ('all N', 'N>=N_max/4', '3 largest'):
            r = [x for x in fits[p] if x[0] == w and x[2] == form]
            if not r: continue
            P(f'| {w} | {p} | {form} | {np.mean([x[3] for x in r]):.3f} | {np.std([x[3] for x in r]):.3f} | {np.mean([x[4] for x in r]):.2f} | {np.mean([x[5] for x in r]):.3f} |')
            summ[(p, w, form)] = (np.mean([x[3] for x in r]), np.std([x[3] for x in r]))
    vals = [v[0] for k, v in summ.items() if k[0] == p]; sc = [v[1] for k, v in summ.items() if k[0] == p]
    P(f'\n**{p} view: s68_inf = {np.mean(vals):.3f}, range over sets and windows {min(vals):.3f} - {max(vals):.3f}, split scatter {np.mean(sc):.3f}.**\n')
json.dump({f'{k[0]}|{k[1]}|{k[2]}': v for k, v in summ.items()}, open(f'{OUT}/asymptotes.json', 'w'), indent=1)
# ---------------------------------------------------------------- Part 3
P3 = json.load(open(f'{OUT}/part3.json'))
P('## Part 3: robustness\n'); P('### (a), (e) time splits and baselines (phi: K, lambda chosen by inner CV in the train part; K_S = 6; paired-bootstrap errors on differences, 400 resamples)\n')
P('| split | n_train/n_test | K | view | pass1 | const | pass1+2 const | cells8 | phi | phi - pass1 | phi - cells8 |'); P('|---|---|---|---|---|---|---|---|---|---|---|')
for k, v in P3.items():
    if k in ('bins', 'index_and_position'): continue
    for p in 'xy':
        s = v[p]['s68']
        P(f'| {k} | {v["ntrain"]}/{v["ntest"]} | {v["K"]} | {p} | {s["pass1"][0]:.3f} | {s["const"][0]:.3f} | {s["pass1c"][0]:.3f} | {s["cells8"][0]:.3f} | {s["phi"][0]:.3f} | {v[p]["d_vs_pass1"]["phi"][0]:+.3f} +- {v[p]["d_vs_pass1"]["phi"][1]:.3f} | {v[p]["phi_minus_cells8"][0]:+.3f} +- {v[p]["phi_minus_cells8"][1]:.3f} |')
P('\n### (b) where the gain comes from (cross-run, both directions)\n')
P('Tail columns: fraction of events with |theta - theta_ref| > 2 deg and > 4 deg (in per cent).\n')
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for tr, bb in P3['bins'].items():
    P(f'**{tr}**\n'); P('| view, subset | n | pass1 s68 | cells8 s68 | phi s68 | pass1 >2/>4 deg | cells8 >2/>4 | phi >2/>4 |'); P('|---|---|---|---|---|---|---|---|')
    for k, r in bb.items():
        P(f'| {k} | {r["n"]} | {r["pass1"]:.3f} | {r["cells8"]:.3f} | {r["phi"]:.3f} | ' + ' | '.join(f'{100 * r[a + "_t2"]:.2f} / {100 * r[a + "_t4"]:.2f}' for a in ('pass1', 'cells8', 'phi')) + ' |')
    P()
    for pi, p in enumerate('xy'):
        labs = ['|tan| 0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.6']; xs = np.arange(4)
        ax[pi].plot(xs, [1 - bb[f'{p} {l}']['phi'] / bb[f'{p} {l}']['pass1'] for l in labs], 'o-', label=f'phi {tr}')
        ax[pi].plot(xs, [1 - bb[f'{p} {l}']['cells8'] / bb[f'{p} {l}']['pass1'] for l in labs], 's--', label=f'cells8 {tr}')
        ax[pi].set_xticks(xs); ax[pi].set_xticklabels(labs); ax[pi].set_ylabel('relative s68 gain vs pass 1'); ax[pi].set_title(f'{p} view'); ax[pi].set_xlabel('|tan_ref|')
ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f'{OUT}/fig_gain_vs_tan.png'); plt.close(fig)
P('### (c), (d) index by the reference position; mid-gap position residual\n')
P('| direction | view | phi, recon. index | phi, ref index | cells8 recon | cells8 ref | position s68 (mm) pass1 | pass1+2c | cells8 | phi |'); P('|---|---|---|---|---|---|---|---|---|---|')
for tr, v in P3['index_and_position'].items():
    for p in 'xy':
        r = v[p]; q = r['pos_mid_s68_mm']
        P(f'| {tr} | {p} | {r["phi_recon"]:.3f} | {r["phi_ref"]:.3f} | {r["cells8_recon"]:.3f} | {r["cells8_ref"]:.3f} | {q["pass1"]:.3f} | {q["pass1c"]:.3f} | {q["cells8"]:.3f} | {q["phi"]:.3f} |')
# ---------------------------------------------------------------- Part 4
try:
    F = json.load(open(f'{OUT}/final_cv.json')); M = json.load(open(f'{OUT}/final_map_det3.json'))
    P('\n## Part 4: final map of the union (CV: 5 time blocks over run 1 then run 2)\n')
    P('| variant | K | K_S | lambda | CV s68 x | CV s68 y | lambda at edge |'); P('|---|---|---|---|---|---|---|')
    for k, v in sorted(F['cvgrid'].items(), key=lambda kv: kv[1]['cv_x'] + kv[1]['cv_y']):
        a, b, c = k.split('|'); P(f'| {a} | {b} | {c} | {v["lam"]:.3g} | {v["cv_x"]:.3f} | {v["cv_y"]:.3f} | {v["edge"]} |')
    P('\nReference arms in the same 5-block CV (maps from the other four blocks): ' + ', '.join(f'{a}: x {v[0]:.3f} / y {v[1]:.3f}' for a, v in F['refs'].items()))
    P(f'\nShipped: `{F["best"]}`, evaluator check max |difference| = {F["check"]:.2e}; file `final_map_det3.json`.')
except Exception as e:
    P(f'\n(final map results missing: {e})')
open(f'{OUT}/tables.md', 'w').write('\n'.join(md))
print('\n'.join(md)[:3000])
