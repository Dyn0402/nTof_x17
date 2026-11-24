import json, sys
sys.path.insert(0, '.')
from d_lib import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
P2 = json.load(open(f'{OUT}/part2.json')); P2b = json.load(open(f'{OUT}/part2b.json')); P2c = json.load(open(f'{OUT}/part2c.json')); P2d = json.load(open(f'{OUT}/part2d.json'))
L = []
# matching check
L.append('| sub-run | train | N det6 | N det7 | matched | max abs diff of M3 slopes (both tables) | corr M3 slope x / y | max abs (rotated tan_ref - m3_tan), det6 / det7 | theta det6 / det7 | rms(tan_ref_x det6 - det7), chamber frame | rms after rotation to M3 |')
L.append('|---|---|---|---|---|---|---|---|---|---|---|')
for v in ('full',):
    for t, c in P2[v]['tests'].items():
        tr_ = {'long': 'longer', 'longer': 'long', 'short': 'long'}[t]
        L.append(f'| {t} | {tr_} | {c["n6"]} | {c["n7"]} | {c["n_matched"]} | {c["max_dm3"]:.1e} | {c["corr_m3x"]:.6f} / {c["corr_m3y"]:.6f} | {c["max_rot6"]:.0e} / {c["max_rot7"]:.0e} | {c["th6"]} / {c["th7"]} | {c["rms_diff_tanref_x_chamberframe"]:.5f} | {c["rms_diff_after_rot_x"]:.0e} |')
open(f'{OUT}/tables_part2_match.md', 'w').write('\n'.join(L))
def fmt(t, d=2): return f'{t[0]:.{d}f}±{t[1]:.{d}f}'
M = []
for var in ('full', 'xhalf'):
    for samp in ('main', 'with_short'):
        r = P2[var][samp]
        M.append(f'\n#### {"det7 x >= 200 mm only" if var=="xhalf" else "all selected events"}, test sets: {"long, longer" if samp=="main" else "long, longer, short"} (matched events {sum(r["n"].values())})\n')
        M.append('| det6 / det7 angle | s68(d) X-axis | s68(d) Y-axis | s68(det6-M3) X / Y | s68(det7-M3) X / Y |')
        M.append('|---|---|---|---|---|')
        names = {'p1|p1': 'pass 1 / pass 1', 'const|const': 'constants / constants'}
        for arm in ('w0+v6', 'w0+v8', 'w06', 'w08'):
            lab = f'{"w0+v" if "v" in arm else "w0"} map n={arm[-1]}'
            names[f'{arm}|{arm}'] = f'{lab} / {lab}'
            names[f'{arm}|const'] = f'{lab} / constants (det6 mapped only)'
            names[f'const|{arm}'] = f'constants / {lab} (det7 mapped only)'
        for c, lab in names.items():
            o = r['s68'][c]
            M.append(f'| {lab} | {o["x"]["d"][0]:.3f}±{o["x"]["d"][1]:.3f} | {o["y"]["d"][0]:.3f}±{o["y"]["d"][1]:.3f} | {o["x"]["r6"][0]:.3f} / {o["y"]["r6"][0]:.3f} | {o["x"]["r7"][0]:.3f} / {o["y"]["r7"][0]:.3f} |')
        M.append('\nChange of s68(d)^2 (deg^2) against the constants/constants configuration, observed and predicted from the falls of s68^2 of the two chambers against M3 (predicted = -(fall6 + fall7)); diff = observed - predicted.\n')
        M.append('| map | configuration | axis | observed change | predicted change | observed - predicted | fall det6 | fall det7 |')
        M.append('|---|---|---|---|---|---|---|---|')
        for arm in ('w0+v6', 'w0+v8', 'w06', 'w08'):
            lab = f'{"w0+v" if "v" in arm else "w0"} n={arm[-1]}'
            for cfg, cl in ((f'{arm}|{arm}', 'both mapped'), (f'{arm}|const', 'det6 only'), (f'const|{arm}', 'det7 only')):
                for ax in 'xy':
                    p = r['pred'][cfg][ax]
                    M.append(f'| {lab} | {cl} | {ax} | {fmt(p["delta_obs"])} | {fmt(p["delta_pred"])} | {fmt(p["diff"])} | {fmt(p["fall6"])} | {fmt(p["fall7"])} |')
open(f'{OUT}/tables_part2.md', 'w').write('\n'.join(M))
V = ['| sample | map | axis | rms/s68 of r6 | rms/s68 of d | var(d) fall obs | pred | obs-pred (both) | det6 only obs-pred | det7 only obs-pred |', '|---|---|---|---|---|---|---|---|---|---|']
for k, o in P2c.items():
    var, arm = k.split('|')
    for ax in 'xy':
        b = o[ax]['both']
        V.append(f'| {var} | {arm} | {ax} | {b["rms_over_s68_r6"][0]:.2f} | {b["rms_over_s68_d"][0]:.2f} | {fmt(b["obs"])} | {fmt(b["pred"])} | {fmt(b["diff"])} | {fmt(o[ax]["det6 only"]["diff"])} | {fmt(o[ax]["det7 only"]["diff"])} |')
open(f'{OUT}/tables_part2c.md', 'w').write('\n'.join(V))
D = ['| sample | map | axis | slope of d on (D6-D7), differential part (real: -1; telescope: 0) | slope of (r6+r7)/2 on (D6+D7)/2, common part | share of correction variance that is common | slope of r6 on D6 | slope of d on D6 (joint fit) | slope of d on D7 (joint fit, real: +1) | corr(D6, D7) |', '|---|---|---|---|---|---|---|---|---|---|']
for k, o in P2d.items():
    var, arm = k.split('|'); b = P2b[k]
    for ax in 'xy':
        D.append(f'| {var} | {arm} | {ax} | {fmt(o[ax]["b_d_on_2Da"])} | {fmt(o[ax]["b_rs_on_Ds"])} | {o[ax]["frac_var_common"][0]:.2f} | {fmt(b[ax]["b_r6"])} | {fmt(b[ax]["b_d6"])} | {fmt(b[ax]["b_d7"])} | {b[ax]["corr_D6_D7"][0]:.2f} |')
open(f'{OUT}/tables_part2d.md', 'w').write('\n'.join(D))
# figure
fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
labs, obs, pred, eo, ep = [], [], [], [], []
r = P2['full']['main']['pred']
for arm in ('w0+v6', 'w0+v8'):
    for cfg, cl in ((f'{arm}|{arm}', 'both'), (f'{arm}|const', 'det6'), (f'const|{arm}', 'det7')):
        for a in 'xy':
            p = r[cfg][a]; labs.append(f'n={arm[-1]} {cl} {a}'); obs.append(p['delta_obs'][0]); eo.append(p['delta_obs'][1]); pred.append(p['delta_pred'][0]); ep.append(p['delta_pred'][1])
x = np.arange(len(labs))
ax[0].bar(x - 0.2, pred, 0.4, yerr=ep, label='predicted from the fall against M3', color='0.65', capsize=1.5)
ax[0].bar(x + 0.2, obs, 0.4, yerr=eo, label='observed', color='C3', capsize=1.5)
ax[0].set_xticks(x); ax[0].set_xticklabels(labs, rotation=70, fontsize=6, ha='right'); ax[0].axhline(0, color='k', lw=0.6)
ax[0].set_ylabel(r'change of s68$^2$ of d (deg$^2$)'); ax[0].legend(fontsize=6); ax[0].set_title('d = det6 - det7, w0+v maps', fontsize=9)
lab2, bd, ed, bs, es = [], [], [], [], []
for k in ('full|w0+v6', 'full|w0+v8', 'full|w06', 'full|w08', 'xhalf|w0+v6', 'xhalf|w06'):
    for a in 'xy':
        o = P2d[k][a]; lab2.append(f'{k.replace("full|","").replace("xhalf|","D x>=200 ")} {a}'); bd.append(o['b_d_on_2Da'][0]); ed.append(o['b_d_on_2Da'][1]); bs.append(o['b_rs_on_Ds'][0]); es.append(o['b_rs_on_Ds'][1])
x = np.arange(len(lab2))
ax[1].errorbar(x - 0.1, bd, ed, fmt='o', ms=4, color='C3', label='differential part, seen in d', capsize=2)
ax[1].errorbar(x + 0.1, bs, es, fmt='s', ms=4, color='C0', label='common part, against M3 only', capsize=2)
ax[1].axhline(-1, color='k', lw=0.6, ls='--'); ax[1].axhline(0, color='k', lw=0.6)
ax[1].set_xticks(x); ax[1].set_xticklabels(lab2, rotation=70, fontsize=6, ha='right'); ax[1].set_ylabel('regression slope of the error on the correction'); ax[1].legend(fontsize=6)
ax[1].set_title('slope -1: the correction removes a real error', fontsize=9)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_part2.png', dpi=160)
