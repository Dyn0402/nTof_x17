import json, sys
sys.path.insert(0, '.')
from d_lib import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
P = json.load(open(f'{OUT}/part34.json'))
ok = lambda v: v['r'] == v['r'] and v['ncell'] >= 12
def rline(k, v):
    kA, kB, p, n, q, fr = k.split('|')
    nul = [x for x in v['null'] if x == x]
    c = v['ceil']
    return (f'| {kA} | {kB} | {p} | {n} | {fr} | {v["r"]:+.2f}±{v["err"]:.2f} | {v["ncell"]} | ' + (f'{c:.2f}' if c == c else 'n/a') + ' | ' +
            (f'{v["r"]/c:+.2f}' if c == c and c > 0.2 else 'n/a') + f' | {min(nul):+.2f} .. {max(nul):+.2f} |')
H = '| row A | row B | view | n | frame | r (tau residual maps) | common cells | noise ceiling | r / ceiling | mirrored-map null |\n|---|---|---|---|---|---|---|---|---|---|'
S = [H]
for k, v in P['same'].items():
    if k.split('|')[4] == 'tau' and ok(v):
        S.append(rline(k, v))
Sv = [H]
for k, v in P['same'].items():
    if k.split('|')[4] == 'v' and ok(v):
        Sv.append(rline(k, v))
X = [H]
for k, v in P['cross'].items():
    if k.split('|')[4] == 'tau' and ok(v):
        X.append(rline(k, v))
Xv = [H]
for k, v in P['cross'].items():
    if k.split('|')[4] == 'v' and ok(v):
        Xv.append(rline(k, v))
open(f'{OUT}/tables_part3.md', 'w').write('#### 3a same chamber, tau = w0/v\n' + '\n'.join(S) + '\n\n#### 3a same chamber, v\n' + '\n'.join(Sv) + '\n\n#### 3b different chambers, same slot, tau\n' + '\n'.join(X) + '\n\n#### 3b different chambers, same slot, v\n' + '\n'.join(Xv))
# summary: per chamber pair best combination (most common cells), tau, n=6 and 8
best = {}
for k, v in P['cross'].items():
    kA, kB, p, n, q, fr = k.split('|')
    if q != 'tau' or not ok(v):
        continue
    key = (kA.split(':')[0], kB.split(':')[0], p, n)
    if key not in best or v['ncell'] > best[key][1]['ncell'] or (v['ncell'] == best[key][1]['ncell'] and v['err'] < best[key][1]['err']):
        best[key] = (k, v)
B = ['| chamber pair | view | n | best row pair | r | common cells | ceiling | r / ceiling | null range |', '|---|---|---|---|---|---|---|---|---|']
for key in sorted(best):
    k, v = best[key]
    nul = [x for x in v['null'] if x == x]; c = v['ceil']
    B.append(f'| {key[0]}-{key[1]} | {key[2]} | {key[3]} | {k.split("|")[0]} / {k.split("|")[1]} | {v["r"]:+.2f}±{v["err"]:.2f} | {v["ncell"]} | ' + (f'{c:.2f}' if c == c else 'n/a') + ' | ' + (f'{v["r"]/c:+.2f}' if c == c and c > 0.2 else 'n/a') + f' | {min(nul):+.2f} .. {max(nul):+.2f} |')
open(f'{OUT}/tables_part3_best.md', 'w').write('\n'.join(B))
# part 4
C = ['| row | N cells x / y | d tau_x/dU | d tau_x/dW | d tau_y/dU | d tau_y/dW | cross difference (tau_x/dW - tau_y/dU) | significance | sum (sign check) |', '|---|---|---|---|---|---|---|---|---|']
fm = lambda t: f'{t[0]*1e3:+.2f}±{t[1]*1e3:.2f}'
for k, o in P['curl'].items():
    z = o['curl'][0] / o['curl'][1] if o['curl'][1] > 0 else np.nan
    C.append(f'| {k} | {o["ncell_x"]} / {o["ncell_y"]} | {fm(o["dtx_dU"])} | {fm(o["dtx_dW"])} | {fm(o["dty_dU"])} | {fm(o["dty_dW"])} | {fm(o["curl"])} | {z:+.1f} sigma | {fm(o["sum"])} |')
open(f'{OUT}/tables_part4.md', 'w').write('(all gradients in 1e-3 per 100 mm; tau = w0/v is dimensionless)\n\n' + '\n'.join(C))
# figures
fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
pts = {'same chamber, different run': [], 'different chambers, same slot': []}
for typ, D in (('same chamber, different run', P['same']), ('different chambers, same slot', P['cross'])):
    for k, v in D.items():
        kA, kB, p, n, q, fr = k.split('|')
        if q == 'tau' and n == '6' and (fr == 'tel') and ok(v) and v['ceil'] == v['ceil'] and v['ceil'] > 0.3:
            pts[typ].append((v['r'], v['err'], v['ceil'], v['ncell']))
for typ, col, off in (('same chamber, different run', 'C0', -0.15), ('different chambers, same slot', 'C3', 0.15)):
    a = np.array(pts[typ])
    if len(a):
        x = np.arange(len(a))
        ax[0].errorbar(np.full(len(a), 0 + off) + np.random.default_rng(1).uniform(-.05, .05, len(a)), a[:, 0], a[:, 1], fmt='o', ms=4, color=col, label=typ + f' ({len(a)})', capsize=1.5)
ax[0].axhline(0, color='k', lw=0.6); ax[0].set_xticks([]); ax[0].set_ylabel('correlation r of residual tilt maps (n=6)'); ax[0].legend(fontsize=6); ax[0].set_ylim(-1, 1.05)
ax[0].set_title('3: residual tilt maps (tau minus plane)', fontsize=9)
rows = list(P['curl'])
cx = [P['curl'][k]['dtx_dW'] for k in rows]; cy = [P['curl'][k]['dty_dU'] for k in rows]
ax[1].errorbar([c[0] * 1e3 for c in cy], [c[0] * 1e3 for c in cx], xerr=[c[1] * 1e3 for c in cy], yerr=[c[1] * 1e3 for c in cx], fmt='o', ms=3, lw=0.8, color='C3')
lim = 12
ax[1].plot([-lim, lim], [-lim, lim], 'k', lw=0.6); ax[1].set_xlim(-lim, lim); ax[1].set_ylim(-lim, lim)
ax[1].set_xlabel(r'$\partial\tau_y/\partial U$ (1e-3 per 100 mm)'); ax[1].set_ylabel(r'$\partial\tau_x/\partial W$ (1e-3 per 100 mm)'); ax[1].set_title('4: curl test, one point per run', fontsize=9)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_part34.png', dpi=160)
print(open(f'{OUT}/tables_part4.md').read()); print(open(f'{OUT}/tables_part3_best.md').read())
