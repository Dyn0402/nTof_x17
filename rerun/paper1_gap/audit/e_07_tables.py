"""e_07_tables.py -- markdown tables for RESULT.md -> $WORK/audit/e/tables.md"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import e_lib as L, e_03_main as M
R = json.load(open(f'{L.OUT}/results.json')); S = pickle.load(open(f'{L.OUT}/slices_default.pkl', 'rb'))
H = json.load(open(f'{L.OUT}/headon.json')); AW = json.load(open(f'{L.OUT}/a_weight.json'))
o = []
f = lambda g, e: f'{g*1e4:+.1f} +- {e*1e4:.1f}'
sc = lambda g: g['err'] * np.sqrt(max(1.0, g['chi2'] / max(g['dof'], 1)))
o.append('## T1 default gradients (units 1e-4 per 100 mm); err = bootstrap, [err*sqrt(chi2/dof) if >1]')
o.append('| view/axis | run | b | D | mean slope (b+qG) | LS slope in range | chi2/dof b, D |')
o.append('|---|---|---|---|---|---|---|')
for k in ('x|U', 'y|W', 'x|W', 'y|U'):
    for lab in ('both', 'run1', 'run2'):
        d = R['default'][f'{k}|{lab}']
        o.append(f'| {k} | {lab} | {f(d["b"]["grad"], d["b"]["err"])} [{sc(d["b"])*1e4:.1f}] | {f(d["D"]["grad"], d["D"]["err"])} [{sc(d["D"])*1e4:.1f}] | {f(d["s_mean"]["grad"], d["s_mean"]["err"])} | {f(d["sl"]["grad"], d["sl"]["err"])} | {d["b"]["chi2"]:.1f}, {d["D"]["chi2"]:.1f} /{d["b"]["dof"]} |')
o.append('\n## T2 significances (both runs): (measured - predicted)/sqrt(err^2 + pred_err^2); pred err from joint.json errors')
J = L.JOINT['arms']['all']['errors']
for k in ('x|U', 'y|W', 'x|W', 'y|U'):
    d = R['default'][f'{k}|both']; P = R['pred'][k]
    own = k in ('x|U', 'y|W'); kk = 'kuu' if k == 'x|U' else 'kww' if k == 'y|W' else 'kuw'
    eh, ec = J[f'h_{kk}'] / 100, J[f'c_{kk}'] / 100
    eb = np.hypot(sc(d['b']), eh)
    z_b = (d['b']['grad'] - P['b']) / eb
    s = f'{k}: b meas {d["b"]["grad"]*1e4:+.0f} pred {P["b"]*1e4:+.0f} -> {z_b:+.1f} sigma | D meas {d["D"]["grad"]*1e4:+.0f}+-{sc(d["D"])*1e4:.0f}: '
    for nm, key, pe in (('audited', 'D_audit', np.hypot(eh, ec)), ('flat', 'D_flat', eh), ('parallel', 'D_par', 0.0)):
        s += f'{nm} {P[key]*1e4:+.0f} -> {(d["D"]["grad"]-P[key])/np.hypot(sc(d["D"]), pe):+.1f}s; '
    o.append(s)
o.append('\n## T3 per-slice values (both runs): slice coord, b, D, mean slope over my range (b+q(z0+z1)), nev; z range')
for k in ('x|U', 'y|W'):
    d = R['default'][f'{k}|both']; res = S[(k[0], k[2], 'both')]
    o.append(f'{k}: z range {d["zr"][0]:.1f}-{d["zr"][1]:.1f} mm')
    for r in res:
        dd = M.derived(r['par'], r['zr']); db = M.derived(r['parb'], r['zr'])
        o.append(f'  {r["coord"]:+.2f}  b {dd["b"]:+.4f}+-{db["b"].std():.4f}  q {dd["q"]*1e4:+.2f}e-4/mm  D {dd["D"]:+.4f}+-{db["D"].std():.4f}  s_far {dd["s_far"]:+.4f}+-{db["s_far"].std():.4f}  s_mean(range) {dd["s_mean"]:+.4f}  s_full {dd["s_full"]:+.4f}  nev {r["nev"]}')
o.append('\n## T4 leave-one-slice-out (both runs)')
for k in ('x|U', 'y|W'):
    res = S[(k[0], k[2], 'both')]
    for drop in (None, 0, 4):
        r = [x for i, x in enumerate(res) if i != drop]
        gb, gD = M.gradient(r, 'b'), M.gradient(r, 'D')
        o.append(f'{k} drop {drop}: b {f(gb["grad"], gb["err"])}  D {f(gD["grad"], gD["err"])}')
o.append('\n## T5 systematics (both runs), own axes: b, D gradients and implied a=(w0v - b)/D')
for name, d in R['sys'].items():
    s = f'{name:42s}'
    for k in ('x|U', 'y|W'):
        if k in d:
            w = R['w0v'][k]['grad']
            a = (w - d[k]['b']['grad']) / d[k]['D']['grad']
            s += f' | {k}: b {f(d[k]["b"]["grad"], d[k]["b"]["err"])} D {f(d[k]["D"]["grad"], d[k]["D"]["err"])} a {a:.2f}'
    o.append(s)
o.append('\n## T6 head-on only (|tan|<0.06, z_k from the full regression)')
for k, v in H.items():
    o.append(f'{k}: b {f(v["b"]["grad"], v["b"]["err"])}  D {f(v["D"]["grad"], v["D"]["err"])}  mean {f(v["s_mean"]["grad"], v["s_mean"]["err"])}')
o.append('\n## T7 w0/v gradient from the pass-1 tables (ladder intercept / v), slices')
for k, v in R['w0v'].items():
    o.append(f'{k}: {f(v["grad"], v["err"])}  per run {[(round(a*1e4,1), round(b*1e4,1)) for a, b in v["per_run"]]}')
o.append('\n## T8 weight a'); o.append(json.dumps(AW))
o.append('\n## T9 4x4 plane fits (gradient per 100 mm, e-4), err bootstrap; chi2 (13 dof)')
for k, v in R['grid'].items():
    o.append(f'{k}: gU {v["gU"]*1e4:+.1f}+-{v["eU"]*1e4:.1f}  gW {v["gW"]*1e4:+.1f}+-{v["eW"]*1e4:.1f}  chi2 {v["chi2"]:.0f}')
open(f'{L.OUT}/tables.md', 'w').write('\n'.join(o)); print('\n'.join(o))
