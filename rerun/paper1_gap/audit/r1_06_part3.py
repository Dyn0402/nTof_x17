"""r1_06_part3.py: the prediction as a tracking correction (s68 of theta - theta_ref).  -> part3.json"""
import sys
from r1_lib2 import *
NBS = 200
D = {r: load(r) for r in ('run1', 'run2', 'r625')}
V = {(r, p): view(D[r]['tab'], p) for r in D for p in 'xy'}
rng = np.random.default_rng(55)
OTHER = {'run1': 'run2', 'run2': 'run1'}

def dang(t, tr):
    return np.degrees(np.arctan(t) - np.arctan(tr))

def apply(v, dw, vr):
    c = TT.robust_const(v['tr'], v['w'], dw, vr)
    return (v['w'] - c[0] - dw) / (c[1] * vr)

def report(arms, v, label):
    """arms: dict name -> tan array (or list of arrays for repeated splits).  paired bootstrap on the events."""
    n = len(v['tr'])
    bidx = [rng.integers(0, n, n) for _ in range(NBS)]
    D_ = {k: ([dang(x, v['tr']) for x in t] if isinstance(t, list) else [dang(t, v['tr'])]) for k, t in arms.items()}
    def s(k, idx=None):
        return float(np.mean([TT.s68(d if idx is None else d[idx]) for d in D_[k]]))
    out = {}
    for k in arms:
        out[k] = dict(s68=s(k))
    base = {k: np.array([s(k, b) for b in bidx]) for k in arms}
    for k in arms:
        out[k]['s68_boot_sd'] = float(base[k].std())
        for ref in ('pass1', 'mesh', 'const'):
            if ref in arms:
                dd = base[k] - base[ref]
                out[k][f'd_{ref}'] = float(out[k]['s68'] - out[ref]['s68']); out[k][f'd_{ref}_err'] = float(dd.std())
    for c in [k for k in arms if k.startswith('direct 8x8')]:
        den0 = out['pass1']['s68'] - out[c]['s68']
        for k in arms:
            num0 = out['pass1']['s68'] - out[k]['s68']
            fr = (base['pass1'] - base[k]) / (base['pass1'] - base[c])
            out[k][f'frac_{c}'] = float(num0 / den0); out[k][f'frac_{c}_err'] = float(fr.std())
    return out, base

J = {}
# ======================================================== golden
for dst in ('run1', 'run2'):
    src = OTHER[dst]
    for p in 'xy':
        v = V[(dst, p)]; vs = V[(src, p)]
        i = 0 if p == 'x' else 1
        arms = {'pass1': v['tp']}
        c0 = TT.robust_const(v['tr'], v['w'], 0.0, 1.0); vv = c0[1]
        arms['const'] = apply(v, 0.0, 1.0)
        gh = hgrad(v['U'], v['W'], 'h')[i]
        arms['mesh'] = apply(v, vv * gh, 1.0)
        fo, fs = Fld(f'T_{src}'), Fld(f'T_{dst}')
        ggo, ggs = fo.grad(v['U'], v['W'])[i], fs.grad(v['U'], v['W'])[i]
        arms['mesh+gap other (T)'] = apply(v, vv * (gh + 0.5 * ggo), 1.0)
        arms['mesh+gap same (T)'] = apply(v, vv * (gh + 0.5 * ggs), 1.0)
        ghp = hgrad(v['U'], v['W'], 'perview')[i]
        arms['mesh (per-view h)'] = apply(v, vv * ghp, 1.0)
        arms['mesh+gap other (per-view h, T)'] = apply(v, vv * (ghp + 0.5 * ggo), 1.0)
        arms['mesh+gap same (per-view h, T)'] = apply(v, vv * (ghp + 0.5 * ggs), 1.0)
        arms['mesh+gap + v_rel, other (per-view h, T)'] = apply(v, vv * (ghp + 0.5 * ggo), fo.vrel(v['U'], v['W']))
        arms['mesh+gap + v_rel, same (per-view h, T)'] = apply(v, vv * (ghp + 0.5 * ggs), fs.vrel(v['U'], v['W']))
        arms['gap only other (T)'] = apply(v, vv * (0.5 * ggo), 1.0)
        arms['mesh+gap + v_rel, other (T)'] = apply(v, vv * (gh + 0.5 * ggo), fo.vrel(v['U'], v['W']))
        arms['mesh+gap + v_rel, same (T)'] = apply(v, vv * (gh + 0.5 * ggs), fs.vrel(v['U'], v['W']))
        arms['v_rel only, other (T)'] = apply(v, 0.0, fo.vrel(v['U'], v['W']))
        fqo, fqs = Fld(f'q_{src}'), Fld(f'q_{dst}')
        ggqo, ggqs = fqo.grad(v['U'], v['W'])[i], fqs.grad(v['U'], v['W'])[i]
        arms['mesh+gap other (q_u90)'] = apply(v, vv * (gh + 0.5 * ggqo), 1.0)
        arms['mesh+gap same (q_u90)'] = apply(v, vv * (gh + 0.5 * ggqs), 1.0)
        # amplitude calibrated on the other run (same-run field there, applied same-run here)
        vsrc = vs; ghs = hgrad(vsrc['U'], vsrc['W'], 'h')[i]
        ggsrc = Fld(f'T_{src}').grad(vsrc['U'], vsrc['W'])[i]
        ab = abfit(vsrc, ghs, ggsrc)
        arms[f'mesh+gap same, (a,b) from other run ({ab["a"]:.2f},{ab["b"]:.2f})'] = apply(v, vv * (ab['a'] * gh + ab['b'] * ggs), 1.0)
        # direct cell maps from the other run
        for n in (4, 8):
            C = [dict(U=c['U'], W=c['W'], w0=c['w0'], v=c['v']) for c in []]
            eu, ew = np.linspace(*CL.UL, n + 1), np.linspace(*CL.WL, n + 1)
            cells = []
            ii, jj = np.searchsorted(eu, vs['U']) - 1, np.searchsorted(ew, vs['W']) - 1
            for a in range(n):
                for b in range(n):
                    m = (ii == a) & (jj == b)
                    w0, vc, *_ = CO.line(vs['tr'][m], vs['w'][m])
                    cells.append(dict(U=0.5 * (eu[a] + eu[a + 1]), W=0.5 * (ew[b] + ew[b + 1]), w0=w0, v=vc))
            vm = np.nanmean([c['v'] for c in cells]); wm = np.nanmean([c['w0'] for c in cells])
            for c in cells:
                c['vrel'] = c['v'] / vm; c['dw0'] = c['w0'] - wm
            vr = TT.bilinear(cells, 'vrel', v['U'], v['W']); dw = TT.bilinear(cells, 'dw0', v['U'], v['W'])
            arms[f'direct {n}x{n} w0 (other run)'] = apply(v, dw, 1.0)
            arms[f'direct {n}x{n} w0+v (other run)'] = apply(v, dw, vr)
        res, base = report(arms, v, f'{dst} {p}')
        J[f'{dst}:{p}'] = dict(res=res, n=len(v['tr']), ab=ab)
        print(dst, p, {k: round(x['s68'], 3) for k, x in res.items()}, flush=True)
json.dump(J, open(f'{OUT}/part3_golden.json', 'w'), indent=1)
