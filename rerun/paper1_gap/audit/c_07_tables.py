import sys, json
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
from c_lib import *
K = json.load(open(f'{OUT}/kfit.json')); R = json.load(open(f'{OUT}/res625.json')); C = json.load(open(f'{OUT}/cells.json'))
G = json.load(open(f'{OUT}/gapdish.json')); M = json.load(open(f'{OUT}/mount.json'))
L = []
def P(s=''): L.append(s)
# fixed-rate vs tan chi2
D = load(ROWS['r625']); chk = []
for p in 'xy':
    Gp = {q: np.concatenate([view(load(ROWS[k]), p)[q] for k in ('run1', 'run2')]) for q in ('U', 'W', 'w', 'tr')}
    T = view(D, p)
    mp = Maps(Gp['U'], Gp['W'], Gp['w'], Gp['tr'], 4)
    ap, af, s, A = mp.ev(T['U'], T['W'])
    X1 = np.vstack([np.ones(len(ap)), T['tr'], T['tr'] * s, ap + af]).T          # tan-unit column (scaled by v')
    X2 = np.vstack([np.ones(len(ap)), T['tr'], T['tr'] * s, mp.vg * (ap + af)]).T  # um/ns column
    c1, k1, r1, _ = lsq_clip(X1, T['w']); c2, k2, r2, _ = lsq_clip(X2, T['w'])
    # RSS on a common event set
    keep = k1 & k2
    rr = lambda X, c: float(np.sum((T['w'] - X @ c)[keep] ** 2))
    chk.append((p, c1[3] / c1[1], c2[3], c2[3] * mp.vg / c1[1], rr(X1, c1), rr(X2, c2), mp.vg / c1[1]))
P('## 0. What differs in the 6-25 mounting')
P()
P('| | run 1 (6-27 sat) | run 2 (6-28 p2) | 6-25 day |')
P('|---|---|---|---|')
def row(name, f): P(f'| {name} | ' + ' | '.join(f(M[k]) for k in ('run1', 'run2', 'r625')) + ' |')
row('theta_align_deg', lambda m: f"{m['theta']:.2f}")
row('alignment z_x / z_y (mm)', lambda m: f"{m['zx']:.0f} / {m['zy']:.0f}")
row('x_offset / y_offset (mm)', lambda m: f"{m['xoff']:.1f} / {m['yoff']:.1f}")
row('selected events (golden-style cuts)', lambda m: f"{m['n']}")
row('ladder slope v_x (um/ns)', lambda m: f"{m['ladder_x']['v']:.2f} +- {m['ladder_x']['ve']:.2f}")
row('ladder slope v_y (um/ns)', lambda m: f"{m['ladder_y']['v']:.2f} +- {m['ladder_y']['ve']:.2f}")
row('ladder offset w0_x (um/ns)', lambda m: f"{m['ladder_x']['w0']:+.3f} +- {m['ladder_x']['w0e']:.3f}")
row('ladder offset w0_y (um/ns)', lambda m: f"{m['ladder_y']['w0']:+.3f} +- {m['ladder_y']['w0e']:.3f}")
row('corr(pass-1 tan, tan_ref) x / y', lambda m: f"{m['sign_tan_pass1_vs_ref_x']:+.3f} / {m['sign_tan_pass1_vs_ref_y']:+.3f}")
row('bundle v_drift / w0 (x,y) / kw (x,y)', lambda m: f"{m['bundle_v']:.1f} / {m['w0']['x']:+.3f},{m['w0']['y']:+.3f} / {m['kw']['x']:.3f},{m['kw']['y']:.3f}")
row('bundle bow block', lambda m: 'same block (source_row sat_det3, identical content)' if m['bowhash'] == M['run1']['bowhash'] else 'DIFFERENT')
P()
P('## 2. Amplitude fits on the 6-25 row (pooled golden map, bootstrap of the 6-25 events, 150 resamples)')
P()
P('Raw k, then ctrl = mean of the two golden-to-golden controls (run1 map on run2, run2 map on run1), then k/ctrl. References: geometric = 1 (in tan units), fixed rate = v_golden/v\' (shown per row).')
P()
P('| view | n | v_g/v\' | k_plane | k_fine | k_s | ctrl k_plane | ctrl k_fine | ctrl k_s | k_plane/ctrl | k_fine/ctrl | k_s/ctrl |')
P('|---|---|---|---|---|---|---|---|---|---|---|---|')
for p in 'xy':
    for n in (4, 6, 8):
        k = K['k'][f'{n}_{p}']; c1 = K['ctrl'][f'{n}_{p}_run1->run2']; c2 = K['ctrl'][f'{n}_{p}_run2->run1']
        s = []; sc = []; sr = []
        for q in ('kp', 'kf', 'ks'):
            s.append(f"{k[q]:.2f} +- {k[q+'_e']:.2f}")
            cm = 0.5 * (c1[q] + c2[q]); ce = 0.5 * np.hypot(c1[q + '_e'], c2[q + '_e']); sc.append(f"{c1[q]:.2f} / {c2[q]:.2f}")
            sr.append(f"{k[q]/cm:.2f} +- {k[q]/cm*np.hypot(k[q+'_e']/k[q], np.hypot(ce, 0.5*abs(c1[q]-c2[q]))/cm):.2f}")
        P(f"| {p.upper()} | {n} | {k['ratio_vg_vp']:.2f} | " + ' | '.join(s) + ' | ' + ' | '.join(sc) + ' | ' + ' | '.join(sr) + ' |')
P()
P('Control checks (same mounting and field; expected 1 up to noise attenuation): noise variance of the golden fine map relative to its cell variance, 1 - lambda:')
P()
P('| view | n | noise/variance golden pooled | noise/variance run 1 | run 2 | noise/variance 6-25 own |')
P('|---|---|---|---|---|---|')
for p in 'xy':
    for n in (4, 6):
        N = C['noise']
        f = lambda k: N[f'{p}{n}:{k}']['mean_err2'] / N[f'{p}{n}:{k}']['var_fine']
        P(f"| {p.upper()} | {n} | {f('gold'):.3f} | {f('run1'):.3f} | {f('run2'):.3f} | {f('r625'):.3f} |")
P()
P('Fixed-rate parametrisation vs tan parametrisation on the same events (4x4, pooled golden map): the two are the same linear model with the column rescaled by v_g/v\', so RSS is identical.')
P()
P('| view | k_fine+plane (tan) | k\' (um/ns map) | k\' v_g/v\' | RSS tan | RSS rate | v_g/v\' |')
P('|---|---|---|---|---|---|---|')
for c in chk:
    P(f"| {c[0].upper()} | {c[1]:.3f} | {c[2]:.3f} | {c[3]:.3f} | {c[4]:.1f} | {c[5]:.1f} | {c[6]:.3f} |")
P()
P('## Cell-level comparison of the maps (slope of second on first, correlation; event bootstrap, 100 resamples)')
P()
P('| view | n | pair | A_fine slope | A_fine corr | A_plane slope | S slope | S corr |')
P('|---|---|---|---|---|---|---|---|')
for key in ('x4', 'y4', 'x6', 'y6'):
    for nm in ('run1_vs_run2', 'gold_vs_r625', 'run1_vs_r625', 'run2_vs_r625'):
        r = C['cells'][key]
        g = lambda q, f: r[f'{nm}:{q}'][f]
        P(f"| {key[0].upper()} | {key[1:]} | {nm} | {g('Afine','slope'):.2f} +- {g('Afine','slope_e'):.2f} | {g('Afine','corr'):.2f} +- {g('Afine','corr_e'):.2f} | {g('Aplane','slope'):.2f} +- {g('Aplane','slope_e'):.2f} | {g('S','slope'):.2f} +- {g('S','slope_e'):.2f} | {g('S','corr'):.2f} |")
P()
P('## 3. Resolution on the 6-25 row: s68 of theta - theta_ref (deg), |tan_ref| <= 0.6, difference to pass 1 with paired bootstrap error')
P()
P('| arm | X s68 | X diff | Y s68 | Y diff |')
P('|---|---|---|---|---|')
names = list(R['x']['r625'].keys())
for a in names:
    x, y = R['x']['r625'][a], R['y']['r625'][a]
    P(f"| {a} | {x['s68']:.3f} | {x['dif']:+.3f} +- {x['dif_e']:.3f} | {y['s68']:.3f} | {y['dif']:+.3f} +- {y['dif_e']:.3f} |")
P()
P('Reference inside the golden mounting (map of one run on the other, same arms; s68 deg):')
P()
P('| view | direction | pass1 | const | 4 w0+v | 6 w0+v | 8 w0+v |')
P('|---|---|---|---|---|---|---|')
for p in 'xy':
    for d, r in R[p]['golden_ref'].items():
        P(f"| {p.upper()} | {d} | {r['pass1']:.3f} | {r['const']:.3f} | {r['4 w0+v']:.3f} | {r['6 w0+v']:.3f} | {r['8 w0+v']:.3f} |")
P()
P('## 4b/5. Gap proxy and dish (cell maps by the reference position at mid-gap, as cellobs.py)')
P()
P('| n | pair | quantity | slope (second on first) | corr |')
P('|---|---|---|---|---|')
for n in ('4', '6'):
    r = G[n]
    for nm in ('run1_vs_run2', 'gold_vs_r625'):
        for q, lab in (('v', 'v_rel (both views)'), ('Q', 'q_u90 rel'), ('P', 'v x q_u90 product'), ('hpx', 'hp_x, plane removed (mm)'), ('hpy', 'hp_y, plane removed (mm)')):
            e = r[f'{nm}:{q}']
            P(f"| {n} | {nm} | {lab} | {e['slope']:.2f} +- {e['slope_e']:.2f} | {e['corr']:.2f} +- {e['corr_e']:.2f} |")
P()
P('Amplitudes (sd over cells, relative): ' + '; '.join(f"{n}x{n} {k.split(':')[1]}: v {v['sd_v']:.3f}, q_u90 {v['sd_Q']:.3f}, product {v['sd_P']:.3f}, mean q_u90 {v['meanQ']:.0f} ns" for n in ('4', '6') for k, v in G[n].items() if k.startswith('amp') and k.split(':')[1] in ('gold', 'r625')))
P()
P('Dish of the position route, h = h0 + lu U + lw W + kuu U^2/2 + kww W^2/2 + kuw U W (mm, U,W in 100 mm). Error = event bootstrap (formal error scaled by sqrt(chi2/dof) is about 1.5 to 2 times larger):')
P()
P('| n | view | mounting | kuu | kww | kuw | chi2/dof |')
P('|---|---|---|---|---|---|---|')
for n in ('4', '6'):
    for p in 'xy':
        for k in ('gold', 'r625', 'run1', 'run2'):
            e = G[n][f'dish:{k}:{p}']; c = e['c']; b = e.get('boot_e', e['e'])
            tag = '' if 'boot_e' in e else ' (formal)'
            P(f"| {n} | {p.upper()} | {k} | {c[3]:.2f} +- {b[3]:.2f}{tag} | {c[4]:.2f} +- {b[4]:.2f}{tag} | {c[5]:.2f} +- {b[5]:.2f}{tag} | {e['rchi2']:.1f} |")
open(f'{OUT}/tables.md', 'w').write('\n'.join(L))
print('\n'.join(L))
