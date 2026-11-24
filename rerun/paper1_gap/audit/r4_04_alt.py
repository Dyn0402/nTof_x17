"""r4_04_alt: q_u50 shape (immune to the 1740 ns window cut), per-view q_u90, injection test of the window cut."""
import sys; sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/rerun/paper1_gap/audit')
import r4_lib as L, numpy as np, json
keys = [L.R625, L.GOLD2, L.GOLD1] + [L.SC + f'/drift_scan_resist_490V_drift_{v}V' for v in (300, 500, 700, 900, 1100)]
out = {}
def show(tag, f):
    c, e = f['c'], f['err']
    print(f'  {tag:14s} N {f["N"]:5d} guu {c[3]:+.3f}+-{e[3]:.3f} gww {c[4]:+.3f}+-{e[4]:.3f} guw {c[5]:+.3f}+-{e[5]:.3f}', flush=True)
for rel in keys:
    print(rel.split('/')[-1][-24:], flush=True)
    d = L.load_events(rel)
    r = {}
    f = L.fit_shape(d, nboot=150); show('q_u90', f); r['q90'] = [f['c'][:6].tolist(), f['err'][:6].tolist()]
    d2 = d.copy(); q50 = 0.5 * (d.x_q_u50 + d.y_q_u50); d2 = d2[q50 > 20].copy(); d2['q'] = 0.5 * (d2.x_q_u50 + d2.y_q_u50); d2['lq'] = np.log(d2.q)
    f = L.fit_shape(d2, nboot=150); show('q_u50', f); r['q50'] = [f['c'][:6].tolist(), f['err'][:6].tolist()]
    d3 = d.copy(); d3['lq'] = np.log(d3.x_q_u90); f = L.fit_shape(d3, nboot=150); show('x_q_u90', f); r['x90'] = [f['c'][:6].tolist(), f['err'][:6].tolist()]
    d3['lq'] = np.log(d3.y_q_u90); f = L.fit_shape(d3, nboot=150); show('y_q_u90', f); r['y90'] = [f['c'][:6].tolist(), f['err'][:6].tolist()]
    d4 = d[(d.x_q_uend < 1700) & (d.y_q_uend < 1700)]
    if len(d4) < len(d): f = L.fit_shape(d4, nboot=150); show('uend<1700', f); r['uend'] = [f['c'][:6].tolist(), f['err'][:6].tolist()]
    out[rel] = r
json.dump(out, open(f'{L.OUT}/fits_alt.json', 'w'))
