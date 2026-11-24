"""D: per-row summary table + reproduction of chamber A's committed track_grid numbers with the loader."""
import json, sys
sys.path.insert(0, '.')
from d_lib import *
rows = {}
for k in ROWS:
    d = load(k); a = align(k); b = bundle(k)
    r = dict(label=ROWS[k][3], det=ROWS[k][0], N=len(d), mesh_V=VOLT[k][0], drift_V=VOLT[k][1], theta=a['theta_deg'],
             z_x=a['z_x'], z_y=a['z_y'], slot='top' if a['z_x'] > 500 else 'bottom', v_drift_bundle=b['v_drift'],
             x_off=a['x_offset'], y_off=a['y_offset'])
    for p in 'xy':
        U, W, w, tr, tp = view(d, p)
        w0, v, ew0, ev, nk = CO.line(tr, w)
        r[f'w0_{p}'] = w0; r[f'v_{p}'] = v; r[f'ev_{p}'] = ev; r[f'ew0_{p}'] = ew0
    if k.startswith('det7'):
        r['N_xhalf'] = len(load(k, xhalf=True))
        U, W, w, tr, tp = view(load(k, xhalf=True), 'x'); w0, v, *_ = CO.line(tr, w); r['w0_x_half'] = w0; r['v_x_half'] = v
    rows[k] = r; print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
json.dump(rows, open(f'{OUT}/rows.json', 'w'), indent=1, default=float)
# reproduction: chamber A, test sat (run1) train p2, and vice versa
ref = json.load(open(f'{REPO}/rerun/paper1_gap/results/track_grid.json'))
for run, (te, tr_) in {'run1': ('det3:sat', 'det3:p2'), 'run2': ('det3:p2', 'det3:sat')}.items():
    dte, dtr = load(te), load(tr_)
    # compare selection with GM.load_table
    g = GM.load_table(WORK, GM.RUNS[run]['rel'])
    print(run, 'N mine', len(dte), 'N GM', len(g))
    for p in 'xy':
        T, S = view(dte, p), view(dtr, p)
        out = {'pass1': TT.metrics(T[4], T[3], T[0], T[1])['s68']}
        for n in (4, 6, 8, 10):
            A = arms(S, T, n)
            for arm in ('w0+v', 'w0'):
                out[f'{n}x{n} {arm}'] = TT.metrics(A[arm], T[3], T[0], T[1])['s68']
        mx = max(abs(out[k] - ref[run][p][k]['s68']) for k in out)
        print(run, p, 'max |diff| vs committed', mx, {k: (round(v, 4), round(ref[run][p][k]['s68'], 4)) for k, v in out.items() if k in ('pass1', '6x6 w0+v', '8x8 w0')})
