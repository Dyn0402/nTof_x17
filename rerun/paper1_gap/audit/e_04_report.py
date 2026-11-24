"""e_04_report.py -- Task E: all numbers of the report -> $WORK/audit/e/results.json (+ pickles for the figures)"""
import sys, json, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import e_lib as L
import e_03_main as M
GM, CO = L.GM, L.CO
P = L.JOINT['arms']['all']['params']
PE = L.JOINT['arms']['all']['errors']

# ---- predictions: gradients per 100 mm of b = h-slope and Delta = c-slope - h-slope
def pred(view, axis):
    own = (view == 'x') == (axis == 'U')
    k = 'kuu' if view == 'x' and axis == 'U' else 'kww' if view == 'y' and axis == 'W' else 'kuw'
    hk, ck = P[f'h_{k}'] / 100, P[f'c_{k}'] / 100
    return dict(b=hk, D_audit=ck - hk, D_flat=-hk, D_par=0.0, s_audit=(hk + ck) / 2)

TAB = {}
for run in ('run1', 'run2'):
    TAB[run] = GM.load_table(L.WORK, GM.RUNS[run]['rel'])

def w0v_gradient(axis, view, runs=('run1', 'run2')):
    """gradient of w0/v across the 5 slices, errors from the per-slice line fits (both runs, weighted)."""
    e = GM.GRIDS['x5'][0] if axis == 'U' else GM.GRIDS['y5'][1]
    col = 'xm' if axis == 'U' else 'ym'
    xs, ys, es = [], [], []
    for i in range(len(e) - 1):
        num, den = [], []
        coords = []
        for run in runs:
            t = TAB[run]; s = t[(t[col] >= e[i]) & (t[col] < e[i + 1])]
            tr = s[f'tan_ref_{view}'].values; sel = np.abs(tr) <= GM.TAN_LADDER
            w0, v, w0e, ve, n = CO.line(tr[sel], s[f'{view}_w'].values[sel] * 1e3)
            # w0/v with a run-specific constant (the runs differ in the constant offset): keep the run's own mean later
            num.append((w0 / v, w0e / v)); coords.append(s[axis].mean() if False else (s[col].mean() - 200) / 100)
        xs.append(np.mean(coords)); ys.append([n_[0] for n_ in num]); es.append([n_[1] for n_ in num])
    xs = np.array(xs); ys = np.array(ys); es = np.array(es)
    # per-run gradient, then inverse-variance mean
    out = []
    for j, run in enumerate(runs):
        c = M.wls_slope(xs, ys[:, j], 1 / es[:, j] ** 2)
        A = np.vstack([np.ones_like(xs), xs]).T
        cov = np.linalg.inv(A.T @ (A / es[:, j][:, None] ** 2))
        out.append((c[1], np.sqrt(cov[1, 1])))
    g = np.array([o[0] for o in out]); e_ = np.array([o[1] for o in out])
    w = 1 / e_ ** 2
    return dict(grad=float((g * w).sum() / w.sum()), err=float(1 / np.sqrt(w.sum())), per_run=[(float(a), float(b)) for a, b in out],
                y=ys.tolist(), x=xs.tolist())

def summarize(res, view, axis, label=''):
    out = dict(view=view, axis=axis, label=label)
    for key in ('b', 'D', 's_mean', 'sl'):
        g = M.gradient(res, key)
        out[key] = dict(grad=g['grad'], err=g['err'], y=g['y'], yerr=g['yerr'], x=g['x'], chi2=g['chi2'], dof=g['dof'], icpt=g['icpt'], icpt_err=g['icpt_err'], boots=g['boots'].tolist())
    out['zr'] = res[0]['zr']
    out['nev'] = [r['nev'] for r in res]
    return out

def fmt(g): return f"{g['grad']*1e4:+.1f}±{g['err']*1e4:.1f}"

def main():
    R = {}
    D = {t: M.load(t) for t in ('', 'n2', 'n5', 'w2', 'w4')}
    # ---------- default, slices, combined and per run
    def sub_run(d, run):
        lo, hi = (10_000_000, 20_000_000) if run == 'run1' else (20_000_000, 30_000_000)
        return d[(d.eid > lo) & (d.eid < hi)]
    default = {}
    allres = {}
    for v in 'xy':
        d0 = M.base_filter(D[''][v])
        for axis in 'UW':
            for lab, d in (('both', d0), ('run1', sub_run(d0, 'run1')), ('run2', sub_run(d0, 'run2'))):
                res = M.slices_fit(d, axis)
                allres[(v, axis, lab)] = res
                default[f'{v}|{axis}|{lab}'] = summarize(res, v, axis, lab)
                print(v, axis, lab, 'b', fmt(default[f'{v}|{axis}|{lab}']['b']), 'D', fmt(default[f'{v}|{axis}|{lab}']['D']), flush=True)
    R['default'] = default
    pickle.dump(allres, open(f'{L.OUT}/slices_default.pkl', 'wb'))
    # ---------- w0/v gradients from tables
    R['w0v'] = {f'{v}|{a}': w0v_gradient(a, v) for v in 'xy' for a in 'UW'}
    R['pred'] = {f'{v}|{a}': pred(v, a) for v in 'xy' for a in 'UW'}
    # ---------- systematics (own-axis, both runs): b, D, s_mean, sl
    S = {}
    def add(name, res_by_view):
        S[name] = {f'{v}|{a}': summarize(res_by_view[(v, a)], v, a, name) for (v, a) in res_by_view}
        print('sys', name, {k: (fmt(x['b']), fmt(x['D'])) for k, x in S[name].items()}, flush=True)
    own = [('x', 'U'), ('y', 'W')]
    cross = [('x', 'W'), ('y', 'U')]
    def run_variant(name, tag='', filt=None, bf=None, tr=(0, 810), edges=M.EDGES, views=own + cross, bal=True):
        out = {}
        for v, a in views:
            d = M.base_filter(D[tag][v], **(bf or {}))
            if filt is not None: d = filt(d)
            out[(v, a)] = M.slices_fit(d, a, tr=tr, edges=edges, bal=bal)
        add(name, out)
    run_variant('default')
    run_variant('not balanced in sign of tan', bal=False)
    for tag, nm in (('n2', 'thr 2 sigma'), ('n5', 'thr 5 sigma'), ('w2', 'window +-(2+4|tan|)'), ('w4', 'window +-(5+9|tan|)')):
        run_variant(nm, tag=tag)
    run_variant('saturated events kept', bf=dict(sat='none'))
    run_variant('saturated samples dropped only', bf=dict(sat='sample'))
    run_variant('masked windows kept', bf=dict(sat='drop'), filt=None) if False else None
    run_variant('inclined only (|tan|>0.12)', filt=lambda d: d[d.tan.abs() > 0.12])
    run_variant('tan > -0.06 (positive + head-on)', filt=lambda d: d[d.tan > -0.06], bal=False)
    run_variant('tan < +0.06 (negative + head-on)', filt=lambda d: d[d.tan < 0.06], bal=False)
    run_variant('balanced, |tan|<0.3', filt=lambda d: d[d.tan.abs() < 0.3])
    run_variant('balanced, |tan|<0.45', filt=lambda d: d[d.tan.abs() < 0.45])
    run_variant('tan in (-0.06, 0.3), not balanced', filt=lambda d: d[(d.tan > -0.06) & (d.tan < 0.3)], bal=False)
    run_variant('tan in (-0.3, 0.06), not balanced', filt=lambda d: d[(d.tan < 0.06) & (d.tan > -0.3)], bal=False)
    run_variant('fit range t 30-690 ns', tr=(0, 700))
    run_variant('fit range t 90-810 ns', tr=(60, 810))
    run_variant('half-sample bins (30 ns)', edges=np.arange(-60, 901, 30.), tr=(0, 810))
    run_variant('amplitude-weighted off: A>2000', filt=lambda d: d[d.A > 2000])
    R['sys'] = S
    # masked-window rows kept (bias near dead strips)
    out = {}
    for v, a in own:
        d = D[''][v]; d = d[~d.sat_ev]
        out[(v, a)] = M.slices_fit(d, a)
    add('dead-strip windows kept', out)
    # inner 3 slices only
    inner = {}
    for (v, a) in own:
        inner[(v, a)] = allres[(v, a, 'both')][1:4]
    S['inner 3 slices'] = {f'{v}|{a}': summarize(r, v, a, 'inner3') for (v, a), r in inner.items()}
    # ---------- 4x4 grid: plane fits of b and D
    grid = {}
    ex, ey = GM.GRIDS['4x4']
    for v in 'xy':
        d = M.base_filter(D[''][v])
        cells = []
        for i in range(4):
            for j in range(4):
                s = d[(d.xm >= ex[i]) & (d.xm < ex[i + 1]) & (d.ym >= ey[j]) & (d.ym < ey[j + 1])]
                r = M.region_fit(s, seed=500 + 4 * i + j)
                r['U'] = float(s.U.mean()); r['W'] = float(s.W.mean()); r['nev'] = int(s.eid.nunique())
                cells.append(r)
        grid[v] = cells
        Z = np.array([[c['U'], c['W']] for c in cells])
        for key in ('b', 'D', 'sl'):
            y = np.array([c['sl'] if key == 'sl' else M.derived(c['par'], c['zr'])[key] for c in cells])
            yb = np.array([c['slb'] if key == 'sl' else M.derived(c['parb'], c['zr'])[key] for c in cells])
            w = 1 / np.maximum(yb.std(1), 1e-9) ** 2
            A = np.vstack([np.ones(16), Z[:, 0], Z[:, 1]]).T; sw = np.sqrt(w)
            c0 = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
            cb = np.array([np.linalg.lstsq(A * sw[:, None], yb[:, i] * sw, rcond=None)[0] for i in range(yb.shape[1])])
            grid.setdefault('fit', {})[f'{v}|{key}'] = dict(icpt=float(c0[0]), gU=float(c0[1]), gW=float(c0[2]),
                                                          eU=float(cb[:, 1].std()), eW=float(cb[:, 2].std()), ei=float(cb[:, 0].std()),
                                                          chi2=float((((y - A @ c0) ** 2) * w).sum()))
    R['grid'] = grid['fit']
    pickle.dump(grid, open(f'{L.OUT}/grid_cells.pkl', 'wb'))
    json.dump(R, open(f'{L.OUT}/results.json', 'w'), indent=1, default=float)
    print('done')

if __name__ == '__main__':
    main()
