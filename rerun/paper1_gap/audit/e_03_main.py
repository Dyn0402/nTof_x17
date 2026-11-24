"""e_03_main.py -- Task E: shape of the drift lines.  Library of the slice/grid analysis + the default run.
usage: e_03_main.py  -> $WORK/audit/e/main.pkl and main_summary.json"""
import sys, json, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import e_lib as L
GM, CO = L.GM, L.CO
G = L.GAP
B = 200
EDGES = np.arange(-60, 901, 60.)


def load(tag='', views='xy'):
    out = {}
    for v in views:
        ds = []
        for i, run in enumerate(('run1', 'run2')):
            d = pd.read_parquet(f'{L.OUT}/cen{tag}_{run}_{v}.parquet')
            d['eid'] = d.eid + 10_000_000 * (i + 1)
            ds.append(d)
        d = pd.concat(ds, ignore_index=True)
        d['U'] = (d.xm - 200) / 100; d['W'] = (d.ym - 200) / 100
        out[v] = d
    return out


def base_filter(d, sat='drop', tmin=None):
    m = ~d.drop_mask
    if sat == 'drop':
        m &= ~d.sat_ev
    elif sat == 'sample':
        m &= ~d.drop_sat
    return d[m]


def fitq(z, dl, w, g=G):
    """delta = a + b z + q z^2 -> (a, b, q) weighted least squares"""
    A = np.vstack([np.ones_like(z), z, z * z]).T
    sw = np.sqrt(w)
    c = np.linalg.lstsq(A * sw[:, None], dl * sw, rcond=None)[0]
    return c


def region_fit(d, edges=EDGES, tr=(0, 810), boot=B, seed=0, keep=None, bal=True):
    """profile + quadratic fit with bootstrap.  returns dict(par (3,), parb (B,3), z, delta, ...)"""
    if bal:
        d = d.assign(wt=L.balance(d))
    p = L.profile_region(d, edges, boot=boot, seed=seed, keep=keep)
    sel = (p['tmid'] >= tr[0]) & (p['tmid'] <= tr[1]) & (p['n'] > 40)
    w = 1 / np.maximum(p['ed'], 1e-3) ** 2
    res = dict(z=p['z'], delta=p['delta'], ed=p['ed'], ez=p['ez'], n=p['n'], tmid=p['tmid'], sel=sel)
    res['par'] = fitq(p['z'][sel], p['delta'][sel], w[sel])
    if boot:
        res['parb'] = np.array([fitq(p['zb'][i][sel], p['db'][i][sel], w[sel]) for i in range(boot)])
        res['zb'], res['db'] = p['zb'], p['db']
    res['zr'] = (float(p['z'][sel].min()), float(p['z'][sel].max()))
    # plain straight-line slope of delta(z) over the same range (no extrapolation): compare with w0/v
    res['sl'] = float(fitlin(p['z'][sel], p['delta'][sel], w[sel])[1])
    if boot:
        res['slb'] = np.array([fitlin(p['zb'][i][sel], p['db'][i][sel], w[sel])[1] for i in range(boot)])
    return res


def fitlin(z, dl, w):
    A = np.vstack([np.ones_like(z), z]).T; sw = np.sqrt(w)
    return np.linalg.lstsq(A * sw[:, None], dl * sw, rcond=None)[0]


def derived(par, zr):
    """b (z=0 slope), slope at far end of range, mean slope over the range, q, Delta = 2 G q (extrapolated to the gap)"""
    a, b, q = par[..., 0], par[..., 1], par[..., 2]
    z0, z1 = zr
    return dict(b=b, q=q, D=2 * G * q, s_far=b + 2 * q * z1, s_mean=b + q * (z0 + z1), s_full=b + q * G)


def slice_defs(view, axis, kind='x5'):
    """edges on U (axis 'U') or W."""
    ex, ey = GM.GRIDS[kind]
    return (ex, 'xm') if axis == 'U' else (ey, 'ym')


def slices_fit(d, axis, nsl=5, **kw):
    """per slice region_fit.  slices on U or W (all of the other coordinate)."""
    e = GM.GRIDS['x5'][0] if axis == 'U' else GM.GRIDS['y5'][1]
    col = 'xm' if axis == 'U' else 'ym'
    # every slice spans the full interior in the other coordinate (the interior cut is already in the table)
    res = []
    for i in range(len(e) - 1):
        s = d[(d[col] >= e[i]) & (d[col] < e[i + 1])]
        r = region_fit(s, seed=100 + i, **kw)
        r['coord'] = float((s[axis].mean())); r['nrows'] = len(s); r['nev'] = int(s.eid.nunique())
        res.append(r)
    return res


def wls_slope(x, y, w):
    A = np.vstack([np.ones_like(x), x]).T; sw = np.sqrt(w)
    c = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
    return c


def gradient(res, key):
    """gradient of a derived quantity across slices, with bootstrap error (slices independent)."""
    x = np.array([r['coord'] for r in res])
    if key == 'sl':
        y = np.array([r['sl'] for r in res]); yb = np.array([r['slb'] for r in res])
    else:
        y = np.array([derived(r['par'], r['zr'])[key] for r in res])
        yb = np.array([derived(r['parb'], r['zr'])[key] for r in res])            # (nsl, B)
    err = yb.std(1)
    w = 1 / np.maximum(err, 1e-9) ** 2
    c = wls_slope(x, y, w)
    cb = np.array([wls_slope(x, yb[:, i], w) for i in range(yb.shape[1])])
    chi2 = float((w * (y - c[0] - c[1] * x) ** 2).sum()); dof = len(x) - 2
    return dict(grad=float(c[1]), err=float(cb[:, 1].std()), chi2=chi2, dof=dof, boots=cb[:, 1], icpt_err=float(cb[:, 0].std()), icpt=float(c[0]), y=y.tolist(), yerr=err.tolist(), x=x.tolist())


def w0v_slices(tabs_by_run, axis, view, nsl=5):
    """w0/v per slice (pooled both runs; own robust line per run, then mean weighted) from the pass-1 table."""
    e = GM.GRIDS['x5'][0] if axis == 'U' else GM.GRIDS['y5'][1]
    col = 'xm' if axis == 'U' else 'ym'
    out = []
    for i in range(len(e) - 1):
        vals = []
        for run, t in tabs_by_run.items():
            s = t[(t[col] >= e[i]) & (t[col] < e[i + 1])]
            tr = s[f'tan_ref_{view}'].values; sel = np.abs(tr) <= GM.TAN_LADDER
            w0, v, w0e, ve, n = CO.line(tr[sel], s[f'{view}_w'].values[sel] * 1e3)
            vals.append((w0, v, w0e, ve))
        out.append(vals)
    return out


def main():
    D = load('')
    res = {}
    for v in 'xy':
        d = base_filter(D[v])
        for axis in 'UW':
            res[(v, axis)] = slices_fit(d, axis)
            print(v, axis, [round(r['par'][1], 4) for r in res[(v, axis)]], flush=True)
    pickle.dump(res, open(f'{L.OUT}/main_slices.pkl', 'wb'))
    return res


if __name__ == '__main__':
    main()
