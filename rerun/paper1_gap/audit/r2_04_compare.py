"""r2_04_compare.py -- predicted tilt (1/2 grad g from the n_TOF drift-time field) against the in-situ A-C cosmic-line tilt map, cell by cell.
 total tilt at n_TOF    T_n  = SGN a_poly(bundle) + a_insitu           (a_insitu = r2 in-situ map: residual beyond the pass-1 bow block)
 model                  T_n  = c0 + k SGN dh_p/dstrip(bench h block) + c SGN dlnT/dstrip      (c nominal = 0.5 G0/(1+beta) = 8.39 mm at beta 0.68)
Cells of the in-situ map (50 mm, n >= 80 lines).  Gradient averaged over a 5x5 sub-grid per cell.  Outputs compare.json."""
import sys, pickle, json
from r2_lib import *
from wft import reco as R
K = 10
FL = pickle.load(open(f'{OUT}/fields.pkl', 'rb'))
BUND = '/media/ucla/work/ntof_pass1/bench_bundles'
CEN = np.arange(-175.0, 176.0, 50.0)
NOM = 0.5 * G0 / (1 + BETA)
SP = Spl(K)
sub = (np.arange(5) - 2) * 10.0


def cell_geometry():
    """U, W strip-frame grids for the 8x8 cells ([ix, iy] = x_local, y_local), with 5x5 sub-grid"""
    X, Y = np.meshgrid(CEN, CEN, indexing='ij')
    xs = X[..., None, None] + sub[None, None, :, None] + 0 * sub[None, None, None, :]
    ys = Y[..., None, None] + 0 * sub[None, None, :, None] + sub[None, None, None, :]
    U = (-(xs) - 0.71) / 100.0
    W = (ys - 0.71) / 100.0
    return U, W


UG, WG = cell_geometry()


def grad_lnT(c):
    """d lnT / d strip coordinate (1/mm), per cell: returns (gx, gy) arrays [8,8] (cell-averaged)"""
    u, w = UG.ravel(), WG.ravel()
    gx = (SP.basis(u, w, 1, 0) @ c) / 100.0
    gy = (SP.basis(u, w, 0, 1) @ c) / 100.0
    shp = UG.shape
    return gx.reshape(shp).mean((2, 3)), gy.reshape(shp).mean((2, 3))


def bow_terms(arm):
    b = json.load(open(f'{BUND}/{arm}/bundle.json'))['bow']
    U, W = UG.mean((2, 3)), WG.mean((2, 3))
    out = {}
    for p in 'xy':
        apoly = R._bow_poly(b[f'a_{p}'], U, W)
        h = b[f'h_{p}']
        if p == 'x':
            gh = (h['gu'] + h['kuu'] * U + h['kuw'] * W) / 100.0
        else:
            gh = (h['gw'] + h['kww'] * W + h['kuw'] * U) / 100.0
        out[p] = dict(apoly=apoly, gh=gh)
    return out


def ntof_map(arm, view):
    m = json.load(open('/home/dylan/PycharmProjects/nTof_x17/rerun/ntof/stage3/angle_map/angle_map_r2.json'))['maps'][f'{arm}_{view}']
    return {k: np.asarray(m[k], float) for k in ('a', 's', 'n')}


def wls(X, y, w):
    sw = np.sqrt(w)
    c, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return c


def pear(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def analyse(arm, view, fkey, rng, nb=400):
    """returns dict for one chamber/view/field-sample"""
    sg = SGN[(arm, view)]
    i = 0 if view == 'x' else 1
    nt = ntof_map(arm, view)
    good = (nt['n'] >= 80)
    bw = bow_terms(arm)[view]
    a_in = nt['a']
    Tn = sg * bw['apoly'] + a_in
    H = sg * bw['gh']
    f = FL[fkey]
    G = sg * NOM * grad_lnT(f['c'])[i]          # tan, nominal c
    wts = nt['n'].astype(float)
    idx = np.where(good.ravel())[0]
    Tn_, H_, G_, a_, ap_, w_ = (z.ravel()[idx] for z in (Tn, H, G, a_in, sg * bw['apoly'], wts))
    out = dict(arm=arm, view=view, field=str(fkey), ncell=len(idx), sd_Tn=float(Tn_.std()), sd_ain=float(a_.std()),
               sd_G=float(G_.std()), sd_H=float(H_.std()), r_ain_G=pear(a_, G_), r_Tn_G=pear(Tn_, G_), r_Tn_H=pear(Tn_, H_),
               r_apoly_H=pear(ap_, H_))
    # (a) total tilt: Tn = c0 + k H + b G'  (G' = G nominal, so b' = c/NOM, expected 1 when b=0.5)
    def fit_tot(ix):
        X = np.c_[np.ones(len(ix)), H_[ix], G_[ix]]
        return wls(X, Tn_[ix], w_[ix])
    c_all = fit_tot(np.arange(len(idx)))
    out['tot_k'], out['tot_bratio'] = float(c_all[1]), float(c_all[2])
    # G only (k = 0 and k = 1 fixed)
    def bfix(kfix, ix):
        y = Tn_[ix] - kfix * H_[ix]
        X = np.c_[np.ones(len(ix)), G_[ix]]
        return wls(X, y, w_[ix])[1]
    allix = np.arange(len(idx))
    out['b_k0'], out['b_k1'] = float(bfix(0, allix)), float(bfix(1, allix))
    # (b) residual map: a_in = c0 + (k H - apoly) + b' G
    def fit_res(ix, kfix=None):
        if kfix is None:
            X = np.c_[np.ones(len(ix)), H_[ix], G_[ix]]
            return wls(X, a_[ix] + ap_[ix], w_[ix])
        X = np.c_[np.ones(len(ix)), G_[ix]]
        return wls(X, a_[ix] + ap_[ix] - kfix * H_[ix], w_[ix])
    # explained variance of a_in (residual) with k = 0, 1, fit  (b' fitted or nominal 1)
    for lab, kf in (('k0', 0.0), ('k1', 1.0)):
        c = fit_res(allix, kf)
        pred = c[0] + kf * H_ + c[1] * G_ - ap_
        out[f'res_{lab}_bratio'] = float(c[1])
        out[f'res_{lab}_r'] = pear(a_, pred)
        out[f'res_{lab}_frac'] = float(1 - np.sum(w_ * (a_ - pred - np.average(a_ - pred, weights=w_)) ** 2) / np.sum(w_ * (a_ - np.average(a_, weights=w_)) ** 2))
        # nominal b' = 1
        pn = kf * H_ + G_ - ap_
        out[f'res_{lab}_nominal_r'] = pear(a_, pn)
        out[f'res_{lab}_nominal_frac'] = float(1 - np.sum(w_ * (a_ - pn - np.average(a_ - pn, weights=w_)) ** 2) / np.sum(w_ * (a_ - np.average(a_, weights=w_)) ** 2))
        out[f'res_{lab}_nominal_slope'] = float(wls(np.c_[np.ones(len(a_)), pn], a_, w_)[1])
    c = fit_res(allix)
    out['res_fit_k'], out['res_fit_bratio'] = float(c[1]), float(c[2])
    pred = c[0] + c[1] * H_ + c[2] * G_ - ap_
    out['res_fit_r'] = pear(a_, pred)
    # bootstrap over cells and over the field replicates
    cb = FL[(fkey[0], fkey[1], 'boot')]['cb']
    bs = {k: [] for k in ('tot_k', 'tot_bratio', 'b_k0', 'b_k1', 'res_fit_k', 'res_fit_bratio', 'r_ain_G', 'res_k1_nominal_r', 'res_k0_nominal_r')}
    for b in range(nb):
        ix = rng.integers(0, len(idx), len(idx))
        rep = rng.integers(0, len(cb))
        Gb = sg * NOM * grad_lnT(cb[rep])[i].ravel()[idx]
        X = np.c_[np.ones(len(ix)), H_[ix], Gb[ix]]
        c1 = wls(X, Tn_[ix], w_[ix]); bs['tot_k'].append(c1[1]); bs['tot_bratio'].append(c1[2])
        for lab, kf in (('b_k0', 0.0), ('b_k1', 1.0)):
            X2 = np.c_[np.ones(len(ix)), Gb[ix]]
            bs[lab].append(wls(X2, Tn_[ix] - kf * H_[ix], w_[ix])[1])
        c2 = wls(X, a_[ix] + ap_[ix], w_[ix]); bs['res_fit_k'].append(c2[1]); bs['res_fit_bratio'].append(c2[2])
        bs['r_ain_G'].append(pear(a_[ix], Gb[ix]))
        bs['res_k1_nominal_r'].append(pear(a_[ix], H_[ix] + Gb[ix] - ap_[ix]))
        bs['res_k0_nominal_r'].append(pear(a_[ix], Gb[ix] - ap_[ix]))
    for k, v in bs.items():
        out[k + '_err'] = float(np.nanstd(v))
    out['_arrays'] = dict(a_in=np.where(good, a_in, np.nan), Tn=np.where(good, Tn, np.nan), H=np.where(good, H, np.nan),
                          G=np.where(good, G, np.nan), apoly=np.where(good, sg * bw['apoly'], np.nan))
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(3)
    R_ = {}
    for arm in 'AC':
        for view in 'xy':
            for w in ('cosmic', 'late'):
                key = (arm, w, 'all')
                o = analyse(arm, view, key, rng)
                R_[f'{arm}_{view}_{w}'] = o
                print(arm, view, w, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items() if k != '_arrays'}, flush=True)
    pickle.dump(R_, open(f'{OUT}/compare.pkl', 'wb'))
    json.dump({k: {a: b for a, b in v.items() if a != '_arrays'} for k, v in R_.items()}, open(f'{OUT}/compare.json', 'w'), indent=1)
