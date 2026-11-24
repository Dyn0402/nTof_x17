"""W3_g4_scale library: the T5 / W2b pointing-scale estimator applied to Geant4 capsule electrons.

Frames (fitted on the HitTree, extract_origin.py f*.parquet, residual < 1e-13 mm):
  arm 2 = A: gx = u - 16.35, gz = w + 204.5     horizontal h = gx, normal n = gz, dh/dn =  tan_u
  arm 3 = C: gx = -u + 17.3, gz = -w - 204.5    h = gx, n = gz, dh/dn = tan_u
  arm 0 = D: gx = w + 204.0, gz = -u + 15.5     h = gz, n = gx, dh/dn = -tan_u
  gy = v in every arm.  Mesh plane w = 30.1 (DriftGas 0.1-30.1, w = 0 faces the source).
The data definition (T5 pm.arm_arrays): A/C e = (p0_x - xs) / (p0_z - zs), D e = (p0_z - zs) / (p0_x - xs);
the scale is q2lib.scale: windowed-median residual (measured - e) in 0.04 bins of e over 0.18 <= |e| <= 0.44, a weighted
line in e, k = 1 + slope.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
G4A = Path('/media/dylan/data/x17/ntof_cosmics/g4_angle')
DIGI = Path('/media/ucla/work/ntof_pass1/g4_digi')
W_MESH = 30.1
FR = {  # arm id: (h0, h_sign_u, n0, n_sign_w, slope sign on tan_u)
    'A': dict(id=2, h=(-16.35, 1.0), n=(204.5, 1.0), s=1.0),
    'C': dict(id=3, h=(17.3, -1.0), n=(-204.5, -1.0), s=1.0),
    'D': dict(id=0, h=(15.5, -1.0), n=(204.0, 1.0), s=-1.0),
}
ID2ARM = {v['id']: k for k, v in FR.items()}


def to_global(arm, u, w, tan_u):
    f = FR[arm]
    h = f['h'][0] + f['h'][1] * np.asarray(u, float)
    n = f['n'][0] + f['n'][1] * np.asarray(w, float)
    return h, n, f['s'] * np.asarray(tan_u, float)


def origin_hn(arm, ox, oz):
    """global origin -> (h, n) of this arm's view"""
    return (np.asarray(ox), np.asarray(oz)) if arm in 'AC' else (np.asarray(oz), np.asarray(ox))


# ---------------------------------------------------------------- estimator (copy of pm.centre / q2lib.scale)
def wmedian(r, w=0.06, n_iter=8, m0=None, nmin=30):
    r = r[np.isfinite(r)]
    if len(r) < nmin:
        return np.nan
    m = np.median(r[np.abs(r) < 0.3]) if m0 is None else m0
    for _ in range(n_iter):
        k = np.abs(r - m) < w
        if k.sum() < 20:
            return np.nan
        m2 = np.median(r[k])
        if abs(m2 - m) < 1e-6:
            break
        m = m2
    return float(m)


def scale(e, meas, emin=0.18, emax=0.44, bw=0.04, nmin=40, w=0.06, return_bins=False):
    """k = 1 + slope of the binned windowed-median residual, unweighted per bin (errors come from the bootstrap)."""
    e, meas = np.asarray(e, float), np.asarray(meas, float)
    EB = np.r_[np.arange(-emax, -emin + 1e-9, bw), np.arange(emin, emax + 1e-9, bw)]
    xs, cs, ns = [], [], []
    for lo, hi in zip(EB[:-1], EB[1:]):
        if lo < 0 < hi:
            continue
        m = (e >= lo) & (e < hi)
        if m.sum() < nmin:
            continue
        c = wmedian(meas[m] - e[m], w)
        if np.isfinite(c):
            xs.append(e[m].mean()); cs.append(c); ns.append(m.sum())
    if len(xs) < 4:
        return (np.nan, np.nan) if not return_bins else (np.nan, np.nan, None)
    xs, cs, ns = map(np.array, (xs, cs, ns))
    X = np.vstack([np.ones_like(xs), xs]).T
    wt = ns.astype(float)            # ~1/err^2 for a median
    beta = np.linalg.solve(X.T @ (X * wt[:, None]), X.T @ (wt * cs))
    if return_bins:
        return 1 + beta[1], beta[0], pd.DataFrame(dict(e=xs, c=cs, n=ns))
    return 1 + beta[1], beta[0]


def boot_scale(e, meas, nboot=200, seed=1, **kw):
    k, a = scale(e, meas, **kw)
    rng = np.random.default_rng(seed)
    n = len(e)
    reps = []
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        reps.append(scale(e[i], meas[i], **kw)[0])
    reps = np.array(reps)
    return k, float(np.nanstd(reps, ddof=1)), a


# ---------------------------------------------------------------- the band crossing (capsule_tof.crossing)
def crossing(pos, disp, n_iter=4, clip=3.0, window=100.0):
    pos, disp = np.asarray(pos, float), np.asarray(disp, float)

    def fit(sel):
        keep = sel.copy()
        for _ in range(n_iter):
            a, b = np.polyfit(pos[keep], disp[keep], 1)
            r = disp - (a * pos + b)
            s = 1.4826 * np.median(np.abs(r[keep]))
            keep = sel & (np.abs(r) < clip * s)
        a, b = np.polyfit(pos[keep], disp[keep], 1)
        x0 = -b / a
        r = disp[keep] - (a * pos[keep] + b)
        err = np.std(r) / abs(a) / np.sqrt(keep.sum()) * np.sqrt(1 + (x0 - pos[keep].mean()) ** 2
                                                                    / max(pos[keep].var(), 1e-9))
        return float(x0), float(err)
    x0, err = fit(np.ones(len(pos), bool))
    for _ in range(2):
        x0, err = fit(np.abs(pos - x0) < window)
    return x0, err


# ---------------------------------------------------------------- samples
def load_reduced(lib='nose'):
    out = []
    for f in sorted(glob.glob(str(G4A / f'neutrons_{lib}' / 'r*.parquet'))):
        r = pd.read_parquet(f, columns=['eventID', 'arm', 'w_lo', 'w_hi', 'tan_gap_u', 'tan_gap_v', 'u_mesh', 'v_mesh',
                                        'dom_track', 'dom_particle', 'dom_ke_gap', 'dom_share', 'wall_same_track',
                                        'tan_dom_u', 'u_mesh_dom'])
        r['job'] = int(Path(f).stem[1:])
        out.append(r)
    R = pd.concat(out, ignore_index=True)
    R['wall'] = R.wall_same_track.astype(object).fillna(False).astype(bool)
    R = R[R.arm.isin([0, 2, 3])].copy()
    R['A'] = R.arm.map(ID2ARM)
    R['full'] = (R.w_hi - R.w_lo) > 20
    return R


def add_global(R, tan='tan_gap_u', u='u_mesh', w=W_MESH, out=('h', 'n', 'sh')):
    h = np.full(len(R), np.nan); n = h.copy(); s = h.copy()
    for arm in 'ACD':
        m = (R.A == arm).to_numpy()
        h[m], n[m], s[m] = to_global(arm, R.loc[m, u], np.full(m.sum(), w), R.loc[m, tan])
    R[out[0]], R[out[1]], R[out[2]] = h, n, s
    return R


def source_from_crossings(R, h='h', n='n', sh='sh', arms_x='AC', arm_z='D'):
    """xs from the A/C crossings (mean of the two, as the data's single XS), zs from D."""
    res = {}
    for arm in 'ACD':
        g = R[R.A == arm]
        pos = g[h].to_numpy(); disp = -g[sh].to_numpy() * g[n].to_numpy()
        res[arm] = crossing(pos, disp)
    xs = np.mean([res[a][0] for a in arms_x])
    zs = res[arm_z][0]
    return xs, zs, res


def expected(R, arm, hs_ns, h='h', n='n'):
    xs, zs = hs_ns
    hs, ns = (xs, zs) if arm in 'AC' else (zs, xs)
    return (R[h].to_numpy() - hs) / (R[n].to_numpy() - ns)
