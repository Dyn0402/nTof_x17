#!/usr/bin/env python3
"""boot_run.py -- the model-free residual bootstrap of the clean-event angle remainder (BOOTSTRAP_2026-11-01.md).

    boot_run.py toy  --det det3 --angle 9 --arms T,N,D,... --data D.pkl [--n 600] --out OUT.npz   (FD_* env as ladder.py)
    boot_run.py data --det det3 --arms dD,dN,dDtshuf --data D.pkl [--n 800] --out OUT.npz

A plane's fit residual R = W - M (boot_extract.py: the gain-corrected window minus the pass model on it) carries
everything the model does not describe on that plane: noise, the ionisation fluctuations about the smooth model, and
whatever structured misfit there is.  Here each truth plane is the pass model rendered alone (no noise, exactly
representable by the fit), a residual from ANOTHER plane is added to it, and the result is fitted with the unchanged
production wft.reco.fit_plane.  The angle error is the refit's against the truth model's own angle:
    d = atan(w_fit / v) - atan(w_truth / v)   [deg; the constant mapping, w0 and kw cancel to first order]

Truths
  toy   the FD toy's full L8 events (FD_BASIS=pass: pass-1 bundles, interior placement, the row's q50 / v(t)), fitted by
        the production fit; the truth is that fit's model.  One angle point per job (3 / 9 / 16 deg).
  data  the clean data planes' own pass models.
Residual sources (arms)
  L8    the toy events themselves (the direct toy: d = the toy fit against the generated angle)
  T     toy residuals (another toy event of the same point): the method's calibration, eps = s68(T) / s68(L8)
  N     Gaussian noise at the window's strip noise only
  D     data residuals (a clean data plane matched in angle bin and model peak)
  D<x>  data residuals with one structure removed (computed on the residual's own plane, before it is moved):
        Dcm      the chip-coherent baseline: per-sample median of the residual over the window's quiet strips
                 (model peak < 10 % of the plane's)
        Dmean    the mean shape error: peak x the pool's median fractional residual pattern (strip offset from the
                 peak strip, oriented by the lean; sample offset from the peak sample)
        Dtshape  a per-plane time-shape mismatch: the weighted LSQ projection of R on {M, dM/dt, (t - t_pk) dM/dt}
        Danti    the antisymmetric part about the peak strip, (R[c+o] - R[c-o]) / 2 for o = 1..3 (the +-1 strip
                 asymmetry, xasym-like)
        Dtshuf   samples permuted within each strip (time coherence destroyed, per-strip amplitude kept)
        Dsshuf   strips permuted within each sample (strip coherence destroyed)
  dD / dN / dDtshuf   the same on data truths.
Moving a residual: oriented by the lean (strips reversed when the two planes lean opposite ways), aligned on the peak
strip and the peak sample (integer samples; the donor is the nearest in t0 phase among up to 15 matched planes),
in ADC (not rescaled); strips or samples it does not cover, and its censored strips, are filled with Gaussian noise at
the truth window's noise.
"""
import argparse
import os
import pickle
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
FD = os.path.join(REPO, 'rerun', 'model_audit', 'followups', 'fd_floor_realism')
sys.path[:0] = [REPO, FD]

BINS = {3: (0.0, 6.0), 9: (6.0, 12.5), 16: (12.5, 25.0)}
DT = 60.0
O_GRID = np.arange(-12, 13)
K_GRID = np.arange(-12, 25)


# ------------------------------------------------------------------ a plane, its residual and its geometry
def plane_rec(W, M, noise, censor, gain, pos, ch, w, t0, v, t0_prior=None, raw_noise=None, view='x'):
    W, M = np.asarray(W, float), np.asarray(M, float)
    R = W - M
    c, k = np.unravel_index(int(np.argmax(M)), M.shape)
    lean = 1 if w * (float(pos[-1]) - float(pos[0])) >= 0 else -1
    return dict(R=R, M=M, noise=np.asarray(noise, float), censor=np.asarray(censor, bool).any(axis=1)
                if np.ndim(censor) == 2 else np.asarray(censor, bool), gain=np.asarray(gain, float),
                pos=np.asarray(pos, float), ch=np.asarray(ch, int), w=float(w), t0=float(t0), c=int(c), k=int(k),
                pk=float(M.max()), lean=lean, theta=float(np.degrees(np.arctan(w * 1e3 / v))),
                phase=float(t0 % DT), t0_prior=t0_prior, raw_noise=raw_noise, view=view)


def oriented(rec, R=None):
    """the residual (or R) with strips in lean order, its peak strip index and censor rows in that order."""
    R = rec['R'] if R is None else R
    if rec['lean'] > 0:
        return R, rec['c'], rec['censor']
    return R[::-1], R.shape[0] - 1 - rec['c'], rec['censor'][::-1]


def pattern(pool):
    """the pool's median fractional residual pattern on (O_GRID, K_GRID), oriented."""
    acc = [[[] for _ in K_GRID] for _ in O_GRID]
    for r in pool:
        R, c, cen = oriented(r)
        for io, o in enumerate(O_GRID):
            s = c + o
            if 0 <= s < R.shape[0] and not cen[s]:
                for ik, kk in enumerate(K_GRID):
                    t = r['k'] + kk
                    if 0 <= t < R.shape[1]:
                        acc[io][ik].append(R[s, t] / r['pk'])
    return np.array([[np.median(a) if len(a) >= 20 else 0.0 for a in row] for row in acc])


def ablate(rec, how, P=None, rng=None):
    """the donor residual with one structure removed, in the donor's own frame (strip order of its window)."""
    R, M, nz, cen = rec['R'].copy(), rec['M'], rec['noise'], rec['censor']
    live = ~cen
    if how == 'cm':
        quiet = live & (M.max(axis=1) < 0.1 * rec['pk'])
        use = quiet if quiet.sum() >= 3 else live
        R -= np.median(R[use], axis=0)[None, :]
    elif how == 'mean':
        sgn = rec['lean']
        for io, o in enumerate(O_GRID):
            s = rec['c'] + sgn * o
            if 0 <= s < R.shape[0]:
                for ik, kk in enumerate(K_GRID):
                    t = rec['k'] + kk
                    if 0 <= t < R.shape[1]:
                        R[s, t] -= rec['pk'] * P[io, ik]
    elif how == 'tshape':
        t = np.arange(M.shape[1]) * DT
        dM = np.gradient(M, DT, axis=1)
        B = np.stack([M, dM, (t - rec['k'] * DT)[None, :] * dM], axis=-1)[live]      # (S, T, 3)
        wgt = 1.0 / nz[live][:, None]
        A = (B * wgt[..., None]).reshape(-1, 3)
        b = (R[live] * wgt).reshape(-1)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        R[live] -= B @ coef
    elif how == 'anti':
        c = rec['c']
        for o in (1, 2, 3):
            a, b = c + o, c - o
            if 0 <= b and a < R.shape[0] and live[a] and live[b]:
                anti = 0.5 * (R[a] - R[b])
                R[a] -= anti
                R[b] += anti
    elif how == 'tshuf':
        for s in np.flatnonzero(live):
            R[s] = R[s, rng.permutation(R.shape[1])]
    elif how == 'sshuf':
        idx = np.flatnonzero(live)
        for t in range(R.shape[1]):
            R[idx, t] = R[rng.permutation(idx), t]
    return R


def transplant(truth, donor, Rd, rng):
    """donor residual Rd (donor frame) placed on the truth window: lean-oriented, peak strip and sample aligned."""
    S, T = truth['M'].shape
    out = rng.normal(0.0, 1.0, (S, T)) * truth['noise'][:, None]
    Ro, cd, cend = oriented(donor, Rd)
    dk = truth['k'] - donor['k']
    for s in range(S):
        so = s if truth['lean'] > 0 else S - 1 - s                 # truth strip in lean order
        ct = truth['c'] if truth['lean'] > 0 else S - 1 - truth['c']
        src = so - ct + cd
        if 0 <= src < Ro.shape[0] and not cend[src]:
            t_lo, t_hi = max(0, dk), min(T, Ro.shape[1] + dk)
            if t_hi > t_lo:
                out[s, t_lo:t_hi] = Ro[src, t_lo - dk:t_hi - dk]
    return out


def pick(truth, pool, rng, self_id=None):
    cand = [j for j, r in enumerate(pool) if j != self_id and 0.75 <= r['pk'] / truth['pk'] <= 1.333]
    if len(cand) < 20:
        cand = [j for j in np.argsort([abs(np.log(r['pk'] / truth['pk'])) for r in pool]) if j != self_id][:20]
    sub = rng.choice(cand, size=min(15, len(cand)), replace=False)
    ph = [min(abs(pool[j]['phase'] - truth['phase']), DT - abs(pool[j]['phase'] - truth['phase'])) for j in sub]
    return int(sub[int(np.argmin(ph))])


def refit(truth, R, view, cal, sigma_prior):
    from wft import reco
    Wr = (truth['M'] + R) * truth['gain'][:, None]
    P = dict(W=Wr, pos=truth['pos'], noise=truth['raw_noise'], ch=truth['ch'])
    tp = (truth['t0'], sigma_prior) if sigma_prior is not None else truth['t0_prior']
    try:
        f = reco.fit_plane(P, view, cal, t0_prior=tp)
    except Exception:
        return np.nan, np.nan
    if f is None:
        return np.nan, np.nan
    v = float(cal.v_drift)
    return (float(np.degrees(np.arctan(f.w * 1e3 / v)) - np.degrees(np.arctan(truth['w'] * 1e3 / v))),
            float(f.chi2 / max(f.dof, 1)))


# ------------------------------------------------------------------ data pools
def load_data(path, view, v):
    D = pickle.load(open(path, 'rb'))
    out = []
    for r in D['planes']:
        if r['view'] != view or r['w_table'] is None or not np.isfinite(r['w_table']) or abs(r['w'] - r['w_table']) > 1e-4:
            continue
        rr = plane_rec(r['W'], r['M'], r['noise'], r['censor'], r['gain'], r['pos'], r['ch'], r['w'], r['t0'], v,
                       t0_prior=r['t0_prior'], raw_noise=np.asarray(r['noise'], float) * np.asarray(r['gain'], float),
                       view=view)
        rr['event_id'] = r['event_id']
        out.append(rr)
    return out, D['meta']


ARM_RES = dict(Tcm='cm', Tanti='anti', Ttshuf='tshuf', Tsshuf='sshuf', Ttshape='tshape', D=None, Dcm='cm', Dmean='mean', Dtshape='tshape', Danti='anti', Dtshuf='tshuf', Dsshuf='sshuf',
               dD=None, dDtshuf='tshuf', dDcm='cm', dDanti='anti', dDtshape='tshape', dDmean='mean')


def run_arms(truths, arms, dpool, tpool, P_bin, view, cal, sigma_prior, rng, self_pool=None):
    res = {}
    for arm in arms:
        d, cd = [], []
        for i, tr in enumerate(truths):
            if arm in ('N', 'dN'):
                R = rng.normal(0.0, 1.0, tr['M'].shape) * tr['noise'][:, None]
            elif arm == 'S':                       # identity: the truth's own residual, not moved -> d = 0
                R = transplant(tr, tr, tr['R'], rng)
            elif arm.startswith('T'):              # toy residuals (T), or with one structure removed (T<x>)
                j = pick(tr, tpool, rng, self_id=i)
                how = ARM_RES.get(arm)
                Rd = tpool[j]['R'] if how is None else ablate(tpool[j], how, rng=rng)
                R = transplant(tr, tpool[j], Rd, rng)
            else:
                j = pick(tr, dpool, rng, self_id=i if self_pool else None)
                how = ARM_RES[arm]
                Rd = dpool[j]['R'] if how is None else ablate(dpool[j], how, P=P_bin, rng=rng)
                R = transplant(tr, dpool[j], Rd, rng)
            a, c = refit(tr, R, view, cal, sigma_prior)
            d.append(a)
            cd.append(c)
        res[arm] = np.array([d, cd], float)
    return res


def cmd_toy(a):
    import ladder as L
    from wft import reco, model as wm
    view = L.VIEW
    L._init(a.det)
    cal = L._CAL
    v = float(cal.v_drift)
    flags = L.flags_for('L8')
    g, h = L.ft.load_template(a.det)
    pulse_c, _, _ = L.pulse_calib(a.det)
    tmpl_gen = (g, L.distorted_template(g, h, pulse_c))
    adc = L.ft.adc_scale(L.ft.params(a.det, 'full'), np.random.default_rng(12345))
    tan = np.tan(np.radians(a.angle))
    rng = np.random.default_rng(1000 * a.angle + 17)
    tp0 = reco.t0_prior_for(cal, view, 0)
    sig = tp0[1] if tp0 is not None else None
    truths, l8 = [], []
    while len(truths) < a.n:
        ev = L.gen_event(a.det, tan, flags, adc, tmpl_gen, rng, L._STATE)
        if ev is None:
            continue
        win, tr = ev
        f = reco.fit_plane(win, view, cal, t0_prior=(tr['t0'], sig) if sig is not None else None)
        if f is None:
            l8.append(np.nan)
            continue
        l8.append(float(np.degrees(np.arctan(f.w * 1e3 / v))) - a.angle)
        Wg, nz, ps, st = wm.prep_plane(win, view)
        M = wm.model_waveforms(view, ps, f.p0, f.w, f.t0, f._q, cal.hyper)
        ch = np.asarray(win['ch'], int)
        truths.append(plane_rec(Wg, M, nz, st, wm.GAIN[view][ch], ps, ch, f.w, f.t0, v,
                                raw_noise=np.asarray(win['noise'], float), view=view))
    print('toy', a.det, view, a.angle, len(truths), 'L8 s68', np.nanpercentile(np.abs(np.array(l8) - np.nanmedian(l8)), 68),
          flush=True)
    lo, hi = BINS[a.angle]
    dpool = []
    if any(x.startswith('D') for x in a.arms.split(',')):
        dall, _meta = load_data(a.data, view, v)
        dpool = [r for r in dall if lo <= abs(r['theta']) < hi]
        print('data pool', len(dpool), 'of', len(dall), flush=True)
    P_bin = pattern(dpool) if 'Dmean' in a.arms.split(',') else None
    res = run_arms(truths, a.arms.split(','), dpool, truths, P_bin, view, cal, sig, rng)
    for k, r in res.items():
        print(k, 's68', np.nanpercentile(np.abs(r[0] - np.nanmedian(r[0])), 68), flush=True)
    np.savez(a.out, L8=np.array(l8, float), angle=a.angle, det=a.det, view=view, n_pool=len(dpool),
             **{k: v_ for k, v_ in res.items()})
    print('wrote', a.out)


def cmd_data(a):
    from wft.calib import CalibrationBundle
    from wft import model as wm
    view = os.environ.get('FD_VIEW', 'x')
    cal = CalibrationBundle.load(a.bundle)
    wm.use_calibration(cal)
    v = float(cal.v_drift)
    dall, _ = load_data(a.data, view, v)
    rng = np.random.default_rng(31)
    idx = rng.choice(len(dall), size=min(a.n, len(dall)), replace=False)
    arms = a.arms.split(',')
    pools = {b: [j for j, r in enumerate(dall) if lo <= abs(r['theta']) < hi] for b, (lo, hi) in BINS.items()}
    pats = {b: pattern([dall[j] for j in pools[b]]) for b in BINS} if any(x.endswith('mean') for x in arms) else {}
    # donors: the truth's own angle bin, the truth plane itself excluded
    res = {arm: [[], []] for arm in arms}
    th = []
    for ti in idx:
        tr = dall[ti]
        b = [k for k, (lo, hi) in BINS.items() if lo <= abs(tr['theta']) < hi]
        if not b:
            continue
        th.append(tr['theta'])
        pool = [dall[j] for j in pools[b[0]] if j != ti]
        r1 = run_arms([tr], arms, pool, pool, pats.get(b[0]), view, cal, None, rng)
        for arm in arms:
            res[arm][0].append(r1[arm][0, 0])
            res[arm][1].append(r1[arm][1, 0])
    np.savez(a.out, theta=np.array(th), det=a.det, view=view,
             **{k: np.array(v_, float) for k, v_ in res.items()})
    print('wrote', a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('toy', 'data'))
    ap.add_argument('--det', required=True)
    ap.add_argument('--angle', type=int, default=9)
    ap.add_argument('--arms', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--bundle', default=None)
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    {'toy': cmd_toy, 'data': cmd_data}[a.cmd](a)


if __name__ == '__main__':
    main()
