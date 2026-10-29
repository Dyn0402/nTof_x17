"""T16: why does a better-shaped model not buy angle resolution?  (Dylan 10-29: "odd that adding a degree of freedom
can't help, given the systematic centre under-fill / +-1 over-fill").

Three questions, on held-out unsaturated planes fitted as in T9 (reference start, t0 pinned, main's LM):
  1. Budget: how much of the chi2 near the peak is the MEAN shape error (what any model fix can remove) and how much
     is plane-to-plane scatter (ionisation fluctuations, delta rays, noise: no mean model removes it)?
     chi2_sys = sum (peak * P / noise)^2 with P the median fractional pattern; chi2_rest = sum ((r - peak * P) / noise)^2.
  2. Symmetry: oriented by the track's lean (+1 = the side the charge moves to with depth), is the pattern
     left-right symmetric?  A symmetric footprint error leaves the charge centroid of every depth slice, and so the
     angle, unchanged to first order; only the antisymmetric part can bias or blur the angle.
  3. Does the per-plane antisymmetric residual correlate with the per-plane angle error?
Usage:
  python -I t16_residual_budget.py run BUNDLE CACHE OUT.npz [jobs]     (condor node)
  python -I t16_residual_budget.py analyze OUT.npz [...]               -> prints, writes OUT.json beside each
"""
import json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, REPO)
O = np.arange(-2, 3)
K = np.arange(-4, 9)
_EV = None


def _init(cache, bundle):
    global _EV
    from wft.calib import CalibrationBundle
    from wft import model as wm
    _EV = pickle.load(open(cache, 'rb'))
    wm.use_calibration(CalibrationBundle.load(bundle)); wm.set_depth_bins(24)
    if wm.COL_GUARD <= 0:
        wm.COL_GUARD = 0.01


def one(eid):
    from wft import model as wm
    ev = _EV[eid]; cal = wm.CAL; out = []
    for vi, plane in enumerate('xy'):
        if plane not in ev:
            continue
        P = ev[plane]; W0 = np.asarray(P['W'])
        if W0.max() >= wm.SAT:
            continue
        if W0.shape[1] != wm.NSAMP:
            wm.set_nsamp(W0.shape[1])
        ft = ev.get(f'ftst_{plane}'); t0a = (cal.t0_abs or {}).get(plane, {})
        if ft is None or int(ft) not in t0a:
            continue
        t0p = float(t0a[int(ft)]); tr = float(ev[f'tan_{plane}']); p0r = float(ev[f'ref_mesh_{plane}'])
        try:
            r = wm.fit_plane_raw(P, plane, p0r, tr * cal.v_drift * 1e-3, t0p, t0_prior=(t0p, 0.05))
        except Exception:
            continue
        if not np.isfinite(r['chi2']):
            continue
        W, noise, pos, sat = wm.prep_plane(P, plane)
        M = wm.model_waveforms(plane, pos, r['p0'], r['w'], r['t0'], r['q'], wm.HYPER)
        c, j0 = np.unravel_index(np.argmax(W), W.shape)
        sgn = 1 if r['w'] * (pos[-1] - pos[0]) >= 0 else -1      # +o = the strip the charge reaches with depth
        R = np.full((len(O), len(K)), np.nan); N = np.full(len(O), np.nan)
        for io, o in enumerate(O):
            s = c + sgn * o
            if 0 <= s < len(W) and not sat[s].any():
                N[io] = noise[s]
                for ik, k in enumerate(K):
                    if 0 <= j0 + k < W.shape[1]:
                        R[io, ik] = W[s, j0 + k] - M[s, j0 + k]
        chi_all = float(np.sum(((W - M) / noise[:, None])[~sat] ** 2))
        out.append(dict(eid=int(eid), view=vi, amp=float(W[c, j0]), mpk=float(max(M.max(), 1.0)), tan=tr, w=float(r['w']),
                        chi2=float(r['chi2']), chi_all=chi_all, dof=int((~sat).sum()), R=R, N=N))
    return out


def run(bundle, cache, outp, jobs=8):
    ids = sorted(pickle.load(open(cache, 'rb')))[180:180 + int(os.environ.get('T16_N', 100000))]
    with ProcessPoolExecutor(int(jobs), initializer=_init, initargs=(cache, bundle)) as pool:
        rows = [r for rs in pool.map(one, ids, chunksize=4) for r in rs]
    np.savez(outp, eid=[r['eid'] for r in rows], view=[r['view'] for r in rows], amp=[r['amp'] for r in rows],
             mpk=[r['mpk'] for r in rows], tan=[r['tan'] for r in rows], w=[r['w'] for r in rows],
             chi2=[r['chi2'] for r in rows], chi_all=[r['chi_all'] for r in rows], dof=[r['dof'] for r in rows],
             R=np.array([r['R'] for r in rows]), N=np.array([r['N'] for r in rows]), bundle=bundle, cache=cache)
    print(outp, len(rows))


def analyze1(path, amin=800.0):
    d = np.load(path); out = {}
    for vi, v in enumerate('xy'):
        m = (d['view'] == vi) & (d['amp'] >= amin)
        R, N, pk, tr, w = d['R'][m], d['N'][m], d['mpk'][m], d['tan'][m], d['w'][m]
        F = R / pk[:, None, None]                                  # fractional residual
        P = np.nanmedian(F, axis=0)                                # the mean shape error (oriented)
        sc = 1.4826 * np.nanmedian(np.abs(F - P), axis=0)          # plane-to-plane scatter (robust)
        Z = R / N[:, :, None]
        sys_ = (pk[:, None, None] * P[None] / N[:, :, None])
        ok = np.isfinite(Z) & np.isfinite(sys_)
        chi_win = np.nansum(np.where(ok, Z ** 2, 0)); chi_sys = np.nansum(np.where(ok, sys_ ** 2, 0))
        chi_rest = np.nansum(np.where(ok, (Z - sys_) ** 2, 0))
        # symmetric / antisymmetric parts of the mean pattern (+o vs -o)
        sym = 0.5 * (P[[3, 4]] + P[[1, 0]]); anti = 0.5 * (P[[3, 4]] - P[[1, 0]])
        # per-plane antisymmetry at the +-1 strips around the peak (k = -1..+3) vs the angle error
        a1 = np.nanmean(F[:, 3, 3:8] - F[:, 1, 3:8], axis=1)
        inc = (np.abs(tr) > 0.1) & (np.abs(tr) < 0.45) & np.isfinite(a1)
        lad = np.median(w[inc] * 1e3 / tr[inc])
        derr = np.degrees(np.arctan(w * 1e3 / lad) - np.arctan(tr))
        k = inc & (np.abs(derr) < 5)
        corr = float(np.corrcoef(a1[k], np.abs(derr[k]))[0, 1]) if k.sum() > 30 else np.nan
        corr_signed = float(np.corrcoef(a1[k], derr[k] * np.sign(tr[k]))[0, 1]) if k.sum() > 30 else np.nan
        i0 = list(K).index(0)
        out[v] = dict(n=int(m.sum()), chi_sys_frac=float(chi_sys / chi_win), chi_rest_frac=float(chi_rest / chi_win),
                      chi_win_frac_of_all=float(chi_win / np.sum(d['chi_all'][m])),
                      centre_k0=float(P[2, i0]), pm1_k0=[float(P[1, i0]), float(P[3, i0])], pm2_k0=[float(P[0, i0]), float(P[4, i0])],
                      scatter_centre_k0=float(sc[2, i0]), scatter_pm1_k0=float(np.nanmean(sc[[1, 3], i0])),
                      sym_pm1_k0=float(sym[0, i0]), anti_pm1_k0=float(anti[0, i0]), anti_pm2_k0=float(anti[1, i0]),
                      anti_pm1_max=float(np.nanmax(np.abs(anti[0]))), corr_anti_abs_err=corr, corr_anti_signed_err=corr_signed,
                      pattern=P.tolist(), scatter=sc.tolist())
        q = out[v]
        print(f"{os.path.basename(path)} {v} n={q['n']}: window chi2 = {100*q['chi_win_frac_of_all']:.0f}% of plane chi2; "
              f"mean-shape part {100*q['chi_sys_frac']:.1f}%, scatter {100*q['chi_rest_frac']:.1f}% | centre {q['centre_k0']:+.3f} "
              f"(scatter {q['scatter_centre_k0']:.3f}) -1/+1 {q['pm1_k0'][0]:+.3f}/{q['pm1_k0'][1]:+.3f} (scatter {q['scatter_pm1_k0']:.3f}) "
              f"-2/+2 {q['pm2_k0'][0]:+.3f}/{q['pm2_k0'][1]:+.3f} | anti ±1 {q['anti_pm1_k0']:+.3f} (max over k {q['anti_pm1_max']:.3f}) "
              f"| corr(anti, |err|) {corr:+.2f}, (anti, signed err) {corr_signed:+.2f}")
    json.dump(out, open(path.replace('.npz', '.json'), 'w'), indent=1)
    return out


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run(*sys.argv[2:])
    else:
        for p in sys.argv[2:]:
            analyze1(p)
