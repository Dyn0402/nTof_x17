#!/usr/bin/env python3
"""analyse_headon.py [--dir DIR] -- sigma0 / Dp / D_rc, ZS bias, time and position resolution from headon_sigma.py output.

Reads <dir>/headon_<tag>.pkl (collected) for the RAW plateaus and the ZS / no-censor arms, writes
<dir>/headon_summary.json and figures into rerun/model_audit/ws_sps/figures/.

  * summed-chi2 profiles in sigma0 and Dp (parabola through the three lowest grid points), errors from an event
    bootstrap (events are the independent unit; the chi2 itself is model-limited, chi2/dof ~ 3-5)
  * sigma0 in amplitude terciles at the global (Dp, D_rc): does the prompt footprint grow with charge (F18)?
  * paired ZS / no-censor shifts on the same events
  * x-y t0 difference (one particle, one FEU: per-plane t0 resolution with no trigger term)
(No position resolution here: the run_71 caches' ref_mesh is det4's own leading strip, not the uRWELL; see
resolution.py for the referenced run_63 measurement.)
"""
import argparse, json, os, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
E_VCM = {'raw700': 243.2, 'raw450': 156.3, 'raw275': 95.5}
# Magboltz, run_71 beam gas (Ar/CF4/iso 88/10/2 + 1.7 % H2O + 0.1 % O2; cloud_basics results/magboltz_beam_w1p7_o0p1.json):
# D_T [um/sqrt(cm)], v [um/ns] at the three fields -> Dp = D_T * sqrt(v) [mm/sqrt(ns)]
MB = {'raw700': (159.1, 12.76), 'raw450': (187.0, 7.6), 'raw275': (233.0, 4.6)}


def dp_magboltz(tag):
    dt, v = MB[tag]
    return dt * 1e-3 / np.sqrt(10.0) * np.sqrt(v * 1e-3)


def parab_min(x, y):
    i = int(np.argmin(y))
    if i == 0 or i == len(y) - 1:
        return float(x[i])
    a, b, c = np.polyfit(x[i - 1:i + 2], y[i - 1:i + 2], 2)
    return float(-b / (2 * a)) if a > 0 else float(x[i])


def best(G, S0, DP, DRC):
    """sigma0, Dp, D_rc at the summed-chi2 minimum (parabolic in each axis through the grid minimum)."""
    S = G.sum(0)
    i, j, k = np.unravel_index(np.argmin(S), S.shape)
    s0 = parab_min(S0, S[:, j, k]); dp = parab_min(DP, S[i, :, k])
    drc = parab_min(DRC, S[i, j, :]) if S.shape[2] > 1 else 0.0
    return s0, dp, drc, (i, j, k), S


def load(d, tag):
    p = os.path.join(d, f'headon_{tag}.pkl')
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else None


def view_rows(D, v):
    return {r['eid']: r[v] for r in D['rows'] if v in r}


def analyse(D, v, nboot=300):
    S0, DP = D['S0'], D['DP']; DRC = D['DRC'] if v == 'y' else np.array([0.0])
    R = view_rows(D, v); eids = np.array(sorted(R))
    G = np.stack([R[e]['G'] for e in eids]); dof = np.array([R[e]['dof'] for e in eids])
    s0, dp, drc, idx, S = best(G, S0, DP, DRC)
    rng = np.random.default_rng(7); bs = []
    for _ in range(nboot):
        b = best(G[rng.integers(0, len(G), len(G))], S0, DP, DRC)
        bs.append(b[:3])
    bs = np.array(bs)
    amax = np.array([R[e]['amax'] for e in eids])
    terc = np.percentile(amax, [33.3, 66.7]); tb = []
    for lo, hi in ((0, terc[0]), (terc[0], terc[1]), (terc[1], 1e9)):
        m = (amax >= lo) & (amax < hi)
        Sm = G[m].sum(0)
        # at the global (Dp, D_rc) and free (Dp, D_rc)
        tb.append(dict(amax_lo=float(lo), amax_hi=float(min(hi, amax.max())), n=int(m.sum()),
                       s0_at_global=parab_min(S0, Sm[:, idx[1], idx[2]]), s0_free=best(G[m], S0, DP, DRC)[0],
                       dp_free=best(G[m], S0, DP, DRC)[1]))
    prof_s0 = (S.min(axis=(1, 2)) - S.min()); prof_dp = (S.min(axis=(0, 2)) - S.min())
    chi2dof = float(S.min() / dof.sum())
    # per-event best sigma0 at the global (Dp, D_rc)
    pe = np.array([parab_min(S0, g[:, idx[1], idx[2]]) for g in G])
    return dict(n=len(eids), s0=s0, dp=dp, drc=drc, s0_err=float(np.std(bs[:, 0])), dp_err=float(np.std(bs[:, 1])),
                drc_err=float(np.std(bs[:, 2])), chi2dof=chi2dof, prof_s0=(prof_s0 / chi2dof).tolist(),
                prof_dp=(prof_dp / chi2dof).tolist(), terciles=tb, s0_event_p=np.percentile(pe, [16, 50, 84]).tolist(),
                frac_event_wide=float(np.mean(pe >= 0.42)), eids=eids, amax=amax, pe=pe)


def t0_pos(D):
    """x-y t0 difference and position residuals (wft p0 and amplitude centroid against the uRWELL prediction)."""
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))
    rows = [r for r in D['rows'] if 'x' in r and 'y' in r]
    dt = np.array([r['x']['t0'] - r['y']['t0'] for r in rows])
    out = dict(n_xy=len(rows), dt_med=float(np.median(dt)), dt_rsig=float(1.4826 * np.median(np.abs(dt - np.median(dt)))))
    return out


def rsig(x):
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def position(D, cache, v):
    """p0 (wft) and amplitude centroid minus the uRWELL prediction, per view, linear alignment removed."""
    E = pickle.load(open(cache, 'rb'))
    R = view_rows(D, v); p, c, r = [], [], []
    for eid, f in R.items():
        e = E[eid]; W = np.asarray(e[v]['W'], float); pos = np.asarray(e[v]['pos'], float)
        a = W.max(1); k = int(np.argmax(a)); s = slice(max(k - 2, 0), k + 3)
        p.append(f['p0']); c.append(float((a[s] * pos[s]).sum() / a[s].sum())); r.append(e[f'ref_mesh_{v}'])
    p, c, r = map(np.asarray, (p, c, r))
    out = {}
    for name, x in (('wft', p), ('centroid', c)):
        d = x - r
        ok = np.abs(d - np.median(d)) < 5
        A = np.vstack([r[ok], np.ones(ok.sum())]).T
        coef = np.linalg.lstsq(A, x[ok], rcond=None)[0]
        res = x - (coef[0] * r + coef[1])
        out[name] = dict(rsig_um=rsig(res[np.abs(res) < 3]) * 1e3, slope=float(coef[0]))
    dd = p - c
    out['wft_minus_centroid_rsig_um'] = rsig(dd[np.abs(dd) < 3]) * 1e3
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', default=os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_sps'))
    ap.add_argument('--cache-dir', default='/media/dylan/data/x17/cosmic_bench/outliers/sps')
    a = ap.parse_args(); os.makedirs(FIG, exist_ok=True)
    summ = {}; keep = {}
    for tag in ('raw700', 'raw450', 'raw275', 'raw700_zs4', 'raw700_zs5', 'raw450_zs4', 'raw275_zs4', 'raw700_nocens',
                'raw700_zs4_zero', 'raw700_zs5_zero', 'raw275_zs4_zero'):
        D = load(a.dir, tag)
        if D is None:
            continue
        summ[tag] = {}
        base = tag.split('_')[0]
        for v in 'xy':
            r = analyse(D, v); keep[(tag, v)] = r
            summ[tag][v] = {k: x for k, x in r.items() if k not in ('eids', 'amax', 'pe')}
            print(f'{tag:14s} {v}: n {r["n"]:4d}  sigma0 {r["s0"]:.3f} ± {r["s0_err"]:.3f}  Dp {r["dp"]:.4f} ± {r["dp_err"]:.4f} '
                  f'(Magboltz {dp_magboltz(base):.4f})  D_rc {r["drc"]:.2e} ± {r["drc_err"]:.1e}  chi2/dof {r["chi2dof"]:.2f}  '
                  f'per-event s0 p16/50/84 {np.round(r["s0_event_p"], 3)}  wide(>=0.42) {r["frac_event_wide"]:.2f}')
            for t in r['terciles']:
                print(f'        amax {t["amax_lo"]:6.0f}-{t["amax_hi"]:6.0f}  n {t["n"]:4d}  s0 (global Dp) {t["s0_at_global"]:.3f}  '
                      f'free s0 {t["s0_free"]:.3f} Dp {t["dp_free"]:.4f}')
        summ[tag]['t0'] = t0_pos(D)
        print(f'        x-y t0: median {summ[tag]["t0"]["dt_med"]:+.1f} ns, rsig {summ[tag]["t0"]["dt_rsig"]:.1f} ns (n {summ[tag]["t0"]["n_xy"]})')
    # paired shifts of the per-event best sigma0 (same events) for the ZS / no-censor arms
    for tag in ('raw700_zs4', 'raw700_zs5', 'raw450_zs4', 'raw275_zs4', 'raw700_nocens', 'raw700_zs4_zero', 'raw700_zs5_zero', 'raw275_zs4_zero'):
        base = tag.split('_')[0]
        for v in 'xy':
            if (tag, v) in keep and (base, v) in keep:
                A, B = keep[(base, v)], keep[(tag, v)]
                common, ia, ib = np.intersect1d(A['eids'], B['eids'], return_indices=True)
                d = B['pe'][ib] - A['pe'][ia]
                summ[tag][v]['paired_event_s0_shift_med'] = float(np.median(d))
                summ[tag][v]['global_shift'] = dict(s0=B['s0'] - A['s0'], dp=B['dp'] - A['dp'], drc=B['drc'] - A['drc'])
                print(f'{tag} {v}: vs RAW, global sigma0 {B["s0"] - A["s0"]:+.3f}, Dp {B["dp"] - A["dp"]:+.4f}, '
                      f'D_rc {B["drc"] - A["drc"]:+.1e}; per-event median shift {np.median(d):+.3f} (n {len(common)})')
    for b in E_VCM:
        summ.setdefault('magboltz_dp', {})[b] = dp_magboltz(b)
    json.dump(summ, open(os.path.join(a.dir, 'headon_summary.json'), 'w'), indent=1, default=float)
    pickle.dump({f'{k[0]}|{k[1]}': {kk: vv for kk, vv in r.items()} for k, r in keep.items()},
                open(os.path.join(a.dir, 'headon_keep.pkl'), 'wb'))


if __name__ == '__main__':
    main()
