"""T10: does the SPS sigma0 measurement (FC, fc_headon_grid.py on det4 run_71 head-on RAW) depend on the template?

FC's measurement is independent of the other calibration issues found here: unsaturated events only (lead peak
250-3200), t0 free per event, D_rc,y / bounded D_x profiled jointly with sigma0 and Dp.  Its one inherited input is
the template (det4 bench, rc_tmpl='x').  This re-runs FC's own per-event grid with the det4 production template and
with the segment-free h_eff (t4_closure.json), on a reduced grid (sigma0 full range, Dp and D near FC's optima),
and reports the summed-chi2 sigma0 minimum per view (profiled over Dp, D).
Usage: python -I t10_sps_sigma0.py CACHE BUNDLE_DIR OUT.pkl [nev]   then   python -I t10_sps_sigma0.py --summ A.pkl B.pkl"""
import os, pickle, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '../../..'))
sys.path.insert(0, os.path.join(REPO, 'rerun/model_audit/followups/fc_sigma0_trade'))
HEALTH = os.path.join(REPO, 'rerun/model_audit/ws_sps/results/health.json')


def run(cache, bundle, out, nev='300'):
    import fc_headon_grid as F
    F.S0 = np.round(np.arange(0.14, 0.461, 0.04), 3)
    F.DP_SPS = np.array([0.005, 0.0075, 0.010])
    F.DV = np.array([1.0e-4, 2.2e-4, 3.5e-4])
    sys.argv = ['fc_headon_grid.py', '--kind', 'sps', '--cache', cache, '--bundle', bundle, '--health', HEALTH,
                '--out', out, '--nev', nev, '--jobs', '3']
    F.main()


def summ(*paths):
    for p in paths:
        R = pickle.load(open(p, 'rb')); S0 = R['S0']
        line = [os.path.basename(p)]
        for v in 'xy':
            G = [r[v]['G'] for r in R['rows'] if v in r and np.isfinite(r[v]['G']).all()]
            if not G:
                continue
            tot = np.sum(G, axis=0)                  # (s0, dp, d)
            prof = tot.reshape(len(S0), -1).min(axis=1)
            i = int(np.argmin(prof)); j = np.clip(i, 1, len(S0) - 2)
            c = np.polyfit(S0[j - 1:j + 2], prof[j - 1:j + 2], 2); smin = -c[1] / (2 * c[0])
            # event-bootstrap error on the minimum
            rng = np.random.default_rng(1); G = np.array(G); bs = []
            for _ in range(200):
                t = G[rng.integers(0, len(G), len(G))].sum(0).reshape(len(S0), -1).min(1)
                k = np.clip(int(np.argmin(t)), 1, len(S0) - 2); cc = np.polyfit(S0[k - 1:k + 2], t[k - 1:k + 2], 2)
                bs.append(-cc[1] / (2 * cc[0]))
            line.append(f'{v}: sigma0 {smin:.3f} ± {np.std(bs):.3f} (n={len(G)})')
        print(' | '.join(line))


if __name__ == '__main__':
    if sys.argv[1] == '--summ':
        summ(*sys.argv[2:])
    else:
        run(*sys.argv[1:])
