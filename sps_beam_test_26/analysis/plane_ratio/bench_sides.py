"""Per-side neighbour area/delay on near-vertical bench cosmics (det3, det4)."""
import sys, numpy as np
sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17_paper/sps_beam_test_26/analysis/sharing_kernel')
import bench_kernel as bk
A3 = '/home/dylan/x17/cosmic_bench/Analysis/'
caches = {'det3': A3 + 'mx17_det3_saturday_scan_6-27-26/long_run_resist_490V_drift_1000V/mx17_3/wft/calib_work/calib_cache.pkl',
          'det4': A3 + 'mx17_det4_day_6-24-26/long_run/mx17_4/wft/calib_work/calib_cache.pkl'}
tmax = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
rng = np.random.default_rng(1)
for det, c in caches.items():
    for v in 'xy':
        A, t, nev = bk.build(v, tmax, 12, cache=c)
        def stats(idx=None):
            W = {d: bk.trim_mean(A[d] if idx is None else A[d][idx]) for d in A}
            s0 = W[0]; p0 = np.clip(s0, 0, None); c0 = (t * p0).sum() / p0.sum()
            o = {}
            for d in (1, -1, 2, -2):
                p = np.clip(W[d], 0, None)
                o[d] = (W[d].sum() / s0.sum(), (t * p).sum() / p.sum() - c0)
            o['r'] = (o[2][0] + o[-2][0]) / (o[1][0] + o[-1][0])
            return o
        o = stats()
        bs = [stats(rng.integers(0, nev, nev))['r'] for _ in range(100)]
        print(f'{det} {v} n={nev:5d} | ' + ' | '.join(f'{d:+d}: {o[d][0]:.3f} {o[d][1]:+4.0f}ns' for d in (1, -1, 2, -2))
              + f' | (±2)/(±1) {o["r"]:.3f} ± {np.std(bs):.3f}')
