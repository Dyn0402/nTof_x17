#!/usr/bin/env python3
"""headon_sigma.py -- the prompt footprint sigma0 and drift diffusion Dp from SPS det4 head-on tracks, per event.

The question (Dylan, 2026-10-27, F18): can a physical sigma0 be fixed from the head-on SPS data and used as the
reasonable value for the other chambers?  Head-on (flat mount) every drift slice lands at the same strip, so the
per-depth footprint sigma(u)^2 = sigma0^2 + Dp^2 u (+ 2 D_rc s along the resistive strips, Y) is the ONLY lateral
extent; there is no tilt to trade it against (the width-tilt degeneracy of F4/F18/F21 is absent by construction).

Method: the production forward model (wft.model.build_matrix_rc, det4's promoted bench bundle: templates, rc_tmpl, K
grown to cover the beam column), on run_71 RAW head-on events (sps_cache.py caches, uRWELL-centred +-4 strip windows):
  1. a free (p0, w, t0) fit at a reference hyper point;
  2. at fixed (p0, w) the chi2 on a (sigma0, Dp[, D_rc]) grid, t0 re-optimised over +-5 ns; summed over events and
     per event.
Every exactly-zero sample is censored (WFT_ZS_CENSOR semantics): in the run_71 RAW caches those are the FEU's dropped
sample groups (23 % of all samples, stored as 0 by sps_cache.py), not zero charge.  '--zs K' emulates DREAM sample-level
zero suppression at K sigma (samples below K*noise set to 0, then censored like real ZS data) to measure what ZS does to
the same events.

Clean-window selection (cloud_basics FINDINGS §35: det4's dead/weak strips fake a narrow footprint): per channel,
health = median of (peak / leading-strip peak) when the channel is the +-1 neighbour, against the view's median;
flagged < 0.5x or > 2x.  An event's view is used only if the leading strip is within +-1 of the window centre, no
flagged strip is within +-3 of it, all 9 strips present, and the window max is 250-3200 ADC.

    headon_sigma.py fit --plateau raw700 [--zs 4] [--nev 500] [--jobs 3] [--chunk I N] [--out DIR]
    headon_sigma.py health              # writes <out>/health.json
    headon_sigma.py collect --plateau raw700 [--zs 4]   # merge the --chunk pickles, summarise
Condor: run_ws_sps.sh / make_ws_sps_sub.py (WS_SPS_DATA, WS_SPS_BUNDLE point at the EOS copies).
Output: <out>/headon_<plateau>[_zs<k>].npz  (per-event chi2 grids + the base fit) and a printed summary.
"""
import argparse, json, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
DATA = os.environ.get('WS_SPS_DATA', '/media/dylan/data/x17/cosmic_bench/outliers/sps')
CACHE = os.path.join(DATA, 'sps_run71_{}.pkl')
BUNDLE = os.environ.get('WS_SPS_BUNDLE', os.path.expanduser('~/x17/cosmic_bench/model_audit/det4/calib_bundle'))
OUT = os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_sps')
K_BINS = 60                       # 3.6 us of drift: covers the 275 V plateau's column inside the 64-sample window
S0 = np.round(np.arange(0.10, 0.601, 0.05), 3)
DP = np.round(np.arange(0.0, 0.0301, 0.005), 4)
DRC = np.array([1.4e-4, 2.2e-4, 3.0e-4, 3.8e-4])
REF = dict(sigma_p0_x=0.238, sigma_p0_y=0.343, Dp=0.012, rc_D_y=2.2e-4)
_EV = None; _CAL = None; _ZS = 0.0; _CENSOR = 1


def _init(plateau, zs, censor=1):
    global _EV, _CAL, _ZS, _CENSOR
    from wft.calib import CalibrationBundle
    from wft import model as wm
    # censor 1: every exact zero censored (RAW drops; with --zs also the suppressed samples, the F16/F20 treatment);
    # 0: nothing censored (the 'nocens' arm: what fits that took the RAW drops as 0 saw);
    # 2 (with --zs): the suppressed samples are fitted as measured zeros, the RAW drops censored (P['missing'])
    wm.ZS_CENSOR = censor == 1
    _CENSOR = censor
    _EV = pickle.load(open(plateau if plateau.endswith('.pkl') else CACHE.format(plateau), 'rb'))
    cal = CalibrationBundle.load(BUNDLE)
    cal.dead, cal.hot = {}, {}                       # bench channel numbering; SPS FEU 03 is cabled differently (F12)
    for k in [k for k in cal.hyper if k.startswith('sigma_p0_prof')]:
        cal.hyper[k] = []
    cal.hyper.update(REF)
    wm.use_calibration(cal); wm.set_depth_bins(K_BINS); wm.set_nsamp(64); wm.COL_GUARD = 0.01
    _CAL = cal; _ZS = zs


def health(plateaus=('raw700', 'raw450', 'raw275')):
    """Per channel, as the +-1 neighbour of the leading strip: median peak ratio, normalised to the view median."""
    rat = {'x': {}, 'y': {}}; lead = {'x': {}, 'y': {}}
    for p in plateaus:
        for e in pickle.load(open(CACHE.format(p), 'rb')).values():
            for v in 'xy':
                W = np.asarray(e[v]['W']); ch = np.asarray(e[v]['ch'])
                pk = W.max(1); c = int(np.argmax(pk))
                if pk[c] < 250 or pk[c] > 3200:
                    continue
                lead[v].setdefault(int(ch[c]), 0); lead[v][int(ch[c])] += 1
                for d in (-1, 1):
                    if 0 <= c + d < len(ch):
                        rat[v].setdefault(int(ch[c + d]), []).append(pk[c + d] / pk[c])
    out = {}
    for v in 'xy':
        med = {c: float(np.median(r)) for c, r in rat[v].items() if len(r) >= 15}
        ref = float(np.median(list(med.values())))
        bad = sorted(c for c, m in med.items() if m < 0.5 * ref or m > 2.0 * ref)
        out[v] = dict(ref=ref, n_judged=len(med), flagged=bad, ratio={str(c): m / ref for c, m in med.items()},
                      lead_counts={str(c): n for c, n in lead[v].items()})
        print(f'{v}: {len(med)} channels judged, median +-1/centre {ref:.3f}, flagged {len(bad)}: {bad}')
    return out


def clean(e, v, bad):
    W = np.asarray(e[v]['W']); ch = np.asarray(e[v]['ch'])
    if W.shape[0] != 9:
        return False
    pk = W.max(1); c = int(np.argmax(pk))
    if abs(c - 4) > 1 or not (250 <= pk[c] <= 3200):
        return False
    near = ch[max(c - 3, 0):c + 4]
    return not np.isin(near, bad).any()


def one(args):
    eid, views = args
    from wft import model as wm
    e = _EV[eid]; out = {'eid': int(eid)}
    for v in views:
        P = dict(e[v]); W = np.asarray(P['W'], float).copy()
        if _ZS > 0:
            if _CENSOR == 2:
                # 'zero' mode: the suppressed samples are measured zeros; only the RAW drops are missing (censored)
                P['missing'] = W == 0.0
            W[W < _ZS * np.asarray(P['noise'], float)[:, None]] = 0.0
        P['W'] = W
        hy = dict(_CAL.hyper)
        p0i, _, t0i = wm.init_guess(P, v)
        try:
            r = wm.fit_plane_raw(P, v, p0i, 0.0, t0i, hyper=hy)
        except Exception:
            continue
        if not np.isfinite(r['chi2']):
            continue
        Wp, noise, pos, sat = wm.prep_plane(P, v)
        drc = DRC if v == 'y' else np.array([0.0])
        G = np.full((len(S0), len(DP), len(drc)), np.inf)
        for i, s in enumerate(S0):
            for j, dp in enumerate(DP):
                for k, d in enumerate(drc):
                    h = dict(hy); h[f'sigma_p0_{v}'] = float(s); h['Dp'] = float(dp); h['rc_D_y'] = float(d)
                    best = np.inf
                    for dt in (-5.0, 0.0, 5.0):
                        c, _ = wm.chi2_plane(v, Wp, noise, pos, sat, r['p0'], r['w'], r['t0'] + dt, h)
                        best = min(best, c)
                    G[i, j, k] = best
        q = np.asarray(r['q'], float)
        out[v] = dict(G=G, p0=r['p0'], w=r['w'], t0=r['t0'], chi2=r['chi2'], dof=r['dof'], amax=float(W.max()),
                      qsum=float(q.sum()), q=q, nsat=int(sat.sum()))
    return out


def summarise(R, v):
    rows = [r[v] for r in R if v in r]
    if not rows:
        return None
    G = np.stack([r['G'] for r in rows]); dof = np.array([r['dof'] for r in rows])
    S = G.sum(0)
    i, j, k = np.unravel_index(np.argmin(S), S.shape)
    # errors: scale the summed chi2 by its own chi2/dof (model-limited data), then delta = 1 along each axis profile
    scale = S[i, j, k] / dof.sum()
    prof_s = (S.min(axis=(1, 2)) - S.min()) / scale
    prof_d = (S.min(axis=(0, 2)) - S.min()) / scale
    # per-event bootstrap of the minimum location (events are the independent unit)
    rng = np.random.default_rng(1); bs = []
    for _ in range(200):
        Sb = G[rng.integers(0, len(G), len(G))].sum(0)
        bs.append(np.unravel_index(np.argmin(Sb), Sb.shape))
    bs = np.array(bs)
    return dict(n=len(rows), s0=float(S0[i]), dp=float(DP[j]), drc=float((DRC if v == 'y' else [0.0])[k]),
                chi2dof=float(scale), prof_s0=prof_s.tolist(), prof_dp=prof_d.tolist(),
                s0_bs=[float(np.percentile(S0[bs[:, 0]], q)) for q in (16, 50, 84)],
                dp_bs=[float(np.percentile(DP[bs[:, 1]], q)) for q in (16, 50, 84)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['fit', 'health', 'collect'])
    ap.add_argument('--chunk', type=int, nargs=2, default=None, metavar=('I', 'N'))
    ap.add_argument('--plateau', default='raw700'); ap.add_argument('--zs', type=float, default=0.0)
    ap.add_argument('--nev', type=int, default=500); ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--cache', default=None, help='any head-on cache pickle (9-strip windows) instead of the run_71 one')
    ap.add_argument('--tag', default=None, help='output tag (with --cache)'); ap.add_argument('--views', default='xy')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--censor', type=int, default=1, choices=[0, 1, 2],
                    help='1: censor exact zeros; 0: fit the RAW drops as zeros (bias of earlier run_71 fits); 2: ZS zeros measured')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    hp = os.path.join(a.out, 'health.json')
    if a.cmd == 'health':
        json.dump(health(), open(hp, 'w'), indent=1); return
    tag = f'{a.plateau}' + (f'_zs{a.zs:g}' if a.zs else '') + {0: '_nocens', 1: '', 2: '_zero'}[a.censor]
    if a.cmd == 'collect':
        import glob
        R = []
        for f in sorted(glob.glob(os.path.join(a.out, f'headon_{tag}_c*.pkl'))):
            R += pickle.load(open(f, 'rb'))['rows']
        with open(os.path.join(a.out, f'headon_{tag}.pkl'), 'wb') as f:
            pickle.dump(dict(S0=S0, DP=DP, DRC=DRC, ref=REF, K=K_BINS, rows=R), f, protocol=4)
        report(R, a.out, tag); return
    H = json.load(open(hp))
    src = a.cache or CACHE.format(a.plateau)
    if a.cache:
        tag = a.tag or os.path.basename(a.cache)[:-4]
    E = pickle.load(open(src, 'rb'))
    tasks = []
    nx = ny = 0
    for eid in sorted(E):
        views = [v for v in a.views if clean(E[eid], v, H[v]['flagged'])]
        if 'x' in views and nx >= a.nev:
            views.remove('x')
        if 'y' in views and ny >= a.nev:
            views.remove('y')
        if views:
            tasks.append((eid, views)); nx += 'x' in views; ny += 'y' in views
    del E
    if a.chunk:
        tasks = tasks[a.chunk[0]::a.chunk[1]]
        tag += f'_c{a.chunk[0]:02d}'
    print(f'{a.plateau} zs={a.zs}: clean views x {nx}, y {ny}', flush=True)
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(src, a.zs, a.censor)) as pool:
        R = list(pool.map(one, tasks, chunksize=4))
    with open(os.path.join(a.out, f'headon_{tag}.pkl'), 'wb') as f:
        pickle.dump(dict(S0=S0, DP=DP, DRC=DRC, ref=REF, K=K_BINS, rows=R), f, protocol=4)
    if not a.chunk:
        report(R, a.out, tag)


def report(R, out, tag):
    summ = {v: summarise(R, v) for v in 'xy'}
    json.dump(summ, open(os.path.join(out, f'headon_{tag}.json'), 'w'), indent=1)
    for v in 'xy':
        s = summ[v]
        if s:
            print(f'  {tag} {v}: n {s["n"]}  best sigma0 {s["s0"]:.2f} (bs {s["s0_bs"][0]:.2f}-{s["s0_bs"][2]:.2f})  Dp {s["dp"]:.3f} '
                  f'(bs {s["dp_bs"][0]:.3f}-{s["dp_bs"][2]:.3f})  D_rc {s["drc"]:.1e}  chi2/dof {s["chi2dof"]:.1f}', flush=True)


if __name__ == '__main__':
    main()
