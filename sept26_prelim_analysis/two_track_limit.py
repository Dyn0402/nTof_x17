#!/usr/bin/env python3
"""
two_track_limit -- where is the real limit on two tracks in one plane?

HANDOFF_TWO_TRACK_LIMIT.md steps 1-2. A ladder of idealisations, each removing
one piece of ignorance, all scored per view (one plane) so they compare:

  R1 asimov   information limit. Two tracks generated noise-free by the forward
              model; the best ONE-track fit leaves chi2 = lambda(d), the
              expected Delta-chi2 of a perfect analysis (the noncentrality).
  R2 oracle   perfect model + white noise at the run's level. One- and
              two-track fits started at the truth AND from a broad start set,
              pure chi2 (no barrier, no guards). Threshold from the same fit on
              synthetic singles at a fixed false-split rate.
  R3 real     the same oracle fitter on real windows: overlay-bench pairs
              (``intra_bench --overlay replace``) and real clean single donors
              for the threshold. Ideal algorithm, real detector.
  R4 prod     production's two-track fit on the same inputs.

    python -m sept26_prelim_analysis.two_track_limit asimov
    python -m sept26_prelim_analysis.two_track_limit oracle --n 100
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sept26_prelim_analysis import paths                   # noqa: E402
from sept26_prelim_analysis import two_track_synth as ts   # noqa: E402

ARMS = ('A', 'C')
SEPS = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0,
                 10.0, 12.0])
TANS = (0.0, 0.3)
NOISE = ts.NOISE_MED            # ADC / strip / sample, run_145 median
QTOT = ts.Q_MED                 # NNLS charge of a clean single track
UEND = 840.0                    # ns, middle of the synthetic 600-1080 range
HALF = 30                       # window half-width [strips]: wide, no cropping
#: symmetric splits tried around the one-track optimum [mm]
SPLITS = (0.4, 0.8, 1.5, 3.0, 6.0, 10.0)


#: a fitted line matches a true one if their r.m.s. distance at the SAME
#: absolute times over the drift column is below this [mm]. Comparing p0 at the
#: mesh instead fails correct lines whose t0 slid by whole depth bins (the
#: t0 <-> q(depth) near-degeneracy moves p0 by w * dt at no chi2 cost).
LINE_MM = 1.0


def line_dist(f, t, uend=UEND) -> float:
    """r.m.s. distance [mm] between lines f and t, each (p0, w, t0), at the
    absolute times t0_true + u, u in [0, uend]. p(t) = p0 + w (t - t0)."""
    tt = t[2] + np.linspace(0.0, uend, 32)
    d = (f[0] + f[1] * (tt - f[2])) - (t[0] + t[1] * (tt - t[2]))
    return float(np.sqrt(np.mean(d * d)))


def match(fits, truths) -> tuple:
    """Best distinct assignment of fitted lines to true ones. Returns
    (found, [distance per truth]); found when every truth is within LINE_MM."""
    import itertools
    best = None
    for perm in itertools.permutations(range(len(fits)), len(truths)):
        ds = [line_dist(fits[i], t) for i, t in zip(perm, truths)]
        if best is None or sum(ds) < sum(best):
            best = ds
    if best is None:
        return False, [np.nan] * len(truths)
    return bool(max(best) < LINE_MM), best


def out_dir() -> Path:
    return paths.out('two_track_limit')


# --------------------------------------------------------------------------- #
# the fitters: pure chi2, many starts. The ceiling, not production.
# --------------------------------------------------------------------------- #
class Plane:
    """One prepared window. chi2 is the model's own, NNLS profiles, no snapping.

    ``eps`` > 0 adds a fractional model error in quadrature to every sample,
    sigma^2 = noise^2 + (eps * model)^2, solved by reweighting (the model from a
    noise-only solve sets the weights of a second solve). A real track is not
    described perfectly and its residual grows with its signal; this is the
    candidate cure (TWO_TRACK_FIT_LOG, 2026-09-30). Saturated samples are
    simply dropped here, not penalised."""

    def __init__(self, plane, W, pos, noise, sat=None, eps: float = 0.0):
        self.plane, self.W, self.pos = plane, W, pos
        self.noise = np.full(len(pos), noise) if np.isscalar(noise) else noise
        self.sat = np.zeros_like(W, bool) if sat is None else sat
        self.eps = float(eps)

    def _chi_eps(self, M):
        from scipy.optimize import nnls
        from wft import model as wm
        ok = ~self.sat.reshape(-1)
        base = np.repeat(self.noise, wm.NSAMP)
        y = self.W.reshape(-1)
        sig = base
        for _ in range(2):
            try:
                q, rn = nnls((M / sig[:, None])[ok], (y / sig)[ok], maxiter=50 * M.shape[1])
            except Exception:
                return np.inf
            sig = np.sqrt(base ** 2 + (self.eps * (M @ q)) ** 2)
        return float(rn * rn)

    def one(self, v):
        from wft import model as wm
        if self.eps > 0:
            return self._chi_eps(wm.build_matrix(self.plane, self.pos, v[0], v[1], v[2],
                                                 wm.HYPER))
        return wm.chi2_plane(self.plane, self.W, self.noise, self.pos, self.sat,
                             v[0], v[1], v[2], wm.HYPER, snap_t0=False)[0]

    def two(self, v):
        """v = (p0a, wa, p0b, wb, t0): tied t0."""
        from wft import model as wm
        if self.eps > 0:
            return self._chi_eps(wm.build_matrix_two(self.plane, self.pos, (v[0], v[1], v[4]),
                                                     (v[2], v[3], v[4]), wm.HYPER))
        return wm.chi2_plane_two(self.plane, self.W, self.noise, self.pos, self.sat,
                                 (v[0], v[1], v[4]), (v[2], v[3], v[4]), wm.HYPER,
                                 snap_t0=False)[0]


def _nm(f, x0, step):
    x0 = np.asarray(x0, float)
    simplex = x0 + np.vstack([np.zeros(len(x0)), np.diag(step)])
    r = minimize(f, x0, method='Nelder-Mead',
                 options=dict(xatol=1e-4, fatol=1e-3, maxiter=3000, maxfev=3000,
                              initial_simplex=simplex))
    # restart once from the optimum: NM stalls on these surfaces
    r2 = minimize(f, r.x, method='Nelder-Mead',
                  options=dict(xatol=1e-4, fatol=1e-3, maxiter=2000, maxfev=2000,
                               initial_simplex=r.x + np.vstack([np.zeros(len(x0)),
                                                                np.diag(step) * 0.3])))
    return (r2.fun, r2.x) if r2.fun < r.fun else (r.fun, r.x)


def fit_one(P: Plane, starts):
    """Best single track over starts [(p0, w, t0)]: all scored, best 3 refined."""
    sc = sorted((P.one(s), tuple(s)) for s in starts)
    best = (np.inf, None)
    for _c, s in sc[:3]:
        b = _nm(P.one, s, [0.3, 2e-3, 20.0])
        if b[0] < best[0]:
            best = b
    return best


def fit_two(P: Plane, one_x, truth=None):
    """Best tied-t0 pair: symmetric splits around the one-track optimum (and
    the truth, when given). All scored, best 4 refined."""
    p0, w, t0 = one_x
    starts = [(p0 - s / 2, w, p0 + s / 2, w, t0) for s in SPLITS]
    starts += [(p0 - s / 2, w - dw, p0 + s / 2, w + dw, t0)
               for s in (0.8, 3.0) for dw in (-4e-3, 4e-3)]
    if truth is not None:
        (pa, wa, ta), (pb, wb, tb) = truth
        starts.append((pa, wa, pb, wb, 0.5 * (ta + tb)))
    sc = sorted((P.two(s), tuple(s)) for s in starts)
    best = (np.inf, None)
    for _c, s in sc[:4]:
        b = _nm(P.two, s, [0.3, 2e-3, 0.3, 2e-3, 20.0])
        if b[0] < best[0]:
            best = b
    return best


# --------------------------------------------------------------------------- #
# synthetic windows
# --------------------------------------------------------------------------- #
def synth_window(plane, tracks, rng, noise=NOISE, flat=False):
    """tracks = [(p0, w, t0, q_tot)]. Wide window centred on the pair."""
    from wft import model as wm
    c = np.mean([t[0] for t in tracks])
    pos = (np.arange(-HALF, HALF + 1) + np.round(c / wm.PITCH)) * wm.PITCH
    W = np.zeros((len(pos), wm.NSAMP))
    for p0, w, t0, qt in tracks:
        if flat:
            q = np.zeros(wm.K)
            q[:int(round(UEND / wm.DT))] = 1.0
        else:
            q = ts._profile(rng.uniform(ts.UEND_LO, ts.UEND_HI), rng)
        q *= qt / q.sum()
        W += (wm.build_matrix(plane, pos, p0, w, t0, wm.HYPER) @ q).reshape(len(pos), wm.NSAMP)
    if noise > 0:
        W = W + rng.normal(0.0, noise, W.shape)
    return Plane(plane, W, pos, max(noise, 1.0))


_ARM = None


def _init(arm):
    global _ARM
    if _ARM != arm:
        ts._init(arm)
        _ARM = arm


# --------------------------------------------------------------------------- #
# R1
# --------------------------------------------------------------------------- #
def _asimov_job(args):
    arm, plane, tan, d = args
    _init(arm)
    w = tan * ts._CAL.v_drift * 1e-3
    p0a = 100.3
    tr = [(p0a, w, 0.0, QTOT), (p0a + d, w, 0.0, QTOT)]
    P = synth_window(plane, tr, None, noise=0.0, flat=True)
    P.noise[:] = NOISE          # chi2 in units of the real noise
    starts = [(p, w, t) for p in np.arange(p0a - 1, p0a + d + 1.001, 0.1)
              for t in (-120, -60, 0, 60, 120)]
    lam, x = fit_one(P, starts)
    return dict(arm=arm, plane=plane, tan=tan, d=d, lam=float(lam),
                peak_snr=float(P.W.max() / NOISE))


def asimov(jobs):
    J = [(a, p, t, d) for a in ARMS for p in 'xy' for t in TANS for d in SEPS]
    with ProcessPoolExecutor(jobs) as ex:
        R = pd.DataFrame(list(ex.map(_asimov_job, J, chunksize=1)))
    R.to_csv(out_dir() / 'r1_asimov.csv', index=False)
    print(R.pivot_table(index='d', columns=['arm', 'plane', 'tan'], values='lam').round(1))


# --------------------------------------------------------------------------- #
# R2
# --------------------------------------------------------------------------- #
def _oracle_job(args):
    arm, plane, tan, d, n_true, seed = args
    _init(arm)
    rng = np.random.default_rng(seed)
    v = ts._CAL.v_drift
    w = tan * v * 1e-3
    t0 = 0.0
    p0a = float(100.0 + rng.uniform(0.0, 0.78))     # random phase on the strips
    tracks = [(p0a, w, t0, QTOT)]
    if n_true == 2:
        tracks.append((p0a + d, w, t0, QTOT))
    t = time.perf_counter()
    P = synth_window(plane, tracks, rng)
    mid = np.mean([x[0] for x in tracks])
    starts = [(x[0], w, t0) for x in tracks] + [(mid, w, t0)]
    c1, x1 = fit_one(P, starts)
    truth = [(x[0], x[1], x[2]) for x in tracks] if n_true == 2 else None
    c2, x2 = fit_two(P, x1, truth)
    row = dict(arm=arm, plane=plane, tan=tan, d=d if n_true == 2 else 0.0,
               n_true=n_true, seed=seed, chi2_one=float(c1), chi2_two=float(c2),
               dchi2=float(c1 - c2), dof=int(P.W.size),
               one_p0=x1[0], one_w=x1[1], one_t0=x1[2],
               sec=time.perf_counter() - t)
    pa, wa, pb, wb, tt = x2
    if pa > pb:
        pa, wa, pb, wb = pb, wb, pa, wa
    row.update(two_pa=pa, two_pb=pb, two_wa=wa, two_wb=wb, two_t0=tt)
    if n_true == 2:
        row['found'], ds = match([(pa, wa, tt), (pb, wb, tt)],
                                 [(p0a, w, t0), (p0a + d, w, t0)])
        row['line_da'], row['line_db'] = ds
    return row


def oracle(n: int, jobs: int, seed: int, arms=ARMS, planes='x'):
    J, k = [], 0
    for a in arms:
        for p in planes:
            for tan in TANS:
                for i in range(4 * n):              # nulls: 4x the pairs, for the tail
                    J.append((a, p, tan, 0.0, 1, seed + k)); k += 1
                for d in SEPS:
                    for i in range(n):
                        J.append((a, p, tan, float(d), 2, seed + k)); k += 1
    t = time.time()
    rows = []
    with ProcessPoolExecutor(jobs) as ex:
        for i, r in enumerate(ex.map(_oracle_job, J, chunksize=4)):
            rows.append(r)
            if (i + 1) % 1000 == 0:
                print(f'[oracle] {i + 1:,}/{len(J):,}  {time.time() - t:.0f} s', flush=True)
    R = pd.DataFrame(rows)
    R.to_parquet(out_dir() / 'r2_oracle.parquet', index=False)
    print(f'[oracle] {len(R):,} planes in {time.time() - t:.0f} s')
    print(power_table(R).to_string())


def _synthprod_job(args):
    """R4 on the synthetics: production's one-track fit, trigger and joint fit
    on the SAME plane as :func:`_oracle_job` (same seed, same draws), cut to a
    production-style window (strips a 5-sigma hit would have fired on + pad)."""
    from wft import model as wm
    from wft import reco as wr
    arm, plane, tan, d, n_true, seed, opts = args
    _init(arm)
    wm.DEAD, wm.HOT = {}, {}             # the oracle sees none; neither may this
    wr.TWO_TRACK_F, wr.TWO_TRACK_F_CORROB = 300.0, 120.0
    wr.TWO_TRACK_T0, wr.TWO_TRACK_RESID_Z = 'tied', 8.0
    wr.TWO_TRACK_SCALE = opts.get('scale', 'one')
    wr.TWO_TRACK_SEARCH = opts.get('search', 'starts')
    rng = np.random.default_rng(seed)
    w = tan * ts._CAL.v_drift * 1e-3
    p0a = float(100.0 + rng.uniform(0.0, 0.78))
    tracks = [(p0a, w, 0.0, QTOT)]
    if n_true == 2:
        tracks.append((p0a + d, w, 0.0, QTOT))
    P = synth_window(plane, tracks, rng)
    row = dict(arm=arm, plane=plane, tan=tan, d=d if n_true == 2 else 0.0,
               n_true=n_true, seed=seed)
    live = np.flatnonzero(P.W.max(axis=1) / NOISE > ts.SIG_SEED)
    if len(live) < 3:
        return row
    lo, hi = max(0, live.min() - ts.PAD), min(len(P.pos) - 1, live.max() + ts.PAD)
    sl = slice(lo, hi + 1)
    ch = np.round(P.pos[sl] / wm.PITCH).astype(int)
    win = dict(W=P.W[sl], pos=P.pos[sl], noise=P.noise[sl], ch=ch)
    row['n_strips'] = int(hi - lo + 1)
    try:
        f = wr.fit_plane(win, plane, ts._CAL)
    except Exception:
        f = None
    if f is None:
        return row
    row.update(prod_one_p0=f.p0, prod_one_chi2=f.chi2)
    probe = wr.two_track_probe(win, plane, f, ts._CAL.hyper)
    if probe is None:
        return row
    trig = wr.two_track_triggers(probe)
    row['prod_trig'] = bool(trig['residual'] or trig['width'])
    try:
        r = wr.fit_plane_two(win, plane, ts._CAL, f, probe=probe)
    except Exception:
        r = None
    if r is None:
        return row
    ca, cb = r['children']
    row.update(prod_fstat=r['fstat'], prod_dchi2=r['dchi2'], prod_guards=r['guards_ok'],
               prod_accepted=bool(r['accepted']), prod_marg=min(r['marg_a'], r['marg_b']),
               prod_chi2_one=r['chi2_one'], prod_chi2_two=r['chi2_two'], prod_dof=r['dof'],
               # the same fit's statistic with the two-track scale (TWO_TRACK_SCALE='two')
               prod_fstat2=min(r['marg_a'], r['marg_b']) / max(r['chi2_two'] / max(r['dof'], 1), 1e-9))
    if n_true == 2:
        row['prod_found'], ds = match([(c.p0, c.w, c.t0) for c in (ca, cb)],
                                      [(p0a, w, 0.0), (p0a + d, w, 0.0)])
        row['prod_line_da'], row['prod_line_db'] = ds
    row.update(prod_pa=ca.p0, prod_wa=ca.w, prod_t0a=ca.t0,
               prod_pb=cb.p0, prod_wb=cb.w, prod_t0b=cb.t0)
    return row


def synthprod(n: int, jobs: int, seed: int, arms=ARMS, planes='x', opts=None,
              tag: str = ''):
    """Same job list, same seeds, as :func:`oracle`. ``opts``: scale, search."""
    opts = opts or {}
    J, k = [], 0
    for a in arms:
        for p in planes:
            for tan in TANS:
                for i in range(4 * n):
                    J.append((a, p, tan, 0.0, 1, seed + k, opts)); k += 1
                for d in SEPS:
                    for i in range(n):
                        J.append((a, p, tan, float(d), 2, seed + k, opts)); k += 1
    t = time.time()
    with ProcessPoolExecutor(jobs) as ex:
        R = pd.DataFrame(list(ex.map(_synthprod_job, J, chunksize=4)))
    R = R.assign(scale=opts.get('scale', 'one'), search=opts.get('search', 'starts'))
    R.to_parquet(out_dir() / f'r4_synthprod{"_" + tag if tag else ""}.parquet', index=False)
    print(f'[synthprod] {len(R):,} planes in {time.time() - t:.0f} s')


def power_table(R: pd.DataFrame, fsr=(0.01, 0.001)) -> pd.DataFrame:
    """Efficiency vs d at the Delta-chi2 threshold giving false-split rate fsr
    on the singles of the same (arm, plane, tan)."""
    out = []
    for key, g in R.groupby(['arm', 'plane', 'tan']):
        null = g[g.n_true == 1].dchi2.to_numpy()
        for f in fsr:
            thr = float(np.quantile(null, 1.0 - f))
            for d, h in g[g.n_true == 2].groupby('d'):
                out.append(dict(zip(['arm', 'plane', 'tan'], key), fsr=f, thr=thr, d=d,
                                eff=float((h.dchi2 > thr).mean()),
                                eff_found=float(((h.dchi2 > thr) & h.found).mean()),
                                n=len(h)))
    return (pd.DataFrame(out).pivot_table(index='d', columns=['arm', 'tan', 'fsr'],
                                          values='eff_found').round(2))


# --------------------------------------------------------------------------- #
# R3 / R4: real windows, their perfect-model twins, and production
# --------------------------------------------------------------------------- #
#: per-view separation bins for the real pairs [mm]
REAL_SEP_BINS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0,
                          24.0])
COINC_NS = 30.0


def real_pairs(D: pd.DataFrame, plane: str, per_bin: int, n_null: int, seed: int):
    """Same-tag, same-phase donor pairs, coincident IN THIS VIEW, stratified on
    this view's separation; plus single donors for the null."""
    rng = np.random.default_rng(seed)
    frames = []
    for _k, g in D.groupby(['tag', 'x_ftst', 'y_ftst']):
        idx = g.index.to_numpy()
        if len(idx) < 2:
            continue
        i, j = np.triu_indices(len(idx), 1)
        frames.append(pd.DataFrame(dict(a=idx[i], b=idx[j])))
    P = pd.concat(frames, ignore_index=True)
    A, B = D.loc[P.a].reset_index(drop=True), D.loc[P.b].reset_index(drop=True)
    P['tag'] = A.tag.to_numpy()
    P['a_eid'], P['b_eid'] = A.event_id.to_numpy(), B.event_id.to_numpy()
    P['sep'] = np.abs(A[f'{plane}_p0'] - B[f'{plane}_p0']).to_numpy()
    P['dt'] = np.abs(A[f'{plane}_t0'] - B[f'{plane}_t0']).to_numpy()
    P = P[P.dt < COINC_NS]
    P['sbin'] = np.digitize(P.sep, REAL_SEP_BINS) - 1
    P = P[(P.sbin >= 0) & (P.sbin < len(REAL_SEP_BINS) - 1)]
    P = P.sample(frac=1.0, random_state=seed).groupby('sbin').head(per_bin)
    S = D.sample(n=min(n_null, len(D)), random_state=seed)
    S = pd.DataFrame(dict(tag=S.tag.to_numpy(), a_eid=S.event_id.to_numpy(), b_eid=-1,
                          sep=0.0, dt=0.0, sbin=-1))
    return pd.concat([P[S.columns], S], ignore_index=True)


def sep_rms(ta, tb, uend=UEND) -> float:
    """R.m.s. transverse distance between two straight tracks over the drift
    column u in [0, uend] ns, tracks given as (p0, w [mm/ns], t0)."""
    u = np.linspace(0.0, uend, 64)
    d = (ta[0] - tb[0]) + (ta[1] - tb[1]) * u
    return float(np.sqrt(np.mean(d * d)))


def _truth(t, plane, v):
    return (float(t[f'{plane}_p0']), float(t[f'{plane}_tan_theta']) * v * 1e-3,
            float(t[f'{plane}_t0']))


def _window_channels(td, plane, centre, half=HALF):
    feu = td.feu[plane]
    o = td.order[plane]
    p = td.pos[feu][o]
    k = int(np.argmin(np.abs(p - centre)))
    return o[max(0, k - half):k + half + 1]


def _prep(W, ch, td, plane):
    from wft import model as wm
    feu = td.feu[plane]
    return wm.prep_plane(dict(W=W[ch], pos=td.pos[feu][ch],
                              noise=np.maximum(td.rdr[plane].noise[ch], 3.0), ch=ch), plane)


def _real_job(job):
    """One real window (a pair overlay or a single), its perfect-model twin,
    and the oracle fits on both."""
    from wft import model as wm
    arm, meta, win, singles, truths = job
    _init(arm)
    rng = np.random.default_rng(int(meta['oid']))
    plane = meta['plane']
    W, noise, pos, sat = win
    if W.shape[1] != wm.NSAMP:
        wm.set_nsamp(W.shape[1])
    row = dict(meta)

    def oracle_on(P, tr):
        mid = np.mean([t[0] for t in tr])
        starts = [tuple(t) for t in tr] + [(mid, tr[0][1], tr[0][2])]
        c1, x1 = fit_one(P, starts)
        c2, x2 = fit_two(P, x1, tr if len(tr) == 2 else None)
        kids = [(x2[0], x2[1], x2[4]), (x2[2], x2[3], x2[4])]
        found, ds = match(kids, tr) if len(tr) == 2 else (False, [np.nan])
        return dict(chi2_one=float(c1), chi2_two=float(c2), dchi2=float(c1 - c2),
                    found=bool(found), line_dmax=float(np.nanmax(ds)),
                    two_pa=x2[0], two_wa=x2[1], two_pb=x2[2], two_wb=x2[3],
                    two_t0=x2[4], one_p0=x1[0], one_w=x1[1], one_t0=x1[2])

    row['dof'] = int((~sat).sum())
    eps_list = row.pop('eps_list', None)
    if eps_list:
        # the model-error study: the ideal fit on the real window at each eps,
        # plus the donors' own fit quality (eps = 0) for the well-modelled cut
        for e in eps_list:
            P = Plane(plane, W, pos, noise, sat, eps=e)
            row.update({f'e{e:g}_{k}': v for k, v in oracle_on(P, truths).items()
                        if k in ('chi2_one', 'chi2_two', 'dchi2', 'found')})
        row['donor_chi2dof'] = float(max(
            fit_one(Plane(plane, Ws, pos, ns, ss), [tr])[0] / max((~ss).sum(), 1)
            for (Ws, ns, _ps, ss), tr in zip(singles, truths)))
        return row
    # real
    Pr = Plane(plane, W, pos, noise, sat)
    row.update({f'real_{k}': v for k, v in oracle_on(Pr, truths).items()})
    # the twin: each donor refit alone on its own real window, then rebuilt
    Wt = np.zeros_like(W)
    for (Ws, ns, _ps, ss), tr in zip(singles, truths):
        Ps = Plane(plane, Ws, pos, ns, ss)
        c, x = fit_one(Ps, [tr])
        _c, q = wm.chi2_plane(plane, Ws, ns, pos, ss, x[0], x[1], x[2], wm.HYPER,
                              snap_t0=False)
        Wt += (wm.build_matrix(plane, pos, x[0], x[1], x[2], wm.HYPER) @ q
               ).reshape(W.shape)
        row.setdefault('donor_chi2dof', []).append(float(c / max((~ss).sum(), 1)))
    Wt += rng.normal(0.0, 1.0, W.shape) * noise[:, None]
    Pt = Plane(plane, Wt, pos, noise, None)
    row.update({f'twin_{k}': v for k, v in oracle_on(Pt, truths).items()})
    row['donor_chi2dof'] = float(np.max(row['donor_chi2dof']))
    return row


#: production variants for ``real --prod-only`` (worker options on top of the
#: bench's final configuration)
PROD_VARIANTS = {
    'current': {},
    'fixed': dict(TWO_TRACK_SCALE='two', TWO_TRACK_SEARCH='grid',
                  TWO_TRACK_RESID_Z=-np.inf, TWO_TRACK_MAX_TRY=99,
                  TWO_TRACK_SELECTED_ONLY=False),
    # every guard-passing split accepted and recorded: thresholds are scanned
    # offline (threshold_scan) against the false-split rate on real singles
    'fixed_f0': dict(TWO_TRACK_SCALE='two', TWO_TRACK_SEARCH='grid',
                     TWO_TRACK_RESID_Z=-np.inf, TWO_TRACK_MAX_TRY=99,
                     TWO_TRACK_SELECTED_ONLY=False, TWO_TRACK_F=0.0,
                     TWO_TRACK_F_CORROB=0.0),
    'current_f0': dict(TWO_TRACK_F=0.0, TWO_TRACK_F_CORROB=0.0),
    'scale': dict(TWO_TRACK_SCALE='two'),
    'notrig': dict(TWO_TRACK_RESID_Z=-np.inf, TWO_TRACK_MAX_TRY=99,
                   TWO_TRACK_SELECTED_ONLY=False),
}


def real(arms, per_bin: int, n_null: int, jobs: int, seed: int, planes='xy',
         with_prod: bool = True, prod_only: str = '', eps_list=None):
    """``prod_only``: skip the oracle, run the named PROD_VARIANTS entry on the
    same windows (same seed, same oids) into r3_prod_<name>.parquet."""
    from ntof_tracking import wft_beam as wb
    from wft import io as wio
    from wft import reco as wr
    from wft.calib import CalibrationBundle
    from sept26_prelim_analysis import intra_bench as ib

    rows, prod, cands, splits = [], [], [], []
    oid = 0
    t_start = time.time()
    for arm in arms:
        bundle = str(ib.reco_dir(arm) / 'calib_bundle_prelim')
        cal = CalibrationBundle.load(bundle)
        cfg = wb.beam_config(arm, run=ib.RUN, sub_run=ib.SUBRUN)
        spos = wio.strip_position_map(cfg)
        v = cal.v_drift
        D = ib.donors(arm)
        Dk = D.set_index(['tag', 'event_id'])
        L = pd.concat([real_pairs(D, p, per_bin, n_null, seed).assign(plane=p)
                       for p in planes], ignore_index=True)
        print(f'[real] arm {arm}: {len(L):,} windows '
              f'({(L.b_eid >= 0).sum():,} pairs)', flush=True)
        rng = np.random.default_rng(seed)
        oracle_jobs, prod_jobs, prod_meta = [], [], {}
        for tag, g in L.groupby('tag'):
            eids = set(g.a_eid) | set(g.b_eid[g.b_eid >= 0])
            td = ib.TagData(arm, tag, cfg, cal, spos, eids, rng, local_mm=16.0,
                            local_mode='rescue', overlay='replace')
            for r in g.itertuples():
                p = r.plane
                a, b = int(r.a_eid), int(r.b_eid)
                ta = Dk.loc[(tag, a)]
                tr = [_truth(ta, p, v)]
                Wa = td.wf[p][a]
                if b >= 0:
                    tb = Dk.loc[(tag, b)]
                    tr.append(_truth(tb, p, v))
                    W = Wa.copy()
                    reg = td.region(b, p)
                    own = np.isin(reg, td.region(a, p))
                    W[reg[own]] += td.wf[p][b][reg[own]]
                    W[reg[~own]] = td.wf[p][b][reg[~own]]
                    srcs = [Wa, td.wf[p][b]]
                else:
                    W, srcs = Wa, [Wa]
                ch = _window_channels(td, p, np.mean([t[0] for t in tr]))
                meta = dict(oid=oid, arm=arm, tag=tag, plane=p, a_eid=a, b_eid=b,
                            n_true=len(tr), sep=float(r.sep), dt=float(r.dt),
                            tan_a=float(ta[f'{p}_tan_theta']),
                            tan_b=float(Dk.loc[(tag, b)][f'{p}_tan_theta']) if b >= 0 else np.nan,
                            pa=tr[0][0], wa=tr[0][1], t0a=tr[0][2],
                            pb=tr[-1][0], wb=tr[-1][1], t0b=tr[-1][2],
                            sep_rms=sep_rms(tr[0], tr[-1]) if b >= 0 else 0.0,
                            qa=float(ta[f'{p}_q_sum']),
                            qb=float(Dk.loc[(tag, b)][f'{p}_q_sum']) if b >= 0 else np.nan)
                if eps_list:
                    meta['eps_list'] = list(eps_list)
                pp = meta['plane']
                if not prod_only:
                    oracle_jobs.append((arm, meta, _prep_main(W, ch, td, pp, cal),
                                        [_prep_main(s, ch, td, pp, cal) for s in srcs], tr))
                if with_prod:
                    pl, _info = td.payload(oid, a, b if b >= 0 else None,
                                           'overlay' if b >= 0 else 'single')
                    if pl is not None:
                        prod_jobs.append(pl)
                    prod_meta[oid] = (meta, tr)
                oid += 1
        t0 = time.time()
        with ProcessPoolExecutor(jobs) as ex:
            for i, r in enumerate(ex.map(_real_job, oracle_jobs, chunksize=2)):
                rows.append(r)
                if (i + 1) % 500 == 0:
                    print(f'[real]   {arm} oracle {i + 1:,}/{len(oracle_jobs):,} '
                          f'{time.time() - t0:.0f} s', flush=True)
        if with_prod:
            pairing = str(ib.out_dir() / f'xy_pairing_{arm}.json')
            opts = dict(TWO_TRACK=True, TWO_TRACK_F=300.0, TWO_TRACK_F_CORROB=120.0,
                        TWO_TRACK_T0='tied', TWO_TRACK_RESID_Z=8.0)
            opts.update(PROD_VARIANTS[prod_only or 'current'])
            with ProcessPoolExecutor(jobs, initializer=wr._worker_init,
                                     initargs=(bundle, pairing, opts)) as ex:
                for out in ex.map(wr._worker_fit, prod_jobs, chunksize=2):
                    meta, tr = prod_meta[out['event_id']]
                    if prod_only:
                        cands.extend(dict(c, oid=meta['oid'], arm=arm)
                                     for c in out.get('_cand', []))
                        splits.extend(dict(r_, oid=meta['oid'], arm=arm)
                                      for r_ in out.get('_splits', []))
                    C = pd.DataFrame(out.get('_cand', []))
                    h = C[C.plane == meta['plane']] if len(C) else C
                    if len(h) and 'split_replaced' in h:
                        h = h[~h.split_replaced.fillna(False).astype(bool)]
                    fits = ([tuple(x) for x in h[['p0', 'w', 't0']].to_numpy(float)]
                            if len(h) else [])
                    found, ds = (match(fits, tr) if len(fits) >= len(tr)
                                 else (False, [np.nan] * len(tr)))
                    prod.append(dict(oid=meta['oid'], prod_n_cand=len(h),
                                     prod_found=bool(found),
                                     prod_line_dmax=float(np.nanmax(ds)),
                                     prod_n_tracks=out['n_tracks'],
                                     prod_split=int(out.get('n_splits', 0) or 0)))
        print(f'[real] arm {arm} done, {time.time() - t_start:.0f} s', flush=True)
    if eps_list:
        pd.DataFrame(rows).to_parquet(out_dir() / 'r3_eps.parquet', index=False)
        print(f'[real] eps study: {len(rows):,} windows in {time.time() - t_start:.0f} s')
        return
    if prod_only:
        pd.DataFrame(prod).to_parquet(out_dir() / f'r3_prod_{prod_only}.parquet', index=False)
        pd.DataFrame(cands).to_parquet(out_dir() / f'r3_cands_{prod_only}.parquet', index=False)
        if splits:
            pd.DataFrame(splits).to_parquet(out_dir() / f'r3_splits_{prod_only}.parquet',
                                            index=False)
        print(f'[real] prod {prod_only}: {len(prod):,} windows in {time.time() - t_start:.0f} s')
        return
    R = pd.DataFrame(rows)
    if prod:
        R = R.merge(pd.DataFrame(prod), on='oid', how='left')
    R.to_parquet(out_dir() / 'r3_real.parquet', index=False)
    print(f'[real] {len(R):,} windows in {time.time() - t_start:.0f} s')


#: thresholds scanned offline; the corroborated one is kept at production's ratio
SCAN_F = (0, 25, 50, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1600, 2000, 2400, 3200,
          4800, 5000)
F_CORROB_RATIO = 120.0 / 300.0


def _plane_sets(C: pd.DataFrame, S: pd.DataFrame, F: float) -> dict:
    """{(oid, plane): candidate lines [(p0, w, t0)]} as production would have
    emitted them at threshold F, rebuilt from a threshold-0 run: each recorded
    split is kept (its children) or undone (its parent)."""
    if len(S):
        thr = np.where(S.corroborated.astype(bool), F * F_CORROB_RATIO, F)
        acc = S.guards_ok.astype(bool) & (S.fstat.to_numpy() >= thr)
        S = S.assign(acc=acc)
    out = {}
    child = C.split_child.fillna(False).astype(bool)
    parent = C.split_replaced.fillna(False).astype(bool)
    base = C[~child & ~parent]
    for key, g in base.groupby(['oid', 'plane']):
        out[key] = [tuple(x) for x in g[['p0', 'w', 't0']].to_numpy(float)]
    Ck, Cp = C[child], C[parent]
    Ck_by = {k: g for k, g in Ck.groupby(['oid', 'plane'])}
    Cp_by = {k: g for k, g in Cp.groupby(['oid', 'plane'])}
    for key, g in (S.groupby(['oid', 'plane']) if len(S) else []):
        kids = Ck_by.get(key)
        pars = Cp_by.get(key)
        lst = out.setdefault(key, [])
        used = set()
        for s in g.itertuples():
            k = kids[np.isclose(kids.split_dchi2, s.dchi2)] if kids is not None else None
            if k is None or not len(k):
                continue            # guard-failed attempt: nothing was replaced
            if s.acc:
                lst.extend(tuple(x) for x in k[['p0', 'w', 't0']].to_numpy(float))
            elif pars is not None and len(pars):
                mid = k.p0.mean()
                order = np.argsort(np.abs(pars.p0.to_numpy() - mid))
                j = next((int(i) for i in order if int(i) not in used), None)
                if j is not None:
                    used.add(j)
                    lst.append(tuple(pars.iloc[j][['p0', 'w', 't0']].to_numpy(float)))
    return out


def threshold_scan(variants=('current_f0', 'fixed_f0')) -> pd.DataFrame:
    """False splits on real singles against efficiency on real pairs, per
    threshold, from the threshold-0 production runs on the R3 windows."""
    d = out_dir()
    R = pd.read_parquet(d / 'r3_real.parquet')
    R['clean'] = R.donor_chi2dof < 2.0
    rows = []
    for v in variants:
        C = pd.read_parquet(d / f'r3_cands_{v}.parquet')
        S = (pd.read_parquet(d / f'r3_splits_{v}.parquet')
             if (d / f'r3_splits_{v}.parquet').exists() else pd.DataFrame())
        for F in SCAN_F:
            sets = _plane_sets(C, S, F)
            if len(S):
                thr = np.where(S.corroborated.astype(bool), F * F_CORROB_RATIO, F)
                split_oids = set(S.oid[S.guards_ok.astype(bool) & (S.fstat >= thr)])
            else:
                split_oids = set()
            for r in R.itertuples():
                if r.n_true == 1:
                    rows.append(dict(variant=v, F=F, oid=r.oid, arm=r.arm, n_true=1,
                                     clean=r.clean, sep_rms=0.0,
                                     split=r.oid in split_oids))
                    continue
                fits = sets.get((r.oid, r.plane), [])
                tr = [(r.pa, r.wa, r.t0a), (r.pb, r.wb, r.t0b)]
                ok = match(fits, tr)[0] if len(fits) >= 2 else False
                rows.append(dict(variant=v, F=F, oid=r.oid, arm=r.arm, n_true=2,
                                 clean=r.clean, sep_rms=r.sep_rms, found=ok))
    T = pd.DataFrame(rows)
    T.to_parquet(d / 'r3_scan.parquet', index=False)
    return T


def _prep_main(W, ch, td, plane, cal):
    """prep_plane needs the model's calibration installed; do it in-process."""
    from wft import model as wm
    if wm.CAL is not cal:
        wm.use_calibration(cal)
    if W.shape[1] != wm.NSAMP:
        wm.set_nsamp(W.shape[1])
    return _prep(W, ch, td, plane)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('asimov')
    s.add_argument('--jobs', type=int, default=14)
    s = sub.add_parser('oracle')
    s.add_argument('--n', type=int, default=100)
    s.add_argument('--jobs', type=int, default=14)
    s.add_argument('--seed', type=int, default=20260929)
    s.add_argument('--arms', nargs='+', default=list(ARMS))
    s.add_argument('--planes', default='x')
    s = sub.add_parser('synthprod')
    s.add_argument('--scale', default='one', choices=['one', 'two'])
    s.add_argument('--search', default='starts', choices=['starts', 'grid'])
    s.add_argument('--tag', default='')
    s.add_argument('--n', type=int, default=100)
    s.add_argument('--jobs', type=int, default=14)
    s.add_argument('--seed', type=int, default=20260929)
    s.add_argument('--arms', nargs='+', default=list(ARMS))
    s.add_argument('--planes', default='x')
    s = sub.add_parser('real')
    s.add_argument('--per-bin', type=int, default=40)
    s.add_argument('--n-null', type=int, default=300)
    s.add_argument('--jobs', type=int, default=14)
    s.add_argument('--seed', type=int, default=20260929)
    s.add_argument('--arms', nargs='+', default=list(ARMS))
    s.add_argument('--planes', default='xy')
    s.add_argument('--no-prod', action='store_true')
    s.add_argument('--prod-only', default='', choices=[''] + list(PROD_VARIANTS))
    s.add_argument('--eps', type=float, nargs='*', default=None,
                   help='model-error study: ideal fit at these eps only (r3_eps.parquet)')
    s = sub.add_parser('scan')
    s.add_argument('variants', nargs='*', default=['current_f0', 'fixed_f0'])
    a = ap.parse_args()
    if a.cmd == 'scan':
        T = threshold_scan(tuple(a.variants))
        print(T.groupby(['variant', 'F', 'arm']).agg(
            fsr=('split', 'mean'), eff=('found', 'mean')).round(3).to_string())
        return 0
    if a.cmd == 'asimov':
        asimov(a.jobs)
    elif a.cmd == 'oracle':
        oracle(a.n, a.jobs, a.seed, a.arms, a.planes)
    elif a.cmd == 'synthprod':
        synthprod(a.n, a.jobs, a.seed, a.arms, a.planes,
                  dict(scale=a.scale, search=a.search), a.tag)
    elif a.cmd == 'real':
        real(a.arms, a.per_bin, a.n_null, a.jobs, a.seed, a.planes,
             not a.no_prod and not a.eps, a.prod_only, a.eps)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
