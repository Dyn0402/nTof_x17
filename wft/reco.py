"""
Per-event reconstruction and the batch driver.

One row per event, one set of columns per plane. The row carries the fit
(position at the mesh, transverse speed, angle), its errors, the charge-profile
summary, and the quality flags — plus enough provenance to know which
calibration produced it.

Reference-free by construction: the seed comes from the detector's own hits
(``wft.seed``), the starting point from the brightest strip and the earliest
half-maximum crossing, and the slope from a wide scan. The M3 reference is
never an input to a fit — otherwise alignment and efficiency would be circular.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np

from .calib import CalibrationBundle, check_xy_pairing
from . import model as wm

# ---------------------------------------------------------------- quality
TAN_MIN_SLOPE = 0.08       # below this the timing carries no slope information
FLOOR_TAN = 0.018          # ~1.0 deg: the measured per-event physics floor
FLOOR_P0_MM = 0.33         # measured charge-centroid jitter per 60 ns bin
CHI2DOF_BAD = 300.0        # showers / multi-track / spark

# slope search: +-0.021 mm/ns covers |tan| ~ 0.57 at v = 36.6 um/ns
W_SCAN_HALF = 0.021        # covers |tan| ~ 0.57 at v = 36.6 um/ns
W_SCAN_STEP = 0.0021
P0_SCAN_HALF = 2.5         # mm around the window's charge centroid
P0_SCAN_STEP = 0.5
T0_SCAN_HALF = 120.0
T0_SCAN_STEP = 40.0

# absolute-t0 prior overrides (None = defer to the calibration bundle). The
# bench harness sets these via reco_globals to A/B the prior without a new
# bundle; production should carry t0_abs/t0_prior_sigma in the bundle itself.
T0_PRIOR_SIGMA: Optional[float] = None
T0_ABS: Optional[dict] = None

# §21.1: p0 is the position AT THE MESH, but the global scan is centred on the
# window's charge centroid — on an inclined track those differ by ~w * (half
# the column), so 21 % of planes start outside the ±2.5 mm box at 5× the
# catastrophic-failure rate. P0_SHEAR evaluates each stage-2 (p0, w) point at
# p0 - w*u_mid instead: the same 11×21 grid, re-centred per slope, zero extra
# cost. Off by default pending its A/B (bench variant 'p0shear').
P0_SHEAR = False

# Two-sided start search (HANDOFF_TRACKING §17). _global_start scans t0 only at
# w = 0 and w only at that t0; a steep track's charge pins neither, so 13-38 %
# of fits at |tan| >= 0.3 land in the mirror basin (wrong sign, |tan| 1.2-4x)
# at a chi2 worse by a median 3,300-5,200, and right-sign fits run t0-late with
# angle (p0 slides 1-8 mm along the track). With WFT_TWO_SIDED=1 every plane
# fit also searches each slope sign on its own -- (p0, t0) at a few |tan|, then
# (p0, w) over that sign's half at the best t0, then the same Nelder-Mead --
# and keeps the lowest chi2. Steep synthetic muons: wrong sign 13-29 % -> 0-3 %
# at |tan| 0.3-0.6, nothing changes below 0.3. ~3,500 extra chi2 evaluations
# per candidate. Read at import, recorded in the stage-2 sidecar.
TWO_SIDED = os.environ.get('WFT_TWO_SIDED', '0') == '1'
TS_PROBE_TAN = (0.15, 0.4, 0.8)        # raw |tan| of the (p0, t0) stage
TS_TAN_LO, TS_TAN_HI, TS_TAN_STEP = 0.05, 1.6, 0.05
TS_P0_HALF, TS_P0_STEP = 6.0, 0.5      # mm around the window's charge centroid
TS_T0_HALF, TS_T0_STEP = 240.0, 40.0   # ns around the one-sided fit's t0

RECO_COLUMNS = [
    'event_id', 'n_hits', 'spark',
    # per plane p in (x, y): p0, w, t0, tan, errors, chi2, dof, profile, flags
]


@dataclass
class PlaneFit:
    p0: float                # track position at the mesh [mm]
    w: float                 # transverse speed [mm/ns]; tan = w / v_drift
    t0: float                # arrival time of charge from the mesh [ns]
    tan_theta: float
    theta_deg: float
    chi2: float
    dof: int
    p0_err: float
    w_err: float
    tan_err: float
    t0_err: float            # 1-sigma t0 from the chi2 curvature [ns]
    q_sum: float             # total fitted charge
    q_u50: float             # median charge arrival time after t0 [ns]
    q_u90: float
    q_uend: float            # last depth bin above 5 % of the profile peak [ns]
    n_strips: int            # strips in the fit window
    n_seed: int              # strips in the seed cluster
    n_dropped: int
    slope_reliable: bool
    quality_ok: bool
    n_candidates: int = 1        # candidate clusters fitted for this plane
    n_flagged_strips: int = 0    # of n_strips, how many are dead/hot wildcards
                                  # (HANDOFF_D_NOISY_CHANNELS.md item 4)


def _profile_summary(q: np.ndarray) -> tuple:
    """(total, median arrival, 90 % arrival, column end) from the NNLS charge
    profile, in ns after t0. Deliberately raw quantiles: the gap/column
    estimators that need a specific definition build it downstream."""
    q = np.asarray(q, float)
    tot = float(q.sum())
    if tot <= 0:
        return 0.0, np.nan, np.nan, np.nan
    u = wm.UK[:len(q)]
    c = np.cumsum(q) / tot
    u50 = float(np.interp(0.5, c, u))
    u90 = float(np.interp(0.9, c, u))
    live = np.where(q > 0.05 * q.max())[0]
    uend = float(u[live[-1]] + 0.5 * wm.DT) if len(live) else np.nan
    return tot, u50, u90, uend


def _errors(P, plane, r, hyper, dp=0.05, dw=2e-4, dt=2.0,
            t0_prior=None) -> tuple:
    """1-sigma (p0, w, t0) from the chi2 curvature, scaled by sqrt(chi2/dof) so
    that model imperfection is absorbed rather than ignored.

    Each error is a 1-D curvature at the minimum: the p0-t0 correlation (the
    slide-along-the-track degeneracy, doc §20) is not propagated, so p0_err is
    ~20 % optimistic (measured pull widths 1.19/1.13)."""
    W, noise, pos, sat = wm.prep_plane(P, plane)

    def chi(p0v, wv, t0v):
        return wm.chi2_plane(plane, W, noise, pos, sat, p0v, wv, t0v,
                             hyper, snap_t0=False, t0_prior=t0_prior)[0]

    try:
        c0 = r['chi2']
        p0, w, t0 = r['p0'], r['w'], r['t0']
        d2p = (chi(p0 + dp, w, t0) - 2 * c0 + chi(p0 - dp, w, t0)) / dp ** 2
        d2w = (chi(p0, w + dw, t0) - 2 * c0 + chi(p0, w - dw, t0)) / dw ** 2
        d2t = (chi(p0, w, t0 + dt) - 2 * c0 + chi(p0, w, t0 - dt)) / dt ** 2
        scale = max(r['chi2'] / max(r['dof'], 1), 1.0)
        ep = float(np.sqrt(2 * scale / d2p)) if d2p > 0 else np.nan
        ew = float(np.sqrt(2 * scale / d2w)) if d2w > 0 else np.nan
        et = float(np.sqrt(2 * scale / d2t)) if d2t > 0 else np.nan
        return ep, ew, et
    except Exception:
        return np.nan, np.nan, np.nan


def _global_start(P, plane, p0_seed, t0_seed, hyper, t0_prior=None):
    """Reference-free global search for the fit's starting point.

    The R&D fits were seeded at the M3 reference (position AND angle), which is
    not available in production and would make alignment/efficiency circular.
    Seeding instead from the brightest strip and a local search lands in the
    wrong basin for ~17 % of planes (measured against the reference-seeded fits:
    those failures sit at 5 deg error with a *higher* chi2, i.e. genuinely
    missed minima, not disagreements). The production hit ladder is no help
    either — it is compressed ~40 %, so an inclined track's seed starts outside
    the basin.

    So: scan. (p0, t0) first at zero slope, then (p0, w) at the best t0. ~310
    chi2 evaluations, all of them NNLS-profiled, ~0.4 s.
    """
    W, noise, pos, sat = wm.prep_plane(P, plane)

    def chi(p0, w, t0):
        return wm.chi2_plane(plane, W, noise, pos, sat, p0, w, t0, hyper,
                             t0_prior=t0_prior)[0]

    # charge-weighted centre of the window as the p0 scan centre
    amp = np.maximum(W.max(axis=1), 0.0)
    p_c = float((pos * amp).sum() / amp.sum()) if amp.sum() > 0 else p0_seed
    p0s = p_c + np.arange(-P0_SCAN_HALF, P0_SCAN_HALF + 1e-9, P0_SCAN_STEP)
    if t0_prior is not None:
        # the external clock collapses the t0 axis of the scan (T1.1): one
        # point at the prediction instead of 7. The stage-2 (p0, w) scan and
        # the Nelder-Mead refinement still see t0 through the penalty.
        t0s = np.array([float(t0_prior[0])])
        t0_seed = float(t0_prior[0])
    else:
        t0s = np.arange(t0_seed - T0_SCAN_HALF, t0_seed + T0_SCAN_HALF + 1e-9,
                        T0_SCAN_STEP)

    best = (np.inf, p0_seed, t0_seed)
    for t0 in t0s:
        for p0 in p0s:
            c = chi(p0, 0.0, t0)
            if c < best[0]:
                best = (c, float(p0), float(t0))
    t0b = best[2]

    ws = np.arange(-W_SCAN_HALF, W_SCAN_HALF + 1e-9, W_SCAN_STEP)
    # centroid-to-anchor lever arm [ns]: True = half the drift column
    # (the measured value, see 12_shear_lever.py); a number = explicit ns
    if P0_SHEAR and wm.CAL is not None:
        shear = (15000.0 / wm.CAL.v_drift if P0_SHEAR is True
                 else float(P0_SHEAR))
    else:
        shear = 0.0
    best2 = (np.inf, best[1], 0.0)
    for p0 in p0s:
        for w in ws:
            p0m = p0 - w * shear
            c = chi(p0m, w, t0b)
            if c < best2[0]:
                best2 = (c, float(p0m), float(w))
    return best2[1], best2[2], t0b


def _two_sided(P, plane, r, cal: CalibrationBundle, hyper, t0_prior=None) -> dict:
    """Re-search each slope sign separately from the one-sided fit ``r`` and
    return the lowest-chi2 fit_plane_raw result (``r`` itself if neither side
    beats it). See TWO_SIDED."""
    W, noise, pos, sat = wm.prep_plane(P, plane)
    v, kw, w0 = cal.v_drift, cal.kw.get(plane, 1.0), cal.w0.get(plane, 0.0)

    def to_w(tan):
        return (tan * kw * v + w0) * 1e-3

    def chi(p0, w, t0):
        return wm.chi2_plane(plane, W, noise, pos, sat, p0, w, t0, hyper,
                             t0_prior=t0_prior)[0]

    shear = 15000.0 / v                     # half the drift column [ns]
    amp = np.maximum(W.max(axis=1), 0.0)
    p_c = float((pos * amp).sum() / amp.sum()) if amp.sum() > 0 else float(r['p0'])
    p0s = p_c + np.arange(-TS_P0_HALF, TS_P0_HALF + 1e-9, TS_P0_STEP)
    t0s = (np.array([float(t0_prior[0])]) if t0_prior is not None else
           np.arange(r['t0'] - TS_T0_HALF, r['t0'] + TS_T0_HALF + 1e-9, TS_T0_STEP))
    tans = np.arange(TS_TAN_LO, TS_TAN_HI + 1e-9, TS_TAN_STEP)
    best = r
    for sgn in (1.0, -1.0):
        ws_probe = [to_w(sgn * t) for t in TS_PROBE_TAN]
        c1 = min((chi(p - w * shear, w, t0), t0)
                 for t0 in t0s for p in p0s for w in ws_probe)
        t0b = c1[1]
        c2 = min((chi(p - w * shear, w, t0b), p - w * shear, w)
                 for p in p0s for w in (to_w(sgn * t) for t in tans))
        rs = wm.fit_plane_raw(P, plane, c2[1], c2[2], t0b, hyper=hyper,
                              t0_prior=t0_prior)
        if rs is not None and np.isfinite(rs['chi2']) and rs['chi2'] < best['chi2']:
            best = rs
    return best


def t0_prior_for(cal: CalibrationBundle, plane: str, ftst) -> Optional[tuple]:
    """(t0_pred, sigma) for one plane of one event, or None if the prior is
    not calibrated/enabled. t0_pred is the bundle's per-ftst-class prediction
    (the trigger is the muon; ftst is its phase against the DREAM clock);
    sigma is the bundle's, overridable via the module global T0_PRIOR_SIGMA."""
    sig = T0_PRIOR_SIGMA if T0_PRIOR_SIGMA is not None else \
        (cal.t0_prior_sigma or None)
    t0a = (T0_ABS or getattr(cal, 't0_abs', None) or {}).get(plane)
    if not sig or not t0a or ftst is None:
        return None
    pred = t0a.get(int(ftst))
    if pred is None:
        return None
    return float(pred), float(sig)


def fit_plane(P, plane: str, cal: CalibrationBundle, hyper: Optional[dict] = None,
              n_seed: int = 0, n_dropped: int = 0,
              t0_prior: Optional[tuple] = None) -> Optional[PlaneFit]:
    """Fit one plane's window. P: dict/PlaneWindow-like with W, pos, noise, ch.
    ``t0_prior=(t0_pred, sigma)``: external-clock t0 penalty (see t0_prior_for)."""
    hyper = hyper or cal.hyper
    W = np.asarray(P['W'])
    if W.shape[1] != wm.NSAMP:
        wm.set_nsamp(W.shape[1])
    p0_seed, _w0, t0_seed = wm.init_guess(P, plane)
    p0_seed, w_seed, t0_seed = _global_start(P, plane, p0_seed, t0_seed, hyper,
                                             t0_prior=t0_prior)
    r = wm.fit_plane_raw(P, plane, p0_seed, w_seed, t0_seed, hyper=hyper,
                         t0_prior=t0_prior)
    if r is None or not np.isfinite(r['chi2']):
        return None
    if TWO_SIDED:
        r = _two_sided(P, plane, r, cal, hyper, t0_prior=t0_prior)
    # Per-plane angle mapping (9dd7d6e; reverted by f9e18d2, restored 8-13).
    # w0/kw are measured from free fits of reference tracks; dropping the w0
    # term is the fleet angle bias, arctan(w0_plane/v) detector by detector.
    tan = ((r['w'] * 1e3 - cal.w0.get(plane, 0.0))
           / (cal.kw.get(plane, 1.0) * cal.v_drift))
    ep, ew, et = _errors(P, plane, r, hyper, t0_prior=t0_prior)
    q_sum, q_u50, q_u90, q_uend = _profile_summary(r['q'])
    ch = np.asarray(P['ch'], dtype=int)
    flagged = np.concatenate([wm.DEAD.get(plane, np.array([], dtype=int)),
                              wm.HOT.get(plane, np.array([], dtype=int))])
    n_flagged = int(np.isin(ch, flagged).sum()) if len(flagged) else 0
    return PlaneFit(
        p0=float(r['p0']), w=float(r['w']), t0=float(r['t0']),
        tan_theta=float(tan), theta_deg=float(np.degrees(np.arctan(tan))),
        chi2=float(r['chi2']), dof=int(r['dof']),
        p0_err=float(np.hypot(ep, FLOOR_P0_MM)) if np.isfinite(ep) else FLOOR_P0_MM,
        w_err=float(ew) if np.isfinite(ew) else np.nan,
        tan_err=float(np.hypot(ew * 1e3 / cal.v_drift, FLOOR_TAN))
        if np.isfinite(ew) else FLOOR_TAN,
        t0_err=float(et) if np.isfinite(et) else np.nan,
        q_sum=q_sum, q_u50=q_u50, q_u90=q_u90, q_uend=q_uend,
        n_strips=int(W.shape[0]), n_seed=int(n_seed), n_dropped=int(n_dropped),
        slope_reliable=bool(abs(tan) >= TAN_MIN_SLOPE),
        quality_ok=bool(r['chi2'] / max(r['dof'], 1) < CHI2DOF_BAD),
        n_flagged_strips=n_flagged)


# --- candidate-cluster selection -------------------------------------------
# A track's charge column crosses the drift gap, so it lasts a few hundred ns
# and its transverse speed is bounded. Coherent noise and stray deposits do not
# satisfy both. Among candidates that do, take the one whose charge the model
# explains best (chi2 improvement over "no signal").
U_MIN_NS = 250.0
U_MAX_NS = 1100.0
# In the bundle's RAW tan (true = raw x k), so its true-angle reach moves with
# the bundle's v.  WFT_TAN_MAX overrides it (read at import, like
# WFT_BEAM_MIN_STRIPS): a versioned re-pass sets it wide and applies the
# angular acceptance as a true-angle cut at stage 3 (build_tracks --tan-max-true).
TAN_MAX = float(os.environ.get('WFT_TAN_MAX', 0.6))


def _candidate_score(P, plane, fit: PlaneFit) -> tuple:
    """(plausible, dchi2) for one candidate cluster's fit."""
    W, noise, pos, sat = wm.prep_plane(P, plane)
    chi_null = float(((W / noise[:, None]) ** 2)[~sat].sum())
    u = fit.q_uend
    plausible = (np.isfinite(u) and U_MIN_NS <= u <= U_MAX_NS
                 and abs(fit.tan_theta) < TAN_MAX)
    return bool(plausible), float(chi_null - fit.chi2)


def fit_plane_candidates(windows: list, plane: str, cal: CalibrationBundle,
                         seeds: Optional[list] = None, return_all: bool = False,
                         t0_prior: Optional[tuple] = None):
    """Fit every candidate cluster of one plane and keep the muon's.

    'Largest cluster wins' is wrong for ~5 % of events, and when it is wrong the
    true track is a median 37 mm outside the fit window — so the failures are
    catastrophic, not marginal. Measured on those failures (det3, 224 events):

        rule                        median |p0 - ref|   within 5 mm
        most strips (old)                 76.6 mm            19 %
        most charge                       47.3 mm            32 %
        best chi2 improvement              3.4 mm            51 %
        plausible + best improvement       1.6 mm            55 %
        (best available candidate)         0.4 mm            95 %

    The right cluster is nearly always among the candidates; this rule finds it
    half the time, which is a large net gain and still leaves headroom.
    """
    best = None
    best_key = None
    n_ok = 0
    ranked = []
    for i, P in enumerate(windows):
        s = (seeds or [None] * len(windows))[i]
        try:
            fit = fit_plane(P, plane, cal,
                            n_seed=getattr(s, 'n_strips', 0) if s else 0,
                            n_dropped=getattr(s, 'n_dropped', 0) if s else 0,
                            t0_prior=t0_prior)
        except Exception:
            fit = None
        if fit is None:
            continue
        n_ok += 1
        plausible, dchi2 = _candidate_score(P, plane, fit)
        fit._plausible, fit._dchi2 = plausible, dchi2
        fit._rescued = bool(getattr(s, 'rescued', False))
        key = (1 if plausible else 0, dchi2)
        ranked.append((key, fit))
        if best_key is None or key > best_key:
            best, best_key = fit, key
    if best is not None:
        best.n_candidates = n_ok
    ranked.sort(key=lambda kv: kv[0], reverse=True)
    for _k, f in ranked:
        f.n_candidates = n_ok
    return (best, [f for _k, f in ranked]) if return_all else best


DT_XY_TOL_NS = 120.0     # how far t0x - t0y may sit from the measured offset


def select_tracks(cand_fits: Dict[str, list], ftst_diff: Optional[int],
                  cal: CalibrationBundle, max_tracks: int = 3,
                  pairing: Optional[dict] = None) -> list:
    """Disjoint time-coincident (x, y) candidate pairs, ranked — the
    multi-track generalisation of :func:`select_pair`.

    ``select_pair`` answers "which single pair is the muon"; this answers "how
    many track-like pairs does the event contain". Pair 0 is select_pair's
    choice (same key, same maximum, kept even when it fails the gate, so the
    single-track answer is unchanged). Every FURTHER pair must earn its place:
    time-coincident AND both members plausible. That gate is the
    double-counting guard — one track split into two clusters (a dead region,
    a delta ray) yields a second pair that is time-coincident with the first
    by construction, but its fragments rarely both pass the column-duration
    plausibility window, and a noise cluster has no reason to be coincident
    at all.

    Candidates flagged ``_rescued`` (seeded only by a rescue-mode local floor,
    ``wft.seed.SIG_FLOOR_LOCAL_MODE``) rank below every production combination,
    so pair 0 is select_pair's choice among the production candidates and a
    rescued candidate can only add a further track, never replace one.

    ``pairing`` (a bundle's ``xy_pairing``) re-assigns y partners among gated
    tracks that are time-degenerate. Summed dchi2 alone pairs the strongest x
    with the strongest y, which is wrong whenever the two planes rank the
    tracks differently (sept26_prelim_analysis/intra_bench.py). Only y members
    move: every x member, the track order, every gate decision and any event
    with a single gated track are unchanged.

    Both are described, with their validation, in wft/MULTITRACK_2026-09-14.md.

    Returns ``[(ix, iy, gated)]`` indices into ``cand_fits['x']/['y']``;
    the event's track count is ``sum(gated)``.
    """
    dt = cal.dt_xy.get(int(ftst_diff), -18.8) if ftst_diff is not None else -18.8
    combos = []
    for i, fx in enumerate(cand_fits.get('x') or []):
        for j, fy in enumerate(cand_fits.get('y') or []):
            if fx is None or fy is None:
                continue
            coincident = int(abs((fx.t0 - fy.t0) - dt) <= DT_XY_TOL_NS)
            plaus = (int(getattr(fx, '_plausible', True))
                     + int(getattr(fy, '_plausible', True)))
            dchi2 = (getattr(fx, '_dchi2', 0.0) or 0.0) + \
                (getattr(fy, '_dchi2', 0.0) or 0.0)
            # a combination using a rescued candidate ranks below every
            # production one, so rescue can add tracks but never replace one
            prod = int(not (getattr(fx, '_rescued', False) or getattr(fy, '_rescued', False)))
            combos.append(((prod, coincident, plaus, dchi2), i, j))
    # stable sort on the key alone: ties keep x-major order, which is the
    # combo select_pair's strict > would have kept
    combos.sort(key=lambda c: c[0], reverse=True)
    used_x, used_y, out = set(), set(), []
    for key, i, j in combos:
        if i in used_x or j in used_y:
            continue
        gated = key[1] == 1 and key[2] == 2
        if out and not gated:
            continue
        out.append((i, j, gated))
        used_x.add(i)
        used_y.add(j)
        if len(out) >= max_tracks:
            break
    if pairing and sum(1 for *_ij, g in out if g) >= 2:
        out = _repair_pairs(out, cand_fits, pairing, dt)
    return out


def _gate(fx, fy, dt: float) -> bool:
    return (abs((fx.t0 - fy.t0) - dt) <= DT_XY_TOL_NS
            and bool(getattr(fx, '_plausible', True))
            and bool(getattr(fy, '_plausible', True)))


def xy_pair_cost(fx, fy, pairing: dict, dt: float) -> float:
    """How unlike an x and a y candidate are, in units of the spread of the
    same x-minus-y quantity on clean single tracks."""
    f = dict(lq=np.log(max(fx.q_sum, 1.0) / max(fy.q_sum, 1.0)),
             u50=fx.q_u50 - fy.q_u50, u90=fx.q_u90 - fy.q_u90,
             t0=(fx.t0 - fy.t0) - dt)
    cost = 0.0
    for k in pairing['features']:
        z = (f[k] - pairing['median'][k]) / pairing['rsig'][k]
        cost += min(z * z, 25.0) if np.isfinite(z) else 25.0
    return cost


def _repair_pairs(out: list, cand_fits: Dict[str, list], pairing: dict, dt: float) -> list:
    """Swap the y partners of two gated tracks when both swapped combinations
    also pass the gate and cost less."""
    xs, ys = cand_fits['x'], cand_fits['y']
    out = list(out)
    for _ in range(len(out)):
        changed = False
        for a in range(len(out)):
            for b in range(a + 1, len(out)):
                ia, ja, ga = out[a]
                ib, jb, gb = out[b]
                if not (ga and gb and _gate(xs[ia], ys[jb], dt) and _gate(xs[ib], ys[ja], dt)):
                    continue
                keep = xy_pair_cost(xs[ia], ys[ja], pairing, dt) + xy_pair_cost(xs[ib], ys[jb], pairing, dt)
                swap = xy_pair_cost(xs[ia], ys[jb], pairing, dt) + xy_pair_cost(xs[ib], ys[ja], pairing, dt)
                if swap < keep:
                    out[a], out[b] = (ia, jb, True), (ib, ja, True)
                    changed = True
        if not changed:
            break
    return out


def load_pairing(path: str) -> dict:
    with open(path) as f:
        p = json.load(f)
    check_xy_pairing(p, where=path)
    return p


def candidate_rows(event_id: int, all_fits: Dict[str, list],
                   pairs: Optional[list] = None,
                   ftst: Optional[dict] = None) -> list:
    """One dict per fitted candidate cluster — the full ranked list that
    :func:`row_from_fits` reduces to a single winner. ``pairs`` (from
    :func:`select_tracks`) stamps each candidate with the track it belongs
    to; ``track_id`` -1 = not part of any selected pair."""
    track_of = {}
    for tid, (ix, iy, gated) in enumerate(pairs or []):
        track_of[('x', ix)] = (tid, gated)
        track_of[('y', iy)] = (tid, gated)
    rows = []
    for plane in ('x', 'y'):
        f_ftst = (ftst or {}).get(plane)
        for rank, f in enumerate(all_fits.get(plane) or []):
            if f is None:
                continue
            row = {'event_id': int(event_id), 'plane': plane, 'rank': int(rank)}
            row.update(asdict(f))
            row['plausible'] = bool(getattr(f, '_plausible', True))
            row['dchi2'] = float(getattr(f, '_dchi2', np.nan))
            row['rescued'] = bool(getattr(f, '_rescued', False))
            tid, gated = track_of.get((plane, rank), (-1, False))
            row['track_id'], row['track_gated'] = int(tid), bool(gated)
            row['isochronous'] = bool(np.isfinite(f.q_uend)
                                      and f.q_uend < U_MIN_NS)
            row['ftst'] = int(f_ftst) if f_ftst is not None else -1
            rows.append(row)
    return rows


def select_pair(cand_fits: Dict[str, list], ftst_diff: Optional[int],
                cal: CalibrationBundle) -> Dict[str, Optional[PlaneFit]]:
    """Choose one cluster per plane using the fact that the muon fired both.

    A muon's X and Y charge arrives at the same time, so ``t0x - t0y`` must sit
    at the measured FEU offset (``dt_xy``, keyed by the ftst difference).
    Coherent noise or a stray deposit in one plane has no reason to be
    time-coincident with the track in the other, which is information that
    single-plane selection cannot use.

    Falls back to the per-plane rule when a plane has only one candidate.
    """
    out = {}
    for plane in ('x', 'y'):
        fits = [f for f in cand_fits.get(plane, []) if f is not None]
        out[plane] = fits[0] if fits else None
    if not (len(cand_fits.get('x', [])) > 1 or len(cand_fits.get('y', [])) > 1):
        return out
    dt = cal.dt_xy.get(int(ftst_diff), -18.8) if ftst_diff is not None else -18.8
    best, best_key = None, None
    for fx in cand_fits.get('x', []) or [None]:
        for fy in cand_fits.get('y', []) or [None]:
            if fx is None or fy is None:
                continue
            coincident = abs((fx.t0 - fy.t0) - dt) <= DT_XY_TOL_NS
            plaus = int(getattr(fx, '_plausible', True)) + int(getattr(fy, '_plausible', True))
            key = (int(coincident), plaus,
                   getattr(fx, '_dchi2', 0.0) + getattr(fy, '_dchi2', 0.0))
            if best_key is None or key > best_key:
                best, best_key = (fx, fy), key
    if best is not None:
        out['x'], out['y'] = best
    return out


def fit_event(windows: Dict[str, object], cal: CalibrationBundle,
              seeds: Optional[dict] = None) -> Dict[str, Optional[PlaneFit]]:
    out = {}
    for plane in ('x', 'y'):
        P = windows.get(plane)
        if P is None:
            out[plane] = None
            continue
        s = (seeds or {}).get(plane)
        out[plane] = fit_plane(P, plane, cal,
                               n_seed=getattr(s, 'n_strips', 0) if s else 0,
                               n_dropped=getattr(s, 'n_dropped', 0) if s else 0)
    return out


def row_from_fits(event_id: int, fits: Dict[str, Optional[PlaneFit]],
                  n_hits: int = 0, spark: bool = False) -> dict:
    row = {'event_id': int(event_id), 'n_hits': int(n_hits), 'spark': bool(spark)}
    for plane in ('x', 'y'):
        f = fits.get(plane)
        if f is None:
            row[f'{plane}_ok'] = False
            for k in ('p0', 'w', 't0', 'tan_theta', 'theta_deg', 'chi2',
                      'p0_err', 'w_err', 'tan_err', 't0_err', 'q_sum', 'q_u50',
                      'q_u90', 'q_uend'):
                row[f'{plane}_{k}'] = np.nan
            for k in ('dof', 'n_strips', 'n_seed', 'n_dropped', 'n_candidates',
                     'n_flagged_strips'):
                row[f'{plane}_{k}'] = 0
            row[f'{plane}_slope_reliable'] = False
            row[f'{plane}_quality_ok'] = False
            row[f'{plane}_isochronous'] = False
        else:
            row[f'{plane}_ok'] = True
            for k, v in asdict(f).items():
                row[f'{plane}_{k}'] = v
            # F32: charge arriving in ≲2 depth bins is a flash/discharge
            # signature, not a track (a vertical muon still fills the gap in
            # TIME). Computed for candidate ranking since day one but never
            # written out — this makes it cuttable downstream.
            row[f'{plane}_isochronous'] = bool(
                np.isfinite(f.q_uend) and f.q_uend < U_MIN_NS)
    return row


# --------------------------------------------------------------- the driver
def _worker_init(bundle_path, pairing_path=None):
    """``pairing_path`` overrides the bundle's ``xy_pairing`` (for A/B runs)."""
    global _CAL, _PAIRING
    _CAL = CalibrationBundle.load(bundle_path)
    wm.use_calibration(_CAL)
    _PAIRING = load_pairing(pairing_path) if pairing_path else (_CAL.xy_pairing or None)


PAIR_SELECT = os.environ.get('WFT_PAIR_SELECT', '0') == '1'
EMIT_CANDIDATES = os.environ.get('WFT_EMIT_CANDIDATES', '1') == '1'
MAX_TRACKS = 3
_PAIRING: Optional[dict] = None


def _worker_fit(payload):
    eid, wins, seeds, n_hits, spark, ftst = payload
    # older beam drivers passed a scalar ftst_diff here; treat anything that
    # is not the per-plane dict as absent rather than dying inside the try
    ftst = ftst if isinstance(ftst, dict) else {}
    fits, all_fits = {}, {}
    for plane in ('x', 'y'):
        cand = wins.get(plane)
        if not cand:
            fits[plane], all_fits[plane] = None, []
            continue
        try:
            best, ranked = fit_plane_candidates(
                cand, plane, _CAL, seeds=seeds.get(plane), return_all=True,
                t0_prior=t0_prior_for(_CAL, plane, ftst.get(plane)))
            fits[plane], all_fits[plane] = best, ranked
        except Exception:
            fits[plane], all_fits[plane] = None, []
    ftst_diff = (ftst['x'] - ftst['y']
                 if ftst.get('x') is not None and ftst.get('y') is not None
                 else None)
    if PAIR_SELECT:
        try:
            fits = select_pair(all_fits, ftst_diff, _CAL)
        except Exception:
            pass
    try:
        pairs = select_tracks(all_fits, ftst_diff, _CAL, max_tracks=MAX_TRACKS,
                              pairing=_PAIRING)
    except Exception:
        pairs = []
    row = row_from_fits(eid, fits, n_hits, spark)
    row['n_tracks'] = int(sum(1 for _i, _j, g in pairs if g))
    for plane in ('x', 'y'):
        f = ftst.get(plane)
        row[f'{plane}_ftst'] = int(f) if f is not None else -1
    if EMIT_CANDIDATES:
        row['_cand'] = candidate_rows(eid, all_fits, pairs, ftst)
    return row


def reconstruct_run(cfg, cal: CalibrationBundle, out_path: str,
                    event_filter: Optional[set] = None, jobs: int = 12,
                    limit: Optional[int] = None, pad_strips: int = 3,
                    bundle_path: Optional[str] = None, verbose: bool = True):
    """Reconstruct one run/subrun into a parquet table.

    cfg           : qa_config run config (paths, FEUs, detector name)
    event_filter  : if given, only these event ids (e.g. events with an M3 ray)
    """
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    from . import io as wio
    from . import seed as wseed

    if bundle_path is None:
        bundle_path = os.path.join(os.path.dirname(out_path), 'calib_bundle')
        cal.save(bundle_path)

    pos_maps = wio.strip_position_map(cfg)
    feu_x, feu_y = cfg.MX17_FEU_X, cfg.MX17_FEU_Y

    if verbose:
        print(f'[wft] {cal.summary()}')
        print(f'[wft] hits -> seeds ...', flush=True)
    hits = _load_hits(cfg)
    seeds = wseed.seeds_from_hits(hits, pos_maps, feu_x, feu_y, hot=cal.hot)
    del hits
    wanted = set(seeds)
    if event_filter is not None:
        wanted &= set(int(e) for e in event_filter)
    wanted = {e for e in wanted
              if not seeds[e]['spark'] and (seeds[e]['x'] or seeds[e]['y'])}
    if limit:
        wanted = set(sorted(wanted)[:limit])
    if verbose:
        print(f'[wft] {len(seeds):,} seeded events, {len(wanted):,} to reconstruct',
              flush=True)

    rows, cand_rows = [], []
    with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init,
                             initargs=(bundle_path,)) as pool:
        for payloads in _stream_windows(cfg, pos_maps, seeds, wanted, pad_strips,
                                        verbose=verbose):
            if not payloads:
                continue
            for r in pool.map(_worker_fit, payloads, chunksize=8):
                cand_rows.extend(r.pop('_cand', []))
                rows.append(r)
            if verbose:
                print(f'[wft]   {len(rows):,} events reconstructed', flush=True)

    df = pd.DataFrame(rows).sort_values('event_id').reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    cand_path = out_path.replace('.parquet', '.candidates.parquet')
    if cand_rows:
        pd.DataFrame(cand_rows).sort_values(
            ['event_id', 'plane', 'rank']).reset_index(drop=True).to_parquet(
            cand_path, index=False)
    meta = dict(n_events=len(df), calibration=bundle_path,
                bundle=dict(detector=cal.detector, run_key=cal.run_key,
                            v_drift=cal.v_drift, hyper=cal.hyper,
                            conditions=cal.conditions,
                            provenance=cal.provenance),
                # Which angle mapping this table is on. A bundle that lost its
                # constants still reconstructs, but on the UNCORRECTED mapping,
                # and nothing downstream could tell -- that is how the fleet
                # bias survived a whole campaign. applied=False here means the
                # angles need the post-hoc w0/kw pass before they are quoted.
                angle_constants=dict(applied=bool(cal.w0 or cal.kw),
                                     w0=dict(cal.w0), kw=dict(cal.kw)),
                t0_prior=dict(sigma=T0_PRIOR_SIGMA if T0_PRIOR_SIGMA is not None
                              else cal.t0_prior_sigma,
                              t0_abs_planes=sorted((T0_ABS or cal.t0_abs
                                                    or {}).keys())),
                run=dict(key=getattr(cfg, 'KEY', ''), run=cfg.RUN,
                         sub_run=cfg.SUB_RUN, detector=cfg.DET_NAME,
                         feu_x=feu_x, feu_y=feu_y),
                selection=dict(sig_rel_floor=wseed.SIG_REL_FLOOR,
                               gap_mm=wseed.GAP_THRESHOLD_MM,
                               sig_floor_local_mm=wseed.SIG_FLOOR_LOCAL_MM,
                               spark_veto=wseed.SPARK_VETO_HITS,
                               pad_strips=pad_strips,
                               event_filter=bool(event_filter)),
                multi_track=dict(emit_candidates=EMIT_CANDIDATES,
                                 xy_pairing=(cal.xy_pairing or {}).get('features'),
                                 max_tracks=MAX_TRACKS,
                                 n_candidate_rows=len(cand_rows),
                                 n_events_multitrack=int(
                                     (df['n_tracks'] >= 2).sum())
                                 if 'n_tracks' in df else 0))
    with open(out_path.replace('.parquet', '.meta.json'), 'w') as f:
        json.dump(meta, f, indent=1, default=str)
    if verbose:
        print(f'[wft] wrote {out_path} ({len(df):,} events)')
    return df


def _load_hits(cfg):
    """Combined hits for the run — used ONLY for seeding (see wft.seed)."""
    import uproot
    import pandas as pd
    files = [f for f in os.listdir(cfg.combined_hits_dir)
             if f.endswith('.root') and '_datrun_' in f]
    df = uproot.concatenate(
        [f'{cfg.combined_hits_dir}{f}:hits' for f in files],
        expressions=['eventId', 'feu', 'channel', 'amplitude', 'significance'],
        library='pd')
    return df[df['feu'].isin(cfg.MX17_FEUS)]


def _stream_windows(cfg, pos_maps, seeds, wanted, pad_strips, verbose=True):
    """Yield lists of (eid, windows, seedinfo, n_hits, spark) per file pair."""
    from . import io as wio
    fx = wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, cfg.MX17_FEU_X)
    fy = wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, cfg.MX17_FEU_Y)
    by_tag = {}
    for f in fx:
        by_tag.setdefault(wio.file_tag(f), {})['x'] = f
    for f in fy:
        by_tag.setdefault(wio.file_tag(f), {})['y'] = f
    for tag in sorted(by_tag):
        pair = by_tag[tag]
        if 'x' not in pair or 'y' not in pair:
            if verbose:
                print(f'[wft]   {tag}: missing a plane, skipped')
            continue
        rx = wio.FeuReader(pair['x'])
        ry = wio.FeuReader(pair['y'])
        want = wanted & (set(rx.event_ids.tolist()) | set(ry.event_ids.tolist()))
        if not want:
            continue
        buf = {}
        for plane, rdr, feu in (('x', rx, cfg.MX17_FEU_X), ('y', ry, cfg.MX17_FEU_Y)):
            for eid, ftst, wfm in rdr.iter_events(want):
                cl = seeds[eid][plane]
                if not cl:
                    continue
                cl = cl if isinstance(cl, list) else [cl]
                wins, used = [], []
                for s in cl:
                    win = wio.extract_window(wfm, rdr.noise, pos_maps[feu],
                                             s.channels, pad_strips)
                    if win is None:
                        continue
                    wins.append(dict(W=win.W, pos=win.pos, noise=win.noise,
                                     ch=win.ch))
                    used.append(s)
                if not wins:
                    continue
                rec = buf.setdefault(eid, {'w': {}, 's': {}})
                rec['w'][plane] = wins
                rec['s'][plane] = used
                rec['ftst_' + plane] = ftst
        payloads = []
        for eid, rec in buf.items():
            ftst = {p: rec.get('ftst_' + p) for p in ('x', 'y')}
            payloads.append((eid, rec['w'], rec['s'], seeds[eid]['n_hits'],
                             seeds[eid]['spark'], ftst))
        if verbose:
            print(f'[wft]   {tag}: {len(payloads):,} events windowed', flush=True)
        yield payloads
