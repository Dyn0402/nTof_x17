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
    f = PlaneFit(
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
    f._q = np.asarray(r['q'], float)     # the depth profile, for x/y pairing
    return f


# --- candidate-cluster selection -------------------------------------------
# A track's charge column crosses the drift gap, so it lasts a few hundred ns
# and its transverse speed is bounded. Coherent noise and stray deposits do not
# satisfy both. Among candidates that do, take the one whose charge the model
# explains best (chi2 improvement over "no signal").
U_MIN_NS = 250.0
U_MAX_NS = 1100.0
TAN_MAX = 0.6


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
        fit._win = i                      # which window this fit came from
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


# ======================================================================== #
# The joint two-track fit
#
# Two tracks closer than the 12 mm seed gap share one cluster, so they share
# one window and are fitted as one compromise line. Below 12 mm that costs
# ~100 % of pairs and at 12-24 mm 62-65 % (intra_bench); small-opening-angle
# pairs -- conversions, low-angle IPC -- are exactly the population it eats.
#
# The fix is not to split the seed (that was tried and it breaks real single
# tracks: wft/MULTITRACK_2026-09-14.md §3.2) but to offer the window a second
# track and let chi2 decide. Design: HANDOFF_JOINT_TWO_TRACK_FIT.md.
#
# CONTRACT. A split REPLACES its parent -- it cannot be additive the way the
# rescue floor is, because the two children occupy the parent's strips and
# counting both would count the charge twice. So:
#   * with WFT_TWO_TRACK_FIT off (the default) the output is bit-identical;
#   * a candidate whose split is not accepted comes out bit-identical;
#   * the parent is kept in the candidates side table flagged ``split_replaced``
#     so downstream can undo any split;
#   * the false-split rate on clean single tracks is measured and bounded.
# ======================================================================== #

#: Master switch. Off = production, candidate for candidate.
TWO_TRACK = os.environ.get('WFT_TWO_TRACK_FIT', '0') == '1'
#: Accept the split when the statistic (``fit_plane_two``'s ``fstat``: the
#: SMALLER of the two children's marginal chi2 improvements, in units of the
#: one-track fit's own chi2/dof) exceeds this. NOT a Wilks number: chi2/dof is
#: 1.4-6.8 on clean single tracks here and 5-12 across the full pass, so a
#: threshold from the asymptotic distribution would split almost everything.
#: Chosen from the efficiency-vs-false-split curve of
#: sept26_prelim_analysis/two_track_synth.py and `intra_bench split-probe`.
TWO_TRACK_F = float(os.environ.get('WFT_TWO_TRACK_F', '300.0'))
#: The same threshold when the other plane resolves two time-coincident
#: plausible candidates where THIS plane resolves fewer (see
#: :func:`_cross_plane_mismatch`). Two tracks must show in both views, so a
#: count mismatch is real external evidence that this plane has merged them and
#: the bar can be lower. Without the mismatch half of that condition the
#: discount fires on the easy case and destroys correct tracks.
TWO_TRACK_F_CORROB = float(os.environ.get('WFT_TWO_TRACK_F_CORROB', '120.0'))
#: Candidates per plane the fit is attempted on, best-ranked first.
TWO_TRACK_MAX_TRY = 2
#: Trigger: the window's charge is wider than one track of the fitted |tan| and
#: column duration can cover, by more than this margin [mm]. Measured on clean
#: single tracks -- `intra_bench split-probe`.
TWO_TRACK_WIDTH_MARGIN = float(os.environ.get('WFT_TWO_TRACK_WIDTH_MM', '6.0'))
#: Strip significance that counts as carrying charge, for the width trigger.
TWO_TRACK_SIG = 5.0
#: Do the two tracks share a t0?
#:
#:   'tied' (default) -- yes, one t0 for both. Two prompt tracks from one vertex
#:       reach the mesh together to within a few ns, and that is the hypothesis
#:       the same-chamber vertex test is about. It also removes a degeneracy
#:       that costs more than it buys: the plane chi2 has near-degenerate minima
#:       one depth bin apart (chi2_plane's docstring -- only ~35 % of free fits
#:       land in the physical one), and with a free second t0 the fit walks into
#:       them. Measured on synthetics 2026-09-16: free t0 splits perfectly
#:       modelled SINGLE tracks at 200 ns offsets and reaches fstat 100+.
#:   'free' -- also offer a free second t0. A STUDY OPTION, not a fallback:
#:       nothing measured so far gives it an advantage. Two co-located parallel
#:       tracks are degenerate at any offset (the profile absorbs it), and on a
#:       time-offset pair with opposite slopes the TIED fit scores higher
#:       (test_two_track.test_free_t0_mode_reaches_a_time_offset_pair).
TWO_TRACK_T0 = os.environ.get('WFT_TWO_TRACK_T0', 'tied')
#: t0 offsets, relative to the parent's, at which the residual is scanned for a
#: second track [ns]. One depth bin apart; wider only when t0 is free.
T0_RESID_SCAN = np.arange(-60.0, 60.1, 60.0)
T0_RESID_SCAN_FREE = np.arange(-180.0, 180.1, 60.0)
#: Trigger: the largest per-strip coherent positive residual the one-track fit
#: leaves, in sigma after de-scaling by the fit's own chi2/dof. Calibrated on
#: clean single tracks -- `intra_bench split-probe`.
TWO_TRACK_RESID_Z = float(os.environ.get('WFT_TWO_TRACK_RESID_Z', '8.0'))
#: Only reconsider candidates the selector actually chose (members of a pair
#: from :func:`select_tracks`). Measured on 13 071 real run_145 candidates:
#: this is 25 % of them and holds 42 % of the splits, the rest being extra
#: clusters in already-busy events -- four candidates a plane, 45 strips,
#: chi2/dof 13. The default was set to save compute, which is no longer a
#: constraint (2026-09-16): whether those other splits are real pairs is
#: unmeasured, and this is on the to-do (wft/TWO_TRACK_FIT_2026-09-16.md §8).
#: Set WFT_TWO_TRACK_ALL_CANDIDATES=1 to reconsider every candidate.
TWO_TRACK_SELECTED_ONLY = os.environ.get('WFT_TWO_TRACK_ALL_CANDIDATES', '0') != '1'
#: Where the split statistic's chi2/dof scale comes from. 'one' (production):
#: the parent one-track fit's -- which on a real pair already holds the second
#: track's unexplained charge, so fstat ~ dchi2 / (1 + dchi2/dof) can never
#: exceed ~dof, and a narrow window (a close vertical pair: ~12 strips x 20
#: samples) cannot reach TWO_TRACK_F = 300 however clear the pair is
#: (two_track_limit.py, 2026-09-29). 'two': the two-track fit's, which equals
#: the parent's on a true single and is not inflated by the thing being tested.
TWO_TRACK_SCALE = os.environ.get('WFT_TWO_TRACK_SCALE', 'one')
#: How the joint fit is started. 'starts' (production): alternating pursuit from
#: the one-track parent plus three symmetric splits, best two refined. On close
#: pairs the parent is a slanted compromise line and the refinement falls into
#: an "X" of two crossing children, chi2 hundreds above the true pair (27/27
#: synthetic misses, two_track_limit.py 2026-09-30). 'grid': additionally scan
#: every pair of parallel lines on a strip-step grid (common slope, tied t0),
#: profiles by NNLS, and refine the best TWO_TRACK_GRID_KEEP of them with a
#: long Nelder-Mead. Costs 2-10 k chi2 evaluations per attempt.
TWO_TRACK_SEARCH = os.environ.get('WFT_TWO_TRACK_SEARCH', 'starts')
TWO_TRACK_GRID_TANS = np.round(np.arange(-0.4, 0.401, 0.1), 3)
TWO_TRACK_GRID_DT = (-60.0, 0.0, 60.0)
TWO_TRACK_GRID_MAX_MM = 14.0
TWO_TRACK_GRID_KEEP = 3


def _plane_extent(W, noise, pos, sig=TWO_TRACK_SIG) -> float:
    """Transverse extent of the window's significant charge [mm]."""
    amp = np.asarray(W).max(axis=1) / np.asarray(noise)
    live = amp > sig
    if live.sum() < 2:
        return 0.0
    p = np.asarray(pos)[live]
    return float(p.max() - p.min())


def one_track_width(fit: PlaneFit) -> float:
    """How wide a window ONE track of this fit can fill [mm]: the transverse
    travel across its own charge column, plus the charge spread and the
    sharing kernel's reach.

    Weak on its own, and worth knowing why: a one-track fit of a MERGED window
    buys width by inflating |w|, so it can "explain" an extent it has no charge
    for (measured on synthetics: a 30 mm pair reads |tan| 1.2 and the trigger
    misses it). Kept as a secondary trigger; the residual is the primary one."""
    u = fit.q_uend if np.isfinite(fit.q_uend) else 0.0
    return abs(fit.w) * u + 4.0 * wm.PITCH + 2.0 * float(
        (wm.HYPER or {}).get('sigma_p0', 0.4))


def two_track_probe(P, plane: str, fit: PlaneFit, hyper) -> Optional[dict]:
    """Everything the trigger and the joint fit both need, computed once: the
    prepared window, the parent's chi2 and charge profile, and the residual it
    leaves.

    The residual is the trigger that matters. A second track the one-track
    model does not cover shows as a run of strips with large *coherent
    positive* residual, and that is exactly the question "does one track
    explain this window" — unlike the strip count, which a steeper fit can
    fake, and unlike chi2/dof, which busy events raise on their own."""
    W, noise, pos, sat = wm.prep_plane(P, plane)
    if W.shape[1] != wm.NSAMP:
        wm.set_nsamp(W.shape[1])
        W, noise, pos, sat = wm.prep_plane(P, plane)
    chi_one, q_one = wm.chi2_plane(plane, W, noise, pos, sat, fit.p0, fit.w,
                                   fit.t0, hyper, snap_t0=False)
    if q_one is None or not np.isfinite(chi_one):
        return None
    dof = max(int((~sat).sum()), 1)
    R = W - (wm.build_matrix(plane, pos, fit.p0, fit.w, fit.t0, hyper) @ q_one
             ).reshape(W.shape)
    # per-strip coherent positive residual, in sigma, de-scaled by the fit's own
    # chi2/dof so that a merely imperfect model does not look like a track
    z = residual_z(R, noise, sat, chi_one / dof)
    return dict(W=W, noise=noise, pos=pos, sat=sat, chi_one=float(chi_one),
                q_one=q_one, dof=dof, resid_z=float(np.max(z)) if len(z) else 0.0,
                extent_mm=_plane_extent(W, noise, pos),
                width_mm=one_track_width(fit))


def two_track_triggers(probe: dict, other_two: bool = False) -> dict:
    """Which of the handoff §4 triggers fire for one candidate.

    The triggers exist to save compute, which is no longer a constraint
    (2026-09-16), and they are measured to be the largest loss at small
    separation: on synthetic coincident pairs 0-6 mm apart the fit alone
    recovers 57 % but this plane's trigger fires on only 25 % of them, leaving
    16 %. Two tracks a few mm apart leave little residual and no excess width.
    Attempting on every candidate is item 1 of the to-do
    (sept26_prelim_analysis/TWO_TRACK_FIT_LOG.md)."""
    return dict(residual=bool(probe['resid_z'] > TWO_TRACK_RESID_Z),
                width=bool(probe['extent_mm'] >
                           probe['width_mm'] + TWO_TRACK_WIDTH_MARGIN),
                cross_plane=bool(other_two),
                resid_z=float(probe['resid_z']),
                extent_mm=float(probe['extent_mm']),
                width_mm=float(probe['width_mm']))


def residual_z(R, noise, sat, chi_scale: float = 1.0) -> np.ndarray:
    """Per-strip coherent positive residual, in sigma, de-scaled by the fit's
    own chi2/dof. Large on a run of strips = charge one track does not cover."""
    live = ~sat
    n_live = np.maximum(live.sum(axis=1), 1)
    z = (R * live).sum(axis=1) / (np.asarray(noise) * np.sqrt(n_live))
    return z / np.sqrt(max(chi_scale, 1.0))


def _scan_positions(R, noise, pos, sat, lo, hi) -> np.ndarray:
    """Where in the window to look for a second track.

    The zero-slope stage costs one NNLS per position, so scanning every strip of
    a 60-strip window is most of the fit's price. The missing charge is not
    everywhere: it is on the strips with coherent positive residual. Scan those
    and one strip either side, and fall back to the whole window when the
    residual is featureless (nothing to lose then -- there is no second track)."""
    z = residual_z(R, noise, sat)
    if len(z) and np.isfinite(z).any() and z.max() > 1.0:
        keep = z >= max(1.0, 0.3 * z.max())
        idx = np.flatnonzero(keep)
        idx = np.unique(np.concatenate([idx - 1, idx, idx + 1]))
        idx = idx[(idx >= 0) & (idx < len(pos))]
        if 0 < len(idx) < len(pos):
            return np.unique(np.asarray(pos, float)[idx])
    return np.arange(lo, hi + 1e-9, wm.PITCH)


def _scan_residual(R, noise, pos, sat, plane, hyper, t0_centre, t0_hints=(),
                   p0_half=2.0, p0_step=0.5, w_step_mult=2, t0_grid=None):
    """Best single track in a residual, found the way ``_global_start`` finds a
    track in the data: (p0, t0) at zero slope where the residual is, then
    (p0, w) around the winner. Waveform samples only — CLAUDE.md: hits never
    set a position, an angle or a depth.

    Returns ``(chi2, (p0, w, t0), q)``."""
    def chi(p0, w, t0):
        return wm.chi2_plane(plane, R, noise, pos, sat, p0, w, t0, hyper)[0]

    lo, hi = float(np.min(pos)), float(np.max(pos))
    p0s = _scan_positions(R, noise, pos, sat, lo, hi)
    # +-180 ns, one depth bin apart. Narrower and a second track more than a
    # bin out of time is unreachable -- including the whole accidental class,
    # two tracks on the same strips at different times, which nothing else can
    # separate (test_two_track.test_time_separated_tracks_at_one_position).
    grid = T0_RESID_SCAN if t0_grid is None else t0_grid
    t0s = sorted({round(float(t), 1) for t in
                  list(t0_centre + np.asarray(grid)) + list(t0_hints)})
    best = (np.inf, float(t0_centre), lo)
    for t0 in t0s:
        for p0 in p0s:
            c = chi(p0, 0.0, t0)
            if c < best[0]:
                best = (c, float(t0), float(p0))
    _c, t0b, p0b = best
    ws = np.arange(-W_SCAN_HALF, W_SCAN_HALF + 1e-9, w_step_mult * W_SCAN_STEP)
    # p0 is the position AT THE MESH but the zero-slope stage finds the charge
    # CENTROID, and on an inclined track those differ by w x half the drift
    # column -- up to 10 mm at |tan| = 0.4. Re-centre the scan per slope (the
    # P0_SHEAR trick, doc §21.1) instead of widening the box: same grid, no
    # extra cost. Without it the second track is missed whenever it is the
    # steeper one (measured on synthetics, 2026-09-16).
    u_mid = 0.5 * wm.K * wm.DT
    best2 = (np.inf, p0b, 0.0)
    for p0 in np.arange(p0b - p0_half, p0b + p0_half + 1e-9, p0_step):
        for w in ws:
            p0m = p0 - w * u_mid
            c = chi(p0m, w, t0b)
            if c < best2[0]:
                best2 = (c, float(p0m), float(w))
    th = (best2[1], best2[2], t0b)
    c, q = wm.chi2_plane(plane, R, noise, pos, sat, th[0], th[1], th[2], hyper)
    return float(c), th, q


def _two_track_starts(W, noise, pos, sat, plane, parent_r, hyper, t0_hints=(),
                      n_alt: int = 2, t0_mode: str = None):
    """Starting points for the joint fit, by alternating matching pursuit.

    A merged window's one-track fit is a compromise line that belongs to
    neither track, so using it as track a's start and only searching for b
    leaves the optimiser in a bad basin (measured on synthetics: the second
    track is missed above ~9 mm separation). Instead alternate — subtract one
    track's model, re-find the other in what is left, repeat — which costs only
    one-track solves and lands both tracks near their own charge. The joint
    Nelder-Mead then has to move them a little, not find them.

    Symmetric splits of the one-track solution are kept as fallback starts for
    a residual the first fit has already absorbed (two tracks a strip apart).
    """
    free = (t0_mode or TWO_TRACK_T0) == 'free'
    grid = T0_RESID_SCAN_FREE if free else T0_RESID_SCAN
    pa = (parent_r['p0'], parent_r['w'], parent_r['t0'])
    qa = parent_r['q']
    pb, qb = None, None
    for _ in range(max(1, n_alt)):
        R = W - (wm.build_matrix(plane, pos, pa[0], pa[1], pa[2], hyper) @ qa
                 ).reshape(W.shape)
        _c, pb, qb = _scan_residual(R, noise, pos, sat, plane, hyper, pa[2],
                                    t0_hints=t0_hints, t0_grid=grid)
        if qb is None or not np.isfinite(_c):
            break
        R = W - (wm.build_matrix(plane, pos, pb[0], pb[1], pb[2], hyper) @ qb
                 ).reshape(W.shape)
        _c, pa2, qa2 = _scan_residual(R, noise, pos, sat, plane, hyper, pb[2],
                                      t0_hints=t0_hints, t0_grid=grid)
        if qa2 is None or not np.isfinite(_c):
            break
        pa, qa = pa2, qa2
    starts = []
    if pb is not None:
        starts.append((pa, pb, not free))
        if free and abs(pb[2] - pa[2]) < wm.DT:
            starts.append((pa, pb, True))
    p0, w, t0 = parent_r['p0'], parent_r['w'], parent_r['t0']
    for d in (2.0, 5.0, 10.0):
        starts.append(((p0 - 0.5 * d, w, t0), (p0 + 0.5 * d, w, t0), True))
    return starts


def _grid_starts(W, noise, pos, sat, plane, parent_r, hyper):
    """The best TWO_TRACK_GRID_KEEP pairs of parallel lines on a strip-step
    grid of the window (common slope, tied t0 near the parent's), each scored
    by the joint NNLS. See TWO_TRACK_SEARCH."""
    v = _v_mm_per_ns()
    ps = np.sort(np.asarray(pos, float))
    scored = []
    for tq in TWO_TRACK_GRID_TANS:
        w = float(tq) * v
        for dt in TWO_TRACK_GRID_DT:
            t0 = parent_r['t0'] + dt
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    if ps[j] - ps[i] > TWO_TRACK_GRID_MAX_MM:
                        break
                    pa, pb = (ps[i], w, t0), (ps[j], w, t0)
                    c = wm.chi2_plane_two(plane, W, noise, pos, sat, pa, pb, hyper,
                                          snap_t0=False)[0]
                    if np.isfinite(c):
                        scored.append((c, pa, pb))
    scored.sort(key=lambda z: z[0])
    return [(pa, pb, True) for _c, pa, pb in scored[:TWO_TRACK_GRID_KEEP]]


def _v_mm_per_ns() -> float:
    return float(wm.CAL.v_drift) * 1e-3


def _two_errors(W, noise, pos, sat, plane, pa, pb, chi0, dof, hyper,
                dp=0.05, dw=2e-4, dt=2.0):
    """1-sigma (p0, w, t0) per child from the curvature of the JOINT chi2 —
    one child's parameter moved, everything else held. Same scaling by
    sqrt(chi2/dof) as the single-track :func:`_errors`."""
    def chi(a, b):
        return wm.chi2_plane_two(plane, W, noise, pos, sat, a, b, hyper,
                                 snap_t0=False)[0]

    scale = max(chi0 / max(dof, 1), 1.0)
    out = []
    for k, (u, v) in enumerate(((pa, pb), (pb, pa))):
        e = []
        for i, h in ((0, dp), (1, dw), (2, dt)):
            up, dn = list(u), list(u)
            up[i] += h
            dn[i] -= h
            args = (lambda z: (tuple(z), v)) if k == 0 else (lambda z: (v, tuple(z)))
            d2 = (chi(*args(up)) - 2 * chi0 + chi(*args(dn))) / h ** 2
            e.append(float(np.sqrt(2 * scale / d2)) if d2 > 0 else np.nan)
        out.append(tuple(e))
    return out


def _child_fit(P, plane, cal, r, which: str, hyper, chi_alone: float) -> PlaneFit:
    """One child of a joint fit as an ordinary :class:`PlaneFit`.

    ``chi2``/``dof`` are the JOINT fit's — the two children share one window
    and one solve, and pretending otherwise would let a downstream chi2/dof cut
    see a quality the fit does not have."""
    p0, w, t0 = r['pa'] if which == 'a' else r['pb']
    q = r['qa'] if which == 'a' else r['qb']
    ep, ew, et = r['err_a'] if which == 'a' else r['err_b']
    tan = (w * 1e3 - cal.w0.get(plane, 0.0)) / (cal.kw.get(plane, 1.0) * cal.v_drift)
    q_sum, q_u50, q_u90, q_uend = _profile_summary(q)
    ch = np.asarray(P['ch'], dtype=int)
    flagged = np.concatenate([wm.DEAD.get(plane, np.array([], dtype=int)),
                              wm.HOT.get(plane, np.array([], dtype=int))])
    f = PlaneFit(
        p0=float(p0), w=float(w), t0=float(t0), tan_theta=float(tan),
        theta_deg=float(np.degrees(np.arctan(tan))),
        chi2=float(r['chi2']), dof=int(r['dof']),
        p0_err=float(np.hypot(ep, FLOOR_P0_MM)) if np.isfinite(ep) else FLOOR_P0_MM,
        w_err=float(ew) if np.isfinite(ew) else np.nan,
        tan_err=float(np.hypot(ew * 1e3 / cal.v_drift, FLOOR_TAN))
        if np.isfinite(ew) else FLOOR_TAN,
        t0_err=float(et) if np.isfinite(et) else np.nan,
        q_sum=q_sum, q_u50=q_u50, q_u90=q_u90, q_uend=q_uend,
        n_strips=int(np.asarray(P['W']).shape[0]), n_seed=0, n_dropped=0,
        slope_reliable=bool(abs(tan) >= TAN_MIN_SLOPE),
        quality_ok=bool(r['chi2'] / max(r['dof'], 1) < CHI2DOF_BAD),
        n_flagged_strips=int(np.isin(ch, flagged).sum()) if len(flagged) else 0)
    f._chi_alone = float(chi_alone)
    f._q = np.asarray(q, float)
    return f


def fit_plane_two(P, plane: str, cal: CalibrationBundle, parent: PlaneFit,
                  hyper: Optional[dict] = None, t0_hints=(),
                  f_thresh: float = None, probe: Optional[dict] = None,
                  t0_mode: str = None) -> Optional[dict]:
    """Try two tracks on a window a one-track fit has already seen.

    ``parent`` is that one-track fit; ``probe`` the :func:`two_track_probe`
    the trigger already computed, so the window is prepared once per candidate.
    Returns None when the window is not re-fittable, else a dict with the two
    children, the statistic and every guard's verdict — ``accepted`` says
    whether the caller should use them.
    """
    hyper = hyper or cal.hyper
    probe = probe or two_track_probe(P, plane, parent, hyper)
    if probe is None:
        return None
    W, noise, pos, sat = probe['W'], probe['noise'], probe['pos'], probe['sat']
    chi_one, q_one = probe['chi_one'], probe['q_one']
    parent_r = dict(p0=parent.p0, w=parent.w, t0=parent.t0, q=q_one)
    # guards and the barrier look only at depth bins the DAQ window constrains
    bins = wm.constrained_bins(parent.t0)
    wgt = np.asarray(q_one, float) * bins[:len(q_one)]
    wgt = wgt * (wgt > 0.05 * wgt.max()) if wgt.max() > 0 else None
    starts = _two_track_starts(W, noise, pos, sat, plane, parent_r, hyper,
                               t0_hints=t0_hints, t0_mode=t0_mode)
    if TWO_TRACK_SEARCH not in ('starts', 'grid'):
        raise ValueError(f'TWO_TRACK_SEARCH must be starts or grid: {TWO_TRACK_SEARCH!r}')
    if TWO_TRACK_SEARCH == 'grid':
        starts = _grid_starts(W, noise, pos, sat, plane, parent_r, hyper) + starts
        r = wm.fit_plane_two_raw(W, noise, pos, sat, plane, starts, hyper=hyper,
                                 wgt=wgt, bins=bins, n_refine=TWO_TRACK_GRID_KEEP + 2,
                                 maxiter=3000, maxiter_polish=2000)
    else:
        r = wm.fit_plane_two_raw(W, noise, pos, sat, plane, starts, hyper=hyper,
                                 wgt=wgt, bins=bins)
    if r is None:
        return None
    dof = int((~sat).sum())
    chi_two = float(r['chi2'])
    r['err_a'], r['err_b'] = _two_errors(W, noise, pos, sat, plane, r['pa'],
                                         r['pb'], chi_two, dof, hyper)
    # chi2 with only ONE of the two children, its own profile re-solved. Two
    # uses: each child's dchi2 on exactly the definition an ordinary candidate
    # gets (chi_null - its own best one-track chi2), and the MARGINAL
    # improvement each child brings to the pair.
    chi_null = float(((W / noise[:, None]) ** 2)[~sat].sum())
    alone = []
    for p in (r['pa'], r['pb']):
        c, _q = wm.chi2_plane(plane, W, noise, pos, sat, p[0], p[1], p[2],
                              hyper, snap_t0=False)
        alone.append(float(c) if np.isfinite(c) else chi_null)
    # The statistic. NOT the total chi2 improvement: that is large whenever the
    # second block absorbs anything at all, including noise, and it is the
    # reason a plain dchi2 threshold splits single tracks. Each child must be
    # individually necessary, so the statistic is the SMALLER of the two
    # marginal improvements -- chi2 without that child minus chi2 with both --
    # in units of the one-track fit's own chi2/dof, because chi2/dof here is
    # 1.4-6.8 on clean single tracks and 5-12 across the full pass.
    marg_a = alone[1] - chi_two          # what a adds to b
    marg_b = alone[0] - chi_two          # what b adds to a
    if TWO_TRACK_SCALE not in ('one', 'two'):
        raise ValueError(f'TWO_TRACK_SCALE must be one or two: {TWO_TRACK_SCALE!r}')
    scale = max((chi_one if TWO_TRACK_SCALE == 'one' else chi_two) / max(dof, 1), 1e-9)
    dchi2 = float(chi_one - chi_two)
    fstat = float(min(marg_a, marg_b) / scale)
    fa = _child_fit(P, plane, cal, r, 'a', hyper, alone[0])
    fb = _child_fit(P, plane, cal, r, 'b', hyper, alone[1])
    qtot = fa.q_sum + fb.q_sum
    qfrac = min(fa.q_sum, fb.q_sum) / qtot if qtot > 0 else 0.0
    plaus = [bool(np.isfinite(f.q_uend) and U_MIN_NS <= f.q_uend <= U_MAX_NS
                  and abs(f.tan_theta) < TAN_MAX) for f in (fa, fb)]
    thr = TWO_TRACK_F if f_thresh is None else f_thresh
    guards = dict(distinguishable=bool(r['dist'] >= 1.0),
                  column_shared=bool(r['overlap'] >= wm.TWO_MIN_OVERLAP),
                  both_plausible=bool(plaus[0] and plaus[1]))
    guards_ok = all(guards.values())
    accepted = bool(fstat >= thr and guards_ok)
    for i, f in enumerate((fa, fb)):
        f._plausible = plaus[i]
        f._dchi2 = float(chi_null - alone[i])
        f._rescued = False
        f._split_child = True
        f._split_dchi2 = dchi2
        f._split_f = fstat
    return dict(children=[fa, fb], profiles=(r['qa'], r['qb']),
                chi2_one=float(chi_one), chi2_two=chi_two,
                chi2_a_alone=alone[0], chi2_b_alone=alone[1],
                dchi2=dchi2, marg_a=float(marg_a), marg_b=float(marg_b),
                fstat=fstat, f_total=float(dchi2 / scale), dof=dof,
                sep=float(r['sep']), dist=float(r['dist']),
                overlap=float(r['overlap']),
                tie_t0=bool(r['tie_t0']), nfev=int(r['nfev']),
                qfrac=float(qfrac), threshold=float(thr),
                guards_ok=bool(guards_ok), accepted=accepted, **guards)


def _n_gated(all_fits: Dict[str, list], ftst_diff, cal=None) -> int:
    """How many gated tracks the selector would report for these candidates."""
    try:
        pairs = select_tracks(all_fits, ftst_diff, cal or _CAL,
                              max_tracks=MAX_TRACKS, pairing=_PAIRING)
    except Exception:
        return 0
    return int(sum(1 for _i, _j, g in pairs if g))


def _n_plausible(fits: list) -> int:
    return sum(1 for f in (fits or []) if f is not None
               and bool(getattr(f, '_plausible', True)))


def _cross_plane_mismatch(this: list, other: list) -> bool:
    """Does the OTHER plane resolve two time-coincident plausible candidates
    where this plane resolves fewer? Handoff §4 trigger 2.

    The count *mismatch* is the whole of it, and leaving the second half out was
    a real defect: with only "the other plane has two", a pair that both planes
    already resolve — two tracks 24 mm apart, the easy case — corroborates
    itself, takes the lower threshold, and splits one of the two correct
    candidates. Measured on the overlay bench 2026-09-16: 91–97 % of the splits
    accepted at ≥ 24 mm were "corroborated" that way, and they cost 10 points of
    both-tracks-found where production was already right. With the mismatch
    required, only 7 % of them survive."""
    ok = [f for f in (other or []) if f is not None
          and bool(getattr(f, '_plausible', True))]
    if len(ok) < 2 or _n_plausible(this) >= len(ok):
        return False
    return any(abs(ok[i].t0 - ok[j].t0) <= DT_XY_TOL_NS
               for i in range(len(ok)) for j in range(i + 1, len(ok)))


def resolve_two_tracks(all_fits: Dict[str, list], windows: Dict[str, list],
                       cal: CalibrationBundle, ftst_diff=None,
                       max_try: int = TWO_TRACK_MAX_TRY,
                       selected: Optional[Dict[str, set]] = None) -> tuple:
    """Offer the merged-looking candidates of both planes a second track.

    Runs AFTER the ordinary per-plane candidate fits, so the one-track answer
    is always available and a candidate whose split is refused is untouched.
    Accepted children replace their parent in the plane's ranked list; the
    parent is returned separately for the side table.

    Returns ``(all_fits, splits, replaced)``: the possibly-rewritten ranked
    lists, one record per attempt (provenance, and the bench's input), and the
    parents a split displaced, per plane, for the candidates side table."""
    splits = []
    replaced = {'x': [], 'y': []}
    if not TWO_TRACK:
        return all_fits, splits, replaced
    out = {p: list(all_fits.get(p) or []) for p in ('x', 'y')}
    if selected is None and TWO_TRACK_SELECTED_ONLY:
        try:
            pre = select_tracks(all_fits, ftst_diff, cal, max_tracks=MAX_TRACKS)
        except Exception:
            pre = []
        selected = {'x': {i for i, _j, _g in pre}, 'y': {j for _i, j, _g in pre}}
    for plane in ('x', 'y'):
        other = 'y' if plane == 'x' else 'x'
        wins = windows.get(plane) or []
        fits = out[plane]
        corrob = _cross_plane_mismatch(all_fits.get(plane), all_fits.get(other))
        hints = []
        if corrob:
            dt = cal.dt_xy.get(int(ftst_diff), -18.8) if ftst_diff is not None else -18.8
            s = 1.0 if plane == 'x' else -1.0
            hints = [f.t0 + s * dt for f in (all_fits.get(other) or [])[:2]
                     if f is not None]
        thr = TWO_TRACK_F_CORROB if corrob else TWO_TRACK_F
        tried = 0
        for rank, f in enumerate(list(fits)):
            if f is None or tried >= max_try:
                continue
            if selected is not None and rank not in selected.get(plane, set()):
                continue
            iw = getattr(f, '_win', None)
            if iw is None or iw >= len(wins):
                continue
            P = wins[iw]
            try:
                probe = two_track_probe(P, plane, f, cal.hyper)
            except Exception:
                probe = None
            if probe is None:
                continue
            trig = two_track_triggers(probe, other_two=corrob)
            if not (trig['residual'] or trig['width'] or trig['cross_plane']):
                continue
            tried += 1
            try:
                r = fit_plane_two(P, plane, cal, f, t0_hints=hints,
                                  f_thresh=thr, probe=probe,
                                  t0_mode=TWO_TRACK_T0)
            except Exception:
                r = None
            if r is None:
                continue
            rec = dict(plane=plane, rank=int(rank), corroborated=bool(corrob),
                       **{k: v for k, v in r.items()
                          if k not in ('children', 'profiles')},
                       **{f'trig_{k}': v for k, v in trig.items()})
            splits.append(rec)
            if r['accepted']:
                ca, cb = r['children']
                ca.n_seed = cb.n_seed = f.n_seed
                ca.n_dropped = cb.n_dropped = f.n_dropped
                ca._win = cb._win = iw
                f._split_replaced = True
                replaced[plane].append(f)
                i = next(k for k, g in enumerate(out[plane]) if g is f)
                out[plane][i:i + 1] = [ca, cb]
        out[plane].sort(key=lambda g: (1 if getattr(g, '_plausible', True) else 0,
                                       getattr(g, '_dchi2', 0.0) or 0.0),
                        reverse=True)
        n = len(out[plane])
        for g in out[plane]:
            g.n_candidates = n
    return out, splits, replaced


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
    if 'lqc' in pairing['features']:
        cx, cy = constrained_charge(fx), constrained_charge(fy)
        f['lqc'] = (np.log(max(cx, 1.0) / max(cy, 1.0))
                    if np.isfinite(cx) and np.isfinite(cy) else np.nan)
    cost = 0.0
    for k in pairing['features']:
        z = (f[k] - pairing['median'][k]) / pairing['rsig'][k]
        cost += min(z * z, 25.0) if np.isfinite(z) else 25.0
    pr = pairing.get('prof')
    if pr:
        d = profile_distance(fx, fy, dt, float(pr.get('sigma_bins', 1.0)))
        cost += float(pr['weight']) * (min(d / float(pr['scale']), 25.0)
                                       if np.isfinite(d) else 25.0)
    return cost


def _constrained_profile(f) -> Optional[np.ndarray]:
    """The depth profile with the bins the DAQ window does not constrain set
    to zero (wft.model.constrained_bins): those hold whatever NNLS parked there."""
    q = getattr(f, '_q', None)
    if q is None:
        return None
    q = np.asarray(q, float).copy()
    q[~wm.constrained_bins(f.t0)[:len(q)]] = 0.0
    return q


def constrained_charge(f) -> float:
    """Fitted charge over the constrained depth bins only (nan without a profile)."""
    q = _constrained_profile(f)
    return float(q.sum()) if q is not None else np.nan


def profile_distance(fx, fy, dt: float, sigma_bins: float = 1.0) -> float:
    """How unlike the depth profiles of an x and a y candidate are.

    Both views sample the same charge column, so one track's x and y profiles
    agree up to gain and noise. Each is resampled on a common absolute-time
    grid (y moved into x's frame by the measured offset ``dt``; comparing bin k
    with bin k fails, because t0 slides by whole bins along the t0 <-> q
    degeneracy), smoothed by ``sigma_bins``, normalised, and compared with a
    chi2-like distance. On clean coincident donor pairs of run_145 this, added
    to the charge ratio, raises correct keep/swap decisions from 87.4 to 91.6 %
    (A) and 89.1 to 92.8 % (C), held-out (TWO_TRACK_FIT_LOG, 2026-09-30).
    Opt-in: used only when the pairing calibration carries a ``prof`` block."""
    from scipy.ndimage import gaussian_filter1d
    qx, qy = _constrained_profile(fx), _constrained_profile(fy)
    if qx is None or qy is None:
        return np.nan
    ref = fx.t0 - 2 * wm.DT
    grid = ref + (np.arange(len(qx) + 6) + 0.5) * wm.DT

    def on_grid(q, t0):
        u = t0 + (np.arange(len(q)) + 0.5) * wm.DT
        return np.interp(grid, u, q, left=0.0, right=0.0)

    px, py = on_grid(qx, fx.t0), on_grid(qy, fy.t0 + dt)
    if sigma_bins > 0:
        px, py = gaussian_filter1d(px, sigma_bins), gaussian_filter1d(py, sigma_bins)
    sx, sy = px.sum(), py.sum()
    if sx <= 0 or sy <= 0:
        return np.nan
    px, py = px / sx, py / sy
    return float(np.sum((px - py) ** 2 / (px + py + 0.01)))


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
                   ftst: Optional[dict] = None,
                   replaced: Optional[Dict[str, list]] = None) -> list:
    """One dict per fitted candidate cluster — the full ranked list that
    :func:`row_from_fits` reduces to a single winner. ``pairs`` (from
    :func:`select_tracks`) stamps each candidate with the track it belongs
    to; ``track_id`` -1 = not part of any selected pair.

    ``replaced``: one-track parents a joint two-track fit displaced. They are
    written with ``rank`` -1 and ``split_replaced`` true, so the split can be
    undone downstream without re-reconstructing (HANDOFF_JOINT_TWO_TRACK_FIT §0.1)."""
    track_of = {}
    for tid, (ix, iy, gated) in enumerate(pairs or []):
        track_of[('x', ix)] = (tid, gated)
        track_of[('y', iy)] = (tid, gated)
    rows = []
    for plane in ('x', 'y'):
        f_ftst = (ftst or {}).get(plane)
        live = list(all_fits.get(plane) or [])
        for rank, f in enumerate(live + list((replaced or {}).get(plane) or [])):
            if f is None:
                continue
            rank = rank if rank < len(live) else -1
            row = {'event_id': int(event_id), 'plane': plane, 'rank': int(rank)}
            row.update(asdict(f))
            row['plausible'] = bool(getattr(f, '_plausible', True))
            row['dchi2'] = float(getattr(f, '_dchi2', np.nan))
            row['rescued'] = bool(getattr(f, '_rescued', False))
            row['split_child'] = bool(getattr(f, '_split_child', False))
            row['split_replaced'] = bool(getattr(f, '_split_replaced', False))
            row['split_dchi2'] = float(getattr(f, '_split_dchi2', np.nan))
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
#: Module globals a worker may be started with. Everything here is otherwise an
#: environment variable read at import, which a forked worker cannot be told
#: about after the fact -- so an A/B harness passes them here instead.
WORKER_OPTS = ('TWO_TRACK', 'TWO_TRACK_F', 'TWO_TRACK_F_CORROB', 'TWO_TRACK_T0',
               'TWO_TRACK_RESID_Z', 'TWO_TRACK_WIDTH_MARGIN', 'TWO_TRACK_MAX_TRY',
               'TWO_TRACK_SELECTED_ONLY', 'TWO_TRACK_SCALE', 'TWO_TRACK_SEARCH')


def _worker_init(bundle_path, pairing_path=None, opts=None):
    """``pairing_path`` overrides the bundle's ``xy_pairing`` (for A/B runs);
    ``opts`` overrides the :data:`WORKER_OPTS` module globals (likewise)."""
    global _CAL, _PAIRING
    _CAL = CalibrationBundle.load(bundle_path)
    wm.use_calibration(_CAL)
    _PAIRING = load_pairing(pairing_path) if pairing_path else (_CAL.xy_pairing or None)
    for k, v in (opts or {}).items():
        if k not in WORKER_OPTS:
            raise KeyError(f'not a worker option: {k!r}; known: {WORKER_OPTS}')
        globals()[k] = v


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
    splits, replaced = [], {}
    if TWO_TRACK:
        try:
            kept = all_fits
            all_fits, splits, replaced = resolve_two_tracks(
                all_fits, wins, _CAL, ftst_diff)
            # A SPLIT MAY NOT COST THE EVENT A TRACK. It replaces its parent, so
            # if the two children then fail to pair with the other plane the
            # event ends up with fewer gated tracks than before -- measured at
            # 0.2 % (A) / 0.8 % (C) of real triggers before this guard. The fit
            # exists to find tracks; a split that loses one is wrong whatever
            # its chi2 says, so the whole event reverts to the unsplit answer.
            if any(replaced.values()) and _n_gated(kept, ftst_diff) > \
                    _n_gated(all_fits, ftst_diff):
                for f in [g for v in replaced.values() for g in v]:
                    f._split_replaced = False
                for r in splits:
                    r['accepted'] = False
                    r['reverted_lost_track'] = True
                all_fits, replaced = kept, {'x': [], 'y': []}
            for plane in ('x', 'y'):
                if replaced.get(plane) and all_fits.get(plane):
                    fits[plane] = all_fits[plane][0]
        except Exception:
            splits, replaced = [], {}
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
    if TWO_TRACK:
        row['n_splits'] = int(sum(1 for r in splits if r['accepted']))
    if EMIT_CANDIDATES:
        row['_cand'] = candidate_rows(eid, all_fits, pairs, ftst, replaced)
    if splits:
        row['_splits'] = [dict(event_id=int(eid), **r) for r in splits]
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

    rows, cand_rows, split_rows = [], [], []
    with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init,
                             initargs=(bundle_path,)) as pool:
        for payloads in _stream_windows(cfg, pos_maps, seeds, wanted, pad_strips,
                                        verbose=verbose):
            if not payloads:
                continue
            for r in pool.map(_worker_fit, payloads, chunksize=8):
                cand_rows.extend(r.pop('_cand', []))
                split_rows.extend(r.pop('_splits', []))
                rows.append(r)
            if verbose:
                print(f'[wft]   {len(rows):,} events reconstructed', flush=True)

    df = pd.DataFrame(rows).sort_values('event_id').reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    if split_rows:
        pd.DataFrame(split_rows).sort_values(['event_id', 'plane']).reset_index(
            drop=True).to_parquet(out_path.replace('.parquet', '.splits.parquet'),
                                  index=False)
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
                                 two_track_fit=TWO_TRACK,
                                 two_track_f=TWO_TRACK_F if TWO_TRACK else None,
                                 two_track_f_corrob=(TWO_TRACK_F_CORROB
                                                     if TWO_TRACK else None),
                                 two_track_width_mm=(TWO_TRACK_WIDTH_MARGIN
                                                     if TWO_TRACK else None),
                                 n_splits=int(sum(r['accepted'] for r in split_rows)),
                                 n_split_attempts=len(split_rows),
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
