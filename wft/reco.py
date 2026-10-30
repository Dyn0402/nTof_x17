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
FLOOR_TAN = 0.018          # ~1.0 deg: det3's measured per-event model floor; a bundle's tan_floor overrides
FLOOR_P0_MM = 0.33         # measured charge-centroid jitter per 60 ns bin
CHI2DOF_BAD = 300.0        # showers / multi-track / spark
#: Per-event (p0, w, t0) errors from the full variable-projection covariance (J^T J)^-1 at the fit's minimum, whatever
#: the outer solver, instead of three 1-D curvatures (branch lm-varpro, 2026-10-28).  The fit itself is untouched; the
#: 1-D curvatures ignore the p0-t0 correlation and read w_err and t0_err 1.9-2.9x small on the bench fleet.  tan_err and
#: p0_err barely move (the per-chamber floors dominate).  A PROTOTYPE, not gated: study override (wft.diag).
COV_ERRORS = False
#: quality flag (GE 2026-10-28): a view is flagged when its chi2/dof exceeds CHI2DOF_REL x the median chi2/dof of the
#: same view over the same table (one chamber, one run condition).  The model error grows with the signal, so the
#: typical chi2/dof runs from ~4 (n_TOF A) to ~740 (det6 Y) and no fixed number means the same on two chambers; ranked
#: against its own table, chi2/dof separates the bad reconstructions (second deposits, showers) from the good with AUC
#: 0.78-0.96 where they add charge, and 5x is the knee (det3: reproduces the old fixed 300).  Blind where the bad views
#: MISS charge (det4 dead stripes, det7 X M1): a hardware limit, not a cut.  Set at table level by quality_flags().
CHI2DOF_REL = 5.0

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
# per candidate. NOT part of the algorithm: on the phys kernel with the pivot
# start it is null on det2/det3/det6/det7 and recovers det4 X far -1.2 pp
# (model audit F19 `ts`); a study override (wft.diag).
TWO_SIDED = False
TS_PROBE_TAN = (0.15, 0.4, 0.8)        # raw |tan| of the (p0, t0) stage
TS_TAN_LO, TS_TAN_HI, TS_TAN_STEP = 0.05, 1.6, 0.05
TS_P0_HALF, TS_P0_STEP = 6.0, 0.5      # mm around the window's charge centroid
TS_T0_HALF, TS_T0_STEP = 240.0, 40.0   # ns around the one-sided fit's t0

# Pivot start (rerun/seedfit, CHECKS C19, 2026-10-24). The chi2 surface along w is single-welled only at the right
# p0; p0 is the mesh END of the track, up to w * (half the column) ~ 5 mm from the charge, so any (p0, w) box around
# the centroid is a diagonal valley. A straight track passes through its own charge centroid (x_c, t_c) whatever its
# slope, and both are model-free: x_c is the charge-weighted strip position, t_c the charge-weighted sample time minus
# the electronics template's own time centroid (convolution adds means). So scan w with the line forced through
# (x_c, t_c) -- p0(w) = x_c - w (t_c - t0) -- and refine the best PIVOT_NREF wells with fit_plane_raw. Same cost as
# _global_start (33 + 2 refinements), fewer wrong-basin fits on every chamber. The pass's start since C19/C21 (it was
# exported by rerun/stage_job.sh); built in since 2026-10-27.  False = the old _global_start (study only, wft.diag).
PIVOT_START = True
PIVOT_W_HALF = 0.032       # mm/ns: |tan| ~ 0.9 at v = 36 um/ns
PIVOT_W_STEP = 0.002       # the true well is 3-5 um/ns wide (seedfit landscapes)
PIVOT_NREF = 2
PIVOT_NSIG = 4.0           # strips entering the centroid: peak above this x noise


def charge_centroid(W, noise, pos, plane, hyper):
    """(x_c [mm], t_c [ns]) of one plane's charge: the charge-weighted position of the strips peaking above
    PIVOT_NSIG x noise, and their charge-weighted sample time minus the impulse response's time centroid -- i.e.
    the mean arrival time of the charge, without deconvolving. None if no strip qualifies."""
    rc = 'rc_D_y' in hyper or 'rc_D_x' in hyper
    tp = 'x' if (rc and hyper.get('rc_tmpl', 'x') in ('x', 1, 1.0)) else plane
    tm = wm.TMPL[tp]
    area = float(np.sum(tm))
    rows = W.max(axis=1) > PIVOT_NSIG * noise
    if not rows.any() or area <= 0:
        return None
    Wr = W[rows]
    q = Wr.sum(axis=1)
    Q = float(q.sum())
    if Q <= 0:
        return None
    ts = np.arange(W.shape[1]) * wm.SNS
    t_tmpl = float(np.sum(wm.TGRID * tm) / area)
    return float((q * pos[rows]).sum() / Q), float((Wr * ts[None, :]).sum() / Q) - t_tmpl


def _pivot_fit(P, plane, hyper, t0_seed, t0_prior=None):
    """fit_plane_raw from the PIVOT_START scan; None if the plane has no charge to pivot on."""
    W, noise, pos, sat = wm.prep_plane(P, plane)
    c = charge_centroid(W, noise, pos, plane, hyper)
    if c is None:
        return None
    xc, tc = c
    t0 = float(t0_prior[0]) if t0_prior is not None else float(t0_seed)
    ws = np.arange(-PIVOT_W_HALF, PIVOT_W_HALF + 1e-12, PIVOT_W_STEP)
    chis = np.array([wm.chi2_plane(plane, W, noise, pos, sat, xc - w * (tc - t0), w, t0, hyper,
                                   t0_prior=t0_prior)[0] for w in ws])
    n = len(ws)
    wells = [i for i in range(n) if (i == 0 or chis[i] <= chis[i - 1]) and (i == n - 1 or chis[i] <= chis[i + 1])]
    best = None
    for i in sorted(wells, key=lambda i: chis[i])[:PIVOT_NREF]:
        r = wm.fit_plane_raw(P, plane, xc - ws[i] * (tc - t0), float(ws[i]), t0, hyper=hyper, t0_prior=t0_prior,
                             pivot=(xc, tc))
        if r is not None and np.isfinite(r['chi2']) and (best is None or r['chi2'] < best['chi2']):
            best = r
    return best


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
    n_band_strips: int = 0       # of n_strips, how many are noise bands in this event (censored, wft.model.BAND_NSIG)
    n_sat_strips: int = 0        # of n_strips, how many reach the ADC rail (sat_adc) in this event


def quality_flags(df, medians: Optional[dict] = None) -> tuple:
    """Set ``quality_ok`` relative to the table (CHI2DOF_REL x the median chi2/dof per view).  Returns (df, medians).

    ``df`` is an events table (``x_chi2``, ``x_dof``, ``x_ok`` ...) or a candidates table (``plane``, ``chi2``, ``dof``,
    ``rank``).  The median runs over the fitted views (events) or the live winners, ``rank == 0`` (candidates).
    ``medians`` = {'x': m, 'y': m} reuses another table's medians (candidates take the events table's; n_TOF takes one
    run x arm over all its sub-runs)."""
    df = df.copy()
    wide = 'plane' not in df.columns
    med = dict(medians or {})
    for p in ('x', 'y'):
        if wide:
            if f'{p}_chi2' not in df:
                continue
            r = df[f'{p}_chi2'] / df[f'{p}_dof'].clip(lower=1)
            ok = df[f'{p}_ok'].astype(bool) & np.isfinite(r)
            if p not in med:
                med[p] = float(np.median(r[ok])) if ok.any() else np.nan
            df[f'{p}_quality_ok'] = (ok & (r <= CHI2DOF_REL * med[p])).to_numpy() if np.isfinite(med[p]) else ok.to_numpy()
        else:
            sel = (df['plane'] == p).to_numpy()
            if not sel.any():
                continue
            r = (df['chi2'] / df['dof'].clip(lower=1)).to_numpy()
            if p not in med:
                w = sel & (df['rank'].to_numpy() == 0) & np.isfinite(r)
                med[p] = float(np.median(r[w])) if w.any() else np.nan
            q = df['quality_ok'].to_numpy(bool).copy() if 'quality_ok' in df else np.ones(len(df), bool)
            q[sel] = np.isfinite(r[sel]) & ((r[sel] <= CHI2DOF_REL * med[p]) if np.isfinite(med[p]) else True)
            df['quality_ok'] = q
    return df, med


def quality_rule(medians: dict) -> dict:
    """The sidecar's record of the flag: a table carrying it is not re-flagged on load (wft.compat.load_table)."""
    return dict(rule='chi2/dof <= CHI2DOF_REL x median(chi2/dof) of the view over this table', rel=CHI2DOF_REL,
                chi2dof_median={p: (None if v is None or not np.isfinite(v) else float(v)) for p, v in medians.items()})


def window_flags(P, plane: str) -> tuple:
    """(n_flagged, n_band, n_sat) strips of one fit window: dead/hot wildcards of the bundle, per-event noise bands,
    strips with a sample at the ADC rail.  They define the event class of the x-y charge pairing (pairing_class)."""
    W = np.asarray(P['W'])
    ch = np.asarray(P['ch'], dtype=int)
    flagged = np.concatenate([wm.DEAD.get(plane, np.array([], dtype=int)),
                              wm.HOT.get(plane, np.array([], dtype=int))])
    n_flag = int(np.isin(ch, flagged).sum()) if len(flagged) else 0
    miss = P.get('missing') if isinstance(P, dict) else None
    n_band = int(wm.band_rows(W, P['noise'], miss).sum())
    n_sat = int((W >= wm.SAT).any(axis=1).sum()) if W.ndim == 2 else 0
    return n_flag, n_band, n_sat


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
    ~20 % optimistic (measured pull widths 1.19/1.13); the full covariance says x2 (algo_review 2026-10-28).  A fit
    that carries ``cov`` (wft.model.FITTER = 'lm') gets its errors from it instead, scaled the same way."""
    cov = r.get('cov') if isinstance(r, dict) else None
    if cov is None and COV_ERRORS and r.get('q') is not None:
        cov = wm.vp_covariance(P, plane, r, hyper, t0_prior=t0_prior)
    if cov is not None:
        scale = max(r['chi2'] / max(r['dof'], 1), 1.0)
        d = np.diag(cov) * scale
        return tuple(float(np.sqrt(x)) if x > 0 else np.nan for x in d)
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


#: Depth at which the position-dependent angle map is evaluated [mm]: the middle of the 30 mm drift gap.
ANGLE_MAP_MID_MM = 15.0


BOW_MID_MM = 15.0


def _bow_poly(c: dict, U: float, W: float, lin=('cu', 'cw'), quad=('auu', 'aww', 'auw')) -> float:
    return (c.get(lin[0], 0.0) * U + c.get(lin[1], 0.0) * W + c.get(quad[0], 0.0) * U * U / 2
            + c.get(quad[1], 0.0) * W * W / 2 + c.get(quad[2], 0.0) * U * W)


def bow_h(cal: CalibrationBundle, U: float, W: float, plane: str = None) -> float:
    """Mesh height [mm, M3 z up] of the bow block at (U, W); 0 without a block.  With ``plane``, that view's own surface
    ``h_<plane>`` when the block has it: the cross-view mean ``h`` makes the positions worse, the view's own one improves
    them 2-3 % (det2/det3) to 10 % (det6) on other datasets of the chamber (rerun/calib/bow/BOW_TRANSFER.md, 10-29)."""
    b = getattr(cal, 'bow', None) or {}
    h = (b.get(f'h_{plane}') if plane else None) or b.get('h')
    return float(_bow_poly(h, U, W, ('gu', 'gw'), ('kuu', 'kww', 'kuw'))) if h else 0.0


def tan_const(cal: CalibrationBundle, plane: str, w: float) -> float:
    """The constant mapping (w*1e3 - w0) / (kw v)."""
    return float((w * 1e3 - cal.w0.get(plane, 0.0)) / (cal.kw.get(plane, 1.0) * cal.v_drift))


def tan_bow(cal: CalibrationBundle, plane: str, w: float, U: float, W: float) -> float:
    """tan = (w - w0 - A_bow(U, W)) / (kw v S): S = FA's v_rel(x, y) at (100 U + 200, 100 W + 200) when the bundle has a
    v_map, else FB's linear scale 1 + su U + sw W (never both: the v map IS the scale variation, FA spec).

    A block with ``a_<plane>`` (the tilt as a geometric tan offset, a = A / (kw v) of the row it was measured on) uses
    tan = (w - w0) / (kw v S) - a(U, W) instead.  The tilt is v r_f dh/du, so in tan it does not change with the drift
    field: a det3 shape measured at 1000 V transfers to its 500 V row in this form, not in um/ns
    (rerun/calib/bow/BOW_TRANSFER.md, 2026-10-29).  Blocks without it are unchanged."""
    b = getattr(cal, 'bow', None) or {}
    A, s = b.get(f'A_{plane}') or {}, b.get(f's_{plane}') or {}
    if (getattr(cal, 'v_map', None) or {}).get('xy'):
        S = v_rel_xy(cal, 100.0 * U + 200.0, 100.0 * W + 200.0)
    else:
        S = 1.0 + s.get('su', 0.0) * U + s.get('sw', 0.0) * W
    a = b.get(f'a_{plane}')
    if a:
        return float((w * 1e3 - cal.w0.get(plane, 0.0)) / (cal.kw.get(plane, 1.0) * cal.v_drift * S) - _bow_poly(a, U, W))
    return float((w * 1e3 - cal.w0.get(plane, 0.0) - _bow_poly(A, U, W)) / (cal.kw.get(plane, 1.0) * cal.v_drift * S))


#: the bow block is fitted on the plate (U, W in 100 mm from its centre: -2...+2); beyond 2.2 a view is not trusted
BOW_UW_MAX = 2.2


def bow_UW(cal: CalibrationBundle, fx=None, fy=None) -> tuple:
    """(U, W) of a track from its views' mesh positions and constant-mapping tangents; a missing view gives 0, and so
    does a view whose point is off the plate (|U|, |W| > BOW_UW_MAX: an absurd fit of that view would otherwise set the
    other view's bow tangent and mesh height, pass-1 review 2026-10-30: <= 0.2 % of events)."""
    U = ((fx.p0 + BOW_MID_MM * tan_const(cal, 'x', fx.w)) - 200.0) / 100.0 if fx is not None else 0.0
    W = ((fy.p0 + BOW_MID_MM * tan_const(cal, 'y', fy.w)) - 200.0) / 100.0 if fy is not None else 0.0
    ok = lambda c: bool(np.isfinite(c) and abs(c) <= BOW_UW_MAX)
    return (U if ok(U) else 0.0), (W if ok(W) else 0.0)


def apply_bow(cal: CalibrationBundle, fx=None, fy=None) -> None:
    """Re-map both views' tangents through the bundle's bow block with BOTH views' coordinates (one iteration, from the
    constant mapping; idempotent: computed from w, not from the stored tangent).  Stores the mesh height on each fit
    (``_bow_h``: the track point at the nominal plane is p0 - tan * h).  No-op without a block."""
    if not (getattr(cal, 'bow', None) or (getattr(cal, 'v_map', None) or {}).get('xy')):
        return
    U, W = bow_UW(cal, fx, fy)
    for p, f in (('x', fx), ('y', fy)):
        if f is None:
            continue
        t = tan_bow(cal, p, f.w, U, W)
        f.tan_theta, f.theta_deg = float(t), float(np.degrees(np.arctan(t)))
        f.slope_reliable = bool(abs(t) >= TAN_MIN_SLOPE)
        f._bow_h = float(bow_h(cal, U, W, p))


def tan_of(cal: CalibrationBundle, plane: str, w: float, p0: float) -> float:
    """Track tangent from the fitted transverse speed w [mm/ns] and mesh position p0 [mm].

    Constant mapping tan = (w*1e3 - w0) / (kw*v), or, when the bundle carries an ``angle_map`` for the plane,
    the position-dependent one tan = (w*1e3 - w0(p)) / (s(p)*kw*v) at the track's mid-gap position p (from the
    constant-mapping tangent; the map varies over ~100 mm, so the first-order p is exact enough)."""
    kv = cal.kw.get(plane, 1.0) * cal.v_drift
    tan = (w * 1e3 - cal.w0.get(plane, 0.0)) / kv
    if getattr(cal, 'bow', None) or (getattr(cal, 'v_map', None) or {}).get('xy'):
        # one view alone: its own coordinate, the other 0 (apply_bow redoes it with both views once paired)
        c = (p0 + BOW_MID_MM * tan - 200.0) / 100.0 if np.isfinite(p0) and np.isfinite(tan) else 0.0
        return tan_bow(cal, plane, w, c if plane == 'x' else 0.0, c if plane == 'y' else 0.0)
    m = (getattr(cal, 'angle_map', None) or {}).get(plane)
    if m and np.isfinite(tan) and np.isfinite(p0):
        p = p0 + ANGLE_MAP_MID_MM * tan
        tan = (w * 1e3 - float(np.interp(p, m['p'], m['w0']))) / (float(np.interp(p, m['p'], m['s'])) * kv)
    return float(tan)


def v_rel_xy(cal: CalibrationBundle, x: float, y: float) -> float:
    """FA's v(x, y): v relative to v_drift at the track's mid-gap (x, y) [mm], common to both views; bilinear between
    the cell centres of ``v_map['xy'] = {'e': edges [mm], 'v_rel': (nx, ny)}`` (4 x 4 cells of 100 mm), clamped at the
    outer centres, empty cells = the median.  1.0 without a map or position."""
    m = (getattr(cal, 'v_map', None) or {}).get('xy')
    if not m or not (np.isfinite(x) and np.isfinite(y)):
        return 1.0
    e = np.asarray(m['e'], float); c = 0.5 * (e[1:] + e[:-1])
    V = np.asarray(m['v_rel'], float)
    V = np.where(np.isfinite(V) & (V > 0), V, np.nanmedian(V[np.isfinite(V) & (V > 0)]))
    fx = np.clip(np.interp(x, c, np.arange(len(c))), 0, len(c) - 1)
    fy = np.clip(np.interp(y, c, np.arange(len(c))), 0, len(c) - 1)
    i0, j0 = int(min(np.floor(fx), len(c) - 2)), int(min(np.floor(fy), len(c) - 2))
    dx, dy = fx - i0, fy - j0
    return float(V[i0, j0] * (1 - dx) * (1 - dy) + V[i0 + 1, j0] * dx * (1 - dy)
                 + V[i0, j0 + 1] * (1 - dx) * dy + V[i0 + 1, j0 + 1] * dx * dy)


def v_rel_t(cal: CalibrationBundle, t: float) -> float:
    """FA's v(t): ``v_map['t'] = {'clock', 't_h': [...], 'v_rel': [...]}``, piecewise linear in the run's time, clamped.
    NOT applied in the reconstruction yet: the event record carries no DREAM timestamp (only ftst) -- FE pending."""
    m = (getattr(cal, 'v_map', None) or {}).get('t')
    if not m or not np.isfinite(t):
        return 1.0
    return float(np.interp(t, np.asarray(m['t_h'], float), np.asarray(m['v_rel'], float)))


def v_scale(cal: CalibrationBundle, x: float = np.nan, y: float = np.nan, t: float = np.nan) -> float:
    """v_rel(t) * v_rel(x, y): the local drift velocity relative to the bundle's v_drift."""
    return v_rel_t(cal, t) * v_rel_xy(cal, x, y)


def tan_floor(cal: CalibrationBundle, plane: str) -> float:
    """Per-event angle error floor [tan] for this chamber and view: the bundle's measured ``tan_floor`` (ws_floor),
    else FLOOR_TAN."""
    v = (getattr(cal, 'tan_floor', None) or {}).get(plane)
    return float(v) if v and v > 0 else FLOOR_TAN


def fit_plane(P, plane: str, cal: CalibrationBundle, hyper: Optional[dict] = None,
              n_seed: int = 0, n_dropped: int = 0,
              t0_prior: Optional[tuple] = None) -> Optional[PlaneFit]:
    """Fit one plane's window. P: dict/PlaneWindow-like with W, pos, noise, ch.
    ``t0_prior=(t0_pred, sigma)``: external-clock t0 penalty (see t0_prior_for)."""
    hyper = hyper or cal.hyper
    W = np.asarray(P['W'])
    if W.shape[1] != wm.NSAMP:
        wm.set_nsamp(W.shape[1])
    r = _fit_once(P, plane, hyper, t0_prior)
    if r is None or not np.isfinite(r['chi2']):
        return None
    # The per-event prompt-footprint choice (``sigma_p0_prof_<plane>``, rerun/det4 2026-10-25) is RETIRED (FE item 12,
    # 2026-10-27): on SPS head-on RAW the footprint is flat with amplitude (+-0.03 mm in thirds, every field) and the
    # per-event pick follows the noise (8-14 % choose >= 0.42 by scatter; ws_sps F18).  One footprint per view, the
    # bundle's sigma_p0_<plane>; a bundle that still carries a profile is fitted with its base value only.
    sig_used = float(hyper.get(f'sigma_p0_{plane}', hyper.get('sigma_p0', np.nan)))
    if TWO_SIDED:
        r = _two_sided(P, plane, r, cal, hyper, t0_prior=t0_prior)
    # Per-plane angle mapping (9dd7d6e; reverted by f9e18d2, restored 8-13).
    # w0/kw are measured from free fits of reference tracks; dropping the w0
    # term is the fleet angle bias, arctan(w0_plane/v) detector by detector.
    tan = tan_of(cal, plane, r['w'], r['p0'])
    ep, ew, et = _errors(P, plane, r, hyper, t0_prior=t0_prior)
    q_sum, q_u50, q_u90, q_uend = _profile_summary(r['q'])
    n_flagged, n_band, n_sat = window_flags(P, plane)
    f = PlaneFit(
        p0=float(r['p0']), w=float(r['w']), t0=float(r['t0']),
        tan_theta=float(tan), theta_deg=float(np.degrees(np.arctan(tan))),
        chi2=float(r['chi2']), dof=int(r['dof']),
        p0_err=float(np.hypot(ep, FLOOR_P0_MM)) if np.isfinite(ep) else FLOOR_P0_MM,
        w_err=float(ew) if np.isfinite(ew) else np.nan,
        tan_err=float(np.hypot(ew * 1e3 / cal.v_drift, tan_floor(cal, plane)))
        if np.isfinite(ew) else tan_floor(cal, plane),
        t0_err=float(et) if np.isfinite(et) else np.nan,
        q_sum=q_sum, q_u50=q_u50, q_u90=q_u90, q_uend=q_uend,
        n_strips=int(W.shape[0]), n_seed=int(n_seed), n_dropped=int(n_dropped),
        slope_reliable=bool(abs(tan) >= TAN_MIN_SLOPE),
        quality_ok=True,             # set relative to the table by quality_flags()
        n_flagged_strips=n_flagged, n_band_strips=n_band, n_sat_strips=n_sat)
    f._q = np.asarray(r['q'], float)     # the depth profile, for x/y pairing
    f._sigma_p0 = sig_used               # the prompt footprint the fit used (profile choice)
    return f


def _fit_once(P, plane, hyper, t0_prior):
    """One start + fit_plane_raw at fixed hypers (the pivot scan when PIVOT_START, else _global_start)."""
    p0_seed, _w0, t0_seed = wm.init_guess(P, plane)
    r = _pivot_fit(P, plane, hyper, t0_seed, t0_prior=t0_prior) if PIVOT_START else None
    if r is None:
        p0_seed, w_seed, t0_seed = _global_start(P, plane, p0_seed, t0_seed, hyper,
                                                 t0_prior=t0_prior)
        r = wm.fit_plane_raw(P, plane, p0_seed, w_seed, t0_seed, hyper=hyper,
                             t0_prior=t0_prior)
    return r


# --- candidate-cluster selection -------------------------------------------
# A track's charge column crosses the drift gap, so it lasts a few hundred ns
# and its transverse speed is bounded. Coherent noise and stray deposits do not
# satisfy both. Among candidates that do, take the one whose charge the model
# explains best (chi2 improvement over "no signal").
U_MIN_NS = 250.0
U_MAX_NS = 1100.0
#: The constants above are the column-duration window at v = 36.6 um/ns (30 mm gap: T = 820 ns; 0.30 T .. 1.35 T).
#: On a slow chamber or a low-field row the real column is longer (det6 24 um/ns: T = 1240 ns; the 500 V drift rows
#: ~1900 ns), so the fixed window called 42-60 % of det6/det7 Y tracks "implausible" (model audit 2026-10-27: the
#: ranking key and the multi-track gate then prefer a noise candidate whose q_uend happens to fall inside).  The
#: window scales with the bundle's v; the constants are used only when no v is known.
#: Window as fractions of the column T = 30 mm / v.  0.30 T reproduces U_MIN_NS at 36.6 um/ns.  The legacy upper
#: bound (1100 ns = 1.34 T) flagged 7-17 % of M3-confirmed det2/det3 single tracks (profile leakage into late bins when
#: t0 sits a bin early) and 23-60 % on det4/det6/det7; 1.6 T keeps the spike / double-column rejection and admits them.
U_MIN_FRAC, U_MAX_FRAC = 0.30, 1.60


def u_window(cal=None) -> tuple:
    """(u_min, u_max) [ns] of a plausible column end for this calibration's drift velocity."""
    v = float(getattr(cal, 'v_drift', 0.0) or 0.0) if cal is not None else float(getattr(wm.CAL, 'v_drift', 0.0) or 0.0)
    if not v > 0:
        return U_MIN_NS, U_MAX_NS
    T = 30000.0 / v                     # the 30 mm column at this v [ns]
    return U_MIN_FRAC * T, U_MAX_FRAC * T
# In the bundle's RAW tan (true = raw x k), so its true-angle reach moves with
# the bundle's v.  A versioned beam re-pass sets it wide (WFT_TAN_MAX, now a
# wft.diag override) and applies the angular acceptance as a true-angle cut at
# stage 3 (build_tracks --tan-max-true).
TAN_MAX = 0.6


def _candidate_score(P, plane, fit: PlaneFit) -> tuple:
    """(plausible, dchi2) for one candidate cluster's fit."""
    W, noise, pos, sat = wm.prep_plane(P, plane)
    chi_null = float(((W / noise[:, None]) ** 2)[~sat].sum())
    u = fit.q_uend
    u_lo, u_hi = u_window()
    plausible = (np.isfinite(u) and u_lo <= u <= u_hi
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
#   * a candidate whose split is not accepted comes out bit-identical to the
#     one-track reconstruction (reco.TWO_TRACK=False, a study override);
#   * the parent is kept in the candidates side table flagged ``split_replaced``
#     so downstream can undo any split;
#   * the false-split rate on clean single tracks is measured and bounded.
# ======================================================================== #

#: Part of the algorithm since 2026-10-27, with the separation floor at the
#: sharing kernel's reach (wft.model.TWO_MIN_SEP_MM, 4 pitches): the fleet gate
#: (rerun/model_audit/ws_unify/REPORT.md) bounds the split rate on clean bench
#: single muons on all five chambers.  False = the one-track reconstruction,
#: candidate for candidate (study override).
TWO_TRACK = True
#: Accept the split when the statistic (``fit_plane_two``'s ``fstat``: the
#: SMALLER of the two children's marginal chi2 improvements, in units of the
#: one-track fit's own chi2/dof) exceeds this. NOT a Wilks number: chi2/dof is
#: 1.4-6.8 on clean single tracks here and 5-12 across the full pass, so a
#: threshold from the asymptotic distribution would split almost everything.
#: Chosen from the efficiency-vs-false-split curve of
#: sept26_prelim_analysis/two_track_synth.py and `intra_bench split-probe`.
TWO_TRACK_F = 300.0
#: The same threshold when the other plane resolves two time-coincident
#: plausible candidates where THIS plane resolves fewer (see
#: :func:`_cross_plane_mismatch`). Two tracks must show in both views, so a
#: count mismatch is real external evidence that this plane has merged them and
#: the bar can be lower. Without the mismatch half of that condition the
#: discount fires on the easy case and destroys correct tracks.
TWO_TRACK_F_CORROB = 120.0
#: Candidates per plane the fit is attempted on, best-ranked first.
TWO_TRACK_MAX_TRY = 2
#: Trigger: the window's charge is wider than one track of the fitted |tan| and
#: column duration can cover, by more than this margin [mm]. Measured on clean
#: single tracks -- `intra_bench split-probe`.
TWO_TRACK_WIDTH_MARGIN = 6.0
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
TWO_TRACK_T0 = 'tied'
#: t0 offsets, relative to the parent's, at which the residual is scanned for a
#: second track [ns]. One depth bin apart; wider only when t0 is free.
T0_RESID_SCAN = np.arange(-60.0, 60.1, 60.0)
T0_RESID_SCAN_FREE = np.arange(-180.0, 180.1, 60.0)
#: Trigger: the largest per-strip coherent positive residual the one-track fit
#: leaves, in sigma after de-scaling by the fit's own chi2/dof. Calibrated on
#: clean single tracks -- `intra_bench split-probe`.
TWO_TRACK_RESID_Z = 8.0
#: Only reconsider candidates the selector actually chose (members of a pair
#: from :func:`select_tracks`). Measured on 13 071 real run_145 candidates:
#: this is 25 % of them and holds 42 % of the splits, the rest being extra
#: clusters in already-busy events -- four candidates a plane, 45 strips,
#: chi2/dof 13. The default was set to save compute, which is no longer a
#: constraint (2026-09-16): whether those other splits are real pairs is
#: unmeasured, and this is on the to-do (wft/TWO_TRACK_FIT_2026-09-16.md §8).
#: A study reconsiders every candidate with WFT_DIAG=reco.TWO_TRACK_SELECTED_ONLY=0.
TWO_TRACK_SELECTED_ONLY = True
#: Where the split statistic's chi2/dof scale comes from. 'one' (production):
#: the parent one-track fit's -- which on a real pair already holds the second
#: track's unexplained charge, so fstat ~ dchi2 / (1 + dchi2/dof) can never
#: exceed ~dof, and a narrow window (a close vertical pair: ~12 strips x 20
#: samples) cannot reach TWO_TRACK_F = 300 however clear the pair is
#: (two_track_limit.py, 2026-09-29). 'two': the two-track fit's, which equals
#: the parent's on a true single and is not inflated by the thing being tested.
TWO_TRACK_SCALE = 'one'
#: How the joint fit is started. 'starts' (production): alternating pursuit from
#: the one-track parent plus three symmetric splits, best two refined. On close
#: pairs the parent is a slanted compromise line and the refinement falls into
#: an "X" of two crossing children, chi2 hundreds above the true pair (27/27
#: synthetic misses, two_track_limit.py 2026-09-30). 'grid': additionally scan
#: every pair of parallel lines on a strip-step grid (common slope, tied t0),
#: profiles by NNLS, and refine the best TWO_TRACK_GRID_KEEP of them with a
#: long Nelder-Mead. Costs 2-10 k chi2 evaluations per attempt.
TWO_TRACK_SEARCH = 'starts'
TWO_TRACK_GRID_TANS = np.round(np.arange(-0.4, 0.401, 0.1), 3)
TWO_TRACK_GRID_DT = (-60.0, 0.0, 60.0)
TWO_TRACK_GRID_MAX_MM = 14.0
TWO_TRACK_GRID_KEEP = 3

#: Threshold ladder, for the split-ab F rescan (TWO_TRACK_LIMIT_RESUME.md):
#: comma-separated F values, '' = off. Neither the attempts nor their fstat
#: depend on the threshold -- only acceptance does -- so one pass replays every
#: F exactly (:func:`two_track_ladder`). The corroborated threshold scales with
#: F at the primary run's TWO_TRACK_F_CORROB / TWO_TRACK_F ratio. The primary
#: output is untouched.
TWO_TRACK_F_LADDER = os.environ.get('WFT_TWO_TRACK_F_LADDER', '')


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
    tan = tan_of(cal, plane, w, p0)
    q_sum, q_u50, q_u90, q_uend = _profile_summary(q)
    n_flagged, n_band, n_sat = window_flags(P, plane)
    f = PlaneFit(
        p0=float(p0), w=float(w), t0=float(t0), tan_theta=float(tan),
        theta_deg=float(np.degrees(np.arctan(tan))),
        chi2=float(r['chi2']), dof=int(r['dof']),
        p0_err=float(np.hypot(ep, FLOOR_P0_MM)) if np.isfinite(ep) else FLOOR_P0_MM,
        w_err=float(ew) if np.isfinite(ew) else np.nan,
        tan_err=float(np.hypot(ew * 1e3 / cal.v_drift, tan_floor(cal, plane)))
        if np.isfinite(ew) else tan_floor(cal, plane),
        t0_err=float(et) if np.isfinite(et) else np.nan,
        q_sum=q_sum, q_u50=q_u50, q_u90=q_u90, q_uend=q_uend,
        n_strips=int(np.asarray(P['W']).shape[0]), n_seed=0, n_dropped=0,
        slope_reliable=bool(abs(tan) >= TAN_MIN_SLOPE),
        quality_ok=True,             # set relative to the table by quality_flags()
        n_flagged_strips=n_flagged, n_band_strips=n_band, n_sat_strips=n_sat)
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
    u_lo, u_hi = u_window(cal)
    plaus = [bool(np.isfinite(f.q_uend) and u_lo <= f.q_uend <= u_hi
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
    return any(abs(col_dt(ok[i], ok[j])) <= TC_TOL_NS
               for i in range(len(ok)) for j in range(i + 1, len(ok)))


def resolve_two_tracks(all_fits: Dict[str, list], windows: Dict[str, list],
                       cal: CalibrationBundle, ftst_diff=None,
                       max_try: int = TWO_TRACK_MAX_TRY,
                       selected: Optional[Dict[str, set]] = None,
                       attempts: Optional[list] = None) -> tuple:
    """Offer the merged-looking candidates of both planes a second track.

    Runs AFTER the ordinary per-plane candidate fits, so the one-track answer
    is always available and a candidate whose split is refused is untouched.
    Accepted children replace their parent in the plane's ranked list; the
    parent is returned separately for the side table.

    Returns ``(all_fits, splits, replaced)``: the possibly-rewritten ranked
    lists, one record per attempt (provenance, and the bench's input), and the
    parents a split displaced, per plane, for the candidates side table.

    ``attempts``, when a list, also collects every attempt WITH its two
    children, accepted or not -- the input of :func:`two_track_ladder`."""
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
            # the other plane's t0 in this plane's clock: same latency, ftst phase difference (ws_t0 F1)
            sh = -FTST_NS * float(ftst_diff) if ftst_diff is not None else 0.0
            s = 1.0 if plane == 'x' else -1.0
            hints = [f.t0 + s * sh for f in (all_fits.get(other) or [])[:2]
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
            if attempts is not None:
                attempts.append(dict(plane=plane, parent=f, win=iw,
                                     children=r['children'], rec=rec))
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


def two_track_ladder(kept: Dict[str, list], attempts: list, ftst_diff,
                     ladder) -> list:
    """Replay the accept/reject of every attempt at each F of ``ladder``.

    ``kept`` is the one-track candidate lists (before any split), ``attempts``
    what :func:`resolve_two_tracks` collected. For each F: replace the parent of
    every attempt with ``fstat >= F`` (``F_corrob`` when corroborated) and its
    guards passed by its two children, apply the same lost-track revert as
    :func:`_worker_fit`, and run the selector. Builds new lists and touches no
    attribute the primary output reads, so the primary answer is unchanged.

    Returns one dict per F: n_splits, reverted, n_tracks, and the gated tracks
    as (x p0, y p0, x tan, y tan)."""
    ratio = TWO_TRACK_F_CORROB / TWO_TRACK_F if TWO_TRACK_F else 0.4
    n_kept = None
    out_rows = []
    for F in ladder:
        Fc = F * ratio
        acc = [a for a in attempts
               if a['rec']['guards_ok']
               and a['rec']['fstat'] >= (Fc if a['rec']['corroborated'] else F)]
        fits = {p: list(kept.get(p) or []) for p in ('x', 'y')}
        for a in acc:
            ca, cb = a['children']
            ca.n_seed = cb.n_seed = a['parent'].n_seed
            ca.n_dropped = cb.n_dropped = a['parent'].n_dropped
            ca._win = cb._win = a['win']
            lst = fits[a['plane']]
            i = next(k for k, g in enumerate(lst) if g is a['parent'])
            lst[i:i + 1] = [ca, cb]
        for p in ('x', 'y'):
            fits[p].sort(key=lambda g: (1 if getattr(g, '_plausible', True) else 0,
                                        getattr(g, '_dchi2', 0.0) or 0.0),
                         reverse=True)
        reverted = False
        if acc:
            if n_kept is None:
                n_kept = _n_gated(kept, ftst_diff)
            if n_kept > _n_gated(fits, ftst_diff):
                fits, reverted = kept, True
        try:
            pairs = select_tracks(fits, ftst_diff, _CAL, max_tracks=MAX_TRACKS,
                                  pairing=_PAIRING)
        except Exception:
            pairs = []
        gated = [(float(fits['x'][i].p0), float(fits['y'][j].p0),
                  float(fits['x'][i].tan_theta), float(fits['y'][j].tan_theta))
                 for i, j, g in pairs if g]
        out_rows.append(dict(F=float(F), F_corrob=float(Fc),
                             n_splits=0 if reverted else len(acc),
                             reverted=reverted, n_tracks=len(gated), gated=gated))
    return out_rows


#: x-y coincidence on the COLUMN TIME (model audit follow-up FE, item 7 / F5, 2026-10-27).  Both views read the same
#: drifting electrons, so the median arrival of the fitted charge, t0 + u50, phase-corrected by the trigger's ftst
#: (+ FTST_NS x ftst per view, the DREAM clock phase; ws_t0 F1), is the same time in x and y up to a fixed FEU offset that
#: measures -2..+6 ns on every bench chamber: the comparison is made at ZERO offset, with no ftst table.  It replaces the
#: t0x - t0y gate against the bundle's ``dt_xy``: a free t0 slides by whole 60 ns depth bins along the t0 <-> p0
#: degeneracy (F6), so that gate needed a 50-140 ns width and an offset table that was itself biased by the tooth; with t0
#: pinned at the trigger latency (bench bundles) a t0 gate would check nothing at all.  u50 is invariant under the tooth.
#: Tolerance: the bundle's ``xy_pairing['tc_tol']`` (p98 of |dtc| on clean single tracks of that chamber and condition,
#: written by rerun/w12/set_w0kw.py --table), else TC_TOL_DEFAULT_NS.  Set per worker by :func:`_worker_init`.
FTST_NS = 10.0
TC_TOL_DEFAULT_NS = 200.0
TC_TOL_NS = TC_TOL_DEFAULT_NS
#: legacy names (the t0 gate is retired); kept so old study scripts that read them still import
DT_XY_TOL_NS = TC_TOL_DEFAULT_NS
DT_XY_TOL_DEFAULT_NS = TC_TOL_DEFAULT_NS
XY_TOL_NSIG = 2.5


def tc_tol(pairing: Optional[dict]) -> float:
    """The x-y column-time tolerance [ns] a bundle's ``xy_pairing`` implies."""
    v = (pairing or {}).get('tc_tol')
    return float(v) if v is not None and np.isfinite(v) and v > 0 else TC_TOL_DEFAULT_NS


dt_xy_tol = tc_tol      # legacy name


def _u50(f) -> float:
    u = getattr(f, 'q_u50', np.nan)
    return float(u) if u is not None and np.isfinite(u) else np.nan


def col_time(f) -> float:
    """t0 + u50 of one fit [ns, window clock] (u50 dropped when the profile is empty)."""
    u = _u50(f)
    return float(f.t0 + (u if np.isfinite(u) else 0.0))


def col_dt(fx, fy, ftst_diff=None) -> float:
    """Phase-corrected column-time difference x - y [ns]; ~0 for one particle.  ``ftst_diff`` = ftst_x - ftst_y.
    Without a profile in one view (u50 undefined) the t0s are compared."""
    if np.isfinite(_u50(fx)) and np.isfinite(_u50(fy)):
        d = col_time(fx) - col_time(fy)
    else:
        d = float(fx.t0 - fy.t0)
    return d + (FTST_NS * float(ftst_diff) if ftst_diff is not None else 0.0)


def _coincident(fx, fy, ftst_diff) -> bool:
    return bool(abs(col_dt(fx, fy, ftst_diff)) <= TC_TOL_NS)


#: x-y charge balance per EVENT CLASS (FE item 11, ws_dead_charge F15).  One width per chamber mixed three populations:
#: clean tracks (rsigma ln(qx/qy) 0.05-0.07), windows with a censored strip (dead or band: 1.3-3x wider) and saturated
#: tracks (clipped charge in one view; up to 0.10-0.20 on det6/det7).  The bundle's ``xy_pairing['lq_class']`` carries
#: {class: {median, rsig, n}}; a pair's class is the worst of its two views (sat > cens > clean).  Absent = the bundle's
#: one 'lq' width.
PAIR_CLASSES = ('clean', 'cens', 'sat')


def view_class(f) -> str:
    if getattr(f, 'n_sat_strips', 0):
        return 'sat'
    if getattr(f, 'n_flagged_strips', 0) or getattr(f, 'n_band_strips', 0):
        return 'cens'
    return 'clean'


def pairing_class(fx, fy) -> str:
    return max(view_class(fx), view_class(fy), key=PAIR_CLASSES.index)


def lq_balance(pairing: Optional[dict], cls: str):
    """(median, rsig) of ln(qx/qy) for this event class, or (None, None)."""
    p = pairing or {}
    c = (p.get('lq_class') or {}).get(cls) or {}
    if c.get('rsig'):
        return float(c['median']), float(c['rsig'])
    if 'lq' in (p.get('rsig') or {}):
        return float(p['median']['lq']), float(p['rsig']['lq'])
    return None, None


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
            coincident = int(_coincident(fx, fy, ftst_diff))
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
        out = _repair_pairs(out, cand_fits, pairing, dt, ftst_diff)
    return out


def _gate(fx, fy, ftst_diff) -> bool:
    return (_coincident(fx, fy, ftst_diff)
            and bool(getattr(fx, '_plausible', True))
            and bool(getattr(fy, '_plausible', True)))


def xy_pair_cost(fx, fy, pairing: dict, dt: float, ftst_diff=None) -> float:
    """How unlike an x and a y candidate are, in units of the spread of the
    same x-minus-y quantity on clean single tracks.  The charge balance (lq, lqc) is
    judged against its own event class's width (pairing_class, ``lq_class``); 'tc' is the
    column-time difference (zero offset), 't0' the retired t0 residual against ``dt``."""
    f = dict(lq=np.log(max(fx.q_sum, 1.0) / max(fy.q_sum, 1.0)),
             u50=fx.q_u50 - fy.q_u50, u90=fx.q_u90 - fy.q_u90,
             t0=(fx.t0 - fy.t0) - dt, tc=col_dt(fx, fy, ftst_diff))
    med_lq, rs_lq = lq_balance(pairing, pairing_class(fx, fy)) if pairing.get('lq_class') else (None, None)
    if 'lqc' in pairing['features']:
        cx, cy = constrained_charge(fx), constrained_charge(fy)
        f['lqc'] = (np.log(max(cx, 1.0) / max(cy, 1.0))
                    if np.isfinite(cx) and np.isfinite(cy) else np.nan)
    cost = 0.0
    for k in pairing['features']:
        if k in ('lq', 'lqc') and rs_lq:
            z = (f[k] - med_lq) / rs_lq
        else:
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


def _repair_pairs(out: list, cand_fits: Dict[str, list], pairing: dict, dt: float, ftst_diff=None) -> list:
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
                if not (ga and gb and _gate(xs[ia], ys[jb], ftst_diff) and _gate(xs[ib], ys[ja], ftst_diff)):
                    continue
                c = lambda a, b: xy_pair_cost(a, b, pairing, dt, ftst_diff)
                keep = c(xs[ia], ys[ja]) + c(xs[ib], ys[jb])
                swap = c(xs[ia], ys[jb]) + c(xs[ib], ys[ja])
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
                                      and f.q_uend < u_window()[0])
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
            coincident = _coincident(fx, fy, ftst_diff)
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
                     'n_flagged_strips', 'n_band_strips', 'n_sat_strips'):
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
                np.isfinite(f.q_uend) and f.q_uend < u_window()[0])
    return row


# --------------------------------------------------------------- the driver
#: Module globals a worker may be started with. Everything here is otherwise an
#: environment variable read at import, which a forked worker cannot be told
#: about after the fact -- so an A/B harness passes them here instead.
WORKER_OPTS = ('TWO_TRACK', 'TWO_TRACK_F', 'TWO_TRACK_F_CORROB', 'TWO_TRACK_T0',
               'TWO_TRACK_RESID_Z', 'TWO_TRACK_WIDTH_MARGIN', 'TWO_TRACK_MAX_TRY',
               'TWO_TRACK_SELECTED_ONLY', 'TWO_TRACK_SCALE', 'TWO_TRACK_SEARCH',
               'TWO_TRACK_F_LADDER', 'PIVOT_START', 'HEADON_RESCUE', 'HEADON_RESCUE_Z')


def _worker_init(bundle_path, pairing_path=None, opts=None):
    """``pairing_path`` overrides the bundle's ``xy_pairing`` (for A/B runs);
    ``opts`` overrides the :data:`WORKER_OPTS` module globals (likewise)."""
    global _CAL, _PAIRING, TC_TOL_NS
    _CAL = CalibrationBundle.load(bundle_path)
    wm.use_calibration(_CAL)
    _PAIRING = load_pairing(pairing_path) if pairing_path else (_CAL.xy_pairing or None)
    TC_TOL_NS = tc_tol(_CAL.xy_pairing)
    for k, v in (opts or {}).items():
        if k not in WORKER_OPTS:
            raise KeyError(f'not a worker option: {k!r}; known: {WORKER_OPTS}')
        globals()[k] = v


PAIR_SELECT = False       # legacy single-pair selector (beam displays); select_tracks decides
EMIT_CANDIDATES = os.environ.get('WFT_EMIT_CANDIDATES', '1') == '1'
MAX_TRACKS = 3
_PAIRING: Optional[dict] = None


#: Cross-view head-on rescue (opt-in, rerun/satstudy 2026-10-26). When one plane has a fitted track and the other
#: has no seed at all, look in that plane's full waveform for a drift column starting at the fitted track's t0
#: (wft.seed.column_rescue, Z >= HEADON_RESCUE_Z), fit it as any candidate, and keep it only if its charge matches the
#: other view's: |ln(q_x / q_y) - median| <= HEADON_RESCUE_QTOL x rsig. One particle leaves the same charge in both
#: views (robust sigma 0.054 on det3 bench singles), which is what keeps 1-strip / low-SNR seeds from turning noise
#: into tracks. The tolerance widens with the rescued cluster's own charge noise, rsig -> hypot(rsig, sigma_Q / Q) over
#: the column window, and Q > 3 sigma_Q is required (headon2.py '+qa': x, noise x2, gain x0.05: 79 -> 84 % found). median / rsig: the bundle's xy_pairing 'lq' if present, else the det3 bench values below.
#: Part of the algorithm since 2026-10-27 (fleet gate, with seed.HEADON_SIG: +0.3..+3.7 pt detected, det4/det7 the
#: most, no cost to the existing tracks).  Every number is in units of a measured width (noise, the bundle's charge
#: balance), so the rule is the same on every chamber; a bundle without xy_pairing falls back to det3's balance.
HEADON_RESCUE = True
HEADON_RESCUE_Z = 10.0
HEADON_RESCUE_QTOL = 3.0
HEADON_RESCUE_LQ = (-0.059, 0.054)


def _headon_rescue(R, plane, other):
    """R: {'W', 'noise', 'pos'} of a plane without seeds; other: the other plane's best fit. -> a PlaneFit or None."""
    from . import io as wio, seed as wseed
    Wr = np.asarray(R['W'])
    band = wm.band_rows(Wr, R['noise'])
    if band.any():                       # a noise band is not a drift column (wft.model.BAND_NSIG)
        Wr = Wr.copy()
        Wr[band] = 0.0
    ch = wseed.column_rescue(Wr, R['noise'], R['pos'], float(other.t0), z_min=HEADON_RESCUE_Z)
    if ch is None:
        return None
    win = wio.extract_window(R['W'], R['noise'], R['pos'], ch, 3, raw_all=R.get('raw'))
    if win is None:
        return None
    P = dict(W=win.W, pos=win.pos, noise=win.noise, ch=win.ch)
    if win.rail_hi is not None:
        P.update(rail_hi=win.rail_hi, rail_lo=win.rail_lo)
    f = fit_plane_candidates([P], plane, _CAL)
    if f is None or not np.isfinite(f.q_sum) or f.q_sum <= 0 or not other.q_sum > 0:
        return None
    lq = np.log(f.q_sum / other.q_sum) if plane == 'x' else np.log(other.q_sum / f.q_sum)
    fx, fy = (f, other) if plane == 'x' else (other, f)
    med, rs = lq_balance(_PAIRING, pairing_class(fx, fy))
    med, rs = (med, rs) if med is not None and rs else HEADON_RESCUE_LQ
    # the cluster's own charge noise widens the tolerance (low gain / high noise): sigma_Q / Q over the column window
    W, nz = R['W'][ch], np.asarray(R['noise'], float)[ch]
    t = np.arange(W.shape[1]) * 60.0
    w = (t >= other.t0 - 120.0) & (t <= other.t0 + 1020.0)
    q = float((W[:, w] - np.median(W[:, ~w], axis=1)[:, None]).sum()) if (~w).sum() >= 4 else 0.0
    sq = float(np.sqrt((nz ** 2).sum() * w.sum()))
    if q <= 3 * sq or abs(lq - med) > HEADON_RESCUE_QTOL * np.hypot(rs, sq / q):
        return None
    f._rescued = True
    return f, P


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
    if HEADON_RESCUE:
        for plane, other in (('x', 'y'), ('y', 'x')):
            R = wins.get(f'_rescue_{plane}')
            if R is None or fits[plane] is not None or fits[other] is None:
                continue
            try:
                r = _headon_rescue(R, plane, fits[other])
            except Exception:
                r = None
            if r is not None:
                fits[plane], all_fits[plane] = r[0], [r[0]]
                wins[plane] = [r[1]]
    ftst_diff = (ftst['x'] - ftst['y']
                 if ftst.get('x') is not None and ftst.get('y') is not None
                 else None)
    splits, replaced, attempts, ladder = [], {}, None, None
    if TWO_TRACK:
        try:
            kept = all_fits
            attempts = [] if TWO_TRACK_F_LADDER else None
            all_fits, splits, replaced = resolve_two_tracks(
                all_fits, wins, _CAL, ftst_diff, attempts=attempts)
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
            splits, replaced, attempts = [], {}, None
        if attempts is not None:
            try:
                ladder = two_track_ladder(
                    kept, attempts, ftst_diff,
                    [float(v) for v in str(TWO_TRACK_F_LADDER).split(',') if v.strip()])
            except Exception:
                ladder = None
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
    # the bow block maps each track's tangents with both views' coordinates (pairs first, then the event's winners)
    if getattr(_CAL, 'bow', None) or (getattr(_CAL, 'v_map', None) or {}).get('xy'):
        for ix, iy, _g in pairs:
            try:
                apply_bow(_CAL, all_fits['x'][ix], all_fits['y'][iy])
            except Exception:
                pass
        if fits.get('x') is not None or fits.get('y') is not None:
            apply_bow(_CAL, fits.get('x'), fits.get('y'))
    row = row_from_fits(eid, fits, n_hits, spark)
    row['n_tracks'] = int(sum(1 for _i, _j, g in pairs if g))
    hb = [getattr(fits.get(p), '_bow_h', None) for p in ('x', 'y')]
    row['bow_h'] = float(next((h for h in hb if h is not None), np.nan))
    for p, h in zip('xy', hb):            # each view's own mesh height: the position at the plane is p0 - tan * h
        row[f'{p}_bow_h'] = float(h) if h is not None else np.nan
    for plane in ('x', 'y'):
        f = ftst.get(plane)
        row[f'{plane}_ftst'] = int(f) if f is not None else -1
    if TWO_TRACK:
        row['n_splits'] = int(sum(1 for r in splits if r['accepted']))
    if EMIT_CANDIDATES:
        row['_cand'] = candidate_rows(eid, all_fits, pairs, ftst, replaced)
    if splits:
        row['_splits'] = [dict(event_id=int(eid), **r) for r in splits]
    if ladder is not None:
        row['_ladder'] = ladder
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
    hits, ped_guard = pedestal_guard(hits, cfg)
    if verbose and ped_guard['corrected']:
        print(f"[wft] pedestal guard: {ped_guard['corrected']}", flush=True)
    dead = (getattr(cal, 'dead', None) or {}) if wseed.DEAD_SEED else {}
    seeds = wseed.seeds_from_hits(hits, pos_maps, feu_x, feu_y, hot=cal.hot, dead=dead)
    wanted = set(seeds)
    if event_filter is not None:
        wanted &= set(int(e) for e in event_filter)
    sparks = {e for e in wanted if seeds[e]['spark']}
    wanted = {e for e in wanted
              if not seeds[e]['spark'] and (seeds[e]['x'] or seeds[e]['y'])}
    if limit:
        wanted = set(sorted(wanted)[:limit])
        sparks = {e for e in sparks if wanted and e <= max(wanted)}
    # noise bands (FE item 13): spark-vetoed events are read too and fitted if, without the hits on their band
    # strips, they are not sparks (band_reseed); the selection of ordinary events (and --limit) is unchanged
    n_spark_checked = len(sparks) if wm.BAND_NSIG > 0 else 0
    if wm.BAND_NSIG > 0:
        wanted |= sparks
        hits = hits.set_index('eventId')
        hits = hits[hits.index.isin(wanted)]
    else:
        hits = None
    if verbose:
        print(f'[wft] {len(seeds):,} seeded events, {len(wanted):,} to reconstruct',
              flush=True)

    rows, cand_rows, split_rows = [], [], []
    with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init,
                             initargs=(bundle_path,)) as pool:
        evtime = {}
        for payloads in _stream_windows(cfg, pos_maps, seeds, wanted, pad_strips,
                                        verbose=verbose, dead=dead, hits=hits, hot=cal.hot, evtime=evtime):
            if not payloads:
                continue
            for r in pool.map(_worker_fit, payloads, chunksize=8):
                # when the event was taken (GC 2026-10-28): its acquisition (file tag; eventId restarts between the
                # acquisitions of one subrun) and DREAM timestamp; v(t) is applied to the tables from these afterwards
                ts, tag = evtime.get(int(r['event_id']), (-1, ''))
                for c in r.get('_cand', []):
                    c['acq_tag'] = tag
                cand_rows.extend(r.pop('_cand', []))
                split_rows.extend(r.pop('_splits', []))
                r['timestamp'], r['acq_tag'] = ts, tag
                rows.append(r)
            if verbose:
                print(f'[wft]   {len(rows):,} events reconstructed', flush=True)

    df = pd.DataFrame(rows).sort_values('event_id').reset_index(drop=True)
    df, q_med = quality_flags(df)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    if split_rows:
        pd.DataFrame(split_rows).sort_values(['event_id', 'plane']).reset_index(
            drop=True).to_parquet(out_path.replace('.parquet', '.splits.parquet'),
                                  index=False)
    cand_path = out_path.replace('.parquet', '.candidates.parquet')
    if cand_rows:
        quality_flags(pd.DataFrame(cand_rows), q_med)[0].sort_values(
            ['event_id', 'plane', 'rank']).reset_index(drop=True).to_parquet(
            cand_path, index=False)
    from . import diag as wdiag
    meta = dict(n_events=len(df), calibration=bundle_path, quality_rule=quality_rule(q_med),
                # the algorithm is the code; anything else is a study and says so here
                diagnostic_overrides=dict(wdiag.ACTIVE),
                algorithm=dict(pivot_start=PIVOT_START, two_sided=TWO_SIDED, col_guard=wm.COL_GUARD,
                               sat_solve=wm.SAT_SOLVE, rc_t0_interp=wm.RC_T0_INTERP, model_frac=wm.MODEL_FRAC,
                               two_track_min_sep_mm=wm.TWO_MIN_SEP_MM,
                               xy_gate='column time t0+u50, zero offset',
                               tc_tol_ns=tc_tol(cal.xy_pairing),
                               band_nsig=wm.BAND_NSIG,
                               sigma_p0_profile=False),
                bundle=dict(detector=cal.detector, run_key=cal.run_key,
                            v_drift=cal.v_drift, hyper=cal.hyper,
                            conditions=cal.conditions,
                            provenance=cal.provenance),
                # Which angle mapping this table is on. A bundle that lost its
                # constants still reconstructs, but on the UNCORRECTED mapping,
                # and nothing downstream could tell -- that is how the fleet
                # bias survived a whole campaign. applied=False here means the
                # angles need the post-hoc w0/kw pass before they are quoted.
                bow=dict(applied=bool(getattr(cal, 'bow', None)),
                         source=(getattr(cal, 'bow', None) or {}).get('source_row')),
                angle_constants=dict(applied=bool(cal.w0 or cal.kw),
                                     w0=dict(cal.w0), kw=dict(cal.kw),
                                     angle_map=sorted((getattr(cal, 'angle_map', None) or {}).keys())),
                t0_prior=dict(sigma=T0_PRIOR_SIGMA if T0_PRIOR_SIGMA is not None
                              else cal.t0_prior_sigma,
                              t0_abs_planes=sorted((T0_ABS or cal.t0_abs
                                                    or {}).keys())),
                run=dict(key=getattr(cfg, 'KEY', ''), run=cfg.RUN,
                         sub_run=cfg.SUB_RUN, detector=cfg.DET_NAME,
                         feu_x=feu_x, feu_y=feu_y),
                selection=dict(u_window_ns=list(u_window(cal)), sig_rel_floor=wseed.SIG_REL_FLOOR,
                               gap_mm=wseed.GAP_THRESHOLD_MM,
                               sig_floor_local_mm=wseed.SIG_FLOOR_LOCAL_MM,
                               spark_veto=wseed.SPARK_VETO_HITS,
                               pad_strips=pad_strips,
                               dead_seed={p: len(c) for p, c in dead.items()},
                               headon_sig=wseed.HEADON_SIG, drop_flat_hits=DROP_FLAT_HITS,
                               pedestal_guard=ped_guard,
                               band=dict(nsig=wm.BAND_NSIG, edge=wm.BAND_EDGE, spark_events_checked=n_spark_checked,
                                         n_reseeded=int(sum(1 for r in seeds.values() if r.get('band_reseeded')))),
                               headon_rescue=dict(on=HEADON_RESCUE, z=HEADON_RESCUE_Z, qtol=HEADON_RESCUE_QTOL),
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


#: Drop every hit that is above threshold for the whole window (trunc_left AND trunc_right) before seeding (opt-in,
#: rerun/satstudy/bands.py 2026-10-26). A particle's signal lasts at most a drift column (~20 samples); a strip above
#: threshold from the first sample to the last is a baseline offset -- the "noise bands", a chamber effect present in
#: 3-19 % of bench triggers -- and otherwise forms competing clusters (n_dropped) and trips the spark veto. 84 % of the
#: hits on band strips, 0.7 % of all others. 'left': drop every hit already above threshold at the FIRST sample
#: (trunc_left), whatever it does later -- 92 % of band hits, 2.4 % of others; the triggering particle's charge
#: cannot be there before its own t0.  'both' is part of the algorithm since 2026-10-27 (fleet gate: +0.0..+1.1 pt
#: detected, rejected_cut -0.3..-0.7 pt, resolution of the existing tracks unchanged); None = the historical hit list.
DROP_FLAT_HITS = 'both'


#: Pedestal mismatch guard for the seeding hits (F11, rerun/precheck_ga/README.md, 2026-10-29).  OFF (0): opt-in study
#: switch, bit-identical when off.  The analyzer's significance is amplitude / the pedestal run's noise.  Per FEU connector
#: (64 channels): r = median over channels of median(amplitude / significance) / the noise of the run's own waveforms
#: (wft.io.FeuReader, first data file); a connector with r < PED_MISMATCH_REL gets its significance scaled by r and the
#: analyzer's floor HIT_SIG re-applied.  Not adopted: g_det4's FEU 6 connectors 7-8 (6-23 pedestal, unplugged) read
#: r = 0.5, but so do the golden det7 row and the high-HV det3/det6 rows (fleet: 5 % of connectors < 0.74, 70 < 0.7;
#: the data run's noise with HV on is above the pedestal run's), so the guard would change seeding fleet-wide; it would
#: need the full-reco gate first.  Stated limitation: sub-threshold noise hits seed weak events there.
PED_MISMATCH_REL = 0
HIT_SIG = 5.0


def pedestal_guard(hits, cfg, files_by_feu=None):
    """(hits, report): the seeding hits with the PED_MISMATCH_REL correction applied (see above).
    report = {feu: {connector: r}} for every corrected connector, plus the ratios of all connectors under 'ratios'."""
    from . import io as wio
    report = {'rel': PED_MISMATCH_REL, 'hit_sig': HIT_SIG, 'corrected': {}, 'ratios': {}}
    if not PED_MISMATCH_REL or hits is None or not len(hits):
        return hits, report
    if files_by_feu is None:
        files_by_feu = {feu: wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu)
                        for feu in (cfg.MX17_FEU_X, cfg.MX17_FEU_Y)}
    keep = np.ones(len(hits), dtype=bool)
    sig = hits['significance'].to_numpy(dtype=float).copy()
    feu_a, ch_a = hits['feu'].to_numpy(), hits['channel'].to_numpy().astype(int)
    amp = hits['amplitude'].to_numpy(dtype=float)
    for feu, files in files_by_feu.items():
        if not files:
            continue
        noise = wio.FeuReader(sorted(files)[0]).noise
        on = (feu_a == feu) & (sig > 0) & (amp > 0)
        if not on.any():
            continue
        ch, q = ch_a[on], amp[on] / sig[on]
        ped_noise = np.full(512, np.nan)
        cnt = np.bincount(ch, minlength=512)
        order = np.argsort(ch, kind='stable')
        starts = np.r_[0, np.cumsum(cnt)[:-1]]
        qs = q[order]
        for c in np.flatnonzero(cnt >= 20):
            ped_noise[c] = np.median(qs[starts[c]:starts[c] + cnt[c]])
        rat = {}
        for k in range(8):
            cs = np.arange(64 * k, 64 * k + 64)
            v = ped_noise[cs] / np.where(noise[cs] > 0, noise[cs], np.nan)
            v = v[np.isfinite(v)]
            if len(v) >= 16:
                rat[k + 1] = float(np.median(v))
        report['ratios'][int(feu)] = {k: round(r, 3) for k, r in rat.items()}
        for k, r in rat.items():
            if r < PED_MISMATCH_REL:
                m = (feu_a == feu) & (ch_a >= 64 * (k - 1)) & (ch_a < 64 * k)
                sig[m] *= r
                keep &= ~(m & (sig < HIT_SIG))
                report['corrected'].setdefault(int(feu), {})[k] = dict(r=round(r, 3), n_removed=int((m & (sig < HIT_SIG)).sum()))
    if not report['corrected']:
        return hits, report
    out = hits.copy()
    out['significance'] = sig
    return out[keep], report


def _load_hits(cfg):
    """Combined hits for the run — used ONLY for seeding (see wft.seed)."""
    import uproot
    import pandas as pd
    files = [f for f in os.listdir(cfg.combined_hits_dir)
             if f.endswith('.root') and '_datrun_' in f]
    cols = ['eventId', 'feu', 'channel', 'amplitude', 'significance']
    df = uproot.concatenate(
        [f'{cfg.combined_hits_dir}{f}:hits' for f in files],
        expressions=cols + (['trunc_left', 'trunc_right'] if DROP_FLAT_HITS else []),
        library='pd')
    df = df[df['feu'].isin(cfg.MX17_FEUS)]
    if DROP_FLAT_HITS == 'both':
        df = df[~(df['trunc_left'] & df['trunc_right'])][cols]
    elif DROP_FLAT_HITS == 'left':
        df = df[~df['trunc_left']][cols]
    return df


def band_reseed(eid, hits_ev, band_ch: dict, pos_maps, feu_x, feu_y, hot, dead, seeder=None):
    """Seeds of one event with the hits on its noise-band strips removed (FE item 13).

    ``band_ch``: {feu: set of channels whose waveform is a band in this event} (wft.model.band_rows on the full event).
    A band strip that kept a hit (not above threshold from first to last sample, so DROP_FLAT_HITS let it through) still
    forms a competing cluster (n_dropped > 2, the event is cut) or pushes the event over the spark veto (no fit at all);
    without it the event is seeded on its particle.  Returns the new seed record, or None if no hit sat on a band strip."""
    from . import seed as wseed
    if hits_ev is None or not len(hits_ev):
        return None
    on_band = np.zeros(len(hits_ev), dtype=bool)
    for feu, chs in band_ch.items():
        if chs:
            on_band |= (hits_ev['feu'].to_numpy() == feu) & np.isin(hits_ev['channel'].to_numpy(), list(chs))
    if not on_band.any():
        return None
    seeder = seeder or wseed.seeds_from_hits
    r = seeder(hits_ev[~on_band], pos_maps, feu_x, feu_y, hot=hot, dead=dead)
    rec = r.get(int(eid)) or {'x': [], 'y': [], 'n_hits': 0, 'spark': False}
    rec = dict(rec)
    rec['band_reseeded'] = int(on_band.sum())
    return rec


def _stream_windows(cfg, pos_maps, seeds, wanted, pad_strips, verbose=True, dead=None, hits=None, hot=None,
                    evtime=None):
    """Yield lists of (eid, windows, seedinfo, n_hits, spark) per file pair.
    ``evtime``: filled with eventId -> (DREAM timestamp, acquisition tag) for every event read (timestamp -1 when the
    file has no timestamp branch).
    ``dead``: {plane: channels} padded across (``wft.seed.DEAD_SEED``); empty = historical.
    ``hits`` (the seeding hits, indexed by eventId): when given, every wanted event is checked for noise bands on its
    full waveforms and re-seeded without them (band_reseed); spark-vetoed events in ``wanted`` are fitted only if the
    re-seed clears the veto."""
    from . import io as wio
    fx = wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, cfg.MX17_FEU_X)
    fy = wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, cfg.MX17_FEU_Y)
    feu_of = {'x': cfg.MX17_FEU_X, 'y': cfg.MX17_FEU_Y}
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
        if evtime is not None:
            evtime.clear()                         # this file pair only: eventId can repeat in the next acquisition
            for rdr in (ry, rx):                  # x wins where both have the event (one DREAM clock)
                ts = rdr.timestamps if rdr.timestamps is not None else np.full(len(rdr.event_ids), -1)
                for e, t in zip(rdr.event_ids.tolist(), ts.tolist()):
                    if e in want:
                        evtime[int(e)] = (int(t), tag)
        raw = {}
        adc = {}                                   # undecoded samples, only when the windows carry rails (wft.io)
        carry = wio.rails_wanted()
        noise = {'x': rx.noise, 'y': ry.noise}
        for plane, rdr in (('x', rx), ('y', ry)):
            for item in rdr.iter_events(want, with_raw=carry):
                eid, ftst, wfm = item[:3]
                raw.setdefault(eid, {})[plane] = (ftst, wfm)
                if carry:
                    adc.setdefault(eid, {})[plane] = item[3]
        buf, n_reseed = {}, 0
        for eid, pl in raw.items():
            sd = seeds[eid]
            if hits is not None and wm.BAND_NSIG > 0:
                band = {feu_of[p]: set(np.flatnonzero(wm.band_rows(w, noise[p])).tolist())
                        for p, (_f, w) in pl.items()}
                try:
                    hev = hits.loc[[eid]] if eid in hits.index else None
                except KeyError:
                    hev = None
                r = band_reseed(eid, hev.reset_index() if hev is not None else None, band, pos_maps,
                                feu_of['x'], feu_of['y'], hot, dead)
                if r is not None:
                    sd, n_reseed = r, n_reseed + 1
                    seeds[eid] = sd
            if sd['spark'] or not (sd['x'] or sd['y']):
                continue
            for plane, (ftst, wfm) in pl.items():
                feu = feu_of[plane]
                cl = sd[plane]
                ra = adc.get(eid, {}).get(plane)
                if not cl:
                    if HEADON_RESCUE:
                        rec = buf.setdefault(eid, {'w': {}, 's': {}})
                        rec['w'][f'_rescue_{plane}'] = dict(W=wfm, noise=noise[plane], pos=pos_maps[feu])
                        if ra is not None:
                            rec['w'][f'_rescue_{plane}']['raw'] = ra
                        rec['ftst_' + plane] = ftst
                    continue
                cl = cl if isinstance(cl, list) else [cl]
                wins, used = [], []
                for s_ in cl:
                    win = wio.extract_window(wfm, noise[plane], pos_maps[feu], s_.channels, pad_strips,
                                             dead=(dead or {}).get(plane), raw_all=ra)
                    if win is None:
                        continue
                    wd = dict(W=win.W, pos=win.pos, noise=win.noise, ch=win.ch)
                    if win.rail_hi is not None:
                        wd.update(rail_hi=win.rail_hi, rail_lo=win.rail_lo)
                    wins.append(wd)
                    used.append(s_)
                if not wins:
                    continue
                rec = buf.setdefault(eid, {'w': {}, 's': {}})
                rec['w'][plane] = wins
                rec['s'][plane] = used
                rec['ftst_' + plane] = ftst
        del raw, adc
        payloads = []
        for eid, rec in buf.items():
            if not any(k in rec['w'] for k in ('x', 'y')):
                continue
            ftst = {p: rec.get('ftst_' + p) for p in ('x', 'y')}
            payloads.append((eid, rec['w'], rec['s'], seeds[eid]['n_hits'],
                             seeds[eid]['spark'], ftst))
        if verbose:
            print(f'[wft]   {tag}: {len(payloads):,} events windowed ({n_reseed} re-seeded without band strips)',
                  flush=True)
        yield payloads


# the one place a study may move a constant of this module (wft.diag); production runs none
from . import diag as _diag  # noqa: E402
_diag.apply('reco', globals())
