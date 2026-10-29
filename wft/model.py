"""
The forward model.

For one plane of one event, the model says: charge ``q_k >= 0`` arrives in each
60 ns slice ``k`` of drift at transverse position ``p0 + w * u_k``; each slice's
charge is shared onto the strips by the geometric strip integral, then onto
their neighbours by the resistive kernel (scaled by ``kY`` on Y), and finally
folded with the measured per-plane impulse response. Fitting ``(p0, w, t0)``
with the charge profile solved by NNLS at each step gives the track's position
at the mesh and its transverse speed ``w``; the angle is
``tan(theta) = w / v_drift``.

The resistive kernel has two forms (``share_mode`` on the bundle):

``delay``   the original parameterisation: a copy of the impulse response,
            amplitude ``c1``, delayed by ``tau_s`` to the +-1 strips (``c2``
            at ``2 tau_s`` to +-2), smeared by ``sigma_s``.
``lp``      the H4-beam-measured structure (M70V_FLAT_ANALYSIS.md §3,
            RAW_RUN71_REANALYSIS §4): the neighbour sees an RC-*dispersed*
            copy — the impulse response convolved with a one-pole low-pass of
            time constant ``tau_s``, cascaded once more for +-2. The copy
            peaks essentially WITH the central strip (shifted only by the RC
            rise, ~+30-60 ns for tau_s of a few hundred ns) and carries the
            long tail the delayed-copy form cannot represent without an
            unphysical ``sigma_p0``. ``c1``/``c2`` keep their meaning as the
            copies' amplitude (area) fractions.

Because the neighbours' delayed copies are *in the model*, they stop being
contamination — which is exactly what a per-strip hit time cannot do.

PRIOR ART (``REFERENCES.md`` in this package has the annotated list). This model
was re-derived from our own data, not lifted from a paper, but it is not new:
the resistive layer as a distributed RC network whose neighbour signal carries
position is Dixit et al., NIM A 518 (2004) 721; the resistive-*strip*
transmission line that makes the copy a cascaded one-pole — and that forces
``c2 < c1`` — is Galan et al., JINST 7 (2012) C04009; and fitting neighbouring
channels *simultaneously* against a spreading-times-electronics model is the
T2K ND280 ERAM analysis, Attie et al., NIM A 1056 (2023) 168534. What is ours
is solving the drift-depth charge profile inside that fit, so the result is a
micro-TPC rather than a sharpened centroid.

This is the packaged form of ``forward_model2.py`` (model v2) and
``forward_model3.py`` (the vectorised fitter) from the R&D directory, with the
module-level calibration replaced by an explicit
:class:`~wft.calib.CalibrationBundle`. The numerics are unchanged and are
regression-tested against the R&D code (``tests/test_model_regression.py``).

Calibration is module-global state, set once per process by
``use_calibration()``; worker processes set it in their initializer.

Two channel wildcards ride the same bundle (``prep_plane``): ``DEAD``
channels are censored samples (no signal in either direction, dropped from
the chi2 like a saturated one); ``HOT`` channels DO carry signal (coherent or
correlated noise fit as a cluster -- HANDOFF_D_NOISY_CHANNELS.md) and stay in
the fit, just down-weighted by inflating their noise (``HOT_NOISE_INFLATION``)
rather than dropped, so a real track that happens to cross one is not
penalised for the crossing.
"""
from __future__ import annotations


import numpy as np
from scipy.optimize import nnls, minimize
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf

from .calib import CalibrationBundle, check_kernel_ordering

# ---------------------------------------------------------------- module state
CAL: CalibrationBundle | None = None
TGRID: np.ndarray | None = None
TMPL: dict | None = None
GAIN: dict | None = None
DT_XY: dict = {}
PITCH = 0.78
SNS = 60.0
SAT = 3550.0
DT = 60.0                 # width of one charge/depth bin [ns] = sample_ns / SUBDIV
SUBDIV = 1                # depth bins per sample period (bundle ``depth_subdiv``; 3 = the 20 ns grid)
NSAMP = 32
K = 18
TS = np.arange(NSAMP) * SNS
UK = (np.arange(K) + 0.5) * DT
HYPER: dict | None = None
SHARE_MODE = 'delay'      # 'delay' | 'lp' — see module docstring
DEAD: dict = {}           # per-plane dead-channel arrays (T1.3); see prep_plane
HOT: dict = {}            # per-plane hot-channel arrays; see prep_plane
#: How far a hot channel's noise is inflated in the chi2 (HANDOFF_D_NOISY_
#: CHANNELS.md item 3: "cap their influence... a per-channel weight in the
#: NNLS design matrix"). Deliberately finite, unlike DEAD's 1e9: a hot
#: channel DOES carry signal and stays in the fit (rows are not censored --
#: `sat` is untouched), just down-weighted by 1/HOT_NOISE_INFLATION**2 in the
#: chi2 sum, so it cannot dominate but a real track crossing it is not
#: penalised for the crossing.
#:
#: **Scanned 2026-09-08 on D/run_145 and it does not matter.** 10, 30 and 100
#: were fitted over one fixed set of 3 600 real triggers and agree to three
#: decimals on every metric, in every stratum -- convergence, chi2/dof, angle,
#: p0. HANDOFF_HOT_WILDCARD_TUNING.md Sec. 4.1 proposed that 10 was "probably
#: too weak" and that this was why the first hot-masked re-run regressed; it
#: was not. Paired per-event, a window whose SEED is unchanged by the mask
#: fits identically with it (chi2/dof 39.45 -> 38.54, median |dp0| 0.00 mm):
#: everything the wildcard does, it does through seeding, not weighting.
#: 10 is kept because nothing argues for anything else.
#: The scan is re-run with ``WFT_DIAG=model.HOT_NOISE_INFLATION=<x>`` (wft.diag).
HOT_NOISE_INFLATION = 10.0
#: Fractional model-error term in the chi2 weights, sigma^2 = noise^2 + (MODEL_FRAC * W)^2.  0: the chi2 is weighted
#: by the noise alone.  The archived RC-ladder model ran production at 0.03 (RECO_BENCH_2026-07-29 §3); on the phys
#: kernel it was REJECTED by the model audit (2026-10-27 F17/F19: chi2/dof falls to a quarter but the far fraction
#: rises +1..+12 pp and s68 worsens 0.1-0.25 deg on det6/det7 -- the bright-sample weighting is what pins the track).
#: Study value only, through wft.diag.
MODEL_FRAC = 0.0
#: Censored samples in the charge solve itself (cloud_basics FINDINGS §41).  Without it saturated samples are left
#: out of the NNLS and only penalised afterwards, so the template fitted to the unclipped samples fills the peak --
#: angle and position come out as well as unclipped, but the charge of clipped strips reads 6-12 % low.  The solve is
#: repeated with every clipped sample the model falls below added as an observation AT the clip level (the active set
#: of a one-sided bound, as in a Tobit fit) until none is below.  Part of the algorithm since 2026-10-27 (model audit
#: F19 `sat` arm: s68 +-0.015 deg, far -0.1..-0.4 pp, chi2 -2..-10 %, and the charge of clipped strips is right, which
#: the x-y charge pairing and the head-on rescue's charge match rely on).
SAT_SOLVE = True
#: Unobservable depth columns (opt-in, 2026-10-23; sept26_prelim_analysis/qsum_runaway.py, OCTOBER O10).  A depth bin
#: whose response lies past the end of the window (time-censored) or off the strip window (space-censored) has a
#: column of ~1e-6..1e-17 in the weighted design matrix, and the unregularised NNLS can put unbounded charge on it
#: (q_sum > 1e6 ADC on 30 % of beam tracks).  With COL_GUARD > 0 every solve offers only the columns whose weighted
#: norm is at least COL_GUARD x the largest; the others get q = 0.  0.01 has been the pass setting since W2 (it was
#: exported by rerun/stage_job.sh); part of the algorithm since 2026-10-27.
COL_GUARD = 0.01
#: t0 continuity of the RC kernel (opt-in, model audit 2026-10-27).  ``build_matrix_rc`` snaps t0 to the 5 ns tensor
#: grid, so chi2 is a staircase in t0 and Nelder-Mead cannot refine t0 (C19); the delay kernel interpolates.  With
#: RC_T0_INTERP the design matrix is interpolated linearly between the two neighbouring grid tensors.  Part of the
#: algorithm since 2026-10-27 as a correctness change (F19 `t0i`: angles unchanged within +-0.02 deg, chi2/dof -1..-4 %,
#: t0 continuous so t0_err is defined on every fit; ws_t0: +1.5-5 % wall time).
RC_T0_INTERP = True
#: Zero-suppressed samples are fitted as MEASURED ZEROS (the FEU saw less than its threshold there): on RAW run_71 with
#: ZS emulated on the same events this is the closest to the RAW answer (t0 <= 27 ns, position 15-27 um, slope
#: 0.03-0.07 um/ns), censoring them is the worst (44-66 ns, 45-86 um, 0.14-0.31 um/ns), the ZS_BOUND form in between
#: (ws_sps REPORT, 2026-10-27).  Samples that were never read out at all (RAW front-end packet loss) are a different
#: thing: the reader marks them ``P['missing']`` and prep_plane censors them.  ZS_CENSOR (censor every exact 0) is a
#: study override only (wft.diag).
ZS_CENSOR = False
#: Zero-suppressed data, the informative form (ws_sps 2026-10-27): a suppressed sample is known to lie BELOW the FEU
#: threshold.  With ZS_BOUND = k > 0 the exactly-zero samples are censored as above AND penalised one-sidedly when the
#: model rises above k x the strip noise (the mirror of the saturation penalty).  k = the run's ZS threshold in sigma
#: (SPS registry zs_sigma).  A PROTOTYPE, not gated (ws_unify): 0, study override only (WFT_ZS_BOUND via wft.diag).
ZS_BOUND = 0.0

#: Noise bands (model audit follow-up FE, item 13, 2026-10-27).  A band is a whole-window baseline offset of a strip
#: (+-50..3000 ADC, both signs, rerun/satstudy/bands.py + rerun/bands/): untriggered, it carries no charge of the
#: triggering particle, and fitted as data it pulls the track or makes a competing cluster.  A strip whose FIRST and LAST
#: BAND_EDGE samples all sit beyond BAND_NSIG x its noise on the same side is a band strip in that event: the waveform
#: equivalent of the hit flags trunc_left AND trunc_right (reco.DROP_FLAT_HITS, which removes such hits from the seeding),
#: but read from the window itself, so it acts in every driver (bench, SPS, n_TOF) and catches negative bands, which never
#: make a hit.  The particle cannot be there in the first samples (its charge arrives at t0 >= ~230 ns on the bench), so a
#: track strip is never flagged by its own pulse.  Band strips are censored exactly like dead strips (unknown charge, not
#: zero; noise inflated so no penalty pulls either way); the event is reconstructed, not dropped.  Counted per fit as
#: ``n_band_strips``.  BAND_NSIG = 0 turns it off (study override, wft.diag).
BAND_NSIG = 5.0
BAND_EDGE = 2
#: The rule needs a window LONGER than a particle's signal (the bench: 32 x 60 ns = 1920 ns against <= 24 samples per
#: particle, satstudy; SPS: 64 samples), so that a strip high at both ends cannot be one particle.  The n_TOF window is
#: 20 x 60 ns = 1200 ns with a trigger asynchronous to the particle: a track can be above threshold from the first
#: sample to the last (1.2 % of run_145 hits are, 18 % start truncated), and bands are absent there (rerun/bands), so on
#: a window shorter than this nothing is flagged.  Measured, not tuned: the bench and SPS windows clear it, n_TOF does not.
BAND_MIN_WINDOW_NS = 1800.0

#: Outer solver of :func:`fit_plane_raw` (algo_review 2026-10-28, HANDOFF "Fitter: LM variable-projection").  The fit
#: is a separable least-squares problem: the K depth charges are linear and non-negative, eliminated by the NNLS at
#: every (p0, w, t0) -- variable projection (Golub & Pereyra 1973) -- so the outer problem is three-dimensional.
#: 'nm' = the 105-point (p0, w, t0) grid + Nelder-Mead and 1-D curvature errors (the algorithm through 2026-10-28);
#: 'lm' = a 7-point t0 ladder + Levenberg-Marquardt on the projected residual with the Kaufman (1975) Jacobian,
#: analytic on the RC kernel, a short Nelder-Mead polish, and the full 3x3 covariance for the errors.  'lm' is the
#: algorithm since the 2026-10-28 fleet gate (lm_r1: fe_nm vs fe_lm on the five golden rows, pinned-t0 FE bundles;
#: rerun/model_audit/algo_review/REPORT_LM.md); 'nm' stays as a study override (``WFT_DIAG=model.FITTER=nm``).
#: With free t0 (n_TOF) any outer solver picks the 60 ns lattice basin by its search path, so t0/p0 labels differ from
#: 'nm' on 13-33 % of planes at the same angle; with the bench pin they do not.
FITTER = 'lm'
#: 'lm' only: walk the depth-bin lattice.  The chi2 surface has a basin at every (p0 + DT w, w, t0 + DT): the lattice
#: shift is not a symmetry of the kernel, and which basin a local search ends in is set by its start -- the grid +
#: Nelder-Mead misses a deeper one on ~10 % of free-t0 planes (algo_review §2), and the bench t0 prior (sigma 5 ns,
#: 144 chi2 units per bin against data chi2 ~ 1e4-1e5) does not pin it either (lm-varpro, 2026-10-28).  After the
#: first LM the fit is re-started one bin over in each direction and keeps hopping while chi2 falls, up to LM_HOPS
#: bins each way.  0 = no hops (the local minimum of the best ladder start).  Default 0: with the bench t0 pin hops
#: find nothing and cost 20 %, and with free t0 the deeper basin is not closer to the trigger or M3 (REPORT_LM.md).
LM_HOPS = 0
#: 'lm' only: the global start.  'ladder' = the 7-point t0 ladder (at the start's p0 and along the pivot line, 13
#: solves); 'grid' = the 'nm' fitter's own 105-point (p0, w, t0) grid, so the basin is chosen exactly as 'nm' chooses
#: it and LM replaces only the Nelder-Mead refinement (the minimal-change form; use with LM_HOPS = 0).
LM_START = 'ladder'
#: 'lm' only: t0 stays within +-LM_T0_RANGE ns of the start (the ladder's +-120 ns plus one bin).  With free t0 a long
#: profile absorbs a time offset (tests/test_two_track.py), and an unbounded LM follows that valley hundreds of ns to a
#: marginally lower chi2 that the grid + Nelder-Mead never reached (a synthetic pair: t0 -267 ns from a start at -17,
#: and the column-time plausibility of its two-track children then fails).  Inactive under the bench t0 pin.
LM_T0_RANGE = 180.0
LM_MAXIT = 30


def band_rows(W, noise, miss=None) -> np.ndarray:
    """Boolean (n_strip,): rows of a (n_strip, n_sample) window that are noise bands in this event (see BAND_NSIG).
    ``miss``: samples never read out (not used as evidence either way)."""
    W = np.asarray(W, dtype=np.float64)
    if (BAND_NSIG <= 0 or W.ndim != 2 or W.shape[1] < 2 * BAND_EDGE + 1
            or W.shape[1] * SNS < BAND_MIN_WINDOW_NS):
        return np.zeros(W.shape[0] if W.ndim == 2 else 0, dtype=bool)
    thr = BAND_NSIG * np.maximum(np.asarray(noise, dtype=np.float64), 3.0)[:, None]
    e = np.concatenate([W[:, :BAND_EDGE], W[:, -BAND_EDGE:]], axis=1)
    if miss is not None:
        m = np.asarray(miss, dtype=bool)
        m = np.concatenate([m[:, :BAND_EDGE], m[:, -BAND_EDGE:]], axis=1)
        up = np.where(m, True, e > thr).all(axis=1) & ~m.all(axis=1)
        dn = np.where(m, True, e < -thr).all(axis=1) & ~m.all(axis=1)
    else:
        up, dn = (e > thr).all(axis=1), (e < -thr).all(axis=1)
    return up | dn


_smear_cache: dict = {}
_lp_cache: dict = {}
_tt_cache: dict = {}
T0_STEP = 5.0             # t0 quantisation for the cached time tensors


def use_calibration(cal: CalibrationBundle) -> None:
    """Install a calibration bundle as the model's calibration."""
    global CAL, TGRID, TMPL, GAIN, DT_XY, PITCH, SNS, SAT, DT, K, UK, HYPER, \
        SHARE_MODE, DEAD, HOT, SUBDIV
    check_kernel_ordering(cal.hyper, where=f'bundle {cal.detector}/{cal.run_key}')
    CAL = cal
    DEAD = {p: np.asarray(sorted(ch), dtype=int)
            for p, ch in getattr(cal, 'dead', {}).items() if len(ch)}
    HOT = {p: np.asarray(sorted(ch), dtype=int)
          for p, ch in getattr(cal, 'hot', {}).items() if len(ch)}
    TGRID = np.asarray(cal.grid, float)
    TMPL = {p: np.asarray(cal.tmpl[p], float) for p in ('x', 'y')}
    GAIN = {p: np.asarray(cal.gain[p], float) for p in ('x', 'y')}
    DT_XY = dict(cal.dt_xy)
    PITCH = float(cal.pitch_mm)
    SNS = float(cal.sample_ns)
    SAT = float(cal.sat_adc)
    SUBDIV = max(int(getattr(cal, 'depth_subdiv', 1) or 1), 1)
    DT = float(cal.sample_ns) / SUBDIV
    K = int(cal.n_depth_bins) * SUBDIV
    UK = (np.arange(K) + 0.5) * DT
    HYPER = dict(cal.hyper)
    HYPER.setdefault('kY', 1.0)
    SHARE_MODE = getattr(cal, 'share_mode', 'delay') or 'delay'
    _smear_cache.clear()
    _lp_cache.clear()
    _tt_cache.clear()
    _rc_cache.clear()
    set_nsamp(NSAMP)


def set_share_mode(mode: str) -> None:
    """Override the sharing-kernel form ('delay' | 'lp')."""
    global SHARE_MODE
    if mode not in ('delay', 'lp'):
        raise ValueError(f'share_mode must be delay|lp, got {mode!r}')
    SHARE_MODE = mode
    _lp_cache.clear()
    _tt_cache.clear()


def set_nsamp(ns: int) -> None:
    """Adapt to a different DAQ window length (det4 mixes 32 and 37 samples)."""
    global NSAMP, TS
    NSAMP = int(ns)
    TS = np.arange(NSAMP) * SNS
    _smear_cache.clear()
    _tt_cache.clear()
    _rc_cache.clear()


def set_depth_bins(k: int) -> None:
    """Extend/shrink the charge basis (K=26 is used at low drift field, where
    the column takes longer than the nominal window to arrive)."""
    global K, UK
    K = int(k) * SUBDIV          # k counted in sample periods (the bundle's n_depth_bins convention)
    UK = (np.arange(K) + 0.5) * DT
    _tt_cache.clear()
    _rc_cache.clear()


def _require_cal():
    if CAL is None:
        raise RuntimeError('wft.model has no calibration: call use_calibration()')


# ------------------------------------------------------------------- pieces
def _templates(plane: str, sigma_s: float):
    """(impulse response, dispersion-smeared impulse response) for a plane."""
    key = (plane, round(float(sigma_s), 1))
    if key not in _smear_cache:
        base = TMPL[plane]
        _smear_cache[key] = (base, gaussian_filter1d(base, max(sigma_s, 1.0) / 10.0))
    return _smear_cache[key]


def _lp_copies(plane: str, sigma_s: float, tau_s: float):
    """(extended grid, once-, twice-RC-convolved smeared template) for the
    ``lp`` share mode. The template grid stops at +1.4 us, but an RC tail with
    tau of a few hundred ns is still alive there, so the template is
    zero-padded to +6 us before convolving. Discrete one-pole with the grid
    step preserves the area, so c1/c2 stay area fractions."""
    key = (plane, round(float(sigma_s), 1), round(float(tau_s), 1))
    hit = _lp_cache.get(key)
    if hit is not None:
        return hit
    _, sm = _templates(plane, sigma_s)
    step = float(TGRID[1] - TGRID[0])
    n_pad = max(0, int(round((6000.0 - TGRID[-1]) / step)))
    ge = np.concatenate([TGRID, TGRID[-1] + step * (1 + np.arange(n_pad))])
    x = np.concatenate([sm, np.zeros(n_pad)])
    a = np.exp(-step / max(float(tau_s), 1.0))
    l1 = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = acc * a + x[i] * (1.0 - a)
        l1[i] = acc
    l2 = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = acc * a + l1[i] * (1.0 - a)
        l2[i] = acc
    if len(_lp_cache) > 256:
        _lp_cache.clear()
    _lp_cache[key] = (ge, l1, l2)
    return ge, l1, l2


def _copy_responses(plane: str, base: np.ndarray, hyper: dict):
    """(H1, H2) neighbour-copy responses on the (NSAMP, K) time offsets in
    ``base``, per the active SHARE_MODE.

    ``tau_y_fac`` scales the RC constant on the Y plane only: the resistive
    strips run along y, so Y's copy is slower as well as stronger (measured
    directly, tau_X 230 / tau_Y 410 ns). NOTE the key is deliberately NOT the
    bundles' ``kTauY``: that constant belongs to the archived RC-ladder
    representation, and switching it on under this kernel form regressed Y
    badly (sigma_Y 1.14 -> 1.57 deg, bench 2026-08-12) — the F19 lesson, RC
    constants are representation-dependent. A per-plane tau must enter here
    through a recalibration that fits/validates ``tau_y_fac`` under THIS
    kernel; no existing bundle carries the key, so nothing changes silently."""
    tau = hyper['tau_s'] * (hyper.get('tau_y_fac', 1.0) if plane == 'y' else 1.0)
    if SHARE_MODE == 'lp':
        ge, l1, l2 = _lp_copies(plane, hyper['sigma_s'], tau)
        H1 = np.interp(base, ge, l1, left=0, right=0)
        H2 = np.interp(base, ge, l2, left=0, right=0)
    else:
        _, sm = _templates(plane, hyper['sigma_s'])
        H1 = np.interp(base - tau, TGRID, sm, left=0, right=0)
        H2 = np.interp(base - 2 * tau, TGRID, sm, left=0, right=0)
    return H1, H2


def _time_tensors(plane: str, t0q: float, hyper: dict):
    """(K, NSAMP) impulse responses of each depth bin, cached on a 5 ns t0 grid."""
    key = (plane, t0q, hyper['tau_s'], round(hyper['sigma_s'], 1), NSAMP, K,
           SHARE_MODE, hyper.get('tau_y_fac', 1.0) if plane == 'y' else 1.0)
    hit = _tt_cache.get(key)
    if hit is not None:
        return hit
    tmpl, _sm = _templates(plane, hyper['sigma_s'])
    base = TS[:, None] - (t0q + UK[None, :])          # (NSAMP, K)
    H0 = np.interp(base, TGRID, tmpl, left=0, right=0)
    H1, H2 = _copy_responses(plane, base, hyper)
    if len(_tt_cache) > 4096:
        _tt_cache.clear()
    _tt_cache[key] = (H0, H1, H2)
    return H0, H1, H2


def strip_fractions(pos, p0, w, sigma_p0, Dp):
    """Fraction of each depth bin's charge landing on each strip: the bin's
    transverse extent (p0 + w*u over the bin) smeared by the initial cloud size
    and diffusion, integrated over the strip pitch."""
    ua = np.arange(K) * DT
    ub = ua + DT
    pa, pb = p0 + w * ua, p0 + w * ub
    pc = 0.5 * (pa + pb)
    half = 0.5 * np.abs(pb - pa)
    sig = np.sqrt(sigma_p0 ** 2 + Dp ** 2 * UK + half ** 2 / 3.0)
    z = 1.0 / (np.sqrt(2) * sig)[None, :]
    hi = (pos[:, None] + PITCH / 2 - pc[None, :]) * z
    lo = (pos[:, None] - PITCH / 2 - pc[None, :]) * z
    return 0.5 * (erf(hi) - erf(lo))                  # (n_strip, K)


#: Time-since-landing edges [ns] for the RC-diffusion kernel: each interval's
#: increment of the strip fractions is applied as one impulse at its midpoint.
#: Fine early (where the spread changes fastest), coarse late.
RC_EDGES = np.array([0.0, 30, 60, 100, 150, 220, 300, 400, 550, 750, 1000,
                     1400, 2000, 3000])
_rc_cache: dict = {}
#: prompt width [mm] of the narrow footprint component (``core_frac_<plane>``): the static induction floor (FC stack-up)
CORE_SIGMA = 0.07


def _rc_tensors(plane: str, t0q: float, tmpl_plane: str):
    """(n_s, NSAMP, K) impulse responses delayed by each RC interval midpoint
    (index 0 = no delay, the prompt landing)."""
    key = (plane, tmpl_plane, t0q, NSAMP, K, float(UK[0]))
    hit = _rc_cache.get(key)
    if hit is not None:
        return hit
    tmpl = TMPL[tmpl_plane]
    shifts = np.r_[0.0, 0.5 * (RC_EDGES[:-1] + RC_EDGES[1:])]
    base = TS[:, None] - (t0q + UK[None, :])
    Hs = np.stack([np.interp(base - s, TGRID, tmpl, left=0, right=0) for s in shifts])
    if len(_rc_cache) > 2048:
        _rc_cache.clear()
    _rc_cache[key] = Hs
    return Hs


def t0q_of(t0: float) -> float:
    return round(t0 / T0_STEP) * T0_STEP


def _rc_shift_tensor(plane: str, t0q: float, tmpl_plane: str, shift_ns: float):
    """(NSAMP, K) impulse response delayed by one arbitrary shift (the asymmetric copy), cached like _rc_tensors."""
    key = ('shift', plane, tmpl_plane, t0q, NSAMP, K, float(UK[0]), round(float(shift_ns), 1))
    hit = _rc_cache.get(key)
    if hit is not None:
        return hit
    base = TS[:, None] - (t0q + UK[None, :])
    H = np.interp(base - shift_ns, TGRID, TMPL[tmpl_plane], left=0, right=0)
    if len(_rc_cache) > 2048:
        _rc_cache.clear()
    _rc_cache[key] = H
    return H


def _fractions_sig(pos, pc, sig):
    """strip_fractions with an explicit width per depth bin (sig: (K,))."""
    z = 1.0 / (np.sqrt(2) * sig)[None, :]
    hi = (pos[:, None] + PITCH / 2 - pc[None, :]) * z
    lo = (pos[:, None] - PITCH / 2 - pc[None, :]) * z
    return 0.5 * (erf(hi) - erf(lo))


def build_matrix_rc(plane, pos, p0, w, t0, hyper):
    """Physical kernel (cloud_basics/FINDINGS.md, 2026-10-09).

    Charge drifting for u lands with a Gaussian footprint of width
    sigma_p0 (prompt: avalanche + induction, a readout constant) and
    Dp^2 u (drift diffusion; Dp^2 = D_T^2 v from Magboltz).  On the view ALONG
    the resistive strips (Y) it then spreads as a 1-D RC line,
    sigma^2 += 2 rc_D_<plane> s (s = time since landing); across them (X)
    rc_D_x = 0.  The readout sees each increment of the charge above a strip
    through the electronics.  No discrete copies: c1, c2, kY, tau_s and
    sigma_s are not used.

    Keys: ``rc_D_y`` (mm^2/ns, enables this kernel), ``rc_D_x`` (default 0),
    ``sigma_p0_<plane>`` / ``sigma_p0``, ``Dp``, ``rc_tmpl`` ('x' or 1 = drive
    both views with the X template, the clean electronics response -- Y's
    measured template already carries RC undershoot; 'own' or 0 = each view
    its own).  ``rc_drain`` (ns, optional) lets the strip's charge decay.
    ``lor_frac_<plane>`` / ``lor_gamma_<plane>`` (optional, mm): a fraction of each prompt landing spread
    as a Lorentzian of half-width gamma instead of the Gaussian (pseudo-Voigt footprint tails, measured
    on X, cloud_basics FINDINGS §23); absent or 0 = bit-identical.  Resolution-neutral on the bench.
    ``core_frac_<plane>`` (optional): a fraction of each prompt landing with the prompt width ``core_sigma``
    (default 0.07 mm, the static induction floor) instead of sigma_p0, drift diffusion and RC growth kept -- a
    two-Gaussian footprint, narrower core with the same tails (template study T15, 2026-10-29).  Each component
    grows under RC exactly, so it survives on Y.  Absent or 0 = bit-identical.
    """
    tp = 'x' if hyper.get('rc_tmpl', 'x') in ('x', 1, 1.0) else plane
    if RC_T0_INTERP:
        # linear interpolation between the two neighbouring 5 ns tensors: chi2 becomes continuous in t0, so the
        # Nelder-Mead refinement can move along t0 (and the p0-t0 degeneracy) instead of stalling on a staircase.
        # The impulse response is smooth over 5 ns, so this is the convolution evaluated at t0 to first order.
        t0a = np.floor(t0 / T0_STEP) * T0_STEP
        f = (t0 - t0a) / T0_STEP
        Ha = _rc_tensors(plane, t0a, tp)
        Hs = Ha if f < 1e-9 else (1.0 - f) * Ha + f * _rc_tensors(plane, t0a + T0_STEP, tp)
    else:
        t0q = round(t0 / T0_STEP) * T0_STEP
        Hs = _rc_tensors(plane, t0q, tp)                          # (ns, NSAMP, K)
    ua = np.arange(K) * DT
    pa, pb = p0 + w * ua, p0 + w * (ua + DT)
    pc = 0.5 * (pa + pb)
    half = 0.5 * np.abs(pb - pa)
    s0 = float(hyper.get(f'sigma_p0_{plane}', hyper['sigma_p0']))
    var0 = s0 ** 2 + hyper['Dp'] ** 2 * UK + half ** 2 / 3.0       # (K,)
    D = float(hyper.get(f'rc_D_{plane}', 0.0))
    cf = float(hyper.get(f'core_frac_{plane}', 0.0))
    dvc = float(hyper.get('core_sigma', CORE_SIGMA)) ** 2 - s0 ** 2

    def frac(v):
        G = _fractions_sig(pos, pc, np.sqrt(v))
        return G if cf <= 0 else (1.0 - cf) * G + cf * _fractions_sig(pos, pc, np.sqrt(v + dvc))
    F0 = frac(var0)                                               # prompt landing
    eta = float(hyper.get(f'lor_frac_{plane}', 0.0))
    if eta > 0:
        # induced-footprint tails (cloud_basics FINDINGS §23): a fraction eta of each landing
        # spread as a Lorentzian of half-width gamma, instead of the Gaussian -- pseudo-Voigt
        gam = float(hyper[f'lor_gamma_{plane}'])
        a_hi = np.arctan((pos[:, None] + PITCH / 2 - pc[None, :]) / gam)
        a_lo = np.arctan((pos[:, None] - PITCH / 2 - pc[None, :]) / gam)
        F0 = (1.0 - eta) * F0 + eta * (a_hi - a_lo) / np.pi
    n = len(pos)
    M = np.einsum('ik,tk->itk', F0, Hs[0])
    asym = float(hyper.get(f'rc_asym_{plane}', 0.0))
    if asym != 0.0:
        # one-sided delayed neighbour copy (cloud_basics FINDINGS 42: the -1 minus +1 stacked pulse on X is ~0.13 x the
        # centre, delayed ~110 ns, a board property): a fraction |asym| of each prompt landing moves to the neighbour
        # at LOWER position (asym > 0) or higher (asym < 0) after rc_asym_tau_<plane> ns; the centre loses it then.
        tau_a = float(hyper.get(f'rc_asym_tau_{plane}', 110.0))
        Ha = _rc_shift_tensor(plane, t0q_of(t0), tp, tau_a)
        Fa = np.zeros_like(F0)
        if asym > 0:
            Fa[:-1] = F0[1:]          # strip i receives what landed on the strip above it in position
        else:
            Fa[1:] = F0[:-1]
        M += abs(asym) * np.einsum('ik,tk->itk', Fa - F0, Ha)
    if D > 0:
        drain = float(hyper.get('rc_drain', 0.0))
        Fprev = F0
        for j, s in enumerate(RC_EDGES[1:]):
            Fj = frac(var0 + 2.0 * D * s)
            if drain > 0:
                Fj = Fj * np.exp(-s / drain)
            M += np.einsum('ik,tk->itk', Fj - Fprev, Hs[j + 1])
            Fprev = Fj
    return M.reshape(n * NSAMP, K)


def build_matrix(plane, pos, p0, w, t0, hyper):
    """Design matrix: column k = the (strip, sample) waveform produced by unit
    charge in depth bin k, sharing and impulse response included."""
    if 'rc_D_y' in hyper:
        return build_matrix_rc(plane, pos, p0, w, t0, hyper)
    t0q = round(t0 / T0_STEP) * T0_STEP
    if abs(t0 - t0q) > 1e-9:
        tmpl, _sm = _templates(plane, hyper['sigma_s'])
        base = TS[:, None] - (t0 + UK[None, :])
        H0 = np.interp(base, TGRID, tmpl, left=0, right=0)
        H1, H2 = _copy_responses(plane, base, hyper)
    else:
        H0, H1, H2 = _time_tensors(plane, t0q, hyper)
    # per-plane amplitude on the discrete (RC) sharing kernel. kY is the
    # long-standing Y multiplier; cX (default 1, i.e. no change) scales the X
    # side — the resistive strips run along y, so X cannot have resistive
    # sharing and its +-1 copy should be diffusion (F6): cX = 0 with Dp
    # refit is the physically motivated test arm (handoff T1.2).
    kY = hyper.get('kY', 1.0) if plane == 'y' else hyper.get('cX', 1.0)
    c1, c2 = hyper['c1'] * kY, hyper['c2'] * kY
    # Per-view ratio (2026-10-09): ``c2_over_c1_<plane>`` overrides the global
    # key on that plane only.  The head-on +-2/+-1 pattern differs by view in
    # sign, not just size -- Y wants MORE delayed +-2 than r06 draws, X less
    # (sps_beam_test_26/analysis/plane_ratio).  Absent keys = old behaviour.
    r = hyper.get(f'c2_over_c1_{plane}', hyper.get('c2_over_c1'))
    if r is not None:
        # SLAVE c2 TO c1.  The +-2 strip is reached only through the +-1
        # strip, so c2 < c1 always -- yet the shipped bundles carry c2 > c1 on
        # every detector (det3 1.14, det2 1.53, det7 1.75, det4 2.12).  That is
        # not a bound artefact: the ref-pinned cosmic chi2 is genuinely flat in
        # this direction (sloppy-mode analysis 2026-08-17), so the fit is free
        # to walk there and does.  The H4 head-on beam data measures the ratio
        # directly and model-free, at 0.45 +- 0.03 over a 2.6x range of drift
        # field (sps_beam_test_26/analysis/sharing_kernel); near-vertical bench
        # cosmics give 0.63 +- 0.10 on det3.  Pinning it costs one hyper and
        # makes the ordering structural.
        # Applied to the BASE hypers, before the per-plane kY/cX scaling, so
        # the ratio is plane-independent.  No existing bundle carries the key.
        c2 = float(r) * c1
    # ``sigma_p0_<plane>`` (optional) gives each view its own prompt lateral
    # spread; absent = the shared sigma_p0.
    F = strip_fractions(pos, p0, w, hyper.get(f'sigma_p0_{plane}',
                                              hyper['sigma_p0']), hyper['Dp'])
    n = len(pos)
    M = np.empty((n, NSAMP, K))
    np.multiply(F[:, None, :], H0[None, :, :], out=M)
    # ``c1_asym_<plane>`` (optional, default 0) makes the +-1 copy one-sided:
    # strip j takes (1 + a) c1 from strip j+1 and (1 - a) c1 from strip j-1.
    # The H4 X view is one-sided (+1 0.27 vs -1 0.43 of the centre) in both
    # mounts and at every field, so it is the board, not a tilt.
    a = float(hyper.get(f'c1_asym_{plane}', 0.0))
    Fs = np.zeros_like(F)
    Fs[1:] = (1.0 - a) * F[:-1]
    Fs[:-1] += (1.0 + a) * F[1:]
    M += (c1 * Fs)[:, None, :] * H1[None, :, :]
    if c2 > 0:
        Fs2 = np.zeros_like(F)
        Fs2[2:] = F[:-2]
        Fs2[:-2] += F[2:]
        M += (c2 * Fs2)[:, None, :] * H2[None, :, :]
    return M.reshape(n * NSAMP, K)


def prep_plane(P, plane):
    """Gain-correct one plane's waveform window. P: dict with W (nstrip, nsamp),
    pos [mm], noise per strip, ch (channel numbers).
    Returns (W, noise, pos, censor mask).

    Dead channels (bundle ``dead``, T1.3) are censored samples — a broken
    connection reads baseline, not zero charge, so their rows are excluded
    from the fit and the dof exactly like saturated samples, and their noise
    is inflated so the one-sided saturation penalty cannot pull on them
    either: no information in either direction.

    Hot channels (bundle ``hot``, HANDOFF_D_NOISY_CHANNELS.md) are NOT
    censored -- they carry real signal, just noise/correlated-noise-shaped
    signal, and a real track can cross one. Their rows stay live in the fit
    (``sat`` untouched) with noise inflated by a bounded
    ``HOT_NOISE_INFLATION`` rather than to 1e9, so they are down-weighted --
    capped influence, not zero -- and cannot pull the fit on their own."""
    W = np.asarray(P['W'], dtype=np.float64).copy()
    ch = np.asarray(P['ch'], dtype=int)
    g = GAIN[plane][ch]
    W /= g[:, None]
    noise = np.maximum(np.asarray(P['noise'], dtype=np.float64), 3.0) / g
    sat = np.asarray(P['W'], dtype=np.float64) >= SAT
    if ZS_CENSOR or ZS_BOUND > 0:
        sat |= np.asarray(P['W'], dtype=np.float64) == 0.0
    miss = P.get('missing') if isinstance(P, dict) else None
    if miss is not None:
        # samples never read out (RAW packet loss): censored, and moved off exact 0 so ZS_BOUND does not bound them
        miss = np.asarray(miss, dtype=bool)
        sat |= miss
        W[miss] = -1e-12
    d = DEAD.get(plane)
    if d is not None and len(d):
        rows = np.isin(ch, d)
        if rows.any():
            sat[rows] = True
            noise[rows] = 1e9
    b = band_rows(P['W'], P['noise'], miss)
    if b.any():
        # noise bands: censored like dead strips (BAND_NSIG); unknown charge, the event is kept
        sat[b] = True
        noise[b] = 1e9
    h = HOT.get(plane)
    if h is not None and len(h):
        rows = np.isin(ch, h)
        if rows.any():
            noise[rows] *= HOT_NOISE_INFLATION
    return W, noise, np.asarray(P['pos'], dtype=np.float64), sat


def chi2_plane(plane, W, noise, pos, sat, p0, w, t0, hyper, censor=True,
               snap_t0=True, t0_prior=None):
    """chi2 of the model at (p0, w, t0), with the charge profile profiled out by
    NNLS. Saturated samples are censored: excluded from the fit, and penalised
    only if the model falls *below* the clipped value.

    ``t0_prior=(t0_pred, sigma)`` adds a Gaussian penalty pinning t0 to an
    external per-event prediction (the scintillator trigger through the ftst
    phase). The chi2 surface has near-degenerate minima 60 ns (one depth bin)
    apart — the profile shifts a bin and p0 slides by w*60 — and only ~35 % of
    free fits land in the physical one (T1.1 gate, 2026-08-11); the prior is
    what selects it."""
    if snap_t0:
        t0 = round(t0 / T0_STEP) * T0_STEP
    M = build_matrix(plane, pos, p0, w, t0, hyper)
    chi, q = _solve_nnls(M, W, noise, sat, censor)
    if q is None:
        return np.inf, None
    if t0_prior is not None:
        chi += ((t0 - t0_prior[0]) / t0_prior[1]) ** 2
    return chi, q


def _solve_nnls(M, W, noise, sat, censor=True):
    """(chi2, q) for a design matrix of any width: the charge profile solved by
    NNLS with the per-strip noise weighting, saturated samples censored.

    Factored out of :func:`chi2_plane` so the two-track model (:func:`chi2_plane_two`)
    uses exactly the same weighting, censoring and dead/hot handling rather than a
    second copy of it. Numerics are unchanged — ``tests/test_model_regression.py``
    pins them."""
    ok = ~sat.reshape(-1)
    if not ok.any():
        return np.inf, None
    if MODEL_FRAC <= 0:
        # production path, expression-for-expression unchanged
        Wt = np.repeat(1.0 / noise, NSAMP)
        A = (M * Wt[:, None])[ok]
        y = (W / noise[:, None]).reshape(-1)[ok]
    else:
        # fractional model-error term in quadrature (ported 2026-10-09 from
        # the archived RC-ladder model): percent-level template mismatch on
        # bright samples otherwise dominates the chi2 (bench chi2/dof ~800),
        # and a kernel fitted on it patches the template instead of
        # describing the sharing.
        Wt = (1.0 / np.sqrt(noise[:, None] ** 2 +
                            (MODEL_FRAC * np.maximum(W, 0.0)) ** 2)).reshape(-1)
        A = (M * Wt[:, None])[ok]
        y = (W.reshape(-1) * Wt)[ok]
    live = None
    if COL_GUARD > 0:
        cn = np.sqrt((A ** 2).sum(0))
        live = cn >= COL_GUARD * cn.max()
        if live.all():
            live = None
    try:
        if live is None:
            q, rn = nnls(A, y, maxiter=50 * M.shape[1])
        else:
            q = np.zeros(M.shape[1])
            q[live], rn = nnls(A[:, live], y, maxiter=50 * M.shape[1])
    except Exception:
        return np.inf, None
    chi = rn * rn
    if censor and SAT_SOLVE and sat.any():
        # active set of the one-sided bound model >= clip on the censored samples
        Wall = (np.repeat(1.0 / noise, NSAMP) if MODEL_FRAC <= 0 else
                (1.0 / np.sqrt(noise[:, None] ** 2 + (MODEL_FRAC * np.maximum(W, 0.0)) ** 2)).reshape(-1))
        Af, yf = M * Wall[:, None], W.reshape(-1) * Wall
        sat1 = sat.reshape(-1)
        act = np.zeros_like(sat1)
        for _ in range(6):
            below = sat1 & ((M @ q) < W.reshape(-1)) & ~act
            if not below.any():
                break
            act |= below
            use = ok | act
            try:
                if live is None:
                    q, rn = nnls(Af[use], yf[use], maxiter=50 * M.shape[1])
                else:
                    q = np.zeros(M.shape[1])
                    q[live], rn = nnls(Af[use][:, live], yf[use], maxiter=50 * M.shape[1])
            except Exception:
                return np.inf, None
        # chi2 over the unclipped samples only; the bound's residual enters through the penalty below
        r = Af[ok] @ q - yf[ok]
        chi = float(r @ r)
    if censor and sat.any():
        model = (M @ q).reshape(W.shape)
        pen = np.maximum(0.0, W[sat] - model[sat]) / np.repeat(
            noise, NSAMP).reshape(W.shape)[sat]
        chi += float((pen ** 2).sum())
        if ZS_BOUND > 0:
            # suppressed samples (exact 0, censored above) are upper bounds: model <= k * noise
            zs = sat & (W == 0.0)
            if zs.any():
                nz = np.broadcast_to(noise[:, None], W.shape)[zs]
                pen = np.maximum(0.0, model[zs] - ZS_BOUND * nz) / nz
                chi += float((pen ** 2).sum())
    return chi, q


def model_waveforms(plane, pos, p0, w, t0, q, hyper):
    return (build_matrix(plane, pos, p0, w, t0, hyper) @ q).reshape(len(pos), NSAMP)


# ------------------------------------------------------- two tracks, one plane
# Nothing in the forward model assumes one track except build_matrix being
# called once: the charge profile is already a free non-negative vector, so a
# second track is a second block of columns in the SAME linear solve.
#
#     W  ~  M(theta_a) qa + M(theta_b) qb ,   qa, qb >= 0
#
# The outer parameters are (p0, w, t0) per track, 6 in all (5 with t0 tied).
# Design and acceptance criteria: sept26_prelim_analysis/HANDOFF_JOINT_TWO_TRACK_FIT.md.

#: Below this the two tracks are not distinguishable and the charge split
#: between their (identical) column blocks is arbitrary. Transverse: a
#: charge-weighted r.m.s. over the drift column, so two tracks that CROSS in
#: this plane still count as separated (a mid-column distance would call them
#: degenerate at the crossing). Temporal: one depth bin, since two tracks at
#: the same place but a bin apart in time are resolved by the profile alone.
#:
#: The transverse floor is the reach of the resistive sharing kernel, not a tuned number: a track puts charge on its
#: own strip and on +-1 and +-2 (c2 < c1; +-2 strips hold 98 % of it), so two tracks less than 4 pitches apart share
#: kernel cores and the split of the charge between them is the model's choice, not the data's.  It was 1.2 mm
#: (~1.5 pitches) until 2026-10-27; the fleet gate showed what lives below 4 pitches on clean bench single muons --
#: the brightest, worst-chi2 tracks (94-99th percentile) explained as two parallel tracks 2 mm apart sharing the
#: charge 40/60: 4.1 % of det6 Y singles, 0.9 % det4, 1.1 % det7 X (rerun/model_audit/ws_unify/REPORT.md).  Same
#: readout pitch and kernel structure on every chamber, so one floor for all of them.
TWO_MIN_SEP_MM = 4 * 398.58 / 512    # 3.11 mm: four strip pitches
TWO_MIN_DT_NS = 60.0          # one depth bin
#: Backstop only. The end-to-end failure -- two straight lines fitting ONE
#: track's column by taking half of it each in depth -- is handled by measuring
#: the separation *where both children have charge* (see
#: :func:`two_track_separation`), not by a threshold on the overlap itself. A
#: threshold there is a bad instrument: with the same quantity in the
#: optimiser's barrier the fit parks exactly on it, and the guard then decides
#: genuine 15 mm pairs on the fourth decimal (measured 2026-09-16).
TWO_MIN_OVERLAP = 0.05
#: Multiplicative barrier on the collapsed basin. Scale-free on purpose: chi2
#: here runs 1e3-1e5 and an additive constant would be a threshold in disguise.
#: It is optimisation hygiene only -- the real protection is the ``guards_ok``
#: verdict that ``wft.reco.fit_plane_two`` applies to the result.
TWO_BARRIER = 2.0


def two_track_separation(pa, pb, wgt=None) -> float:
    """R.m.s. transverse distance between two tracks over the drift column [mm],
    weighted by ``wgt`` -- one weight per depth bin.

    The weight is what makes this the right question. Weighted by the depths
    where BOTH children carry charge, it is large for two tracks side by side
    (including two that cross: the crossing is one depth out of eighteen) and
    zero for one track cut in half end to end, where there is no depth at which
    both children exist. A zero-sum weight therefore returns 0 -- not a fallback
    to the unweighted r.m.s., which is exactly the number the end-to-end failure
    makes look enormous."""
    d = (pa[0] - pb[0]) + (pa[1] - pb[1]) * UK
    if wgt is None:
        return float(np.sqrt(np.mean(d * d)))
    w = np.asarray(wgt, float)[:len(d)]
    s = w.sum()
    return float(np.sqrt((w * d * d).sum() / s)) if s > 0 else 0.0


def two_track_distinguishability(pa, pb, wgt=None) -> float:
    """How far apart the two tracks are, in units of the resolvability floor;
    < 1 means the pair is degenerate. Transverse and temporal separation add
    in quadrature: either one alone is enough."""
    sep = two_track_separation(pa, pb, wgt)
    dt = abs(pa[2] - pb[2])
    return float(np.hypot(sep / TWO_MIN_SEP_MM, dt / TWO_MIN_DT_NS))


def constrained_bins(t0: float) -> np.ndarray:
    """Which depth bins the DAQ window actually constrains, for a track at t0.

    A bin whose charge arrives after the last sample contributes an almost-zero
    column, so NNLS is free to park an arbitrary amount of charge in it -- which
    it does: run_145 tracks carry fitted ``q_sum`` up to 1e26, and ``q_uend``
    reads the last bin for nearly every track. That is a property of the
    production model, not of this fit, but any guard built on the charge profile
    has to look only where the profile means something."""
    last = float(TS[-1]) if len(TS) else 0.0
    arr = t0 + UK
    return (arr >= -0.5 * SNS) & (arr <= last)


def profile_overlap(qa, qb, bins=None) -> float:
    """How much of the drift column the two children share, 0 to 1: the
    histogram intersection of their normalised charge profiles. Two real
    coincident tracks each cross the whole gap and overlap almost completely;
    two halves of one column do not overlap at all."""
    qa, qb = _masked(qa, bins), _masked(qb, bins)
    sa, sb = qa.sum(), qb.sum()
    if sa <= 0 or sb <= 0:
        return 0.0
    return float(np.minimum(qa / sa, qb / sb).sum())


def _masked(q, bins=None) -> np.ndarray:
    q = np.asarray(q, float)
    return q if bins is None else q * np.asarray(bins, float)[:len(q)]


def common_weight(qa, qb, bins=None) -> np.ndarray:
    """Per depth bin, how much of the column the two children SHARE: the
    pointwise minimum of their normalised charge profiles. Zero everywhere for
    one track cut in half end to end."""
    ma, mb = _masked(qa, bins), _masked(qb, bins)
    sa, sb = ma.sum(), mb.sum()
    return (np.minimum(ma / sa, mb / sb) if sa > 0 and sb > 0
            else np.zeros_like(ma))


def build_matrix_two(plane, pos, pa, pb, hyper):
    """(strip x sample, 2K) design matrix of two straight tracks."""
    return np.hstack([build_matrix(plane, pos, pa[0], pa[1], pa[2], hyper),
                      build_matrix(plane, pos, pb[0], pb[1], pb[2], hyper)])


def chi2_plane_two(plane, W, noise, pos, sat, pa, pb, hyper, censor=True,
                   snap_t0=True):
    """chi2 of the two-track model, both charge profiles profiled out together.

    ``pa``/``pb`` are ``(p0, w, t0)``. Returns ``(chi2, qa, qb)``. Setting
    ``qb = 0`` reproduces the one-track model exactly, so chi2 here can never
    exceed the one-track chi2 at the same ``pa`` — which is what makes
    ``dchi2 = chi2_one - chi2_two`` a model-selection statistic and not a fit
    artefact."""
    if snap_t0:
        pa = (pa[0], pa[1], round(pa[2] / T0_STEP) * T0_STEP)
        pb = (pb[0], pb[1], round(pb[2] / T0_STEP) * T0_STEP)
    M = build_matrix_two(plane, pos, pa, pb, hyper)
    chi, q = _solve_nnls(M, W, noise, sat, censor)
    if q is None:
        return np.inf, None, None
    return chi, q[:K], q[K:]


def _two_pack(pa, pb, tie_t0):
    return (np.array([pa[0], pa[1], pb[0], pb[1], 0.5 * (pa[2] + pb[2])])
            if tie_t0 else np.array([pa[0], pa[1], pa[2], pb[0], pb[1], pb[2]]))


def _two_unpack(v, tie_t0):
    if tie_t0:
        return (v[0], v[1], v[4]), (v[2], v[3], v[4])
    return (v[0], v[1], v[2]), (v[3], v[4], v[5])


_TWO_SIMPLEX_FREE = np.array([[0, 0, 0, 0, 0, 0], [0.4, 0, 0, 0, 0, 0],
                              [0, 1.5e-3, 0, 0, 0, 0], [0, 0, 20, 0, 0, 0],
                              [0, 0, 0, 0.4, 0, 0], [0, 0, 0, 0, 1.5e-3, 0],
                              [0, 0, 0, 0, 0, 20]], float)
_TWO_SIMPLEX_TIED = np.array([[0, 0, 0, 0, 0], [0.4, 0, 0, 0, 0],
                              [0, 1.5e-3, 0, 0, 0], [0, 0, 0.4, 0, 0],
                              [0, 0, 0, 1.5e-3, 0], [0, 0, 0, 0, 20]], float)


#: How many of the offered starting points are actually refined. All of them
#: are scored first (one chi2 each, ~1 ms); Nelder-Mead, which costs a few
#: hundred of those, runs only on the best few.
TWO_N_REFINE = 2


def fit_plane_two_raw(W, noise, pos, sat, plane, starts, hyper=None,
                      wgt=None, maxiter=220, maxiter_polish=140,
                      n_refine=TWO_N_REFINE, bins=None):
    """Fit two tracks to one prepared plane window from several starting points.

    ``starts``: list of ``(pa, pb, tie_t0)``. All are scored, the best
    ``n_refine`` are refined by Nelder-Mead on (p0, w, t0) per track — 6
    parameters, or 5 with ``tie_t0`` — with the collapsed basin held off by a
    multiplicative barrier, and the winner is polished once more. Returns the
    best result as a dict, or None.

    The caller prepares the window (``prep_plane``) once and passes it in: the
    two-track fit is always a SECOND look at a window a one-track fit has
    already seen, so re-preparing it here would be waste."""
    _require_cal()
    hyper = hyper or HYPER
    dof = int((~sat).sum())

    def obj(v, tie_t0):
        pa, pb = _two_unpack(v, tie_t0)
        c, qa, qb = chi2_plane_two(plane, W, noise, pos, sat, pa, pb, hyper,
                                   snap_t0=False)
        if not np.isfinite(c):
            return np.inf
        # the barrier measures exactly what the guard will: separation at the
        # depths where both children have charge. One quantity, so the
        # optimiser is not pushed onto a threshold the guard then rules on.
        d = two_track_distinguishability(pa, pb, common_weight(qa, qb, bins))
        return c * (1.0 + TWO_BARRIER * (1.0 - d) ** 2) if d < 1.0 else c

    scored = sorted((obj(_two_pack(pa, pb, tie), tie), i, (pa, pb, tie))
                    for i, (pa, pb, tie) in enumerate(starts))
    nfev = len(starts)
    best = None
    for c0, _i, (pa, pb, tie) in scored[:max(1, n_refine)]:
        if not np.isfinite(c0):
            continue
        v0 = _two_pack(pa, pb, tie)
        simplex = v0 + (_TWO_SIMPLEX_TIED if tie else _TWO_SIMPLEX_FREE)
        r = minimize(obj, v0, args=(tie,), method='Nelder-Mead',
                     options=dict(xatol=1e-3, fatol=0.3, maxiter=maxiter,
                                  initial_simplex=simplex))
        nfev += r.nfev
        if best is None or r.fun < best[0]:
            best = (float(r.fun), r.x, tie)
    if best is None:
        return None
    r = minimize(obj, best[1], args=(best[2],), method='Nelder-Mead',
                 options=dict(xatol=5e-4, fatol=0.15, maxiter=maxiter_polish))
    nfev += r.nfev
    pa, pb = _two_unpack(r.x, best[2])
    chi, qa, qb = chi2_plane_two(plane, W, noise, pos, sat, pa, pb, hyper,
                                 snap_t0=False)
    if qa is None:
        return None
    # label order: the track at the smaller position in the middle of the drift
    # column is 'a'. Mid-column rather than at the mesh because crossing tracks
    # swap order at the mesh but rarely at mid-column.
    u_mid = 0.5 * K * DT
    if pa[0] + pa[1] * u_mid > pb[0] + pb[1] * u_mid:
        pa, pb, qa, qb = pb, pa, qb, qa
    # the guards are measured on the FITTED profiles, not the parent's: the
    # separation that matters is the one at the depths where both tracks have
    # charge, and that is also what makes an end-to-end pair fail.
    common = common_weight(qa, qb, bins)
    return dict(chi2=float(chi), dof=dof, tie_t0=bool(best[2]), nfev=int(nfev),
                pa=tuple(float(x) for x in pa), pb=tuple(float(x) for x in pb),
                qa=qa, qb=qb, overlap=profile_overlap(qa, qb, bins),
                sep=two_track_separation(pa, pb, common),
                dist=two_track_distinguishability(pa, pb, common))


# ------------------------------------------------- variable projection (FITTER = 'lm')
def _frac_derivs(pos, pc, var):
    """Gaussian strip fractions (as :func:`_fractions_sig`) and their derivatives d/d(centre), d/d(variance)."""
    s = np.sqrt(var)[None, :]
    a = pos[:, None] + PITCH / 2 - pc[None, :]
    b = pos[:, None] - PITCH / 2 - pc[None, :]
    z = 1.0 / (np.sqrt(2) * s)
    G = 0.5 * (erf(a * z) - erf(b * z))
    pa = np.exp(-(a * z) ** 2) / (np.sqrt(2 * np.pi) * s)
    pb = np.exp(-(b * z) ** 2) / (np.sqrt(2 * np.pi) * s)
    return G, pb - pa, -(a * pa - b * pb) / (2.0 * s ** 2)


def _rc_dmq(plane, pos, p0, w, t0, hyper, q):
    """(n_strip * NSAMP, 3): d(M q)/d(p0, w, t0) of :func:`build_matrix_rc` at fixed q, analytic.  p0 and w enter
    only through each bin's centre p0 + w UK and its width (half^2 / 3 = w^2 DT^2 / 12); t0 through the linear
    interpolation between the 5 ns tensors (the derivative of the function that is actually evaluated).  None for t0
    when RC_T0_INTERP is off (chi2 is a staircase in t0 there; the caller differences it)."""
    tp = 'x' if hyper.get('rc_tmpl', 'x') in ('x', 1, 1.0) else plane
    if RC_T0_INTERP:
        t0a = np.floor(t0 / T0_STEP) * T0_STEP
        f = (t0 - t0a) / T0_STEP
        Ha, Hb = _rc_tensors(plane, t0a, tp), _rc_tensors(plane, t0a + T0_STEP, tp)
        Hs = Ha if f < 1e-9 else (1.0 - f) * Ha + f * Hb
        dHs = (Hb - Ha) / T0_STEP
    else:
        Hs, dHs = _rc_tensors(plane, round(t0 / T0_STEP) * T0_STEP, tp), None
    Hq = Hs * q[None, None, :]
    dHq = dHs * q[None, None, :] if dHs is not None else None
    pc = p0 + w * UK
    s0 = float(hyper.get(f'sigma_p0_{plane}', hyper['sigma_p0']))
    var0 = s0 ** 2 + hyper['Dp'] ** 2 * UK + (w * DT) ** 2 / 12.0
    dvar_dw = w * DT ** 2 / 6.0
    cf = float(hyper.get(f'core_frac_{plane}', 0.0))
    dvc = float(hyper.get('core_sigma', CORE_SIGMA)) ** 2 - s0 ** 2

    def fderiv(v):
        G, gc, gv = _frac_derivs(pos, pc, v)
        if cf <= 0:
            return G, gc, gv
        Gc, gcc, gvc = _frac_derivs(pos, pc, v + dvc)
        return (1 - cf) * G + cf * Gc, (1 - cf) * gc + cf * gcc, (1 - cf) * gv + cf * gvc
    G, gc, gv = fderiv(var0)
    eta = float(hyper.get(f'lor_frac_{plane}', 0.0))
    F0, Fc, Fv = G, gc, gv
    if eta > 0:
        gam = float(hyper[f'lor_gamma_{plane}'])
        ah = pos[:, None] + PITCH / 2 - pc[None, :]
        al = pos[:, None] - PITCH / 2 - pc[None, :]
        L = (np.arctan(ah / gam) - np.arctan(al / gam)) / np.pi
        Lc = -(gam / (gam ** 2 + ah ** 2) - gam / (gam ** 2 + al ** 2)) / np.pi
        F0, Fc, Fv = (1 - eta) * G + eta * L, (1 - eta) * gc + eta * Lc, (1 - eta) * gv
    n = len(pos)
    out = np.zeros((n, NSAMP, 3))

    def add(F, dFp, dFw, H, dH):
        out[:, :, 0] += dFp @ H.T
        out[:, :, 1] += dFw @ H.T
        if dH is not None:
            out[:, :, 2] += F @ dH.T

    dF0p, dF0w = Fc, Fc * UK[None, :] + Fv * dvar_dw
    add(F0, dF0p, dF0w, Hq[0], dHq[0] if dHq is not None else None)
    asym = float(hyper.get(f'rc_asym_{plane}', 0.0))
    if asym != 0.0:
        Hs_a = _rc_shift_tensor(plane, t0q_of(t0), tp, float(hyper.get(f'rc_asym_tau_{plane}', 110.0))) * q[None, :]

        def shifted(A):
            B = np.zeros_like(A)
            if asym > 0:
                B[:-1] = A[1:]
            else:
                B[1:] = A[:-1]
            return abs(asym) * (B - A)
        add(shifted(F0), shifted(dF0p), shifted(dF0w), Hs_a, None)   # snapped t0: no t0 derivative
    D = float(hyper.get(f'rc_D_{plane}', 0.0))
    if D > 0:
        drain = float(hyper.get('rc_drain', 0.0))
        Fp, Fpp, Fpw = F0, dF0p, dF0w
        for j, s in enumerate(RC_EDGES[1:]):
            Gj, gcj, gvj = fderiv(var0 + 2.0 * D * s)
            e = np.exp(-s / drain) if drain > 0 else 1.0
            Fj, Fjp, Fjw = e * Gj, e * gcj, e * (gcj * UK[None, :] + gvj * dvar_dw)
            add(Fj - Fp, Fjp - Fpp, Fjw - Fpw, Hq[j + 1], dHq[j + 1] if dHq is not None else None)
            Fp, Fpp, Fpw = Fj, Fjp, Fjw
    J = out.reshape(n * NSAMP, 3)
    if dHs is None:
        J[:, 2] = np.nan
    return J


def _dmq(plane, pos, th, hyper, q):
    """d(M q)/d(p0, w, t0): analytic on the RC kernel, central differences otherwise (and for t0 without interp)."""
    J = _rc_dmq(plane, pos, th[0], th[1], th[2], hyper, q) if 'rc_D_y' in hyper else None
    hs = (0.02, 1e-4, T0_STEP if 'rc_D_y' in hyper else 1.0)
    for j in range(3):
        if J is not None and not np.isnan(J[0, j]):
            continue
        tp, tm = np.array(th, float), np.array(th, float)
        tp[j] += hs[j]
        tm[j] -= hs[j]
        col = (build_matrix(plane, pos, *tp, hyper) @ q - build_matrix(plane, pos, *tm, hyper) @ q) / (2 * hs[j])
        if J is None:
            J = np.empty((len(col), 3))
        J[:, j] = col
    return J


def _vp_eval(plane, W, noise, pos, sat, th, hyper, censor, t0_prior):
    """chi2_plane at an unsnapped t0, also returning the design matrix."""
    M = build_matrix(plane, pos, th[0], th[1], th[2], hyper)
    chi, q = _solve_nnls(M, W, noise, sat, censor)
    if q is None:
        return np.inf, None, M
    if t0_prior is not None:
        chi += ((th[2] - t0_prior[0]) / t0_prior[1]) ** 2
    return chi, q, M


def _vp_resid_jac(plane, W, noise, pos, sat, th, q, M, hyper, censor, t0_prior):
    """(r, J) with r @ r = chi2 of :func:`_solve_nnls` (+ prior) at theta, and J the variable-projection Jacobian in
    Kaufman's form: d(M q)/d(theta) at fixed q, projected off the span of the free (q > 0) weighted columns over the
    rows that determine q (the live samples and the clipped samples the model falls below)."""
    if MODEL_FRAC <= 0:
        Wt = np.repeat(1.0 / noise, NSAMP)
    else:
        Wt = (1.0 / np.sqrt(noise[:, None] ** 2 + (MODEL_FRAC * np.maximum(W, 0.0)) ** 2)).reshape(-1)
    y = W.reshape(-1)
    m = M @ q
    ok = ~sat.reshape(-1)
    dmq = _dmq(plane, pos, th, hyper, q)
    rows_r, rows_J = [((m - y) * Wt)[ok]], [(dmq * Wt[:, None])[ok]]
    A = [(M * Wt[:, None])[ok]]
    if censor and sat.any():
        s1 = sat.reshape(-1)
        nz = np.repeat(noise, NSAMP)
        pen = s1 & (m < y)                       # the one-sided bound is active (= the SAT_SOLVE active set)
        if pen.any():
            rows_r.append((m - y)[pen] / nz[pen])
            rows_J.append(dmq[pen] / nz[pen][:, None])
            if SAT_SOLVE:
                A.append(M[pen] / nz[pen][:, None])
        if ZS_BOUND > 0:
            zs = s1 & (y == 0.0)
            act = zs & (m > ZS_BOUND * nz)
            if act.any():
                rows_r.append((m[act] - ZS_BOUND * nz[act]) / nz[act])
                rows_J.append(dmq[act] / nz[act][:, None])
    nA = sum(len(a) for a in A)
    J = np.vstack(rows_J)
    F = q > 0
    if F.any():
        Qf, _ = np.linalg.qr(np.vstack(A)[:, F])
        J[:nA] -= Qf @ (Qf.T @ J[:nA])
    r = np.concatenate(rows_r)
    if t0_prior is not None:
        r = np.r_[r, (th[2] - t0_prior[0]) / t0_prior[1]]
        J = np.vstack([J, [0.0, 0.0, 1.0 / t0_prior[1]]])
    return r, J


def _lm_refine(plane, W, noise, pos, sat, th0, hyper, censor=True, t0_prior=None, maxit=None, t0_win=None):
    """Levenberg-Marquardt (Marquardt diagonal scaling) on the NNLS-profiled residual.  Every trial point is judged
    by the exact chi2, so the Jacobian only has to point downhill.  Returns (chi2, theta, q, nfev, njac)."""
    maxit = LM_MAXIT if maxit is None else maxit
    th = np.array(th0, float)
    chi, q, M = _vp_eval(plane, W, noise, pos, sat, th, hyper, censor, t0_prior)
    nfev, njac = 1, 0
    if not np.isfinite(chi):
        return np.inf, th, None, nfev, njac
    lam = 1e-3
    for _ in range(maxit):
        r, J = _vp_resid_jac(plane, W, noise, pos, sat, th, q, M, hyper, censor, t0_prior)
        njac += 1
        if not (np.isfinite(r).all() and np.isfinite(J).all()):
            break                                  # no usable direction here: stop at the current (finite) point
        g, H = J.T @ r, J.T @ J
        D = np.maximum(np.diag(H), 1e-12)
        accepted = False
        for _ in range(8):
            try:
                step = -np.linalg.solve(H + lam * np.diag(D), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            if not np.isfinite(step).all() or (t0_win is not None and not t0_win[0] <= th[2] + step[2] <= t0_win[1]):
                lam *= 10
                continue
            c2, q2, M2 = _vp_eval(plane, W, noise, pos, sat, th + step, hyper, censor, t0_prior)
            nfev += 1
            if np.isfinite(c2) and c2 < chi:
                dchi = chi - c2
                th, chi, q, M = th + step, c2, q2, M2
                lam = max(lam / 5, 1e-7)
                accepted = True
                break
            lam *= 10
        if not accepted:
            break
        if dchi < 0.05 or (abs(step[0]) < 1e-4 and abs(step[1]) < 1e-6 and abs(step[2]) < 0.05):
            break
    return float(chi), th, q, nfev, njac


def _fit_plane_lm(W, noise, pos, sat, plane, p0_init, w_init, t0_init, hyper, t0_prior, pivot):
    """FITTER = 'lm' body of :func:`fit_plane_raw` (same starts in t0 as the grid, see FITTER)."""
    nfev = njac = 0
    # global start in t0: the grid's 7-point, 40 ns ladder, along the pivot line when the start came from it
    # (and at the fixed p0 of the grid: the two families end in different lattice basins on ~1/3 of planes)
    best = (np.inf, None)
    if LM_START == 'grid':
        # the 'nm' grid, point for point (t0 snapped to the tensor grid as chi2_plane does there)
        starts = [(p0_init + dp, w_init + dw, round(t0 / T0_STEP) * T0_STEP)
                  for t0 in np.arange(t0_init - 120, t0_init + 121, 40.0) for dp in (-0.8, 0.0, 0.8)
                  for dw in (-2.4e-3, -0.8e-3, 0.0, 0.8e-3, 2.4e-3)]
    else:
        starts = []
        for t0 in np.arange(t0_init - 120, t0_init + 121, 40.0):
            starts += [(p0, w_init, t0) for p0 in
                       ({p0_init} if pivot is None else {p0_init, pivot[0] - w_init * (pivot[1] - t0)})]
    for st in starts:
        c, _q, _M = _vp_eval(plane, W, noise, pos, sat, st, hyper, True, t0_prior)
        nfev += 1
        if c < best[0]:
            best = (c, st)
    if best[1] is None:
        return dict(chi2=np.inf, p0=float(p0_init), w=float(w_init), t0=float(t0_init), q=None, nfev=nfev)
    win = (t0_init - LM_T0_RANGE, t0_init + LM_T0_RANGE)
    chi, th, q, nf, nj = _lm_refine(plane, W, noise, pos, sat, best[1], hyper, t0_prior=t0_prior, t0_win=win)
    nfev, njac = nfev + nf, njac + nj
    if LM_HOPS > 0 and np.isfinite(chi):
        th_c = th
        for s in (-1.0, 1.0):
            th_h = th_c
            for _ in range(LM_HOPS):
                c_s, th2, q2, nf, nj = _lm_refine(plane, W, noise, pos, sat,
                                                  th_h + s * np.array([DT * th_h[1], 0.0, DT]), hyper,
                                                  t0_prior=t0_prior, t0_win=win)
                nfev, njac = nfev + nf, njac + nj
                if not c_s < chi:
                    break
                chi, th, q, th_h = c_s, th2, q2, th2
    # short polish: the surface is only piecewise smooth (NNLS active-set changes, the 5 ns tensor interpolation)

    def obj(v):
        if not win[0] <= v[2] <= win[1]:
            return np.inf
        return _vp_eval(plane, W, noise, pos, sat, v, hyper, True, t0_prior)[0]
    r = minimize(obj, th, method='Nelder-Mead',
                 options=dict(xatol=1e-4, fatol=0.02, maxiter=80,
                              initial_simplex=th + np.array([[0, 0, 0], [0.1, 0, 0], [0, 4e-4, 0], [0, 0, 5.0]])))
    nfev += r.nfev
    if r.fun < chi:
        th = np.asarray(r.x, float)
    chi, q, M = _vp_eval(plane, W, noise, pos, sat, th, hyper, True, t0_prior)
    cov = None
    if q is not None:
        _r, J = _vp_resid_jac(plane, W, noise, pos, sat, th, q, M, hyper, True, t0_prior)
        njac += 1
        try:
            cov = np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            cov = None
    return dict(chi2=chi, p0=float(th[0]), w=float(th[1]), t0=float(th[2]), q=q, nfev=nfev, njac=njac, cov=cov)


def vp_covariance(P, plane, r, hyper=None, t0_prior=None):
    """Unscaled 3x3 (p0, w, t0) covariance (J^T J)^-1 of a fit result ``r`` (any FITTER), from the variable-projection
    Jacobian at its minimum; None if singular."""
    hyper = hyper or HYPER
    W, noise, pos, sat = prep_plane(P, plane)
    th = np.array([r['p0'], r['w'], r['t0']], float)
    chi, q, M = _vp_eval(plane, W, noise, pos, sat, th, hyper, True, t0_prior)
    if q is None:
        return None
    _r, J = _vp_resid_jac(plane, W, noise, pos, sat, th, q, M, hyper, True, t0_prior)
    try:
        return np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return None


# --------------------------------------------------------------------- fits
def fit_plane_raw(P, plane, p0_init, w_init, t0_init, hyper=None, fix_p0w=None,
                  t0_prior=None, pivot=None):
    """Two-stage fit: coarse (p0, w, t0) grid, then Nelder-Mead. With
    ``fix_p0w=(p0, w)`` only t0 and the charge profile are fitted — that is the
    'ref-pinned' configuration used for calibration and for the chi2(v) scan.
    ``t0_prior=(t0_pred, sigma)`` is passed through to :func:`chi2_plane`.
    ``pivot=(x_c, t_c)``: the start lies on the pivot line (wft.reco.PIVOT_START); FITTER = 'lm' then moves p0 with
    t0 along it.  With FITTER = 'lm' the result also carries ``cov``, the 3x3 (p0, w, t0) covariance (unscaled)."""
    _require_cal()
    hyper = hyper or HYPER
    W, noise, pos, sat = prep_plane(P, plane)
    dof = int((~sat).sum())
    if FITTER == 'lm' and fix_p0w is None:
        r = _fit_plane_lm(W, noise, pos, sat, plane, p0_init, w_init, t0_init, hyper, t0_prior, pivot)
        r['dof'] = dof
        return r

    if fix_p0w is not None:
        p0f, wf = fix_p0w
        grid = np.arange(t0_init - 240, t0_init + 241, 20.0)
        cs = [chi2_plane(plane, W, noise, pos, sat, p0f, wf, t, hyper,
                         t0_prior=t0_prior)[0]
              for t in grid]
        j = int(np.argmin(cs))
        g2 = np.arange(grid[j] - 20, grid[j] + 21, T0_STEP)
        cs2 = [chi2_plane(plane, W, noise, pos, sat, p0f, wf, t, hyper,
                          t0_prior=t0_prior)[0]
               for t in g2]
        t0b = float(g2[int(np.argmin(cs2))])
        chi, q = chi2_plane(plane, W, noise, pos, sat, p0f, wf, t0b, hyper,
                            t0_prior=t0_prior)
        return dict(chi2=chi, dof=dof, p0=p0f, w=wf, t0=t0b, q=q, nfev=len(cs) + len(cs2))

    nfev = 0
    best = (np.inf, p0_init, w_init, t0_init)
    for t0 in np.arange(t0_init - 120, t0_init + 121, 40.0):
        for dp in (-0.8, 0.0, 0.8):
            for dw in (-2.4e-3, -0.8e-3, 0.0, 0.8e-3, 2.4e-3):
                c, _ = chi2_plane(plane, W, noise, pos, sat,
                                  p0_init + dp, w_init + dw, t0, hyper,
                                  t0_prior=t0_prior)
                nfev += 1
                if c < best[0]:
                    best = (c, p0_init + dp, w_init + dw, t0)

    def obj(v):
        return chi2_plane(plane, W, noise, pos, sat, v[0], v[1], v[2], hyper,
                          snap_t0=False, t0_prior=t0_prior)[0]

    v0 = np.array(best[1:])
    r = minimize(obj, v0, method='Nelder-Mead',
                 options=dict(xatol=1e-3, fatol=0.3, maxiter=140,
                              initial_simplex=v0 + np.array(
                                  [[0, 0, 0], [0.4, 0, 0], [0, 1.5e-3, 0],
                                   [0, 0, 20]])))
    chi, q = chi2_plane(plane, W, noise, pos, sat, r.x[0], r.x[1], r.x[2],
                        hyper, snap_t0=False, t0_prior=t0_prior)
    return dict(chi2=chi, dof=dof, p0=float(r.x[0]), w=float(r.x[1]),
                t0=float(r.x[2]), q=q, nfev=nfev + r.nfev)


def fit_joint(evx, evy, ftst_diff, p0x_i, wx_i, p0y_i, wy_i, t0_i, hyper=None):
    """Joint two-plane fit: one shared charge profile (per-plane scale), t0
    tied through the measured FEU offset. Its value is stabilising a
    near-vertical plane, where timing carries no slope information."""
    _require_cal()
    hyper = hyper or HYPER
    dt = DT_XY.get(int(ftst_diff), -18.8)
    Wx, nx, px_, sx = prep_plane(evx, 'x')
    Wy, ny, py_, sy = prep_plane(evy, 'y')
    dof = int((~sx).sum() + (~sy).sum())

    def solve(p0x, wx, p0y, wy, t0):
        Mx = build_matrix('x', px_, p0x, wx, t0, hyper)
        My = build_matrix('y', py_, p0y, wy, t0 - dt, hyper)
        okx, oky = ~sx.reshape(-1), ~sy.reshape(-1)
        Ax = (Mx * np.repeat(1.0 / nx, NSAMP)[:, None])[okx]
        Ay = (My * np.repeat(1.0 / ny, NSAMP)[:, None])[oky]
        yx = (Wx / nx[:, None]).reshape(-1)[okx]
        yy = (Wy / ny[:, None]).reshape(-1)[oky]
        alpha = 1.0
        my = None
        for _ in range(2):
            try:
                q, _rn = nnls(np.vstack([Ax, alpha * Ay]),
                              np.concatenate([yx, yy]), maxiter=50 * K)
            except Exception:
                return np.inf, None, 1.0
            my = Ay @ q
            den = float(my @ my)
            alpha = float(my @ yy / den) if den > 0 else 1.0
        chi = float(((Ax @ q - yx) ** 2).sum() + ((alpha * my - yy) ** 2).sum())
        return chi, q, alpha

    v0 = np.array([p0x_i, wx_i, p0y_i, wy_i, t0_i])
    r = minimize(lambda v: solve(*v)[0], v0, method='Nelder-Mead',
                 options=dict(xatol=1e-3, fatol=0.5, maxiter=500,
                              initial_simplex=v0 + np.array(
                                  [[0, 0, 0, 0, 0], [0.8, 0, 0, 0, 0],
                                   [0, 0.004, 0, 0, 0], [0, 0, 0.8, 0, 0],
                                   [0, 0, 0, 0.004, 0], [0, 0, 0, 0, 50]])))
    r = minimize(lambda v: solve(*v)[0], r.x, method='Nelder-Mead',
                 options=dict(xatol=5e-4, fatol=0.3, maxiter=300))
    chi, q, alpha = solve(*r.x)
    return dict(chi2=chi, dof=dof, p0x=float(r.x[0]), wx=float(r.x[1]),
                p0y=float(r.x[2]), wy=float(r.x[3]), t0=float(r.x[4]),
                q=q, alpha=alpha)


def init_guess(P, plane, tan_seed=0.0, p0_seed=None, v_drift=None):
    """Starting point for the fit. Deliberately crude and reference-free: the
    brightest strip for position, the earliest half-maximum crossing for t0."""
    W = np.asarray(P['W'], dtype=np.float64)
    noise = np.maximum(np.asarray(P['noise'], dtype=np.float64), 3.0)
    pos = np.asarray(P['pos'], dtype=np.float64)
    amax = W.max(axis=1)
    if p0_seed is None:
        p0_seed = float(pos[int(np.argmax(amax))])
    sel = amax > np.maximum(6 * noise, 60)
    if sel.sum() == 0:
        sel = amax == amax.max()
    leads = []
    for wv in W[sel]:
        ipk = int(np.argmax(wv))
        a = wv.max()
        for kk in range(1, ipk + 1):
            if wv[kk] >= 0.5 * a > wv[kk - 1]:
                leads.append(SNS * (kk - 1 + (0.5 * a - wv[kk - 1]) /
                                    (wv[kk] - wv[kk - 1])))
                break
    t0g = min(leads) if leads else 400.0
    v = v_drift if v_drift is not None else (CAL.v_drift if CAL else 36.6)
    return float(p0_seed), float(tan_seed * v * 1e-3), float(t0g)


# the one place a study may move a constant of this module (wft.diag); production runs none
from . import diag as _diag  # noqa: E402
_diag.apply('model', globals())
