"""edgefit.py — Compton-edge extraction from a linear amplitude spectrum.

Generalises the machinery proven in `22_y88_edges.py` (July-17 Y-88 scan) to the
two-source campaign, where a channel may show one edge (Cs-137, 477 keVee) or
two (Y-88, 699 + 1612 keVee), on top of a measured and subtracted background.

Three things are different from 22, all forced by the new data:

1. **Nothing is positioned by hand.** 22 hard-coded fit windows in mV (12-48 mV
   for the 699 edge), tuned to the HV of runs 224476-79; these runs sit ~2.5x
   higher. Here every window is built around a position the data itself
   supplies: for a Cs-137 spectrum the most prominent steepest-descent point
   (`seed_candidates`), for a Y-88 spectrum the position handed down from the
   same channel's Cs-137 fit, and for the outer Y-88 edge the position the
   energy ratio predicts from the fitted 699 one. Centres stay free inside
   their windows, so every ratio remains a measurement.

2. **Background is subtracted, not eyeballed.** The caller passes the dark-run
   template and its live-time scale; the fit weights use the full variance
   sig + scale^2 * bkg, and the bootstrap resamples both.

3. **Two estimators are always reported.** `edge_mv` is the fitted
   resolution-smeared step centre (or bump centre) — the same convention as 22,
   so the two campaigns can be compared directly. `edge_mv_halfheight` is the
   model-independent cross-check (where the smoothed spectrum falls to half its
   pre-edge plateau); the two differ by O(sigma/2) and that difference is the
   convention systematic to carry into any simulation-anchored comparison.

Models: plastics see the Compton continuum step DOWN at the edge (erfc step);
the SiPM-wall bars and the liquids show a localised bump (Gaussian + linear
background), exactly as in 22.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

SQRT2 = np.sqrt(2.0)
VALLEY_HI = 16.0        # noise turn-on / valley sits below this (mV)
EDGE_MIN_MV = 8.0       # never call an edge below this (noise region)
SMOOTH_MV = 1.0         # Gaussian smoothing sigma (mV), seeding only
N_BOOT = 200
SEED = 224588


def kernel(sigma_bins):
    half = int(np.ceil(4 * sigma_bins))
    x = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    return k / k.sum()


def smooth(y, kern):
    return np.convolve(y, kern, mode='same')


# --- models ----------------------------------------------------------------
def single_step(A, bg, m, s, E, k):
    """One resolution-smeared Compton edge on a SLOPED continuum: bg + m*(A-E)
    below/above, plus a step of height s falling through E with resolution |k|.

    22_y88_edges.py used a flat background. That is not good enough here for two
    reasons, both of which put a real continuum under the edge: (i) in a Y-88
    run the 1836 keV gamma's own Compton continuum runs straight through the
    699 keVee edge, and (ii) Y-88's 898 and 1836 keV gammas are a CASCADE, so a
    source sitting on the bar produces true-coincidence summing — two gammas in
    one event — which adds counts above every single-gamma edge. A flat
    background forces that slope into the step and biases the edge outward.
    """
    from scipy.special import erfc
    return bg + m * (A - E) + s * 0.5 * erfc((A - E) / (SQRT2 * abs(k)))


def gauss_lin(A, a, mu, sig, m, b):
    """Localised edge bump on a linear background (walls, liquids)."""
    return a * np.exp(-0.5 * ((A - mu) / sig) ** 2) + m * A + b


def _fit(model, A, y, sigma, p0, bounds):
    try:
        popt, pcov = curve_fit(model, A, y, p0=p0, sigma=sigma, bounds=bounds,
                               maxfev=20000)
        return popt, pcov
    except Exception:
        return None, None


# --- seeding ---------------------------------------------------------------
def valley_of(cen, sm):
    """End of the noise turn-on: the first local minimum of the smoothed
    spectrum. 22 took the global minimum below a fixed 16 mV, which was tied to
    the 07-17 gains; these runs sit ~2.5x higher and the valley moves with the
    gain, so it is found rather than assumed (the fixed window stays as the
    fallback when there is no clean local minimum)."""
    search = cen < 0.35 * cen[-1]
    if search.sum() > 8:
        y = sm[search]
        mins, props = find_peaks(-y, prominence=max(0.02 * np.nanmax(y), 1.0))
        if len(mins):
            return max(float(cen[search][mins[0]]), EDGE_MIN_MV)
    low = cen < VALLEY_HI
    v = cen[low][np.argmin(sm[low])] if low.any() else EDGE_MIN_MV
    return max(float(v), EDGE_MIN_MV)


def seed_candidates(cen, sm, valley, sigma=None, min_counts=25.0, hi=None):
    """Edge seeds = STEEPEST-DESCENT points, i.e. local maxima of -dN/dA above
    the noise valley, returned in ascending amplitude with their prominence.

    This is used autonomously only on the Cs-137 spectra, where there is exactly
    one edge and the caller takes the most prominent candidate. Three other
    rules were tried on this data first and each is wrong in a way worth
    recording, because all three look reasonable on paper:

    * `-d(log N)/dA` (relative fall) is scale-free and therefore attractive, but
      the steepest *relative* fall in any of these spectra is the point where
      the spectrum runs out of statistics. On a Y-88 bar that fake seed beat
      both real edges — it labelled the 1612 keVee edge as 699 and then found
      its "partner" out in the empty tail at 220 mV.
    * the LOWEST significant candidate is unstable at the few-% prominence
      level: a wiggle just above the noise valley passes in one run and not the
      next, which moved the AR 699 keVee edge between 20 and 31 mV across two
      runs whose spectra are the same shape to 2 %.
    * the MOST PROMINENT candidate is right on a Cs-137 spectrum but not on a
      Y-88 one: on the brightest bars (BL, 8400 hits/trigger) the pileup
      shoulder near the threshold carries more |dN/dA| than the real edge, so
      the fit lands on it and is then rejected, losing the bar entirely.

    Hence the Cs-anchored design in 34: fit the clean single-gamma Cs-137 edge
    per channel first, and hand that position to the Y-88 fit as `prior`.

    Significance: a seed is kept only where the smoothed spectrum still holds
    `min_counts` and stands 5x above its own error — which is what stops the
    fitter from finding structure in the tail.
    """
    hi = hi if hi is not None else cen[-1]
    win = (cen > valley + 2.0) & (cen < hi)
    if win.sum() < 8:
        return [], np.zeros_like(cen)
    d = -np.gradient(sm, cen)
    idx = np.where(win)[0]
    dmax = float(np.nanmax(d[idx]))
    if not np.isfinite(dmax) or dmax <= 0:
        return [], d
    pk, props = find_peaks(d[idx], prominence=0.10 * dmax)
    out = []
    for i, p in enumerate(pk):
        j = idx[p]
        lvl = float(sm[j])
        if lvl < min_counts:
            continue
        if sigma is not None and lvl < 5 * float(sigma[j]):
            continue
        out.append((float(cen[j]), float(props['prominences'][i]),
                    contrast(cen, sm, float(cen[j]))))
    return out, d


def contrast(cen, sm, E):
    """How far the spectrum falls ACROSS a candidate: median just below it over
    median just above. A Compton edge drops by a large factor; a threshold or
    pileup shoulder barely dips. On a single-edge (Cs-137) spectrum this ranks
    candidates correctly where prominence does not — on the brightest bars the
    pileup shoulder carries more |dN/dA| than the edge itself (BL: prominence
    170 at the 20 mV shoulder vs 86 at the real 43 mV edge; contrast 1.0 vs
    3.7). It is NOT usable on a two-edge Y-88 spectrum, where the outer edge
    always wins on contrast because the spectrum ends there."""
    below = (cen > 0.75 * E) & (cen < 0.90 * E)
    above = (cen > 1.15 * E) & (cen < 1.35 * E)
    if below.sum() < 2 or above.sum() < 2:
        return 1.0
    return float(np.median(sm[below]) / max(float(np.median(sm[above])), 1.0))


def half_height(cen, sm, edge):
    """Model-independent cross-check: where the smoothed spectrum falls to half
    of its plateau just below the edge."""
    plateau_win = (cen > 0.55 * edge) & (cen < 0.85 * edge)
    if plateau_win.sum() < 3:
        return None
    plateau = float(np.median(sm[plateau_win]))
    if plateau <= 0:
        return None
    above = np.where((cen > 0.85 * edge) & (sm < 0.5 * plateau))[0]
    return float(cen[above[0]]) if len(above) else None


# --- extractors ------------------------------------------------------------
def fit_steps(cen, y, sigma, kern, energies, prior=None):
    """Plastic: one erfc step per expected edge, fitted low-to-high in windows
    built from the seeds. Returns (positions, curves) with NaN for a failure."""
    sm = smooth(y, kern)
    valley = valley_of(cen, sm)
    if prior is not None and np.isfinite(prior) and prior > 0:
        # Position handed down from this channel's Cs-137 measurement (see 34):
        # a clean single-gamma edge is a far better anchor than any feature
        # search on a Y-88 spectrum, which carries two edges, the cascade-sum
        # continuum and — on the brightest bars — a pileup shoulder that
        # outweighs the real edge. The centre stays free inside the window, so
        # the measured 699/477 ratio is still a measurement.
        s1 = float(prior)
    else:
        cands, _ = seed_candidates(cen, sm, valley, sigma=sigma)
        if not cands:
            return [np.nan] * len(energies), [], valley
        if len(energies) == 1:
            # single-edge (Cs-137) spectrum: rank by how far the spectrum falls
            # across the candidate, which the real edge always wins
            s1 = max(cands, key=lambda c: c[2])[0]
        else:
            # unanchored Y-88 fallback: contrast would pick the OUTER edge, so
            # rank by |dN/dA| prominence instead (the 699 keVee edge sits on a
            # far higher continuum than the 1612 one)
            s1 = max(cands, key=lambda c: c[1])[0]

    def one(lo, hi, seed):
        """Fit a sloped-continuum step in [lo, hi] seeded at `seed`.

        Rejected (returns None) unless the fit actually found an edge: the step
        height must stand 3 sigma above its own error, and the centre must sit
        clear of the window bounds. Without the bound check the outer Y-88 edge
        rails at the low bound and reports a number that is really "no step
        here" — which is what the 1836 keV continuum does on these bars, where
        multiple scattering and cascade summing wash the 1612 keVee edge out.
        """
        m = (cen >= lo) & (cen <= hi)
        if m.sum() < 8:
            return None, None
        A, yy, ss = cen[m], y[m], sigma[m]
        top = max(float(yy.max()), 1.0)
        bg0 = max(float(sm[np.argmin(np.abs(cen - hi))]), 0.5)
        slope_max = 5 * top / max(hi - lo, 1.0)
        p, cov = _fit(single_step, A, yy, ss,
                      [bg0, 0.0, top, seed, max(0.12 * seed, 1.0)],
                      ([-2 * abs(bg0) - 1, -slope_max, 0, lo + 0.5, 0.3],
                       [3 * top, slope_max, 6 * top, hi - 0.5, 0.6 * seed]))
        if p is None:
            return None, None
        E, step = float(p[3]), float(p[2])
        span = hi - lo
        if not (lo + 0.05 * span < E < hi - 0.05 * span):
            return None, None
        if cov is not None and np.isfinite(cov[2, 2]) and cov[2, 2] > 0:
            if step < 3 * np.sqrt(cov[2, 2]):
                return None, None
        xx = np.linspace(A[0], A[-1], 300)
        return E, (xx, single_step(xx, *p))

    out, curves = [], []
    # --- dominant (lowest-energy) edge -------------------------------------
    # The window is kept LOCAL to the edge (roughly +-60 % of the seed): the
    # Compton continuum under it is strongly curved, so a linear background is
    # only honest over a short span. 22's wide fixed windows biased the step
    # centre low here by several mV.
    E1, curve = one(max(valley + 1.0, 0.6 * s1), min(cen[-1], 1.7 * s1), s1)
    if E1 is None:
        return [np.nan] * len(energies), [], valley
    out.append(E1)
    curves.append(curve)

    # --- weaker outer edge (Y-88 1612 keVee) --------------------------------
    # Searched around the position the energy ratio predicts from the FITTED
    # first edge, but with a free centre — so the measured 1612/699 ratio stays
    # a genuine cross-check, it is only the search region that uses the prior.
    # This edge sits on the cascade-summing continuum, so it is the less
    # trustworthy of the two and 34 marks it `secondary`.
    if len(energies) == 2:
        s2 = out[0] * energies[1] / energies[0]
        E2, curve2 = one(max(1.35 * out[0], 0.65 * s2),
                         min(cen[-1], 1.6 * s2), s2)
        out.append(E2 if E2 is not None else np.nan)
        if curve2 is not None:
            curves.append(curve2)
    return out, curves, valley


def fit_bump(cen, y, sigma, kern, energies, prior=None):
    """Wall / liquid: the edge is a localised bump on a falling background. Only
    the lowest expected edge is fitted (the 1612 bump is never significant on
    these channels); the returned list is padded with NaN."""
    sm = smooth(y, kern)
    valley = valley_of(cen, sm)
    seek = (cen > valley + 1) & (cen < 0.6 * cen[-1])
    if seek.sum() < 5:
        return [np.nan] * len(energies), [], valley
    idx = np.where(seek)[0]
    logsm = np.log(np.clip(sm[idx], 1.0, None))
    pk, props = find_peaks(logsm, prominence=0.15)
    if len(pk) == 0:
        return [np.nan] * len(energies), [], valley
    mu0 = float(cen[idx][pk[np.argmax(props['prominences'])]])
    if mu0 < EDGE_MIN_MV + 2:
        return [np.nan] * len(energies), [], valley
    w = max(6.0, 0.35 * mu0)
    fr = (cen >= mu0 - w) & (cen <= mu0 + w)
    if fr.sum() < 6:
        return [np.nan] * len(energies), [], valley
    A, yy, ss = cen[fr], y[fr], sigma[fr]
    bg0 = float(np.interp(mu0, cen[idx][[0, -1]], sm[idx][[0, -1]]))
    a0 = max(float(sm[np.argmin(np.abs(cen - mu0))]) - bg0, 1.0)
    p, _ = _fit(gauss_lin, A, yy, ss, [a0, mu0, w / 2.5, 0.0, bg0],
                ([0, mu0 - w, 1.0, -np.inf, -abs(yy.max())],
                 [max(yy.max(), 1.0) * 2, mu0 + w, w, np.inf,
                  max(yy.max(), 1.0) * 2]))
    if p is None:
        return [np.nan] * len(energies), [], valley
    a, mu = p[0], float(p[1])
    bg_here = p[3] * mu + p[4]
    if a < 3 * np.sqrt(max(abs(bg_here), 1)) or not (mu0 - w < mu < mu0 + w):
        return [np.nan] * len(energies), [], valley
    xx = np.linspace(A[0], A[-1], 300)
    return ([mu] + [np.nan] * (len(energies) - 1), [(xx, gauss_lin(xx, *p))],
            valley)


# --- public entry point ----------------------------------------------------
def extract(cen, sig, bkg, scale, kind, energies, n_boot=N_BOOT, prior=None):
    """Fit the Compton edge(s) of one channel.

    cen       bin centres (mV)
    sig       raw counts in this run
    bkg       summed counts of the dark-run template for this channel
    scale     live-time factor applied to bkg (n_trig_run / n_trig_dark_total)
    kind      'PSS' (step model) | 'WAL' | 'LIQ' (bump model)
    energies  expected edge energies, ascending (keVee)

    Returns dict with per-edge position, bootstrap error, half-height
    cross-check, plus the smoothed spectrum and fit curves for the diagnostic
    figure.
    """
    sig = np.asarray(sig, float)
    bkg = np.asarray(bkg, float)
    sub = sig - scale * bkg
    var = np.clip(sig + scale ** 2 * bkg, 1.0, None)
    sigma = np.sqrt(var)
    binw = cen[1] - cen[0]
    kern = kernel(SMOOTH_MV / binw)
    fitter = fit_steps if kind == 'PSS' else fit_bump

    pos, curves, valley = fitter(cen, sub, sigma, kern, energies, prior)
    sm = smooth(sub, kern)

    boots = {i: [] for i, p in enumerate(pos) if np.isfinite(p)}
    if boots and n_boot:
        rng = np.random.default_rng(SEED)
        for _ in range(n_boot):
            rs = rng.poisson(np.clip(sig, 0, None)).astype(float)
            rb = rng.poisson(np.clip(bkg, 0, None)).astype(float)
            bp, _, _ = fitter(cen, rs - scale * rb, sigma, kern,
                              energies, prior)
            for i in boots:
                if i < len(bp) and np.isfinite(bp[i]):
                    boots[i].append(bp[i])

    edges = []
    for i, E in enumerate(energies):
        if i >= len(pos) or not np.isfinite(pos[i]):
            continue
        b = boots.get(i, [])
        err = max(float(np.std(b)) if len(b) > 10 else np.inf, binw / 2)
        edges.append(dict(kevee=E, edge_mv=round(float(pos[i]), 2),
                          edge_mv_err=round(err, 2),
                          edge_mv_halfheight=(
                              round(half_height(cen, sm, pos[i]), 2)
                              if half_height(cen, sm, pos[i]) else None),
                          n_boot_ok=len(b),
                          confidence='primary' if i == 0 else 'secondary'))
    # How much source there is at all, so "no edge" can be told apart from "no
    # signal" — the walls sit in FRONT of the plastic and the source is clamped
    # on the bar, so several channels legitimately see nothing to fit.
    above = cen > valley
    n_exc = float(sub[above].sum())
    n_bkg = float((scale * bkg)[above].sum())
    return dict(edges=edges, valley=round(valley, 2), sub=sub, sm=sm,
                curves=curves, n_excess=round(n_exc, 1),
                excess_over_bkg=(round(n_exc / n_bkg, 3) if n_bkg > 0 else None))
