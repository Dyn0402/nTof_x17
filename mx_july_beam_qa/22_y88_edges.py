"""22_y88_edges.py — Compton-edge extraction from the Y-88 source spectra (T2).

Input: the fine linear-mV histograms cached by 21_y88_spectra.py. Output: per
source-arm channel, the Compton-edge position(s) in mV with bootstrap errors,
plus a per-run diagnostic figure so each extraction can be eyeballed.

Convention (stated, per handoff §4.2 — a resolution-smeared Compton edge has no
unique position): the PRIMARY estimator is the steepest-descent point, i.e. the
amplitude of the minimum of the smoothed derivative dN/dA on the high side of
the continuum. A HALF-HEIGHT estimator (amplitude where the smoothed spectrum
falls to half of the local pre-edge plateau) is reported as a cross-check; the
two differ by O(sigma/2) and that offset is the convention systematic to carry
if a simulation-anchored edge is wanted later.

Shapes: on the SiPM walls the edge is a bump (the falling side is the edge); on
the plastics it is a shoulder on a falling Compton continuum, and BOTH Y-88
edges (699 & 1612 keVee) are usually visible — expect their mV ratio ~ 2.31.

Errors: Poisson-bootstrap the fine linear bins (fixed seed, 200 resamples),
re-smooth and re-extract; the estimator is continuous under smoothing so the
error is floored at half a fine bin (0.1 mV).

Outputs:
  calib/y88_edges_<run>.json          per channel: edges (mV) + errors + method
  figures/21_y88/edges_<run>.png      diagnostic grid (source arm)
Usage: python 22_y88_edges.py [run_stem ...]   (default: all four)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import erfc

BASE = Path(__file__).parent
CACHE = BASE / 'cache'
OUT = BASE / 'figures' / '21_y88'
CALIB = BASE / 'calib'
OUT.mkdir(parents=True, exist_ok=True)
CALIB.mkdir(exist_ok=True)

RUNS = ['run224476', 'run224477', 'run224478', 'run224479']
NCH = {'PSS': 2, 'WAL': 8, 'LIQ': 1}

SMOOTH_MV = 1.0          # Gaussian smoothing sigma (mV), for seeding only
VALLEY_HI = 16.0         # noise turn-on / valley sits below this (mV)
EDGE_MIN_MV = 8.0        # never call an edge below this (noise region)
N_BOOT = 200
SEED = 224488
E_RATIO = 1612.06 / 698.63   # = 2.307, the two Compton-edge keVee ratio
SQRT2 = np.sqrt(2.0)


def _kernel(sigma_bins):
    half = int(np.ceil(4 * sigma_bins))
    x = np.arange(-half, half + 1)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    return k / k.sum()


# --- physics models --------------------------------------------------------
# Plastic (two gammas): the Compton continuum steps DOWN at each edge, so the
# spectrum above the noise valley is a sum of two resolution-smeared steps with
# the edge ratio FIXED to 2.307 (both edges from one source) — a strong prior
# that makes the two-edge fit robust. Resolution sigma(E) = k*sqrt(E).
def two_step(A, bg, s1, s2, E1, k):
    E2 = E_RATIO * E1
    sig1 = k * np.sqrt(max(E1, 1e-6))
    sig2 = k * np.sqrt(max(E2, 1e-6))
    return (bg + s1 * 0.5 * erfc((A - E1) / (SQRT2 * sig1))
            + s2 * 0.5 * erfc((A - E2) / (SQRT2 * sig2)))


# Wall (SiPM bars): the Compton edge appears as a localized bump on the falling
# low-amplitude background — Gaussian peak + linear background; the peak center
# is the edge landmark.
def gauss_lin(A, a, mu, sig, m, b):
    return a * np.exp(-0.5 * ((A - mu) / sig) ** 2) + m * A + b


def _fit(model, A, y, p0, bounds):
    try:
        sig = np.sqrt(np.clip(y, 1, None))
        popt, _ = curve_fit(model, A, y, p0=p0, sigma=sig, bounds=bounds,
                            maxfev=20000)
        return popt
    except Exception:
        return None


def _valley(cen, sm):
    lowmask = cen < VALLEY_HI
    v = cen[lowmask][np.argmin(sm[lowmask])] if lowmask.any() else EDGE_MIN_MV
    return max(v, EDGE_MIN_MV)


def single_step(A, bg, s, E, k):
    """One resolution-smeared step (Compton edge): background + amplitude s that
    drops through the edge at E with resolution sigma |k|."""
    return bg + s * 0.5 * erfc((A - E) / (SQRT2 * abs(k)))


def _fit_plastic(cen, counts, sm, valley):
    """Two INDEPENDENT single-erfc-step fits, one per Y-88 Compton edge, with NO
    ratio assumption. The 698.63 keVee edge is the dominant step (stronger
    898 keV gamma) and is fit first over the whole continuum above the noise
    turn-on; the weaker 1612 keVee edge is fit in the upper window above it. The
    measured ratio E1612/E699 (returned) is then a genuine cross-check of the
    energy assignment and the linearity, not an input. Returns
    (E699, E1612, (p699, p1612), (fr699, fr1612), ratio) or Nones."""
    # 699 (dominant) step: fit over [turn-on end, 48 mV]. Capping the window
    # below the 1612 roll-off (52-71 mV) and the resolution at sigma<=15 stops a
    # single wide erfc from stretching across the whole continuum and railing at
    # the low bound (the PSSC1 failure mode).
    lo1 = max(valley + 3.0, 12.0)
    hi1 = 48.0
    m1 = (cen >= lo1) & (cen <= hi1)
    A1, y1 = cen[m1], counts[m1]
    if len(A1) < 6 or y1.max() <= 0:
        return None
    bg1 = max(sm[np.argmin(np.abs(cen - hi1))], 1.0)
    p699 = _fit(single_step, A1, y1,
                [bg1, y1.max(), 26.0, 7.0],
                ([bg1 * 0.2, 0, lo1 + 1, 1.0],
                 [y1.max() * 2, y1.max() * 4, hi1 - 2, 15.0]))
    if p699 is None:
        return None
    E699 = p699[2]
    # weaker outer (1612) edge, in the window above the 699 edge
    lo2 = 1.4 * E699
    m2 = (cen >= lo2) & (cen <= cen[-1])
    A2, y2 = cen[m2], counts[m2]
    p1612 = _fit(single_step, A2, y2,
                 [max(sm[cen > 100].min(), 1.0), y2.max(), 2.3 * E699, 12.0],
                 ([0, 0, 1.3 * E699, 2.0],
                  [y2.max() * 2, y2.max() * 4, 3.3 * E699, 40.0])) \
        if len(A2) >= 6 else None
    if p1612 is None:
        return E699, np.nan, (p699, None), ((A1[0], A1[-1]), None), None
    E1612 = p1612[2]
    return (E699, E1612, (p699, p1612),
            ((A1[0], A1[-1]), (A2[0], A2[-1])),
            round(float(E1612 / E699), 2))


def _fit_wall(cen, counts, sm, valley):
    """Gaussian-bump fit on the falling background. Returns (mu, popt, fitrange)
    or Nones if no significant bump."""
    seek = (cen > valley + 1) & (cen < 70)
    if seek.sum() < 5:
        return None
    # bump = the most prominent local maximum of the log-smoothed spectrum in
    # the window (log space so the steep low-amplitude falloff doesn't swamp a
    # real bump sitting on it, e.g. WALx3-type marginal edges)
    idx = np.where(seek)[0]
    logsm = np.log(np.clip(sm[idx], 1.0, None))
    peaks, props = find_peaks(logsm, prominence=0.15)
    if len(peaks) == 0:
        return None
    mu0 = cen[idx][peaks[np.argmax(props['prominences'])]]
    if mu0 < EDGE_MIN_MV + 2:
        return None
    w = max(6.0, 0.35 * mu0)
    fr = (cen >= mu0 - w) & (cen <= mu0 + w)
    A, y = cen[fr], counts[fr]
    if len(A) < 6:
        return None
    bg0 = float(np.interp(mu0, cen[idx][[0, -1]], sm[idx][[0, -1]]))
    a0 = max(sm[np.argmin(np.abs(cen - mu0))] - bg0, 1.0)
    p0 = [a0, mu0, w / 2.5, 0.0, bg0]
    bounds = ([0, mu0 - w, 1.0, -np.inf, 0],
              [counts.max() * 2, mu0 + w, w, np.inf, counts.max() * 2])
    popt = _fit(gauss_lin, A, y, p0, bounds)
    if popt is None:
        return None
    a, mu, sg = popt[0], popt[1], abs(popt[2])
    # require the bump to stand ~3 sigma_stat above its local background
    bg_here = popt[3] * mu + popt[4]
    if a < 3 * np.sqrt(max(bg_here, 1)) or not (mu0 - w < mu < mu0 + w):
        return None
    return mu, popt, (A[0], A[-1])


def extract_channel(cen, counts, kern, kind):
    """Fit the Compton edge(s). Returns (edges, valley, sm, fitcurves, ratio)
    where edges is a list of dicts (ascending mV) and fitcurves is a list of
    (x, y) model segments for the diagnostic plot."""
    sm = np.convolve(counts, kern, mode='same')
    valley = _valley(cen, sm)
    binw = cen[1] - cen[0]

    # LIQ (liquid scintillator) shows the same localized Compton bump as the
    # SiPM walls, so it uses the same Gaussian-bump extractor.
    def edges_from(cnts):
        if kind == 'PSS':
            r = _fit_plastic(cen, cnts, np.convolve(cnts, kern, 'same'), valley)
            return (None if r is None else (r[0], r[1]), r)
        r = _fit_wall(cen, cnts, np.convolve(cnts, kern, 'same'), valley)
        return (None if r is None else (r[0],), r)

    vals, r = edges_from(counts)
    if vals is None:
        return [], float(valley), sm, [], None
    idx_valid = [i for i, v in enumerate(vals) if np.isfinite(v)]

    # bootstrap each measured edge
    rng = np.random.default_rng(SEED)
    boots = {i: [] for i in idx_valid}
    for _ in range(N_BOOT):
        rc = rng.poisson(np.clip(counts, 0, None)).astype(float)
        bv, _ = edges_from(rc)
        if bv is None:
            continue
        for i in idx_valid:
            if i < len(bv) and np.isfinite(bv[i]):
                boots[i].append(bv[i])

    kevee = [698.63, 1612.06] if kind == 'PSS' else [698.63]
    conf = ['primary', 'secondary'] if kind == 'PSS' else ['primary']
    edges = []
    for i in idx_valid:
        err = max(np.std(boots[i]) if len(boots[i]) > 10 else np.inf, binw / 2)
        edges.append(dict(edge_mv=round(float(vals[i]), 2),
                          edge_mv_err=round(float(err), 2),
                          kevee=kevee[i], confidence=conf[i]))
    free_ratio = r[4] if kind == 'PSS' else None
    # fitted curve(s) for the plot
    fitcurves = []
    if kind == 'PSS':
        for popt, fr in zip(r[2], r[3]):
            if popt is not None:
                xx = np.linspace(fr[0], fr[1], 300)
                fitcurves.append((xx, single_step(xx, *popt)))
        return edges, float(valley), sm, fitcurves, free_ratio
    else:
        popt, (lo, hi) = r[1], r[2]
        xx = np.linspace(lo, hi, 300)
        fitcurve = (xx, gauss_lin(xx, *popt))
    return edges, float(valley), sm, [fitcurve], free_ratio


def process_run(run_stem):
    z = np.load(CACHE / f'21_y88_{run_stem}.npz')
    arm = str(z['source_arm'])
    le = z['lin_edges']
    cen = 0.5 * (le[:-1] + le[1:])
    kern = _kernel(SMOOTH_MV / (cen[1] - cen[0]))

    result = {'run': run_stem, 'source_arm': arm,
              'convention': 'PSS: two INDEPENDENT single-erfc-step fits (no ratio '
                            'assumption) — 698.63 keVee is the dominant step, '
                            '1612.06 keVee the weaker one above it; edge_mv = '
                            'smeared-step center. edge_ratio_1612_over_699 is a '
                            'measured cross-check (expect 2.307). WAL/LIQ: '
                            'Gaussian-bump + linear-bg fit; edge_mv = peak center '
                            '(698.63 keVee edge). '
                            f'{N_BOOT} Poisson bootstraps (seed {SEED}); '
                            'err floored at half a fine bin (0.1 mV).',
              'hv_note': 'HV during 224476-79 not recorded in DAQsettings — '
                         'plastic HV was RAISED (edges ~20x nominal MIP scale); '
                         'confirm exact values with Dylan before transporting '
                         'to nominal (T3.2).',
              'channels': {}}

    panels = [('PSS', arm, c) for c in range(NCH['PSS'])] + \
             [('LIQ', arm, 0)] + \
             [('WAL', arm, c) for c in range(NCH['WAL'])]
    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    axf = axes.flat
    for k, (kind, a, c) in enumerate(panels):
        tree = f'{kind}{a}'
        counts = z[f'{tree}_lin'][c]
        edges, valley, sm, fitcurves, free_ratio = extract_channel(
            cen, counts, kern, kind)
        ch = f'{tree}{c + 1}'
        result['channels'][ch] = dict(kind=kind, edges=edges,
                                       edge_ratio_1612_over_699=free_ratio,
                                       n_hits=int(counts.sum()))
        ax = axf[k]
        ax.step(cen, counts, where='mid', color='0.7', lw=0.8)
        ax.plot(cen, sm, color='steelblue', lw=1.0, alpha=0.7)
        for fc in fitcurves:
            ax.plot(fc[0], fc[1], color='k', lw=1.4)
        for e in edges:
            col = 'crimson' if e['kevee'] > 1000 else 'darkorange'
            ax.axvline(e['edge_mv'], color=col, lw=1.4)
            ax.axvspan(e['edge_mv'] - e['edge_mv_err'],
                       e['edge_mv'] + e['edge_mv_err'], color=col, alpha=0.25)
            ax.text(e['edge_mv'], 1.5, f" {e['edge_mv']:.1f}", color=col,
                    fontsize=8, rotation=90, va='bottom')
        ax.axvline(valley, color='green', lw=0.7, ls=':')
        ax.set_yscale('log')
        ax.set_xlim(0, 110)
        ax.set_ylim(bottom=1)
        rlabel = ''
        if kind == 'PSS' and len(edges) == 2:
            fr = f'{free_ratio:.2f}' if free_ratio else 'na'
            rlabel = f'  (699 orange + 1612 red; ratio={fr}, exp 2.31)'
        ax.set_title(f'{ch}  n={counts.sum():,.0f}{rlabel}', fontsize=9)
        ax.tick_params(labelsize=7)
        if k >= 8:
            ax.set_xlabel('amplitude [mV]', fontsize=8)
    for k in range(len(panels), len(axf)):
        axf[k].axis('off')
    fig.suptitle(f'{run_stem}: Y-88 Compton-edge extraction, source arm {arm} '
                 '(gray=raw, blue=smoothed, orange=699 keVee, red=1612 keVee, '
                 'green=valley)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = OUT / f'edges_{run_stem}.png'
    fig.savefig(p, dpi=140)
    plt.close(fig)

    (CALIB / f'y88_edges_{run_stem}.json').write_text(json.dumps(result, indent=2))
    print(f'{run_stem} (arm {arm}): '
          + ' '.join(f'{ch}={[e["edge_mv"] for e in v["edges"]]}'
                     for ch, v in result['channels'].items() if v['kind'] == 'PSS'))
    print(f'  -> {p}  &  calib/y88_edges_{run_stem}.json')
    return result


def main():
    for s in (sys.argv[1:] or RUNS):
        process_run(s)


if __name__ == '__main__':
    main()
