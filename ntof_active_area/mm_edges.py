#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mm_edges.py -- the MX17 chamber active area, measured on n_TOF beam data.

Method
------
The July beam illuminates each chamber smoothly and well past its edges: the
source is the He-3 target 235 mm away plus the whole neutron flight path, so
nothing in the *illumination* changes over a few millimetres.  A physical edge
of the active area does: it is a step.  So we look for steps.

The observable is the position of a **paired track**: exactly one particle-like
cluster on each plane of the same chamber in the same event, with the two
planes' charges balanced.  The balance requirement matters -- an MX17 avalanche
splits ~50/50 between the two strip planes, so requiring it rejects the
uncorrelated per-plane noise that swamps a raw occupancy at the board edges.

For each plane and each end we fit, over a +-`WINDOW_MM` window,

    N(s) = b + (A + B (s - s0)) * Phi(+-(s - s0) / sigma)

i.e. a locally linear illumination times an error-function turn-on, on a flat
background.  `s0` is the 50 % point -- the same definition the June cosmic-bench
telescope measurement used (common/mx17_active_area.py), so the two numbers are
directly comparable.

Coordinates
-----------
`x` plane = u, the chamber's tangential coordinate;  `y` plane = v, along the
beam.  (ntof_tracking/run79_merge_prelim.track_frame.)  Both planes are 512
strips of 0.78 mm spanning 0 .. 398.58 mm.

Run this as
    .venv/bin/python -m ntof_active_area.mm_edges
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.beam_july_paths import BASE
from .clusters import (BENCH_ALIAS, CHAMBERS, ClusterCuts, N_STRIPS, PITCH_MM,
                       STRIP_MAX_MM, scan_subrun, subrun_files)

OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'
RUN = 'run_79'
SUB_RUNS = ('stat090_0000', 'stat090_0001')

# paired-track quality: the two planes share one avalanche
Q_RATIO = (0.6, 1.6)
MIN_WIDTH = 4                 # strips, on each plane

WINDOW_MM = 22.0              # half-window of the edge fit
GUESS = {'lo': 18.0, 'hi': 380.0}
MIN_PLATEAU = 25              # counts inside the window, above background


HOT_FACTOR = 5.0              # a strip this far above its neighbours is not a track
HOT_FLOOR = 20                # ...and only if it has this many counts at all


def _phi(z):
    from scipy.special import erf
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def hot_strip_mask(centroid_strip: np.ndarray) -> np.ndarray:
    """Strips whose paired-track centroid count spikes far above their
    neighbourhood.

    A real track population has centroids spread continuously across the strip
    pitch, so a single strip carrying several times its neighbours' counts is a
    coincidence artefact (one noisy channel on each plane firing together), not
    ionisation.  Returns a boolean 512-vector, True = drop.
    """
    n = np.bincount(np.clip(np.rint(centroid_strip).astype(int), 0, N_STRIPS - 1),
                    minlength=N_STRIPS).astype(float)
    hot = np.zeros(N_STRIPS, bool)
    half = 10
    for i in range(N_STRIPS):
        lo, hi = max(0, i - half), min(N_STRIPS, i + half + 1)
        nb = np.delete(n[lo:hi], i - lo)
        med = np.median(nb)
        if n[i] > HOT_FLOOR and n[i] > HOT_FACTOR * max(med, 1.0):
            hot[i] = True
    return hot


def fit_edge(pos_mm: np.ndarray, end: str, guess: float):
    """Error-function turn-on/off fit.  `end` is 'lo' or 'hi'.

    Returns a dict with the 50 % point, its fit error, the turn-on width and a
    `ok` flag.  `ok` is False when there is no step to fit -- either no plateau
    inside the window (the chamber is already dark there for another reason) or
    the fit did not converge.
    """
    from scipy.optimize import curve_fit

    sign = +1.0 if end == 'lo' else -1.0
    edges = np.arange(guess - WINDOW_MM, guess + WINDOW_MM + PITCH_MM, PITCH_MM)
    n, _ = np.histogram(pos_mm, bins=edges)
    s = 0.5 * (edges[1:] + edges[:-1])

    inside = (s - guess) * sign > 0
    plateau = float(np.median(n[inside]))
    outside = float(np.median(n[~inside]))
    bad = dict(ok=False, edge_mm=None, edge_err_mm=None, width_mm=None,
               plateau=plateau, background=outside, n_window=int(n.sum()))
    if plateau - outside < MIN_PLATEAU / len(n[inside]) or plateau <= outside:
        return bad

    def model(x, b, A, B, s0, sigma):
        return b + (A + B * (x - s0)) * _phi(sign * (x - s0) / max(sigma, 1e-3))

    p0 = [outside, plateau - outside, 0.0, guess, 1.5]
    sig = np.sqrt(np.maximum(n, 1.0))
    try:
        p, cov = curve_fit(model, s, n, p0=p0, sigma=sig, absolute_sigma=True,
                           maxfev=20000,
                           bounds=([0, 0, -np.inf, guess - WINDOW_MM, 0.2],
                                   [np.inf, np.inf, np.inf, guess + WINDOW_MM, 15.0]))
    except Exception:
        return bad
    err = float(np.sqrt(np.diag(cov))[3])
    resid = (n - model(s, *p)) / sig
    return dict(ok=True, edge_mm=float(p[3]), edge_err_mm=err,
                width_mm=float(p[4]), plateau=plateau, background=outside,
                n_window=int(n.sum()), chi2_dof=float((resid ** 2).sum() / (len(s) - 5)))


def select_pairs(p: np.ndarray) -> np.ndarray:
    if p.size == 0:
        return p
    r = p[:, 2] / np.where(p[:, 3] == 0, np.nan, p[:, 3])
    m = ((r > Q_RATIO[0]) & (r < Q_RATIO[1])
         & (p[:, 4] >= MIN_WIDTH) & (p[:, 5] >= MIN_WIDTH))
    return p[m]


LIVE_FRAC = 0.25              # of the local interior level
LIVE_REF = 30                 # strips inward used as that local reference


def span_profile(p: np.ndarray, plane: str) -> np.ndarray:
    """Per strip, how many selected tracks had that strip inside their cluster.

    Built from the cluster's first/last strip, so it counts *participation*,
    not centroids: a centroid can never reach the outermost live strip (the
    cluster is truncated there) but a span can.  This is the profile whose
    edges are the active-area edges.
    """
    if p.size == 0:
        return np.zeros(N_STRIPS)
    first = np.rint(p[:, 6] if plane == 'u' else p[:, 8]).astype(int)
    last = np.rint(p[:, 7] if plane == 'u' else p[:, 9]).astype(int)
    first = np.clip(first, 0, N_STRIPS - 1)
    last = np.clip(last, 0, N_STRIPS - 1)
    d = np.zeros(N_STRIPS + 1)
    np.add.at(d, first, 1.0)
    np.add.at(d, last + 1, -1.0)
    return np.cumsum(d)[:N_STRIPS]


# How many consecutive dark strips end the chamber.  It has to be wider than
# the widest *interior* dead band, or that band is mistaken for the edge:
# chamber C carries a ~20-strip dead stripe near u = 190 mm and D is worse.
PERSIST = 40

# Below this fraction of the plane's peak the profile is noise, not a chamber:
# an "edge" found there is wherever the noise dipped.  D's damaged planes hit
# this and are reported as undetermined rather than given a number.
NOISE_FRAC = 0.04


def live_edges(span: np.ndarray):
    """Outermost live strip at each end of a span profile.

    Walk **outward from the interior**, never inward: the reference level is
    always the median of the `LIVE_REF` strips just *inside* the strip under
    test, which are known live.  (Taking a symmetric local median instead lets
    the dead region set its own reference and the edge dissolves.)  A strip is
    dead when it and the next `PERSIST` strips outward all sit below
    `LIVE_FRAC` of that reference, so one dead channel in the middle of a live
    region is not mistaken for the end of the chamber.

    Returns (lo_strip, hi_strip, lo_step, hi_step).  Each `step` is the pair
    (inside, outside): the mean level of the 10 strips just inside the edge and
    the 10 just outside, both divided by the reference level further in.  A
    hard boundary reads (~1, ~0); a slow illumination fade reads (~0.4, ~0.2)
    and should not be quoted as an edge.
    """
    if span.sum() == 0:
        return None, None, None, None
    smooth = np.convolve(span, np.ones(9) / 9.0, mode='same')
    peak = int(np.argmax(smooth))

    def contrast(s, step, ref):
        def band(a, b):
            idx = [s + step * j for j in range(a, b)]
            idx = [t for t in idx if 0 <= t < N_STRIPS]
            return float(np.mean(span[idx])) if idx else 0.0
        return [round(band(-9, 1) / ref, 3), round(band(1, 11) / ref, 3)]

    peak_level = float(np.max(smooth))

    def walk(step):
        s = peak
        while 0 <= s + step < N_STRIPS:
            nxt = s + step
            ref = float(np.median(span[min(s, s - step * LIVE_REF):
                                       max(s, s - step * LIVE_REF) + 1]))
            # Once the reference itself has decayed into the noise floor there
            # is no contrast left to find an edge with; walking on would just
            # report wherever the noise happened to dip.  Say so instead.
            if ref <= 0 or ref < NOISE_FRAC * peak_level:
                return s, None
            tail = [span[t] for t in range(nxt, nxt + step * (PERSIST + 1), step)
                    if 0 <= t < N_STRIPS]
            if tail and max(tail) < LIVE_FRAC * ref:
                return s, contrast(s, step, ref)
            s = nxt
        return s, None

    lo, lo_c = walk(-1)
    hi, hi_c = walk(+1)
    return lo, hi, lo_c, hi_c


def dead_bands(span: np.ndarray, lo: int, hi: int, min_strips: int = 4):
    """Dead stretches *inside* [lo, hi] -- holes in the chamber, not its edges.

    These are as much a part of "what area is live" as the outline is, so the
    report quotes them rather than smoothing them away.
    """
    if lo is None or hi is None:
        return []
    seg = span[lo:hi + 1]
    # rolling local level, so the smooth illumination gradient across the
    # chamber is not read as a hole
    half = 30
    ref = np.array([np.percentile(seg[max(0, i - half):i + half + 1], 90)
                    for i in range(len(seg))])
    dark = seg < LIVE_FRAC * np.maximum(ref, 1.0)
    out, start = [], None
    for i, d in enumerate(dark):
        if d and start is None:
            start = i
        elif not d and start is not None:
            if i - start >= min_strips:
                out.append([int(lo + start), int(lo + i - 1)])
            start = None
    if start is not None and len(dark) - start >= min_strips:
        out.append([int(lo + start), int(hi)])
    return out


def connector_health(occ: dict) -> dict:
    """Per 64-strip detector connector, its median cluster occupancy relative
    to the plane's interior median.  A connector at ~0 is unplugged or dead --
    a readout fact about *this run*, not a property of the chamber, and the two
    have to be reported apart or the sim inherits a cabling accident.
    """
    out = {}
    for (ch, plane), o in occ.items():
        med = max(float(np.median(o[100:400])), 1.0)
        out[f'{ch}{plane}'] = [round(float(np.median(o[i * 64:(i + 1) * 64]) / med), 3)
                               for i in range(8)]
    return out


def measure(save: bool = True) -> dict:
    files = [f for s in SUB_RUNS for f in subrun_files(BASE, RUN, s)]
    if not files:
        raise SystemExit(f'no combined_hits under {BASE}/runs/{RUN}')
    cuts = ClusterCuts()
    occ, cent, pairs, n_events = scan_subrun(files, cuts)

    res = {'run': RUN, 'sub_runs': list(SUB_RUNS), 'n_files': len(files),
           'n_events': int(n_events),
           'cuts': dict(significance=cuts.sig, min_strips=cuts.min_strips,
                        max_gap=cuts.max_gap, q_ratio=list(Q_RATIO),
                        min_width=MIN_WIDTH),
           'strip_max_mm': STRIP_MAX_MM, 'pitch_mm': PITCH_MM,
           'connector_health': connector_health(occ),
           'chambers': {}}

    for ch in CHAMBERS:
        sel = select_pairs(pairs[ch])
        # drop coincidences sitting on a hot strip of either plane
        hot_u = hot_strip_mask(sel[:, 0]) if len(sel) else np.zeros(N_STRIPS, bool)
        hot_v = hot_strip_mask(sel[:, 1]) if len(sel) else np.zeros(N_STRIPS, bool)
        if len(sel):
            iu = np.clip(np.rint(sel[:, 0]).astype(int), 0, N_STRIPS - 1)
            iv = np.clip(np.rint(sel[:, 1]).astype(int), 0, N_STRIPS - 1)
            sel = sel[~(hot_u[iu] | hot_v[iv])]
        entry = {'bench_alias': BENCH_ALIAS[ch], 'n_pairs_raw': int(len(pairs[ch])),
                 'n_pairs': int(len(sel)),
                 'n_hot_strips': [int(hot_u.sum()), int(hot_v.sum())], 'planes': {}}
        for plane, col in (('u', 0), ('v', 1)):
            pos = sel[:, col] * PITCH_MM if len(sel) else np.zeros(0)
            span = span_profile(sel, plane)
            lo_s, hi_s, lo_sh, hi_sh = live_edges(span)
            entry['planes'][plane] = {
                'strip_plane': 'x' if plane == 'u' else 'y',
                'live_lo_strip': None if lo_s is None else int(lo_s),
                'live_hi_strip': None if hi_s is None else int(hi_s),
                'live_lo_mm': None if lo_s is None else lo_s * PITCH_MM,
                'live_hi_mm': None if hi_s is None else hi_s * PITCH_MM,
                'lo_sharpness_strips': lo_sh,
                'hi_sharpness_strips': hi_sh,
                # an end is only "determined" if the walk actually found a step;
                # None means it ran out of contrast (D) or ran off the board (B)
                # An end is "determined" when the walk found a real step.  It
                # can also stop for two other reasons, which mean different
                # things and must not be merged: it reached the last strip on
                # the board (the plane is live all the way out -- still an
                # answer), or the profile decayed into noise first (no answer).
                'lo_determined': lo_sh is not None,
                'hi_determined': hi_sh is not None,
                'lo_at_board_end': lo_s == 0,
                'hi_at_board_end': hi_s == N_STRIPS - 1,
                'dead_bands_mm': [[a * PITCH_MM, b * PITCH_MM]
                                  for a, b in dead_bands(span, lo_s, hi_s)],
                'span_profile': span.astype(int).tolist(),
                'lo': fit_edge(pos, 'lo', GUESS['lo']) if len(pos) else {'ok': False},
                'hi': fit_edge(pos, 'hi', GUESS['hi']) if len(pos) else {'ok': False},
                'n': int(len(pos))}
        res['chambers'][ch] = entry

    if save:
        np.savez_compressed(OUT / 'profiles.npz',
                            **{f'occ_{k[0]}{k[1]}': occ[k] for k in occ},
                            **{f'cent_{k[0]}{k[1]}': cent[k] for k in cent},
                            **{f'pairs_{c}': pairs[c] for c in CHAMBERS})
        (OUT / 'results_mm.json').write_text(json.dumps(res, indent=1))
    return res


def _print(res):
    print(f"{res['run']} {res['n_files']} files, {res['n_events']} events")
    for ch, e in res['chambers'].items():
        print(f"\n== chamber {ch} ({e['bench_alias']})  {e['n_pairs']} selected pairs")
        for plane, pe in e['planes'].items():
            if pe['live_lo_strip'] is None:
                print(f"  {plane} ({pe['strip_plane']} plane): no tracks")
                continue
            span = (pe['live_hi_mm'] - pe['live_lo_mm']) / 10.0
            notes = [f'{e} ' + ('live to board end' if pe[f'{e}_at_board_end']
                                 else 'undetermined')
                     for e in ('lo', 'hi') if not pe[f'{e}_determined']]
            flag = '  [' + '; '.join(notes) + ']' if notes else ''
            print(f"  {plane} ({pe['strip_plane']} plane): live strips "
                  f"{pe['live_lo_strip']}..{pe['live_hi_strip']}  = "
                  f"{pe['live_lo_mm']:6.2f} .. {pe['live_hi_mm']:6.2f} mm "
                  f"({span:.2f} cm)  step lo={pe['lo_sharpness_strips']} "
                  f"hi={pe['hi_sharpness_strips']}  n={pe['n']}{flag}")
            if pe['dead_bands_mm']:
                print('        dead inside: '
                      + ', '.join(f'{a:.0f}-{b:.0f}' for a, b in pe['dead_bands_mm']))


if __name__ == '__main__':
    _print(measure())
