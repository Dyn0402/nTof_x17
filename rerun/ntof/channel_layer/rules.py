"""rules.py -- the channel-layer rules on one view (one FEU) of an n_TOF chamber, from hits.

Every rule here is an existing pass-1 rule, re-expressed on one view with that view's channel -> position map
(``pos``: (512,) mm in the chamber's strip-map frame, the same on bench and n_TOF: cabling.py).  Nothing here
reconstructs a position, angle or depth: hits are used for QA only (RECONSTRUCTION_BASIS.md).

  dead / weak / silent   rerun/model_audit/ws_dead_charge/flank_census.py (imported, unchanged): a strip whose two
                         +-1 (or +-2) position neighbours both read > 300 ADC must fire (> 50 ADC); dead f < 0.25,
                         weak < 0.6; silent = occupancy < 5 % of the +-32-strip neighbourhood median (a whole dead
                         connector has no live flank).  The accumulation is split here (flank_accumulate /
                         flank_decide) so acquisitions can be summed over an era before deciding.
  shorts                 rerun/seedfit/channel_census.py: amplitude cosine over events with 2-40 good hits
                         (significance > 5, unsaturated) > 0.7 with a strip >= 2 pitches away; a correlated group that
                         fills >= 70 % of its span is a band, not a short.
  hot (seed probe)       rerun/model_audit/sps_build_cache.py hot_in_blocks / hot_channels: a channel that seeds
                         (is the brightest hit of the view) in > 3 % of a 1000-event block and > 10x the median of its
                         LIVE +-4 position neighbours, in any block.
  near oscillators       sps_build_cache.near_oscillators: within 2 ranks of a known oscillator, ON (> 300 ADC) in
                         >= 1 % and >= 20 events at >= 5x its live neighbours, not a copy (<= 50 % shared ON events).
                         At SPS the known oscillators were det4 Y 372/510; at n_TOF there is no prior list, so the known
                         oscillators are this probe's hot channels.
  coupled channels       sps_build_cache.coupled_channels: for an oscillator that oscillates (ON rate > 10x its live
                         neighbours'), a channel within +-4 ranks lit (> 100 or > 300 ADC) in >= 10 ON events with an
                         ON/OFF lit-rate ratio > 10x that of the strips away from every oscillator.

The constants are the pass-1 values; they are imported where the source module can be imported without side effects
(flank_census) and copied (with the source named) where it cannot (sps_build_cache loads the det4 H4 map at import).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'rerun', 'model_audit', 'ws_dead_charge'))
import flank_census as FC  # noqa: E402

PITCH = FC.PITCH
# flank_census: FIRE 50, SIG 5, BIG 300, DEAD_REL 0.25, WEAK_REL 0.6, MIN_FLANK 20, SILENT_REL 0.05
FLANK_MAX_HITS = 60          # rerun/model_audit/ws_dead_charge/flank_ntof.py: drop flash / shower events on the FEU
# rerun/seedfit/channel_census.py
SHORT_CORR, BAND_FILL, SHORT_MIN_HITS, SHORT_MAX_HITS = 0.7, 0.7, 2, 40
# rerun/model_audit/sps_build_cache.py (values of code 48244778, the final SPS rule)
HOT_BLOCK, HOT_FRAC, HOT_RATIO, HOT_NB = 1000, 0.03, 10.0, 4
OSC_NEAR, OSC_MIN, OSC_RATIO, OSC_COPY, OSC_MIN_EVENTS = 2, 0.01, 5.0, 0.5, 20
COUPLE_ON_ADC, COUPLE_LIT_ADC, COUPLE_NB, COUPLE_MIN, COUPLE_RATIO, COUPLE_OSC_RATIO = 300.0, (100.0, 300.0), 4, 10, 10.0, 10.0
#: the n_TOF beam seeder ignores a plane with > 150 hits (busy, R4 of the plan); its brightest strip is not a seed
SEED_BUSY = 150
#: a channel is called silent only where silence is significant: the expected count under the reference occupancy
#: must be >= this (flank_census has no floor; it ran on 20-file samples.  A 100-event tail acquisition otherwise reads
#: half of every connector as silent)
SILENT_MIN_EXPECTED = 20.0
#: acquisitions shorter than this do not vote in the era's hot persistence (one block of < 1000 events)
HOT_MIN_EVENTS = 1000


def order(pos):
    ok = np.flatnonzero(np.isfinite(pos))
    return ok[np.argsort(pos[ok])]


def live_neighbours(c, o, rank, nb=4, skip=None):
    """channels within +-nb position ranks of c (o: channels in position order, rank: channel -> index), not c,
    not in skip (bool[512])."""
    i = rank[c]
    out = np.r_[o[max(0, i - nb):i], o[i + 1:i + nb + 1]]
    return [int(x) for x in out if skip is None or not skip[x]]


# ----------------------------------------------------------------------------------------------- dense event rows
def dense(ev, ch, amp):
    """(A [n_events, 512] float32 per-event max amplitude, nhits [n_events], event ids)"""
    ue, inv = np.unique(ev, return_inverse=True)
    A = np.zeros((len(ue), 512), np.float32)
    np.maximum.at(A, (inv, ch), amp)
    nh = np.bincount(inv, minlength=len(ue))
    return A, nh, ue


# ----------------------------------------------------------------------------------------------- flank / silent
def flank_accumulate(A, good, pos, nh):
    """summable flank counts: occupancy over all events, n1/f1/n2/f2 over events with <= FLANK_MAX_HITS hits.
    A: per-event max amplitude of GOOD hits (flank_census keeps sig > 5 or amp > 300)."""
    fired = good > FC.FIRE
    out = dict(n_events=len(A), occ=fired.sum(0).astype(np.int64))
    k = nh <= FLANK_MAX_HITS
    G, F = good[k], fired[k]
    out['n_flank_events'] = int(k.sum())
    for kk in (1, 2):
        L, R = FC.neighbours(pos, kk)
        cs = np.flatnonzero((L >= 0) & (R >= 0))
        both = (G[:, L[cs]] > FC.BIG) & (G[:, R[cs]] > FC.BIG)
        n = np.zeros(512, np.int64); f = np.zeros(512, np.int64)
        n[cs] = both.sum(0); f[cs] = (both & F[:, cs]).sum(0)
        out[f'n{kk}'], out[f'f{kk}'] = n, f
    return out


def flank_decide(acc, pos):
    """flank_census.flank's decision on (summed) accumulators: dead / weak / silent / judged, f, occupancy."""
    n1, f1, n2, f2 = (np.asarray(acc[k], float) for k in ('n1', 'f1', 'n2', 'f2'))
    j1, j2 = n1 >= FC.MIN_FLANK, n2 >= FC.MIN_FLANK
    f = np.where(j1, f1 / np.maximum(n1, 1), np.where(j2, f2 / np.maximum(n2, 1), np.nan))
    judged = j1 | j2
    with np.errstate(invalid='ignore'):
        dead = judged & (f < FC.DEAD_REL)
        weak = judged & (f >= FC.DEAD_REL) & (f < FC.WEAK_REL)
    occ_r = np.asarray(acc['occ'], float) / max(acc['n_events'], 1)
    o = order(pos)
    ref = np.full(512, np.nan)
    for i, c in enumerate(o):
        ref[c] = np.median(occ_r[o[max(0, i - 32):i + 33]])
    vref = np.median(occ_r[o]) if len(o) else np.nan
    with np.errstate(invalid='ignore'):
        rr = np.maximum(ref, 0.5 * vref)
        silent = np.isfinite(pos) & (occ_r < FC.SILENT_REL * rr) & (rr * acc['n_events'] >= SILENT_MIN_EXPECTED)
    return dict(dead=dead, weak=weak, silent=silent, judged=judged, f=f, occ=occ_r)


# ----------------------------------------------------------------------------------------------- shorts
def short_accumulate(ev, ch, amp, sig, sat):
    """summable (S [512,512], N [512]) amplitude-product sums over events with 2-40 good unsaturated hits."""
    g = (sig > FC.SIG) & ~sat
    ev, ch, amp = ev[g], ch[g], amp[g].astype(np.float64)
    ue, inv, cnt = np.unique(ev, return_inverse=True, return_counts=True)
    keep = (cnt[inv] >= SHORT_MIN_HITS) & (cnt[inv] <= SHORT_MAX_HITS)
    inv, ch, amp = inv[keep], ch[keep], amp[keep]
    from scipy.sparse import csr_matrix
    M = csr_matrix((amp, (inv, ch)), shape=(len(ue), 512))
    S = (M.T @ M).toarray()
    N = np.asarray(M.multiply(M).sum(0)).ravel()
    return S, N


def short_decide(S, N, pos):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    C = S / np.sqrt(np.maximum(np.outer(N, N), 1e-9)); np.fill_diagonal(C, 0)
    P = np.where(np.isfinite(pos), pos, 1e9)
    A = (C > SHORT_CORR) & (np.abs(P[:, None] - P[None, :]) > 1.5 * PITCH)
    ng, lab = connected_components(csr_matrix(A), directed=False)
    groups = sorted([np.flatnonzero(lab == g).tolist() for g in range(ng) if (lab == g).sum() > 1], key=len, reverse=True)
    bands = [g for g in groups if len(g) >= 4 and len(g) >= BAND_FILL * (g[-1] - g[0] + 1)]
    shorts = np.zeros(512, bool)
    for g in groups:
        if g not in bands:
            shorts[g] = True
    return dict(shorts=shorts, groups=[g for g in groups if g not in bands], bands=bands)


# ----------------------------------------------------------------------------------------------- hot seed probe
def seeds(A, nh):
    """brightest channel per event (-1: no hit, or a busy plane the seeder ignores)."""
    s = np.argmax(A, axis=1)
    s[(A.max(axis=1) <= 0) | (nh > SEED_BUSY)] = -1
    return s


def hot_channels(cnt, n, pos, skip):
    o = order(pos); rank = {int(c): i for i, c in enumerate(o)}
    out = np.zeros(512, bool)
    for c in np.flatnonzero(cnt > HOT_FRAC * n):
        if not np.isfinite(pos[c]):
            continue
        nb = live_neighbours(int(c), o, rank, HOT_NB, skip)
        out[c] = cnt[c] > HOT_RATIO * max(np.median(cnt[nb]) if nb else 0.0, 1.0)
    return out


def block_seed_counts(s):
    """[n_blocks, 512] seed counts in consecutive HOT_BLOCK-event blocks (a tail < half a block joins the last)."""
    n = len(s); edges = list(range(0, n, HOT_BLOCK)) + [n]
    if len(edges) > 2 and edges[-1] - edges[-2] < HOT_BLOCK // 2:
        edges.pop(-2)
    out = np.zeros((len(edges) - 1, 512), np.int32); sizes = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        ss = s[lo:hi]; ss = ss[ss >= 0]
        out[i] = np.bincount(ss, minlength=512); sizes.append(hi - lo)
    return out, np.array(sizes)


def hot_in_blocks(bcnt, sizes, pos, skip):
    out = np.zeros(512, bool)
    for cnt, n in zip(bcnt, sizes):
        out |= hot_channels(cnt.astype(float), n, pos, skip)
    return out


def near_oscillators(A, known, pos, skip):
    """SPS near-oscillator rule around ``known`` (bool[512]) oscillators."""
    n = len(A); out = np.zeros(512, bool)
    if n == 0 or not known.any():
        return out
    o = order(pos); rank = {int(c): i for i, c in enumerate(o)}
    on = A > COUPLE_ON_ADC; rate = on.mean(0)
    acc = [int(c) for c in np.flatnonzero(known)]
    cand = set()
    for b in acc:
        cand |= {c for c in live_neighbours(b, o, rank, OSC_NEAR, skip) if not known[c]}
    scored = []
    for c in cand:
        nb = [x for x in live_neighbours(c, o, rank, HOT_NB, skip) if x not in acc]
        ratio = rate[c] / max(np.median(rate[nb]) if nb else 0.0, 1.0 / n)
        if rate[c] >= OSC_MIN and int(on[:, c].sum()) >= OSC_MIN_EVENTS and ratio >= OSC_RATIO:
            scored.append((ratio, c))
    for ratio, c in sorted(scored, reverse=True):
        mine = on[:, c]
        if any((mine & on[:, a]).sum() > OSC_COPY * mine.sum() for a in acc):
            continue
        out[c] = True; acc.append(c)
    return out


def coupled_channels(A, osc, pos, skip):
    """{oscillator: [coupled channels]} (SPS rule; ``osc`` bool[512]), and {oscillator: ON fraction}."""
    o = order(pos); rank = {int(c): i for i, c in enumerate(o)}
    oscs = [int(c) for c in np.flatnonzero(osc)]
    out, onfrac = {}, {}
    if not oscs or len(A) == 0:
        return out, onfrac
    lits = [A > t for t in COUPLE_LIT_ADC]
    near = set()
    for c in oscs:
        i = rank[c]; near |= {int(x) for x in o[max(0, i - COUPLE_NB - 4):i + COUPLE_NB + 5]}
    far = np.array([c for c in o if int(c) not in near], int)
    for c in oscs:
        on = A[:, c] > COUPLE_ON_ADC; n1, n0 = int(on.sum()), int((~on).sum())
        onfrac[c] = round(n1 / len(A), 4)
        if n1 < COUPLE_MIN or n0 < COUPLE_MIN:
            continue
        i = rank[c]
        nbs = [int(x) for x in o[max(0, i - COUPLE_NB):i + COUPLE_NB + 1] if int(x) not in oscs]
        live = [x for x in live_neighbours(c, o, rank, COUPLE_NB, skip) if x not in oscs]
        nb_rate = np.median((A[:, live] > COUPLE_ON_ADC).mean(0)) if live else 0.0
        if on.mean() <= COUPLE_OSC_RATIO * max(nb_rate, 1.0 / len(A)):
            continue
        got = []
        for lit in lits:
            f1, f0 = lit[on][:, far].mean(), lit[~on][:, far].mean()
            base = (f1 / f0) if f0 > 0 else (np.inf if f1 > 0 else 1.0)
            for x in nbs:
                k1, k0 = int(lit[on, x].sum()), int(lit[~on, x].sum())
                r = (k1 / n1) / (max(k0, 1) / n0)
                if x not in got and k1 >= COUPLE_MIN and r > COUPLE_RATIO * max(base, 1.0):
                    got.append(x)
        if got:
            out[c] = sorted(got)
    return out, onfrac
