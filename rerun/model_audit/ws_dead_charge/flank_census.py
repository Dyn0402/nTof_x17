#!/usr/bin/env python3
"""flank_census.py -- a reference-free dead-strip test that runs identically on every dataset (bench, SPS, n_TOF).

A strip whose two POSITION neighbours both read a big pulse in the same event should fire too: charge spreads over
2-3 strips (98 % within +-2, det4-charge-sharing-xy).  Per channel:
    flank1  events where both +-1 neighbours read > BIG ADC;  f1 = fraction of those in which the channel fired (> FIRE)
    flank2  same with the +-2 neighbours (judges the middle of a 2-3 strip dead run, whose +-1 neighbours are dead)
dead: f < DEAD_REL (0.25) on the flank with >= MIN_FLANK events (flank1 first, flank2 if flank1 is unjudged);
weak: f < WEAK_REL (0.6).  A dead amplification stripe (det4) is not found this way (its neighbours do not fire
either): that is an amplification property common to both views and to every readout, not a channel fault.
A short does not read silent (it carries its group's charge), so shorts need channel_census.py's correlation test.

Inputs: hits (eventId, channel, amplitude, significance) per FEU and the channel -> position map of that dataset.
    from flank_census import flank
    out = flank(ev, ch, amp, pos)        # pos: (512,) mm, nan = unmapped
"""
import numpy as np

FIRE, SIG, BIG, DEAD_REL, WEAK_REL, MIN_FLANK, SILENT_REL = 50.0, 5.0, 300.0, 0.25, 0.6, 20, 0.05
PITCH = 0.78


def neighbours(pos, k):
    """(left, right) channel k strips away in position (-1 = none), only if the position step is k pitches."""
    L = np.full(512, -1); R = np.full(512, -1)
    ok = np.flatnonzero(np.isfinite(pos))
    o = ok[np.argsort(pos[ok])]
    ps = pos[o]
    for i, c in enumerate(o):
        if i - k >= 0 and abs(ps[i] - ps[i - k] - k * PITCH) < 0.3 * PITCH:
            L[c] = o[i - k]
        if i + k < len(o) and abs(ps[i + k] - ps[i] - k * PITCH) < 0.3 * PITCH:
            R[c] = o[i + k]
    return L, R


def flank(ev, ch, amp, pos, sig=None):
    """dict with per-channel flank counts / fractions and the dead / weak lists (FEU channel numbers)."""
    ev = np.asarray(ev); ch = np.asarray(ch, int); amp = np.asarray(amp, float)
    keep = np.isfinite(amp)
    if sig is not None:
        keep &= (np.asarray(sig) > SIG) | (amp > BIG)
    ev, ch, amp = ev[keep], ch[keep], amp[keep]
    # dense per-event amplitude rows, events relabelled 0..n-1
    ue, inv = np.unique(ev, return_inverse=True)
    n = len(ue)
    out = dict(n_events=int(n))
    res = {}
    # process in event blocks to bound memory (n x 512 float32)
    blk = 20000
    cnt = {k: np.zeros(512) for k in ('n1', 'f1', 'n2', 'f2')}
    occ = np.zeros(512)
    LR = {k: neighbours(pos, k) for k in (1, 2)}
    for s in range(0, n, blk):
        m = (inv >= s) & (inv < s + blk)
        A = np.zeros((min(blk, n - s), 512), np.float32)
        np.maximum.at(A, (inv[m] - s, ch[m]), amp[m])
        fired = A > FIRE
        occ += fired.sum(0)
        for k in (1, 2):
            L, R = LR[k]
            v = (L >= 0) & (R >= 0)
            cs = np.flatnonzero(v)
            both = (A[:, L[cs]] > BIG) & (A[:, R[cs]] > BIG)
            cnt[f'n{k}'][cs] += both.sum(0)
            cnt[f'f{k}'][cs] += (both & fired[:, cs]).sum(0)
    f1 = cnt['f1'] / np.maximum(cnt['n1'], 1); f2 = cnt['f2'] / np.maximum(cnt['n2'], 1)
    j1 = cnt['n1'] >= MIN_FLANK; j2 = cnt['n2'] >= MIN_FLANK
    f = np.where(j1, f1, np.where(j2, f2, np.nan))
    judged = j1 | j2
    with np.errstate(invalid='ignore'):
        dead = judged & (f < DEAD_REL); weak = judged & (f >= DEAD_REL) & (f < WEAK_REL)
    # silent: occupancy < SILENT_REL x the median occupancy of the view's mapped channels within +-32 strips in position
    # (a whole dead connector -- n_TOF A-x connector 8 through run_79 -- has no live flank, but it is silent)
    occ_r = occ / max(n, 1)
    ok = np.flatnonzero(np.isfinite(pos)); o = ok[np.argsort(pos[ok])]
    ref = np.full(512, np.nan)
    for i, c in enumerate(o):
        nb = o[max(0, i - 32):i + 33]
        ref[c] = np.median(occ_r[nb])
    vref = np.median(occ_r[ok]) if len(ok) else np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        silent = np.isfinite(pos) & (occ_r < SILENT_REL * np.maximum(ref, 0.5 * vref))
    L = lambda mm: np.flatnonzero(mm).tolist()
    out.update(silent=L(silent))
    out.update(dead=L(dead), weak=L(weak), judged=L(judged), f=np.round(np.nan_to_num(f, nan=-1), 3).tolist(),
               n1=cnt['n1'].astype(int).tolist(), n2=cnt['n2'].astype(int).tolist(), occupancy=(occ / max(n, 1)).round(5).tolist(),
               cuts=dict(fire=FIRE, sig=SIG, big=BIG, dead_rel=DEAD_REL, weak_rel=WEAK_REL, min_flank=MIN_FLANK))
    return out
