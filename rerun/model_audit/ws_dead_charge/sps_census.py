#!/usr/bin/env python3
"""sps_census.py <pair_*.npz> --tag NAME --out census/sps_<tag>.json

The bench W3 dead/weak census (rerun/seedfit/channel_census.py) redone for det4 at H4, with the uRWELL telescope in
place of M3: per FEU-03 channel, the probability that it fires (hit > FIRE_ADC, significance > 5) when the uRWELL
track at det4 crosses its strip (|d| < 0.6 pitch), over the median of its clean neighbours (+-6 in position):
dead < 0.25, weak < 0.6, judged with >= MIN_CROSS crossings.  Same cuts as the bench, so the two lists compare.

Track at det4: front/back uRWELL points (single cluster in all four planes) interpolated to z_det4 (scanned for the
minimum residual against det4's own brightest strip per view), then one affine fit to det4-local (POSITION_MM frame,
which is the bench strip-map frame: DET4_URW_MAPPING_2026-08-01.md).  Flat-mount datasets only (head-on tracks).
Also: per channel occupancy and median amplitude (for the band / gain picture) and the channel's bench equivalent
(bench FEU channel at the same position: X ch + 128, Y ch - 128).
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment'))
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM      # noqa: E402

FIRE_ADC, SIG, DEAD_REL, WEAK_REL, MIN_CROSS = 50.0, 5.0, 0.25, 0.6, 15
Z_BACK = 1370.0


def lead_pos(h_ev, h_ch, h_amp, n_ev, view):
    k = VIEW[h_ch] == view
    ev, ch, amp = h_ev[k], h_ch[k], h_amp[k]
    o = np.lexsort((amp, ev)); ev, ch = ev[o], ch[o]
    last = np.r_[ev[1:] != ev[:-1], True]
    L = np.full(n_ev, np.nan); L[ev[last]] = POSITION_MM[ch[last]]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pair'); ap.add_argument('--tag', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--max-events', type=int, default=2_000_000)
    a = ap.parse_args()
    P = np.load(a.pair)
    n_ev = min(len(P['ev_id']), a.max_events)
    clean = np.ones(n_ev, bool)
    for k in ('fx', 'fy', 'bx', 'by'):
        clean &= (P[k + '_n'][:n_ev] == 1) & np.isfinite(P[k + '_p'][:n_ev])
    fx, fy, bx, by = (P[k + '_p'][:n_ev] for k in ('fx', 'fy', 'bx', 'by'))
    h_ev = P['h_ev']; m = h_ev < n_ev
    h_ev, h_ch, h_amp, h_sig = h_ev[m], P['h_ch'][m].astype(int), P['h_amp'][m], P['h_sig'][m]
    good = (h_amp > FIRE_ADC) & (h_sig > SIG)
    h_ev, h_ch, h_amp = h_ev[good], h_ch[good], h_amp[good]
    LX, LY = lead_pos(h_ev, h_ch, h_amp, n_ev, 'x'), lead_pos(h_ev, h_ch, h_amp, n_ev, 'y')
    best = None
    for z in np.arange(1000, 1301, 20.0):
        f = z / Z_BACK
        tx, ty = fx + (bx - fx) * f, fy + (by - fy) * f
        keep = clean & np.isfinite(LX + LY + tx + ty)
        U = np.column_stack([tx, ty, np.ones(n_ev)])
        A, *_ = np.linalg.lstsq(U[keep], np.column_stack([LX, LY])[keep], rcond=None)
        for _ in range(4):
            r = np.hypot(*(np.column_stack([LX, LY]) - U @ A).T)
            g = keep & (r < 3 * np.nanmedian(r[keep]))
            A, *_ = np.linalg.lstsq(U[g], np.column_stack([LX, LY])[g], rcond=None)
        r = np.hypot(*(np.column_stack([LX, LY]) - U @ A).T)
        med = float(np.nanmedian(r[keep]))
        if best is None or med < best[1]:
            best = (z, med, A)
    z, med, A = best
    f = z / Z_BACK
    U = np.column_stack([fx + (bx - fx) * f, fy + (by - fy) * f, np.ones(n_ev)])
    pred = U @ A
    print(f'{a.tag}: {n_ev} events, {clean.sum()} clean; z {z:.0f} mm, median |res| {med:.2f} mm', flush=True)

    # fired matrix as (event, channel) pairs
    fired = set()
    key = h_ev.astype(np.int64) * 512 + h_ch
    fired_keys = np.unique(key)
    cross = np.zeros(512); fire = np.zeros(512)
    for view, col in (('x', 0), ('y', 1)):
        chs = np.flatnonzero(VIEW == view)
        pv = pred[:, col]
        ev_ok = np.flatnonzero(clean & np.isfinite(pv))
        # sort the strips by position; for each event the strips within 0.6 pitch of the track
        ps = POSITION_MM[chs]; o = np.argsort(ps); ps, cs = ps[o], chs[o]
        lo = np.searchsorted(ps, pv[ev_ok] - 0.6 * PITCH_MM); hi = np.searchsorted(ps, pv[ev_ok] + 0.6 * PITCH_MM)
        for d in range(0, 3):                      # at most two strips within +-0.6 pitch
            j = lo + d; ok = j < hi
            ee, cc = ev_ok[ok], cs[j[ok]]
            np.add.at(cross, cc, 1)
            hit = np.isin(ee.astype(np.int64) * 512 + cc, fired_keys, assume_unique=False)
            np.add.at(fire, cc[hit], 1)
    fr = fire / np.maximum(cross, 1)
    judged = cross >= MIN_CROSS
    occ = np.bincount(h_ch, minlength=512) / max(n_ev, 1)
    amp_med = np.array([np.median(h_amp[h_ch == c]) if (h_ch == c).sum() >= 10 else np.nan for c in range(512)])
    # neighbour reference in POSITION within the view (+-6 strips), judged channels only
    ref = np.full(512, np.nan)
    for view in 'xy':
        chs = np.flatnonzero(VIEW == view); o = np.argsort(POSITION_MM[chs]); cs = chs[o]
        for i, c in enumerate(cs):
            nb = cs[max(0, i - 6):i + 7]
            nb = nb[judged[nb]]
            if len(nb) >= 3:
                ref[c] = np.median(fr[nb])
    # the bench census takes the reference from clean channels; with det4's dead stripes a +-6 window inside a
    # stripe has no live reference, so also use the view's median over judged live channels as a floor
    vmed = {v: float(np.percentile(fr[judged & (VIEW == v)], 75)) for v in 'xy'}
    refv = np.array([max(ref[c] if np.isfinite(ref[c]) else 0.0, 0.5 * vmed[VIEW[c]]) for c in range(512)])
    rel = fr / refv
    # bench-identical rule (neighbour reference only): a channel inside a dead stripe has dead neighbours, so it is
    # judged against them and is NOT called dead -- exactly as channel_census.py does on the bench
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_nb = fr / ref
    dead_nb = judged & (rel_nb < DEAD_REL)
    dead = judged & (rel < DEAD_REL); weak = judged & (rel >= DEAD_REL) & (rel < WEAK_REL)
    bench_ch = np.where(VIEW == 'x', np.arange(512) + 128, np.arange(512) - 256 + 128)
    L = lambda mm: np.flatnonzero(mm).tolist()
    out = dict(tag=a.tag, pair=os.path.abspath(a.pair), z_det4=z, median_resid_mm=med, n_events=int(n_ev), n_clean=int(clean.sum()),
               cuts=dict(fire_adc=FIRE_ADC, sig=SIG, dead_rel=DEAD_REL, weak_rel=WEAK_REL, min_cross=MIN_CROSS),
               view_fire_median=vmed)
    for v in 'xy':
        mv = VIEW == v
        out[v] = dict(dead=L(dead & mv), dead_nb=L(dead_nb & mv), weak=L(weak & mv), judged=L(judged & mv),
                      dead_nb_bench_ch=sorted(int(bench_ch[c]) for c in L(dead_nb & mv)),
                      dead_bench_ch=sorted(int(bench_ch[c]) for c in L(dead & mv)),
                      judged_bench_ch=sorted(int(bench_ch[c]) for c in L(judged & mv)),
                      dead_pos_mm=sorted(round(float(POSITION_MM[c]), 2) for c in L(dead & mv)))
    out['fire'] = fr.round(3).tolist(); out['rel'] = np.round(rel, 3).tolist(); out['crossings'] = cross.astype(int).tolist()
    out['occupancy'] = occ.round(5).tolist(); out['amp_median'] = np.round(amp_med, 1).tolist()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w'))
    print(f"{a.tag}: judged X {len(out['x']['judged'])} Y {len(out['y']['judged'])}; dead X {len(out['x']['dead'])} Y {len(out['y']['dead'])}; "
          f"weak X {len(out['x']['weak'])} Y {len(out['y']['weak'])}; view fire medians {vmed}")


if __name__ == '__main__':
    main()
