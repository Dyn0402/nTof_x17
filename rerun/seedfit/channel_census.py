#!/usr/bin/env python3
"""channel_census.py <run_key> --feu-x F --feu-y F [--subs s1 s2 ...] --out census.json

The W3 channel layer of one chamber on one night, from the hits (candidate finding / QA only) and the M3 reference;
no position, angle or depth is reconstructed from hit times here.

Per plane:
  noisy    pedestal rms (hits_root 'pedestals') > NOISY_X x the plane median.  The wft noise is a post-CNS MAD and
           does not see these: on det7 X the noisy channels swing +-~900 ADC, bipolar, in ~25 % of triggers, wherever
           the track is (global pickup), and read 13 ADC to the MAD against 11 for clean channels.
  shorts   channels whose amplitude over events correlates > SHORT_CORR with a strip >= 2 pitches away (cosine of the
           amplitude products over events with 2-40 hits, significance > 5, unsaturated).  Shorted channels carry
           the charge of every strip of the group at once, a few mm apart, so the fit reads a tilt: positions
           survive, angles do not.  det3 X: one pair; det7 X: 114 channels, mostly in groups 5 apart.
  dead/weak  firing probability (hit > FIRE_ADC) when the M3 track at the mesh crosses the channel's mapped strip
           (|d| < 0.6 pitch), over the median of its clean neighbours (+-6): dead < 0.25, weak < 0.6.
  map      per 64-channel connector, the median M3 coordinate of the events in which a channel is the largest hit
           (> 200 ADC) against its mapped position: slope (should be +-pitch), residual rms, Spearman.  A connector
           plugged reversed or permuted shows up here; det7 X connectors 1 and 3 follow the map (Spearman -0.96/-0.91).

    ../../.venv/bin/python channel_census.py g_det7_long --feu-x 6 --feu-y 8 --subs long_run longer_run --out c.json
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]

NOISY_X = 3.0
SHORT_CORR = 0.7
FIRE_ADC = 50.0
MAX_ADC = 200.0
PITCH = 0.78
BAND_FILL = 0.7          # a correlated group filling >= this fraction of its span is a band, not a short
MIN_CROSS = 15            # crossings needed to judge a channel (one 8 k-event sub-run gives ~20)


def rays_at(cfg, sub):
    from qa_config import M3_CHI2_CUT, M3_MIN_NCLUS
    from M3RefTracking import M3RefTracking
    sp = os.path.join(cfg.BASE_PATH, cfg.RUN, sub)
    d = os.path.join(sp, 'm3_tracking_root_v2')
    d = d if os.path.isdir(d) else os.path.join(sp, 'm3_tracking_root')
    rays = M3RefTracking(d + '/', chi2_cut=M3_CHI2_CUT, min_nclus=M3_MIN_NCLUS)
    x, y, evn = (np.asarray(v) for v in rays.get_xy_positions(cfg.DET_PLANE_Z))
    return {int(e): (a, b) for e, a, b in zip(evn, x, y)}


def read_hits(f):
    import uproot
    a = uproot.open(f)['hits'].arrays(['eventId', 'channel', 'amplitude', 'significance', 'saturated'], library='np')
    o = np.argsort(a['eventId'], kind='stable')
    return {k: v[o] for k, v in a.items()}


def plane(cfg, subs, feu, pos):
    import uproot
    files = {s: sorted(glob.glob(os.path.join(cfg.BASE_PATH, cfg.RUN, s, 'hits_root', f'*_{feu:02d}_hits.root')))
             for s in subs}
    first = next(f for fs in files.values() for f in fs)
    rms = uproot.open(first)['pedestals'].arrays(library='np')['rms'].astype(float)
    noisy = rms > NOISY_X * np.median(rms)

    S = np.zeros((512, 512)); N = np.zeros(512)
    mch, mcrd = [], []                                   # largest hit per event -> M3 (x, y)
    fire = np.zeros(512); cross = np.zeros(512)
    events = []
    for s, fs in files.items():
        T = rays_at(cfg, s)
        for f in fs:
            a = read_hits(f)
            e, c, q = a['eventId'], a['channel'].astype(int), a['amplitude']
            good = (a['significance'] > 5) & ~a['saturated']
            b = np.flatnonzero(np.diff(e)) + 1
            for s0, s1 in zip(np.r_[0, b], np.r_[b, len(e)]):
                g = good[s0:s1]; cc, qq = c[s0:s1][g], q[s0:s1][g]
                if 2 <= len(cc) <= 40:
                    S[np.ix_(cc, cc)] += np.outer(qq, qq); N[cc] += qq ** 2
                t = T.get(int(e[s0]))
                if t is None:
                    continue
                events.append((t, dict(zip(cc.tolist(), qq.tolist()))))
                if len(qq) and qq.max() > MAX_ADC:
                    mch.append(int(cc[np.argmax(qq)])); mcrd.append(t)
    C = S / np.sqrt(np.maximum(np.outer(N, N), 1e-9)); np.fill_diagonal(C, 0)
    idx = np.arange(512)
    A = (C > SHORT_CORR) & (np.abs(pos[:, None] - pos[None, :]) > 1.5 * PITCH)     # not physical neighbours (connectors meet inverted)
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    ng, lab = connected_components(csr_matrix(A), directed=False)
    groups = sorted([np.flatnonzero(lab == g).tolist() for g in range(ng) if (lab == g).sum() > 1], key=len, reverse=True)
    # A group that fills most of its own span is a band of strips that fire together on wide clusters (det4's
    # low-gain stripes, which fire only on big events), not a short: shorts skip the channels in between.
    bands = [g for g in groups if len(g) >= 4 and len(g) >= BAND_FILL * (g[-1] - g[0] + 1)]
    shorts = np.zeros(512, bool)
    for g in groups:
        if g not in bands:
            shorts[g] = True

    # M3 -> local strip position: which M3 axis, sign and offset, from the per-channel medians of the largest hit
    mch = np.array(mch); mcrd = np.array(mcrd)
    med = {c: np.median(mcrd[mch == c], axis=0) for c in np.unique(mch) if (mch == c).sum() >= 8}
    cs = np.array(sorted(med)); M = np.array([med[c] for c in cs]); P = pos[cs]
    k = int(abs(np.corrcoef(P, M[:, 1])[0, 1]) > abs(np.corrcoef(P, M[:, 0])[0, 1]))
    sgn = np.sign(np.corrcoef(P, M[:, k])[0, 1])
    off = np.median(P - sgn * M[:, k])
    for _ in range(5):
        r = P - (sgn * M[:, k] + off); off += np.median(r[np.abs(r) < 5 * PITCH]) if (np.abs(r) < 5 * PITCH).any() else 0
    mapping = []
    for b in range(8):
        m = (cs >= 64 * b) & (cs < 64 * b + 64)
        if m.sum() < 10:
            mapping.append(dict(conn=b, n=int(m.sum())))
            continue
        loc = sgn * M[m, k] + off
        p = np.polyfit(pos[cs[m]], loc, 1); res = loc - np.polyval(p, pos[cs[m]])
        rho = np.corrcoef(np.argsort(np.argsort(pos[cs[m]])), np.argsort(np.argsort(loc)))[0, 1]
        mapping.append(dict(conn=b, n=int(m.sum()), slope=round(float(p[0]), 3), resid_rms_mm=round(float(res.std()), 2),
                            spearman=round(float(rho), 3), offset_mm=round(float(np.median(loc - pos[cs[m]])), 2)))

    for t, hits in events:
        loc = sgn * t[k] + off
        for c in np.flatnonzero(np.abs(pos - loc) < 0.6 * PITCH):
            cross[c] += 1; fire[c] += hits.get(int(c), 0.0) > FIRE_ADC
    f = fire / np.maximum(cross, 1); judged = cross >= MIN_CROSS
    clean = judged & ~noisy & ~shorts
    ref = np.array([np.nanmedian(np.where(clean, f, np.nan)[max(0, i - 6):i + 7]) if clean[max(0, i - 6):i + 7].any()
                    else np.nan for i in range(512)])
    r = f / ref
    dead = judged & (r < 0.25); weak = judged & (r >= 0.25) & (r < 0.6)
    L = lambda m: np.flatnonzero(m).tolist()
    return dict(noisy=L(noisy), shorts=L(shorts), short_groups=[g for g in groups if g not in bands], bands=bands, dead=L(dead), weak=L(weak),
                unjudged=L(~judged), mapping=mapping, m3_axis='xy'[k], m3_sign=int(sgn), m3_offset_mm=round(float(off), 2),
                rms=rms.round(1).tolist(), fire=f.round(3).tolist(), crossings=cross.astype(int).tolist(),
                fire_median_clean=round(float(np.median(f[clean])), 3) if clean.any() else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_key'); ap.add_argument('--feu-x', type=int, required=True); ap.add_argument('--feu-y', type=int, required=True)
    ap.add_argument('--subs', nargs='*', default=None); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from qa_config import get_config, setup_paths
    setup_paths()
    from wft.io import strip_position_map
    cfg = get_config(a.run_key); pm = strip_position_map(cfg)
    subs = a.subs or [cfg.SUB_RUN]
    out = dict(run_key=a.run_key, detector=cfg.DET_NAME, subs=subs,
               cuts=dict(noisy_x=NOISY_X, short_corr=SHORT_CORR, fire_adc=FIRE_ADC, max_adc=MAX_ADC))
    for p, feu in (('x', a.feu_x), ('y', a.feu_y)):
        out[p] = plane(cfg, subs, feu, np.asarray(pm[feu], float))
        o = out[p]
        print(f"{a.run_key} {p} (FEU {feu}): noisy {len(o['noisy'])} shorts {len(o['shorts'])} bands {sum(map(len, o['bands']))} dead {len(o['dead'])} "
              f"weak {len(o['weak'])} unjudged {len(o['unjudged'])}; map " +
              ' '.join(f"c{m['conn']}:{m.get('spearman', float('nan')):+.2f}/{m.get('resid_rms_mm', float('nan')):.1f}" for m in o['mapping']))
    json.dump(out, open(a.out, 'w'))


if __name__ == '__main__':
    main()
