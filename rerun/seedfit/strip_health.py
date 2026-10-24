#!/usr/bin/env python3
"""strip_health.py -- dead channels and relative strip gains from a calibration cache's own waveforms.

For every event and plane, strips are ordered by position; for each strip whose two neighbours both peak above
NSIG x noise, r = peak / mean(neighbour peaks).  Per channel, the median r over events (n >= NMIN), divided by the
chamber median, is the channel's relative response g.  dead: g < DEAD_REL (censored by the fit);
gain: g for DEAD_REL <= g, clipped to [GMIN, GMAX] (1.0 where unjudged).  A strip next to a dead one is judged only
against live neighbours (the dead one is skipped), so a dead block does not hide its edges.

    strip_health.py --cache cache.pkl --out health.json
"""
import argparse, json, pickle
import numpy as np

NSIG, NMIN, DEAD_REL, GMIN, GMAX = 5.0, 15, 0.25, 0.3, 2.0


def health(ev, plane, dead_prev=None):
    dead_prev = set(dead_prev or ())
    acc = {}
    seen = {}
    for e in ev.values():
        if plane not in e:
            continue
        P = e[plane]
        ch = np.asarray(P['ch'], int); pos = np.asarray(P['pos'], float)
        pk = np.asarray(P['W'], float).max(1); nz = np.maximum(np.asarray(P['noise'], float), 3.0)
        o = np.argsort(pos); ch, pos, pk, nz = ch[o], pos[o], pk[o], nz[o]
        live = np.array([c not in dead_prev for c in ch])
        for j in range(len(ch)):
            seen[ch[j]] = seen.get(ch[j], 0) + 1
            l = j - 1
            while l >= 0 and not live[l]:
                l -= 1
            r = j + 1
            while r < len(ch) and not live[r]:
                r += 1
            if l < 0 or r >= len(ch) or pos[r] - pos[l] > 4 * 0.78:
                continue
            if pk[l] > NSIG * nz[l] and pk[r] > NSIG * nz[r] and pk[l] < 3500 and pk[r] < 3500 and pk[j] < 3500:
                acc.setdefault(int(ch[j]), []).append(pk[j] / (0.5 * (pk[l] + pk[r])))
    rate = {c: float(np.median(v)) for c, v in acc.items() if len(v) >= NMIN}
    med = float(np.median(list(rate.values())))
    g = {c: v / med for c, v in rate.items()}
    dead = sorted(c for c, v in g.items() if v < DEAD_REL)
    return dict(median=med, n_judged=len(g), dead=dead, gain={str(c): float(np.clip(v, GMIN, GMAX)) for c, v in g.items()
                                                                 if v >= DEAD_REL})


def responsiveness(ev, plane, near=2.0 * 0.78):
    """Per channel: of the events where a strip within `near` mm peaks above 10 sigma, the fraction where this
    channel itself peaks above 3 sigma.  Catches dead blocks that the neighbour ratio cannot judge."""
    lit, tot = {}, {}
    for e in ev.values():
        if plane not in e:
            continue
        P = e[plane]
        ch = np.asarray(P['ch'], int); pos = np.asarray(P['pos'], float)
        pk = np.asarray(P['W'], float).max(1); nz = np.maximum(np.asarray(P['noise'], float), 3.0)
        strong = pk > 10 * nz
        for j in range(len(ch)):
            d = np.abs(pos - pos[j])
            if np.any(strong & (d > 0) & (d <= near)):
                tot[ch[j]] = tot.get(ch[j], 0) + 1
                lit[ch[j]] = lit.get(ch[j], 0) + int(pk[j] > 3 * nz[j])
    return {int(c): (lit[c] / tot[c], tot[c]) for c in tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ev = pickle.load(open(a.cache, 'rb'))
    out = {}
    for p in ('x', 'y'):
        h = health(ev, p)
        rs = responsiveness(ev, p)
        unresp = sorted(c for c, (f, n) in rs.items() if n >= NMIN and f < 0.3)
        h = health(ev, p, sorted(set(h['dead']) | set(unresp)))   # second pass: judge dead blocks' edges against live neighbours
        h['dead'] = sorted(set(h['dead']) | set(unresp))
        h['unresponsive'] = unresp
        out[p] = h
        gs = np.array(list(h['gain'].values()))
        print(f"{p}: judged {h['n_judged']}  dead {len(h['dead'])} {h['dead'][:30]}  gain p5/p50/p95 "
              f"{np.percentile(gs, 5):.2f}/{np.percentile(gs, 50):.2f}/{np.percentile(gs, 95):.2f}  <0.6: {np.sum(gs < 0.6)}")
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
