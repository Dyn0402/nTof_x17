#!/usr/bin/env python3
"""Do the plastics ring? Part 1: the after-pulse spectrum behind isolated hits.

If the PSS channels ring, every real pulse is followed by a *deterministic*
train of secondary PSA hits: excess at fixed delays, with follower amplitude
proportional to the leader's. That is distinguishable from

  - accidentals    (follow the singles rate, follower amplitude independent),
  - PMT afterpulsing (stochastic, ion-feedback delays ~0.1-3 us, no amplitude
                    proportionality),
  - real physics   (no amplitude link to the leader either).

Construction: take ISOLATED LARGE leaders (nothing on the same channel for
`--quiet` ns before them, amp_0 above `--lead-amp`), then histogram the delay to
every PSA hit that follows on the same channel within `--max-dt`. The accidental
level is *measured*, not modelled: the identical construction is repeated with
each leader's time transplanted into a different bunch of the same channel,
which carries the same rate profile, the same dead time and no correlation.

    python afterpulse_spectrum.py --parts 1 2 -o afterpulse.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import uproot

REPROC = Path('/media/dylan/data/x17/ntof_reproc/v12_liqpileup')
BR = ['segment', 'BunchNumber', 'detn', 'tof', 'amp', 'amp_0', 'fwhm',
      'chi2', 'pileup1', 'pileup2', 'satuflag', 'pulseshape']
# physics times only: the flash sits at ~11.6 us and the walls are diverted for
# ~1 us around it, so everything below 20 us is flash territory.
T_PHYS_LO = 20_000.0
T_PHYS_HI = 19_000_000.0
BANDS = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 500),
         (500, 1000), (1000, 5000), (5000, 20000)]


def load(det, parts):
    out = []
    for p in parts:
        f = REPROC / f'run224572_{p:04d}.root'
        if not f.exists():
            continue
        with uproot.open(f) as fh:
            out.append(fh[det].arrays(BR, library='np'))
    return {k: np.concatenate([a[k] for a in out]) for k in BR}


def followers(key, t, grp, lead_t, lead_grp, max_dt, span, cap=5000):
    """Every hit within `max_dt` after each leader, on the leader's own group.

    Returns (dt, follower index into t, leader index into lead_t). The leader
    times are passed separately so the same routine serves the event-mixed
    control, where a leader's time is looked up inside a *different* group.

    `key = group_index * span + t` is the globally sorted search key. Searching
    on `t` alone would be wrong: (grp, t) is only *lexicographically* sorted, so
    t restarts at every group boundary and is not monotonic -- searchsorted on it
    silently returns positions from an unrelated group.
    """
    start = np.searchsorted(key, lead_grp * span + lead_t, side='right')
    dts, foll, lead = [], [], []
    for k in range(cap):
        j = start + k
        alive = j < t.size
        jj = np.where(alive, j, t.size - 1)
        ok = alive & (grp[jj] == lead_grp) & (t[jj] > lead_t) \
            & (t[jj] - lead_t <= max_dt)
        if not ok.any():
            break
        dts.append(t[jj[ok]] - lead_t[ok])
        foll.append(jj[ok])
        lead.append(np.flatnonzero(ok))
    else:
        print(f'  ! follower depth hit the cap of {cap}')
    if not dts:
        z = np.array([], dtype=int)
        return np.array([]), z, z
    return np.concatenate(dts), np.concatenate(foll), np.concatenate(lead)


def analyse(det, parts, args, edges):
    a = load(det, parts)
    phys = (a['tof'] > T_PHYS_LO) & (a['tof'] < T_PHYS_HI)
    if not phys.any():
        return None
    grp = ((a['segment'][phys].astype(np.int64) * 100000
            + a['BunchNumber'][phys]) * 100 + a['detn'][phys])
    t = a['tof'][phys].astype(np.float64)
    amp = a['amp_0'][phys].astype(np.float64)
    if args.reverse:
        # run the clock backwards: "followers" become "preceders". A pulse-tail
        # artifact is strictly forward in time, an accidental is symmetric.
        t = T_PHYS_HI - t
    order = np.lexsort((t, grp))
    grp, t, amp = grp[order], t[order], amp[order]
    # compact group index, so that key = gi * span + t is globally monotonic
    ugrp, gi = np.unique(grp, return_inverse=True)
    span = T_PHYS_HI + 1.0
    key = gi * span + t

    # ---- leaders: large, and isolated on their own channel -------------------
    prev_dt = np.full(t.size, np.inf)
    prev_dt[1:] = np.where(grp[1:] == grp[:-1], t[1:] - t[:-1], np.inf)
    li = np.flatnonzero((amp > args.lead_amp) & (prev_dt > args.quiet))
    if li.size == 0:
        return None

    dt, foll, lead = followers(key, t, gi, t[li], gi[li], args.max_dt, span)
    counts = np.histogram(dt, bins=edges)[0]

    # ---- event-mixed control -------------------------------------------------
    # transplant each leader into the next bunch of the same channel: same rate
    # profile, same dead time, no correlation.
    bunch_of, chan = grp // 100, grp % 100
    ubun = np.unique(bunch_of)
    nxt = {b: ubun[(i + 1) % ubun.size] for i, b in enumerate(ubun)}
    mix_grp = np.array([nxt[k] for k in bunch_of[li]]) * 100 + chan[li]
    pos = np.searchsorted(ugrp, mix_grp)
    keep = (pos < ugrp.size) & (ugrp[np.minimum(pos, ugrp.size - 1)] == mix_grp)
    mdt, _, _ = followers(key, t, gi, t[li][keep], pos[keep], args.max_dt, span)
    mix = np.histogram(mdt, bins=edges)[0]
    mix_scale = li.size / max(int(keep.sum()), 1)

    # ---- amplitude relation: follower amplitude vs the leader's, per band ----
    lead_amp, foll_amp = amp[li][lead], amp[foll]
    bands = []
    for blo, bhi in BANDS:
        sel = (dt >= blo) & (dt < bhi)
        n = int(sel.sum())
        b = dict(lo=blo, hi=bhi, n=n)
        if n >= 20:
            la, fa = lead_amp[sel], foll_amp[sel]
            b |= dict(foll_amp_median=float(np.median(fa)),
                      ratio_median=float(np.median(fa / la)),
                      corr=float(np.corrcoef(la, fa)[0, 1]))
        bands.append(b)

    # follower yield in the ring band as a function of leader amplitude: a
    # ringing follower must scale with the leader, an accidental must not.
    q = np.quantile(amp[li], [0, .25, .5, .75, .9, 1.0])
    yield_vs_amp = []
    for i in range(len(q) - 1):
        m = (amp[li] >= q[i]) & (amp[li] < q[i + 1] if i < len(q) - 2
                                 else amp[li] <= q[i + 1])
        if m.sum() < 10:
            continue
        in_band = np.isin(lead, np.flatnonzero(m)) & (dt < args.ring_band)
        yield_vs_amp.append(dict(amp_lo=float(q[i]), amp_hi=float(q[i + 1]),
                                 n_lead=int(m.sum()),
                                 per_lead=float(in_band.sum() / m.sum())))

    pairs = None
    if args.dump_pairs:
        near = dt < args.dump_pairs
        pairs = dict(dt=dt[near], lead_amp=lead_amp[near],
                     foll_amp=foll_amp[near], mix_dt=mdt[mdt < args.dump_pairs])

    return dict(det=det, n_phys=int(phys.sum()), n_leaders=int(li.size),
                lead_amp_median=float(np.median(amp[li])),
                counts=counts.tolist(), mixed=mix.tolist(),
                mix_scale=float(mix_scale), bands=bands,
                yield_vs_amp=yield_vs_amp), pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', type=int, nargs='+', default=[1])
    ap.add_argument('--dets', nargs='+',
                    default=['PSSA', 'PSSB', 'PSSC', 'PSSD',
                             'WALA', 'WALB', 'LIQA', 'LIQB'])
    ap.add_argument('--max-dt', type=float, default=20_000.0)
    ap.add_argument('--quiet', type=float, default=5_000.0,
                    help='ns of silence required before a leader')
    ap.add_argument('--lead-amp', type=float, default=3_000.0)
    ap.add_argument('--ring-band', type=float, default=500.0,
                    help='ns; the band the yield-vs-leader-amplitude uses')
    ap.add_argument('--reverse', action='store_true',
                    help='run the clock backwards: measure the excess BEFORE '
                         'each leader instead of after')
    ap.add_argument('--dump-pairs', type=float, default=2000.0,
                    help='also save every pair below this Delta-t to <out>.npz')
    ap.add_argument('-o', '--out', default='afterpulse.json')
    args = ap.parse_args()

    edges = np.concatenate([np.arange(0, 500, 1.0),
                            np.geomspace(500, args.max_dt, 120)])
    res = {'edges': edges.tolist(), 'args': vars(args), 'dets': {}}
    dump = {}
    for det in args.dets:
        out = analyse(det, args.parts, args, edges)
        if out is None:
            print(f'{det}: nothing to do')
            continue
        r, pairs = out
        res['dets'][det] = r
        if pairs is not None:
            dump |= {f'{det}_{k}': v for k, v in pairs.items()}
        exc = sum(r['counts']) - sum(r['mixed']) * r['mix_scale']
        print(f'{det}: {r["n_leaders"]:,} isolated leaders '
              f'(median amp_0 {r["lead_amp_median"]:,.0f}) | followers '
              f'{sum(r["counts"]):,} vs mixed '
              f'{sum(r["mixed"]) * r["mix_scale"]:,.0f} '
              f'(excess {exc:+,.0f}, {exc / max(r["n_leaders"], 1):+.2f}/leader)')
    Path(args.out).write_text(json.dumps(res))
    print(f'wrote {args.out}')
    if dump:
        np.savez_compressed(Path(args.out).with_suffix('.npz'), **dump)
        print(f'wrote {Path(args.out).with_suffix(".npz")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
