#!/usr/bin/env python3
"""What the after-pulse flag does to the DREAM/PSS match, and how to set it.

Runs the flag of `afterpulse_flag.py` against a real slim and answers the three
questions the slim actually needs:

  1. how much of the plastic LATE TAIL does it remove,
  2. what does it cost in the CORE (|dt| < 25 ns), which is genuine signal, and
  3. how often does it fire on a hit that CANNOT be an after-pulse -- measured,
     not assumed, on the slim's own +100 us accidental control.

(3) is the honest cost axis. At the plastics' ~720 kHz singles a blanket
dead time throws away real hits at a rate that has nothing to do with
after-pulsing, and the control window measures exactly that rate.

Everything is background-subtracted against the same control, so "removed"
means removed from the excess and not from the accidental floor.

It also builds the per-trigger metric: for each (DREAM trigger, arm) the
PRIMARY plastic hit -- the earliest one that is not itself in the shadow of a
bigger hit -- and how far it sits from the prediction.

The flag is computed on the FULL n_TOF hit stream, not on the slim, so a hit
whose parent falls outside the slim window is still caught.

    python veto_on_dream.py <ntof_hits_*.root> --scan
    python veto_on_dream.py <ntof_hits_*.root> --t-hold 300 --ratio 0.10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from afterpulse_flag import prev_max_amp

SOURCE = Path('/media/dylan/data/x17/ntof_reproc/v12_liqpileup')
PSS = {'PSSA': 4, 'PSSB': 5, 'PSSC': 6, 'PSSD': 7}
# the n_TOF tof axis runs to ~80 ms; only the flash region is excluded here
T_PHYS_LO, T_PHYS_HI = 20_000.0, 1e8
CORE_NS = 25.0
LATE_NS = 150.0
T_HOLDS = [100.0, 200.0, 300.0, 500.0, 1000.0]
RATIOS = [0.02, 0.05, 0.10, 0.20, 0.35]
AMP_FLOORS = [100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 600.0]


def hit_key(bunch, det, detn, tof):
    """Collision-free int64 key for (bunch, detector, channel, time)."""
    return (((bunch.astype(np.int64) * 12 + det) * 4 + detn) * (1 << 40)
            + np.round(tof * 100.0).astype(np.int64))


def source_prevmax(run, bunches, t_holds, parts):
    """(sorted keys, prev-max-amp per lookback, amp) over the full n_TOF stream."""
    keys, pms, amps = [], [], []
    lo, hi = int(bunches.min()), int(bunches.max())
    for p in parts:
        f = SOURCE / f'run{run}_{p:04d}.root'
        if not f.exists():
            continue
        with uproot.open(f) as fh:
            for det, code in PSS.items():
                a = fh[det].arrays(['BunchNumber', 'detn', 'tof', 'amp_0'],
                                   library='np')
                m = ((a['BunchNumber'] >= lo) & (a['BunchNumber'] <= hi)
                     & (a['tof'] > T_PHYS_LO) & (a['tof'] < T_PHYS_HI))
                if not m.any():
                    continue
                b = a['BunchNumber'][m].astype(np.int64)
                dn = a['detn'][m].astype(np.int64)
                tof = a['tof'][m].astype(np.float64)
                amp = a['amp_0'][m].astype(np.float64)
                pm = prev_max_amp(b * 100 + dn, tof, amp, t_holds)
                keys.append(hit_key(b, code, dn, tof))
                pms.append(pm)
                amps.append(amp)
        print(f'  read part {p}')
    k = np.concatenate(keys)
    pm = np.concatenate(pms, axis=1)
    am = np.concatenate(amps)
    o = np.argsort(k)
    return k[o], pm[:, o], am[o]


def excess(dt, ctrl, keep, lo, hi):
    """Background-subtracted counts with |dt| in [lo, hi)."""
    s = np.abs(dt) >= lo
    s &= np.abs(dt) < hi
    return int(np.count_nonzero(s & ~ctrl & keep)
               - np.count_nonzero(s & ctrl & keep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('slim')
    ap.add_argument('--scan', action='store_true')
    ap.add_argument('--t-hold', type=float, default=300.0)
    ap.add_argument('--ratio', type=float, default=0.10)
    ap.add_argument('--parts', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--run', type=int, default=224572)
    ap.add_argument('-o', '--out', default='veto_on_dream.json')
    args = ap.parse_args()

    with uproot.open(args.slim) as fh:
        h = fh['hits'].arrays(['eventId', 'det', 'detn', 'tof', 'dt_ns',
                               'amp_0', 'is_control'], library='np')
        ev = fh['events'].arrays(['eventId', 'bunch'], library='np')

    ev_id = ev['eventId'].astype(np.int64)
    ev_bunch = np.zeros(int(ev_id.max()) + 1, dtype=np.int64)
    ev_bunch[ev_id] = ev['bunch'].astype(np.int64)
    bunch = ev_bunch[h['eventId'].astype(np.int64)]

    is_pss = (h['det'] >= 4) & (h['det'] <= 7)
    print(f'{is_pss.sum():,} plastic hits of {h["det"].size:,} in the slim')

    t_holds = T_HOLDS if args.scan else [args.t_hold]
    print('computing prev-max amplitude on the full n_TOF stream...')
    keys, pm, _am = source_prevmax(args.run, ev['bunch'].astype(np.int64),
                                   t_holds, args.parts)
    hk = hit_key(bunch[is_pss], h['det'][is_pss].astype(np.int64),
                 h['detn'][is_pss].astype(np.int64), h['tof'][is_pss])
    pos = np.searchsorted(keys, hk)
    ok = (pos < keys.size) & (keys[np.minimum(pos, keys.size - 1)] == hk)
    print(f'  joined {ok.mean() * 100:.3f} % of slim plastic hits to the source')
    if ok.mean() < 0.99:
        print('  ! join incomplete -- flags default to "not flagged" for the rest')
    pmax = np.zeros((len(t_holds), hk.size))
    pmax[:, ok] = pm[:, pos[ok]]

    dt = h['dt_ns'][is_pss].astype(np.float64)
    ctrl = h['is_control'][is_pss].astype(bool)
    amp = h['amp_0'][is_pss].astype(np.float64)
    allk = np.ones(dt.size, bool)
    base_core = excess(dt, ctrl, allk, 0, CORE_NS)
    base_late = excess(dt, ctrl, allk, LATE_NS, 1000.0)
    base_mid = excess(dt, ctrl, allk, CORE_NS, LATE_NS)
    print(f'\nbefore any flag: core {base_core:,}  25-150 ns {base_mid:,}  '
          f'150-1000 ns {base_late:,}')

    # An amplitude floor alone is a serious competitor, because the after-pulse
    # amplitude is ~120 ADC almost independently of its parent while a genuine
    # plastic coincidence is a MIP. Scan it, and scan it TOGETHER with the
    # shadow flag, so the recommendation is made against the simpler option
    # rather than assuming the more complicated one wins.
    print('\namplitude floor alone:')
    print('   amp_0 > | core kept | 25-150 cut | 150-1k cut')
    amp_rows = []
    for a0 in AMP_FLOORS:
        k = amp > a0
        core = excess(dt, ctrl, k, 0, CORE_NS)
        mid = excess(dt, ctrl, k, CORE_NS, LATE_NS)
        late = excess(dt, ctrl, k, LATE_NS, 1000.0)
        amp_rows.append(dict(amp_floor=a0, core_kept=core / base_core,
                             mid_removed=1 - mid / base_mid,
                             late_removed=1 - late / base_late))
        print(f'   {a0:7.0f} |  {core / base_core * 100:6.2f} % |'
              f'   {(1 - mid / base_mid) * 100:6.2f} % |'
              f'   {(1 - late / base_late) * 100:6.2f} %')

    rows = []
    ratios = RATIOS if args.scan else [args.ratio]
    print('\n T_HOLD  RATIO | core kept | 25-150 cut | 150-1k cut | control-window'
          '\n                |           |            |            | hits vetoed')
    for w, th in enumerate(t_holds):
        for r in ratios:
            fl = amp < r * pmax[w]
            keep = ~fl
            core = excess(dt, ctrl, keep, 0, CORE_NS)
            mid = excess(dt, ctrl, keep, CORE_NS, LATE_NS)
            late = excess(dt, ctrl, keep, LATE_NS, 1000.0)
            cost = float(fl[ctrl].mean())
            rows.append(dict(t_hold=th, ratio=r,
                             core_kept=core / base_core,
                             mid_removed=1 - mid / base_mid,
                             late_removed=1 - late / base_late,
                             control_vetoed=cost,
                             flagged=float(fl.mean())))
            print(f'  {th:5.0f}   {r:.2f} |  {core / base_core * 100:6.2f} % |'
                  f'   {(1 - mid / base_mid) * 100:6.2f} % |'
                  f'   {(1 - late / base_late) * 100:6.2f} % |'
                  f'   {cost * 100:6.2f} %')

    # the combination, at the operating point
    print('\ncombined: amplitude floor AND the shadow flag')
    print('   amp_0 > | T_HOLD RATIO | core kept | 25-150 cut | 150-1k cut')
    comb = []
    for a0 in (150.0, 200.0, 250.0):
        for w, th in enumerate(t_holds):
            for r in (0.02, 0.05):
                k = (amp > a0) & ~(amp < r * pmax[w])
                core = excess(dt, ctrl, k, 0, CORE_NS)
                mid = excess(dt, ctrl, k, CORE_NS, LATE_NS)
                late = excess(dt, ctrl, k, LATE_NS, 1000.0)
                comb.append(dict(amp_floor=a0, t_hold=th, ratio=r,
                                 core_kept=core / base_core,
                                 mid_removed=1 - mid / base_mid,
                                 late_removed=1 - late / base_late))
                if th in (300.0, 1000.0):
                    print(f'   {a0:7.0f} | {th:5.0f} {r:5.2f} |'
                          f'  {core / base_core * 100:6.2f} % |'
                          f'   {(1 - mid / base_mid) * 100:6.2f} % |'
                          f'   {(1 - late / base_late) * 100:6.2f} %')

    res = {'base': dict(core=base_core, mid=base_mid, late=base_late),
           'scan': rows, 'amp_scan': amp_rows, 'combined': comb,
           'join_rate': float(ok.mean())}

    # ---- the per-trigger metric ---------------------------------------------
    w = t_holds.index(args.t_hold) if args.t_hold in t_holds else 0
    fl = amp < args.ratio * pmax[w]
    arm = h['det'][is_pss].astype(np.int64) - 4
    evid = h['eventId'][is_pss].astype(np.int64)
    with uproot.open(args.slim) as fh:
        ev2 = fh['events'].arrays(['eventId', 'arm', 'matched'], library='np')
    ev_arm = np.full(int(ev_id.max()) + 1, -1, dtype=np.int64)
    ev_arm[ev2['eventId'].astype(np.int64)] = ev2['arm'].astype(np.int64)
    print(f'\nper-trigger PRIMARY plastic hit (T={t_holds[w]:.0f} ns, '
          f'R={args.ratio:.2f}), signal window only:')
    own_arm = ev_arm[evid] == (h['det'][is_pss].astype(np.int64) - 4)
    for lab, sel in (('earliest hit', ~ctrl),
                     ('earliest unflagged', ~ctrl & ~fl),
                     ('largest amplitude', ~ctrl),
                     ('largest, amp>250', ~ctrl & (amp > 250)),
                     ('largest, trigger arm', ~ctrl & own_arm),
                     ('largest, arm & amp>250', ~ctrl & own_arm & (amp > 250))):
        e, a_, d_, am_ = evid[sel], arm[sel], dt[sel], amp[sel]
        key = e * 4 + a_
        rank = d_ if lab.startswith('earliest') else -am_
        o = np.lexsort((rank, key))
        key_s, d_s = key[o], d_[o]
        first = np.ones(key_s.size, bool)
        first[1:] = key_s[1:] != key_s[:-1]
        fd = d_s[first]
        within = float(np.mean(np.abs(fd) < CORE_NS))
        res.setdefault('primary', {})[lab] = dict(
            n=int(fd.size), within_core=within, median=float(np.median(fd)))
        print(f'  {lab:20s} {fd.size:8,d} (trigger, arm) pairs   '
              f'{within * 100:5.1f} % within +-25 ns   '
              f'median {np.median(fd):+8.1f} ns')

    Path(args.out).write_text(json.dumps(res, indent=1))
    np.savez_compressed(Path(args.out).with_suffix('.npz'), dt=dt, ctrl=ctrl,
                        flagged=fl, arm=arm, evid=evid, amp=amp,
                        pmax=pmax, t_holds=np.array(t_holds),
                        own_arm=own_arm, t_hold=t_holds[w], ratio=args.ratio)
    print(f'\nwrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
