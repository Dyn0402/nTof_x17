#!/usr/bin/env python3
"""DREAM-side burst forensics: is the tagged flash trigger the real flash?

Per-bunch lag scans (2026-08-11 night) show every bunch of a failed sub-run
carries its own ~1.6-13 ms offset: the burst's tagged time reference precedes
the true gamma flash by a random ms-scale amount. If so, the FIRST trigger of
each burst in a failed sub-run should be an ordinary/small event, with the
real flash (a huge-multiplicity event) sitting ms later in the burst.

Reads ONLY DREAM combined_hits. For each sub-run:
  - burst structure from trigger timestamps (bunch_join.dream_events)
  - hit multiplicity per event (rows per eventId in the hits tree)
  - per burst: multiplicity of first trigger, index and time offset of the
    max-multiplicity event, gap first->second trigger
Prints one comparison row per sub-run and writes a JSON.

Usage: dream_forensics.py <run>/<subrun> [<run>/<subrun> ...]
Env:   X17_BEAM_JULY must point at the beam_july tree.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uproot                                              # noqa: E402
from ntof_dream_merge import bunch_join as bj              # noqa: E402
from ntof_july_analysis import pulse_match as pm           # noqa: E402


def event_mult(run, subrun):
    """eventId -> number of hit rows (and summed amplitude if present)."""
    ids, amps = [], []
    amp_branch = None
    for f in pm._combined_files(run, subrun):
        with uproot.open(f) as uf:
            if 'hits' not in uf:
                continue
            t = uf['hits']
            if amp_branch is None:
                for cand in ('amplitude', 'amp', 'charge', 'adc'):
                    if cand in t.keys():
                        amp_branch = cand
                        break
            cols = ['eventId'] + ([amp_branch] if amp_branch else [])
            a = t.arrays(cols, library='np')
        ids.append(a['eventId'])
        if amp_branch:
            amps.append(a[amp_branch].astype(np.float64))
    eid = np.concatenate(ids)
    uid, cnt = np.unique(eid, return_counts=True)
    mult = dict(zip(uid.tolist(), cnt.tolist()))
    asum = {}
    if amps:
        amp = np.concatenate(amps)
        order = np.argsort(eid, kind='stable')
        eid_s, amp_s = eid[order], amp[order]
        bounds = np.searchsorted(eid_s, uid)
        sums = np.add.reduceat(amp_s, bounds)
        asum = dict(zip(uid.tolist(), sums.tolist()))
    return mult, asum, amp_branch


def forensics(run, subrun):
    ev = bj.dream_events(run, subrun)
    mult, asum, ab = event_mult(run, subrun)
    ev = ev.sort_values(['burst_id', 'trigger_ns']).reset_index(drop=True)
    ev['mult'] = ev['eventId'].map(mult).fillna(0).astype(int)

    rows = []
    for bid, g in ev.groupby('burst_id'):
        m = g['mult'].to_numpy()
        t = g['trigger_ns'].to_numpy()
        if len(g) < 5:
            continue
        imax = int(m.argmax())
        rows.append(dict(
            burst=int(bid), n=len(g),
            first_mult=int(m[0]), max_mult=int(m[imax]),
            imax=imax,
            dt_max_first_ms=float((t[imax] - t[0]) / 1e6),
            gap12_ms=float((t[1] - t[0]) / 1e6),
        ))
    r = {k: np.array([x[k] for x in rows]) for k in rows[0]}
    frac_first_is_max = float((r['imax'] == 0).mean())
    dt = r['dt_max_first_ms'][r['imax'] != 0]
    out = dict(
        subrun=f'{run}/{subrun}', n_bursts=len(rows),
        amp_branch=ab,
        frac_first_is_max_mult=frac_first_is_max,
        first_mult_median=float(np.median(r['first_mult'])),
        max_mult_median=float(np.median(r['max_mult'])),
        gap12_ms_median=float(np.median(r['gap12_ms'])),
        dt_max_first_ms_p10=float(np.percentile(dt, 10)) if dt.size else 0.0,
        dt_max_first_ms_med=float(np.median(dt)) if dt.size else 0.0,
        dt_max_first_ms_p90=float(np.percentile(dt, 90)) if dt.size else 0.0,
        n_first_not_max=int((r['imax'] != 0).sum()),
    )
    print(f"{out['subrun']:<28} bursts {out['n_bursts']:>4}  "
          f"first==maxmult {frac_first_is_max:6.1%}  "
          f"first_mult {out['first_mult_median']:>6.0f}  "
          f"max_mult {out['max_mult_median']:>7.0f}  "
          f"gap1->2 {out['gap12_ms_median']:7.3f} ms  "
          f"dt(max-first) med {out['dt_max_first_ms_med']:7.3f} ms "
          f"[p10 {out['dt_max_first_ms_p10']:.3f}, "
          f"p90 {out['dt_max_first_ms_p90']:.3f}]")
    return out


def main():
    res = []
    print(f"{'sub-run':<28} {'':>10}")
    for arg in sys.argv[1:]:
        run, subrun = arg.split('/')
        try:
            res.append(forensics(run, subrun))
        except Exception as e:
            print(f'{arg}: FAILED {e}')
    with open('dream_forensics.json', 'w') as fh:
        json.dump(res, fh, indent=1)
    print('wrote dream_forensics.json')


if __name__ == '__main__':
    main()
