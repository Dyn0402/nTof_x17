#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pss_tail_probe.py -- what IS the plastic late tail, and how wide must we slim?

    python pss_tail_probe.py --bunches 150            # on lxplus, ~5 min
    python pss_tail_probe.py --plot probe.json        # anywhere, from the json

The +-150 ns slim keeps only 77 % of the background-subtracted PSS coincident
yield, and the integral scan (window_yield_wide.json) says the rest runs out to
microseconds. Five integral points cannot tell us WHY, and the two candidate
explanations want different windows:

  ringing / a split pulse   -> echoes at FIXED delays. Discrete peaks. The
                               window must reach past the last echo, and the
                               hits are an artifact to be merged or dropped.
  afterpulsing / late light -> a smooth exponential. The window is a choice of
                               how much of a continuum to keep.

So this slims ONE sub-run over a subset of bunches at +-10 us and looks at the
shape directly. It reuses the production path (`run_segment` with a bunch
subset), so what it measures is what the campaign would write.

Outputs `pss_tail_probe.json`: per-family dt histograms, signal and the +100 us
control, plus the cumulative background-subtracted capture as a function of
window -- which is the number the window decision actually needs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ntof_processing.slim_pipeline import config as C            # noqa: E402

ARMS = ('A', 'B', 'C', 'D')
FAMILIES = ('WAL', 'PSS', 'LIQ')
PROBE_NS = 10_000.0        # how far out to look
BIN_NS = 5.0               # fine enough to resolve a discrete echo


def measure(dream_run, dream_subrun, ntof_run, n_bunches, source, out,
            probe_ns=PROBE_NS):
    from ntof_processing.slim_pipeline.slim import Segment, run_segment
    from ntof_processing.slim_pipeline import segments as SEG

    # Pick the bunches from the middle of the overlap, away from the edges
    # where a sub-run boundary could bias what is available.
    props = [p for p in SEG.for_ntof_run(ntof_run)
             if p.dream_run == dream_run and p.dream_subrun == dream_subrun]
    if not props:
        raise SystemExit(f'no proposal for {dream_run}/{dream_subrun} '
                         f'x {ntof_run}')

    from ntof_dream_merge.bunch_join import dream_event_to_bunch
    ev = dream_event_to_bunch(dream_run, dream_subrun, ntof_run)
    ub = np.unique(ev.loc[ev['BunchNumber'] > 0, 'BunchNumber'])
    mid = len(ub) // 2
    keep = ub[max(0, mid - n_bunches // 2):mid + n_bunches // 2]
    print(f'probing {len(keep)} bunches ({keep.min()}..{keep.max()}) of '
          f'{len(ub)} at +-{probe_ns/1000:g} us')

    seg = Segment(dream_run, dream_subrun, ntof_run,
                  ntof_source=Path(source) if source else None,
                  bunches=keep)
    path, meta = run_segment(seg, out_base=Path(out), slim_ns=probe_ns)
    print(f'wrote {path}')
    return path, meta


def shape(path, out_json, probe_ns=PROBE_NS):
    import uproot
    from ntof_processing.slim_pipeline.clock_qa import _arrays
    with uproot.open(path) as f:
        hits = _arrays(f, 'hits')
        ev = _arrays(f, 'events')
    n_phys = int((ev['is_flash'] == 0).sum())
    det = {t: i for i, t in enumerate(C.SCINT_TREES)}
    edges = np.arange(-probe_ns, probe_ns + BIN_NS, BIN_NS)
    sig = hits['is_control'] == 0

    rec = dict(file=str(path), n_physics=n_phys, bin_ns=BIN_NS,
               lo_ns=float(edges[0]), families={})
    for fam in FAMILIES:
        ids = [det[f'{fam}{a}'] for a in ARMS]
        m = np.isin(hits['det'], ids)
        hs, _ = np.histogram(hits['dt_ns'][m & sig], bins=edges)
        hc, _ = np.histogram(hits['dt_ns'][m & ~sig], bins=edges)
        # Background-subtracted capture vs half-window, the decision number.
        centres = 0.5 * (edges[:-1] + edges[1:])
        excess = hs.astype(float) - hc.astype(float)
        cap = {}
        for W in (25, 50, 100, 150, 250, 500, 1000, 2000, 5000, 10000):
            cap[W] = float(excess[np.abs(centres) <= W].sum())
        tot = cap[10000] or 1.0
        rec['families'][fam] = dict(
            signal=[int(x) for x in hs], control=[int(x) for x in hc],
            n_signal=int((m & sig).sum()), n_control=int((m & ~sig).sum()),
            capture={str(k): v / tot for k, v in cap.items()},
            excess_total=tot)
    Path(out_json).write_text(json.dumps(rec, indent=1))
    print(f'wrote {out_json}')
    return rec


def describe(rec):
    """Print the two things the window decision turns on."""
    print(f'\ncumulative background-subtracted capture vs half-window')
    ws = ['25', '50', '100', '150', '250', '500', '1000', '2000', '5000']
    print(f'{"fam":5}' + ''.join(f'{w:>8}' for w in ws))
    for fam in FAMILIES:
        c = rec['families'][fam]['capture']
        print(f'{fam:5}' + ''.join(f'{c[w]:8.3f}' for w in ws))

    print(f'\nlate-side structure: background-subtracted excess per 50 ns, '
          f'PSS only')
    f = rec['families']['PSS']
    ex = np.array(f['signal'], float) - np.array(f['control'], float)
    lo, b = rec['lo_ns'], rec['bin_ns']
    centres = lo + b * (np.arange(ex.size) + 0.5)
    k = int(50 / b)
    n = (ex.size // k) * k
    cc = centres[:n].reshape(-1, k).mean(axis=1)
    ee = ex[:n].reshape(-1, k).sum(axis=1)
    late = (cc > 0) & (cc < 3000)
    peak = ee[np.abs(cc) < 50].max() or 1.0
    for c_, e_ in zip(cc[late][:40], ee[late][:40]):
        bar = '#' * int(round(40 * max(e_, 0) / peak))
        print(f'  {c_:+7.0f} ns {e_:9,.0f}  {bar}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='run_79')
    ap.add_argument('--subrun', default='stat090_0000')
    ap.add_argument('--ntof', type=int, default=224572)
    ap.add_argument('--bunches', type=int, default=150)
    ap.add_argument('--source', default=None)
    ap.add_argument('--out', default='probe_out')
    ap.add_argument('--json', default='pss_tail_probe.json')
    ap.add_argument('--plot', default=None,
                    help='skip measuring; describe an existing json')
    a = ap.parse_args()

    if a.plot:
        describe(json.loads(Path(a.plot).read_text()))
        return 0
    path, _ = measure(a.run, a.subrun, a.ntof, a.bunches, a.source, a.out)
    describe(shape(path, a.json))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
