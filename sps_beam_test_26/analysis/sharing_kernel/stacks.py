#!/usr/bin/env python3
"""stacks.py -- per-plateau peak-aligned neighbour stacks, with the per-event
matrices kept so downstream code can BOOTSTRAP.

The existing d4_kernel_fit_raw*.npz carry only the mean stacks, which is enough
to see a kernel shape but not to put an error bar on it.  This rebuilds the
same clean selection (robust_waveforms.build_clean: oscillating channels out,
pre-window gate, per-event baseline, leading strip as centre) and writes the
peak-aligned per-event traces themselves, normalised to the event's central
peak, for d = 0, +-1, +-2, +-3.

All THREE run_71 RAW drift plateaus are built -- 700 V (243 V/cm) as well as
450 and 275.  The 700 V block is the best-populated of the three; the earlier
deconvolution used only two fields because it consumed the pre-made npz.

    ../../../.venv/bin/python stacks.py
writes stacks_run71_raw.npz
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.dirname(HERE)
sys.path.insert(0, ANA)

import datasets                                          # noqa: E402
from robust_waveforms import build_clean                 # noqa: E402

SNS = 60.0
NREL = 30                    # +-30 samples about the central peak
DS = 'run71_raw'
OFFSETS = (0, 1, -1, 2, -2, 3, -3)


def build(dataset=DS, q0=(400.0, 3000.0)):
    D = datasets.get(dataset)
    wf = D['stage'] + f'wf_{dataset}_det4only.npz'
    C = build_clean(wf, D, q0[0], q0[1])
    nsmp, nal = C.nsmp, 2 * NREL + 1
    out = {'t_rel': (np.arange(nal) - NREL) * SNS}
    meta = []
    for v in ('x', 'y'):
        isv = C.t_view == v
        cmap = C.cmap[v]
        for lab, *_ in D['plateaus']:
            evsel = np.flatnonzero((C.plateau == lab) & (cmap >= 0))
            if len(evsel) < 50:
                continue
            # the event list is the SAME for every offset, so a bootstrap can
            # resample events and pick the matching rows in each d-matrix.
            ev_ok = np.zeros(C.n_ev, bool)
            ev_ok[evsel] = True
            cen = cmap[evsel]
            q0_ev = C.peak_amp[cen]
            pk_ev = C.peak_smp[cen]
            row_of_ev = np.full(C.n_ev, -1, np.int64)
            row_of_ev[evsel] = np.arange(len(evsel))
            meta.append((lab, v, len(evsel)))
            out[f'q0_{lab}_{v}'] = q0_ev
            out[f'pk_{lab}_{v}'] = pk_ev * SNS
            for dd in OFFSETS:
                sel = np.flatnonzero(isv & (C.t_d == dd) & ev_ok[C.t_ev])
                # one row per event; missing events stay NaN
                A = np.full((len(evsel), nal), np.nan, np.float32)
                if len(sel):
                    r = row_of_ev[C.t_ev[sel]]
                    cols = pk_ev[r][:, None] + (np.arange(nal) - NREL)[None, :]
                    ok = (cols >= 0) & (cols < nsmp)
                    src = np.broadcast_to(sel[:, None], cols.shape)
                    vals = np.full(cols.shape, np.nan, np.float32)
                    vals[ok] = C.trace[src[ok], np.clip(cols, 0, nsmp - 1)[ok]]
                    A[r] = vals / q0_ev[r][:, None]
                out[f'A_{lab}_{v}_{dd:+d}'] = A
    return out, meta


def main():
    out, meta = build()
    p = os.path.join(HERE, f'stacks_{DS}.npz')
    np.savez_compressed(p, **out)
    print(f'\n{"plateau":>8} {"view":>4} {"events":>7}')
    for lab, v, n in meta:
        print(f'{lab:>8} {v:>4} {n:7d}')
    print(f'\nwrote {p}  ({os.path.getsize(p) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
