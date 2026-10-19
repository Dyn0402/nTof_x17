#!/usr/bin/env python3
"""Whole-run beam state for each processed run, from the `index` tree.

`index` is replicated IN FULL in every partial (one row per bunch of the whole
run), so a single open per run gives the beam state of the entire run rather
than of the partial that happens to be sampled.  This matters because a run with
no protons produces a small, quiet file that looks like a failed processing and
is not -- and because the physics comparison is only meaningful where there was
beam.

Usage:  python beam_state.py <run-dir> [<run-dir> ...]   [--json=out.json]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import uproot

BEAM = 1e12          # protons; below this the pulse carried no usable beam


def main():
    outjson, dirs = None, []
    for a in sys.argv[1:]:
        if a.startswith('--json='):
            outjson = a.split('=', 1)[1]
        else:
            dirs.append(a)

    rows = []
    hdr = (f'{"run":>7} {"bunches":>8} {"beam":>8} {"beam%":>7} {"protons_1e12":>13} '
           f'{"mean_I":>10} {"state":>10}')
    print(hdr)
    print('-' * len(hdr))
    for d in dirs:
        m = re.search(r'(\d{6})', Path(d).name)
        run = int(m.group(1)) if m else -1
        ps = sorted(Path(d).glob('run[0-9]*_[0-9]*.root'),
                    key=lambda q: int(q.stem.split('_')[-1]))
        if not ps:
            print(f'{run:>7}  no partials')
            continue
        try:
            idx = uproot.open(ps[0])['index'].arrays(
                ['BunchNumber', 'PulseIntensity'], library='np')
        except Exception as e:
            print(f'{run:>7}  index unreadable: {type(e).__name__}')
            continue
        I = idx['PulseIntensity']
        good = I > BEAM
        tot = float(np.nansum(I[good])) / 1e12
        frac = float(good.mean()) * 100
        state = 'beam' if frac > 50 else ('mixed' if frac > 1 else 'NO BEAM')
        rows.append({'run': run, 'bunches': int(I.size), 'beam_bunches': int(good.sum()),
                     'beam_pct': frac, 'protons_1e12': tot,
                     'mean_intensity': float(np.nanmean(I[good])) if good.any() else 0.0,
                     'state': state})
        print(f'{run:>7} {I.size:8d} {int(good.sum()):8d} {frac:7.1f} {tot:13.1f} '
              f'{(np.nanmean(I[good]) if good.any() else 0):10.3g} {state:>10}')

    if outjson:
        Path(outjson).write_text(json.dumps(rows, indent=1))
        print(f'\nwrote {outjson}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
