#!/usr/bin/env python3
"""Which X17 runs are finished and will not change, and which are still moving.

Run this on lxplus. It is a pure metadata scan -- no ROOT files are opened -- so
it takes a couple of minutes over all 445 runs and can be re-run whenever the
picture needs refreshing.

THE CRITERION IS THE MERGED FILE, and that is an observation rather than a
convention. Through 2026-08-11 every run with a non-empty `done/run<N>.root` was
left completely untouched, while every run whose partials were complete but
whose merge had not produced a file was **wiped and reconstructed again** --
224454, 224652, 224653, 224666 and 224671 each lost a complete 39-54 partial set
that way in a single afternoon. So a merged file is what marks a run as settled;
a complete partial set on its own does not.

Two qualifications the scan reports separately:

  * a run can be complete, unmerged and still stable -- 224405, 224451-224453 and
    224667 have not been touched since 08-05/08-07. They are usable; they are
    simply not blessed. 224405 and 224667 additionally carry a ZERO-BYTE merged
    file, which `exists()` will happily return true for.
  * a zero-byte merged file can also be a merge IN FLIGHT. `done/run224637.root`
    was 0 bytes at 18:41 on 08-11 and 36 GB at 18:43. Size alone does not
    separate the two; only a second look does.

Usage:
    python3 -u settled_runs.py [--out=settled_runs.txt] [--stable-hours=24]
"""
import os
import sys
import time
from datetime import datetime
from pathlib import Path

DAQ = Path('/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement')
COMPLETED = Path('/eos/experiment/ntof/processing/official/completed')
DONE = Path('/eos/experiment/ntof/processing/official/done')


def ranges(rs):
    """[1,2,3,7] -> ['1-3', '7']"""
    rs = sorted(rs)
    if not rs:
        return []
    out, s, prev = [], rs[0], rs[0]
    for r in rs[1:]:
        if r == prev + 1:
            prev = r
            continue
        out.append(str(s) if s == prev else f'{s}-{prev}')
        s = prev = r
    out.append(str(s) if s == prev else f'{s}-{prev}')
    return out


def main():
    out_path, stable_h = None, 24
    for a in sys.argv[1:]:
        if a.startswith('--out='):
            out_path = a.split('=', 1)[1]
        elif a.startswith('--stable-hours='):
            stable_h = float(a.split('=', 1)[1])

    now = time.time()
    finished, stable_unmerged, moving, zero_byte = [], [], [], []
    for d in sorted(DAQ.iterdir()):
        if not d.is_dir() or not d.name.isdigit():
            continue
        run = int(d.name)
        cdir = COMPLETED / d.name
        try:
            parts = sum(1 for n in os.listdir(cdir)
                        if n.startswith(f'run{run}_') and n.endswith('.root'))
            mtime = cdir.stat().st_mtime
        except OSError:
            parts, mtime = 0, 0
        m = DONE / f'run{run}.root'
        try:
            msize = m.stat().st_size
        except OSError:
            msize = -1

        if msize > 0:
            finished.append(run)
        elif parts and (now - mtime) > stable_h * 3600:
            stable_unmerged.append(run)
            if msize == 0:
                zero_byte.append(run)
        else:
            moving.append(run)
            if msize == 0:
                zero_byte.append(run)

    L = [
        '# n_TOF X17 EAR2 2026 -- runs that are finished and not moving',
        f'# generated {datetime.now():%Y-%m-%d %H:%M} by campaign_qa/settled_runs.py',
        '# criterion: a non-empty merged file in official/done/. Every such run has',
        '# been left untouched, while complete-but-unmerged runs were wiped and',
        '# reconstructed again -- see the module docstring.',
        '',
        f'## FINISHED -- {len(finished)} runs',
    ]
    r = ranges(finished)
    L += [', '.join(r[i:i + 8]) for i in range(0, len(r), 8)]
    L += [
        '',
        f'## STABLE BUT UNMERGED -- {len(stable_unmerged)} runs',
        f'# complete partial sets, untouched for more than {stable_h:g} h. Usable,',
        '# but not blessed, and a re-run would cost them their partials for hours.',
        ', '.join(ranges(stable_unmerged)) or '(none)',
        '',
        f'## MOVING -- {len(moving)} runs. Do not read these.',
        ', '.join(ranges(moving)) or '(none)',
        '',
        f'## zero-byte done/run<N>.root -- {len(zero_byte)}',
        '# either a failed merge or one in flight; look twice before concluding.',
        ', '.join(ranges(zero_byte)) or '(none)',
    ]
    text = '\n'.join(L) + '\n'
    print(text)
    if out_path:
        Path(out_path).write_text(text)
        print(f'wrote {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
