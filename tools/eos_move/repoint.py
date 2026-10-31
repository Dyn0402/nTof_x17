#!/usr/bin/env python3
"""repoint.py [--apply] [--moved LIST] -- point code, submit files, PARAMETERS and READMEs at the n_TOF EOS copies
of what moved out of /eos/user/d/dneff on 2026-10-31 (rerun/EOS_MOVE_2026-10-31.md).

The old paths are FUSE symlinks to the new ones, so docs and result records keep working unchanged; what does
not survive the move is xrootd access (root://eosuser.cern.ch cannot follow a symlink into the experiment
instance), so every code/script reference is rewritten, with root://eosuser -> root://eosexperiment on the same
line.  --moved: a file of old paths that actually moved (only those are rewritten); default = all of MAP.
"""
import argparse
import os
import re
import subprocess

U, X = '/eos/user/d/dneff', '/eos/experiment/ntof/data/x17'
SMALL = ('asym_beam bands calorimetry cloud_basics mx17_hdr_audit_20261016 pairing_sim plane_ratio satblob seedfit sim '
         'vladder x17dig_out scratch test').split()
HELD = ('sps_gasfit', 'sps_junk')            # still in use on 2026-10-31 (not moved); see EOS_MOVE_2026-10-31.md
CODE = ('.py', '.sh', '.sub', '.toml', '.cfg', '.yaml', '.yml', '.dag')


def mapping(moved):
    m = [(f'{U}/x17/sps_run53_det4_check', f'{X}/p2_sps_july/sps_run53_det4_check'),
         (f'{U}/tmp/gx', f'{X}/response_sim/scratch_gx_2026-10'),
         (f'{U}/tmp/d0ar', f'{X}/response_sim/scratch_d0ar_2026-10')]
    m += [(f'{U}/{d}', f'{X}/analysis/user_misc/{d}') for d in SMALL]
    m.append((f'{U}/x17rerun/', f'{X}/cosmic_bench/x17rerun/'))      # per subdir, filtered below
    m.append((f'{U}/x17/', f'{X}/analysis/'))                        # sept26_*, cosmics_fullpass, ... (not reproc)
    return [(o, n) for o, n in m if moved is None or any(p == o.rstrip('/') or p.startswith(o) for p in moved)]


def rewrite(text, M, moved):
    def sub_line(line):
        new = line
        for o, n in M:
            def rep(mt):
                tail = mt.group(1)
                first = tail.split('/')[0] if o.endswith('/') else ''
                if o.endswith('/x17rerun/') and (first in HELD or not first):
                    return mt.group(0)
                if o.endswith('/x17/') and first in ('reproc', ''):
                    return mt.group(0)
                if moved is not None and o.endswith('/'):
                    if (o + first).rstrip('/') not in moved:
                        return mt.group(0)
                return n + tail
            new = re.sub(re.escape(o) + r'([A-Za-z0-9_.\-/${}]*)', rep, new)
        if new != line:
            new = new.replace('root://eosuser.cern.ch/', 'root://eosexperiment.cern.ch/')
        return new
    return '\n'.join(sub_line(l) for l in text.split('\n'))


def targets(root):
    out = subprocess.run(['git', '-C', root, 'ls-files'], capture_output=True, text=True).stdout.split()
    for f in out:
        b = os.path.basename(f)
        if f.endswith(CODE) or f == 'docs/PARAMETERS.md' or b.startswith('README'):
            yield os.path.join(root, f)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--apply', action='store_true')
    p.add_argument('--moved')
    p.add_argument('--root', default='.')
    a = p.parse_args()
    moved = set(l.strip().rstrip('/') for l in open(a.moved)) if a.moved else None
    M = mapping(moved)
    n = 0
    for f in targets(a.root):
        try:
            t = open(f).read()
        except (UnicodeDecodeError, OSError):
            continue
        if U not in t:
            continue
        t2 = rewrite(t, M, moved)
        if t2 != t:
            n += 1
            print(os.path.relpath(f, a.root))
            if a.apply:
                open(f, 'w').write(t2)
    print(f'{n} files {"rewritten" if a.apply else "would change"}')
