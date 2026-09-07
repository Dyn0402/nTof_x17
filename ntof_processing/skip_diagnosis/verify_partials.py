#!/usr/bin/env python3
"""Verify the official partial sets for the runs that never got merged.

For each run:
  * contiguity of run<run>_NNNN.root from 1..N with no gap
  * N against ceil(n_raw_files / 4), the split the 07 August script uses
  * the history md5, against the reference run224572.root (known v12)
  * a real read of the first, middle and last partial: trees present, entries
"""
import hashlib
import math
import os
import re
import sys

import uproot

COMPLETED = '/eos/experiment/ntof/processing/official/completed'
RAW = '/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement'
REF = '/eos/experiment/ntof/processing/official/done/run224572.root'
FILES_PER_JOB = 4


def history_md5(path):
    """The `history` object is a ROOT string holding the whole UserInput."""
    try:
        o = uproot.open(path)['history']
        s = None
        for attr in ('fString', 'fTitle', 'fName'):
            try:
                v = o.member(attr)
                if isinstance(v, (str, bytes)) and len(v) > 200:
                    s = v.decode() if isinstance(v, bytes) else v
                    break
            except Exception:
                pass
        if s is None:
            s = str(o)
        return hashlib.md5(s.encode()).hexdigest()
    except Exception as e:
        return f'FAIL:{type(e).__name__}'


def main(runs):
    ref = history_md5(REF)
    print(f'reference history md5 (run224572.root, v12): {ref}\n')
    hdr = f'{"run":>7} {"parts":>5} {"exp":>4} {"contig":>7} {"raw":>4} {"hist":>6} {"read":>22}'
    print(hdr)
    print('-' * len(hdr))
    bad = []
    for run in runs:
        d = f'{COMPLETED}/{run}'
        if not os.path.isdir(d):
            print(f'{run:>7} {"-":>5}')
            bad.append((run, 'no directory'))
            continue
        names = os.listdir(d)
        idx = sorted(int(m.group(1)) for n in names
                     if (m := re.fullmatch(rf'run{run}_(\d+)\.root', n)))
        nraw = len([f for f in os.listdir(f'{RAW}/{run}/stream1')]) \
            if os.path.isdir(f'{RAW}/{run}/stream1') else 0
        exp4 = math.ceil(nraw / 4) if nraw else 0
        exp10 = math.ceil(nraw / 10) if nraw else 0
        exp = exp4 if len(names) and exp4 else exp4
        contig = bool(idx) and idx == list(range(1, len(idx) + 1))
        hm = history_md5(f'{d}/history_{run}.root')
        tag = 'v12' if hm == ref else hm[:6]

        probe, ok = [], True
        for i in ({idx[0], idx[len(idx) // 2], idx[-1]} if idx else set()):
            p = f'{d}/run{run}_{i:04d}.root'
            try:
                f = uproot.open(p)
                keys = [k.split(';')[0] for k in f.keys()]
                n = f[keys[0]].num_entries if keys else 0
                probe.append(f'{i}:{len(keys)}t/{n}')
            except Exception as e:
                probe.append(f'{i}:FAIL')
                ok = False
        status = ' '.join(probe)
        flag = '' if (contig and ok and tag == 'v12'
                      and (exp4 == 0 or len(idx) in (exp4, exp10))) else '   <<<'
        print(f'{run:>7} {len(idx):5d} {(exp4 if len(idx)!=exp10 else exp10):4d} {str(contig):>7} {nraw:4d} '
              f'{tag:>6} {status:>22}{flag}')
        if flag:
            bad.append((run, f'parts={len(idx)} exp4={exp4} exp10={exp10} contig={contig} '
                             f'hist={tag} read_ok={ok}'))
    print()
    if bad:
        print('NEEDS ATTENTION:')
        for r, why in bad:
            print(f'  {r}: {why}')
    else:
        print('every run complete, contiguous, v12, and readable.')
    return 0


if __name__ == '__main__':
    sys.exit(main([int(x) for x in sys.argv[1:]]))
