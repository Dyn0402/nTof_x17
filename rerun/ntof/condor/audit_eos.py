#!/usr/bin/env python3
"""audit_eos.py {inputs,outputs} -- read-only completeness audit of the n_TOF pass files on EOS.  Run on lxplus.

inputs:  every decoded_root_fa80c06 FEU file against its re-decode record (bytes) and its own ROOT header (fEND), and the
         rr1_cm0 per-FEU hits files (fEND; a hits file without a usable size is listed).  Found 5 cut files on 2026-11-08.
outputs: every stage-2 job with a .meta.json: the JSON parses and each parquet ends in the PAR1 footer magic.

Prints one line per problem and a summary; exit 1 if any.  A file's EOS checksum proves nothing here: EOS checksums what
arrived, so a cut file is "healthy".  A short acquisition is legitimate (small but complete).
"""
import glob
import json
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

EOS = '/eos/experiment/ntof/data/x17/july_beam'
RUNS = f'{EOS}/runs'
RECORDS = f'{EOS}/Analysis_pass1/checks/redecode'
STAGE2 = f'{EOS}/Analysis_pass1/stage2'
SIDE = f'{EOS}/Analysis_pass1/side'
HOME = os.path.expanduser('~/x17rerun/ntof')


def root_size_end(path):
    size = os.stat(path).st_size
    with open(path, 'rb') as f:
        h = f.read(64)
    if h[:4] != b'root':
        return size, None
    version = struct.unpack('>i', h[4:8])[0]
    return size, struct.unpack('>q', h[16:24])[0] if version >= 1000000 else struct.unpack('>i', h[12:16])[0]


def check_input(item):
    run, sub, tag, feu = item
    d = f'{RUNS}/run_{run}/{sub}'
    base = f'Mx17_{sub}_datrun_{tag}_{feu}'
    label = f'{run} {sub} {tag} {feu}'
    out = []
    try:
        rec = json.load(open(f'{RECORDS}/{run}_{sub}_{tag}_f{feu}.json'))
    except (OSError, ValueError) as e:
        return [f'NORECORD {label} {e}']
    if 'no fdf' in str(rec.get('error', '')):
        return []
    try:
        size, end = root_size_end(f'{d}/decoded_root_fa80c06/{base}.root')
        if size != end or size != rec.get('bytes'):
            out.append(f'DECODED {label} size={size} header={end} record={rec.get("bytes")}')
    except OSError as e:
        out.append(f'DECODED_MISSING {label} {e}')
    try:
        size, end = root_size_end(f'{d}/hits_root_rr1_cm0/{base}_hits.root')
        if size != end:
            out.append(f'HITS {label} size={size} header={end}')
    except OSError as e:
        out.append(f'HITS_MISSING {label} {e}')
    return out


def check_output(meta):
    out = []
    try:
        json.load(open(meta))
    except (OSError, ValueError) as e:
        out.append(f'META {meta} {e}')
    stem = meta[:-len('.meta.json')]
    parquets = glob.glob(stem + '*.parquet')
    if stem + '.parquet' not in parquets:
        out.append(f'NOPARQUET {stem}.parquet')
    for p in parquets:
        try:
            with open(p, 'rb') as f:
                f.seek(-4, os.SEEK_END)
                if f.read(4) != b'PAR1':
                    out.append(f'PARQUET_CUT {p} size={os.stat(p).st_size}')
        except OSError as e:
            out.append(f'PARQUET_BAD {p} {e}')
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'inputs':
        items = [(*line.split(), f'{i:02d}') for line in open(f'{HOME}/redecode/jobs.txt') if len(line.split()) == 3
                 for i in range(1, 9)]
        fn = check_input
    elif mode == 'outputs':
        items = glob.glob(f'{STAGE2}/run_*/*/*/*.meta.json') + glob.glob(f'{SIDE}/*/run_*/*/*/*.meta.json')
        fn = check_output
    else:
        sys.exit(__doc__)
    bad = 0
    with ThreadPoolExecutor(24) as ex:
        for rows in ex.map(fn, items):
            for row in rows:
                bad += 1
                print(row, flush=True)
    print(f'checked {len(items)}: {bad} problems', flush=True)
    sys.exit(1 if bad else 0)


main()
