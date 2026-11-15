#!/usr/bin/env python3
"""pass1_complete.py <job list> [...] -- per job line of an n_TOF pass-1 list, check its four EOS outputs.  Run on lxplus.

A job line is `run subrun tag arm bundle outdir`; its outputs are <outdir>/run_<run>_<subrun>_<tag>_<arm>.{parquet,
candidates.parquet,splits.parquet,meta.json}.  Checks: all four present and non-empty; meta parses, partial false,
tags_done == [tag], ntof_pass.code == the expected code, n_events > 0.  Also checks that the arms of one acquisition
agree on n_events (they read the same triggers).  audit_eos.py outputs checks the files' integrity (PAR1 footers);
this checks that every job of the list has them.  Writes a TSV of (list, run, subrun, tag, arm, n_events, bytes) to
stdout's sibling --tsv and prints problems + a summary.  Read-only.
"""
import argparse, collections, gzip, json, os, sys
from concurrent.futures import ThreadPoolExecutor

SUFFIXES = ('.parquet', '.candidates.parquet', '.splits.parquet', '.meta.json')


def check(item):
    lst, (run, sub, tag, arm, bundle, outdir) = item
    stem = f'{outdir}/run_{run}_{sub}_{tag}_{arm}'
    probs, size = [], 0
    for s in SUFFIXES:
        try:
            n = os.stat(stem + s).st_size
            size += n
            if n == 0:
                probs.append(f'EMPTY {stem}{s}')
        except OSError:
            probs.append(f'MISSING {stem}{s}')
    nev = None
    try:
        m = json.load(open(stem + '.meta.json'))
        nev = m.get('n_events')
        if m.get('partial') is not False:
            probs.append(f'PARTIAL {stem} partial={m.get("partial")}')
        if m.get('tags_done') != [tag]:
            probs.append(f'TAGS {stem} tags_done={m.get("tags_done")}')
        code = (m.get('ntof_pass') or {}).get('code')
        if code != ARGS.code:
            probs.append(f'CODE {stem} code={code}')
        if not nev:
            probs.append(f'NOEVENTS {stem} n_events={nev}')
    except (OSError, ValueError) as e:
        if not any(p.startswith('MISSING') and p.endswith('.meta.json') for p in probs):
            probs.append(f'META {stem} {e}')
    return lst, (run, sub, tag, arm), nev, size, probs


def lines(path):
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt') as f:
        return [l.split()[:6] for l in f if len(l.split()) >= 6]


ap = argparse.ArgumentParser()
ap.add_argument('lists', nargs='+')
ap.add_argument('--code', default='ed0b1162')
ap.add_argument('--tsv', required=True)
ARGS = ap.parse_args()
items = [(os.path.basename(p), l) for p in ARGS.lists for l in lines(p)]
bad, per_acq = 0, collections.defaultdict(dict)
with ThreadPoolExecutor(24) as ex, open(ARGS.tsv, 'w') as tsv:
    tsv.write('list\trun\tsubrun\ttag\tarm\tn_events\tbytes\n')
    for lst, key, nev, size, probs in ex.map(check, items):
        tsv.write('\t'.join([lst, *key, str(nev), str(size)]) + '\n')
        for p in probs:
            bad += 1
            print(p, flush=True)
        if nev is not None:
            per_acq[(lst, *key[:3])][key[3]] = nev
for k, arms in sorted(per_acq.items()):
    if len(set(arms.values())) > 1:
        bad += 1
        print('NEVENTS_DIFFER', *k, json.dumps(arms))
print(f'checked {len(items)} job lines in {len(ARGS.lists)} lists: {bad} problems')
sys.exit(1 if bad else 0)
