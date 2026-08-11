#!/usr/bin/env python3
"""Run-by-run ledger: what n_TOF's official processing has, and what we have.

Runs on lxplus, where the files are.  For every run of the X17 EAR2 2026
campaign it records

  * the official state  -- MERGED / MERGE_EMPTY / PARTIALS_ONLY / IN_FLIGHT /
    RAW_ONLY / NOTHING, with partial counts and volumes;
  * the recipe the official product was actually made with, read out of its own
    `history_<run>.root` (normalised: path prefixes and the header file name
    dropped, since those differ without carrying physics);
  * whether we processed the run ourselves, under which production
    (`prod_v11`, `prod_v12`, or one of the 224572 variant studies), with how
    many partials and which recipe.

IN_FLIGHT is the state that the 08-10 inventory could not express: a
`completed/<run>/` directory that holds fewer partials than it did, with fresh
timestamps -- n_TOF wiped it and is reprocessing the run right now.  Treating
that as NOTHING would read as data loss.

Usage:
    python3 -u official_ledger.py [--csv=ledger.csv] [--json=ledger.json]
"""
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

DAQ = Path('/eos/experiment/ntof/DAQ/2026/EAR2/X17_measurement')
DONE = Path('/eos/experiment/ntof/processing/official/done')
COMPLETED = Path('/eos/experiment/ntof/processing/official/completed')
REPROC = Path('/eos/experiment/ntof/data/x17/reproc')

# a completed/ dir touched this recently is being written, not abandoned
FRESH_S = 6 * 3600


def norm_history(text):
    """Recipe fingerprint: drop absolute paths and the header file name."""
    out = []
    for i, line in enumerate(text.splitlines()):
        line = line.rstrip()
        if not line:
            continue
        if i == 0:                      # UserInput_<...>.h
            continue
        line = re.sub(r'(/[\w.\-]+)+/([\w.\-]+)', r'\2', line)
        out.append(line)
    return '\n'.join(out)


def read_history(path):
    """(header_name, variant_tag, normalised_md5) or None."""
    import ROOT
    ROOT.gROOT.SetBatch(True)
    f = ROOT.TFile.Open(str(path))
    if not f or f.IsZombie():
        return None
    o = f.Get('history')
    if not o:
        f.Close()
        return None
    s = o.GetString().Data()
    f.Close()
    lines = [l for l in s.splitlines() if l.strip()]
    header = lines[0].strip() if lines else '?'
    m = re.search(r'variant\s+(\S+)', s)
    return header, (m.group(1) if m else ''), \
        hashlib.md5(norm_history(s).encode()).hexdigest()[:12]


def dir_stat(d, pattern):
    """(n files, bytes) for entries of `d` whose name matches `pattern`."""
    try:
        names = os.listdir(d)
    except OSError:
        return 0, 0
    n = tot = 0
    for name in names:
        if pattern.match(name):
            try:
                tot += os.stat(os.path.join(d, name)).st_size
            except OSError:
                pass
            n += 1
    return n, tot


# run 224572 exists under all thirteen variant-study directories; the one that
# counts is the production configuration, so name the preference explicitly
# rather than letting `sorted()` pick v10 because it sorts first.
PROD_RANK = ['prod_v12', 'prod_v11', 'v12_liqpileup']


def our_products():
    """{run: (production, n_partials, bytes, history_path, [all productions])}."""
    all_of = {}
    for prod in sorted(REPROC.iterdir()):
        if not prod.is_dir():
            continue
        # production runs are <prod>/<run>/completed/<run>/, variant studies of
        # the single reference run are <variant>/completed/<run>/
        for rd in list(prod.glob('*/completed/*')) + list(prod.glob('completed/*')):
            if not rd.is_dir() or not rd.name.isdigit():
                continue
            run = int(rd.name)
            n, b = dir_stat(rd, re.compile(rf'^run{run}_\d+\.root$'))
            if n:
                all_of.setdefault(run, []).append(
                    (prod.name, n, b, rd / f'history_{run}.root'))

    found = {}
    for run, entries in all_of.items():
        entries.sort(key=lambda e: (PROD_RANK.index(e[0])
                                    if e[0] in PROD_RANK else len(PROD_RANK),
                                    e[0]))
        best = entries[0]
        found[run] = (*best, [e[0] for e in entries])
    return found


def main():
    out_csv = out_json = None
    for a in sys.argv[1:]:
        if a.startswith('--csv='):
            out_csv = a.split('=', 1)[1]
        elif a.startswith('--json='):
            out_json = a.split('=', 1)[1]

    runs = sorted(int(d.name) for d in DAQ.iterdir()
                  if d.is_dir() and d.name.isdigit())
    print(f'{len(runs)} campaign runs staged under {DAQ}', flush=True)

    ours = our_products()
    print(f'{len(ours)} runs processed by us under {REPROC}', flush=True)

    now = time.time()
    recipe_cache = {}
    rows = []
    for run in runs:
        pat = re.compile(rf'^run{run}_\d+\.root$')
        raw_n, raw_b = dir_stat(DAQ / run.__str__() / 'stream1', re.compile('.'))

        cdir = COMPLETED / str(run)
        parts, parts_b = dir_stat(cdir, pat)
        cmtime = cdir.stat().st_mtime if cdir.is_dir() else 0

        merged = DONE / f'run{run}.root'
        mb = merged.stat().st_size if merged.exists() else -1

        if mb > 0:
            state = 'MERGED'
        elif mb == 0:
            state = 'MERGE_EMPTY' if parts else 'MERGE_EMPTY_NOPARTS'
        elif parts:
            state = 'PARTIALS_ONLY'
        elif raw_n:
            state = 'RAW_ONLY'
        else:
            state = 'NOTHING'
        # a completed/ dir that exists but is short and freshly written is a
        # reprocessing in progress, not an absence
        if cdir.is_dir() and (now - cmtime) < FRESH_S and state != 'MERGED':
            state = 'IN_FLIGHT'

        off_recipe = off_header = ''
        hp = cdir / f'history_{run}.root'
        if hp.exists():
            key = str(hp)
            if key not in recipe_cache:
                recipe_cache[key] = read_history(hp)
            r = recipe_cache[key]
            if r:
                off_header, _, off_recipe = r

        oprod = oparts = ob = ''
        our_recipe = our_variant = our_all = ''
        if run in ours:
            oprod, oparts, ob, ohp, prods = ours[run]
            our_all = ' '.join(prods)
            if ohp.exists():
                key = str(ohp)
                if key not in recipe_cache:
                    recipe_cache[key] = read_history(ohp)
                r = recipe_cache[key]
                if r:
                    _, our_variant, our_recipe = r

        rows.append(dict(
            run=run, raw_files=raw_n, raw_GB=round(raw_b / 2**30, 1),
            off_state=state, off_parts=parts, off_GB=round(parts_b / 2**30, 1),
            off_merged_bytes=mb, off_header=off_header, off_recipe=off_recipe,
            ours_prod=oprod, ours_parts=oparts,
            ours_GB=round(ob / 2**30, 1) if ob != '' else '',
            ours_variant=our_variant, ours_recipe=our_recipe,
            ours_all_productions=our_all,
        ))
        print(f'{run} {state:14s} parts={parts:3d} merged={mb:>13} '
              f'off={off_recipe} ours={oprod or "-"}/{our_recipe}', flush=True)

    if out_csv:
        with open(out_csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {out_csv}')
    if out_json:
        Path(out_json).write_text(json.dumps(rows, indent=1))
        print(f'wrote {out_json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
