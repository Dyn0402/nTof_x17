#!/usr/bin/env python3
"""What the waveform-pull campaign actually produced, read from the products.

    python -m ntof_processing.waveform_pull.fleet_report [--dest <dir>] [--out f]

Every number here comes from the `*_provenance.json` and `*_verify.json` that
travel WITH each product on EOS -- never from the text of a condor log.  That is
not a style preference: parsing pass/fail out of prose produced four confident
wrong numbers in one night on this project, and a job log is also the one
artefact that disappears when a sandbox is reused.

A run is only COMPLETE here if every one of its segments has a product, that
product carries provenance (written last, so its presence means "finished"),
and the closure check on it passed.  Anything else is named and counted.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DEST = '/eos/experiment/ntof/data/x17/wf_pull/out'
SLIM = '/eos/experiment/ntof/data/x17/wf_pull/slim_input'
NAME = re.compile(r'ntof_wf_(run_\d+)_(.+)_(\d+)$')


def scan(dest: Path) -> dict:
    """{(dream_run, subrun, ntof_run): {...}} for everything published."""
    out: dict[tuple, dict] = {}
    for prov in dest.rglob('ntof_wf_*_provenance.json'):
        stem = prov.name[:-len('_provenance.json')]
        m = NAME.match(stem)
        if not m:
            continue
        key = (m.group(1), m.group(2), int(m.group(3)))
        rec: dict = {'stem': stem, 'root': prov.with_name(stem + '.root')}
        try:
            rec['prov'] = json.loads(prov.read_text())
        except Exception as e:                      # truncated mid-copy
            rec['prov'], rec['prov_error'] = None, str(e)
        vf = prov.with_name(stem + '_verify.json')
        if vf.is_file():
            try:
                rec['verify'] = json.loads(vf.read_text())
            except Exception as e:
                rec['verify'], rec['verify_error'] = None, str(e)
        out[key] = rec

    # products with no provenance beside them: a job died mid-write, or the
    # publish was interrupted between the two files.
    for root in dest.rglob('ntof_wf_*.root'):
        m = NAME.match(root.stem)
        if not m:
            continue
        key = (m.group(1), m.group(2), int(m.group(3)))
        if key not in out:
            out[key] = {'stem': root.stem, 'root': root, 'prov': None,
                        'orphan': True}
    return out


def expected(slim: Path) -> set:
    """Every segment that HAS a slim, i.e. every segment we could build."""
    want = set()
    for p in slim.rglob('ntof_hits_*.root'):
        m = re.match(r'ntof_hits_(run_\d+)_(.+)_(\d+)$', p.stem)
        if m:
            want.add((m.group(1), m.group(2), int(m.group(3))))
    return want


def classify(rec: dict) -> str:
    if rec.get('orphan') or rec.get('prov') is None:
        return 'NO_PROVENANCE'
    p = rec['prov']
    if p.get('n_bunches_missing'):
        return 'BUNCHES_MISSING'
    v = rec.get('verify')
    if v is None:
        return 'UNVERIFIED'
    if not str(v.get('status', '')).startswith('PASS'):
        return 'VERIFY_FAIL'
    return 'OK'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dest', type=Path, default=Path(DEST))
    ap.add_argument('--slim', type=Path, default=Path(SLIM))
    ap.add_argument('--out', type=Path, help='also write the report here')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args(argv)

    if not a.dest.is_dir():
        print(f'no such destination: {a.dest}', file=sys.stderr)
        return 1
    got = scan(a.dest)
    want = expected(a.slim) if a.slim.is_dir() else set()

    by = defaultdict(list)
    for k, rec in sorted(got.items()):
        by[classify(rec)].append(k)
    missing = sorted(want - set(got))

    tot_bytes = sum(r['prov'].get('file_bytes', 0) for r in got.values()
                    if r.get('prov'))
    tot_blocks = sum(r['prov'].get('n_blocks', 0) for r in got.values()
                     if r.get('prov'))

    L = [f'waveform-pull fleet report  --  {a.dest}', '']
    if want:
        L.append(f'segments with a slim (buildable) : {len(want)}')
    L += [f'segments published               : {len(got)}',
          f'  of which complete and PASS     : {len(by["OK"])}',
          f'total size                       : {tot_bytes/1e9:.1f} GB',
          f'total blocks                     : {tot_blocks:,}', '']
    for state in ('VERIFY_FAIL', 'BUNCHES_MISSING', 'NO_PROVENANCE', 'UNVERIFIED'):
        if not by[state]:
            continue
        L.append(f'{state} ({len(by[state])}):')
        for k in by[state][:20]:
            rec = got[k]
            det = ''
            if state == 'VERIFY_FAIL' and rec.get('verify'):
                det = '  ' + str(rec['verify'].get('status', ''))[:90]
            elif state == 'BUNCHES_MISSING' and rec.get('prov'):
                det = f"  {rec['prov']['n_bunches_missing']} bunches absent from raw"
            L.append(f'  {k[0]}/{k[1]} x{k[2]}{det}')
        if len(by[state]) > 20:
            L.append(f'  ... and {len(by[state])-20} more')
        L.append('')
    if missing:
        runs = sorted({k[2] for k in missing})
        L += [f'NOT YET PULLED ({len(missing)} segments over {len(runs)} n_TOF runs):',
              '  ' + ' '.join(str(r) for r in runs), '']

    txt = '\n'.join(L)
    print(json.dumps({k: [list(x) for x in v] for k, v in by.items()},
                     indent=2) if a.json else txt)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(txt)
    return 0


if __name__ == '__main__':
    sys.exit(main())
