#!/usr/bin/env python3
"""pass1_ledger.py [--root A] [--out PASS1_LEDGER.md] -- the pass's completion ledger (PASS_PLAN §3: no row missing, STAGES
show no overrides, code hash = the archive).  Run on lxplus (reads EOS).

Per non-excluded pass_registry row: the pass table (wft/events.parquet + meta), its 02/03/04 outputs, the bundle it used,
and every STAGES/*.json of the row: code commit and diagnostic overrides.  Per sps_registry row: events.parquet + meta
(code, bundle).  Expected code: 6a0bb1f everywhere, d34cf25 for the three run_61 rows whose clock was fixed.
"""
import argparse
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
CODE = {'6a0bb1f'}
CODE_SPS_FIX = {'run61_rot15_ladder__d210', 'run61_rot15_ladder__d140', 'run61_rot15_ladder__d070'}


def bench(root):
    rows, bad = [], []
    for r in csv.DictReader(open(os.path.join(HERE, 'pass_registry.csv'))):
        if r['cls'] == 'excluded':
            continue
        d = os.path.join(root, r['run'], r['subrun'], r['det'])
        w = os.path.join(d, 'wft')
        have = dict(table=os.path.exists(os.path.join(w, 'events.parquet')),
                    meta=os.path.exists(os.path.join(w, 'events.meta.json')),
                    eff=bool(glob.glob(os.path.join(w, '*efficiency*')) or glob.glob(os.path.join(w, 'eff*'))),
                    angles=bool(glob.glob(os.path.join(w, 'angles*')) or glob.glob(os.path.join(w, '*angular*'))),
                    maps=bool(glob.glob(os.path.join(w, 'maps*')) or glob.glob(os.path.join(w, '*map*'))))
        n = None; bundle = None
        if have['meta']:
            m = json.load(open(os.path.join(w, 'events.meta.json')))
            n, bundle = m.get('n_events'), m.get('calibration')
        codes, overrides = set(), []
        for s in glob.glob(os.path.join(d, 'STAGES', '*.json')):
            try:
                j = json.load(open(s))
            except Exception:
                continue
            codes.add(str(j.get('code') or '?'))
            if j.get('diagnostic_overrides'):
                overrides.append(os.path.basename(s))
        row = dict(key=r['key'], det=r['det'], cls=r['cls'], n=n, bundle=bundle, codes=sorted(codes), overrides=overrides, **have)
        problems = [k for k in ('table', 'meta', 'eff', 'angles', 'maps') if not have[k]]
        if overrides:
            problems.append('overrides')
        if codes and not codes <= CODE:
            problems.append(f'code {sorted(codes)}')
        row['problems'] = problems
        rows.append(row)
        if problems:
            bad.append(row)
    return rows, bad


def sps(root):
    rows, bad = [], []
    for r in csv.DictReader(open(os.path.join(HERE, 'sps_registry.csv'))):
        d = os.path.join(root, 'sps', r['key'])
        mp = os.path.join(d, 'events.meta.json')
        m = json.load(open(mp)) if os.path.exists(mp) else {}
        want = {'d34cf25'} if r['key'] in CODE_SPS_FIX else CODE
        problems = []
        if not os.path.exists(os.path.join(d, 'events.parquet')):
            problems.append('table')
        if m and m.get('code') not in want:
            problems.append(f"code {m.get('code')}")
        if m and m.get('diagnostic_overrides'):
            problems.append('overrides')
        if m and 'Analysis_pass1' not in str(m.get('bundle')):
            problems.append('bundle not pass1')
        row = dict(key=r['key'], n=m.get('n_events'), code=m.get('code'), problems=problems)
        rows.append(row)
        if problems:
            bad.append(row)
    return rows, bad


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default=A); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    B, Bbad = bench(a.root); S, Sbad = sps(a.root)
    L = [f'# Pass 1 completion ledger', '',
         f'Bench: {len(B)} non-excluded rows, {len(B) - len(Bbad)} complete; SPS: {len(S)} rows, {len(S) - len(Sbad)} complete.',
         f'Expected code: {sorted(CODE)} (SPS run_61 clock rows: d34cf25).', '']
    if Bbad:
        L += ['## Bench rows with problems', '| row | cls | n | problems |', '|---|---|---|---|']
        L += [f"| {r['key']} | {r['cls']} | {r['n']} | {', '.join(r['problems'])} |" for r in Bbad]
    if Sbad:
        L += ['', '## SPS rows with problems', '| row | n | problems |', '|---|---|---|']
        L += [f"| {r['key']} | {r['n']} | {', '.join(r['problems'])} |" for r in Sbad]
    if not Bbad and not Sbad:
        L += ['**No row missing, no overrides, code = the archive.**']
    tot = sum(r['n'] or 0 for r in B)
    L += ['', f'Bench events reconstructed: {tot:,}; SPS events: {sum(r["n"] or 0 for r in S):,}']
    txt = '\n'.join(L) + '\n'
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt)
        json.dump(dict(bench=B, sps=S), open(a.out.replace('.md', '.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
