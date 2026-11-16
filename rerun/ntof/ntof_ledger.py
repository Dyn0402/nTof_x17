#!/usr/bin/env python3
"""ntof_ledger.py -- the n_TOF pass-1 ledger (NTOF_PASS_PLAN P6): one canonical output per (registry row, arm), and
the checks that it is the output the plan says it is.  Run on lxplus (reads EOS), through the lock venv:

    ~/bin/lockpy ntof_ledger.py [--root A] [--canon ledger_canon.json] [--no-yield] [--out DIR]

Rows: every non-excluded row of ntof_registry.csv x arms A B C D.  Canonical output: A/stage2/run_<r>/<sub>/<arm>/
unless ledger_canon.json says otherwise (D run_104: side/run104_D_L1, decided 11-16 in run104/README.md; a re-run of
a set of rows lands there too).  Per (row, arm), from the .meta.json:

  present    .parquet, .candidates.parquet and .meta.json exist; partial false; tags_done == [tag]
  code       ntof_pass.code in the canon's code set (ed0b1162 for the main pass)
  registry   ntof_pass.registry run/subrun/mask_era/tan_max equal the registry row's
  bundle     v_drift = the expected bundle's v; the t0 pin (provenance.ntof_bundle.t0.C) = the expected bundle's
             (B: no pin); mask era of the bundle = the row's.  Expected bundle = make_ntof_jobs.py --per-row's choice
             (or the canon's)
  overrides  reco_config and selection equal the arm's reference (the most common value over the pass), ignoring
             the per-job counters (two_track.n_splits, dead_seed counts per era)
  yield      (unless --no-yield) the fraction of events with x and y ok and quality_ok; flagged when below a third of
             the median of its (arm, run, trigger class) by more than x0.6 / x1.5, on acquisitions of >= 100 events (a
             5-MAD rule flagged 71 rows on 11-16, mostly 1-10 % moves on stable runs).  A flag is a row to look at, not
             a failure.

Writes OUT/ntof_ledger.tsv.gz (one line per row x arm), OUT/NTOF_LEDGER.json (summary, every problem) and prints the
summary.  Exit 1 when any row fails a check other than yield.
"""
import argparse, collections, csv, gzip, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
A = '/eos/experiment/ntof/data/x17/july_beam/Analysis_pass1'
ARMS = 'ABCD'
CLASS = {'cosmic': 'cos', 'production': 'beam'}
IGNORE = {('reco_config', 'two_track', 'n_splits'), ('selection', 'dead_seed', 'x'), ('selection', 'dead_seed', 'y')}


def load_canon(path):
    c = json.load(open(path)) if path and os.path.exists(path) else {}
    c.setdefault('code', ['ed0b1162'])
    c.setdefault('rules', [])
    return c


def rule_for(canon, r, arm):
    """The last matching canon rule: {arm, runs?, lat_epoch?, rclass?, dir, bundle?, code?}."""
    hit = None
    for u in canon['rules']:
        if u['arm'] != arm:
            continue
        if 'runs' in u and int(r['run']) not in u['runs']:
            continue
        if 'lat_epoch' in u and r['lat_epoch'] != u['lat_epoch']:
            continue
        if 'rclass' in u and r['rclass'] != u['rclass']:
            continue
        hit = u
    return hit


def expected_bundle(r, arm):
    era = r['mask_era'].split('_')[0]
    lc = CLASS[r['rclass']]
    if arm == 'D' and r['lat_epoch'] == 'L1':
        return f'{arm}_{era}_{lc}_L1'
    return f'{arm}_{era}_{lc}'


def flat(o, p=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, p + (k,))
    else:
        yield p, json.dumps(o, sort_keys=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=A)
    ap.add_argument('--registry', default=os.path.join(HERE, 'ntof_registry.csv'))
    ap.add_argument('--canon', default=os.path.join(HERE, 'ledger_canon.json'))
    ap.add_argument('--no-yield', action='store_true')
    ap.add_argument('--out', default='.')
    a = ap.parse_args()
    canon = load_canon(a.canon)
    reg = [r for r in csv.DictReader(open(a.registry)) if r['excluded'] != 'True']
    bcache = {}

    def bundle_info(path):
        if path not in bcache:
            b = json.load(open(f'{path}/bundle.json'))
            t0 = b.get('provenance', {}).get('ntof_bundle', {}).get('t0')
            bcache[path] = dict(v=b.get('v_drift'), t0=t0.get('C') if isinstance(t0, dict) else t0,
                                era=b.get('conditions', {}).get('mask_era'))
        return bcache[path]

    rows, metas = [], []
    for r in reg:
        for arm in ARMS:
            u = rule_for(canon, r, arm)
            d = f"{a.root}/{u['dir']}" if u else f'{a.root}/stage2'
            d = f"{d}/run_{r['run']}/{r['subrun']}/{arm}"
            bname = (u or {}).get('bundle') or expected_bundle(r, arm)
            bpath = bname if bname.startswith('/') else f'{a.root}/bundles/{bname}'
            codes = set((u or {}).get('code') or canon['code'])
            name = f"run_{r['run']}_{r['subrun']}_{r['tag']}_{arm}"
            row = dict(run=int(r['run']), subrun=r['subrun'], tag=r['tag'], arm=arm, rclass=r['rclass'],
                       lat_epoch=r['lat_epoch'], dir=d, bundle=os.path.basename(bpath), problems=[])
            P = row['problems']
            m = None
            for ext in ('.parquet', '.candidates.parquet', '.meta.json'):
                if not os.path.exists(f'{d}/{name}{ext}'):
                    P.append(f'missing{ext}')
            if os.path.exists(f'{d}/{name}.meta.json'):
                try:
                    m = json.load(open(f'{d}/{name}.meta.json'))
                except Exception as e:
                    P.append(f'meta unreadable: {e}')
            if m is not None:
                np_ = m.get('ntof_pass', {})
                rg = np_.get('registry', {})
                if m.get('partial') is not False:
                    P.append('partial')
                if m.get('tags_done') != [r['tag']]:
                    P.append(f"tags_done {m.get('tags_done')}")
                if np_.get('code') not in codes:
                    P.append(f"code {np_.get('code')}")
                for k in ('subrun', 'mask_era', 'tan_max'):
                    if str(rg.get(k)) != str(r[k]):
                        P.append(f'registry {k} {rg.get(k)} != {r[k]}')
                if str(rg.get('run')) != r['run']:
                    P.append(f"registry run {rg.get('run')}")
                bi = bundle_info(bpath)
                mb = m.get('bundle', {})
                if mb.get('v_drift') != bi['v']:
                    P.append(f"v {mb.get('v_drift')} != bundle {bi['v']}")
                mt0 = mb.get('provenance', {}).get('ntof_bundle', {}).get('t0')
                mt0 = mt0.get('C') if isinstance(mt0, dict) else mt0
                if mt0 != bi['t0']:
                    P.append(f't0 pin {mt0} != bundle {bi["t0"]}')
                if mb.get('conditions', {}).get('mask_era') != r['mask_era']:
                    P.append(f"bundle mask_era {mb.get('conditions', {}).get('mask_era')}")
                row['n_events'] = m.get('n_events')
                row['t0x'] = mt0.get('x') if isinstance(mt0, dict) else mt0
                row['t0y'] = mt0.get('y') if isinstance(mt0, dict) else mt0
                row['code'] = np_.get('code')
                metas.append((row, {p: v for p, v in flat({'reco_config': m.get('reco_config', {}),
                                                           'selection': m.get('selection', {})}) if p not in IGNORE}))
            rows.append(row)
        if len(rows) % 4000 == 0:
            print(f'{len(rows):,} rows', file=sys.stderr, flush=True)

    # overrides: every reco_config/selection key against the arm's most common value
    ref = {}
    for arm in ARMS:
        cnt = collections.defaultdict(collections.Counter)
        for row, f in metas:
            if row['arm'] == arm:
                for k, v in f.items():
                    cnt[k][v] += 1
        ref[arm] = {k: c.most_common(1)[0][0] for k, c in cnt.items()}
    for row, f in metas:
        R = ref[row['arm']]
        diff = sorted('.'.join(k) for k in set(R) | set(f) if f.get(k) != R.get(k))
        if diff:
            row['problems'].append('override ' + ','.join(diff))

    if not a.no_yield:
        import pandas as pd, numpy as np
        for i, row in enumerate(rows):
            f = f"{row['dir']}/run_{row['run']}_{row['subrun']}_{row['tag']}_{row['arm']}.parquet"
            try:
                e = pd.read_parquet(f, columns=['x_ok', 'y_ok', 'x_quality_ok', 'y_quality_ok'])
                row['yield'] = float((e.x_ok.astype(bool) & e.y_ok.astype(bool) & e.x_quality_ok.astype(bool)
                                      & e.y_quality_ok.astype(bool)).mean()) if len(e) else 0.0
            except Exception:
                row['yield'] = None
            if i % 2000 == 0:
                print(f'yield {i:,}/{len(rows):,}', file=sys.stderr, flush=True)
        df = pd.DataFrame([r for r in rows if r.get('yield') is not None])
        for (arm, run, rc), g in df.groupby(['arm', 'run', 'rclass']):
            med = g['yield'].median()
            rel = g['yield'] / med if med > 0 else g['yield'] * 0 + 1
            flag = ((rel < 0.6) | (rel > 1.5)) & (g['n_events'].fillna(0) >= 100)
            for ix in g.index[flag]:
                rr = df.loc[ix]
                for row in rows:
                    if (row['run'], row['subrun'], row['tag'], row['arm']) == (rr.run, rr.subrun, rr.tag, rr.arm):
                        row['yield_flag'] = f'yield {rr["yield"]:.3f} vs median {med:.3f} ({arm} run_{run} {rc})'

    os.makedirs(a.out, exist_ok=True)
    cols = ['run', 'subrun', 'tag', 'arm', 'rclass', 'lat_epoch', 'bundle', 'code', 'n_events', 't0x', 't0y', 'yield',
            'dir', 'problems', 'yield_flag']
    with gzip.open(f'{a.out}/ntof_ledger.tsv.gz', 'wt') as fo:
        fo.write('\t'.join(cols) + '\n')
        for r in rows:
            fo.write('\t'.join('|'.join(r[c]) if c == 'problems' else str(r.get(c, '')) for c in cols) + '\n')
    bad = [r for r in rows if r['problems']]
    kinds = collections.Counter(p.split(' ')[0] for r in bad for p in r['problems'])
    flags = [r for r in rows if r.get('yield_flag')]
    S = dict(n_rows=len(reg), n_outputs=len(rows), n_bad=len(bad), problem_kinds=dict(kinds),
             n_yield_flags=len(flags), canon=canon,
             reference={arm: {'.'.join(k): v for k, v in R.items()} for arm, R in ref.items()},
             per_dir=dict(collections.Counter(os.path.relpath(r['dir'], a.root).split('/run_')[0] + f" {r['arm']}"
                                              for r in rows)),
             problems=[{k: r[k] for k in ('run', 'subrun', 'tag', 'arm', 'problems')} for r in bad][:2000],
             yield_flags=[{k: r.get(k) for k in ('run', 'subrun', 'tag', 'arm', 'yield_flag')} for r in flags])
    json.dump(S, open(f'{a.out}/NTOF_LEDGER.json', 'w'), indent=1, default=str)
    print(json.dumps({k: S[k] for k in ('n_rows', 'n_outputs', 'n_bad', 'problem_kinds', 'n_yield_flags', 'per_dir')},
                     indent=1))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
