#!/usr/bin/env python3
"""pass1_ledger.py [--root A] [--out PASS1_LEDGER.md] -- the pass's completion ledger (PASS_PLAN §3: no row missing, STAGES
show no overrides, code hash = the archive).  Run on lxplus (reads EOS).

Per non-excluded pass_registry row: the pass table (wft/events.parquet + meta), its 02/03/04 outputs, the bundle it used,
and every STAGES/*.json of the row: code commit and diagnostic overrides.  Per sps_registry row: events.parquet + meta
(code, bundle, the seed-rule meta, the head-on estimator columns).  Expected code: 6a0bb1f on the bench; 3ac0e79 on every
SPS row (the final SPS re-run of 2026-10-30, rerun/sps_audit/README.md; before it: d34cf25 for three run_61 rows, a72b889
for the RAW run_71 rows).

Since the post phase (2026-10-30, rerun/pass1_review/REVIEW_2026-10-30.md P1): the latest 01 (rows that align themselves),
02, 03 and 04 stage of a row must be on CODE_POST, and two checks the presence ledger could not make:
- alignment: z_x/z_y of the row's alignment within ALIGN_TOL_MM of the median over its night's self-aligned rows of the
  chamber (nights with >= 3 of them), and |z_x - z_y| <= ALIGN_TOL_MM;
- efficiency: reco_far (5 mm) <= FAR_MAX_PCT (a far fraction this large is a wrong alignment, not the chamber).
"""
import argparse
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
A = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
CODE = {'6a0bb1f'}
CODE_POST = {'b5a29af'}
POST_STAGES = ('01_alignment', '02_efficiency', '03_angles', '04_maps')
#: the bow transfer gate (rerun/calib/bow/results_pass1/) ran bow_node on the run71 archive; it writes only
#: bow_gate_* beside the products, so it is checked against its own code, not the pass's
CODE_BOW = {'a72b889'}
ALIGN_TOL_MM = 5.0
FAR_MAX_PCT = 15.0
#: every SPS row since the 2026-10-30 final SPS re-run (rerun/sps_audit/README.md): one code version, with the seed fix
#: (oscillating ch 372/510 and the per-plateau hot channels kept out of the seed and censored) and the head-on estimators
CODE_SPS = {'3ac0e79'}
#: meta keys and table columns the final SPS rows must carry
SPS_META_KEYS = ('hot_seed_channels',)
SPS_COLUMNS = ('x_p_mid', 'y_p_mid', 'x_c_pos', 'y_c_pos', 'x_c_n_strips', 'y_c_n_strips', 'ftst')
#: the rows re-run on 2026-10-31 for the oscillation's neighbours (rerun/sps_junk/README.md): one code version, the
#: coupled-channel meta and per-event columns.  The other rows' events do not change under it (rerun/sps_junk census),
#: so they stay on CODE_SPS.
SPS_COUPLED_ROWS = os.path.join(HERE, 'sps_junk', 'results', 'promoted_rows.txt')
CODE_SPS_COUPLED = {'c36322ec', '48244778'}   # 48244778 = c36322ec + the near-oscillator event floor (no pass row changes)
SPS_COUPLED_META = ('hot_seed_channels', 'coupled_channels', 'n_events_coupled_censored')
SPS_COUPLED_COLUMNS = SPS_COLUMNS + ('x_n_coupled', 'y_n_coupled')


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
        codes, overrides, latest = set(), [], {}
        for s in sorted(glob.glob(os.path.join(d, 'STAGES', '*.json'))):     # names start with a timestamp
            try:
                j = json.load(open(s))
            except Exception:
                continue
            c = str(j.get('code') or '?')
            stage = os.path.basename(s).split('_', 2)[-1][:-5]
            if stage in POST_STAGES:
                latest[stage] = c
            elif stage == 'bow_node':
                if c not in CODE_BOW:
                    codes.add(c)
            else:
                codes.add(c)
            if j.get('diagnostic_overrides'):
                overrides.append(os.path.basename(s))
        own_al = r['key'] == (r.get('align_key') or r['calib_key'])
        post_bad = [f'{st} {latest.get(st)}' for st in POST_STAGES
                    if (st != '01_alignment' or own_al) and latest.get(st) not in CODE_POST]
        al, far = None, None
        if own_al and os.path.exists(os.path.join(w, 'alignment', 'alignment.json')):
            al = json.load(open(os.path.join(w, 'alignment', 'alignment.json')))
        ef = os.path.join(w, 'efficiency', 'efficiency_breakdown_cut.json')
        if os.path.exists(ef):
            far = json.load(open(ef)).get('reco_far')
        row = dict(key=r['key'], det=r['det'], cls=r['cls'], run=r['run'], n=n, bundle=bundle, codes=sorted(codes),
                   overrides=overrides, post=latest, z=(al['z_x'], al['z_y']) if al else None, far=far, **have)
        problems = [k for k in ('table', 'meta', 'eff', 'angles', 'maps') if not have[k]]
        if post_bad:
            problems.append('post code ' + ', '.join(post_bad))
        if al and abs(al['z_x'] - al['z_y']) > ALIGN_TOL_MM:
            problems.append(f"z_x {al['z_x']:.0f} vs z_y {al['z_y']:.0f}")
        if far is not None and far > FAR_MAX_PCT:
            problems.append(f'far {far:.1f} %')
        if overrides:
            problems.append('overrides')
        if codes and not codes <= CODE:
            problems.append(f'code {sorted(codes)}')
        row['problems'] = problems
        rows.append(row)
    # z against the night: the median over the chamber's self-aligned rows of that run
    for r in rows:
        if not r['z']:
            continue
        night = [o['z'] for o in rows if o['z'] and o['run'] == r['run'] and o['det'] == r['det']]
        if len(night) >= 3:
            for i, v in enumerate('xy'):
                med = sorted(z[i] for z in night)[len(night) // 2]
                if abs(r['z'][i] - med) > ALIGN_TOL_MM:
                    r['problems'].append(f"z_{v} {r['z'][i]:.0f} vs night {med:.0f}")
    bad = [r for r in rows if r['problems']]
    return rows, bad


def sps(root):
    rows, bad = [], []
    coupled = set(open(SPS_COUPLED_ROWS).read().split()) if os.path.exists(SPS_COUPLED_ROWS) else set()
    for r in csv.DictReader(open(os.path.join(HERE, 'sps_registry.csv'))):
        d = os.path.join(root, 'sps', r['key'])
        mp = os.path.join(d, 'events.meta.json')
        m = json.load(open(mp)) if os.path.exists(mp) else {}
        new = r['key'] in coupled
        want = CODE_SPS_COUPLED if new else CODE_SPS
        problems = []
        tp = os.path.join(d, 'events.parquet')
        if not os.path.exists(tp):
            problems.append('table')
        else:
            import pyarrow.parquet as pq
            cols = set(pq.read_schema(tp).names)
            miss = [c for c in (SPS_COUPLED_COLUMNS if new else SPS_COLUMNS) if c not in cols]
            if miss:
                problems.append(f"columns {','.join(miss)}")
        if m:
            miss = [k for k in (SPS_COUPLED_META if new else SPS_META_KEYS) if k not in m]
            if miss:
                problems.append(f"meta {','.join(miss)}")
            elif 'RAW_CM_BAD' not in str(m.get('seeding', '')):
                problems.append('seeding rule')
        if m and m.get('code') not in want:
            problems.append(f"code {m.get('code')}")
        if m and m.get('diagnostic_overrides'):
            problems.append('overrides')
        if m and 'Analysis_pass1' not in str(m.get('bundle')):
            problems.append('bundle not pass1')
        row = dict(key=r['key'], n=m.get('n_events'), code=m.get('code'), hot=m.get('hot_seed_channels'), problems=problems)
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
         f'Expected code: bench {sorted(CODE)}, SPS {sorted(CODE_SPS)} (seed fix + head-on estimators: meta {list(SPS_META_KEYS)}, columns {list(SPS_COLUMNS)}); the rows of '
         f'{os.path.relpath(SPS_COUPLED_ROWS, HERE)} {sorted(CODE_SPS_COUPLED)} (+ coupled channels: meta {list(SPS_COUPLED_META)}, '
         f'columns x/y_n_coupled); 01-04: {sorted(CODE_POST)}. Alignment within '
         f'{ALIGN_TOL_MM:g} mm of its night, reco_far <= {FAR_MAX_PCT:g} %.', '']
    if Bbad:
        L += ['## Bench rows with problems', '| row | cls | n | problems |', '|---|---|---|---|']
        L += [f"| {r['key']} | {r['cls']} | {r['n']} | {', '.join(r['problems'])} |" for r in Bbad]
    if Sbad:
        L += ['', '## SPS rows with problems', '| row | n | problems |', '|---|---|---|']
        L += [f"| {r['key']} | {r['n']} | {', '.join(r['problems'])} |" for r in Sbad]
    if not Bbad and not Sbad:
        L += ['**No row missing, no overrides, code = the archive, every alignment within its night, no far collapse.**']
    tot = sum(r['n'] or 0 for r in B)
    L += ['', f'Bench events reconstructed: {tot:,}; SPS events: {sum(r["n"] or 0 for r in S):,}']
    txt = '\n'.join(L) + '\n'
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt)
        json.dump(dict(bench=B, sps=S), open(a.out.replace('.md', '.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
