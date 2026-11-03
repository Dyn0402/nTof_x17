#!/usr/bin/env python3
"""a1_table.py -- the Paper I appendix A1 table from the a1_recat.py products (2026-11-03).

    .venv/bin/python rerun/effbudget/pass1/a1_table.py --eff <paper1_eff[_twotrack]> --analysis <Analysis_pass1[_twotrack]>
        [--p1-eff <paper1_eff> (pass-1 recat, for the merged-in-projection follow-up)] [--keys sat_det3,<c26 key>]
        [--out results/a1_table.json] [--tex papers/paper1/sections/a1_table_rows.tex]

Pools the interior rays of the chosen golden rows (default: the two det3 rows, Runs 1 + 2) from <eff>/results/a1.json
(counts per class) and maps the a1_recat classes onto the table rows.  Also derives from the products:
  * pass 1 -> two-track re-pass: how many of the pass-1 'multi:m3_not_found' (merged in projection) events have a
    reference-matched kept track (track_id >= 0 within 5 mm of the reference in both views) in the re-pass tree
    (the old "8/37" of the pre-pass joint-fit test), and the classes they end up in;
  * head-on: fit_failed events with |tan_x| < 0.03 (a1_recat 'fit_failed_headon_frac').
Writes the JSON and the \\input snippet of the table body (rows in table order)."""
import argparse, json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WD = os.environ.get('WORK_DATA', '/media/ucla/work')
TT = WD + '/claude-jobs/twotrack'
P1 = WD + '/claude-jobs/17c92c46/x17rerun/paper1_eff'
#: (label, figure id, [a1_recat classes], recoverable text)
ROWS = [
    ('noise band on a strip patch', 'loss_band', ['band'], 'largely: band hits are censored in the pass (Run 1: 1.43\\,\\% pre-pass, 0.22\\,\\% now)'),
    ('second particle, reference particle fitted, view cut', 'loss_multi_m3picked', ['multi:m3_picked'],
     'yes, if a view with two tracks is kept (multi-track credit)'),
    ('second particle, the other one fitted', 'loss_multi_otherpicked', ['multi:other_picked'],
     'no: needs a rule to choose the triggering track'),
    ('second particle, merged in projection', 'loss_multi_merged', ['multi:m3_not_found'], '@MERGED@'),
    ('``spark\'\': discharge', 'loss_spark_discharge', ['spark:discharge'], 'no'),
    ('``spark\'\': muon + unrelated deposit', 'loss_spark_muonother', ['spark:muon+other'],
     'partly: veto off + competing-cluster cut relaxed'),
    ('``spark\'\': shower at the crossing', 'loss_spark_shower', ['spark:shower'], 'no'),
    ('$\\delta$-ray on the track', 'loss_delta', ['far:DELTA'], 'not with a straight-line model'),
    ('saturated $\\delta$ blob, fit pulled off', 'loss_satblob', ['far:FIT_SAT'], '@SAT@'),
    ('fit off, no other flag', 'loss_fit', ['far:FIT'], 'no'),
    ('fit failed in one view (the pre-pass head-on class)', 'loss_headon', ['fit_failed'], '@HEAD@'),
    ('position tail, mostly bent reference tracks', 'loss_postail', ['far:POS_TAIL'], 'no (reference)'),
    ('edge of the other coordinate', 'loss_edge', ['far:EDGE'], 'no (geometry)'),
    ('second deposit, not particle-like', 'loss_multi_deposit', ['far:MULTI'], 'no'),
    ('other (cuts, charge elsewhere)', None, ['cut:other', 'other:charge_elsewhere', 'other:not_seeded', 'other:no_hit',
                                              'other:one_view_silent', 'other:hw_known'], '---'),
]


def merged_followup(rows, tt_eff, tt_ana, p1_eff):
    """pass-1 multi:m3_not_found events -> credited / class in the re-pass tree."""
    out = dict(n=0, credited=0, to={}, per_row={})
    for r in rows:
        rel = r['rel']; d = r['det']
        old = pd.read_parquet(f'{p1_eff}/a1/{rel}/recat.parquet')
        new = pd.read_parquet(f'{tt_eff}/a1/{rel}/recat.parquet')
        R = pd.read_parquet(f'{tt_eff}/budget/{rel}/rays_{d}.parquet')
        R = R[~R.not_measured & (R.region == 'interior')].drop_duplicates('event_id').set_index('event_id')
        C = pd.read_parquet(f'{tt_ana}/{rel}/wft/events.candidates.parquet')
        C = C[C.track_id >= 0]
        ev = old.index[old.recat == 'multi:m3_not_found']
        ev = [e for e in ev if e in new.index]
        ok = {}
        for v in 'xy':
            c = C[C.plane == v].copy()
            c['m'] = R[f'mesh_{v}'].reindex(c.event_id).to_numpy()
            ok[v] = set(c.event_id[(c.p0 - c.m).abs() <= 5.0])
        cred = [e for e in ev if e in ok['x'] and e in ok['y']]
        out['per_row'][r['key'][:12]] = dict(n=len(ev), credited=len(cred))
        out['n'] += len(ev); out['credited'] += len(cred)
        for k, v in new.recat.reindex(ev).value_counts().items():
            out['to'][k] = out['to'].get(k, 0) + int(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eff', default=TT + '/paper1_eff_twotrack'); ap.add_argument('--analysis', default=TT + '/Analysis_pass1_twotrack')
    ap.add_argument('--p1-eff', default=P1)
    ap.add_argument('--keys', default='', help='comma list of a1.json keys (default: all in the file)')
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'a1_table.json'))
    ap.add_argument('--tex', default=os.path.join(HERE, '..', '..', '..', 'papers', 'paper1', 'sections', 'a1_table_rows.tex'))
    ap.add_argument('--pop', default='interior')
    ap.add_argument('--satblob', default=os.path.join(HERE, '..', '..', 'satstudy', 'results_tt', 'satblob_numbers.json'),
                    help='saturated-blob test on the re-pass (rerun/satstudy/satblob_numbers.py); missing -> \\requote kept')
    a = ap.parse_args()
    A = json.load(open(f'{a.eff}/results/a1.json'))
    keys = [k for k in a.keys.split(',') if k] or list(A)
    rows = [A[k] | {'key': k} for k in keys]
    run1 = [r for r in rows if r['key'] == 'sat_det3']
    pools = {'pooled': rows, 'run1': run1}
    res = dict(keys=keys, eff=a.eff, analysis=a.analysis, pop=a.pop)
    tot = {}
    for nm, rs in pools.items():
        n = sum(r[a.pop]['n'] for r in rs)
        cnt = {}
        for r in rs:
            for k, v in r[a.pop]['counts'].items():
                cnt[k] = cnt.get(k, 0) + v
        tot[nm] = dict(n=n, counts=cnt, detected_pct=100 * cnt.get('detected', 0) / n)
    res['tot'] = tot
    mf = merged_followup(rows, a.eff, a.analysis, a.p1_eff)
    res['merged_followup'] = mf
    ff = {nm: sum(r[a.pop]['counts'].get('fit_failed', 0) for r in rs) for nm, rs in pools.items()}
    res['fit_failed_headon_frac'] = {r['key'][:12]: r[a.pop]['fit_failed_headon_frac'] for r in rows}
    table = []
    lines = []
    for lab, fid, cls, rec in ROWS:
        vals = {}
        for nm in pools:
            c = sum(tot[nm]['counts'].get(k, 0) for k in cls)
            vals[nm] = dict(count=c, pct=100 * c / tot[nm]['n'])
        table.append(dict(label=lab, fig=fid, classes=cls, **vals))
        if rec == '@MERGED@':
            rec = (f'partly: joint two-track fit with charge corroboration, {mf["credited"]}/{mf["n"]} of the pass-1 events of '
                   'this class have a reference-matched fitted track (Run 1: ' + str(mf['per_row'].get('sat_det3', {}).get('credited')) + '/' + str(mf['per_row'].get('sat_det3', {}).get('n')) + ')')
        elif rec == '@SAT@':
            sb = json.load(open(a.satblob)) if os.path.exists(a.satblob) else None
            if sb:
                rec = (f'no: neither censoring nor a track\\,+\\,compact-deposit fit (test on {sb["n_views"]} saturated views of Run 1 on the re-pass: '
                       f'{sb["agree_prod"]:.0f}\\,$\\to$\\,{sb["agree_best"]:.0f}\\,\\% agree)')
            else:
                rec = 'no: neither censoring nor a track\\,+\\,compact-deposit fit (pre-pass test on 350 saturated views: \\requote{71}\\,$\\to$\\,\\requote{74}\\,\\% agree; not repeated on the re-pass)'
        elif rec == '@HEAD@':
            rec = 'yes: head-on seeding in the pass (Run 1: 0.26\\,\\% pre-pass, 17 tracks; 0.02\\,\\% now, none head-on)'
        ref = f'(\\ref{{fig:{fid}}})' if fid else ''
        lines.append(f'    {lab} {ref} & {vals["pooled"]["pct"]:.2f} & {vals["run1"]["pct"]:.2f} & {rec} \\\\')
    res['table'] = table
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    open(a.tex, 'w').write('% generated by rerun/effbudget/pass1/a1_table.py -- do not edit\n' + '\n'.join(lines) + '\n    \\hline\n')  # closing rule here: \hline right after \input in a tabular is a Misplaced \noalign
    print(json.dumps({k: res[k] for k in ('tot', 'merged_followup', 'fit_failed_headon_frac')}, indent=1))
    for t in table:
        print(f'{t["label"][:55]:55s} pooled {t["pooled"]["pct"]:5.2f} ({t["pooled"]["count"]})  run1 {t["run1"]["pct"]:5.2f}')


if __name__ == '__main__':
    main()
