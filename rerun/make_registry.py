#!/usr/bin/env python3
"""make_registry.py -- the pass registry (RERUN_PLAN W7 + W8): every June bench row, what it is, what it inherits.

One row per (subrun, detector) of the campaign manifest on EOS.  Columns:

  key, det, run, subrun, resist_V, drift_V   from the manifest
  cls        golden_calib | golden_quoted | consistency | skeptical | excluded   (plan §5, Dylan 10-23: quote from the
             largest runs unless they show a problem; the final choice is made after the analysis)
  group      (det, run, drift_V): rows that share gas, night and drift field
  calib_key  the row whose calibration serves the group: the golden calibration row of the chamber for its own
             group; otherwise the group's reference row (nominal resist HV, most decoded data) for a v refit
  refit      golden = full W1 calibration (kernel + v, reference from the waveform alignment)
             v      = the row's own angle scale: w0/kw (v) refitted on the row itself against M3, with the chamber's
                      golden kernel pinned
             none   = inherits calib_key's bundle
             Rule (GC 2026-10-28): every row with at least N_OWN_V clean triggers gets its own v.  The gas was still
             exchanging during several nights (det4 6-23, det6/det7 6-26: v 17.8 -> 31.0 um/ns on det6 over 12 h), so
             a row that inherits another row's v reads its angles 9-32 % off; and the resist HV moves the angle
             scale too (det3: -3 % at 520-525 V, saturation).  A row too small to fit its own kw to 0.5 %
             (< N_OWN_V) inherits the row of its group, with its own calibration, nearest to it in time.
             Row times and clean-trigger counts: GC's column-duration pass (ROW_INFO); rows missing there keep the
             group reference.  A group with no golden row whose reference row has < N_MIN_V clean triggers cannot
             calibrate v at all: its rows take the chamber's golden bundle and become skeptical, angle scale not
             calibrated (F9 2026-10-29: det2 6-17, det3 6-26 quick).
  align_key  the row whose waveform alignment the row uses: its calib_key, except for rows that take another
             night's (golden) bundle, which keep their own group's reference row (that row aligns itself)
  K          depth bins of 60 ns: 24 (1440 ns) by default; 30 on drift fields <= 500 V, where the column overruns the
             32-sample window (stated limitation)
  note

    python rerun/make_registry.py  ->  rerun/pass_registry.csv
"""
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MANIFEST = os.path.join(REPO, 'mx_june_wft', 'condor', 'campaign_manifest.csv')

GOLDEN_CALIB = {   # chamber -> (run, subrun): the 8-12 calibration rows (plan §5)
    'mx17_2': ('mx17_det2_det3_overnight_6-22-26', 'longer_run'),
    'mx17_3': ('mx17_det3_saturday_scan_6-27-26', 'long_run_resist_490V_drift_1000V'),
    'mx17_4': ('mx17_det4_day_6-24-26', 'long_run'),
    'mx17_6': ('mx17_det6_det7_overnight_6-26-26', 'long_run'),
    'mx17_7': ('mx17_det6_det7_overnight_6-26-26', 'long_run'),
}
GOLDEN_QUOTED = {  # largest clean runs (quoted candidates; det4/6/7: the calibration run, held out by event)
    'mx17_2': ('mx17_det2_det3_overnight_6-22-26', 'long_run'),
    'mx17_3': ('mx17_det3_p2_det1_overnight_6-27-26', 'long_run_p2_det1_sanity_check'),
}
SKEPTICAL = {
    ('mx17_det1_det2_overnight_6-17-26', 'mx17_2'): 'half-dead X; its decode had fused events (re-judge on the re-decode)',
    ('mx17_det3_det4_overnight_6-23-26', 'mx17_3'): 'degraded bottom-slot night (re-judge on the re-decode)',
    ('mx17_det3_test_6-22-26', 'mx17_3'): 'test run, two acquisitions in one subrun',
    ('mx17_det2_det3_overnight_6-22-26', 'mx17_3'): 'the 6-22 Monday det3 run: micro-TPC angle unusable (r ~ 0.62)',
}
#: runs that share their night, gas and chambers with another run (grouped together for the calibration):
#: the 6-26 det6/det7 HV scan (11H06 on) followed the overnight run (06H03) the same day, same drift field
SAME_NIGHT = {'mx17_det6_det7_hv_scan_6-26-26': 'mx17_det6_det7_overnight_6-26-26'}
#: clean single-cluster triggers in the chamber (GC colsum) needed for a row to fit its own w0/kw against M3: ~250
#: M3-matched tracks give sigma(kw)/kw ~ 0.026 / (0.3 sqrt 250) ~ 0.5 %, below the 0.8 % (0.2 deg at 30 deg) that matters
N_OWN_V = 500
#: a group's reference row below this many clean triggers cannot fit v at all (GC's own 'missing' floor); between
#: N_MIN_V and N_OWN_V a reference row with no calibrated row in its group still fits its own v (the det3 100 V drift
#: point, 475: the golden 1000 V v would be far worse there)
N_MIN_V = 100
#: per row: first/last trigger time (acquisition stem + DREAM timestamp) and clean-trigger count (GC 2026-10-28)
ROW_INFO = os.path.join(HERE, 'model_audit', 'followups', 'gc_gas_time', 'results', 'bench_rows.json')
NO_V_NOTE = 'group reference row < N_MIN_V clean triggers: golden bundle, angle scale NOT calibrated (positions/efficiency only)'
EXCLUDED = {
    'mx17_1': 'det1: no calibration (8-12 manifest)',
}
#: single rows with no usable data (key -> reason)
EXCLUDED_ROWS = {
    # F9 + pass1 10-30: resist 1.3-2.7 uA decaying, hits in a few % of triggers, X one-cluster in 1 %, 0 clean triggers;
    # the waveform alignment finds no X track (01_alignment z scan all-NaN, 4 tries)
    'c26__mx17_det3_quick_6-26-26__resist_530V_drift_1200V__mx17_3':
        'no usable data: chamber charging up, not amplifying (0 clean triggers; alignment finds no X track)',
}
# pass1 10-30: the subrun's v2 M3 ray file is empty (6.6 kB, 0 entries): no reference, nothing to match or calibrate
for _d in ('6', '7'):
    EXCLUDED_ROWS[f'c26__mx17_det6_det7_hv_scan_6-26-26__resist_det6_450V_det7_450V_drift_700V__mx17_{_d}'] = \
        'no M3 reference: the v2 ray file of this subrun is empty (0 rays)'


def nominal_resist(rows):
    """The group's most common resist HV among its biggest subruns."""
    return max(rows, key=lambda r: (r['resist_V'] in ('490.0', '495.0'), int(r['n_dec_files'] or 0)))


def own_v(out):
    """The v rule (module docstring): own w0/kw for every row big enough, else the nearest-in-time calibrated row."""
    import json
    rows_info = json.load(open(ROW_INFO))
    info = {x['key']: x for x in rows_info if not x.get('missing')}
    n_clean = {x['key']: x.get('n_clean', 0) for x in rows_info}
    mid = lambda k: 0.5 * (info[k]['t_start'] + info[k]['t_end'])
    for o in out:
        if o['cls'] != 'excluded' and o['refit'] != 'golden' and o['key'] in info \
                and info[o['key']]['n_clean'] >= N_OWN_V:
            o['refit'], o['calib_key'] = 'v', o['key']
    by_group = defaultdict(list)
    for o in out:
        by_group[o['group']].append(o)
    # A group whose reference row is too small to fit any v (F9, 2026-10-29: det2 6-17, half-dead X, <= 210 clean
    # triggers in any row; det3 6-26 quick 530 V / 1200 V, resist drawing 1.3-2.7 uA and hits in a few % of triggers,
    # 0 clean) has no row that can calibrate v: it takes the chamber's golden bundle, angle scale not calibrated.
    gold = {o['det']: o['key'] for o in out if o['refit'] == 'golden'}
    for rs in by_group.values():
        if any(o['refit'] == 'golden' for o in rs):
            continue
        for o in rs:
            if o['refit'] == 'v' and n_clean.get(o['key'], 0) < N_MIN_V:
                for p in rs:
                    if p['cls'] != 'excluded' and p['calib_key'] == o['key']:
                        # the bundle comes from the golden row, the geometry does not: another night's mounting
                        p['align_key'] = o['key']
                        p['calib_key'], p['refit'], p['cls'] = gold[p['det']], 'none', 'skeptical'
                        p['note'] = '; '.join(x for x in (p['note'], NO_V_NOTE) if x)
    for rs in by_group.values():
        cal = [o for o in rs if o['refit'] in ('golden', 'v') and o['key'] in info]
        for o in rs:
            if o['cls'] == 'excluded' or o['refit'] != 'none' or o['key'] not in info or not cal:
                continue
            o['calib_key'] = min(cal, key=lambda c: abs(mid(c['key']) - mid(o['key'])))['key']


def main():
    with open(MANIFEST) as f:
        rows = [r for r in csv.DictReader(f) if 'june_tests' in (r['eos_run_dir'] or '')]
    groups = defaultdict(list)
    for r in rows:
        groups[(r['det'], SAME_NIGHT.get(r['run'], r['run']), r['drift_V'])].append(r)
    out = []
    for (det, run, drift), rs in sorted(groups.items()):
        gc = GOLDEN_CALIB.get(det)
        golden_here = [r for r in rs if gc and (r['run'], r['subrun']) == gc]
        ref = golden_here[0] if golden_here else nominal_resist(rs)
        K = 30 if float(drift) <= 500 else 24
        for r in rs:
            note = ''
            if det in EXCLUDED:
                cls, refit, note = 'excluded', '-', EXCLUDED[det]
            elif gc and (r['run'], r['subrun']) == gc:
                cls, refit = 'golden_calib', 'golden'
            elif (run, det) in SKEPTICAL:
                cls, note = 'skeptical', SKEPTICAL[(run, det)]
                refit = 'v' if r is ref else 'none'
            else:
                cls = 'golden_quoted' if GOLDEN_QUOTED.get(det) == (r['run'], r['subrun']) else 'consistency'
                refit = ('none' if golden_here else 'v') if r is ref else 'none'
            if r['tier'] == 'C' and r['reason'].startswith('no decoded'):
                cls, refit, note = 'excluded', '-', r['reason']
            if r['key'] in EXCLUDED_ROWS:
                cls, refit, note = 'excluded', '-', EXCLUDED_ROWS[r['key']]
            out.append(dict(key=r['key'], det=det, run=r['run'], subrun=r['subrun'], resist_V=r['resist_V'],
                            drift_V=drift, cls=cls, group=f'{det}|{run}|{drift}', calib_key=ref['key'],
                            refit=refit, K=K, note=note))
    for o in out:
        o.setdefault('align_key', '')
    own_v(out)
    for o in out:
        o['align_key'] = o['align_key'] or o['calib_key']
    p = os.path.join(HERE, 'pass_registry.csv')
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    from collections import Counter
    print(f'{len(out)} rows -> {p}')
    print(' classes:', dict(Counter(o['cls'] for o in out)))
    print(' refits :', dict(Counter(o['refit'] for o in out)))
    print(' groups :', len(groups), '; K:', dict(Counter(o['K'] for o in out)))


if __name__ == '__main__':
    main()
