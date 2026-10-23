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
             v      = v (+ t0 table) refitted on calib_key with the chamber's golden kernel pinned
             none   = inherits calib_key's bundle (HV-scan points: resist HV changes the gain, absorbed by the fit's
                      free charge profile)
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
EXCLUDED = {
    'mx17_1': 'det1: no calibration (8-12 manifest)',
}


def nominal_resist(rows):
    """The group's most common resist HV among its biggest subruns."""
    return max(rows, key=lambda r: (r['resist_V'] in ('490.0', '495.0'), int(r['n_dec_files'] or 0)))


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
            out.append(dict(key=r['key'], det=det, run=r['run'], subrun=r['subrun'], resist_V=r['resist_V'],
                            drift_V=drift, cls=cls, group=f'{det}|{run}|{drift}', calib_key=ref['key'],
                            refit=refit, K=K, note=note))
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
