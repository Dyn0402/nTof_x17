#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_daq_calib.py -- export the DREAM<->n_TOF calibration to the DAQ repo.

Writes machine-readable constants to nTof_x17_DAQ/calibrations/dream_ntof/,
which is the operational home for anything anyone outside this analysis has to
look up. Three files, deliberately separated because they have three different
lifetimes:

  time_map_*.json          OFFLINE. Fitted here, per (DREAM run, n_TOF
                           processing) pair. Nothing to set on any instrument.
  ntof_internal_*.json     OFFLINE. A property of the PROCESSING, re-measured
                           per reprocessing. Says which offsets to apply
                           (currently: none) and which to measure per file.
  n1081b_thresholds_*.json DAQ. Hardware state, read back from the run's own
                           n1081b_config.json -- an INPUT to this analysis, not
                           an output of it. This is the one a shifter changes.

Regenerate after fit_timebase.py / fit_perbunch.py / align_survey.py /
window_scan.py / bias_check.py have run.

USAGE
    python export_daq_calib.py [--out <calibrations dir>] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from study_common import DATA, ROOT, NTOF_RUN, DREAM_RUN, SUBRUNS

DEFAULT_OUT = ROOT.parent / 'nTof_x17_DAQ' / 'calibrations' / 'dream_ntof'
VARIANT = 'v12_liqpileup'
ARMS = ('A', 'B', 'C', 'D')
WINDOW_NS = 25.0
POINT = 'tight_25'


def _load():
    d = {}
    for name in ('timebase', 'alignment', 'recommended_window',
                 'perbunch_wp_summary', 'bias_check_wp',
                 'window_scan_summary_perbunch', 'tb_offsets_official_vs_v12'):
        p = DATA / f'{name}.json'
        if not p.exists():
            raise SystemExit(f'missing {p} -- run the pipeline first')
        d[name] = json.loads(p.read_text())
    return d


def _thresholds():
    """The discriminator settings the run actually held, per sub-run."""
    import sys
    sys.path.insert(0, str(ROOT))
    from ntof_dream_merge.dream_trigger import load_thresholds
    out = {}
    for sub in SUBRUNS:
        t = load_thresholds(DREAM_RUN, sub)
        out[sub] = dict(wall_mV={a: float(t['wall'][a]) for a in ARMS},
                        plastic_mV={a: float(t['plastic'][a]) for a in ARMS},
                        plastic_pmts={a: list(map(int, t['pmts'][a]))
                                      for a in ARMS},
                        polled_at=t['polled_at'])
    return out


def time_map(D, stamp):
    tb = D['timebase']
    pb = D['perbunch_wp_summary']
    bias = D['bias_check_wp']
    pts = D['window_scan_summary_perbunch']['legs']
    arm = {a: float(np.mean([v['a'] for v in tb['per_arm'][a].values()]))
           for a in ARMS}
    arm_spread = {a: float(abs(np.diff([v['a'] for v in
                                        tb['per_arm'][a].values()])[0]))
                  for a in ARMS}
    return {
        'what': ('the map from a DREAM trigger timestamp to the n_TOF time base, '
                 'and the accept window around it'),
        'usage': ('t_nTOF [ns] = t_DREAM * (1 + K + dk_b) + T0 + a_arm + da_b ; '
                  'accept |t_candidate - t_nTOF| < window_ns'),
        'kind': 'OFFLINE analysis constant -- nothing here is set on any instrument',
        'IMPORTANT': ('K, T0 and the per-arm offsets are properties of the PAIR '
                      '(DREAM run, n_TOF processing) and DO NOT TRANSFER. The '
                      'constants fitted on the official processing of the same '
                      'run leave a -45 ns offset and a 1.35 % rate error on v12. '
                      'Re-fit per run pair: match_study/scripts/fit_timebase.py.'),
        'K': tb['fitted']['K'],
        'T0_ns': tb['fitted']['T0'],
        'arm_offset_ns': arm,
        'arm_offset_reproducibility_ns': arm_spread,
        'per_bunch': {
            'note': ('da_b and dk_b are fitted per bunch from that bunch own '
                     'matched triggers and are ALWAYS re-fitted; the numbers here '
                     'describe the population, they are not constants to apply'),
            'fit': 'least squares on |r| < 200 ns, one trim pass, >= 20 triggers',
            'median_triggers_per_bunch': float(np.mean(
                [bias[s]['median_events_per_bunch'] for s in SUBRUNS])),
            'min_triggers_per_bunch': float(min(
                bias[s]['min_events_per_bunch'] for s in SUBRUNS)),
            'offset_rms_ns': {s: pb[s]['offset_rms'] for s in SUBRUNS},
            'rate_rms_ppm': {s: pb[s]['rate_rms'] * 1e6 for s in SUBRUNS},
            'real_drift_rms_ppm': {s: bias[s]['split_half']['drift_rms_k_ppm']
                                   for s in SUBRUNS},
            'fit_noise_rms_ppm': {s: bias[s]['split_half']['noise_rms_k_ppm']
                                  for s in SUBRUNS},
            'bunches_fitted': int(sum(bias[s]['n_bunches'] for s in SUBRUNS)),
        },
        'window_ns': WINDOW_NS,
        'window_bands': [[-WINDOW_NS, WINDOW_NS]],
        'window_note': ('ONE band. The [+250,+450] ns satellite of the earlier '
                        'calibration was a delayed wall lobe of the OLD pulse '
                        'reconstruction: on v12 it adds 0.00 points of efficiency '
                        'and 0.21 points of background. Criterion for the width: '
                        'the tightest window within 0.5 % (relative) of the '
                        'efficiency plateau; the knee is at '
                        f'{D["recommended_window"]["legs"]["wp"]["half_width_ns"]:.1f}'
                        ' ns.'),
        'performance_at_window': {
            leg: {
                'efficiency': pts[leg]['points'][POINT]['eff'],
                'accidental': pts[leg]['points'][POINT]['false'],
                'accidental_minus_control': pts[leg]['points'][POINT]['false_minus'],
                'purity': pts[leg]['points'][POINT]['purity'],
                'frac_multi_candidate': pts[leg]['points'][POINT]['frac_multi'],
                'frac_multi_arm': pts[leg]['points'][POINT]['frac_multi_arm'],
                'per_time_since_flash_ms': pts[leg]['points'][POINT]['per_t'],
            } for leg in ('wp', 'w')},
        'match_resolution_ns': {
            'note': '68 % half-width, cross-validated, flat over 1-80 ms',
            'per_time_since_flash_ms': {
                k: float(np.mean([bias[s]['widths'][k]['xval'] for s in SUBRUNS]))
                for k in bias[SUBRUNS[0]]['widths']}},
        'provenance': _prov(stamp, D),
    }


def ntof_internal(D, stamp):
    A = D['alignment']
    ch = np.array([v['median'] for v in A['coincidence']['channel'].values()])
    tbl = D['tb_offsets_official_vs_v12']
    return {
        'what': ('the internal time alignment of the n_TOF detectors, measured in '
                 'situ on the file being analysed'),
        'kind': ('OFFLINE. A property of the PROCESSING, not of the hardware -- '
                 're-measure after any reprocessing or UserInput change'),
        'variant': VARIANT,
        'run': NTOF_RUN,
        'offsets_to_apply': {
            'value': 'none',
            'why': ('on this processing every subsystem is already within a few ns '
                    'of the others: liquid-vs-wall -0.8..+0.2 ns, wall-vs-plastic '
                    'per channel RMS 2.3 ns inside a 20 ns logic pulse. Applying a '
                    'stale offset is as harmful as ignoring a live one -- measure, '
                    'then decide.')},
        'measure_per_file': {
            'wall_top_bottom_ns': (
                'the per-segment t_top - t_bottom used to pair the two ends of a '
                'bar. On v12 it is within +-5.5 ns; on the OFFICIAL processing of '
                'the same run it is +-32..39 ns with one -77.5 ns outlier. That '
                'structure was the old flash-finder / leading-edge timing, removed '
                'by the wall shape fitting of v4_walshapes. Pairing around a 38 ns '
                'offset that is no longer there loses most genuine pairs. Use '
                'fast_singles.measure_tb_offsets (seconds).'),
            'measured': {'official': tbl['official'], 'v12': tbl['v12'],
                         'bunches': tbl['bunches']}},
        'tflash_repair': {
            'setting': 'OFF',
            'why': ('ntof_dream_merge/tflash_repair.py was built for the BROKEN '
                    'official flash finding. On v12 it is not a no-op: it would '
                    'shift LIQC/LIQD by ~15 ns and add 25 ns RMS on PSSC, while '
                    'the stored time base already has the liquids within 1 ns of '
                    'the walls. ntof_io defaults it ON -- turn it off explicitly.')},
        'flash_vs_pickup_ns': {
            t: {'median': v['median'], 'per_bunch_sigma': v['std_core']}
            for t, v in A['flash'].items()},
        'flash_vs_divert_off_calibration': {
            'reference': ('nTof_x17_DAQ/calibrations/flash_timing/'
                          'flash_time_constants.json (LIQ monitor values)'),
            'liquids_agree_to_ns': {
                t: round(A['flash'][t]['median'] - A['calibration'][t]['C_ns'], 2)
                for t in ('LIQA', 'LIQB', 'LIQC', 'LIQD')},
            'plastics_disagree_by_ns': {
                t: round(A['flash'][t]['median'] - A['calibration'][t]['C_ns'], 1)
                for t in ('PSSA', 'PSSB', 'PSSC', 'PSSD')},
            'note': ('the liquids reproducing the divert-off calibration to '
                     '0.1-0.5 ns is two independent measurements confirming each '
                     'other. The plastics are 31-50 ns away, exactly as '
                     'flash_timing/README.md warns: take PSS per run.')},
        'wall_vs_plastic_ns': {
            'per_arm_peak': {a: v['peak']
                             for a, v in A['coincidence']['station'].items()},
            'per_arm_sigma': {a: v['sigma']
                              for a, v in A['coincidence']['station'].items()},
            'per_channel_rms': float(ch.std()),
            'per_channel_range': [float(ch.min()), float(ch.max())]},
        'liquid_vs_wall_ns': {a: v['peak']
                              for a, v in A['coincidence']['liq'].items()},
        'provenance': _prov(stamp, D, n_bunches=A['n_bunches']),
    }


def n1081b(stamp, thr):
    return {
        'what': 'the N1081B discriminator thresholds the run actually held',
        'kind': ('DAQ hardware state. This is an INPUT to the offline trigger '
                 'emulation, not something the analysis fits. It is the file a '
                 'shifter changes.'),
        'IMPORTANT': ('read per sub-run from <run>/<subrun>/n1081b_config.json, '
                      'which is the only record of what the discriminators were '
                      'set to. Do not assume they held across a run.'),
        'trigger_chain': ('M1 = 428F analogue SUM of the two bar ends over the '
                          'wall threshold, ORed over the 4 bar segments; '
                          'M2 = any plastic bar of that arm over its threshold; '
                          'M3 = M1 .AND. M2 inside the 20 ns logic pulse'),
        'recommended_thresholds': ('see ../wal_trigger/ and ../pss_trigger/ for '
                                   'what to SET; this file records what WAS set'),
        'run': DREAM_RUN,
        'ntof_run': NTOF_RUN,
        'sub_runs': thr,
        'provenance': {'exported_utc': stamp,
                       'source': 'nTof_x17/ntof_dream_merge/dream_trigger.py '
                                 'load_thresholds(), reading the per-sub-run '
                                 'n1081b_config.json staged from the DAQ machine'},
    }


def _prov(stamp, D, **extra):
    p = dict(
        exported_utc=stamp,
        dream_run=DREAM_RUN, dream_sub_runs=list(SUBRUNS),
        ntof_run=NTOF_RUN, ntof_processing=VARIANT,
        n_bunches=2061, n_dream_triggers=213420,
        stored_tflash='used as-is, laptop repair OFF',
        source='nTof_x17/ntof_dream_merge/match_study/',
        document='nTof_x17/ntof_dream_merge/DREAM_NTOF_CALIBRATION.md',
        regenerate='cd match_study/scripts && python export_daq_calib.py',
    )
    p.update(extra)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    D = _load()
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    tag = f'run{NTOF_RUN}_{VARIANT}'

    files = {
        f'time_map_{DREAM_RUN}_{tag}.json': time_map(D, stamp),
        f'ntof_internal_alignment_{tag}.json': ntof_internal(D, stamp),
        f'n1081b_thresholds_{DREAM_RUN}.json': n1081b(stamp, _thresholds()),
    }

    out = Path(args.out)
    if args.dry_run:
        for name, obj in files.items():
            print(f'--- {name}\n{json.dumps(obj, indent=1)[:600]}\n')
        return 0
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in files.items():
        (out / name).write_text(json.dumps(obj, indent=1) + '\n')
        print(f'  {name}')
    print(f'-> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
