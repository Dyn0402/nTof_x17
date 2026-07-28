#!/usr/bin/env python3
"""Export the flash-time constants to the DAQ calibrations tree.

Mirrors the convention of nTof_x17_DAQ/calibrations: flat machine-readable
constants tagged by source run, with a provenance block, plus a CSV that can be
read without a JSON parser.  Regenerate rather than hand-edit.

    python export_daq_calib.py [dest_dir]
"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / 'data' / 'flash_timing_calibration.json'
DEST = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path.home() / 'PycharmProjects' / 'nTof_x17_DAQ' / 'calibrations' / 'flash_timing'

WALLS = ['WALA', 'WALB', 'WALC', 'WALD']


def main():
    cal = json.loads(SRC.read_text())
    C = cal['constants']
    T = cal['transport_monitor']['per_tree']
    DEST.mkdir(parents=True, exist_ok=True)

    # ---- flat per-channel JSON, the thing an online/offline tool reads
    chans = {}
    for tree in WALLS:
        for ch, v in C[tree]['channels'].items():
            chans[f'{tree}_{ch}'] = {
                'tree': tree, 'detn': int(ch),
                'C_ns': v['C_2026_07_16'],
                'C_ns_epoch_2026_07_11': v['C_2026_07_11'],
                'per_bunch_sigma_ns': v['per_bunch_sigma_ns'],
            }
    out = {
        'what': 'gamma-flash arrival time of each detector channel, referenced to the '
                'beam pickup (PKUP) pulse of the same bunch',
        'usage': 't_flash_at_channel(bunch) [ns] = tof_PKUP(bunch) + C_ns',
        'IMPORTANT': 'C is PER CHANNEL. Do not average to one constant per wall: a '
                     'per-detector constant is no better than a single global one '
                     '(3.42 vs 3.43 ns rms residual) because the four wall means agree '
                     'to 0.7 ns while channels within a wall differ by up to 13.3 ns.',
        'sign': 'C is NEGATIVE: the flash reaches the detectors ~1.72 us BEFORE the '
                'pickup pulse appears in the digitiser window',
        'epochs': {
            'C_ns': 'the 2026-07-16 epoch — use for runs >= 224400 (the state that held '
                    'to the end of the campaign)',
            'C_ns_epoch_2026_07_11': 'use for runs < 224400',
            'note': 'the 07-11 -> 07-16 shift is per-channel (WALB ch3 -0.6 ns, '
                    'WALD ch7 -6.9 ns); do not mix epochs',
        },
        'corrections': {
            'beam_intensity_ns': -5.0,
            'beam_intensity_note': 'C shifts by -5.0 ns going from parasitic (4.1e12 p) '
                                   'to dedicated (8.5e12 p). Front-end saturation, not a '
                                   'real change of flash arrival. Values here are the '
                                   'run-average of a mixed-intensity sample; apply per '
                                   'bunch from PulseIntensity if you need better than 5 ns.',
        },
        'accuracy_ns': cal['recommended_use']['accuracy'],
        'ungated_detectors': {
            'note': 'LIQ and PSS are never blanked, so they see the flash in every run. '
                    'LIQ is stable across the campaign and can be used as a live monitor '
                    'of the time base; PSS is NOT stable (moves tens of ns) — take it per run.',
            'LIQ_C_ns': {k: T[k]['C_ns'] for k in ('LIQA', 'LIQB', 'LIQC', 'LIQD') if k in T},
            'LIQ_stability_ns': {k: T[k]['std_over_runs_ns']
                                 for k in ('LIQA', 'LIQB', 'LIQC', 'LIQD') if k in T},
        },
        'provenance': {
            'exported_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'source_runs': cal['measured']['runs'],
            'why_these_runs': cal['measured']['why_these_runs'],
            'method': cal['measured']['method'],
            'analysis': 'nTof_x17/ntof_processing/flash_timing/ (report: latex/flash_timing_calibration.pdf)',
            'regenerate': 'python nTof_x17/ntof_processing/flash_timing/scripts/export_daq_calib.py',
            'validated_against': 'the independently reprocessed run 224572 (v4_walshapes): '
                                 'walls agree to 2.07 ns rms, liquids to 0.27 ns rms',
            'caveats': [
                'measured on the SEVEN runs of the campaign in which the SiPM-wall divert '
                'was disabled; every other run has the walls blanked and cannot show the flash',
                'the wall front end saturates on the undiverted flash — these runs give '
                'TIMING only, no amplitude/energy information',
                'valid for the 2026 EAR2 X17 hardware state; re-measure after any change '
                'to wall HV/thresholds or cabling (the 07-11 -> 07-16 wall equalisation '
                'moved individual channels by up to 6.9 ns)',
            ],
        },
        'channels': chans,
    }
    (DEST / 'flash_time_constants.json').write_text(json.dumps(out, indent=2) + '\n')

    # ---- CSV twin
    with open(DEST / 'flash_time_constants_per_channel.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['tree', 'detn', 'C_ns_2026_07_16', 'C_ns_2026_07_11', 'per_bunch_sigma_ns'])
        for tree in WALLS:
            for ch in sorted(C[tree]['channels'], key=int):
                v = C[tree]['channels'][ch]
                w.writerow([tree, ch, v['C_2026_07_16'], v['C_2026_07_11'],
                            v['per_bunch_sigma_ns']])
    print(f'wrote {DEST}/flash_time_constants.json  (+ .csv, {len(chans)} channels)')


if __name__ == '__main__':
    main()
