"""
adc_mv.py — Per-channel ADC -> mV conversion factors from DAQsettings.

factor[tree][detn-1] = fullScalemV / ADCrange  (mV per ADC count).
Cached to calib/adc_to_mv_<run>.json on first use.

Usage:
    from adc_mv import mv_factors
    fac = mv_factors()            # {'WALA': array(8), ..., 'PSSD': array(2)}
"""

import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
DEFAULT_RUN = Path.home() / 'x17/beam_july/data/run224404.root'


def mv_factors(run_file=DEFAULT_RUN):
    cal = BASE / 'calib' / f'adc_to_mv_{Path(run_file).stem}.json'
    if not cal.exists():
        import uproot
        t = uproot.open(run_file)['DAQsettings'].arrays(library='np')
        out = {}
        for name, detn, fs, rng in zip(t['detectorName'], t['detectorNumber'],
                                       t['fullScalemV'], t['ADCrange']):
            name = str(name)
            if name.startswith(('WAL', 'PSS')):
                out.setdefault(name, {})[int(detn)] = fs / rng
        cal.parent.mkdir(exist_ok=True)
        cal.write_text(json.dumps({'run': Path(run_file).stem,
                                   'unit': 'mV per ADC count',
                                   'factors': out}, indent=2))
    d = json.loads(cal.read_text())['factors']
    return {tree: np.array([chans[str(i + 1)] for i in range(len(chans))])
            for tree, chans in d.items()}
