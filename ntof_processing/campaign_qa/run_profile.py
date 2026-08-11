#!/usr/bin/env python3
"""Per-run profile of a processed n_TOF partial set, normalised so that OUR
runs and n_TOF's official runs can be put side by side.

A raw hit count is not comparable between runs: it scales with the proton
intensity of the bunches that happen to be in the file, and with the DAQ's
zero-suppression threshold, neither of which is a property of the processing.
So for each run this reports

  * beam:   mean PulseIntensity over the sampled bunches, and the fraction of
            bunches with essentially no protons (empty PS pulses)
  * DAQ:    the zero-suppression threshold and full scale per detector family,
            straight out of the DAQsettings tree
  * rate:   hits/bunch AND hits/bunch per 1e12 protons, per tree
  * flash:  fraction of bunches whose tflash is >150 ns off the tree's mode
            (the check that failed 37-85 % in the broken official processing)
  * time:   median tof and the fraction of hits inside the first 100 us

Usage:
    python run_profile.py label=<dir-or-file>[,<file>...] [label2=...] \
                          [--partials=2] [--json=out.json]

A directory argument is expanded to its run<run>_NNNN.root partials and the
first `--partials` of them are read.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import uproot

TREES = ([f'WAL{a}' for a in 'ABCD'] + [f'PSS{a}' for a in 'ABCD']
         + [f'LIQ{a}' for a in 'ABCD'] + ['PKUP', 'SILI'])
BRANCHES = ['BunchNumber', 'tflash', 'tof', 'amp', 'PulseIntensity']


def expand(arg, npart):
    """label=<paths> -> (label, [Path, ...]), directories expanded in order."""
    label, _, spec = arg.partition('=')
    if not spec:
        label, spec = Path(arg).name, arg
    files = []
    for token in spec.split(','):
        p = Path(token)
        if p.is_dir():
            parts = sorted(p.glob('run[0-9]*_[0-9]*.root'),
                           key=lambda q: int(q.stem.split('_')[-1]))
            files += parts[:npart]
        else:
            files.append(p)
    return label, files


def mode(v, binw=10.0):
    v = v[np.isfinite(v)]
    if not v.size:
        return np.nan
    h, e = np.histogram(v, bins=np.arange(0.0, 20000.0, binw))
    return float(e[h.argmax()] + binw / 2)


def daq_settings(path):
    """zero-suppression threshold per detector family, from the file itself."""
    try:
        d = uproot.open(path)['DAQsettings'].arrays(
            ['detectorName', 'zeroSuppThrmV', 'fullScalemV', 'samplingRate'],
            library='np')
    except Exception:
        return {}
    out = {}
    for name, thr, fs, sr in zip(d['detectorName'], d['zeroSuppThrmV'],
                                 d['fullScalemV'], d['samplingRate']):
        fam = re.sub(r'[0-9]+$', '', str(name))[:4]
        out.setdefault(fam, set()).add((round(float(thr), 3), round(float(fs), 1),
                                        round(float(sr), 4)))
    return {k: sorted(v) for k, v in out.items()}


def profile(label, files):
    print(f'\n{"=" * 78}\n{label}   ({len(files)} file(s))')
    for f in files:
        print(f'    {f}')
    res = {'label': label, 'files': [str(f) for f in files], 'trees': {}}

    res['daq'] = {k: v for k, v in daq_settings(files[0]).items()}
    print('\n  DAQsettings (thr_mV, fullscale_mV, GHz):')
    for fam, vals in sorted(res['daq'].items()):
        print(f'    {fam:<5} {vals}')

    # beam, from the index tree (one row per bunch, no hit weighting)
    inten, nb_tot = [], 0
    for f in files:
        try:
            idx = uproot.open(f)['index'].arrays(
                ['BunchNumber', 'PulseIntensity'], library='np')
        except Exception:
            continue
        inten.append(idx['PulseIntensity'])
        nb_tot += idx['BunchNumber'].size
    inten = np.concatenate(inten) if inten else np.array([])
    empty = float((inten < 1e-3).mean()) if inten.size else np.nan
    res['beam'] = {'index_bunches': nb_tot,
                   'mean_intensity': float(np.nanmean(inten)) if inten.size else np.nan,
                   'median_intensity': float(np.nanmedian(inten)) if inten.size else np.nan,
                   'frac_empty_pulses': empty}
    print(f'\n  beam: index rows={nb_tot}  mean PulseIntensity='
          f'{res["beam"]["mean_intensity"]:.3g}  median='
          f'{res["beam"]["median_intensity"]:.3g}  empty={empty * 100:.1f} %')

    hdr = (f'\n  {"tree":<6} {"hits":>10} {"bunch":>6} {"hits/bunch":>11} '
           f'{"per 1e12p":>10} {"flashbad%":>9} {"med tof us":>11} '
           f'{"<100us %":>9} {"med amp":>9}')
    print(hdr)
    print('  ' + '-' * (len(hdr) - 3))
    for t in TREES:
        cols = {}
        for f in files:
            try:
                h = uproot.open(f)
            except Exception:
                continue
            if t not in {k.split(';')[0] for k in h.keys()}:
                continue
            want = [b for b in BRANCHES if b in h[t].keys()]
            a = h[t].arrays(want, library='np')
            for k, v in a.items():
                cols.setdefault(k, []).append(v)
        if not cols:
            continue
        a = {k: np.concatenate(v) for k, v in cols.items()}
        bn = a['BunchNumber']
        nb = np.unique(bn).size
        n = bn.size
        inten_h = a.get('PulseIntensity')
        # per-bunch mean intensity, hit-weighted removed by taking unique bunches
        if inten_h is not None and n:
            first = np.unique(bn, return_index=True)[1]
            ib = inten_h[first]
            tot_p = float(np.nansum(ib)) / 1e12          # in units of 1e12 protons
        else:
            tot_p = np.nan
        per_p = n / tot_p if tot_p and np.isfinite(tot_p) and tot_p > 0 else np.nan

        # flash health: fraction of bunches whose tflash is >150 ns off the mode
        badf = np.nan
        if 'tflash' in a and n:
            tf = a['tflash']
            first = np.unique(bn, return_index=True)[1]
            tfb = tf[first]
            m = mode(tfb)
            if np.isfinite(m):
                badf = float(np.mean(np.abs(tfb - m) > 150.0)) * 100.0

        tof = a.get('tof', np.array([]))
        med_tof = float(np.nanmedian(tof)) / 1e3 if tof.size else np.nan   # ns->us
        early = float(np.mean(tof < 1e5)) * 100.0 if tof.size else np.nan
        amp = a.get('amp', np.array([]))
        med_amp = float(np.nanmedian(amp)) if amp.size else np.nan

        res['trees'][t] = {'hits': int(n), 'bunches': int(nb),
                           'hits_per_bunch': n / nb if nb else np.nan,
                           'hits_per_1e12p': per_p, 'flash_bad_pct': badf,
                           'median_tof_us': med_tof, 'frac_first_100us': early,
                           'median_amp': med_amp}
        print(f'  {t:<6} {n:>10} {nb:>6} {n / nb:>11.1f} {per_p:>10.1f} '
              f'{badf:>9.2f} {med_tof:>11.1f} {early:>9.1f} {med_amp:>9.1f}')
    return res


def main():
    npart = 2
    outjson = None
    args = []
    for a in sys.argv[1:]:
        if a.startswith('--partials='):
            npart = int(a.split('=', 1)[1])
        elif a.startswith('--json='):
            outjson = a.split('=', 1)[1]
        else:
            args.append(a)
    out = []
    for a in args:
        label, files = expand(a, npart)
        files = [f for f in files if f.exists()]
        if not files:
            print(f'{label}: no files')
            continue
        try:
            out.append(profile(label, files))
        except Exception as e:
            print(f'{label}: FAILED {type(e).__name__}: {e}')
    if outjson:
        Path(outjson).write_text(json.dumps(out, indent=1, default=float))
        print(f'\nwrote {outjson}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
