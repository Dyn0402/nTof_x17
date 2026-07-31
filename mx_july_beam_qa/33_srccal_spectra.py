"""33_srccal_spectra.py — read pass for the 2026-07-28 Y-88 + Cs-137 plastic
calibration (T1). One run in, one small .npz cache out. No matplotlib: this is
the script that runs on lxplus next to the EOS data (see lxplus/srccal.sub).

Successor of `21_y88_spectra.py`. Same fine linear-mV histogram (0-160 mV,
0.2 mV bins) so the two campaigns are directly comparable, plus what the
two-source design and the higher rates need:

  * EVERY channel of all four arms is histogrammed, not just the source arm.
    In this campaign each run has 4-5 completely dark arms, so the dark runs of
    a given arm form a proper, rate-normalised BACKGROUND TEMPLATE for it
    (`srccal_runs.dark_runs_for`). 34 subtracts it before fitting.
  * a pileup-vetoed copy of every histogram (`*_lin_np`, pileup1 == 0). The
    source runs sit at ~50 kHz/channel in a 20 ms window, so pileup is a real
    edge-position systematic and has to be measurable, not assumed away.
  * per-channel saturation counts (amp above the ~31000 ADC headroom common to
    every tree) — an edge fit is meaningless if the spectrum is clipped.
  * a coarse tof profile per tree. Beam is OFF, so the profile must be FLAT;
    structure would mean a DAQ/baseline artefact contaminating the spectra.
  * trigger count (index tree) so every spectrum can be turned into a rate.

Outputs (per run, ~1 MB):
  cache/33_srccal_<run>.npz
  calib/adc_to_mv_<run>.json     (written by adc_mv, one per run)
Usage:
  python 33_srccal_spectra.py <run.root | run stem | run number> [...]
  (default: all nine campaign runs, resolved under $SRCCAL_DATA or ~/x17/beam_july/data)
"""

import os
import sys
from pathlib import Path

import numpy as np
import uproot

import srccal_runs as S
from adc_mv import mv_factors

BASE = Path(__file__).parent
DATA = Path(os.environ.get('SRCCAL_DATA', Path.home() / 'x17' / 'beam_july' / 'data'))
CACHE = BASE / 'cache'
CACHE.mkdir(exist_ok=True)

# 21_y88_spectra.py stopped at 160 mV, which was right for the 07-17 gains
# (699 keVee edge at 20-30 mV). Here the gains run from about the same (arm A)
# up to ~2x that (BL, CL): the 699 keVee edge spans 29-65 mV and the 1612 keVee
# edge reaches 125-136 mV, so the old ceiling would sit right on top of the
# outer edge and cut off the continuum above it that the fit needs for its
# background. The grid therefore runs to 400 mV, at 0.25 mV — still fine enough
# that the edge error is set by counting statistics, not by the binning (the
# extraction floors its error at half a bin).
LIN_EDGES = np.arange(0.0, 400.0 + 1e-9, 0.25)
LOG_EDGES = np.geomspace(0.5, 1000.0, 121)
TOF_EDGES = np.linspace(0.0, 2.0e7, 201)      # 20 ms acquisition, 100 us bins
SAT_ADC = 31000.0        # amp headroom to the near end of range (all trees)
BRANCHES = ['amp', 'detn', 'pileup1', 'satuflag', 'tof']


def resolve(token):
    """Accept a path, a stem ('run224588') or a bare number ('224588')."""
    p = Path(token)
    if p.suffix == '.root' and p.exists():
        return p
    stem = token if str(token).startswith('run') else f'run{token}'
    return DATA / f'{stem}.root'


def n_triggers(f):
    """Acquisition (bunch) count — the exact live-time normalisation, since all
    windows are the same length. index first, PKUP as fallback."""
    for t in ('index', 'PKUP'):
        if t in f:
            n = f[t].num_entries
            if n:
                return int(n)
    return 0


def spectra_for_run(run_file):
    run_file = Path(run_file)
    stem = run_file.stem
    fac = mv_factors(run_file)
    f = uproot.open(run_file)
    ntrig = n_triggers(f)

    store = {'run': stem, 'n_triggers': ntrig, 'window_ms': S.WINDOW_MS,
             'lin_edges': LIN_EDGES, 'log_edges': LOG_EDGES,
             'tof_edges': TOF_EDGES, 'sat_adc': SAT_ADC}
    # ALL_MAP, not SOURCE_MAP: the same read pass also serves the 07-17 legacy
    # runs used by the equalization cross-check (36).
    for src, bar in S.ALL_MAP[stem].items():
        store[f'source_{src}'] = bar if bar else ''

    for kind in ('PSS', 'WAL', 'LIQ'):
        nch = S.NCH[kind]
        for a in S.ARMS:
            tree = f'{kind}{a}'
            if tree not in f:
                print(f'  {tree}: MISSING from file')
                continue
            lin = np.zeros((nch, len(LIN_EDGES) - 1))
            lin_np = np.zeros_like(lin)          # pileup1 == 0
            log = np.zeros((nch, len(LOG_EDGES) - 1))
            tofp = np.zeros((nch, len(TOF_EDGES) - 1))
            nhit = np.zeros(nch, np.int64)
            nsat = np.zeros(nch, np.int64)       # amp > SAT_ADC
            nflag = np.zeros(nch, np.int64)      # satuflag set
            npile = np.zeros(nch, np.int64)
            fmv = fac[tree]

            for ch in f[tree].iterate(BRANCHES, library='np', step_size='200 MB'):
                amp = ch['amp'].astype(np.float64)
                detn = ch['detn']
                pile = ch['pileup1'] != 0
                for c in range(nch):
                    m = detn == (c + 1)
                    if not m.any():
                        continue
                    a_adc = amp[m]
                    a_mv = a_adc * fmv[c]
                    lin[c] += np.histogram(a_mv, bins=LIN_EDGES)[0]
                    log[c] += np.histogram(a_mv, bins=LOG_EDGES)[0]
                    tofp[c] += np.histogram(ch['tof'][m], bins=TOF_EDGES)[0]
                    keep = ~pile[m]
                    lin_np[c] += np.histogram(a_mv[keep], bins=LIN_EDGES)[0]
                    nhit[c] += m.sum()
                    nsat[c] += int((a_adc > SAT_ADC).sum())
                    nflag[c] += int((ch['satuflag'][m] != 0).sum())
                    npile[c] += int(pile[m].sum())

            store[f'{tree}_lin'] = lin
            store[f'{tree}_lin_np'] = lin_np
            store[f'{tree}_log'] = log
            store[f'{tree}_tof'] = tofp
            store[f'{tree}_nhit'] = nhit
            store[f'{tree}_nsat'] = nsat
            store[f'{tree}_nsatuflag'] = nflag
            store[f'{tree}_npileup'] = npile
            store[f'{tree}_mv'] = fmv

    out = CACHE / f'33_srccal_{stem}.npz'
    np.savez_compressed(out, **store)

    src = ', '.join(f'{k}->{v}' for k, v in S.sources_in(stem).items())
    print(f'{stem}: {ntrig} triggers ({ntrig * S.WINDOW_MS / 1000:.1f} s live), '
          f'sources {src}  -> {out.name}')
    for a in S.ARMS:
        row = []
        for kind in ('PSS', 'WAL', 'LIQ'):
            k = f'{kind}{a}_nhit'
            if k in store:
                n = store[k]
                sat = store[f'{kind}{a}_nsat'].sum()
                row.append(f'{kind}={n.sum() / max(ntrig, 1):8.0f} hit/trig'
                           + (f' (sat {100 * sat / max(n.sum(), 1):.2f}%)'
                              if sat else ''))
        lit = S.source_on_arm(stem, a)
        print(f'   arm {a} {"<" + lit + ">" if lit else "(dark)":9s} '
              + '  '.join(row))
    return store


def main():
    toks = sys.argv[1:] or S.RUNS
    for t in toks:
        p = resolve(t)
        if not p.exists():
            print(f'MISSING: {p}')
            continue
        spectra_for_run(p)


if __name__ == '__main__':
    main()
