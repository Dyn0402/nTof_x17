#!/usr/bin/env python3
"""Census of detector names in the n_TOF official processed files.

Run this ON LXPLUS (uproot from LCG_105); it only touches the ROOT header and
the tiny `DAQsettings` tree, so a full run costs ~1 s of EOS latency per file.

    source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh
    python3 scan_processed_detectors.py runs.txt > census.tsv

`runs.txt` is one n_TOF run number per line.  Output columns:

    run <TAB> TREES=<comma list> <TAB> DETS=<comma list>

TREES is what the PSA actually produced; DETS is what the DAQ had configured.
A detector that appears in DETS but not in TREES was digitised but never
processed -- that is exactly how MMA/MMB show up (see NTOF_MICROMEGAS_SIGNALS.md).
"""
import sys
from concurrent.futures import ThreadPoolExecutor

import uproot

DONE = '/eos/experiment/ntof/processing/official/done'


def scan(run):
    try:
        f = uproot.open(f'{DONE}/run{run}.root')
        trees = sorted(k.split(';')[0] for k in f.keys(recursive=False))
        dets = []
        if 'DAQsettings' in f:
            seen = set()
            for n in f['DAQsettings']['detectorName'].array(library='np'):
                n = n if isinstance(n, str) else n.decode()
                if n not in seen:
                    seen.add(n)
                    dets.append(n)
        return f"{run}\tTREES={','.join(trees)}\tDETS={','.join(sorted(dets))}"
    except Exception as exc:                      # empty/corrupt files exist
        return f'{run}\tERROR\t{type(exc).__name__}: {exc}'


def main():
    runs = [int(l) for l in open(sys.argv[1]) if l.strip()]
    with ThreadPoolExecutor(8) as ex:
        for line in ex.map(scan, runs):
            print(line, flush=True)


if __name__ == '__main__':
    main()
