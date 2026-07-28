#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ntof_io.py -- per-bunch access into the official n_TOF hit trees.

The official file is one ~26 GB ROOT file per run with one tree per detector
(WALA-D SiPM walls, PSSA-D plastics, LIQA-D liquid scintillators, SILI, plus the
PKUP beam-pickup tree and a per-bunch `index`). A tree is a flat hit list: one
entry = one PSA-fitted pulse. Run 224572 carries ~610 M hits over 3018 bunches.

Reading a whole tree to get a handful of bunches is hopeless, but it turns out
`BunchNumber` is monotonically non-decreasing in entry order in every tree, so a
single pass over that one branch (0.8-2 s per tree, it compresses well) gives
entry offsets per bunch, and everything after that is an `entry_start/stop` read.
The offsets are cached per (run, tree).

Time base: each hit carries `tof` and its bunch's `tflash`, both in NANOSECONDS
(not the microseconds the mx_july README quotes). t_since_flash = tof - tflash is
the gamma-flash-referenced time this analysis joins on. `tflash` differs between
trees (~11.2 us on WALA vs ~13.3 us on PKUP) because each detector has its own
cable/electronics delay, so subtracting each tree's OWN tflash is what puts them
on a common t=0 -- do not take tflash from PKUP for a scintillator hit.

TFLASH IS REPAIRED HERE, NOT TRUSTED. The official PSA mis-identifies the gamma
flash on the PSS trees in 37-85 % of bunches (essentially every parasitic pulse),
shifting the stored tflash by up to 11.6 us and with it every t_since_flash of
that (tree, bunch) -- this is what masqueraded as "n_TOF records a plastic for
only ~52 % of DREAM triggers". `read_bunches` therefore computes
t_since_flash_ns against tflash_repair.corrected_tflash() (per-tree cable mode +
bunch-common jitter from the stable trees) instead of the stored value. The raw
`tflash` branch is still returned untouched when asked for. See
tflash_repair.py for the measurements. Set repair_tflash=False to get the old
(broken for PSS) behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.beam_july_paths import NTOF_DATA_DIR, ANALYSIS_DIR  # noqa: E402

WALL_TREES = ('WALA', 'WALB', 'WALC', 'WALD')
PLASTIC_TREES = ('PSSA', 'PSSB', 'PSSC', 'PSSD')
LIQ_TREES = ('LIQA', 'LIQB', 'LIQC', 'LIQD')
SCINT_TREES = WALL_TREES + PLASTIC_TREES + LIQ_TREES

CACHE_DIR = ANALYSIS_DIR / 'ntof_dream_merge' / 'cache'

# Hit branches worth carrying into the merge. `amp`/`area` are the PSA pulse
# height/integral, `satuflag` marks a clipped pulse and `pileup1` a pulse the PSA
# had to disentangle -- both matter when the amplitude is used as a MIP proxy.
HIT_BRANCHES = ('BunchNumber', 'detn', 'tof', 'tflash', 'amp', 'area',
                'satuflag', 'pileup1')


def ntof_paths(run: int) -> list:
    """Every ROOT file holding this run, in bunch order.

    A run is EITHER a single merged `run<run>.root` OR a directory
    `run<run>.parts/` of the per-job partials that RunProcessing.sh leaves in
    `<out>/completed/<run>/`. The partials are preferred when both exist.

    Reading the partials directly is the normal case, not a fallback: the
    official merge step cannot produce a merged EAR2 run at all (its condor job
    ships the partials through condor file transfer and dies on
    `max total download bytes exceeded (max=1024 MB)`), and merging by hand with
    hadd costs an hour of serial I/O and a second 26 GB copy per run. The
    partials are already contiguous and bunch-ordered, so chaining them is both
    cheaper and the only thing that scales to the whole campaign.
    """
    parts = NTOF_DATA_DIR / f'run{run}.parts'
    if parts.is_dir():
        files = sorted(parts.glob(f'run{run}_[0-9]*.root'),
                       key=lambda p: int(p.stem.split('_')[-1]))
        if files:
            return files
    p = NTOF_DATA_DIR / f'run{run}.root'
    if p.exists():
        return [p]
    raise FileNotFoundError(
        f'neither {p} nor {parts}/ is staged. Get it with '
        'ntof_dream_merge/stage_reference_pair.sh ntof')


def ntof_path(run: int) -> Path:
    """Back-compat: the single file of a merged run. Prefer ntof_paths()."""
    files = ntof_paths(run)
    if len(files) != 1:
        raise ValueError(f'run{run} is stored as {len(files)} partials; '
                         'use ntof_paths()')
    return files[0]


def _read_range(run: int, tree: str, branches, lo: int = 0, hi: int = None,
                counts=None) -> dict:
    """Read a GLOBAL entry range [lo, hi) of `tree`, spanning partials."""
    files = ntof_paths(run)
    if counts is None:
        counts = _tree_counts(run, tree)
    starts = np.concatenate([[0], np.cumsum(counts)])
    if hi is None:
        hi = int(starts[-1])
    out = {k: [] for k in branches}
    for path, s, e in zip(files, starts[:-1], starts[1:]):
        a0, b0 = max(lo, int(s)), min(hi, int(e))
        if b0 <= a0:
            continue
        with uproot.open(path) as f:
            a = f[tree].arrays(list(branches), entry_start=a0 - int(s),
                               entry_stop=b0 - int(s), library='np')
        for k in branches:
            out[k].append(a[k])
    return {k: (np.concatenate(v) if v else np.array([])) for k, v in out.items()}


def _tree_counts(run: int, tree: str) -> np.ndarray:
    """Entries per file for `tree` (cached alongside the bunch index)."""
    cache = CACHE_DIR / f'bunchidx_{run}_{tree}.npz'
    if cache.exists():
        z = np.load(cache)
        if 'counts' in z.files:
            return z['counts']
    counts = []
    for path in ntof_paths(run):
        with uproot.open(path) as f:
            counts.append(f[tree].num_entries)
    return np.asarray(counts, dtype=np.int64)


def bunch_edges(run: int, tree: str, rebuild: bool = False) -> np.ndarray:
    """
    Entry offsets per bunch: hits of BunchNumber b are entries [e[b-1], e[b]).

    Index 0 is bunch 1, so the array has (max_bunch + 1) elements. Bunches with no
    hits give an empty (equal-valued) range rather than an error. Entry numbers
    are GLOBAL across the run's partials, in the order of ntof_paths().
    """
    cache = CACHE_DIR / f'bunchidx_{run}_{tree}.npz'
    if cache.exists() and not rebuild:
        return np.load(cache)['edges']

    bn, counts = [], []
    for path in ntof_paths(run):
        with uproot.open(path) as f:
            b = f[tree]['BunchNumber'].array(library='np')
        bn.append(b)
        counts.append(len(b))
    bn = np.concatenate(bn) if bn else np.array([], dtype=np.int64)
    if np.any(np.diff(bn) < 0):
        raise ValueError(f'{tree} in run{run}: BunchNumber is not sorted by entry; '
                         'the searchsorted index is invalid for this file set '
                         '(are the partials in the right order?)')
    edges = np.searchsorted(bn, np.arange(1, int(bn.max()) + 2))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, edges=edges,
                        counts=np.asarray(counts, dtype=np.int64))
    return edges


_TFLASH_FIX_CACHE = {}


def _tflash_fix(run: int) -> dict:
    if run not in _TFLASH_FIX_CACHE:
        from ntof_dream_merge.tflash_repair import corrected_tflash
        _TFLASH_FIX_CACHE[run] = corrected_tflash(run)
    return _TFLASH_FIX_CACHE[run]


def read_bunches(run: int, tree: str, bunches, branches=HIT_BRANCHES,
                 repair_tflash: bool = True) -> dict:
    """
    Hits of the given bunches from one tree, as a dict of flat arrays.

    Contiguous bunch runs are read as one entry range -- the reference-pair
    bunches are a contiguous block of the run, so this is usually a single read
    rather than one per bunch. Adds `t_since_flash_ns` = tof - tflash, where
    tflash is the REPAIRED per-bunch flash time (see module docstring and
    tflash_repair.py) unless repair_tflash=False.
    """
    bunches = np.unique(np.asarray(bunches, dtype=np.int64))
    if bunches.size == 0:
        return {b: np.array([]) for b in tuple(branches) + ('t_since_flash_ns',)}
    edges = bunch_edges(run, tree)
    want = set(branches) | {'tof', 'tflash'} | ({'BunchNumber'} if repair_tflash else set())

    # group consecutive bunch numbers into blocks -> one entry range per block
    breaks = np.where(np.diff(bunches) > 1)[0] + 1
    out = {k: [] for k in want}
    counts = _tree_counts(run, tree)
    for grp in np.split(bunches, breaks):
        lo, hi = edges[grp[0] - 1], edges[grp[-1]]
        if hi <= lo:
            continue
        a = _read_range(run, tree, want, int(lo), int(hi), counts)
        for k in want:
            out[k].append(a[k])
    res = {k: (np.concatenate(v) if v else np.array([])) for k, v in out.items()}
    if repair_tflash and tree in _tflash_fix(run) and res['tof'].size:
        tf = _tflash_fix(run)[tree][res['BunchNumber'].astype(np.int64)]
        res['t_since_flash_ns'] = res['tof'] - tf
    else:
        res['t_since_flash_ns'] = res['tof'] - res['tflash']
    return {k: v for k, v in res.items()
            if k in branches or k == 't_since_flash_ns'}


def _index_epoch(run: int) -> tuple[np.ndarray, np.ndarray]:
    """(BunchNumber, wall-clock seconds) from the `index` tree's Date/Time fields.

    Date is DYYMMDD and Time is HHMMSS as integers, both LOCAL (the DAQ writes
    local = UTC+2), truncated to the second. Coarse, but -- unlike psTime -- it is
    filled for every bunch, which is what makes the psTime repair below possible.
    """
    a = _read_range(run, 'index', ['BunchNumber', 'Date', 'Time'])
    o = np.argsort(a['BunchNumber'])
    d, t = a['Date'][o], a['Time'][o]
    yy, mm, dd = 2000 + (d // 10000) % 100, (d // 100) % 100, d % 100
    hh, mi, ss = t // 10000, (t // 100) % 100, t % 100
    epoch = (np.array([np.datetime64(f'{y:04d}-{m:02d}-{D:02d}T{H:02d}:{M:02d}:{S:02d}')
                       for y, m, D, H, M, S in zip(yy, mm, dd, hh, mi, ss)])
             - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
    return a['BunchNumber'][o].astype(np.int64), epoch.astype(np.float64)


PS_PERIOD_S = 1.2      # PS basic period; every bunch of run224572 sits exactly on
                       # this grid (all 3017 spacings are integer multiples, and
                       # the residual mod 1.2 s is 0.0000 s over 2978 good bunches)


def pkup_bunches(run: int) -> dict:
    """
    Per-bunch beam record: BunchNumber, psTime_s (UTC), intensity (1e10 p), tflash.

    ~1.3 % of bunches carry an unfilled `psTime` -- denormal garbage ~1e-310, two
    20-bunch blocks in run224572 (bunches 2038-2057 and 2638-2657, segments 102 and
    132). PLAN.md describes this as one contiguous block 2038-2077 and suggests
    interpolating; it is actually two blocks, and interpolation is the wrong repair
    because the pulse spacing is irregular (1.2 to 12 s), so a linear fill across 20
    bunches lands seconds away and the burst match then fails outright -- that is
    exactly the 20 bursts run_79/stat090_0001 was losing.

    The right repair uses the beam structure: PS pulses sit on an EXACT 1.2 s grid.
    `index.Date/Time` is filled for every bunch and is good to +-0.5 s, which is
    well inside the 0.6 s half-period, so each bad bunch snaps to a unique grid
    point and the recovered psTime is exact rather than approximate.

    `pstime_recovered` flags the repaired bunches.
    """
    a = _read_range(run, 'PKUP',
                    ['BunchNumber', 'psTime', 'PulseIntensity', 'tflash'])
    o = np.argsort(a['BunchNumber'])
    bn = a['BunchNumber'][o].astype(np.int64)
    ps = a['psTime'][o] / 1e9
    good = ps > 1e9                       # a plausible 2026 unix timestamp

    ps_filled = ps.copy()
    if (~good).any():
        ibn, iep = _index_epoch(run)
        if not np.array_equal(ibn, bn):
            iep = iep[np.searchsorted(ibn, bn)]
        # index.Time is local; take the constant local->UTC shift from the bunches
        # where both clocks exist, rather than trusting this machine's timezone.
        shift = np.median(ps[good] - iep[good])
        approx = iep[~good] + shift
        ref = ps[good][0]
        ps_filled[~good] = ref + np.round((approx - ref) / PS_PERIOD_S) * PS_PERIOD_S

    return dict(BunchNumber=bn, psTime_s=ps_filled,
                intensity_e10=a['PulseIntensity'][o] / 1e10,
                tflash_ns=a['tflash'][o],
                pstime_recovered=~good)


if __name__ == '__main__':
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 224572
    p = pkup_bunches(run)
    n_bad = int(p['pstime_interpolated'].sum())
    print(f'run{run}: {len(p["BunchNumber"])} bunches, '
          f'{n_bad} with interpolated psTime ({100*n_bad/len(p["BunchNumber"]):.2f} %)')
    print(f'  span UTC {np.datetime64(int(p["psTime_s"].min()), "s")} -> '
          f'{np.datetime64(int(p["psTime_s"].max()), "s")}')
    for tree in SCINT_TREES:
        e = bunch_edges(run, tree)
        print(f'  {tree}: {e[-1]:>12,} hits, median {np.median(np.diff(e)):>7,.0f}/bunch')
