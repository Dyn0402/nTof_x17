"""hitcache.py — fast I/O + pairing layer for the July-beam read pass.

Two speedups over the original per-script uproot reads:

1. I/O: `fastread/extract_hits` (C++, compiled against ROOT) reads each hit tree
   of the official n_TOF file ONCE, keeps only the 9 branches the QA pipeline
   uses, sorts by (BunchNumber, tof) and dumps flat .npy arrays next to the run
   file (`<dir>/hitcache/<stem>/`). `load()` memory-maps those. When the cache
   dir is absent (e.g. on lxplus without the binary), it falls back to the old
   chunked uproot read + lexsort, so scripts behave identically either way.

2. Pairing: the per-bunch python loops are replaced by ONE global searchsorted
   pass. Hits are keyed as bunch * BUNCH_KEY_STRIDE + tof; the stride (1e9 ns)
   dwarfs both the 20 ms acquisition window (2.1e7 ns) and any dt window, so
   same-bunch pairs are found with a plain sorted-array window search and
   cross-bunch pairs are excluded automatically. `iter_pairs()` yields index
   blocks bounded by `max_pairs` to cap memory. Compute dt from the original
   tof arrays (not the keys) — bit-identical to the old per-bunch code.
"""

from pathlib import Path

import numpy as np

try:
    import uproot
except ImportError:          # fine when only npy caches are used
    uproot = None

BUNCH_KEY_STRIDE = 1e9       # ns

_NPY_NAME = {'BunchNumber': 'bunch', 'tof': 'tof', 'tflash': 'tflash',
             'amp': 'amp', 'area': 'area', 'fwhm': 'fwhm', 'detn': 'detn',
             'satuflag': 'satu', 'pileup1': 'pileup'}
_FLAGS = ('satuflag', 'pileup1')     # stored/returned as uint8 (value != 0)


def cache_dir(run_file):
    """hitcache dir for run_file, or None if not extracted yet."""
    d = Path(run_file).parent / 'hitcache' / Path(run_file).stem
    return d if (d / 'meta.json').exists() else None


def bunch_intensity(run_file):
    """Per-bunch PulseIntensity sorted by bunch number (index tree; PKUP
    fallback for runs with an empty index, e.g. run224460)."""
    d = cache_dir(run_file)
    if d is not None:
        return np.load(d / 'index_intensity.npy')
    f = uproot.open(run_file)
    idx = f['index'].arrays(['BunchNumber', 'PulseIntensity'], library='np')
    if len(idx['BunchNumber']) == 0:
        idx = f['PKUP'].arrays(['BunchNumber', 'PulseIntensity'], library='np')
    return idx['PulseIntensity'][np.argsort(idx['BunchNumber'], kind='stable')]


def load(run_file, tree, branches, good_bunches=None):
    """Per-hit arrays for `tree`, sorted by (BunchNumber, tof), optionally
    restricted to `good_bunches`. satuflag/pileup1 come back as uint8 (!=0)."""
    d = cache_dir(run_file)
    if d is not None:
        return _load_npy(d, tree, branches, good_bunches)
    return _load_uproot(run_file, tree, branches, good_bunches)


def _load_npy(d, tree, branches, good_bunches):
    mm = {br: np.load(d / f'{tree}_{_NPY_NAME[br]}.npy', mmap_mode='r')
          for br in branches}
    if good_bunches is None:
        return {br: np.asarray(v) for br, v in mm.items()}
    b = np.load(d / f'{tree}_bunch.npy', mmap_mode='r')
    lo = np.searchsorted(b, good_bunches, side='left')
    hi = np.searchsorted(b, good_bunches, side='right')
    blocks = [np.arange(a, c) for a, c in zip(lo, hi) if c > a]
    idx = (np.concatenate(blocks) if blocks else np.empty(0, np.int64))
    return {br: np.asarray(mm[br][idx]) for br in branches}


def _load_uproot(run_file, tree, branches, good_bunches):
    f = uproot.open(run_file)
    read = list(dict.fromkeys(list(branches) + ['BunchNumber', 'tof']))
    out = {br: [] for br in read}
    for chunk in f[tree].iterate(read, library='np', step_size='300 MB'):
        keep = (np.isin(chunk['BunchNumber'], good_bunches)
                if good_bunches is not None
                else np.ones(len(chunk['BunchNumber']), bool))
        for br in read:
            out[br].append(chunk[br][keep])
    arrs = {br: np.concatenate(v) for br, v in out.items()}
    order = np.lexsort((arrs['tof'], arrs['BunchNumber']))
    arrs = {br: arrs[br][order] for br in branches}
    for br in _FLAGS:
        if br in arrs:
            arrs[br] = (arrs[br] != 0).astype(np.uint8)
    return arrs


def bunch_key(bunch, tof):
    """Global sort key: same order as (bunch, tof), same-bunch dt preserved."""
    return bunch.astype(np.float64) * BUNCH_KEY_STRIDE + tof


def first_by_bunch(bunch_sorted, values, n_bunches, fill=np.nan):
    """values[first hit of each bunch], as an array indexed by bunch number
    (0..n_bunches inclusive; bunches with no hits get `fill`)."""
    out = np.full(n_bunches + 1, fill)
    ub, fi = np.unique(bunch_sorted, return_index=True)
    out[ub] = values[fi]
    return out


def iter_pairs(key_ref, key_other, dt_lo, dt_hi, t_ref, t_other,
               max_pairs=20_000_000):
    """Yield (ref_idx, other_idx) blocks covering every same-bunch pair with
    dt = t_other[oi] - t_ref[ri] in [dt_lo, dt_hi) — the exact window the old
    per-bunch searchsorted code selected. Keys locate candidates (both key
    arrays must be sorted; bunch_key output is); because a key rounds the true
    (bunch, tof) by up to ~1e-3 ns, the key search is widened by a margin and
    the raw-tof window is then applied exactly. Blocks are capped at
    ~max_pairs pairs."""
    margin = 0.1                       # ns; >> key rounding at any bunch count
    lo = np.searchsorted(key_other, key_ref + (dt_lo - margin))
    hi = np.searchsorted(key_other, key_ref + (dt_hi + margin))
    counts = (hi - lo).astype(np.int64)
    cum = np.cumsum(counts)
    start = 0
    n = len(counts)
    while start < n:
        base = cum[start - 1] if start else 0
        end = max(int(np.searchsorted(cum, base + max_pairs, side='right')),
                  start + 1)
        c = counts[start:end]
        tot = int(c.sum())
        if tot:
            ref_idx = np.repeat(np.arange(start, end), c)
            offs = np.repeat(np.cumsum(c) - c, c)
            other_idx = np.repeat(lo[start:end], c) + (np.arange(tot) - offs)
            # candidates are same-bunch by construction (key window << bunch
            # stride), so the raw-tof cut is safe and exact
            dt = t_other[other_idx] - t_ref[ref_idx]
            ok = (dt >= dt_lo) & (dt < dt_hi)
            yield ref_idx[ok], other_idx[ok]
        start = end
