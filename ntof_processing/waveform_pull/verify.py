#!/usr/bin/env python3
"""Closure check on a waveform product, against the slim it was built from.

The check that matters is simple and exact, and it does not reuse any of the
machinery that built the file: **every hit the slim recorded inside the pulled
window must sit inside a kept block.** A hit is a (bunch, det, detn, tof); a
block covers [tof0, tof0 + n). If a hit is not covered, the window arithmetic or
the tflash bridge is wrong, and nothing downstream would notice.

    python -m ntof_processing.waveform_pull.verify <ntof_wf_*.root> [--slim <path>]

Reports, and fails on, four things:
  * hits inside the window that no block covers          (window/tflash error)
  * bunches requested that carry no block at all         (missing raw)
  * blocks whose samples are shorter than their declared n  (truncated raw)
  * the fraction of pulled samples that no hit explains  (informational: this
    is the whole point of a block-driven pull, so it should be LARGE)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import uproot

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'ntof_processing.waveform_pull'

from . import config as C                                          # noqa: E402


def find_slim(wf_path: Path, meta: dict) -> Path | None:
    p = Path(meta.get('slim', ''))
    if p.is_file():
        return p
    guess = wf_path.parent.parent / 'ntof_hits' / \
        wf_path.name.replace('ntof_wf_', 'ntof_hits_')
    return guess if guess.is_file() else None


def verify(wf_path: Path, slim_path: Path | None = None,
           edge_tol_ns: float = 8.0) -> dict:
    wf_path = Path(wf_path)
    meta_path = wf_path.parent / (wf_path.stem + '_provenance.json')
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    slim_path = Path(slim_path) if slim_path else find_slim(wf_path, meta)

    with uproot.open(wf_path) as f:
        blk = f['blocks'].arrays(['bunch', 'det', 'detn', 'tof0', 'n'],
                                 library='np')
        n_entries = f['blocks'].num_entries
        ev = f['events'].arrays(['bunch'], library='np') if 'events' in f else None

    res = {'file': str(wf_path), 'n_blocks': int(n_entries),
           'n_samples': int(blk['n'].sum()), 'window_ns': meta.get('window_ns'),
           'coverage_reported': meta.get('coverage')}

    # ---- blocks the raw declared longer than it actually held. Counted during
    # the scan, which is the only place the declared length survives; reading it
    # back off the product cannot work, because the product stores the length it
    # actually wrote. (Reading the samples array to measure it also crashed on
    # awkward 1.10, whose Index64 has no `.data`, and would pull GBs into RAM.)
    short = int(meta.get('scan', {}).get('blocks_short', 0))
    res['blocks_truncated'] = short

    if slim_path is None:
        res['slim'] = None
        res['status'] = 'PARTIAL (no slim to check against)'
        return res
    res['slim'] = str(slim_path)

    with uproot.open(slim_path) as f:
        hits = f['hits'].arrays(['det', 'detn', 'tof', 'dt_ns', 'eventId',
                                 'is_control'], library='np')
        sev = f['events'].arrays(['eventId', 'bunch'], library='np')

    o = np.argsort(sev['eventId'])
    hb = sev['bunch'][o][np.searchsorted(sev['eventId'][o], hits['eventId'])]

    win = meta.get('window_ns', C.WINDOW_NS)
    inwin = np.abs(hits['dt_ns']) <= win
    if meta.get('control_shift_ns') is None:
        inwin &= hits['is_control'] == 0
    # Coverage is measured only where the pull actually had raw to read.
    # Bunches that were requested but absent from the raw are a SEPARATE and
    # louder failure (n_bunches_missing) -- folding them in here would drown
    # the arithmetic check this function exists for.
    pulled = meta.get('bunches_pulled')
    if pulled is not None:
        inwin &= np.isin(hb, np.asarray(pulled, np.int64))
    res['hits_in_window'] = int(inwin.sum())

    # ---- every such hit must fall inside a kept block on its own channel
    bkey = (blk['bunch'].astype(np.int64) * 100_000
            + blk['det'].astype(np.int64) * 1000 + blk['detn'].astype(np.int64))
    order = np.lexsort((blk['tof0'], bkey))
    bkey_s, tof0_s, n_s = bkey[order], blk['tof0'][order], blk['n'][order]

    hkey = (hb[inwin].astype(np.int64) * 100_000
            + hits['det'][inwin].astype(np.int64) * 1000
            + hits['detn'][inwin].astype(np.int64))
    htof = hits['tof'][inwin]

    # Candidate = last block of the SAME channel starting at or before the hit.
    # This has to be a single lexicographic search on (channel, tof0): searching
    # tof0 alone is wrong because tof0 only rises within a channel, not across
    # the file, and np.searchsorted on that non-monotonic array silently returns
    # nonsense (it read 2.9 % coverage on a file that was in fact complete).
    SCALE = 100_000_000                       # > the 80 ms acquisition window
    kb = bkey_s * SCALE + tof0_s
    kh = hkey * SCALE + np.floor(htof).astype(np.int64)
    j = np.searchsorted(kb, kh, 'right') - 1
    ok = j >= 0
    jc = np.clip(j, 0, max(len(kb) - 1, 0))
    same = ok & (bkey_s[jc] == hkey)
    covered = same & (htof >= tof0_s[jc]) & (htof < tof0_s[jc] + n_s[jc])
    res['hits_covered'] = int(covered.sum())
    res['hits_uncovered'] = int((~covered).sum())
    res['hit_coverage'] = float(covered.mean()) if covered.size else 1.0

    # A hit landing a nanosecond or two past the end of the block that holds its
    # pulse is a time-base convention, not lost data: PRE_SAMPLES = 259 was
    # measured on LIQA alone, and the PSA's own tof for some pulse classes sits
    # exactly 2 ns beyond the block end (measured on PSSD, 3/3, zero spread).
    # The waveform IS in the file, so this is reported apart from a real miss.
    past = htof - (tof0_s[jc] + n_s[jc])
    edge = same & ~covered & (past >= 0) & (past <= edge_tol_ns)
    res['hits_edge_within_tol'] = int(edge.sum())
    res['edge_tol_ns'] = edge_tol_ns
    if edge.any():
        res['edge_gap_ns_p50_p100'] = [float(np.median(past[edge])),
                                       float(past[edge].max())]
    res['hits_no_block_on_channel'] = int((~same).sum())
    res['hits_missing'] = int((~covered & ~edge).sum())
    res['hit_coverage_with_edge'] = (
        float((covered | edge).mean()) if covered.size else 1.0)
    if res['hits_missing']:
        bad = np.nonzero(~covered & ~edge)[0][:10]
        res['missing_examples'] = [
            {'bunch': int(hb[inwin][k]), 'det': int(hits['det'][inwin][k]),
             'detn': int(hits['detn'][inwin][k]), 'tof': float(htof[k]),
             'dt_ns': float(hits['dt_ns'][inwin][k]),
             'has_block_on_channel': bool(same[k])} for k in bad]

    # ---- bunches asked for but carrying nothing
    if ev is not None:
        want = set(pulled) if pulled is not None \
            else set(np.unique(ev['bunch']).tolist())
        got = set(np.unique(blk['bunch']).tolist())
        res['bunches_requested'] = len(want)
        res['bunches_with_blocks'] = len(want & got)
        res['bunches_empty'] = len(want - got)

    # ---- how much of the pull is NOT explained by a slim hit (should be most)
    res['samples_per_hit_in_window'] = (
        float(res['n_samples'] / max(res['hits_in_window'], 1)))

    fails = []
    if res['hits_missing']:
        fails.append(f"{res['hits_missing']} slim hits inside the window are "
                     f"not covered by any kept block")
    if short:
        fails.append(f'{short} blocks are shorter than their declared length')
    if meta.get('n_bunches_missing'):
        fails.append(f"{meta['n_bunches_missing']} requested bunches were "
                     f"never seen in the raw")
    res['status'] = 'PASS' if not fails else 'FAIL: ' + '; '.join(fails)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('waveform_file', nargs='+', type=Path)
    ap.add_argument('--slim', type=Path)
    ap.add_argument('--edge-tol-ns', type=float, default=8.0,
                    help='a hit this far past a block end still counts as held')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--write-json', action='store_true',
                    help='also write <product>_verify.json beside each product, '
                         'so the fleet verdict can be read from summaries '
                         'instead of grepped out of job logs')
    a = ap.parse_args(argv)

    bad = 0
    for p in a.waveform_file:
        r = verify(p, a.slim, a.edge_tol_ns)
        bad += not r['status'].startswith('PASS')
        if a.write_json:
            Path(p).with_name(Path(p).stem + '_verify.json').write_text(
                json.dumps(r, indent=2, sort_keys=True, default=str))
        if a.json:
            print(json.dumps(r, indent=2, sort_keys=True))
        else:
            print(f"{Path(r['file']).name}")
            print(f"  {r['n_blocks']:,} blocks, {r['n_samples']*2/1e6:.0f} MB "
                  f"of samples, window +-{r['window_ns']} ns")
            if r.get('hits_in_window') is not None:
                print(f"  slim hits in window: {r['hits_in_window']:,}, "
                      f"covered {r['hit_coverage']:.6%} "
                      f"(+{r['hits_edge_within_tol']:,} within "
                      f"{r['edge_tol_ns']:g} ns of a block edge "
                      f"-> {r['hit_coverage_with_edge']:.6%})")
                if r['hits_missing']:
                    print(f"  MISSING {r['hits_missing']:,} "
                          f"({r['hits_no_block_on_channel']:,} have no block "
                          f"at all on their channel)")
                print(f"  {r['samples_per_hit_in_window']:.0f} samples pulled "
                      f"per hit in window (block-driven surplus)")
            if r.get('bunches_requested'):
                print(f"  bunches: {r['bunches_with_blocks']:,}/"
                      f"{r['bunches_requested']:,} carry blocks")
            print(f"  {r['status']}")
            for ex in r.get('missing_examples', [])[:5]:
                print(f"    missing: {ex}")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
