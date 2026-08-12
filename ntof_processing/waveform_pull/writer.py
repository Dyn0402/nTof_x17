"""Buffered writer for one segment's waveform product.

A segment at +-5 us is a few GB of samples, so nothing is held whole: blocks are
buffered to a size cap and flushed into the tree as they arrive.  The raw files
of a run are time-ordered, so a segment's blocks arrive contiguously and the
buffer stays small.
"""
from __future__ import annotations

import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from . import config as C

FLUSH_SAMPLES = 20_000_000        # ~40 MB of int16 per flush


class SegmentWriter:
    """`blocks` grows incrementally; `events`, `tflash` and the metadata are
    written once at close, so a half-written file is recognisably half-written
    (no `events` tree) rather than quietly short."""

    def __init__(self, path: Path, compression=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # ZLIB(1): on real sample bytes level 1 gives 2.09x against level 6's
        # 2.16x (measured per family on 224572), for a fraction of the CPU.
        self._f = uproot.recreate(
            self.path, compression=compression or uproot.ZLIB(1))
        # The tree is created from the FIRST batch rather than declared with
        # mktree, so no awkward type object has to be spelled: this runs against
        # awkward 2.8 / uproot 5.7 locally and awkward 1.10 / uproot 4.3 on
        # LCG_105, whose type APIs do not agree.
        self._created = False
        self._buf = {k: [] for k in ('bunch', 'det', 'detn', 'tof0', 'n')}
        self._samples: list[np.ndarray] = []
        self._pending = 0
        self.n_blocks = 0
        self.n_samples = 0

    def add(self, bunch: int, det: int, detn: int, tof0: int,
            samples: np.ndarray) -> None:
        self._buf['bunch'].append(bunch)
        self._buf['det'].append(det)
        self._buf['detn'].append(detn)
        self._buf['tof0'].append(tof0)
        self._buf['n'].append(len(samples))
        self._samples.append(samples)
        self._pending += len(samples)
        self.n_blocks += 1
        self.n_samples += len(samples)
        if self._pending >= FLUSH_SAMPLES:
            self.flush()

    def flush(self) -> None:
        if not self._samples:
            return
        # Build the jagged array from a flat buffer and per-block counts.
        # `ak.Array(list_of_ndarrays)` is the obvious spelling and it is 60 s
        # per 10 k blocks -- three orders of magnitude slower than this, and it
        # would have dominated the whole campaign. Measured 2026-08-12.
        # `ak.unflatten` rather than a hand-built ListOffsetArray because the
        # layout classes moved between awkward 1 and 2 and LCG_105 has 1.10;
        # unflatten is present and equally cheap in both.
        n = np.array(self._buf['n'], np.int32)
        payload = {
            'bunch': np.array(self._buf['bunch'], np.int32),
            'det': np.array(self._buf['det'], np.uint8),
            'detn': np.array(self._buf['detn'], np.int32),
            'tof0': np.array(self._buf['tof0'], np.int64),
            'n': n,
            'samples': ak.unflatten(np.concatenate(self._samples),
                                    n.astype(np.int64))}
        if self._created:
            self._f['blocks'].extend(payload)
        else:
            self._f['blocks'] = payload
            self._created = True
        for v in self._buf.values():
            v.clear()
        self._samples = []
        self._pending = 0

    def close(self, events: dict, tflash_table: dict, meta: dict) -> None:
        """`events` is the slim's own event table, trimmed; `tflash_table` is
        {(det, bunch): (value, source)}.  Both are copied in so the product
        stands alone -- a window can be recomputed from this file with no
        reference to the slim it came from."""
        self.flush()
        if not self._created:
            # A segment can legitimately end up with no blocks (every bunch
            # missing from the raw). Write the tree empty rather than omitting
            # it, so a consumer sees "nothing here" instead of "malformed".
            self._f['blocks'] = {
                'bunch': np.zeros(0, np.int32), 'det': np.zeros(0, np.uint8),
                'detn': np.zeros(0, np.int32), 'tof0': np.zeros(0, np.int64),
                'n': np.zeros(0, np.int32),
                'samples': ak.unflatten(np.zeros(0, np.int16),
                                        np.zeros(0, np.int64))}
            self._created = True
        self._f['events'] = {
            'eventId': np.asarray(events['eventId'], np.uint64),
            'bunch': np.asarray(events['bunch'], np.int32),
            't_pred_ns': np.asarray(events['t_pred_ns'], np.float64),
            'is_flash': np.asarray(events['is_flash'], np.int8),
            'matched': np.asarray(events['matched'], np.int8)}
        if tflash_table:
            keys = sorted(tflash_table)
            self._f['tflash'] = {
                'det': np.array([k[0] for k in keys], np.uint8),
                'detn': np.array([k[1] for k in keys], np.int32),
                'bunch': np.array([k[2] for k in keys], np.int32),
                'tflash_ns': np.array([tflash_table[k][0] for k in keys],
                                      np.float64),
                'source': np.array([tflash_table[k][1] for k in keys], np.uint8)}
        self._f.close()
        meta = dict(meta, n_blocks=self.n_blocks, n_samples=self.n_samples,
                    bytes_samples=self.n_samples * C.BYTES_PER_SAMPLE,
                    file_bytes=self.path.stat().st_size)
        (self.path.parent / (self.path.stem + '_provenance.json')).write_text(
            json.dumps(meta, indent=2, sort_keys=True, default=str))

    def abort(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
