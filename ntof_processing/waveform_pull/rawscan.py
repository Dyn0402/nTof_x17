"""One streaming pass over a raw stream1 file, keeping only the blocks that
overlap a requested time window.

The reader is deliberately NOT the vendored `ntof_raw.iter_banks`: that one
reads every bank payload into memory, and most of a raw file is bunches and
channels we do not want.  This one seeks past them, which is what makes a
0.5 TB run affordable.

Time base: a block is described in `tof` space (the PSA's own), so
    tof0 = start - PRE_SAMPLES   (start > 0)
    tof0 = 0                     (start == 0, the mandatory flash block)
and the block covers [tof0, tof0 + n).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from . import config as C

HDR = struct.Struct('<4sIII')
HDR_SIZE = HDR.size
TOP_TAGS = {b'RCTR', b'MODH', b'EVEH', b'ADDH', b'ACQC', b'EVDH'}
ACQC_HDR = struct.Struct('<4sII')      # det, chan, flags


@dataclass
class ScanStats:
    """What the pass saw, so a caller can tell 'quiet' from 'absent'."""
    events: int = 0
    bunches: set = field(default_factory=set)
    banks_seen: int = 0
    banks_read: int = 0
    blocks_seen: int = 0
    blocks_kept: int = 0
    samples_kept: int = 0
    bytes_read: int = 0

    def merge(self, other: 'ScanStats') -> None:
        self.events += other.events
        self.bunches |= other.bunches
        for k in ('banks_seen', 'banks_read', 'blocks_seen', 'blocks_kept',
                  'samples_kept', 'bytes_read'):
            setattr(self, k, getattr(self, k) + getattr(other, k))


def merge_intervals(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Sort and coalesce touching/overlapping [lo, hi) into an (K, 2) array."""
    if len(lo) == 0:
        return np.zeros((0, 2), np.int64)
    lo = np.asarray(lo, np.int64)
    hi = np.asarray(hi, np.int64)
    o = np.argsort(lo, kind='stable')
    lo, hi = lo[o], hi[o]
    keep_lo = [lo[0]]
    keep_hi = [hi[0]]
    for i in range(1, len(lo)):
        if lo[i] <= keep_hi[-1]:
            if hi[i] > keep_hi[-1]:
                keep_hi[-1] = hi[i]
        else:
            keep_lo.append(lo[i])
            keep_hi.append(hi[i])
    return np.stack([np.array(keep_lo, np.int64),
                     np.array(keep_hi, np.int64)], axis=1)


def overlaps(intervals: np.ndarray, blo: np.ndarray, bhi: np.ndarray) -> np.ndarray:
    """Vectorised: does [blo, bhi) hit any of the merged, sorted `intervals`?

    Because the intervals are disjoint and sorted, the only candidate is the
    last one starting at or before `bhi`; a block overlaps iff that candidate
    ends after `blo`.
    """
    if len(intervals) == 0 or len(blo) == 0:
        return np.zeros(len(blo), bool)
    j = np.searchsorted(intervals[:, 0], bhi, side='left') - 1
    ok = j >= 0
    j = np.clip(j, 0, len(intervals) - 1)
    return ok & (intervals[j, 1] > blo)


def _index_blocks(payload: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk the block chain of an ACQC payload (samples not touched).

    Returns (start, n, offset) where `offset` is the byte position of the first
    sample within `payload`.  The walk is serial by construction -- a block's
    length is what locates the next one.
    """
    starts, ns, offs = [], [], []
    pos, end = 0, len(payload)
    while pos + 16 <= end:
        start, n = struct.unpack_from('<QQ', payload, pos)
        if n == 0 or pos + 16 + 2 * n > end + 2:
            break                                   # trailing pad word
        starts.append(start)
        ns.append(n)
        offs.append(pos + 16)
        pos += 16 + 2 * n
    return (np.array(starts, np.int64), np.array(ns, np.int64),
            np.array(offs, np.int64))


def scan_file(path, windows: dict, keep_flash: bool = False,
              stats: ScanStats | None = None):
    """Yield (bunch, det, chan, tof0, samples) for every kept block.

    `windows` maps bunch -> {det_name: {chan: (K, 2) int64 merged [lo, hi)}} in
    tof ns.  A bunch absent from `windows` is skipped with seeks and costs
    nothing but its EVEH header; a detector or channel absent from a bunch's
    entry likewise.  The two levels exist so a whole detector's banks can be
    skipped without decoding a channel id.

    `samples` is a fresh int16 copy, so the caller may hold it after the
    payload it came from is released.
    """
    st = stats if stats is not None else ScanStats()
    with open(path, 'rb') as f:
        bunch = None
        want = None
        while True:
            head = f.read(HDR_SIZE)
            if len(head) < HDR_SIZE:
                return                              # clean end, or truncated tail
            tag, _ver, _res, length = HDR.unpack(head)
            nbytes = length * 4
            if tag not in TOP_TAGS or nbytes < 0 or nbytes > (1 << 31):
                raise ValueError(f'{path}: bad bank at {f.tell() - HDR_SIZE}: '
                                 f'tag={tag!r} len={length}')
            st.bytes_read += HDR_SIZE

            if tag == b'EVEH':
                payload = f.read(nbytes)
                if len(payload) < nbytes:
                    return
                st.bytes_read += nbytes
                bunch = int(struct.unpack_from('<10I', payload, 0)[1])
                st.events += 1
                st.bunches.add(bunch)
                want = windows.get(bunch)
                continue

            if tag != b'ACQC':
                f.seek(nbytes, 1)
                continue

            st.banks_seen += 1
            if want is None or nbytes < ACQC_HDR.size:
                f.seek(nbytes, 1)
                continue
            hdr = f.read(ACQC_HDR.size)
            st.bytes_read += ACQC_HDR.size
            det = hdr[0:4].decode('ascii', 'replace')
            chan = int(struct.unpack_from('<I', hdr, 4)[0])
            per_chan = want.get(det)
            iv = per_chan.get(chan) if per_chan is not None else None
            if iv is None:
                f.seek(nbytes - ACQC_HDR.size, 1)
                continue

            payload = f.read(nbytes - ACQC_HDR.size)
            if len(payload) < nbytes - ACQC_HDR.size:
                return
            st.bytes_read += len(payload)
            st.banks_read += 1

            start, n, off = _index_blocks(payload)
            if len(start) == 0:
                continue
            st.blocks_seen += len(start)
            tof0 = np.where(start > 0, start - C.PRE_SAMPLES, 0)
            sel = overlaps(iv, tof0, tof0 + n)
            if not keep_flash:
                sel &= start > 0
            for k in np.nonzero(sel)[0]:
                avail = (len(payload) - off[k]) // C.BYTES_PER_SAMPLE
                cnt = int(min(n[k], avail))
                s = np.frombuffer(payload, dtype=C.SAMPLE_DTYPE,
                                  offset=int(off[k]), count=cnt).copy()
                st.blocks_kept += 1
                st.samples_kept += cnt
                yield bunch, det, chan, int(tof0[k]), s
