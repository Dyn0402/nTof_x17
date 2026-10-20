#!/usr/bin/env python3
"""Pull the raw waveforms of the gamma flash and its recovery from a few raw
stream1 files of one n_TOF run.

    python -m ntof_processing.flash_pull.pull 224572 \
        --raw /media/dylan/data/x17/ntof_raw_224572/head_20.bin --idx 20 --out /tmp/flash

For EVERY bunch in each file and every scintillator channel plus PKUP, every
zero-suppressed block overlapping `config.WINDOWS_NS` is kept whole, the
always-kept flash block included.  No slim, no PSA, no time calibration is
needed to decide what to keep, so the product is the raw restricted in time and
nothing else.  One product per raw file:

    flash_<run>_<idx>.root      trees `blocks`, `bunches`, `channels`
    flash_<run>_<idx>_provenance.json   written LAST: its presence = finished

`blocks`   bunch, det (config.DET_CODE: 0-11 as the slim, 12 = PKUP), detn
           (raw ACQC channel), tof0, n, samples (signed int16).  Sample j of a
           block is at tof = tof0 + j; tof0 = 0 for the flash block.
`bunches`  one row per bunch in the file: blocks and samples kept, and, from the
           official processed file when it exists, PulseIntensity, psTime,
           date, time and the PKUP flash tof (NaN / -1 otherwise).
`channels` one row per (bunch, det, detn) seen: whether its flash block was
           present and how long it is -- so "quiet" and "absent" differ.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import uproot

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'ntof_processing.flash_pull'

from . import config as F                                      # noqa: E402
from ..waveform_pull.rawscan import ScanStats, scan_file       # noqa: E402
from ..waveform_pull.writer import SegmentWriter               # noqa: E402


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


class _AnyChannel(dict):
    """`per_chan.get(chan)` -> the same intervals for every channel, so a channel
    nobody listed (a spare, a re-cabled one) is kept rather than silently dropped."""

    def __init__(self, iv):
        super().__init__()
        self._iv = iv

    def get(self, _chan, default=None):
        return self._iv


class _AnyBunch(dict):
    def __init__(self, per_det):
        super().__init__()
        self._d = per_det

    def get(self, _bunch, default=None):
        return self._d


def windows() -> _AnyBunch:
    iv = np.array(F.WINDOWS_NS, np.int64)
    return _AnyBunch({d: _AnyChannel(iv) for d in F.DETS})


def official_bunches(run: int) -> dict:
    """bunch -> (PulseIntensity, psTime, date, time, PKUP flash tof), or {}."""
    f = F.OFFICIAL / f'run{run}.root'
    if not f.exists():
        return {}
    try:
        t = uproot.open(f)['PKUP']
        a = t.arrays(['BunchNumber', 'PulseIntensity', 'psTime', 'date', 'time', 'tof', 'amp'],
                     library='np')
    except Exception as e:                       # noqa: BLE001 -- metadata only
        log(f'  official PKUP unreadable ({e}); bunch metadata left empty')
        return {}
    o = np.argsort(-a['amp'])                     # the flash: largest PKUP hit
    b, i = np.unique(a['BunchNumber'][o], return_index=True)
    k = o[i]
    return {int(x): (float(a['PulseIntensity'][j]), float(a['psTime'][j]), int(a['date'][j]),
                     int(a['time'][j]), float(a['tof'][j])) for x, j in zip(b, k)}


def _put(f, name, payload):
    """Write `payload` as a TTree on every uproot in play.  uproot >= 5.7 turns
    `f[name] = dict` into an RNTuple; uproot 4.3 (LCG_105, the batch workers)
    has no RNTuple and makes a TTree.  One product format, so spell it out
    where `mktree` takes type strings."""
    if not hasattr(f, 'mkrntuple'):
        f[name] = payload
        return
    types = {k: (f'var * {np.asarray(v[0] if len(v) else np.zeros(0, np.int16)).dtype}'
                 if not isinstance(v, np.ndarray) else v.dtype) for k, v in payload.items()}
    f.mktree(name, types)
    f[name].extend(payload)


class FlashWriter(SegmentWriter):
    def flush(self) -> None:
        if not self._samples:
            return
        import awkward as ak
        n = np.array(self._buf['n'], np.int32)
        payload = {
            'bunch': np.array(self._buf['bunch'], np.int32),
            'det': np.array(self._buf['det'], np.uint8),
            'detn': np.array(self._buf['detn'], np.int32),
            'tof0': np.array(self._buf['tof0'], np.int64),
            'n': n,
            'samples': ak.unflatten(np.concatenate(self._samples), n.astype(np.int64))}
        if self._created:
            self._f['blocks'].extend(payload)
        else:
            _put(self._f, 'blocks', payload)
            self._created = True
        for v in self._buf.values():
            v.clear()
        self._samples = []
        self._pending = 0

    def close(self, bunches: dict, chans: dict, meta: dict) -> None:   # noqa: D401
        self.flush()
        if not self._created:
            raise RuntimeError('no blocks at all -- refusing to write an empty product')
        bl = sorted(bunches)
        cols = list(zip(*[bunches[b] for b in bl])) if bl else [[]] * 9
        _put(self._f, 'bunches', {
            'bunch': np.array(bl, np.int32),
            'n_blocks': np.array(cols[0], np.int32), 'n_samples': np.array(cols[1], np.int64),
            'n_channels': np.array(cols[2], np.int32),
            'PulseIntensity': np.array(cols[3], np.float64), 'psTime': np.array(cols[4], np.float64),
            'date': np.array(cols[5], np.int32), 'time': np.array(cols[6], np.int32),
            'tof_pkup_official': np.array(cols[7], np.float64),
            'has_official': np.array(cols[8], np.int8)})
        ck = sorted(chans)
        _put(self._f, 'channels', {
            'bunch': np.array([k[0] for k in ck], np.int32),
            'det': np.array([k[1] for k in ck], np.uint8),
            'detn': np.array([k[2] for k in ck], np.int32),
            'flash_block_n': np.array([chans[k][0] for k in ck], np.int32),
            'n_blocks': np.array([chans[k][1] for k in ck], np.int32)})
        self._f.close()
        meta = dict(meta, n_blocks=self.n_blocks, n_samples=self.n_samples,
                    bytes_samples=self.n_samples * 2, file_bytes=self.path.stat().st_size)
        (self.path.parent / (self.path.stem + '_provenance.json')).write_text(
            json.dumps(meta, indent=2, sort_keys=True, default=str))


def stage(url: str, stage_dir: Path) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    dest = stage_dir / Path(url).name
    t0 = time.time()
    r = subprocess.run(['xrdcp', '--force', '--silent', url, str(dest)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'xrdcp failed for {url}: {r.stderr.strip()}')
    mb = dest.stat().st_size / 1e6
    log(f'  staged {dest.name}: {mb:.0f} MB in {time.time() - t0:.0f} s')
    return dest


def pull_one(run: int, idx: int, src: str, out: Path, stage_dir: Path | None, off: dict) -> dict:
    local = stage(src, stage_dir) if (stage_dir and '://' in src) else Path(src)
    path = out / F.out_name(run, idx)
    w = FlashWriter(path)
    st = ScanStats()
    t0 = time.time()
    per_bunch: dict = {}
    chans: dict = {}
    try:
        for bunch, det, chan, tof0, s in scan_file(local, windows(), keep_flash=True, stats=st):
            code = F.DET_CODE[det]
            w.add(bunch, code, chan, tof0, s)
            pb = per_bunch.setdefault(bunch, [0, 0])
            pb[0] += 1
            pb[1] += len(s)
            c = chans.setdefault((bunch, code, chan), [0, 0])
            c[1] += 1
            if tof0 == 0:
                c[0] = len(s)
    except Exception:
        w.abort()
        raise
    finally:
        if local != Path(src):
            local.unlink(missing_ok=True)       # never leave raw in the condor scratch
    bunches = {}
    for b in st.bunches:
        nb, ns = per_bunch.get(b, (0, 0))
        nch = sum(1 for k in chans if k[0] == b)
        o = off.get(b)
        bunches[b] = (nb, ns, nch, *(o if o else (np.nan, np.nan, -1, -1, np.nan)), int(o is not None))
    no_flash = sorted({(k[1], k[2]) for k, v in chans.items() if v[0] == 0})
    # a bunch with fewer channels than the file's fullest one was cut short
    # (a truncated file) or lost a channel; either way it must not pass as quiet
    nmax = max((v[2] for v in bunches.values()), default=0)
    incomplete = sorted(b for b, v in bunches.items() if v[2] < nmax)
    meta = dict(schema=F.SCHEMA, run=run, raw_index=idx, raw_source=src, windows_ns=F.WINDOWS_NS,
                det_code=F.DET_CODE, events=st.events, bunches=len(st.bunches),
                bunches_without_blocks=sorted(set(st.bunches) - set(per_bunch)),
                channel_bunches=len(chans), channels_missing_flash_block=no_flash,
                bunches_incomplete=incomplete,
                blocks_short=st.blocks_short, bytes_read=st.bytes_read,
                has_official=bool(off), seconds=round(time.time() - t0, 1),
                created=time.strftime('%Y-%m-%dT%H:%M:%S'))
    w.close(bunches, chans, meta)
    log(f'  {path.name}: {len(st.bunches)} bunches, {w.n_blocks} blocks, '
        f'{w.n_samples * 2 / 1e6:.0f} MB samples -> {path.stat().st_size / 1e6:.0f} MB, '
        f'short {st.blocks_short}, {meta["seconds"]} s')
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('run', type=int)
    ap.add_argument('--raw', nargs='+', required=True, help='raw paths or xrootd URLs')
    ap.add_argument('--idx', nargs='+', type=int, help='raw file indices (default: parsed from names)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--stage-dir', type=Path, help='xrdcp remote files here first, delete after')
    a = ap.parse_args(argv)
    import re
    idx = a.idx or [int(re.search(r'_(\d+)_s1\.raw', r).group(1)) for r in a.raw]
    a.out.mkdir(parents=True, exist_ok=True)
    off = official_bunches(a.run)
    log(f'run {a.run}: {len(a.raw)} raw files, official metadata for {len(off)} bunches')
    bad = 0
    for i, src in zip(idx, a.raw):
        m = pull_one(a.run, i, src, a.out, a.stage_dir, off)
        if m['bunches_without_blocks'] or m['blocks_short'] or m['bunches_incomplete']:
            bad += 1
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
