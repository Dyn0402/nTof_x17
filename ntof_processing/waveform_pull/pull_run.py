#!/usr/bin/env python3
"""Pull raw waveforms in the DREAM trigger neighbourhoods for one n_TOF run.

One job = one n_TOF run = every DREAM sub-run that overlaps it, because the raw
is read once and it is the only expensive thing here.

    python -m ntof_processing.waveform_pull.pull_run 224572 \
        --slim-base /media/dylan/data/x17/slim_campaign_2026-08-12 \
        --raw-dir  /media/dylan/data/x17/ntof_raw_224572 \
        --out /tmp/wf --window-ns 5000

The product is block-driven, NOT hit-driven: every zero-suppressed block that
overlaps a window is kept whether or not the PSA found a hit in it.  That is
the whole point -- a hit-driven pull cannot show you what the PSA missed.

Fails, loudly, if a bunch the slim asked for never appeared in the raw that was
scanned.  A short raw file set produces a short product that looks exactly like
a quiet detector, and the campaign has already been bitten once by exactly that
class of silent hole.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import uproot

if __package__ in (None, ''):                       # allow direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'ntof_processing.waveform_pull'

from . import config as C                                          # noqa: E402
from . import windows as W                                         # noqa: E402
from .rawscan import ScanStats, scan_file                          # noqa: E402
from .writer import SegmentWriter                                  # noqa: E402

RAW_RE = re.compile(r'run(\d+)_(\d+)_s1\.raw(\.finished)?$')


def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ------------------------------------------------------------------ discovery
def find_slims(ntof_run: int, slim_base: Path) -> list[Path]:
    """Every slim product built against this n_TOF run, under `slim_base`."""
    pats = (f'**/ntof_hits_*_{ntof_run}.root', f'ntof_hits_*_{ntof_run}.root')
    out: list[Path] = []
    for p in pats:
        out += [q for q in Path(slim_base).glob(p) if q.is_file()]
    return sorted(set(out))


def segment_id(path: Path) -> tuple[str, str]:
    """('run_79', 'stat090_0000') from the file name."""
    m = re.match(r'ntof_hits_(run_\d+)_(.+)_(\d+)\.root$', path.name)
    if not m:
        raise ValueError(f'cannot parse segment from {path.name}')
    return m.group(1), m.group(2)


def find_raw(ntof_run: int, raw_dir: str | None, raw_list: Path | None) -> list[str]:
    if raw_list:
        return [l.strip() for l in Path(raw_list).read_text().splitlines()
                if l.strip()]
    if not raw_dir:
        raise SystemExit('one of --raw-dir / --raw-list is required')
    files = []
    for name in os.listdir(raw_dir):
        m = RAW_RE.match(name)
        if m and int(m.group(1)) == ntof_run:
            files.append((int(m.group(2)), os.path.join(raw_dir, name)))
    return [p for _, p in sorted(files)]


def raw_index(url: str) -> int:
    m = RAW_RE.search(url)
    return int(m.group(2)) if m else -1


# --------------------------------------------------------------------- staging
def stage(url: str, stage_dir: Path) -> Path:
    """xrdcp one raw file to local scratch.  Returns the local path."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    dest = stage_dir / Path(url).name
    if dest.exists():
        return dest
    t0 = time.time()
    r = subprocess.run(['xrdcp', '--force', '--silent', url, str(dest)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'xrdcp failed for {url}: {r.stderr.strip()}')
    mb = dest.stat().st_size / 1e6
    log(f'    staged {dest.name}: {mb:.0f} MB in {time.time()-t0:.0f} s '
        f'({mb/max(time.time()-t0, 1e-9):.0f} MB/s)')
    return dest


# ------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ntof_run', type=int)
    ap.add_argument('--slim-base', required=True, type=Path,
                    help='directory tree holding the slim products')
    ap.add_argument('--raw-dir', help='directory of run<R>_<i>_s1.raw[.finished]')
    ap.add_argument('--raw-list', type=Path, help='file of raw paths or xrootd URLs')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--stage-dir', type=Path,
                    help='xrdcp each raw file here before scanning, then delete')
    ap.add_argument('--window-ns', type=float, default=C.WINDOW_NS)
    ap.add_argument('--control-shift-ns', type=float, default=C.CONTROL_SHIFT_NS)
    ap.add_argument('--no-control', action='store_true',
                    help='drop the +100 us accidental control windows')
    ap.add_argument('--no-pkup', action='store_true')
    ap.add_argument('--keep-flash', action='store_true',
                    help='also keep the mandatory 30 us flash block of every '
                         'bunch (adds ~2.6 GB per segment)')
    ap.add_argument('--ntof-source', type=Path,
                    help='directory of processed n_TOF partials; default is '
                         'the slim pipeline\'s own resolution (done/, then '
                         'completed/<run>/)')
    ap.add_argument('--tflash-sample-blocks', type=int, default=5,
                    help='stretches of 10 bunches to cross-check tflash on '
                         '(default 5); the full read costs ~15 min per segment')
    ap.add_argument('--no-tflash-crosscheck', action='store_true',
                    help='skip reading the processed file; back-solve tflash '
                         'from the slim alone (needs a hit per det and bunch)')
    ap.add_argument('--max-files', type=int, help='stop after N raw files (dev)')
    ap.add_argument('--bunches', help='restrict to these bunches: a comma list '
                                      'or a file of one per line. For targeted '
                                      're-pulls and for partial-raw tests.')
    a = ap.parse_args(argv)

    only = None
    if a.bunches:
        txt = Path(a.bunches).read_text() if Path(a.bunches).is_file() else a.bunches
        only = {int(x) for x in txt.replace(',', '\n').split() if x.strip()}

    t_start = time.time()
    slims = find_slims(a.ntof_run, a.slim_base)
    if not slims:
        raise SystemExit(f'no slim products for {a.ntof_run} under {a.slim_base}')
    log(f'{a.ntof_run}: {len(slims)} segment(s)')

    # ------------------------------------------------ per-segment windows
    segments = []
    for sp in slims:
        dream_run, dream_subrun = segment_id(sp)
        with uproot.open(sp) as f:
            ev = f['events'].arrays(
                ['eventId', 'bunch', 't_pred_ns', 'is_flash', 'matched'],
                library='np')
            hits = f['hits'].arrays(
                ['eventId', 'det', 'detn', 'tof', 'dt_ns', 'is_control'],
                library='np')

        tf_slim = W.tflash_from_slim(hits, ev, a.control_shift_ns)
        if a.no_tflash_crosscheck:
            table = {k: (v, 1) for k, (v, _n, _s) in tf_slim.items()}
            report = {'n_processed': 0, 'n_slim_only': len(table),
                      'n_cross_checked': 0, 'n_total': len(table)}
        else:
            # Cross-check on a SAMPLE, then keep the slim's own values for
            # everything. Reading every bunch of all twelve trees costs ~15 min
            # per segment -- as much as the raw read it protects -- and a
            # systematic offset in the bridge shows up in fifty bunches just as
            # plainly. Validated 2026-08-12 on 224615: full read, 29,583
            # channel-bunches cross-checked, zero disagreement at 1 ns.
            sample = W.sample_bunches(ev['bunch'], a.tflash_sample_blocks)
            tf_proc = W.tflash_from_processed(a.ntof_run, sample,
                                              ntof_source=a.ntof_source)
            _, report = W.reconcile(tf_proc, {k: v for k, v in tf_slim.items()
                                              if k[2] in set(sample.tolist())})
            table = {k: (v, 1) for k, (v, _n, _s) in tf_slim.items()}
            for k, v in tf_proc.items():
                table.setdefault(k, (v, 0))
            report['n_sampled_bunches'] = int(sample.size)
        table, fill = W.fill_gaps(table, ev['bunch'])
        report.update(fill)
        log(f'  {dream_run}/{dream_subrun}: {len(ev["eventId"]):,} triggers, '
            f'tflash {len(table):,} (det, bunch) '
            f'[{report["n_cross_checked"]:,} cross-checked, '
            f'{report["n_slim_only"]:,} slim-only, '
            f'{report["n_filled_from_neighbour_bunch"]:,} from a neighbour '
            f'bunch, {report["n_filled_dead_channel"]:,} dead-channel]')
        if report['dets_with_unstable_tflash']:
            log(f'    WARNING: tflash is not near-constant on detectors '
                f'{report["dets_with_unstable_tflash"]} -- the neighbour-bunch '
                f'fill is not safe there')
        if report['dets_without_any_tflash']:
            log(f'    WARNING: no tflash at all for '
                f'{report["dets_without_any_tflash"]}: they get no windows')

        wrep: dict = {}
        win = W.build(ev, table, window_ns=a.window_ns,
                      control_shift_ns=a.control_shift_ns,
                      with_control=not a.no_control,
                      extra_dets=() if a.no_pkup else (C.PKUP_DET,),
                      report=wrep)
        report.update(wrep)
        if wrep['n_skipped_det_bunch']:
            log(f'    WARNING: {wrep["n_skipped_det_bunch"]:,} (det, bunch) '
                f'have no tflash and get no window: {wrep["skipped_det_bunch"]}')
        if only is not None:
            win = {b: v for b, v in win.items() if b in only}
        out = a.out / 'runs' / dream_run / dream_subrun / 'ntof_wf'
        segments.append({
            'dream_run': dream_run, 'dream_subrun': dream_subrun,
            'slim': sp, 'events': ev, 'windows': win,
            'tflash': table, 'tflash_report': report,
            'writer': SegmentWriter(
                out / C.out_name(dream_run, dream_subrun, a.ntof_run))})

    # ------------------------------------------- union of windows, bunch routing
    union: dict[int, dict[str, np.ndarray]] = {}
    routing: dict[int, list[int]] = {}
    for si, seg in enumerate(segments):
        for b, per_det in seg['windows'].items():
            routing.setdefault(b, []).append(si)
            tgt = union.setdefault(b, {})
            for det, per_chan in per_det.items():
                t = tgt.setdefault(det, {})
                for chan, iv in per_chan.items():
                    t[chan] = iv if chan not in t else \
                        W.merge_intervals(np.r_[t[chan][:, 0], iv[:, 0]],
                                          np.r_[t[chan][:, 1], iv[:, 1]])
    want_samples = W.requested_samples(union)
    log(f'  {len(union):,} bunches requested, '
        f'{want_samples/1e9:.2f} G sample-ns of window '
        f'(+-{a.window_ns:g} ns{"" if a.no_control else " + control"})')

    # ------------------------------------------------------------- the raw pass
    raws = find_raw(a.ntof_run, a.raw_dir, a.raw_list)
    if a.max_files:
        raws = raws[:a.max_files]
    if not raws:
        raise SystemExit(f'no raw files found for {a.ntof_run}')
    idx = [raw_index(r) for r in raws]
    gaps = sorted(set(range(min(idx), max(idx) + 1)) - set(idx)) if idx else []
    log(f'  {len(raws)} raw files, indices {min(idx)}..{max(idx)}'
        + (f', MISSING {len(gaps)}: {gaps[:20]}' if gaps else ', contiguous'))

    stats = ScanStats()
    for i, url in enumerate(raws):
        local = Path(url)
        staged = False
        if a.stage_dir and not local.exists():
            local = stage(url, a.stage_dir)
            staged = True
        st = ScanStats()
        try:
            for bunch, det, chan, tof0, samples in scan_file(
                    local, union, keep_flash=a.keep_flash, stats=st):
                code = C.DET_CODE.get(det, 255)
                for si in routing.get(bunch, ()):
                    segments[si]['writer'].add(bunch, code, chan, tof0, samples)
        finally:
            if staged:
                local.unlink(missing_ok=True)
        stats.merge(st)
        log(f'  [{i+1}/{len(raws)}] {Path(url).name}: {st.events} events, '
            f'{st.blocks_kept:,} blocks kept of {st.blocks_seen:,} seen, '
            f'{st.samples_kept*2/1e6:.0f} MB')

    # ------------------------------------------------------------- the hard check
    missing = sorted(set(union) - stats.bunches)
    coverage = 1 - len(missing) / max(len(union), 1)
    log(f'  scan done: {stats.events:,} events, {len(stats.bunches):,} bunches '
        f'seen, {stats.blocks_kept:,} blocks kept, '
        f'{stats.samples_kept*2/1e6:.0f} MB of samples '
        f'[{time.time()-t_start:.0f} s]')

    meta_common = {
        'ntof_run': a.ntof_run,
        'window_ns': a.window_ns,
        'control_shift_ns': None if a.no_control else a.control_shift_ns,
        'keep_flash': a.keep_flash,
        'pkup': not a.no_pkup,
        'pre_samples': C.PRE_SAMPLES,
        'sample_dtype': C.SAMPLE_DTYPE,
        'block_driven': True,
        'raw_files': [Path(r).name for r in raws],
        'raw_file_gaps': gaps,
        'bunches_requested': len(union),
        'bunches_seen': len(stats.bunches),
        'bunches_missing': missing[:200],
        'n_bunches_missing': len(missing),
        'coverage': coverage,
        'scan': {k: getattr(stats, k) for k in
                 ('events', 'banks_seen', 'banks_read', 'blocks_seen',
                  'blocks_kept', 'samples_kept', 'bytes_read')},
        'runtime_s': round(time.time() - t_start, 1),
        'produced': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    for seg in segments:
        # what this segment actually got: requested AND present in the raw.
        # verify.py measures hit coverage on exactly this set, so a partial raw
        # set shows up as missing bunches rather than as fake uncovered hits.
        pulled = sorted(set(seg['windows']) & stats.bunches)
        seg['writer'].close(
            seg['events'], seg['tflash'],
            dict(meta_common, dream_run=seg['dream_run'],
                 dream_subrun=seg['dream_subrun'], slim=str(seg['slim']),
                 tflash_report=seg['tflash_report'],
                 n_triggers=int(len(seg['events']['eventId'])),
                 n_bunches=len(seg['windows']),
                 bunches_pulled=pulled,
                 n_bunches_missing=len(set(seg['windows']) - stats.bunches)))
        log(f'  wrote {seg["writer"].path.name}: '
            f'{seg["writer"].n_blocks:,} blocks, '
            f'{seg["writer"].path.stat().st_size/1e6:.0f} MB')

    (a.out / f'pull_{a.ntof_run}.json').write_text(
        json.dumps(meta_common, indent=2, sort_keys=True, default=str))

    if missing:
        log(f'FAIL: {len(missing)} of {len(union)} requested bunches never '
            f'appeared in the {len(raws)} raw files scanned. The product is '
            f'incomplete and a missing bunch is indistinguishable from a quiet '
            f'one downstream. First: {missing[:20]}')
        return 2
    if gaps:
        log(f'FAIL: raw file index has {len(gaps)} gaps: {gaps[:20]}')
        return 2
    log('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
