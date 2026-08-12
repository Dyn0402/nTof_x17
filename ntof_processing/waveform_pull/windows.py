"""Turn a slim product into the time windows the raw scan must keep.

The slim's `t_pred_ns` lives in **t_since_flash** space; a raw block lives in
**tof** space.  The bridge is `tflash`, which is per detector AND per bunch:

    tof_window_centre(det, bunch, event) = t_pred_ns[event] + tflash[det, bunch]

`tflash` is taken UNREPAIRED, because that is what the slim matched on
(`slim.py` reads its hits with `repair_tflash=False` and absorbs the per-arm
offsets in the fit).  Using the repaired value here would move the window off
the very hits the slim found.

Two independent sources are used and compared, because getting this wrong
shifts every window by a few hundred ns and would be invisible in the output:

  * the processed n_TOF file's own `tflash` branch  -- authoritative, complete
  * back-solved from the slim itself, tflash = tof - dt_ns - t_pred  -- exact
    wherever a hit exists, and available for nothing else

Disagreement beyond a nanosecond is a hard failure, not a warning.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .rawscan import merge_intervals

TFLASH_TOL_NS = 1.0


def tflash_from_slim(hits: dict, events: dict,
                     control_shift_ns=C.CONTROL_SHIFT_NS) -> dict:
    """{(det, detn, bunch): (value, n_hits, spread_ns)} back-solved from the slim.

    Keyed per CHANNEL, and that is not fussiness: tflash differs between the
    eight channels of one wall detector by up to 23 ns while being constant to
    0.00 ns within a channel (measured, det0/bunch398 of the reference pair). A
    per-detector key pools those eight into one number, and the pooled value
    then disagrees with the processed file's own pooled value by 1-15 ns purely
    from how the mixture fell out -- which is what the cross-check tripped on.

    Control hits count too, with their +100 us shift removed. Excluding them
    (the obvious reading of 'use the signal hits') silently loses every
    channel whose only hits are control ones -- not a rare corner: on the
    reference pair LIQC in bunches 398/399 is exactly that, and it then got no
    window and no waveforms at all.
    """
    eid = np.asarray(events['eventId'])
    o = np.argsort(eid)
    eid_s, pred_s, bun_s = eid[o], np.asarray(events['t_pred_ns'])[o], \
        np.asarray(events['bunch'])[o]

    if np.size(hits['eventId']) == 0:
        return {}
    j = np.searchsorted(eid_s, np.asarray(hits['eventId']))
    j = np.clip(j, 0, len(eid_s) - 1)
    shift = np.where(np.asarray(hits['is_control']) != 0, control_shift_ns, 0.0)
    tf = (np.asarray(hits['tof']) - np.asarray(hits['dt_ns'])
          - pred_s[j] - shift)
    key = ((np.asarray(hits['det']).astype(np.int64) * 1000
            + np.asarray(hits['detn']).astype(np.int64)) * 10_000_000 + bun_s[j])

    out = {}
    order = np.argsort(key, kind='stable')
    key, tf = key[order], tf[order]
    edges = np.flatnonzero(np.diff(key)) + 1
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(key)]):
        v = tf[lo:hi]
        k = int(key[lo])
        out[(k // 10_000_000 // 1000, k // 10_000_000 % 1000, k % 10_000_000)] = (
            float(np.median(v)), int(hi - lo), float(v.max() - v.min()))
    return out


def sample_bunches(bunches, n_blocks=5, block=10):
    """A few short CONTIGUOUS stretches spread across the run.

    The cross-check exists to catch a systematic offset in the time bridge, and
    a systematic offset shows up in fifty bunches as clearly as in a thousand.
    Reading every bunch of every tree costs ~15 min per segment -- as much as
    the raw read it is meant to protect. Contiguous stretches, not a scatter,
    because `read_bunches` turns each run of consecutive bunches into ONE entry
    range and a scattered sample would be one read apiece.
    """
    b = np.unique(np.asarray(bunches, np.int64))
    if b.size <= n_blocks * block:
        return b
    starts = np.linspace(0, b.size - block, n_blocks).astype(int)
    return np.unique(np.concatenate([b[s:s + block] for s in starts]))


def tflash_from_processed(ntof_run: int, bunches, dets=C.SCINT_DETS,
                          ntof_source=None) -> dict:
    """{(det_code, detn, bunch): value} from the processed file's `tflash`.

    Binding is delegated to `slim._bind_ntof`, not re-implemented here.
    `ntof_io.ntof_paths` resolves against the LOCAL staging tree
    (`$X17_BEAM_JULY/ntof_data/`), which does not exist on a worker, and its
    bunch-index and tflash caches are keyed by run number alone -- so an official
    and a reprocessed run224572 sharing a cache directory silently mix
    (`../REVIEW.md` section 5). `_bind_ntof` points the module at the EOS file
    set and fingerprints the cache on it; duplicating that here would be one
    more copy to keep in step with it.
    """
    from ntof_dream_merge import ntof_io
    from ntof_processing.slim_pipeline.slim import Segment, _bind_ntof

    _bind_ntof(Segment(dream_run='', dream_subrun='', ntof_run=ntof_run,
                       ntof_source=ntof_source))

    bunches = np.unique(np.asarray(bunches, np.int64))
    out = {}
    for det in dets:
        a = ntof_io.read_bunches(ntof_run, det, bunches,
                                 branches=('BunchNumber', 'detn', 'tflash'),
                                 repair_tflash=False)
        if a['tflash'].size == 0:
            continue
        k = (a['detn'].astype(np.int64) * 10_000_000
             + a['BunchNumber'].astype(np.int64))
        t = a['tflash'].astype(np.float64)
        o = np.argsort(k, kind='stable')
        k, t = k[o], t[o]
        edges = np.flatnonzero(np.diff(k)) + 1
        for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(k)]):
            out[(C.DET_CODE[det], int(k[lo] // 10_000_000),
                 int(k[lo] % 10_000_000))] = float(np.median(t[lo:hi]))
    return out


def reconcile(from_proc: dict, from_slim: dict, tol=TFLASH_TOL_NS):
    """Merge the two sources; the processed value wins where both exist.

    Returns (table, report).  `table` is {(det, bunch): (value, source)} with
    source 0 = processed, 1 = slim-derived.  Raises on disagreement.
    """
    bad = []
    for k, (v_slim, n, spread) in from_slim.items():
        if k in from_proc and abs(from_proc[k] - v_slim) > tol:
            bad.append((k, from_proc[k], v_slim, n, spread))
    if bad:
        head = '\n'.join(f'    det={d} detn={c} bunch={b}: processed {p:.2f} ns '
                         f'vs slim-derived {s:.2f} ns ({n} hits, spread {sp:.2f})'
                         for (d, c, b), p, s, n, sp in bad[:10])
        raise ValueError(
            f'tflash disagrees between the processed file and the slim on '
            f'{len(bad)} (det, bunch) pairs, tolerance {tol} ns:\n{head}\n'
            f'  The windows would be centred on the wrong place. Do not pull '
            f'until this is understood.')

    table = {k: (v, 0) for k, v in from_proc.items()}
    n_slim = 0
    for k, (v, _n, _sp) in from_slim.items():
        if k not in table:
            table[k] = (v, 1)
            n_slim += 1
    return table, {'n_processed': len(from_proc), 'n_slim_only': n_slim,
                   'n_cross_checked': sum(1 for k in from_slim if k in from_proc),
                   'n_total': len(table)}


def fill_gaps(table: dict, bunches, dets=C.SCINT_DETS, step_tol_ns=200.0):
    """Give every (det, detn, bunch) a tflash, so no channel is silently skipped.

    A channel with no hit in a bunch has no measured flash time; the value from
    the NEAREST bunch ON THE SAME CHANNEL stands in. Nearest-bunch, not the run
    median: tflash is ~11.6 us and stable to ~20 ns across a run, but LIQA/LIQB
    carry ~120 ns outliers and PSS the known ~350-400 ns per-bunch mis-tags, and
    a median inherits all of that where the neighbour inherits only the ~4 ns
    bunch-to-bunch step.

    A channel that never appears anywhere in the run -- a dead one -- has no
    neighbour either, and falls back to the median over its detector's other
    channels (source 3). Those are exactly the channels a block-driven pull most
    wants to look at, so giving them no window would defeat the purpose.

    Filled entries are marked (source 2 / 3) and counted, never hidden.
    """
    per_chan: dict[tuple, list[tuple[int, float]]] = {}
    for (d, c, b), (v, _s) in table.items():
        per_chan.setdefault((d, c), []).append((b, v))

    per_det: dict[int, list[float]] = {}
    for (d, _c), vs in per_chan.items():
        per_det.setdefault(d, []).extend(v for _b, v in vs)

    out = dict(table)
    filled = filled_dead = 0
    no_source = []
    step = {}
    want = np.unique(np.asarray(bunches, np.int64))

    for det in dets:
        d = C.DET_CODE[det]
        chans = sorted(c for (dd, c) in per_chan if dd == d)
        if not chans:
            no_source.append(det)
            continue
        det_median = float(np.median(per_det[d]))
        for c in chans:
            a = np.array(sorted(per_chan[(d, c)]), dtype=np.float64)
            kb, kv = a[:, 0].astype(np.int64), a[:, 1]
            if len(kv) > 1:
                step[(d, c)] = float(np.median(np.abs(np.diff(kv))))
            missing = want[~np.isin(want, kb)]
            if missing.size == 0:
                continue
            j = np.clip(np.searchsorted(kb, missing), 0, len(kb) - 1)
            jm = np.clip(j - 1, 0, len(kb) - 1)
            pick = np.where(np.abs(kb[jm] - missing) <= np.abs(kb[j] - missing),
                            jm, j)
            for b, q in zip(missing.tolist(), pick.tolist()):
                out[(d, c, int(b))] = (float(kv[q]), 2)
                filled += 1
        # channels of this detector that never appeared at all
        for c in range(1, C.DET_NCHAN.get(det, 0) + 1):
            if c in chans:
                continue
            for b in want.tolist():
                out[(d, c, int(b))] = (det_median, 3)
                filled_dead += 1

    unstable = sorted({int(d) for (d, _c), v in step.items() if v > step_tol_ns})
    return out, {'n_filled_from_neighbour_bunch': filled,
                 'n_filled_dead_channel': filled_dead,
                 'dets_without_any_tflash': no_source,
                 'chan_tflash_bunch_step_ns_p95': (
                     round(float(np.percentile(list(step.values()), 95)), 2)
                     if step else 0.0),
                 'dets_with_unstable_tflash': unstable}


def build(events: dict, tflash: dict, window_ns=C.WINDOW_NS,
          control_shift_ns=C.CONTROL_SHIFT_NS, with_control=True,
          dets=C.SCINT_DETS, extra_dets=(C.PKUP_DET,), report=None):
    """{bunch: {det_name: {chan: (K, 2) merged [lo, hi) in tof ns}}}.

    Two levels, not one, so the scan can skip a whole detector's bank with a
    seek before it ever looks at a channel.

    Flash triggers are excluded -- they are referenced to the flash itself and
    the slim gives them no n_TOF hits.

    `extra_dets` (PKUP) have no tflash of their own; they inherit the windows of
    the whole bunch, i.e. the union over the scintillator detectors, so the
    pickup trace covering each trigger comes along.
    """
    bun = np.asarray(events['bunch'], np.int64)
    pred = np.asarray(events['t_pred_ns'], np.float64)
    live = (np.asarray(events['is_flash']) == 0) & np.isfinite(pred)
    bun, pred = bun[live], pred[live]

    shifts = (0.0, control_shift_ns) if with_control else (0.0,)
    W = float(window_ns)

    per_bunch: dict[int, dict[str, dict]] = {}
    skipped: dict[str, int] = {}
    order = np.argsort(bun, kind='stable')
    bun, pred = bun[order], pred[order]
    edges = np.flatnonzero(np.diff(bun)) + 1
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(bun)]):
        b = int(bun[lo])
        p = pred[lo:hi]
        entry: dict[str, dict] = {}
        allc_lo, allc_hi = [], []
        for det in dets:
            for chan in range(1, C.DET_NCHAN.get(det, 0) + 1):
                tf = tflash.get((C.DET_CODE[det], chan, b))
                if tf is None:
                    # No flash time for this channel in this bunch => no window
                    # and therefore no waveforms. Counted, never silent:
                    # `fill_gaps` exists so this stays at zero in production.
                    skipped[det] = skipped.get(det, 0) + 1
                    continue
                if isinstance(tf, tuple):    # (value, source), as reconcile returns
                    tf = tf[0]
                c = np.concatenate([p + tf + s for s in shifts])
                iv = merge_intervals(np.floor(c - W).astype(np.int64),
                                     np.ceil(c + W).astype(np.int64) + 1)
                entry.setdefault(det, {})[chan] = iv
                allc_lo.append(iv[:, 0])
                allc_hi.append(iv[:, 1])
        if not entry:
            continue
        if extra_dets and allc_lo:
            union = merge_intervals(np.concatenate(allc_lo),
                                    np.concatenate(allc_hi))
            for det in extra_dets:
                entry[det] = {c: union
                              for c in range(1, C.DET_NCHAN.get(det, 1) + 1)}
        per_bunch[b] = entry
    if report is not None:
        report['skipped_det_bunch'] = skipped
        report['n_skipped_det_bunch'] = sum(skipped.values())
    return per_bunch


def requested_samples(windows: dict) -> int:
    """Upper bound on kept samples if every window were solid signal -- the
    number to compare the product against when asking 'did we get it all'."""
    return int(sum(int((iv[:, 1] - iv[:, 0]).sum())
                   for per_det in windows.values()
                   for per_chan in per_det.values()
                   for iv in per_chan.values()))
