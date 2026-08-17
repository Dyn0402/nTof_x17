#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slim.py -- turn one (DREAM sub-run x n_TOF run) segment into an `ntof_hits` file.

Two passes over the n_TOF source, and the clock is fully fitted between them, so
NOTHING is ever cut on a provisional calibration:

  [0] join    DREAM eventId -> BunchNumber, t_since_flash, is_flash
              (bunch_join; beam record only, independent of the PSA settings),
              then DROP the bunches whose PS pulse delivered no protons -- see
              `bunch_table` and C.EMPTY_PULSE_E10. They stay in the `bunches`
              tree with has_beam = 0 so the file still says what was dropped.
  [1] pass 1  wall top/bottom offsets, then the N1081B SINGLES emulation over
              the whole segment -> the candidate list. Reads WAL + PSS.
  [2] fit     K, T0, per-arm offsets, then per-bunch (da_b, dk_b). Seconds.
  [3] pass 2  read all twelve scintillator trees and keep hits within
              +-SLIM_NS of the FULLY CORRECTED prediction, plus the same width
              at +CONTROL_SHIFT_NS as the accidental control. Reads everything.
  [4] write   one ROOT file with three trees, plus JSON sidecars.

Flash triggers are tagged in the `events` tree and get no n_TOF hits -- the
source stays on EOS if they are ever wanted.

Why two passes and no intermediate buffer: pass 1 has to read WAL + PSS (78 % of
the hits) to build candidates at all, so caching the hits it streams would save
78 % of one read at the cost of an intermediate format. Simplicity won; if the
EOS I/O turns out to hurt, the buffer slots into pass 1 without changing
anything downstream.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from ntof_processing.slim_pipeline import clockfit as cf    # noqa: E402
from ntof_processing.slim_pipeline import config as C       # noqa: E402

OFFSET_BUNCHES = 150      # sample for the modal top/bottom offsets
CAND_CHUNK = 250          # bunches per candidate chunk (memory, not speed)
HIT_CHUNK = 150           # bunches per pass-2 read


@dataclass
class Segment:
    dream_run: str
    dream_subrun: str
    ntof_run: int
    ntof_source: Path | None = None      # dir of partials; None -> C.NTOF_DONE
    bunches: np.ndarray | None = None    # restrict; None -> all the sub-run's
    processing: str = C.NTOF_PROCESSING
    files: list = field(default_factory=list)
    # boundary-sliver recovery (2026-08-12, ntof_processing/join_mislock/):
    # the majority side's fitted burst->pulse delta, transferred within the
    # same DREAM sub-run so the truncated minority side does not refit it
    delta_hint_s: float | None = None
    # A burst-to-pulse lock established by EVIDENCE rather than by the count
    # scan -- a bunch-shift scan, or coincidence_arbiter. Forces pulse_match to
    # take the nearest candidate lock to this offset. Until 2026-08-13 there was
    # no path at all from "we know the right lock" to "produce the product",
    # which is why every recovery had to be done by hand.
    accept_offset_s: float | None = None
    # which evidence established `accept_offset_s` -- 'verified' (a hand-run
    # bunch-shift scan) or 'coincidence' (coincidence_arbiter). Recorded as
    # lock_chosen_by, and the override result is CACHED, so this label outlives
    # the run that set it.
    accept_source: str = 'verified'
    # Half-width of the candidate-lock enumeration, seconds; None = pulse_match's
    # default (+-120 s). Widened for the segments whose lock the 2026-08-14
    # bunch-shift scans placed OUTSIDE that window (+172.8 s = four
    # supercycles). Recorded in the join block as `lock_search_s`.
    search_s: float | None = None
    # Small-segment lever (2026-08-16): the clock bootstrap's peak-count floor,
    # None = clockfit.BOOT_MIN_PEAK (150). Lowered ONLY for a segment whose
    # lock was established burst by burst (burst_bruteforce.py) and which is
    # too small to reach 150 counts -- a sub-run tail of 6-17 real bursts.
    # clockfit clamps it at BOOT_MIN_PEAK_FLOOR; the value used is recorded in
    # calibration.json fit.bootstrap.min_peak.
    boot_min_peak: int | None = None
    # Per-burst overrides established by burst_bruteforce.py, keyed by
    # burst_id: {'ntof_run': int (a bunch number is meaningless without it --
    # every n_TOF run numbers from 1), 'bunch': int (the bunch whose hits
    # coincide), 'flash_shift_ns':
    # float (the DREAM flash mis-tag: add to t_since_flash_ns), plus free-form
    # evidence keys}. Applied in `join_events` before anything downstream sees
    # the events; recorded in calibration.json join.burst_fix and in
    # burst_map.json. See ../slim_pipeline/burst_fixes.json.
    burst_fix: dict | None = None

    def __str__(self):
        return f'{self.dream_run}/{self.dream_subrun} x n_TOF {self.ntof_run}'


def _bind_ntof(seg: Segment):
    """Point ntof_io at this segment's files and give them their own cache.

    ntof_io's bunch index and tflash caches are keyed by RUN NUMBER only, so an
    official and a reprocessed run224572 sharing a cache directory silently mix
    (ntof_processing/REVIEW.md section 5). `variant_cache` fingerprints on the
    file set instead.
    """
    import ntof_dream_merge.ntof_io as io
    import ntof_dream_merge.tflash_repair as rep

    files = C.ntof_files(seg.ntof_run, seg.ntof_source)
    # the directory the files ACTUALLY came from -- `ntof_files` falls back to
    # the unmerged partials under completed/<run>/, so C.NTOF_DONE would name
    # the wrong variant here. The fingerprint is on the file set either way.
    src = Path(seg.ntof_source) if seg.ntof_source else files[0].parent
    io.ntof_paths = lambda r: files
    io.ntof_path = lambda r: files[0]
    if C.CACHE_BASE:
        # On a worker, $X17_BEAM_JULY is EOS and ntof_io would try to build its
        # index there. Redirect to node-local scratch before variant_cache runs.
        io.CACHE_DIR = Path(C.CACHE_BASE)
        io.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rep.CACHE_DIR = io.CACHE_DIR = io.variant_cache(src, files)
    io._TFLASH_FIX_CACHE.clear()
    seg.files = files
    return io, files


def join_events(seg: Segment, log=print, events=None):
    """DREAM events of the sub-run that belong to this n_TOF run."""
    from ntof_dream_merge.bunch_join import dream_event_to_bunch
    ev = dream_event_to_bunch(seg.dream_run, seg.dream_subrun, seg.ntof_run,
                              delta_hint_s=seg.delta_hint_s,
                              accept_offset_s=getattr(seg, 'accept_offset_s',
                                                      None),
                              accept_source=getattr(seg, 'accept_source',
                                                    'verified'),
                              events=events)
    attrs = dict(ev.attrs)         # filtering below may drop DataFrame.attrs
    if seg.burst_fix:
        ev, attrs = apply_burst_fix(seg, ev, attrs, log=log)
    ev = ev[ev['BunchNumber'] > 0].reset_index(drop=True)
    if seg.bunches is not None:
        ev = ev[ev['BunchNumber'].isin(seg.bunches)].reset_index(drop=True)
    ev.attrs.update(attrs)
    flash = ev['is_flash'].to_numpy()
    log(f'  joined {len(ev):,} DREAM events, {flash.sum():,} flash triggers, '
        f'{ev["BunchNumber"].nunique()} bunches')
    return ev


# A trigger this close to (or before) the re-referenced flash IS the flash, or
# the flash's own tail; the N93B gate admits singles from ~1 ms after it.
FLASH_RETAG_NS = 1000.0


def apply_burst_fix(seg: Segment, ev, attrs, log=print):
    """Apply burst_bruteforce.py's per-burst overrides to the joined frame.

    Two levers, both per burst_id, both established by scanning the burst
    against every bunch in reach and finding the one whose wall+plastic
    candidates line up with its triggers at the production accept window:

      bunch           the join put this burst on the wrong pulse (or on none);
                      move its events to the bunch that actually coincides
      flash_shift_ns  the burst's DREAM time base is referenced to the wrong
                      trigger (a mis-tagged flash: the sub-run's first burst
                      recorded mid-gate, or the flash trigger dropped and the
                      first single ~1 ms later tagged in its place). Measured
                      in n_TOF ns by the scan; added to t_since_flash_ns as
                      DREAM ns (divided by 1 + K_SEED) so the events sit on
                      the sub-run's clock again; triggers that then fall
                      within FLASH_RETAG_NS of the true flash are re-tagged
                      is_flash and get no hits, like every other flash.

    Nothing is fitted here; these are numbers the scan measured, and they are
    recorded verbatim in the product so the burst stays distinguishable from
    one the join placed on its own.
    """
    import ntof_dream_merge.ntof_io as io
    pk = io.pkup_bunches(seg.ntof_run)
    inten = dict(zip(pk['BunchNumber'].tolist(), pk['intensity_e10'].tolist()))
    bm = attrs.get('burst_map') or {}
    bm_idx = {b: i for i, b in enumerate(bm.get('burst_id', []))}
    applied = {}
    for bid, fx in seg.burst_fix.items():
        bid = int(bid)
        # A BUNCH NUMBER MEANS NOTHING WITHOUT ITS n_TOF RUN: every run starts
        # at 1, so bunch 677 exists in both 224642 and 224643 as unrelated
        # pulses. A sub-run that straddles a run boundary is joined once per
        # n_TOF run, and without this guard the override would fire in BOTH --
        # moving the burst onto a real, wrong pulse of the other run, which
        # the PKUP membership check below cannot catch because that bunch does
        # exist there. Found 2026-08-17 by the flash sweep, on
        # run_118/stat090_0005 (224642 + 224643); no product was affected,
        # because only the correct segment happened to be re-made.
        want = fx.get('ntof_run')
        if want is not None and int(want) != int(seg.ntof_run):
            log(f'  burst_fix: burst {bid} belongs to n_TOF {int(want)}, '
                f'not {seg.ntof_run} -- not applied here')
            continue
        m = (ev['burst_id'] == bid).to_numpy()
        if not m.any():
            log(f'  !! burst_fix: burst {bid} not in this sub-run -- ignored')
            continue
        rec = dict(fx)
        was = int(ev.loc[m, 'BunchNumber'].iloc[0])
        if fx.get('bunch') is not None:
            b = int(fx['bunch'])
            if b not in inten:
                log(f'  !! burst_fix: bunch {b} is not in n_TOF {seg.ntof_run} '
                    f'-- burst {bid} left as joined')
                continue
            ev.loc[m, 'BunchNumber'] = b
            ev.loc[m, 'bunch_intensity_e10'] = inten[b]
            ev.loc[m, 'join_resid_s'] = np.nan
            if bid in bm_idx:
                bm['bunch'][bm_idx[bid]] = b
                bm['resid_ms'][bm_idx[bid]] = None
        sh = float(fx.get('flash_shift_ns') or 0.0)
        n_retag = 0
        if sh:
            # `flash_shift_ns` is measured in the n_TOF time base (it is the
            # burst's lag against its neighbours'), and t_since_flash_ns is
            # DREAM time, which the clock map stretches by (1 + K) -- so
            # convert, or a 4.38 ms shift lands 482 ns off and the per-bunch
            # fit (+-400 ns search) never sees the burst (run_102/stat090_0002
            # burst 0, first attempt 2026-08-16). K_SEED is within ~1e-6 of
            # any fitted K, i.e. a few ns at these shifts.
            t = ev.loc[m, 't_since_flash_ns'].to_numpy().astype(np.int64) \
                + np.int64(round(sh / (1.0 + cf.K_SEED)))
            ev.loc[m, 't_since_flash_ns'] = t
            # re-tag from scratch: the flash is whatever sits at t' ~ 0. A
            # positive shift (first burst recorded mid-gate) frees the trigger
            # that was standing in for the flash; a negative one (orphan ahead
            # of the flash) tags the true flash and leaves the orphan tagged.
            new_flash = t < FLASH_RETAG_NS
            n_retag = int((new_flash != ev.loc[m, 'is_flash'].to_numpy()).sum())
            ev.loc[m, 'is_flash'] = new_flash
        rec.update(was_bunch=was, n_events=int(m.sum()), n_retagged=n_retag)
        applied[str(bid)] = rec
        log(f'  burst_fix: burst {bid} ({int(m.sum())} events) bunch '
            f'{was} -> {int(ev.loc[m, "BunchNumber"].iloc[0])}, '
            f'flash shift {sh:+.0f} ns, {n_retag} flash tag(s) changed')
    if applied:
        bm['fix'] = applied
        attrs['burst_map'] = bm
    attrs['burst_fix'] = applied
    return ev, attrs


def bunch_table(ev, log=print):
    """(per-bunch table, has_beam per EVENT) from the joined events.

    One row per bunch the sub-run touched, EMPTY PULSES INCLUDED, because the
    accounting is the point: the rows say what was dropped and why, and the
    fraction of them with beam is this segment's beam availability. Bunches the
    beam record does not describe at all (NaN intensity, which the join only
    produces for a burst it could not place) count as beam and are kept -- an
    unknown is not a reason to throw data away.
    """
    b = ev['BunchNumber'].to_numpy().astype(np.int64)
    inten = ev['bunch_intensity_e10'].to_numpy().astype(np.float64)
    ub, first, cnt = np.unique(b, return_index=True, return_counts=True)
    bi = inten[first]                       # intensity is a property of the bunch
    beam = ~(bi < C.EMPTY_PULSE_E10)        # NaN -> True, deliberately
    tbl = dict(bunch=ub, n_triggers=cnt, intensity_e10=bi, has_beam=beam)
    if ub.size == 0:                    # nothing joined; the caller says why
        return tbl, np.zeros(b.size, bool)
    n_drop = int(cnt[~beam].sum())
    log(f'  beam record: {int(beam.sum()):,} of {ub.size:,} bunches carried '
        f'protons ({beam.mean():.2%} availability); dropping {ub.size - int(beam.sum()):,} '
        f'empty pulses and their {n_drop:,} triggers')
    if beam.any():
        par = float(np.mean(bi[beam] < C.PARASITIC_E10))
        log(f'  beam mix: {par:.1%} parasitic (< {C.PARASITIC_E10:g}e10), '
            f'median intensity {np.median(bi[beam]):.0f}e10')
    # An empty pulse should hold a HANDFUL of triggers, because DREAM's gate
    # opens on the PS timing but only background walks through it: measured
    # 1-2 against ~92 in a beam bunch. An empty pulse holding a FULL burst is
    # therefore not a beam statement at all -- it means bursts are landing on
    # the wrong bunches. Seen 2026-08-10 on run_116/stat090_0013 x 224636, a
    # 13 %-overlap proposal whose join fitted a -1,324 s offset and paired
    # unrelated bursts to unrelated pulses: 22 "empty" bunches holding 66-108
    # triggers each, one burst apiece. That segment fails its clock fit anyway,
    # but the ratio sees it several minutes earlier and says why.
    if (~beam).any() and beam.any():
        r = float(np.median(cnt[~beam]) / max(np.median(cnt[beam]), 1))
        if r >= C.EMPTY_TRIGGER_RATIO_WARN:
            log(f'  !! the dropped pulses hold {np.median(cnt[~beam]):.0f} '
                f'triggers each against {np.median(cnt[beam]):.0f} in a beam '
                f'bunch (ratio {r:.2f}). A no-beam pulse cannot produce a full '
                f'DREAM burst -- suspect the burst-to-bunch assignment, not '
                f'the beam')
    return tbl, np.isin(b, ub[beam])


def pass1_candidates(seg: Segment, bunches, log=print, offsets=None):
    """Wall top/bottom offsets and the wall AND plastic SINGLES candidates.

    `offsets` supplies the top/bottom table instead of measuring it here. The
    offsets are a property of the n_TOF RUN, not of `bunches`, so a caller that
    works on a handful of bunches at a time (coincidence_arbiter, via
    arbiter_measure) must measure them once on a proper sample and pass them in
    -- measuring them on its own 8-bunch sample lands them on noise, and a wrong
    window keeps only ~28 % of genuine top/bottom pairs.
    """
    from ntof_dream_merge import dream_trigger as dt
    from ntof_dream_merge import fast_singles as fs

    # The reprocessed file's OWN stored tflash. The laptop-side repair is built
    # for the broken official flash finding and would shift LIQC/D by 15 ns and
    # add 25 ns RMS on PSSC here.
    fs.REPAIR_TFLASH = False

    thr = dt.load_thresholds(seg.dream_run, seg.dream_subrun)
    adc = dt.load_adc_mv()
    log('  thresholds: ' + '  '.join(
        f'{a} wall {thr["wall"][a]:.0f}/pss {thr["plastic"][a]:.0f} mV'
        for a in dt.ARMS))

    t0 = time.time()
    # Instrumental constants, so measured once on a sample of ~1e5 pairs rather
    # than per chunk. They are per PROCESSING, not per cabling: on the official
    # file they are +-32-39 ns, on v12 within +-5.5 ns. A stored table would
    # pair the bar ends around an offset that is no longer there.
    if offsets is None:
        n_off = min(OFFSET_BUNCHES, len(bunches))
        offs = {a: fs.measure_tb_offsets(seg.ntof_run, bunches[:n_off], a)
                for a in dt.ARMS}
        src = f'on {n_off} bunches [{time.time()-t0:.0f} s]'
    else:
        offs = offsets
        src = 'supplied by the caller'
    log(f'  top/bottom offsets {src}: ' +
        '; '.join(f'{a} ' + ','.join(f'{offs[a][g]:+.1f}' for g in range(4))
                  for a in dt.ARMS))

    acc, t0 = [], time.time()
    chunks = [bunches[i:i + CAND_CHUNK]
              for i in range(0, len(bunches), CAND_CHUNK)]
    for i, ch in enumerate(chunks):
        acc.append(fs.all_arms(seg.ntof_run, ch, thr, adc, offsets=offs,
                               require_plastic=True))
        log(f'    chunk {i+1}/{len(chunks)} (bunches {ch[0]}-{ch[-1]}): '
            f'{sum(a["t"].size for a in acc):,} candidates '
            f'[{time.time()-t0:.0f} s]')
    keys = ('bunch', 't', 'wall_mv', 'seg', 'pss_dt', 'pss_mv', 'arm')
    cd = {k: np.concatenate([a[k] for a in acc]) for k in keys}
    o = np.lexsort((cd['t'], cd['bunch']))
    cd = {k: v[o] for k, v in cd.items()}
    log(f'  {cd["t"].size:,} candidates '
        f'({cd["t"].size/max(len(bunches),1):.0f}/bunch) [{time.time()-t0:.0f} s]')
    return cd, offs, thr


def shadow_prev(grp, t, amp, t_hold=C.SHADOW_HOLD_NS):
    """(largest earlier amp on the same channel within t_hold, ns since it).

    The plastics ring: every large pulse is trailed by real secondary pulses
    out to ~1 us (`../pss_ringing/`), and the adopted cut for them is
    `amp_0 < ratio * shadow_amp`. Computed here on the FULL per-bunch stream,
    BEFORE the slim window cut, because an after-pulse whose parent falls just
    outside the window is exactly what a slim-only recomputation gets wrong.
    Returns (0, -1) where nothing precedes the hit. Same walk-back as
    `pss_ringing/afterpulse_flag.py`: work is proportional to in-window PAIRS
    (~0.7/hit at the plastics' 720 kHz), not depth x length.
    """
    grp = np.asarray(grp)
    t = np.asarray(t, np.float64)
    amp = np.asarray(amp, np.float64)
    order = np.lexsort((t, grp))
    g, tt, aa = grp[order], t[order], amp[order]
    smax = np.zeros(t.size)
    sdt = np.full(t.size, -1.0)
    active = np.arange(1, t.size)
    for k in range(1, t.size):
        j = active - k
        keep = j >= 0
        active, j = active[keep], j[keep]
        if active.size == 0:
            break
        inwin = (g[j] == g[active]) & (tt[active] - tt[j] <= t_hold) \
            & (tt[active] - tt[j] > 0)
        active, j = active[inwin], j[inwin]
        if active.size == 0:
            break
        upd = aa[j] > smax[active]
        ai, aj = active[upd], j[upd]
        smax[ai] = aa[aj]
        sdt[ai] = tt[ai] - tt[aj]
    out_a, out_d = np.empty_like(smax), np.empty_like(sdt)
    out_a[order], out_d[order] = smax, sdt
    return out_a, out_d


def pass2_hits(seg: Segment, ev_bunch, ev_t, ev_id, corr, K, T0,
               slim_ns=C.SLIM_NS, control_ns=C.CONTROL_SHIFT_NS, log=print):
    """Every scintillator hit within +-slim_ns of a corrected prediction.

    The window is centred ARM-AGNOSTICALLY: the slim cannot know which arm fired
    before it has the hits, and the per-arm offsets span only -16.8..+7.5 ns,
    which the window absorbs. The control window is the same width at
    +control_ns and is what makes the file able to measure its own background.
    """
    import ntof_dream_merge.ntof_io as io

    pred = cf.predict(ev_t, K, T0) + np.nan_to_num(np.asarray(corr, float))
    keys, ids, tags = [], [], []
    for tag, sh in ((0, 0.0), (1, control_ns)):
        k = cf.pack(ev_bunch, pred + sh)
        o = np.argsort(k)
        keys.append(k[o]); ids.append(np.asarray(ev_id)[o])
        tags.append(np.full(k.size, tag, np.uint8))
    allk = np.concatenate(keys)
    order = np.argsort(allk)
    allk, allid = allk[order], np.concatenate(ids)[order]
    alltag = np.concatenate(tags)[order]

    bunches = np.unique(np.asarray(ev_bunch))
    cols = {k: [] for k in ('eventId', 'det', 'detn', 'tof', 'dt_ns', 'amp',
                            'amp_0', 'area_0', 'fwhm', 'risetime', 'chi2',
                            'satuflag', 'pileup1', 'pulseshape', 'is_control',
                            'shadow_amp', 'shadow_dt')}
    n_src = 0
    t0 = time.time()
    for ti, tree in enumerate(C.SCINT_TREES):
        for i in range(0, bunches.size, HIT_CHUNK):
            blk = bunches[i:i + HIT_CHUNK]
            a = io.read_bunches(seg.ntof_run, tree, blk,
                                branches=C.HIT_BRANCHES, repair_tflash=False)
            if a['tof'].size == 0:
                continue
            n_src += a['tof'].size
            # Ringing shadow, on the FULL chunk stream before any window cut.
            # Chunks hold whole bunches and tof restarts per bunch, so the
            # (bunch, channel) grouping never needs lookback across a chunk.
            sh_amp, sh_dt = shadow_prev(
                a['BunchNumber'].astype(np.int64) * 100
                + a['detn'].astype(np.int64), a['tof'], a['amp_0'])
            hk = cf.pack(a['BunchNumber'], a['t_since_flash_ns'])
            j = np.searchsorted(allk, hk)
            j0 = np.clip(j - 1, 0, allk.size - 1)
            j1 = np.clip(j, 0, allk.size - 1)
            d0, d1 = allk[j0] - hk, allk[j1] - hk
            take = np.abs(d0) <= np.abs(d1)
            pick = np.where(take, j0, j1)
            dt = np.where(take, -d0, -d1)          # hit minus prediction
            keep = np.abs(dt) <= slim_ns
            if not keep.any():
                continue
            k = np.nonzero(keep)[0]
            cols['eventId'].append(allid[pick[k]])
            cols['is_control'].append(alltag[pick[k]])
            cols['det'].append(np.full(k.size, ti, np.uint8))
            cols['dt_ns'].append(dt[k])
            cols['shadow_amp'].append(sh_amp[k])
            cols['shadow_dt'].append(sh_dt[k])
            for b in ('detn', 'tof', 'amp', 'amp_0', 'area_0', 'fwhm',
                      'risetime', 'chi2', 'satuflag', 'pileup1', 'pulseshape'):
                cols[b].append(a[b][k])
        log(f'    {tree}: {sum(x.size for x in cols["det"]):,} kept so far '
            f'[{time.time()-t0:.0f} s]')
    hits = {k: (np.concatenate(v) if v else np.array([])) for k, v in cols.items()}
    n = hits['eventId'].size
    log(f'  {n:,} hits kept of {n_src:,} read ({n/max(n_src,1):.4%}), '
        f'{n/max(np.size(ev_id),1):.2f} per trigger [{time.time()-t0:.0f} s]')
    o = np.lexsort((hits['det'], hits['is_control'], hits['eventId']))
    return {k: v[o] for k, v in hits.items()}, n_src


def write(seg: Segment, out: Path, hits, events, bunches_tbl, meta, log=print):
    out.mkdir(parents=True, exist_ok=True)
    p = out / f'ntof_hits_{seg.dream_run}_{seg.dream_subrun}_{seg.ntof_run}.root'
    with uproot.recreate(p, compression=uproot.ZLIB(4)) as f:
        f['hits'] = {
            'eventId': hits['eventId'].astype(np.uint64),
            'det': hits['det'].astype(np.uint8),
            'detn': hits['detn'].astype(np.int32),
            'tof': hits['tof'].astype(np.float64),
            'dt_ns': hits['dt_ns'].astype(np.float32),
            'amp': hits['amp'].astype(np.float32),
            'amp_0': hits['amp_0'].astype(np.float32),
            'area_0': hits['area_0'].astype(np.float32),
            'fwhm': hits['fwhm'].astype(np.float32),
            'risetime': hits['risetime'].astype(np.float32),
            'chi2': hits['chi2'].astype(np.float32),
            'satuflag': hits['satuflag'].astype(np.int32),
            'pileup1': hits['pileup1'].astype(np.int32),
            'pulseshape': hits['pulseshape'].astype(np.int32),
            'is_control': hits['is_control'].astype(np.uint8),
            'shadow_amp': hits['shadow_amp'].astype(np.float32),
            'shadow_dt': hits['shadow_dt'].astype(np.float32)}
        f['events'] = events
        f['bunches'] = bunches_tbl
    (out / 'calibration.json').write_text(json.dumps(meta['calibration'], indent=2))
    (out / 'qa.json').write_text(json.dumps(meta['qa'], indent=2))
    (out / 'provenance.json').write_text(json.dumps(meta['provenance'], indent=2))
    # THE BURST MAP, its own sidecar. Every burst of the sub-run that this
    # segment's join looked at, matched or not (bunch -1, resid null), so the
    # pulse ledger has a denominator. It is deliberately NOT in calibration.json:
    # that file is per-segment calibration and a ~1000-row table does not belong
    # in it. Bursts that found no bunch exist ONLY here -- `join_events` filters
    # them at `BunchNumber > 0` and they are otherwise lost.
    if meta.get('burst_map'):
        (out / 'burst_map.json').write_text(json.dumps(meta['burst_map']))
    mb = p.stat().st_size / 1e6
    log(f'  -> {p.name}  {mb:.1f} MB  '
        f'({p.stat().st_size/max(events["eventId"].size,1):.0f} B/trigger)')
    return p


def load_burst_fixes(path=None) -> dict:
    """{'run/subrun': {'lock': {...}, 'bursts': {burst_id: {...}}}} or {}."""
    p = Path(path) if path else C.BURST_FIXES
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {k: v for k, v in d.items() if not k.startswith('_')}


def apply_fixes(seg: Segment, fixes: dict, log=print):
    """Put a burst_fixes.json entry onto the Segment. Returns the entry used.

    A `lock` entry applies only when its ntof_run is this segment's: it sets
    accept_offset_s (source 'bruteforce') and boot_min_peak. `bursts` entries
    apply whenever their bunch is in this n_TOF run (checked in
    apply_burst_fix), so a fixed burst on the other side of a run boundary is
    simply ignored by the segment it does not belong to.
    """
    e = fixes.get(f'{seg.dream_run}/{seg.dream_subrun}')
    if not e:
        return None
    used = {}
    lock = e.get('lock')
    if lock and int(lock.get('ntof_run', -1)) == int(seg.ntof_run):
        seg.accept_offset_s = float(lock['offset_s'])
        seg.accept_source = 'bruteforce'
        if lock.get('boot_min_peak') is not None:
            seg.boot_min_peak = int(lock['boot_min_peak'])
        used['lock'] = lock
        log(f'  burst_fixes: lock {seg.accept_offset_s:+.3f} s '
            f'(boot_min_peak {seg.boot_min_peak}) -- {lock.get("source", "")}')
    if e.get('bursts'):
        seg.burst_fix = {int(k): v for k, v in e['bursts'].items()}
        used['bursts'] = e['bursts']
        log(f'  burst_fixes: {len(seg.burst_fix)} per-burst override(s)')
    return used or None


# Kept for one release as an escape hatch: set False and the slim falls back to
# letting pulse_match decide (weak count scan, silent tie-break). There is no
# reason to use it except to reproduce a pre-2026-08-13 product bit-for-bit.
ARBITRATE_AMBIGUOUS = True


def _join_by_coincidence(seg: Segment, log=print, events=None):
    """Join at the lock the COINCIDENCE chooses, every segment, every time.

    `pulse_match` enumerates candidates; it no longer decides. The count scan
    it runs is a 50 ms-tolerance cluster match that ties routinely under the PS
    supercycle -- that silent tie-break cost 25.7 % of the July campaign beam.
    The wall+plastic coincidence separates the same question by three orders of
    magnitude: measured 2026-08-13 over ~190 candidate evaluations, a correct
    lock puts 87-98 % of a pulse's triggers on a same-arm coincidence and the
    highest any wrong lock reached was 0.00 %.

    So the ordering is inverted from what the pipeline had: the strong
    instrument chooses, and the weak one is only used to propose. If no
    candidate clears the bar the segment REFUSES -- "none of these locks is
    right" is a real answer, and the refusal carries the evidence in
    `.arbiter`.
    """
    sys.path.insert(0, str(HERE.parents[1] / 'ntof_july_analysis'))
    import pulse_match as pm
    from ntof_processing.slim_pipeline import arbiter_measure as AM
    from ntof_processing.slim_pipeline import coincidence_arbiter as CA

    if seg.accept_offset_s is not None:
        # a lock already established by evidence (a hand scan, or a caller that
        # arbitrated) -- apply it, do not re-decide
        return join_events(seg, log=log, events=events)

    enum = pm.enumerate_locks(seg.dream_run, seg.dream_subrun,
                              search_s=seg.search_s)
    if not enum or not enum.get('locks'):
        raise pm.NoLock(
            f'{seg}: pulse_match enumerated no candidate lock at all -- no '
            f'events, no beam-record coverage, or nothing matching the CSV. '
            f'Nothing for the coincidence to choose between.')
    log(f'  [0] {len(enum["locks"])} candidate lock(s) within '
        f'+-{enum["search_s"]:g} s over {enum["n_clusters"]:,} clusters; '
        f'the coincidence chooses')
    v = CA.arbitrate(enum['locks'],
                     AM.make_measurer(seg.dream_run, seg.dream_subrun,
                                      seg.ntof_run, ntof_source=seg.ntof_source,
                                      log=log),
                     log=log)
    if not v.ok:
        exc = pm.AmbiguousLock(
            f'{seg}: the coincidence could not settle the lock '
            f'({v.reason}; {v.tested} measurement(s) over '
            f'{len(enum["locks"])} candidates)')
        exc.arbiter = dict(
            resolved=False, reason=v.reason, tested=v.tested,
            n_candidates=len(enum['locks']),
            rejected=[[float(o), float(f)] for o, f in v.rejected],
            unmeasured=[float(o) for o in v.unmeasured])
        raise exc
    log(f'  coincidence chose {v.offset_s:+.4f} s -- {v.reason}')
    seg.accept_offset_s = v.offset_s
    seg.accept_source = 'coincidence'
    return join_events(seg, log=log, events=events)


class LowJoin(RuntimeError):
    """Too few DREAM events joined to this n_TOF run to be worth fitting.

    Not an error in the data -- the segment list is proposed from wall-clock
    overlap, which is an estimate. Callers should record it and move on.
    """


def run_segment(seg: Segment, out_base: Path | None = None,
                slim_ns: float = C.SLIM_NS, min_events: int = C.MIN_EVENTS,
                log=print):
    """The whole chain for one segment. Returns (path, qa)."""
    t_start = time.time()
    log(f'== {seg}')
    io, files = _bind_ntof(seg)
    log(f'  {len(files)} n_TOF file(s), cache {io.CACHE_DIR}')

    # THE CENSUS BEFORE THE JOIN, because a segment that REFUSES is exactly the
    # one whose denominator must not vanish. Every burst of the sub-run, from
    # the DREAM files alone, written before anything downstream can raise. The
    # same frame is then handed to the join, so this costs no extra read.
    from ntof_dream_merge.bunch_join import dream_events
    ev_all = dream_events(seg.dream_run, seg.dream_subrun)
    try:
        from ntof_processing.slim_pipeline import pulse_ledger
        cdir = C.out_dir(seg.dream_run, seg.dream_subrun, out_base)
        cdir.mkdir(parents=True, exist_ok=True)
        pulse_ledger.write_census_from_events(ev_all, seg.dream_run,
                                              seg.dream_subrun, cdir)
        log(f'  [0] burst census -> {cdir/"burst_census.json"}')
    except Exception as e:                                   # noqa: BLE001
        # Auxiliary to the product, so it must not kill a segment -- but say so
        # loudly, because a missing census is a hole in the accounting and the
        # standalone census path has to be run to fill it.
        log(f'  !! burst census NOT written ({type(e).__name__}: {e}); '
            f'run pulse_ledger census for {seg.dream_run}/{seg.dream_subrun}')

    ev = _join_by_coincidence(seg, log=log, events=ev_all)
    join_attrs = dict(ev.attrs)     # survives the filters below

    # Empty pulses out, BEFORE anything is fitted or read. They are PS pulses
    # that delivered no protons (C.EMPTY_PULSE_E10, and the measurement behind
    # it); the triggers DREAM took during their gates are detector background
    # with a time base referenced to a background trigger rather than a flash,
    # so they can only add junk to the fit, the file and every analysis
    # downstream. The bunches themselves stay in the `bunches` tree with
    # has_beam = 0, which is where beam availability is read from.
    btbl, keep = bunch_table(ev, log=log)
    if not keep.all():
        ev = ev[keep].reset_index(drop=True)

    phys = ~ev['is_flash'].to_numpy() if len(ev) else np.zeros(0, bool)
    if int(phys.sum()) < min_events:
        # Say which of the two it is. "No beam" and "no overlap" look the same
        # from here -- both end with an empty frame -- and calling a zero join
        # a no-beam segment is a lie about the accelerator. Seen on
        # run_116/stat090_0017 x 224636, which joined nothing at all.
        n_b = int(btbl['bunch'].size)
        why = ('the proposed overlap did not pan out' if n_b == 0
               or btbl['has_beam'].any() else
               f'all {n_b} joined bunches were empty pulses -- no beam '
               f'delivered in this segment at all')
        raise LowJoin(f'{int(phys.sum()):,} physics events joined with beam, '
                      f'below the {min_events:,} needed to fit a clock -- {why}')
    ev_id = ev['eventId'].to_numpy().astype(np.int64)
    ev_b = ev['BunchNumber'].to_numpy().astype(np.int64)
    ev_t = ev['t_since_flash_ns'].to_numpy().astype(np.float64)
    bunches = np.unique(ev_b[phys])

    log('  [1] candidates')
    cd, offs, thr = pass1_candidates(seg, bunches, log=log)

    log('  [2] clock')
    K, T0, arm_off, ginfo = cf.fit_global(
        ev_b[phys], ev_t[phys], cd['bunch'], cd['t'], cd['arm'], log=log,
        min_peak=(seg.boot_min_peak if seg.boot_min_peak is not None
                  else cf.BOOT_MIN_PEAK))
    corr_in, corr_cv, pb = cf.fit_perbunch(ev_b[phys], ev_t[phys],
                                           cd['bunch'], cd['t'], cd['arm'],
                                           K, T0, arm_off, log=log)
    qa_in = cf.efficiency(ev_b[phys], ev_t[phys], cd['bunch'], cd['t'],
                          cd['arm'], K, T0, arm_off, corr_in, C.ACCEPT_NS)
    qa_cv = cf.efficiency(ev_b[phys], ev_t[phys], cd['bunch'], cd['t'],
                          cd['arm'], K, T0, arm_off, corr_cv, C.ACCEPT_NS)
    log(f'    accept +-{C.ACCEPT_NS:g} ns: efficiency {qa_in["efficiency"]:.4%} '
        f'(cross-validated {qa_cv["efficiency"]:.4%}), '
        f'accidental {qa_in["accidental"]:.4%}, '
        f'purity {qa_in["purity"]:.4%}')

    # Corrections and match results are per PHYSICS event; flash triggers keep
    # NaN / unmatched and are written with no hits.
    full = lambda x, fill: (                                    # noqa: E731
        np.full(ev_id.size, fill, dtype=np.asarray(x).dtype))
    corr_all = np.full(ev_id.size, np.nan); corr_all[phys] = corr_in
    corrcv_all = np.full(ev_id.size, np.nan); corrcv_all[phys] = corr_cv
    matched = np.zeros(ev_id.size, bool); matched[phys] = qa_in['matched']
    resid = np.full(ev_id.size, np.nan); resid[phys] = qa_in['residual_ns']
    arm = np.full(ev_id.size, -1, np.int8); arm[phys] = qa_in['arm']

    log('  [3] slim')
    hits, n_src = pass2_hits(seg, ev_b[phys], ev_t[phys], ev_id[phys],
                             corr_in, K, T0, slim_ns=slim_ns, log=log)

    da = np.full(ev_id.size, np.nan); dk = np.full(ev_id.size, np.nan)
    for b, (a_, k_, _) in pb.items():
        m = ev_b == b
        da[m], dk[m] = a_, k_
    events = dict(
        eventId=ev_id.astype(np.uint64), bunch=ev_b.astype(np.int32),
        t_dream_ns=ev_t, is_flash=(~phys).astype(np.uint8),
        t_pred_ns=cf.predict(ev_t, K, T0) + np.nan_to_num(corr_all),
        matched=matched.astype(np.uint8), residual_ns=resid.astype(np.float32),
        arm=arm, da_ns=da.astype(np.float32), dk=dk.astype(np.float32),
        corr_ns=corr_all.astype(np.float32),
        corr_cv_ns=corrcv_all.astype(np.float32))

    # Per-bunch fit parameters belong on the bunch, not smeared over its events:
    # a drifting clock shows up as structure in da_ns/dk ACROSS bunches, and
    # that is the first thing to look at when a segment misbehaves.
    #
    # The table spans every bunch the sub-run touched, empty pulses included, so
    # it doubles as the record of what was filtered: `has_beam` says whether the
    # PS delivered, `intensity_e10` says how much, and `n_triggers` is what the
    # bunch held before the cut. Analyses select on it -- has_beam for a clean
    # sample, intensity for the dedicated/parasitic split.
    ub = btbl['bunch']
    pb_a = np.array([pb.get(int(b), (np.nan,) * 3)[0] for b in ub])
    pb_k = np.array([pb.get(int(b), (np.nan,) * 3)[1] for b in ub])
    pb_n = np.array([pb.get(int(b), (0, 0, 0))[2] for b in ub])
    bunches_tbl = dict(bunch=ub.astype(np.int32),
                       n_triggers=btbl['n_triggers'].astype(np.int32),
                       has_beam=btbl['has_beam'].astype(np.uint8),
                       intensity_e10=btbl['intensity_e10'].astype(np.float32),
                       fitted=np.isin(ub, list(pb)).astype(np.uint8),
                       da_ns=pb_a.astype(np.float32),
                       dk=pb_k.astype(np.float32),
                       n_core=np.asarray(pb_n).astype(np.int32))
    beam = btbl['has_beam']
    bi = btbl['intensity_e10']

    meta = dict(
        # every burst the join looked at, matched or not -- see the sidecar
        # write in `write_slim`. Carried through join_attrs because the event
        # frame it came on has already had its unmatched rows filtered out.
        burst_map=join_attrs.get('burst_map'),
        calibration=dict(
            K=K, T0_ns=T0,
            arm_offset_ns={a: float(arm_off[i]) for i, a in enumerate(cf.ARMS)},
            accept_ns=C.ACCEPT_NS, slim_ns=slim_ns,
            control_shift_ns=C.CONTROL_SHIFT_NS,
            shadow_hold_ns=C.SHADOW_HOLD_NS, shadow_ratio=C.SHADOW_RATIO,
            # `measure_tb_offsets` returns {group: offset}, so iterating it
            # yields the GROUP INDICES: every slim written before 2026-08-10
            # recorded its top/bottom calibration as [0, 1, 2, 3]. The data was
            # never affected -- the real dict goes to `fast_singles.all_arms` --
            # but the file's record of its own calibration was wrong, which is
            # the one thing a provenance sidecar exists to get right.
            tb_offsets_ns={a: [float(offs[a][g]) for g in range(4)]
                           for a in offs},
            thresholds_mv=dict(wall=thr['wall'], plastic=thr['plastic'],
                               polled_at=str(thr.get('polled_at'))),
            empty_pulse_e10=C.EMPTY_PULSE_E10, parasitic_e10=C.PARASITIC_E10,
            n_bunches_fitted=len(pb), fit=ginfo,
            # Join provenance (2026-08-12, ntof_processing/join_mislock/):
            # a recovered segment must stay distinguishable from an
            # originally-clean one, and a confident join from a lucky one.
            # `pulse_match_chosen_by` is 'count' (clear win), 'intensity'
            # (near-tie arbitrated by the fluctuation correlation) or
            # 'verified' (scan-verified override); `delta_hint_s` non-null
            # means a sliver joined on its majority side's transferred delta.
            join=dict(
                lock_search_s=seg.search_s,
                # burst_bruteforce.py overrides actually applied (2026-08-16):
                # {burst_id: {bunch, flash_shift_ns, was_bunch, ...}}; absent
                # or empty on every segment the join placed on its own
                burst_fix=join_attrs.get('burst_fix') or None,
                pulse_match_offset_s=join_attrs.get('pulse_match_offset_s'),
                pulse_match_margin=join_attrs.get('pulse_match_margin'),
                pulse_match_chosen_by=join_attrs.get('pulse_match_chosen_by'),
                pulse_match_r_sig=join_attrs.get('pulse_match_r_sig'),
                delta_s=join_attrs.get('delta_s'),
                delta_margin=join_attrs.get('delta_margin'),
                delta_hint_s=join_attrs.get('delta_hint_s'),
                # Burst counts, so the overhang can be read in the units the
                # bootstrap bug actually depends on. `overlap_frac` in the
                # coverage map is a fraction of TIME and diverges from the
                # fraction of BURSTS wherever beam density is not uniform --
                # measured at 0.335 burst against 0.611 time on
                # run_81/stat090_0001 x 224581, a 3.2x density step across the
                # boundary. It does not merely blur the classification, it can
                # invert it: that sub-run's other side reads 0.311 overhang
                # nominally and 0.691 in bursts, opposite sides of the 50 %
                # threshold. Recording both counts means the question never has
                # to be reconstructed from a wall-clock proposal again.
                n_bursts=join_attrs.get('n_bursts'),
                n_bursts_matched=join_attrs.get('n_matched'),
                # The MEASURED extent of this sub-run's beam-synchronised
                # triggering. Not a general sub-run duration: it ends with the
                # last flash burst, so a sub-run whose beam stops before its
                # DAQ does will read short. It is the right denominator for
                # overhang -- a stretch with no bursts cannot move a median
                # over bursts -- and nothing else.
                subrun_span_s=join_attrs.get('subrun_span_s'))),
        qa=dict(
            efficiency=qa_in['efficiency'], efficiency_cv=qa_cv['efficiency'],
            accidental=qa_in['accidental'], purity=qa_in['purity'],
            n_events=int(ev_id.size), n_physics=int(phys.sum()),
            n_flash=int((~phys).sum()), n_candidates=int(cd['t'].size),
            n_hits=int(hits['eventId'].size),
            n_hits_signal=int((hits['is_control'] == 0).sum()),
            n_hits_control=int((hits['is_control'] == 1).sum()),
            n_hits_read=int(n_src),
            hits_per_trigger=float(hits['eventId'].size / max(phys.sum(), 1)),
            # The beam, as this segment saw it. `beam_availability` is a PS
            # statement, not a data-quality one, and the two are worth keeping
            # apart: a segment can be perfect and still sit at 0.86.
            n_bunches=int(beam.size), n_bunches_beam=int(beam.sum()),
            n_bunches_empty=int((~beam).sum()),
            beam_availability=float(beam.mean()),
            n_triggers_empty=int(btbl['n_triggers'][~beam].sum()),
            parasitic_fraction=(float(np.mean(bi[beam] < C.PARASITIC_E10))
                                if beam.any() else float('nan')),
            intensity_median_e10=(float(np.median(bi[beam]))
                                  if beam.any() else float('nan')),
            seconds=round(time.time() - t_start, 1)),
        provenance=dict(
            dream_run=seg.dream_run, dream_subrun=seg.dream_subrun,
            ntof_run=seg.ntof_run, ntof_processing=seg.processing,
            ntof_files=[str(f) for f in files],
            trees=list(C.SCINT_TREES), det_code={t: i for i, t in
                                                 enumerate(C.SCINT_TREES)},
            hit_branches=list(C.HIT_BRANCHES),
            created=time.strftime('%Y-%m-%dT%H:%M:%S')))

    out = C.out_dir(seg.dream_run, seg.dream_subrun, out_base)
    p = write(seg, out, hits, events, bunches_tbl, meta, log=log)
    log(f'== done in {meta["qa"]["seconds"]:.0f} s\n')
    return p, meta
