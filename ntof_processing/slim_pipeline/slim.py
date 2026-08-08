#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slim.py -- turn one (DREAM sub-run x n_TOF run) segment into an `ntof_hits` file.

Two passes over the n_TOF source, and the clock is fully fitted between them, so
NOTHING is ever cut on a provisional calibration:

  [0] join    DREAM eventId -> BunchNumber, t_since_flash, is_flash
              (bunch_join; beam record only, independent of the PSA settings)
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
    src = Path(seg.ntof_source) if seg.ntof_source else C.NTOF_DONE
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


def join_events(seg: Segment, log=print):
    """DREAM events of the sub-run that belong to this n_TOF run."""
    from ntof_dream_merge.bunch_join import dream_event_to_bunch
    ev = dream_event_to_bunch(seg.dream_run, seg.dream_subrun, seg.ntof_run)
    ev = ev[ev['BunchNumber'] > 0].reset_index(drop=True)
    if seg.bunches is not None:
        ev = ev[ev['BunchNumber'].isin(seg.bunches)].reset_index(drop=True)
    flash = ev['is_flash'].to_numpy()
    log(f'  joined {len(ev):,} DREAM events, {flash.sum():,} flash triggers, '
        f'{ev["BunchNumber"].nunique()} bunches')
    return ev


def pass1_candidates(seg: Segment, bunches, log=print):
    """Wall top/bottom offsets and the wall AND plastic SINGLES candidates."""
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
    n_off = min(OFFSET_BUNCHES, len(bunches))
    offs = {a: fs.measure_tb_offsets(seg.ntof_run, bunches[:n_off], a)
            for a in dt.ARMS}
    log(f'  top/bottom offsets on {n_off} bunches [{time.time()-t0:.0f} s]: ' +
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
                            'satuflag', 'pileup1', 'pulseshape', 'is_control')}
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
            'is_control': hits['is_control'].astype(np.uint8)}
        f['events'] = events
        f['bunches'] = bunches_tbl
    (out / 'calibration.json').write_text(json.dumps(meta['calibration'], indent=2))
    (out / 'qa.json').write_text(json.dumps(meta['qa'], indent=2))
    (out / 'provenance.json').write_text(json.dumps(meta['provenance'], indent=2))
    mb = p.stat().st_size / 1e6
    log(f'  -> {p.name}  {mb:.1f} MB  '
        f'({p.stat().st_size/max(events["eventId"].size,1):.0f} B/trigger)')
    return p


def run_segment(seg: Segment, out_base: Path | None = None,
                slim_ns: float = C.SLIM_NS, log=print):
    """The whole chain for one segment. Returns (path, qa)."""
    t_start = time.time()
    log(f'== {seg}')
    io, files = _bind_ntof(seg)
    log(f'  {len(files)} n_TOF file(s), cache {io.CACHE_DIR}')

    ev = join_events(seg, log=log)
    phys = ~ev['is_flash'].to_numpy()
    ev_id = ev['eventId'].to_numpy().astype(np.int64)
    ev_b = ev['BunchNumber'].to_numpy().astype(np.int64)
    ev_t = ev['t_since_flash_ns'].to_numpy().astype(np.float64)
    bunches = np.unique(ev_b[phys])

    log('  [1] candidates')
    cd, offs, thr = pass1_candidates(seg, bunches, log=log)

    log('  [2] clock')
    K, T0, arm_off, ginfo = cf.fit_global(ev_b[phys], ev_t[phys],
                                          cd['bunch'], cd['t'], cd['arm'],
                                          log=log)
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

    ub, cnt = np.unique(ev_b, return_counts=True)
    bunches_tbl = dict(bunch=ub.astype(np.int32), n_triggers=cnt.astype(np.int32),
                       fitted=np.isin(ub, list(pb)).astype(np.uint8))

    meta = dict(
        calibration=dict(
            K=K, T0_ns=T0,
            arm_offset_ns={a: float(arm_off[i]) for i, a in enumerate(cf.ARMS)},
            accept_ns=C.ACCEPT_NS, slim_ns=slim_ns,
            control_shift_ns=C.CONTROL_SHIFT_NS,
            tb_offsets_ns={a: [float(x) for x in offs[a]] for a in offs},
            thresholds_mv=dict(wall=thr['wall'], plastic=thr['plastic'],
                               polled_at=str(thr.get('polled_at'))),
            n_bunches_fitted=len(pb), fit=ginfo),
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
