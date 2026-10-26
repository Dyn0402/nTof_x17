"""
Waveform input: decoded_root -> pedestal-subtracted, common-mode-corrected
windows around a seed cluster.

This is the only place raw waveforms are read. The pedestal / CNS / noise
recipe is the one validated in the R&D study (``01_build_cache.py``):

  * pedestal   = per-channel median over the first ``n_ped`` events
  * CNS        = per-sample median over each 64-channel block, subtracted
  * noise      = 1.4826 x MAD per channel over the pedestal events, post-CNS

Common-mode subtraction is not optional on this bench: FEU 6/8 have raw sigma
~115 ADC against ~10 after CNS.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence

import numpy as np

N_PED_EVENTS = 300
CNS_BLOCK = 64
CHUNK = 400
#: Signal-masked common mode (opt-in, 2026-10-23; rerun/CHECKS_2026-10-23.md C9).  The plain per-sample block median
#: moves with the signal when a track fills a large part of a 64-channel block (bench clusters are ~17 strips wide at
#: operating HV): on det7 the baseline error it puts on every strip of the block is 2.5-4 ADC at the pulse peak
#: (median), 15-51 ADC at p90 and up to ~100 ADC elsewhere in the window -- 10-25 % of a neighbour's tail.  With
#: CM_MASK = M > 0 the median is computed twice: once plain, then again without every channel that sits within M of
#: a contiguous cluster (>= 2 adjacent channels above 5 sigma in 2 consecutive samples).  A block left with fewer than
#: CM_MIN_KEEP channels keeps its plain median.  0 (default) = the plain median, bit-identical.
CM_MASK = int(os.environ.get('WFT_CM_MASK', 0))
CM_MIN_KEEP = 16
CM_SIGMA = 5.0
#: Coherent-event guard: the masked median is only valid if the channels it is taken from are quiet.  In a spark or
#: whole-chip saturation (det7 6-26: chip median ~3700 ADC) the plain median cancels the coherent shift, while the
#: median of the unmasked channels leaves +-900 ADC on every strip.  If the unmasked channels' spread around the
#: masked median (1.4826 x MAD over samples and channels) exceeds CM_GUARD x the chip's median noise, the plain median
#: is kept for that chip.  Found 2026-10-23 (rerun/m3_audit judge: masked CM lost 9 % of det7 track matches).
CM_GUARD = 3.0


def masked_block_median(wfm: np.ndarray, noise: np.ndarray, mask_half: int = None) -> np.ndarray:
    """(n_sample, 512) pedestal-subtracted window -> the same with the signal-masked block common mode removed."""
    mask_half = CM_MASK if mask_half is None else mask_half
    ns = wfm.shape[0]
    nblk = 512 // CNS_BLOCK
    blk = wfm.reshape(ns, nblk, CNS_BLOCK)
    cms = np.median(blk, axis=2)
    if mask_half <= 0:
        return wfm - np.repeat(cms, CNS_BLOCK, axis=1)
    wu = wfm - np.repeat(cms, CNS_BLOCK, axis=1)
    above = wu > CM_SIGMA * np.maximum(noise, 1.0)[None, :]
    persist = (above[1:] & above[:-1]).any(0)
    sig = np.zeros(512, bool)
    idx = np.flatnonzero(persist)
    if len(idx) >= 2:
        for g in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(g) >= 2:
                sig[max(0, g[0] - mask_half):g[-1] + mask_half + 1] = True
    if not sig.any():
        return wu
    keep = ~sig.reshape(nblk, CNS_BLOCK)
    for b in range(nblk):
        if keep[b].all() or keep[b].sum() < CM_MIN_KEEP:
            continue
        m = np.median(blk[:, b, keep[b]], axis=1)
        spread = 1.4826 * np.median(np.abs(blk[:, b, keep[b]] - m[:, None]))
        if spread > CM_GUARD * max(np.median(noise[b * CNS_BLOCK:(b + 1) * CNS_BLOCK]), 1.0):
            continue
        cms[:, b] = m
    return wfm - np.repeat(cms, CNS_BLOCK, axis=1)


@dataclass
class PlaneWindow:
    """One plane's waveform window for one event (the fit's input)."""
    ch: np.ndarray          # channel numbers
    pos: np.ndarray         # strip positions [mm], sorted ascending
    W: np.ndarray           # (n_strip, n_sample) pedestal+CNS subtracted ADC
    noise: np.ndarray       # per-strip sigma [ADC]


def subrun_files(base_path: str, run: str, sub_run: str, feu: int) -> list:
    return sorted(glob.glob(os.path.join(
        base_path, run, sub_run, 'decoded_root', f'*_{feu:02d}.root')))


def file_tag(path: str) -> str:
    """The '<date>_<time>_<idx>' part that pairs files across FEUs."""
    m = re.search(r'_datrun_(.+)_\d\d\.root$', os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


class FeuReader:
    """Streams one decoded FEU file, applying pedestal + CNS on the fly."""

    def __init__(self, path: str, n_ped: int = N_PED_EVENTS):
        import uproot
        self.path = path
        self.tree = uproot.open(path)['nt']
        self.n_entries = self.tree.num_entries
        a0 = self.tree.arrays(['amplitude'], entry_stop=min(n_ped, self.n_entries),
                              library='np')['amplitude']
        # Window length can vary event to event (det4's 6-24 run mixes 32 and 37
        # samples), so build the pedestal from the modal length only.
        lens = np.array([len(a) // 512 for a in a0])
        self.n_sample = int(np.bincount(lens).argmax())
        stack = np.stack([a.reshape(self.n_sample, 512)
                          for a, l in zip(a0, lens) if l == self.n_sample]
                         ).astype(np.float32)
        self.ped = np.median(stack, axis=(0, 1))
        sub = stack - self.ped[None, None, :]
        nblk = 512 // CNS_BLOCK
        cms = np.median(sub.reshape(len(stack), self.n_sample, nblk, CNS_BLOCK),
                        axis=3)
        sub -= np.repeat(cms, CNS_BLOCK, axis=2)
        self.noise = (1.4826 * np.median(np.abs(sub), axis=(0, 1))).astype(np.float32)
        self.event_ids = self.tree.arrays(['eventId'], library='np')['eventId']
        self.incomplete: Dict[int, int] = {}     # eventId -> samples present, for events iter_events skipped

    def iter_events(self, wanted: Optional[set] = None) -> Iterator[tuple]:
        """Yield (eventId, ftst, W[512, n_sample]) for the wanted events."""
        idx = (np.where(np.isin(self.event_ids, np.fromiter(wanted, dtype=np.int64)))[0]
               if wanted is not None else np.arange(self.n_entries))
        nblk = 512 // CNS_BLOCK
        for lo in range(0, len(idx), CHUNK):
            block = idx[lo:lo + CHUNK]
            arr = self.tree.arrays(['eventId', 'amplitude', 'ftst'],
                                   entry_start=int(block[0]),
                                   entry_stop=int(block[-1]) + 1, library='np')
            base = int(block[0])
            for i in block:
                j = i - base
                a = arr['amplitude'][j]
                # An event is usable only if every sample of the file's window is there.  The nt entry carries
                # no sample index in this reshape, so an event missing a 5-sample packet (FEU bandwidth loss,
                # 0.1-0.3 % of bench events) would come out with its later samples shifted early; an event longer
                # than the window is two triggers fused by the pre-fa80c06 decoder (det4 6-24's "37-sample"
                # events).  Both are skipped and listed in self.incomplete (rerun/decoder_audit, 2026-10-23).
                if len(a) != self.n_sample * 512:
                    self.incomplete[int(arr['eventId'][j])] = len(a) // 512
                    continue
                wfm = a.reshape(-1, 512).astype(np.float32) - self.ped
                ns = wfm.shape[0]
                if CM_MASK > 0:
                    wfm = masked_block_median(wfm, self.noise)
                else:
                    cms = np.median(wfm.reshape(ns, nblk, CNS_BLOCK), axis=2)
                    wfm -= np.repeat(cms, CNS_BLOCK, axis=1)
                yield int(arr['eventId'][j]), int(arr['ftst'][j]), wfm.T


def strip_position_map(cfg) -> Dict[int, np.ndarray]:
    """channel -> strip position [mm] per FEU, from the run's strip map."""
    from common.Mx17StripMap import RunConfig
    rc = RunConfig(cfg.run_config_path, cfg.MAP_CSV_PATH)
    det = rc.get_detector(cfg.DET_NAME)
    out = {}
    for feu, axis in ((cfg.MX17_FEU_X, 0), (cfg.MX17_FEU_Y, 1)):
        p = np.full(512, np.nan)
        for ch in range(512):
            hit = det.map_hit(feu, ch)
            if hit is not None and hit[axis] is not None:
                p[ch] = hit[axis]
        out[feu] = p
    return out


def extract_window(wfm_all: np.ndarray, noise_all: np.ndarray,
                   pos_map: np.ndarray, seed_channels: Sequence[int],
                   pad_strips: int = 3, dead=None) -> Optional[PlaneWindow]:
    """Cut the fit window: the seed cluster's strips plus `pad_strips` on each
    side in *position* order (the sharing kernel reaches +-2 strips, so the pad
    must be at least 2 for the model to see where the shared charge went).

    ``dead`` (channel numbers; callers pass ``cal.dead`` when ``wft.seed.DEAD_SEED``, the default): the pad counts LIVE
    strips only and takes the dead ones it crosses along, so a cluster that
    stops at a dead run (dead strips never fire) still has `pad_strips` strips
    of data beyond it. None = the historical pad.

    Note what does NOT happen here: no reference track, no hit times. Only
    'which strips' comes from the seed.
    """
    order = np.argsort(pos_map)
    order = order[np.isfinite(pos_map[order])]
    if len(order) == 0:
        return None
    rank = {int(c): i for i, c in enumerate(order)}
    ranks = [rank[int(c)] for c in seed_channels if int(c) in rank]
    if not ranks:
        return None
    lo = max(0, min(ranks) - pad_strips)
    hi = min(len(order) - 1, max(ranks) + pad_strips)
    if dead is not None and len(dead):
        isd = np.isin(order, np.asarray(dead, dtype=int))
        lo, n = min(ranks), 0
        while lo > 0 and n < pad_strips:
            lo -= 1
            n += not isd[lo]
        hi, n = max(ranks), 0
        while hi < len(order) - 1 and n < pad_strips:
            hi += 1
            n += not isd[hi]
    ch = order[lo:hi + 1]
    return PlaneWindow(ch=ch.astype(np.int16), pos=pos_map[ch].astype(np.float32),
                       W=wfm_all[ch].astype(np.float32),
                       noise=np.maximum(noise_all[ch], 3.0).astype(np.float32))
