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
                wfm = arr['amplitude'][j].reshape(-1, 512).astype(np.float32) - self.ped
                # the window length is per EVENT, not per file: det4's 6-24 run
                # mixes 32- and 37-sample events, so taking it from the pedestal
                # events crashes the common-mode reshape partway through a file
                ns = wfm.shape[0]
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
                   pad_strips: int = 3) -> Optional[PlaneWindow]:
    """Cut the fit window: the seed cluster's strips plus `pad_strips` on each
    side in *position* order (the sharing kernel reaches +-2 strips, so the pad
    must be at least 2 for the model to see where the shared charge went).

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
    ch = order[lo:hi + 1]
    return PlaneWindow(ch=ch.astype(np.int16), pos=pos_map[ch].astype(np.float32),
                       W=wfm_all[ch].astype(np.float32),
                       noise=np.maximum(noise_all[ch], 3.0).astype(np.float32))
