#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
digitise.py -- put Geant4 (or synthetic) drift-gas ionisation through the
detector response and the PRODUCTION reconstruction, unchanged.
HANDOFF_TRACKING_2026-10-06.md §12-13.

THE QUESTION.  With an ideal line through the gap ionisation, the capsule
estimators read ~1 and flat in angle on the simulated beam electrons (§12).
The data, on a bundle that closes on cosmic muons (`is2_A`), read 1.04 -> 0.73
between |tan| 0.1 and 0.45.  Does the real reconstruction compress electron
angles, where it does not compress muon angles?

WHAT IS SIMULATED, per electron of ionisation (one DriftGas step -> Poisson
edep/W_ION electrons):
  drift      arrival u = d / v_true after t0 (d = W_MESH - w), with transverse
             diffusion sigma^2 = sigma_p0^2 + Dp^2 u -- the bundle's own pair,
             sampled per electron, not smeared
  gain       Polya (Gamma, shape POLYA_K) per electron; the x/y split fixed
  strips     nearest strip of the run's strip map (pitch 0.78)
  response   the bundle's impulse template and resistive kernel
             (c1, c2 = c2_over_c1 c1, kY on y, share_mode), each electron at
             its own arrival time on a 5 ns grid, through wft.model's own
             pieces, then x the bundle's per-channel gain
  readout    added to a REAL quiet beam trigger's raw ADC (same run, same
             FEUs), clipped at the 12-bit rail, then the exact FeuReader
             pedestal + 64-channel common-mode subtraction
  hits       emulated: per channel the peak sample above 5 sigma, sigma from
             the real hit finder (median amplitude/significance per channel)
Then: `wft_beam.seeds_from_hits_beam` -> `wft.io.extract_window` ->
`wft.reco._worker_fit`, i.e. the production path, under any bundle.

So the response model IS the fit model: a straight, uniform track would be
reproduced exactly.  What the test isolates is what the fit model does not
contain -- curved / scattered / delta-ray ionisation, per-electron
fluctuations, real noise and CNS, the seeder, window truncation -- which is
exactly what differs between a few-MeV electron and a muon.  It does NOT test
a mismatch between the model and the real chamber; that part is calibrated on
cosmic muons (is2 closes on the A-C line).

Gate: synthetic straight muons must read the bundle's 1/kw, flat in angle.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd

W_MESH = 30.1            # mm, sim local w of the mesh (DriftGas 0.1-30.1)
W_ION = 26.0             # eV per ion pair, Ar/iC4H10 90/10
POLYA_K = 1.5            # avalanche gain shape (Gamma)
T_BIN = 5.0              # ns, arrival-time grid of the response
ADC_RAIL = 4095.0
HIT_SIG = 5.0
FEUS = {'A': (3, 4), 'C': (7, 8)}


# ----------------------------------------------------------------- overlay
class Overlay:
    """Quiet real triggers of one file tag: raw - pedestal per FEU, for the
    signal to be added to before the FeuReader's own common-mode step."""

    def __init__(self, arm: str, run: str, sub: str, tag: str, max_hits: int = 4,
                 n_max: int = 400):
        import uproot
        from ntof_tracking import wft_beam as WB
        from wft import io as wio
        self.cfg = WB.beam_config(arm, run, sub)
        self.pos_maps = wio.strip_position_map(self.cfg)
        fx, fy = self.cfg.MX17_FEU_X, self.cfg.MX17_FEU_Y
        self.feu = {'x': fx, 'y': fy}
        hp = WB.hits_file_for_tag(self.cfg, tag)
        H = WB.read_hits_tag(hp, (fx, fy))
        # the real hit finder's per-channel sigma
        H = H[H.significance > 0]
        self.hf_noise = {}
        for p, f in self.feu.items():
            s = (H[H.feu == f].amplitude / H[H.feu == f].significance).groupby(H.channel).median()
            arr = np.full(512, np.nan)
            arr[s.index.to_numpy(int)] = s.to_numpy()
            self.hf_noise[p] = arr
        files = {p: [x for x in wio.subrun_files(self.cfg.BASE_PATH, run, sub, f)
                     if wio.file_tag(x) == tag][0] for p, f in self.feu.items()}
        self.rdr = {p: wio.FeuReader(files[p]) for p in files}
        for p in 'xy':
            m = ~np.isfinite(self.hf_noise[p])
            self.hf_noise[p][m] = np.maximum(self.rdr[p].noise[m], 3.0)
        # quiet: few hits on both planes
        nh = H.groupby(['eventId', 'feu']).size().unstack(fill_value=0)
        ids = set(self.rdr['x'].event_ids.tolist()) & set(self.rdr['y'].event_ids.tolist())
        busy = set(nh[(nh.get(fx, 0) > max_hits) | (nh.get(fy, 0) > max_hits)].index)
        quiet = sorted(ids - busy)[:n_max]
        self.events = []
        raw = {}
        for p in 'xy':
            r = self.rdr[p]
            idx = np.where(np.isin(r.event_ids, quiet))[0]
            raw[p] = {}
            for lo in range(0, len(idx), 400):
                b = idx[lo:lo + 400]
                arr = r.tree.arrays(['eventId', 'amplitude', 'ftst'], entry_start=int(b[0]),
                                    entry_stop=int(b[-1]) + 1, library='np')
                for i in b:
                    j = i - int(b[0])
                    a = arr['amplitude'][j].reshape(-1, 512).astype(np.float32)
                    if a.shape[0] != r.n_sample:
                        continue
                    raw[p][int(arr['eventId'][j])] = (a, int(arr['ftst'][j]))
        for e in quiet:
            if e in raw['x'] and e in raw['y']:
                self.events.append(dict(eid=e, x=raw['x'][e][0], y=raw['y'][e][0],
                                        ftst_x=raw['x'][e][1], ftst_y=raw['y'][e][1]))
        self.n_sample = self.rdr['x'].n_sample
        self.ped = {p: self.rdr[p].ped for p in 'xy'}
        self.noise = {p: self.rdr[p].noise for p in 'xy'}

    def state(self) -> dict:
        """Picklable pieces a worker needs."""
        return dict(pos_maps={p: self.pos_maps[self.feu[p]] for p in 'xy'},
                    feu=self.feu, hf_noise=self.hf_noise, ped=self.ped,
                    noise=self.noise, n_sample=self.n_sample)


# ------------------------------------------------------------------ signal
def electrons(steps: pd.DataFrame, rng, v_um_ns: float, hyper: dict):
    """Per-electron (u, v, arrival time after t0, gain) at the mesh."""
    n = rng.poisson(np.clip(steps.edep.to_numpy(float), 0, None) / W_ION)
    keep = n > 0
    if not keep.any():
        return None
    n = n[keep]
    rep = lambda c: np.repeat(steps[c].to_numpy(float)[keep], n)   # noqa: E731
    u, v, w = rep('u'), rep('v'), rep('w')
    tp = rep('time') if 'time' in steps else 0.0
    d = np.clip(W_MESH - w, 0.0, None)
    ta = d / (v_um_ns * 1e-3)                                       # ns
    sig = np.sqrt(hyper['sigma_p0'] ** 2 + hyper['Dp'] ** 2 * ta)
    u = u + rng.normal(0, 1, len(u)) * sig
    v = v + rng.normal(0, 1, len(v)) * sig
    g = rng.gamma(POLYA_K, 1.0 / POLYA_K, len(u))
    return dict(u=u, v=v, t=ta + (tp - np.min(tp) if np.ndim(tp) else 0.0), g=g)


def plane_waveform(plane: str, coord: np.ndarray, t: np.ndarray, q: np.ndarray,
                   pos_map: np.ndarray, t0: float, hyper: dict) -> np.ndarray:
    """(512, NSAMP) pedestal-subtracted ADC for one plane, before gain."""
    from wft import model as wm
    order = np.argsort(pos_map)
    order = order[np.isfinite(pos_map[order])]
    ps = pos_map[order]
    out = np.zeros((512, wm.NSAMP))
    # nearest strip in position order
    j = np.searchsorted(ps, coord)
    j = np.clip(j, 1, len(ps) - 1)
    j = np.where(np.abs(coord - ps[j - 1]) < np.abs(coord - ps[j]), j - 1, j)
    inside = (coord > ps[0] - 0.39) & (coord < ps[-1] + 0.39)
    j, t, q = j[inside], t[inside], q[inside]
    if not len(j):
        return out
    tb = np.round(t / T_BIN).astype(int)
    lo, hi = max(j.min() - 3, 0), min(j.max() + 3, len(ps) - 1)
    ns = hi - lo + 1
    tbu, inv = np.unique(tb, return_inverse=True)
    Q = np.zeros((ns, len(tbu)))
    np.add.at(Q, (j - lo, inv), q)
    tmpl, _ = wm._templates(plane, hyper['sigma_s'])
    base = wm.TS[:, None] - (t0 + tbu[None, :] * T_BIN)              # (NSAMP, nb)
    H0 = np.interp(base, wm.TGRID, tmpl, left=0, right=0)
    H1, H2 = wm._copy_responses(plane, base, hyper)
    kY = hyper.get('kY', 1.0) if plane == 'y' else hyper.get('cX', 1.0)
    c1, c2 = hyper['c1'] * kY, hyper['c2'] * kY
    if hyper.get('c2_over_c1') is not None:
        c2 = float(hyper['c2_over_c1']) * c1
    Q1 = np.zeros_like(Q)
    Q1[1:] += Q[:-1]
    Q1[:-1] += Q[1:]
    W = Q @ H0.T + c1 * (Q1 @ H1.T)
    if c2 > 0:
        Q2 = np.zeros_like(Q)
        Q2[2:] += Q[:-2]
        Q2[:-2] += Q[2:]
        W += c2 * (Q2 @ H2.T)
    out[order[lo:hi + 1]] = W
    return out


def _cns(wfm: np.ndarray) -> np.ndarray:
    """FeuReader.iter_events' common-mode step on (ns, 512)."""
    from wft import io as wio
    ns = wfm.shape[0]
    nblk = 512 // wio.CNS_BLOCK
    cms = np.median(wfm.reshape(ns, nblk, wio.CNS_BLOCK), axis=2)
    return wfm - np.repeat(cms, wio.CNS_BLOCK, axis=1)


def emulate_hits(W: np.ndarray, hf_noise: np.ndarray, feu: int, eid: int) -> pd.DataFrame:
    """(512, ns) -> hits rows with the real hit finder's columns the seeder reads."""
    amp = W.max(axis=1)
    sig = amp / hf_noise
    ch = np.where(sig >= HIT_SIG)[0]
    k = W[ch].argmax(axis=1)
    # parabolic peak interpolation, as the hit finder reports a fractional sample
    kk = np.clip(k, 1, W.shape[1] - 2)
    a, b, c = W[ch, kk - 1], W[ch, kk], W[ch, kk + 1]
    den = a - 2 * b + c
    frac = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
    ms = np.where((k >= 1) & (k <= W.shape[1] - 2), kk + np.clip(frac, -0.5, 0.5), k)
    return pd.DataFrame(dict(eventId=eid, feu=feu, channel=ch, amplitude=amp[ch],
                             significance=sig[ch], max_sample=ms))


# ------------------------------------------------------------------- event
_ST: dict = {}


def worker_init(bundle_path: str, state: dict, min_strips: int):
    os.environ['WFT_BEAM_MIN_STRIPS'] = str(min_strips)
    from wft import reco as wreco
    from wft import model as wm
    wreco._worker_init(bundle_path)
    wm.set_nsamp(state['n_sample'])
    _ST.clear()
    _ST.update(state)
    _ST['min_strips'] = min_strips


def digitise_event(steps: pd.DataFrame, ov: dict, t0: float, adc_per_e: float,
                   xy_split: float, v_true: float, rng, pos_offset=(0.0, 0.0)):
    """One event's two planes: (W_x, W_y) as (512, ns) CNS'd ADC, plus truth."""
    from wft import model as wm
    from wft.reco import _CAL
    hyper = dict(wm.HYPER)
    el = electrons(steps, rng, v_true, hyper)
    W = {}
    for p in 'xy':
        if el is None:
            sig = np.zeros((512, wm.NSAMP))
        else:
            # strip-map position = HALF - local (build_tracks IN_PLANE_SIGN = -1)
            from ntof_tracking.run145_target_imaging import STRIP_MAP_HALF
            coord = STRIP_MAP_HALF - (el['u'] if p == 'x' else el['v']) + pos_offset[p == 'y']
            share = xy_split if p == 'x' else 1.0 - xy_split
            t0p = t0 if p == 'x' else t0 - _CAL.dt_xy.get(str(int(ov['ftst_x'] - ov['ftst_y'])),
                                                          _CAL.dt_xy.get(int(ov['ftst_x'] - ov['ftst_y']), -18.8))
            sig = plane_waveform(p, coord, el['t'], el['g'] * adc_per_e * share,
                                 _ST['pos_maps'][p], t0p, hyper)
            sig *= wm.GAIN[p][:, None]
        raw = ov[p] + sig.T                                               # (ns, 512) raw - nothing
        raw = np.clip(raw + 0.0, None, ADC_RAIL)
        W[p] = _cns(raw - _ST['ped'][p][None, :]).T
    return W


def reco_event(eid: int, W: dict, ftst: dict):
    from ntof_tracking import wft_beam as WB
    from wft import io as wio
    from wft import reco as wreco
    H = pd.concat([emulate_hits(W[p], _ST['hf_noise'][p], _ST['feu'][p], eid) for p in 'xy'])
    pm = {_ST['feu'][p]: _ST['pos_maps'][p] for p in 'xy'}
    seeds = WB.seeds_from_hits_beam(H, pm, _ST['feu']['x'], _ST['feu']['y'],
                                    hot=wreco._CAL.hot, min_strips=_ST['min_strips'])
    if eid not in seeds:
        return None
    wins, used = {}, {}
    for p in 'xy':
        cl = seeds[eid][p]
        ws, us = [], []
        for s in cl:
            win = wio.extract_window(W[p], _ST['noise'][p], _ST['pos_maps'][p], s.channels, 3)
            if win is None:
                continue
            ws.append(dict(W=win.W, pos=win.pos, noise=win.noise, ch=win.ch))
            us.append(s)
        if ws:
            wins[p], used[p] = ws, us
    if not wins:
        return None
    r = wreco._worker_fit((eid, wins, used, seeds[eid]['n_hits'], False, ftst))
    r.pop('_cand', None)
    return r


def run_one(job):
    """Worker: digitise + reconstruct one event.  job = dict(steps, ov, t0, ...)."""
    rng = np.random.default_rng(job['seed'])
    W = digitise_event(job['steps'], job['ov'], job['t0'], job['adc_per_e'],
                       job['xy_split'], job['v_true'], rng, job.get('pos_offset', (0.0, 0.0)))
    try:
        r = reco_event(job['eid'], W, dict(x=job['ov']['ftst_x'], y=job['ov']['ftst_y']))
    except Exception as err:                                               # noqa: BLE001
        r = dict(error=repr(err)[:200])
    r = r or {}
    r.update(job['truth'])
    return r


# --------------------------------------------------------------- synthetic
def straight_steps(rng, u0: float, v0: float, tan_u: float, tan_v: float,
                   clusters_per_mm: float = 2.7) -> pd.DataFrame:
    """A straight minimum-ionising track through the gap: Poisson clusters,
    1/n^2 cluster sizes (n <= 50).  u0/v0 at the mesh."""
    L = W_MESH - 0.1
    n = rng.poisson(clusters_per_mm * L * np.sqrt(1 + tan_u ** 2 + tan_v ** 2))
    w = rng.uniform(0.1, W_MESH, n)
    nn = np.arange(1, 51)
    p = 1.0 / nn ** 2
    sz = rng.choice(nn, n, p=p / p.sum())
    return pd.DataFrame(dict(u=u0 + tan_u * (w - W_MESH), v=v0 + tan_v * (w - W_MESH), w=w,
                             edep=sz * W_ION, time=0.0))
