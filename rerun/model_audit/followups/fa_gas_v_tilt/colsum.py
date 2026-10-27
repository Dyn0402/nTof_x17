#!/usr/bin/env python3
"""colsum.py <key> --bundle B --out OUT.npz [--max-events N] [--jobs J]  -- model-free column duration, every trigger.

FA item 1 (v(t), v(x, y)) and item 2 (per-view column duration).  Runs inside rerun/stage_job.py (the row's decoded
waveforms are staged; ``qa_config`` resolves the row).  No M3, no track model, no hit times: per trigger and view

  1. pedestal + common mode + noise exactly as the reconstruction reads them (wft.io.FeuReader);
  2. strips with a pulse (peak > NSIG x noise, the bundle's dead channels left out), grouped in position order into
     clusters (gaps of <= GAP_STRIPS strips bridged); the cluster with the largest summed peak is the track;
  3. the strip sum S(t) of that cluster +- PAD strips.  Summed over strips every lateral term of the forward model drops
     out (strip fractions sum to one; the RC increments telescope to zero), so for ANY track angle S is the column's
     charge arrival profile q(u) convolved with the electronics template h (ws_spsv/boxfit.py):
         S(t) = a [G(t-t0) - G(t-t0-T)] + b [ (J(t-t0) - J(t-t0-T))/T - G(t-t0-T) ]
     a box of duration T = gap / v with a linear charge gradient, its far end smeared by SIG_END;
  4. (t0, T) on a 5 ns x 10 ns grid, (a, b) by linear least squares at every grid point (vectorised: the 2x2 normal
     matrices are precomputed), a > 0 and a + b > 0; then a parabola through the chi2 minimum in T.

Saved per trigger and view: eventId, DREAM timestamp, ftst, acquisition index, number of pulse strips, number of
clusters, charge centroid [mm] and rms width, summed charge, peak of S, saturation flag, fitted t0, T, a, b, chi2/dof,
and S itself (float32) so the fit can be redone offline.  The template is the bundle's ``tmpl_x`` for both views (the
clean response; the Y RC spreading cancels in the strip sum) and, for the per-view check, ``tmpl_y`` on Y as well.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]

NSIG = 5.0          # pulse strip: peak > NSIG x noise
GAP_STRIPS = 2      # bridge up to this many quiet strips inside a cluster
PAD = 2             # strips added on each side of the cluster before summing (the sharing kernel reaches +-2)
SIG_END = 30.0      # far-end smearing of the box [ns] (ws_spsv)
DTS = 60.0
T0_GRID = np.arange(0.0, 900.0 + 1e-9, 5.0)
T_GRID = np.arange(300.0, 2000.0 + 1e-9, 10.0)


class Kern:
    """Closed-form box integrals through the template's running integrals (ws_spsv/boxfit.py)."""

    def __init__(self, grid, tmpl, sig_end=SIG_END):
        g = np.arange(grid[0], grid[-1] + 6000.0, 1.0)
        h = np.interp(g, grid, tmpl, left=0.0, right=0.0)
        G = np.cumsum(h); J = np.cumsum(G)
        x = np.arange(-int(4 * sig_end), int(4 * sig_end) + 1)
        k = np.exp(-0.5 * (x / sig_end) ** 2); k /= k.sum(); pad = len(x) // 2
        self.Gs = np.convolve(np.r_[np.zeros(pad), G, np.full(pad, G[-1])], k, 'valid')
        self.Js = np.convolve(np.r_[np.zeros(pad), J, J[-1] + G[-1] * np.arange(1, pad + 1)], k, 'valid')
        self.g, self.G, self.J = g, G, J

    def basis(self, ts, t0, T):
        tau = ts - t0
        G0 = np.interp(tau, self.g, self.G, left=0.0); J0 = np.interp(tau, self.g, self.J, left=0.0)
        G1 = np.interp(tau - T, self.g, self.Gs, left=0.0); J1 = np.interp(tau - T, self.g, self.Js, left=0.0)
        return np.stack([G0 - G1, (J0 - J1) / max(T, 1.0) - G1], 1)


class GridFit:
    """All (t0, T) grid points at once: B [n_t0, n_T, ns, 2], with the inverse normal matrices."""

    def __init__(self, kern, ns):
        ts = np.arange(ns) * DTS
        B = np.empty((len(T0_GRID), len(T_GRID), ns, 2))
        for i, t0 in enumerate(T0_GRID):
            for j, T in enumerate(T_GRID):
                B[i, j] = kern.basis(ts, t0, T)
        M = np.einsum('ijsk,ijsl->ijkl', B, B)
        det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] ** 2
        good = det > 1e-9 * np.maximum(M[..., 0, 0] * M[..., 1, 1], 1e-30)
        inv = np.zeros_like(M)
        inv[..., 0, 0] = M[..., 1, 1]; inv[..., 1, 1] = M[..., 0, 0]
        inv[..., 0, 1] = inv[..., 1, 0] = -M[..., 0, 1]
        inv /= np.where(good, det, 1.0)[..., None, None]
        self.B, self.inv, self.good, self.ns = B, inv, good, ns

    def fit(self, y):
        Bty = np.einsum('ijsk,s->ijk', self.B, y)
        c = np.einsum('ijkl,ijl->ijk', self.inv, Bty)
        chi = y @ y - np.einsum('ijk,ijk->ij', c, Bty)
        ok = self.good & (c[..., 0] > 0) & (c[..., 0] + c[..., 1] > 0)
        chi = np.where(ok, chi, np.inf)
        i, j = np.unravel_index(np.argmin(chi), chi.shape)
        if not np.isfinite(chi[i, j]):
            return (np.nan,) * 5
        T = T_GRID[j]
        if 0 < j < len(T_GRID) - 1 and np.all(np.isfinite(chi[i, j - 1:j + 2])):
            c0, c1, c2 = chi[i, j - 1], chi[i, j], chi[i, j + 1]
            den = c0 - 2 * c1 + c2
            if den > 0:
                T = T + 0.5 * (c0 - c2) / den * (T_GRID[1] - T_GRID[0])
        return float(T0_GRID[i]), float(T), float(c[i, j, 0]), float(c[i, j, 1]), float(chi[i, j] / max(self.ns - 4, 1))


def cluster(W, noise, pos, dead):
    """main cluster of a plane: (channel list to sum, n pulse strips, n clusters, centroid, rms, saturated peak)."""
    pk = W.max(axis=1)
    live = np.isfinite(pos) & ~dead
    sig = live & (pk > NSIG * np.maximum(noise, 3.0))
    if sig.sum() < 3:
        return None
    order = np.argsort(np.where(np.isfinite(pos), pos, np.inf))
    order = order[np.isfinite(pos[order])]
    rk_sig = np.flatnonzero(sig[order])
    groups, cur = [], [rk_sig[0]]
    for r in rk_sig[1:]:
        if r - cur[-1] <= GAP_STRIPS + 1:
            cur.append(r)
        else:
            groups.append(cur); cur = [r]
    groups.append(cur)
    score = [pk[order[g]].sum() for g in groups]
    gi = int(np.argmax(score))
    ncl = int(sum(s > 0.2 * score[gi] for s in score))
    g = groups[gi]
    if len(g) < 3:
        return None
    lo, hi = max(0, g[0] - PAD), min(len(order) - 1, g[-1] + PAD)
    ch = order[lo:hi + 1]
    ch = ch[live[ch]]
    q = np.clip(W[ch], 0, None).sum(axis=1)
    if q.sum() <= 0:
        return None
    cen = float((pos[ch] * q).sum() / q.sum())
    rms = float(np.sqrt(((pos[ch] - cen) ** 2 * q).sum() / q.sum()))
    return ch, len(g), ncl, cen, rms, float(pk[ch].max())


def plane_pass(args):
    """one FEU file: per event the cluster summary and strip sum."""
    path, pos, dead_ch, sat_adc = args
    from wft import io as wio
    rdr = wio.FeuReader(path)
    dead = np.zeros(512, bool); dead[[c for c in dead_ch if 0 <= c < 512]] = True
    ts = rdr.tree.arrays(['eventId', 'timestamp'], library='np')
    tsmap = dict(zip(ts['eventId'].astype(np.int64).tolist(), ts['timestamp'].astype(np.int64).tolist()))
    out = {}
    for eid, ftst, W in rdr.iter_events(None):
        r = cluster(W, rdr.noise, pos, dead)
        if r is None:
            continue
        ch, nsig, ncl, cen, rms, pkmax = r
        S = W[ch].sum(axis=0)
        out[eid] = (tsmap.get(eid, -1), ftst, nsig, ncl, cen, rms, float(np.clip(S, 0, None).sum()), float(S.max()),
                    pkmax >= sat_adc, S.astype(np.float32))
    return path, out


def fit_chunk(args):
    S, grid, tmpl_x, tmpl_y, ns = args
    gx = GridFit(Kern(grid, tmpl_x), ns)
    gy = GridFit(Kern(grid, tmpl_y), ns) if tmpl_y is not None else None
    rx = np.array([gx.fit(s.astype(float)) for s in S])
    ry = np.array([gy.fit(s.astype(float)) for s in S]) if gy is not None else None
    return rx, ry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key'); ap.add_argument('--bundle', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=8); ap.add_argument('--max-files', type=int, default=0)
    a = ap.parse_args()
    from qa_config import get_config
    from wft import io as wio
    from wft.calib import CalibrationBundle
    cfg = get_config(a.key)
    cal = CalibrationBundle.load(a.bundle)
    z = np.load(os.path.join(a.bundle, 'arrays.npz'))
    grid, tx, ty = z['grid'], z['tmpl_x'], z['tmpl_y']
    pmaps = wio.strip_position_map(cfg)
    tasks, tags = [], {}
    for plane, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
        files = wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu)
        if a.max_files:
            files = files[:a.max_files]
        for f in files:
            tags[f] = (plane, wio.file_tag(f))
            tasks.append((f, pmaps[feu], list(cal.dead.get(plane, [])), float(cal.sat_adc)))
    print(f'[colsum] {a.key}: {len(tasks)} FEU files', flush=True)
    t1 = time.time()
    per = {'x': {}, 'y': {}}
    acq = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for path, out in ex.map(plane_pass, tasks):
            plane, tag = tags[path]
            k = acq.setdefault(tag, len(acq))
            for eid, v in out.items():
                per[plane][(k, eid)] = v
            print(f'[colsum] {os.path.basename(path)}: {len(out)} events with a cluster ({time.time() - t1:.0f} s)', flush=True)
    keys = sorted(set(per['x']) | set(per['y']))
    n = len(keys); ns = None
    for p in 'xy':
        for v in per[p].values():
            ns = len(v[-1]); break
    rec = dict(acq=np.array([k[0] for k in keys], np.int16), eid=np.array([k[1] for k in keys], np.int64))
    for p in 'xy':
        cols = dict(ts=np.full(n, -1, np.int64), ftst=np.full(n, -1, np.int16), nsig=np.zeros(n, np.int16),
                    ncl=np.zeros(n, np.int16), cen=np.full(n, np.nan, np.float32), rms=np.full(n, np.nan, np.float32),
                    q=np.full(n, np.nan, np.float32), smax=np.full(n, np.nan, np.float32), sat=np.zeros(n, bool),
                    S=np.zeros((n, ns), np.float32), has=np.zeros(n, bool))
        for i, k in enumerate(keys):
            v = per[p].get(k)
            if v is None:
                continue
            cols['has'][i] = True
            for j, name in enumerate(['ts', 'ftst', 'nsig', 'ncl', 'cen', 'rms', 'q', 'smax', 'sat', 'S']):
                cols[name][i] = v[j]
        for name, arr in cols.items():
            rec[f'{p}_{name}'] = arr
    print(f'[colsum] {n} triggers with a cluster in at least one view; fitting the strip sums', flush=True)
    for p in 'xy':
        idx = np.flatnonzero(rec[f'{p}_has'])
        chunks = np.array_split(idx, max(1, a.jobs * 4))
        res_x = np.full((n, 5), np.nan); res_y = np.full((n, 5), np.nan)
        with ProcessPoolExecutor(a.jobs) as ex:
            for ch, (rx, ry) in zip(chunks, ex.map(fit_chunk, [(rec[f'{p}_S'][c], grid, tx, ty if p == 'y' else None, ns)
                                                                for c in chunks])):
                if len(ch):
                    res_x[ch] = rx
                    if ry is not None:
                        res_y[ch] = ry
        for j, name in enumerate(['t0', 'T', 'a', 'b', 'chi2dof']):
            rec[f'{p}_{name}'] = res_x[:, j].astype(np.float32)
            if p == 'y':
                rec[f'{p}_{name}_ty'] = res_y[:, j].astype(np.float32)
        print(f'[colsum] {p}: fitted {np.isfinite(res_x[:, 1]).sum()} ({time.time() - t1:.0f} s)', flush=True)
    for p in 'xy':
        pass  # S kept float32 (saturated sums exceed float16)
    rec['acq_tags'] = np.array(sorted(acq, key=acq.get))
    rec['meta'] = np.array([f'key={a.key} bundle={a.bundle} NSIG={NSIG} GAP={GAP_STRIPS} PAD={PAD} SIG_END={SIG_END}'])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **rec)
    print(f'[colsum] wrote {a.out} ({n} triggers, {time.time() - t1:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
