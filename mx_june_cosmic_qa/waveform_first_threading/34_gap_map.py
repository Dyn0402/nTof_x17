#!/usr/bin/env python3
"""Gap topography: U50 endpoint vs track position on det3.

If the soft ensemble edge (27->30 mm) is a bowed/tilted cathode, U50 varies
coherently with (x,y); if it is a uniform per-event effect, the map is flat
at ~28 mm with the same soft edge everywhere.
Ref-pinned fits, K=24, all matched events with |tan|>=0.08 in either plane.
"""
import os, sys, pickle, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/dylan/PycharmProjects/nTof_x17/mx_june_cosmic_qa/waveform_first_threading')
import forward_model2 as fm2

B3 = ('/home/dylan/x17/cosmic_bench/Analysis/mx17_det3_saturday_scan_6-27-26/'
      'long_run_resist_490V_drift_1000V/mx17_3/waveform_first')
hj = json.load(open(os.path.join(B3, 'hyper_v2.json')))
H3 = {k: hj[k] for k in ('c1', 'c2', 'kY', 'tau_s', 'sigma_s', 'sigma_p0', 'Dp')}
V = hj['v']
DT = 60.0
K = 24

ev3 = pickle.load(open(os.path.join(B3, 'wfcache.pkl'), 'rb'))['events']
sel = [e for e, ev in ev3.items()
       if max(abs(ev['tan_x']), abs(ev['tan_y'])) > 0.08]
print(f'{len(sel)} events', flush=True)

def fit_one(eid):
    fm2.K = K
    fm2.UK = (np.arange(K) + 0.5) * DT
    ev = ev3[eid]
    out = []
    for plane in ('x', 'y'):
        tn = ev[f'tan_{plane}']
        if abs(tn) < 0.08:
            continue
        P = ev[plane]
        W, noise, pos, sat = fm2.prep_plane(P, plane)
        best = (np.inf, None)
        for t0 in np.arange(150.0, 640.0, 30.0):
            c, q = fm2.chi2_plane(plane, W, noise, pos, sat,
                                  ev[f'ref_mesh_{plane}'], tn * V * 1e-3,
                                  float(t0), H3)
            if c < best[0]:
                best = (c, q)
        q = best[1]
        if q is not None and 0 < q.sum() and q.max() < 2e4:
            out.append((ev['ref_mesh_x'], ev['ref_mesh_y'], q / q.sum()))
    return out

def U50g(uk, prof, pb=(1, 6)):
    plat = np.median(prof[pb[0]:pb[1]])
    below = np.where(prof < 0.5 * plat)[0]
    below = below[below >= pb[1] - 1]
    if len(below) == 0 or plat <= 0:
        return np.nan
    j = below[0]
    x0, x1 = uk[j - 1], uk[j]
    y0, y1 = prof[j - 1], prof[j]
    return x0 + (0.5 * plat - y0) / (y1 - y0) * (x1 - x0) if y1 != y0 else x1

if __name__ == '__main__':
    t_ = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=14) as pool:
        for o in pool.map(fit_one, sel, chunksize=8):
            rows.extend(o)
    print(f'{len(rows)} plane-profiles in {time.time()-t_:.0f}s', flush=True)
    X = np.array([r[0] for r in rows])
    Y = np.array([r[1] for r in rows])
    L = max(len(r[2]) for r in rows)
    P = np.zeros((len(rows), L))
    for i, r in enumerate(rows):
        P[i, :len(r[2])] = r[2]
    uk = (np.arange(L) + 0.5) * DT

    # 3x3 map over the populated area
    xq = np.percentile(X, [0, 33, 66, 100])
    yq = np.percentile(Y, [0, 33, 66, 100])
    Umap = np.full((3, 3), np.nan)
    Nmap = np.zeros((3, 3), int)
    for i in range(3):
        for j in range(3):
            m = (X >= xq[i]) & (X < xq[i + 1]) & (Y >= yq[j]) & (Y < yq[j + 1])
            if m.sum() > 150:
                Umap[i, j] = U50g(uk, P[m].mean(axis=0))
                Nmap[i, j] = m.sum()
    print('U50 map [ns] (rows=x terciles, cols=y terciles):')
    print(np.round(Umap, 0))
    print('-> column [mm]:')
    print(np.round(Umap * V * 1e-3, 1))
    print('N per cell:'); print(Nmap)

    # finer 1D profiles vs x and vs y
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, coord, lab in ((axs[0], X, 'x'), (axs[1], Y, 'y')):
        bins = np.percentile(coord, np.linspace(0, 100, 9))
        cent, Us, Ns = [], [], []
        for a, b in zip(bins[:-1], bins[1:]):
            m = (coord >= a) & (coord < b)
            if m.sum() > 120:
                cent.append(0.5 * (a + b))
                Us.append(U50g(uk, P[m].mean(axis=0)) * V * 1e-3)
                Ns.append(m.sum())
        ax.plot(cent, Us, 'ko-')
        ax.axhline(29, color='r', ls='--'); ax.axhline(30, color='gray', ls=':')
        ax.set_xlabel(f'track {lab} at mesh [mm]')
        ax.set_ylabel('column [mm]'); ax.grid(alpha=0.3)
        ax.set_title(f'column vs {lab}')
        print(f'vs {lab}:', [f'{c:.0f}:{u:.1f}' for c, u in zip(cent, Us)])
    im = axs[2].imshow(Umap.T * V * 1e-3, origin='lower', cmap='viridis')
    plt.colorbar(im, ax=axs[2], label='column [mm]')
    axs[2].set_title('3x3 column map [mm]')
    axs[2].set_xlabel('x tercile'); axs[2].set_ylabel('y tercile')
    fig.tight_layout()
    fig.savefig(os.path.join(B3, 'gap_map.png'), dpi=110)
    print('saved gap_map.png')
