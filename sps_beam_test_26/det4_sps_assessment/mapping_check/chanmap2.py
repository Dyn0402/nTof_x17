"""Measured det4 channel -> position map, take 2.

Adds the in-time cut (the det4 drift window is 600-1850 ns, read straight off
the hit-time spectrum) and replaces the plain median with an excess-correlation
centroid, so a flat accidental pedestal cannot drag every channel towards the
middle of the beam spot.
"""
import numpy as np

BASE = ("/media/dylan/data/x17/sps_run53_det4_check/flat_ArCO2iso_95-3-2__run53-56/"
        "mapping_check/")
D = np.load(BASE + "det4_run_53_raw.npz")
fx, fy = D["fx"], D["fy"]
h_ev, h_ch, h_amp, h_t = D["h_ev"], D["h_ch"], D["h_amp"], D["h_time"]

good_ev = np.isfinite(fx) & np.isfinite(fy)
sel = good_ev[h_ev] & (h_amp > 60.0) & (h_t > 600.0) & (h_t < 1850.0)
ch, ux, uy = h_ch[sel], fx[h_ev[sel]], fy[h_ev[sel]]
print(f"hits used: {sel.sum()}")

BINW = 2.0
def excess_centroid(pos):
    lo, hi = 0.0, 128.0
    nb = int((hi - lo) / BINW)
    ib = np.clip(((pos - lo) / BINW).astype(int), 0, nb - 1)
    N = np.zeros((512, nb))
    np.add.at(N, (ch, ib), 1.0)
    marg = N.sum(0)
    ncy = N.sum(1)
    E = np.outer(ncy, marg) / max(N.sum(), 1)
    X = N - E                                  # excess over the accidental floor
    centers = lo + BINW * (np.arange(nb) + 0.5)
    cen = np.full(512, np.nan)
    frac = np.full(512, np.nan)
    for c in range(512):
        w = X[c].copy()
        if ncy[c] < 100:
            continue
        w[w < 0] = 0
        if w.sum() <= 0:
            continue
        # tighten onto the peak: keep bins within 15 mm of the argmax
        pk = centers[np.argmax(w)]
        m = np.abs(centers - pk) <= 15.0
        cen[c] = (w[m] * centers[m]).sum() / w[m].sum()
        frac[c] = w.sum() / ncy[c]
    return cen, frac, N

cen_x, frac_x, Nx = excess_centroid(ux)
cen_y, frac_y, Ny = excess_centroid(uy)
np.savez(BASE + "chanmap2.npz", cen_x=cen_x, cen_y=cen_y,
         frac_x=frac_x, frac_y=frac_y, n=Nx.sum(1))

lc = np.arange(64)
print(f"\n{'conn':>4} {'nch':>4} | {'slope vs uRW-x':>15} {'r':>6} |"
      f" {'slope vs uRW-y':>15} {'r':>6} | verdict")
for c in range(8):
    lo, hi = c * 64, c * 64 + 64
    m = np.isfinite(cen_x[lo:hi]) & np.isfinite(cen_y[lo:hi])
    if m.sum() < 10:
        print(f"{c:4d} {m.sum():4d} | -- too few live channels --")
        continue
    res = []
    for cen in (cen_x, cen_y):
        v = cen[lo:hi][m]
        s, i = np.polyfit(lc[m], v, 1)
        r = np.corrcoef(lc[m], v)[0, 1]
        res.append((s, i, r))
    (sx, ix, rx), (sy, iy, ry) = res
    if abs(rx) > abs(ry):
        which, s, i = "measures uRW-x", sx, ix
    else:
        which, s, i = "measures uRW-y", sy, iy
    print(f"{c:4d} {m.sum():4d} | {sx:15.4f} {rx:6.2f} | {sy:15.4f} {ry:6.2f} |"
          f" {which}  pitch {s:+.3f} mm/ch, ch0 at {i:6.1f} mm")
