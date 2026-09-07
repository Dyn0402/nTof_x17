#!/usr/bin/env python3
"""det4 spatial resolution at H4, with the uRWELL reference MEASURED, not assumed.

The question this answers: the cosmic bench has always reported ~0.5 mm and it was
never clear how much of that was the M3 reference.  Here the reference contribution
is measured from the data, because the back uRWELL carries THREE STRIP PITCHES
(0.5 / 1.5 / 1.0 mm) in one plane and the beam illuminates all three.  Its
contribution to the det4 residual therefore varies across the plane by a known
factor while everything else stays put -- that is the lever.

Geometry (one rail, `SPS_BEAM_GEOMETRY_2026-07-31.md`):

    uRWELL front  z =    0    1.0 mm pitch, both views
    P2_IN/MID/OUT z = 320/630/940   pad detectors, 3.4 mm residual -- useless here
    det4          z = 1120   0.78 mm pitch  <- the DUT
    uRWELL back   z = 1370   0.5 / 1.5 / 1.0 mm zones

The track is the front->back interpolation, so a straight track is reproduced
exactly and the beam divergence cancels; what is left at det4 is

    sigma_res^2 = sigma_det4^2 + 0.6683 * sigma_back^2 + 0.0333 * sigma_front^2

Run:  ../../../.venv/bin/python resolution.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "det4_sps_assessment"))
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM          # noqa: E402

DATA = Path("/media/dylan/data/x17/sps_run53_det4_check/"
            "flat_ArCO2iso_95-3-2__run53-56/mapping_check")
Z_DET4, Z_BACK = 1120.0, 1370.0
W_BACK = (Z_DET4 / Z_BACK) ** 2                 # 0.6683
W_FRONT = (1 - Z_DET4 / Z_BACK) ** 2            # 0.0333
BINARY = 1.0 / np.sqrt(12)                      # 0.2887 * pitch

# back-plane zones, from analysis/urw_mapping/mapping_urwell.csv
ZONES = {"x": [(0.0, 15.5, 0.5), (16.88, 63.38, 1.5), (64.44, 127.44, 1.0)],
         "y": [(0.0, 15.5, 0.5), (16.50, 63.00, 1.5), (64.25, 127.25, 1.0)]}


# ---------------------------------------------------------------- estimators
def gauss_sigma(r, win=1.2, nbins=60):
    """Gaussian fit to the core inside a FIXED window.

    A fixed window rather than an iterative n-sigma clip: the coincidence sample
    has a percent-level tail of mis-associated tracks (rms(all) is ~16 mm), and an
    adaptive clip latches onto that tail whenever a bin is thinly populated.
    """
    r = r[np.abs(r) < win]
    if len(r) < 150:
        return np.nan, np.nan, len(r)
    h, e = np.histogram(r, bins=nbins, range=(-win, win))
    c = 0.5 * (e[1:] + e[:-1])
    try:
        p, cov = curve_fit(lambda x, A, mu, s: A * np.exp(-0.5 * ((x - mu) / s) ** 2),
                           c, h, p0=[h.max(), 0.0, 0.3],
                           sigma=np.sqrt(np.maximum(h, 1)), maxfev=30000)
    except Exception:
        return np.nan, np.nan, len(r)
    return abs(p[2]), float(np.sqrt(abs(cov[2, 2]))), len(r)


def robust_lstsq(M, target, niter=6):
    """Least squares with a 3-sigma MAD trim -- the alignment must not chase tails."""
    keep = np.ones(len(target), bool)
    for _ in range(niter):
        coef, *_ = np.linalg.lstsq(M[keep], target[keep], rcond=None)
        r = target - M @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        keep = np.abs(r - np.median(r)) < 3 * s
    return coef, target - M @ coef


# ------------------------------------------------------------------- loading
def det4_clusters(ev, pos, amp, nev, gap_mm=3.0):
    """Charge-weighted leading cluster per event, clustered in POSITION.

    Clustering on the raw channel index would split every cluster straddling a
    connector boundary -- the plugs are inverted (DET4_URW_MAPPING_2026-08-01.md).
    """
    lead = np.full(nev, np.nan)
    ncl = np.zeros(nev, np.int16)
    size = np.zeros(nev, np.int16)
    order = np.lexsort((pos, ev))
    ev, pos, amp = ev[order], pos[order], amp[order]
    new = np.empty(len(ev), bool)
    new[0] = True
    new[1:] = (ev[1:] != ev[:-1]) | (np.diff(pos) > gap_mm)
    cid = np.cumsum(new) - 1
    nc = cid[-1] + 1
    cq = np.bincount(cid, weights=amp, minlength=nc)
    cpos = np.bincount(cid, weights=pos * amp, minlength=nc) / np.maximum(cq, 1e-9)
    csize = np.bincount(cid, minlength=nc)
    cev = np.zeros(nc, np.int64)
    cev[cid] = ev
    np.add.at(ncl, cev, 1)
    biggest = np.argsort(cq, kind="stable")          # ascending -> last write wins
    lead[cev[biggest]] = cpos[biggest]
    size[cev[biggest]] = csize[biggest]
    return lead, ncl, size


def load(fname, t_lo, t_hi, amp_min=60.0):
    """Clean 4-view uRWELL + single-cluster det4 coincidences."""
    d = np.load(DATA / fname)
    nev = len(d["fx_p"])
    ev, ch, amp, t = d["h_ev"], d["h_ch"], d["h_amp"], d["h_time"]
    sel = (amp > amp_min) & (t > t_lo) & (t < t_hi)   # det4 drift gate, per run
    ev, ch, amp = ev[sel], ch[sel], amp[sel]
    pos, is_x = POSITION_MM[ch], VIEW[ch] == "x"
    dx, nx, sx = det4_clusters(ev[is_x], pos[is_x], amp[is_x], nev)
    dy, ny, sy = det4_clusters(ev[~is_x], pos[~is_x], amp[~is_x], nev)
    m = ((d["fx_n"] == 1) & (d["fy_n"] == 1) & (d["bx_n"] == 1) & (d["by_n"] == 1)
         & (nx == 1) & (ny == 1) & np.isfinite(dx) & np.isfinite(dy)
         & np.isfinite(d["fx_p"]) & np.isfinite(d["fy_p"])
         & np.isfinite(d["bx_p"]) & np.isfinite(d["by_p"]))
    return dict(fx=d["fx_p"][m], fy=d["fy_p"][m], bx=d["bx_p"][m], by=d["by_p"][m],
                dx=dx[m], dy=dy[m], sx=sx[m], sy=sy[m], n=int(m.sum()))


def residuals(D, z=Z_DET4):
    """Align det4 to the interpolated track with a free affine, return residuals."""
    tx = D["fx"] + (D["bx"] - D["fx"]) * z / Z_BACK
    ty = D["fy"] + (D["by"] - D["fy"]) * z / Z_BACK
    M = np.column_stack([D["dx"], D["dy"], np.ones(len(D["dx"]))])
    cx, rx = robust_lstsq(M, tx)
    cy, ry = robust_lstsq(M, ty)
    A = np.array([[cx[0], cx[1]], [cy[0], cy[1]]])
    return rx, ry, A


# ------------------------------------------------------------------ analysis
def main():
    out = {}
    flat = load("det4_run_53_v2.npz", 600, 1850)
    rx, ry, A = residuals(flat)
    sv = np.linalg.svd(A, compute_uv=False)
    out["run53"] = dict(n=flat["n"], det_A=float(np.linalg.det(A)),
                        singular=[float(s) for s in sv],
                        strips_x=float(flat["sx"].mean()), strips_y=float(flat["sy"].mean()))
    print(f"run 53 (flat): {flat['n']} clean coincidences, det(A)={np.linalg.det(A):+.4f}")
    print(f"  det4 cluster size: X {flat['sx'].mean():.2f} strips, Y {flat['sy'].mean():.2f}")

    # -- (1) residual width per back-plane pitch zone, and the decomposition
    out["zones"] = {}
    for coord, r, bpos in [("uRW-x", rx, flat["bx"]), ("uRW-y", ry, flat["by"])]:
        rows = []
        for lo, hi, pitch in sorted(ZONES[coord[-1]], key=lambda z: z[2]):
            m = (bpos >= lo) & (bpos <= hi)
            s, es, n = gauss_sigma(r[m])
            if np.isnan(s) or n < 300:
                continue
            point = W_BACK * (BINARY * pitch) ** 2 + W_FRONT * (BINARY * 1.0) ** 2
            rows.append(dict(pitch=pitch, sigma_res=s, err=es, n=n,
                             pointing=float(np.sqrt(point)),
                             sigma_det4=float(np.sqrt(max(s ** 2 - point, 0)))))
            print(f"  {coord} back pitch {pitch}: res {s*1e3:6.1f}+-{es*1e3:.1f} um  "
                  f"pointing {np.sqrt(point)*1e3:5.1f}  -> det4 {rows[-1]['sigma_det4']*1e3:6.1f} um  (n={n})")
        out["zones"][coord] = rows

        if len(rows) == 3:   # free-slope fit: sigma_res^2 = c + W_back*(f*pitch)^2
            P = np.array([q["pitch"] for q in rows])
            y = np.array([q["sigma_res"] ** 2 for q in rows])
            ye = np.array([2 * q["sigma_res"] * q["err"] for q in rows])
            X = np.column_stack([np.ones(3), P ** 2])
            Ci = np.diag(1 / ye ** 2)
            cov = np.linalg.inv(X.T @ Ci @ X)
            b = cov @ (X.T @ Ci @ y)
            chi2 = float((y - X @ b) @ Ci @ (y - X @ b))
            f = float(np.sqrt(max(b[1], 0) / W_BACK))
            out["fit"] = dict(coord=coord, f_back=f, f_binary=float(BINARY),
                              sigma_det4=float(np.sqrt(max(b[0], 0))),
                              sigma_det4_err=float(0.5 * np.sqrt(cov[0, 0]) / np.sqrt(max(b[0], 1e-9))),
                              chi2=chi2)
            print(f"  -> FIT {coord}: back plane = {f:.3f} x pitch (binary {BINARY:.3f}), "
                  f"det4 = {np.sqrt(max(b[0],0))*1e3:.1f} um, chi2/1dof = {chi2:.1f}")

    # -- (2) the control: does the width STEP at the zone boundary?
    edges = np.arange(20, 105, 5.0)
    prof = {}
    for tag, r, bpos in [("signal_x", rx, flat["bx"]),      # pitch of THIS coord changes
                         ("control_x", ry, flat["bx"])]:    # same det4 region, pitch fixed
        pts = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (bpos >= lo) & (bpos < hi)
            s, es, n = gauss_sigma(r[m])
            if not np.isnan(s) and n > 400:
                pts.append(dict(z=float(0.5 * (lo + hi)), sigma=float(s), err=float(es), n=n))
        prof[tag] = pts
    out["profile"] = prof

    # -- (3) z-scan: the interpolation is exact, so the minimum locates det4
    zs = []
    zone = (flat["bx"] >= 64.44) & (flat["bx"] <= 127.44)
    for z in [800, 900, 1000, 1050, 1100, 1120, 1150, 1200, 1300, 1370]:
        rxz, _, _ = residuals(flat, z=z)
        s, _, _ = gauss_sigma(rxz[zone])
        zs.append(dict(z=z, sigma=float(s)))
        print(f"  z={z:5d}  sigma_res={s*1e3:6.1f} um")
    out["zscan"] = zs
    # divergence from the extra width at z=Z_BACK, once the weight change is removed
    s0 = [q["sigma"] for q in zs if q["z"] == 1120][0]
    s1 = [q["sigma"] for q in zs if q["z"] == 1370][0]
    d_point = (BINARY * 1.0) ** 2 - (W_BACK * (BINARY * 1.0) ** 2 + W_FRONT * (BINARY * 1.0) ** 2)
    theta = float(np.sqrt(max(s1 ** 2 - s0 ** 2 - d_point, 0)) / (Z_BACK - Z_DET4))
    out["divergence_rad"] = theta
    print(f"  -> beam divergence from the z-scan curvature: {theta*1e3:.2f} mrad")

    # -- (4) the same detector at 25.4 deg -- the inclined-track projection
    tilt = load("det4_run_57_v2.npz", 600, 3600)
    rxt, ryt, At = residuals(tilt)
    svt = np.linalg.svd(At, compute_uv=False)
    yaw = float(np.degrees(np.arccos(min(1.0, svt[1] / svt[0]))))
    tilt_rows = []
    for coord, r, bpos in [("uRW-x", rxt, tilt["bx"]), ("uRW-y", ryt, tilt["by"])]:
        m = (bpos >= 64.44) & (bpos <= 127.44) if coord == "uRW-x" else \
            (bpos >= 64.25) & (bpos <= 127.25)
        s, es, n = gauss_sigma(r[m], win=4.0, nbins=100)
        tilt_rows.append(dict(coord=coord, sigma_res=float(s), n=n))
        print(f"  run57 {coord}: sigma_res = {s*1e3:.0f} um (n={n})")
    out["tilt"] = dict(n=tilt["n"], yaw_deg=yaw, rows=tilt_rows)
    print(f"  run 57 yaw from the singular values: {yaw:.1f} deg")

    (HERE / "results.json").write_text(json.dumps(out, indent=2))
    np.savez(HERE / "residuals.npz", rx=rx, ry=ry, bx=flat["bx"], by=flat["by"],
             rx_tilt=rxt, ry_tilt=ryt, bx_tilt=tilt["bx"], by_tilt=tilt["by"])
    print(f"\nwrote {HERE/'results.json'} and residuals.npz")


if __name__ == "__main__":
    main()
