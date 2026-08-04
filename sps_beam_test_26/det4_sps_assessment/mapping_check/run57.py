"""Does the measured map hold on the rotated mount, and what is the yaw angle?

A yaw about the vertical axis foreshortens only the horizontal detector axis, so
after the alignment the affine's two singular values differ by 1/cos(yaw).
"""
import numpy as np

BASE = ("/media/dylan/data/x17/sps_run53_det4_check/flat_ArCO2iso_95-3-2__run53-56/"
        "mapping_check/")
l = np.arange(512) % 64
Dn = np.arange(512) // 64
isx = Dn < 4
POS_NEW = 99.84 + 49.92 * (Dn % 4) + (49.14 - 0.78 * l)
POS_OLD = 99.84 + 49.92 * (Dn % 4) + 0.78 * l


def run(tag, npz, z=1120.0):
    D = np.load(BASE + npz)
    h_ev, h_ch, h_amp, h_t = D["h_ev"], D["h_ch"], D["h_amp"], D["h_time"]
    n_ev = len(D["fx_p"])
    clean = np.ones(n_ev, bool)
    for k in ("fx", "fy", "bx", "by"):
        clean &= (D[k + "_n"] == 1) & np.isfinite(D[k + "_p"])
    tx = D["fx_p"] + (D["bx_p"] - D["fx_p"]) * z / 1370.0
    ty = D["fy_p"] + (D["by_p"] - D["fy_p"]) * z / 1370.0

    print(f"\n================ {tag} ================")
    print(f"clean uRWELL tracks: {clean.sum()} of {n_ev}")
    for name, lut in (("old map", POS_OLD), ("measured map", POS_NEW)):
        sel = (h_amp > 60.0) & (h_t > 600.0) & (h_t < 1850.0) & clean[h_ev]
        lead, ncl = {}, {}
        for v, want in (("x", True), ("y", False)):
            k = sel & (isx[h_ch] == want)
            ev, pos, amp = h_ev[k], lut[h_ch[k]], h_amp[k]
            o = np.lexsort((pos, ev))
            ev, pos, amp = ev[o], pos[o], amp[o]
            new = np.empty(len(ev), bool); new[0] = True
            new[1:] = (ev[1:] != ev[:-1]) | (np.diff(pos) > 3.0)
            cid = np.cumsum(new) - 1
            nc = cid[-1] + 1
            cq = np.bincount(cid, weights=amp, minlength=nc)
            cp = np.bincount(cid, weights=pos * amp, minlength=nc) / np.maximum(cq, 1e-9)
            cev = np.zeros(nc, np.int64); cev[cid] = ev
            n_ = np.zeros(n_ev, np.int32); np.add.at(n_, cev, 1)
            L = np.full(n_ev, np.nan)
            o2 = np.argsort(cq, kind="stable")
            L[cev[o2]] = cp[o2]
            lead[v], ncl[v] = L, n_
        ok = clean & (ncl["x"] == 1) & (ncl["y"] == 1) \
            & np.isfinite(lead["x"] + lead["y"] + tx + ty)
        U = np.column_stack([tx[ok], ty[ok], np.ones(ok.sum())])
        V = np.column_stack([lead["x"][ok], lead["y"][ok]])
        A, *_ = np.linalg.lstsq(U, V, rcond=None)
        for _ in range(3):
            r = np.hypot(*(V - U @ A).T)
            g = r < np.percentile(r, 80)
            A, *_ = np.linalg.lstsq(U[g], V[g], rcond=None)
        r = np.hypot(*(V - U @ A).T)
        M = A[:2].T
        sv = np.linalg.svd(M, compute_uv=False)
        print(f"  [{name:12s}] n={ok.sum():6d}  median |res| {np.median(r):6.2f} mm"
              f"  <1 mm {np.mean(r < 1):5.1%}  rot {np.degrees(np.arctan2(M[1,0], M[0,0])):+7.2f} deg"
              f"  singular values {sv[0]:.4f}/{sv[1]:.4f}"
              f"  -> yaw {np.degrees(np.arccos(min(sv)/max(sv))):5.1f} deg")


run("run 53  (flat mount)", "det4_run_53_v2.npz")
run("run 57  (rotated ~25 deg)", "det4_run_57_v2.npz")
