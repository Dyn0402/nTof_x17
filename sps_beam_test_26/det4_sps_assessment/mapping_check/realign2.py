"""det4 <-> uRWELL alignment on clean single-cluster events: old map vs measured map."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = ("/media/dylan/data/x17/sps_run53_det4_check/flat_ArCO2iso_95-3-2__run53-56/"
        "mapping_check/")
D = np.load(BASE + "det4_run_53_v2.npz")
h_ev, h_ch, h_amp, h_t = D["h_ev"], D["h_ch"], D["h_amp"], D["h_time"]
n_ev = len(D["fx_p"])

clean = np.ones(n_ev, bool)
for k in ("fx", "fy", "bx", "by"):
    clean &= (D[k + "_n"] == 1) & np.isfinite(D[k + "_p"])
print(f"uRWELL clean single-cluster tracks: {clean.sum()} of {n_ev}")

Z4, ZB = 1155.0, 1370.0
tx = D["fx_p"] + (D["bx_p"] - D["fx_p"]) * Z4 / ZB
ty = D["fy_p"] + (D["by_p"] - D["fy_p"]) * Z4 / ZB

l = np.arange(512) % 64
Dn = np.arange(512) // 64
isx = Dn < 4
CONN0, PITCH = 49.92, 0.78
MAPS = {
    "old  (channels forward)":  99.84 + CONN0 * (Dn % 4) + PITCH * l,
    "measured (channels reversed)": 99.84 + CONN0 * (Dn % 4) + (49.14 - PITCH * l),
}


def cluster_det4(pos_lut, gap=3.0):
    sel = (h_amp > 60.0) & (h_t > 600.0) & (h_t < 1850.0) & clean[h_ev]
    res = {}
    for v, want in (("x", True), ("y", False)):
        k = sel & (isx[h_ch] == want)
        ev, pos, amp = h_ev[k], pos_lut[h_ch[k]], h_amp[k]
        o = np.lexsort((pos, ev))
        ev, pos, amp = ev[o], pos[o], amp[o]
        lead = np.full(n_ev, np.nan)
        ncl = np.zeros(n_ev, np.int32)
        if len(ev):
            new = np.empty(len(ev), bool)
            new[0] = True
            new[1:] = (ev[1:] != ev[:-1]) | (np.diff(pos) > gap)
            cid = np.cumsum(new) - 1
            nc = cid[-1] + 1
            cq = np.bincount(cid, weights=amp, minlength=nc)
            cp = np.bincount(cid, weights=pos * amp, minlength=nc) / np.maximum(cq, 1e-9)
            cev = np.zeros(nc, np.int64); cev[cid] = ev
            np.add.at(ncl, cev, 1)
            order = np.argsort(cq, kind="stable")
            lead[cev[order]] = cp[order]
        res[v] = (lead, ncl)
    return res


def align(dx, dy, keep, tag):
    ok = keep & np.isfinite(dx + dy + tx + ty)
    U = np.column_stack([tx[ok], ty[ok], np.ones(ok.sum())])
    V = np.column_stack([dx[ok], dy[ok]])
    A, *_ = np.linalg.lstsq(U, V, rcond=None)
    # one robust re-fit: drop the worst 20 %, refit
    for _ in range(3):
        r = np.hypot(*(V - U @ A).T)
        good = r < np.percentile(r, 80)
        A, *_ = np.linalg.lstsq(U[good], V[good], rcond=None)
    res = V - U @ A
    r = np.hypot(res[:, 0], res[:, 1])
    M = A[:2].T
    print(f"\n--- {tag} ---")
    print(f"  events {ok.sum()};  rotation {np.degrees(np.arctan2(M[1,0], M[0,0])):+.2f} deg, "
          f"det(A) {np.linalg.det(M):+.4f}")
    print(f"  |residual| median {np.median(r):6.2f} mm | within 5 mm {np.mean(r<5):6.1%}"
          f" | within 2 mm {np.mean(r<2):6.1%} | within 1 mm {np.mean(r<1):6.1%}")
    core = r < np.percentile(r, 80)
    print(f"  core (best 80 %) rms x/y {res[core,0].std():.2f} / {res[core,1].std():.2f} mm")
    print(f"  transform: [[{M[0,0]:+.4f} {M[0,1]:+.4f}] [{M[1,0]:+.4f} {M[1,1]:+.4f}]]"
          f"  t = ({A[2,0]:.1f}, {A[2,1]:.1f}) mm")
    return ok, res, r


out = {}
for name, lut in MAPS.items():
    c = cluster_det4(lut)
    dx, ncx = c["x"]
    dy, ncy = c["y"]
    keep = (ncx == 1) & (ncy == 1)
    print(f"\n[{name}] det4 single-cluster in both views: {keep.sum()}")
    out[name] = align(dx, dy, keep, name) + (dx, dy, keep)

fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
for row, (name, (ok, res, r, dx, dy, keep)) in enumerate(out.items()):
    ax[row, 0].hist2d(ty[ok], dx[ok], bins=100, cmap="Blues")
    ax[row, 0].set(xlabel="uRWELL track y at det4 [mm]", ylabel="det4 local X [mm]",
                   title=f"{name}\ndet4-X vs uRW-y  r={np.corrcoef(ty[ok], dx[ok])[0,1]:+.3f}")
    ax[row, 1].hist2d(tx[ok], dy[ok], bins=100, cmap="Oranges")
    ax[row, 1].set(xlabel="uRWELL track x at det4 [mm]", ylabel="det4 local Y [mm]",
                   title=f"det4-Y vs uRW-x  r={np.corrcoef(tx[ok], dy[ok])[0,1]:+.3f}")
    ax[row, 2].hist(r, bins=np.arange(0, 20, 0.2), color="tab:blue")
    ax[row, 2].axvline(np.median(r), color="tab:red", ls="--")
    ax[row, 2].set(xlabel="|residual| [mm]",
                   title=f"median {np.median(r):.2f} mm, <2 mm {np.mean(r<2):.0%}")
fig.suptitle("det4 <-> uRWELL, run 53, clean single-cluster events", y=1.0)
fig.tight_layout()
fig.savefig(BASE + "realign_clean.png", dpi=110)
print("\nwrote", BASE + "realign_clean.png")
