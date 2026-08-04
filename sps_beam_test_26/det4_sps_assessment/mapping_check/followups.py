"""With the map fixed: where is det4 along the beam, where is the beam on det4,
and how efficient is det4 where the beam actually lands."""
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

l = np.arange(512) % 64
Dn = np.arange(512) // 64
isx = Dn < 4
POS = 99.84 + 49.92 * (Dn % 4) + (49.14 - 0.78 * l)


def cluster_det4(gap=3.0):
    sel = (h_amp > 60.0) & (h_t > 600.0) & (h_t < 1850.0) & clean[h_ev]
    res = {}
    for v, want in (("x", True), ("y", False)):
        k = sel & (isx[h_ch] == want)
        ev, pos, amp = h_ev[k], POS[h_ch[k]], h_amp[k]
        o = np.lexsort((pos, ev))
        ev, pos, amp = ev[o], pos[o], amp[o]
        lead = np.full(n_ev, np.nan); ncl = np.zeros(n_ev, np.int32)
        new = np.empty(len(ev), bool); new[0] = True
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


c = cluster_det4()
dx, ncx = c["x"]
dy, ncy = c["y"]
keep = clean & (ncx == 1) & (ncy == 1) & np.isfinite(dx + dy)


def resid_at_z(z):
    tx = D["fx_p"] + (D["bx_p"] - D["fx_p"]) * z / 1370.0
    ty = D["fy_p"] + (D["by_p"] - D["fy_p"]) * z / 1370.0
    ok = keep & np.isfinite(tx + ty)
    U = np.column_stack([tx[ok], ty[ok], np.ones(ok.sum())])
    V = np.column_stack([dx[ok], dy[ok]])
    A, *_ = np.linalg.lstsq(U, V, rcond=None)
    for _ in range(3):
        r = np.hypot(*(V - U @ A).T)
        g = r < np.percentile(r, 80)
        A, *_ = np.linalg.lstsq(U[g], V[g], rcond=None)
    r = np.hypot(*(V - U @ A).T)
    M = A[:2].T
    return np.median(r), np.mean(r < 1.0), np.linalg.det(M)


print("=== 1. det4 z along the beam (config has 1155 mm PLACEHOLDER) ===")
print(f"{'z [mm]':>8} {'median |res| [mm]':>18} {'within 1 mm':>12} {'det(A)':>8}")
zs = np.arange(700, 1801, 100.0)
best = None
for z in zs:
    m, f1, dt = resid_at_z(z)
    print(f"{z:8.0f} {m:18.3f} {f1:11.1%} {dt:8.4f}")
    if best is None or m < best[0]:
        best = (m, z)
zf = np.arange(best[1] - 100, best[1] + 101, 20.0)
fine = [(resid_at_z(z)[0], z) for z in zf]
zbest = min(fine)[1]
print(f"  -> best z = {zbest:.0f} mm  (median residual {min(fine)[0]:.3f} mm)")

print("\n=== 2. where the beam lands on det4, in detector-local mm ===")
for nm, v in (("local X (the striped coordinate)", dx[keep]),
              ("local Y", dy[keep])):
    p = np.percentile(v, [5, 25, 50, 75, 95])
    print(f"  {nm:34s} 5/25/50/75/95 % = "
          + " ".join(f"{q:6.1f}" for q in p))
print("  June bands nearby: 146-164 (highest charge), 178-216 ('the band to use')")

print("\n=== 3. det4 response where the beam lands ===")
tx = D["fx_p"] + (D["bx_p"] - D["fx_p"]) * zbest / 1370.0
ty = D["fy_p"] + (D["by_p"] - D["fy_p"]) * zbest / 1370.0
ok = keep & np.isfinite(tx + ty)
U = np.column_stack([tx[ok], ty[ok], np.ones(ok.sum())])
V = np.column_stack([dx[ok], dy[ok]])
A, *_ = np.linalg.lstsq(U, V, rcond=None)
for _ in range(3):
    r = np.hypot(*(V - U @ A).T)
    g = r < np.percentile(r, 80)
    A, *_ = np.linalg.lstsq(U[g], V[g], rcond=None)
# predict the det4-local impact point for EVERY clean uRWELL track
allt = clean & np.isfinite(tx + ty)
P = np.column_stack([tx[allt], ty[allt], np.ones(allt.sum())]) @ A
predX, predY = P[:, 0], P[:, 1]
fired_x = np.isfinite(dx[allt])
fired_y = np.isfinite(dy[allt])
both = fired_x & fired_y
print(f"  clean uRWELL tracks pointing at det4: {allt.sum()}")
inside = (predX > 100) & (predX < 299) & (predY > 100) & (predY < 299)
print(f"  ... of which inside the instrumented window: {inside.sum()}")
print(f"  X view fired: {fired_x[inside].mean():.1%}   "
      f"Y view fired: {fired_y[inside].mean():.1%}   both: {both[inside].mean():.1%}")

edges = np.arange(100, 300, 4.0)
ib = np.clip(((predX - 100) / 4).astype(int), 0, len(edges) - 2)
tot = np.bincount(ib[inside], minlength=len(edges) - 1)
hit = np.bincount(ib[inside & both], minlength=len(edges) - 1)
eff = np.divide(hit, tot, out=np.zeros_like(hit, float), where=tot > 20)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
axes[0].plot(zs, [resid_at_z(z)[0] for z in zs], "o-")
axes[0].axvline(zbest, color="tab:red", ls="--", label=f"best {zbest:.0f} mm")
axes[0].axvline(1155, color="0.5", ls=":", label="config placeholder 1155")
axes[0].set(xlabel="assumed det4 z [mm]", ylabel="median |residual| [mm]",
            title="1. det4 position along the beam")
axes[0].legend()
axes[1].hist2d(predX[inside], predY[inside], bins=80, cmap="viridis")
axes[1].set(xlabel="det4 local X [mm]", ylabel="det4 local Y [mm]",
            title="2. beam spot on det4 (from the uRWELL)")
for lo, hi in ((146, 164), (178, 216)):
    axes[1].axvspan(lo, hi, color="w", alpha=0.25)
axes[2].step(edges[:-1] + 2, eff, where="mid")
for lo, hi in ((146, 164), (178, 216)):
    axes[2].axvspan(lo, hi, color="tab:green", alpha=0.2)
axes[2].set(xlabel="det4 local X [mm]", ylabel="both views fired",
            title="3. det4 efficiency vs local X (green = June live bands)",
            ylim=(0, 1))
fig.tight_layout()
fig.savefig(BASE + "followups.png", dpi=110)
print("\nwrote", BASE + "followups.png")
