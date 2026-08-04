"""Fingerprint test, final: score the discrete, physically possible cablings.

The four cabled connectors must be four of the board's eight 49.92 mm connector
blocks, so there are only  (5 windows) x (block order) x (channel order) = 20
candidate maps.  Score each against the June stripe fingerprint.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = ("/media/dylan/data/x17/sps_run53_det4_check/flat_ArCO2iso_95-3-2__run53-56/"
        "mapping_check/")
F = np.load(BASE + "fingerprint.npz")
eff, w_near = F["eff"], F["w_near"]

J = np.load("/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
            "det4_sps_assessment/stripes_g_det4.npz")
jx, jmed, jlive = J["c"], J["med"], J["live"]
jlogq = np.log10(np.maximum(jmed, 1.0))

l = (np.arange(256) % 64).astype(float)
Dn = (np.arange(256) // 64).astype(float)
illum = w_near[:256] > 0.05 * w_near[:256].max()
e = eff[:256]
print(f"scoring {illum.sum()} illuminated X-view channels "
      f"(per Dream: {[int(illum[i*64:(i+1)*64].sum()) for i in range(4)]})")

CONN0, PITCH = 49.92, 0.78
rows = []
for first in range(1, 6):                     # detector connectors first..first+3
    for border in ("forward", "reversed"):
        for chorder in ("forward", "reversed"):
            conn = first + (Dn if border == "forward" else 3 - Dn)
            base = (conn - 1) * CONN0
            X = base + (PITCH * l if chorder == "forward" else 49.14 - PITCH * l)
            q = np.interp(X[illum], jx, jlogq)
            c = np.corrcoef(e[illum], q)[0, 1]
            lv = np.interp(X[illum], jx, jlive.astype(float)) > 0.5
            auc = (e[illum][lv][:, None] > e[illum][~lv][None, :]).mean() \
                if lv.any() and (~lv).any() else np.nan
            rows.append((c, auc, first, border, chorder, X))

rows.sort(key=lambda t: -t[0])
print(f"\n{'conns':>7} {'block order':>12} {'ch order':>9} {'corr':>7} {'AUC':>6}"
      f"   {'uRW-consistent?':>15}")
for c, auc, first, border, chorder, X in rows:
    ok = "YES" if (border, chorder) in (("reversed", "forward"),
                                        ("forward", "reversed")) else "no (excluded)"
    print(f"{first}-{first+3:<5} {border:>12} {chorder:>9} {c:7.3f} {auc:6.3f}"
          f"   {ok:>15}")

# --- figure ----------------------------------------------------------------
allowed = [r for r in rows if (r[3], r[4]) in (("reversed", "forward"),
                                               ("forward", "reversed"))]
best = allowed[0]
runner = allowed[1]
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
for ax, r, tag in zip(axes[1:], (best, runner), ("BEST", "runner-up")):
    c, auc, first, border, chorder, X = r
    ax.semilogy(jx, np.maximum(jmed, 1), color="0.6", lw=1,
                label="June bench: median charge vs local X")
    for lo, hi in J["bands"]:
        ax.axvspan(lo, hi, color="tab:green", alpha=0.15)
    ax2 = ax.twinx()
    ax2.plot(X[illum], e[illum], ".", ms=4, color="tab:red",
             label="run 53: beam-normalised hit rate")
    ax2.set_ylabel("run 53 rate", color="tab:red")
    ax.set_ylabel("June charge [ADC]")
    ax.set_title(f"{tag}: connectors {first}-{first+3}, blocks {border}, "
                 f"channels {chorder}   (corr {c:.3f}, AUC {auc:.3f})")
    ax.set_xlim(0, 400)
axes[0].semilogy(jx, np.maximum(jmed, 1), "k-", lw=1)
for lo, hi in J["bands"]:
    axes[0].axvspan(lo, hi, color="tab:green", alpha=0.2)
axes[0].set_title("June det4 stripe fingerprint (green = live bands)")
axes[0].set_ylabel("median charge [ADC]")
axes[-1].set_xlabel("detector-local X [mm]")
fig.tight_layout()
fig.savefig(BASE + "fingerprint_match.png", dpi=110)
print("\nwrote", BASE + "fingerprint_match.png")
np.savez(BASE + "fingerprint3.npz", X_best=best[5], first=best[2],
         border=best[3], chorder=best[4], eff=e, illum=illum)
