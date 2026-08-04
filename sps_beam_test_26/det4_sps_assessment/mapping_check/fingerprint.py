"""Break the mirror ambiguity with det4's own dead-stripe fingerprint.

The channel->uRW response (chanmap2) fixes the connector ORDER and the
within-connector direction only up to a reflection of the detector-local
coordinate.  det4's stripe pattern is irregular, so it is a fingerprint: map the
run-53 X-view channels onto detector-local X under each hypothesis, and see
which one lines the live channels up with the June live bands.
"""
import json

import numpy as np

BASE = ("/media/dylan/data/x17/sps_run53_det4_check/flat_ArCO2iso_95-3-2__run53-56/"
        "mapping_check/")
D = np.load(BASE + "det4_run_53_raw.npz")
fx, fy = D["fx"], D["fy"]
h_ev, h_ch, h_amp, h_t = D["h_ev"], D["h_ch"], D["h_amp"], D["h_time"]

good_ev = np.isfinite(fx) & np.isfinite(fy)
sel = good_ev[h_ev] & (h_amp > 60.0) & (h_t > 600.0) & (h_t < 1850.0)
ch, ev = h_ch[sel], h_ev[sel]
uy_hit = fy[ev]
ev_y = fy[good_ev]                     # the beam's own uRW-y profile

PITCH_URW = 0.755                      # measured mm of uRW-y per det4 channel
# --- fit the block model  uRW_y = p*l + q*D + r  on the clean X-view ramps ----
C = np.load(BASE + "chanmap2.npz")
cen_y = C["cen_y"]
clean = np.zeros(512, bool)
for lo, hi in ((74, 87), (113, 128), (171, 191)):
    clean[lo:hi] = True
clean &= np.isfinite(cen_y)
l = np.arange(512) % 64
Dn = np.arange(512) // 64
A_ = np.column_stack([l[clean], Dn[clean], np.ones(clean.sum())])
coef, *_ = np.linalg.lstsq(A_, cen_y[clean], rcond=None)
p, q, r = coef
print(f"uRW_y = {p:+.4f}*local_ch {q:+.3f}*dream {r:+.2f}   "
      f"(block step {q:+.2f} mm = {q/p:+.1f} channels)")
y_pred = p * l + q * Dn + r

# --- per-channel excess (correlated) hit count, X view only ------------------
xv = ch < 256
n_c = np.bincount(ch[xv], minlength=512).astype(float)
# a hit counts as correlated if the event's uRW-y sits near the channel's own
# predicted position; the accidental rate is measured far away and subtracted
near = np.abs(uy_hit[xv] - y_pred[ch[xv]]) < 6.0
far = np.abs(uy_hit[xv] - y_pred[ch[xv]]) > 20.0
n_near = np.bincount(ch[xv][near], minlength=512).astype(float)
n_far = np.bincount(ch[xv][far], minlength=512).astype(float)
# expected accidental fraction in the near window, from the beam profile
hb, eb = np.histogram(ev_y, bins=np.arange(-1, 130, 1.0))
cb = 0.5 * (eb[1:] + eb[:-1])
w_near = np.array([hb[np.abs(cb - y) < 6.0].sum() for y in y_pred])
w_far = np.array([hb[np.abs(cb - y) > 20.0].sum() for y in y_pred])
excess = n_near - n_far * np.divide(w_near, w_far, out=np.zeros(512), where=w_far > 0)
# efficiency-like: excess per beam particle that crossed this channel
eff = np.divide(excess, w_near / len(ev_y) * len(ev_y), out=np.zeros(512),
                where=w_near > 0)
eff = np.divide(excess, np.maximum(w_near, 1.0))

# --- June fingerprint -------------------------------------------------------
J = np.load("/home/dylan/PycharmProjects/nTof_x17/sps_beam_test_26/"
            "det4_sps_assessment/stripes_g_det4.npz")
jx, jmed, jlive = J["c"], J["med"], J["live"]

illum = (w_near > 0.02 * w_near.max()) & (np.arange(512) < 256)
print(f"X-view channels inside the beam: {illum.sum()}")

print(f"\n{'hypothesis':>34} {'best A':>8} {'corr(log q)':>12} {'AUC live':>9}")
best = None
for sign, name in ((+1, "X = A + 0.78*l - 49.92*D  (roll +90)"),
                   (-1, "X = A - 0.78*l + 49.92*D  (roll -90)")):
    for A in np.arange(0.0, 400.0, 0.5):
        X = A + sign * (0.78 * l - 49.92 * Dn)
        m = illum & (X > jx[0]) & (X < jx[-1])
        if m.sum() < 60:
            continue
        q_j = np.interp(X[m], jx, np.log10(np.maximum(jmed, 1.0)))
        v = eff[m]
        c = np.corrcoef(v, q_j)[0, 1]
        lv = np.interp(X[m], jx, jlive.astype(float)) > 0.5
        if lv.sum() < 5 or (~lv).sum() < 5:
            continue
        auc = (v[lv][:, None] > v[~lv][None, :]).mean()
        if best is None or c > best[0]:
            best = (c, name, A, auc, m.copy(), X.copy())
        if abs(A - round(A / 0.5) * 0.5) < 1e-9 and False:
            pass
    # report this hypothesis' own best
    bh = None
    for A in np.arange(0.0, 400.0, 0.5):
        X = A + sign * (0.78 * l - 49.92 * Dn)
        m = illum & (X > jx[0]) & (X < jx[-1])
        if m.sum() < 60:
            continue
        q_j = np.interp(X[m], jx, np.log10(np.maximum(jmed, 1.0)))
        c = np.corrcoef(eff[m], q_j)[0, 1]
        lv = np.interp(X[m], jx, jlive.astype(float)) > 0.5
        if lv.sum() < 5 or (~lv).sum() < 5:
            continue
        auc = (eff[m][lv][:, None] > eff[m][~lv][None, :]).mean()
        if bh is None or c > bh[0]:
            bh = (c, A, auc)
    print(f"{name:>34} {bh[1]:8.1f} {bh[0]:12.3f} {bh[2]:9.3f}")

c, name, A, auc, m, X = best
print(f"\nBEST: {name}   A = {A:.1f} mm   corr = {c:.3f}   AUC = {auc:.3f}")
np.savez(BASE + "fingerprint.npz", eff=eff, excess=excess, w_near=w_near,
         y_pred=y_pred, X_best=X, mask=m, A=A, name=name)
