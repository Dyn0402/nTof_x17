"""What the channel-order reversal does to the channel-space topology cuts.

spark_check_tilt_aware.py keeps an event only if  nx <= 1  and  y_span <= 30,
both computed on the RAW channel index.  With the measured map, physically
adjacent strips across a connector boundary are 127 channels apart, so a normal
cluster sitting on a boundary looks like two clusters 127 channels wide.
"""
import numpy as np

BASE = ("/media/dylan/data/x17/sps_run53_det4_check/flat_ArCO2iso_95-3-2__run53-56/"
        "mapping_check/")
D = np.load(BASE + "det4_run_53_v2.npz")
h_ev, h_ch = D["h_ev"], D["h_ch"]
n_ev = len(D["fx_p"])
POS = 99.84 + 49.92 * (h_ch // 64 % 4) + (49.14 - 0.78 * (h_ch % 64))
isx = (h_ch // 64) < 4
PITCH = 0.78


def ncl_and_span(mask, key):
    """cluster count and span per event, for the hits in `mask`, ordered by `key`"""
    ev, k = h_ev[mask], key[mask]
    o = np.lexsort((k, ev))
    ev, k = ev[o], k[o]
    gap = 2.0 if key is not POS else 2.0 * PITCH   # >2 channels / >2 pitches apart
    new = np.empty(len(ev), bool)
    new[0] = True
    new[1:] = (ev[1:] != ev[:-1]) | (np.diff(k) > gap)
    ncl = np.zeros(n_ev, np.int32)
    np.add.at(ncl, ev[new], 1)
    lo = np.full(n_ev, np.inf)
    hi = np.full(n_ev, -np.inf)
    np.minimum.at(lo, ev, k)
    np.maximum.at(hi, ev, k)
    span = np.where(np.isfinite(lo) & np.isfinite(hi), hi - lo, 0.0)
    return ncl, span


nx_ch, _ = ncl_and_span(isx, h_ch.astype(float))
nx_mm, _ = ncl_and_span(isx, POS)
_, span_ch = ncl_and_span(~isx, h_ch.astype(float))
_, span_mm = ncl_and_span(~isx, POS)

have = np.zeros(n_ev, bool)
have[h_ev] = True
Y_SPAN_MAX, Y_SPAN_MM = 30.0, 30 * PITCH
pass_ch = have & (nx_ch <= 1) & (span_ch <= Y_SPAN_MAX)
pass_mm = have & (nx_mm <= 1) & (span_mm <= Y_SPAN_MM)

print(f"det4 events with hits: {have.sum()}")
print(f"\npassing the channel-space cut (what the script does): {pass_ch.sum():8d}"
      f"  ({pass_ch.sum()/have.sum():.1%})")
print(f"passing the same cut in mm (what it should be):       {pass_mm.sum():8d}"
      f"  ({pass_mm.sum()/have.sum():.1%})")
lost = have & pass_mm & ~pass_ch
print(f"\ngood events thrown away by the channel-space cut: {lost.sum():8d}"
      f"  = {lost.sum()/max(pass_mm.sum(),1):.1%} of those that should pass")
print(f"   nx > 1 in channels but == 1 in mm: "
      f"{(lost & (nx_ch > 1) & (nx_mm == 1)).sum()}")
print(f"   y span > 30 ch but <= 23 mm:       "
      f"{(lost & (span_ch > Y_SPAN_MAX) & (span_mm <= Y_SPAN_MM)).sum()}")
print(f"bad events kept by the channel-space cut:         "
      f"{(have & pass_ch & ~pass_mm).sum():8d}")
