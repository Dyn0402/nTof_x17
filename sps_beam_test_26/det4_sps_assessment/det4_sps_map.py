#!/usr/bin/env python3
"""det4 ("Detector E", mx17_E) FEU-3 channel -> detector-local position at H4.

Measured from run 53 against the uRWELL reference, 2026-08-01; see
DET4_URW_MAPPING_2026-08-01.md.  The cabling recorded in banco's
`run_config_beam.py` is correct as written -- detector connectors x_3..x_6 on
Dream 0..3 and y_3..y_6 on Dream 4..7, every plug **inverted** -- and this
module is just that record turned into an array.

The thing to not get wrong: 'inverted' reverses the channel order INSIDE each
64-channel connector, so FEU channel 0 is the connector's LAST strip.  Physically
adjacent strips across a connector boundary are 127 FEU channels apart.  Any
clustering or span cut done on the raw channel index is therefore wrong at the
three internal boundaries -- cluster in position, not in channel number.

    from det4_sps_map import POSITION_MM, VIEW
    x_mm = POSITION_MM[channel]          # nan-free, 0..511
    is_x = VIEW[channel] == 'x'
"""
import numpy as np

CHANNELS = 512
CH_PER_CONN = 64
PITCH_MM = 0.78
CONN_PITCH_MM = 49.92            # connector k starts at (k-1) * 49.92 mm
FIRST_CONNECTOR = 3              # x_3 / y_3 are the first cabled ones
N_CABLED = 4                     # x_3..x_6, y_3..y_6

_dream = np.arange(CHANNELS) // CH_PER_CONN          # 0..7
_local = np.arange(CHANNELS) % CH_PER_CONN           # 0..63

#: 'x' for FEU channels 0-255, 'y' for 256-511 -- the coordinate the strip MEASURES
VIEW = np.where(_dream < N_CABLED, "x", "y")

#: detector-side connector (1-8) each FEU channel belongs to
CONNECTOR = FIRST_CONNECTOR + (_dream % N_CABLED)

#: strip index inside that connector, after undoing the inverted plug
STRIP_IN_CONNECTOR = (CH_PER_CONN - 1) - _local

#: detector-local position [mm] of each FEU channel, in its own view
POSITION_MM = (CONNECTOR - 1) * CONN_PITCH_MM + STRIP_IN_CONNECTOR * PITCH_MM

#: the instrumented window, in both views
WINDOW_MM = (POSITION_MM.min(), POSITION_MM.max())

# Alignment to the uRWELL reference, run 53 (flat mount), z_det4 = 1120 mm:
#   (X, Y)_det4 = A @ (x, y)_uRWELL_track + t
# A came out orthonormal (det +1.0034, no shear), i.e. det4 is rolled +90.2 deg
# with its dead/live bands horizontal, exactly as it was mounted.
RUN53_ROLL_DEG = 90.20
RUN53_RESIDUAL_MEDIAN_MM = 0.46
RUN57_YAW_DEG = 25.4             # rotated mount; config's DAQ_DETE_ROT_Y was 25.64


def cluster_positions(channel, amplitude, gap_mm=3.0):
    """Amplitude-weighted cluster positions for one event, one view.

    Clusters in POSITION, which is the whole point -- doing it on `channel`
    splits every cluster that straddles a connector boundary.
    """
    pos = POSITION_MM[channel]
    order = np.argsort(pos)
    pos, amp = pos[order], amplitude[order]
    if len(pos) == 0:
        return np.array([]), np.array([])
    cut = np.flatnonzero(np.diff(pos) > gap_mm) + 1
    out_p, out_q = [], []
    for p, a in zip(np.split(pos, cut), np.split(amp, cut)):
        q = a.sum()
        out_p.append((p * a).sum() / q if q > 0 else p.mean())
        out_q.append(q)
    return np.array(out_p), np.array(out_q)


if __name__ == "__main__":
    print(f"instrumented window: {WINDOW_MM[0]:.2f} - {WINDOW_MM[1]:.2f} mm "
          f"(both views)")
    for d in range(8):
        lo, hi = d * 64, d * 64 + 63
        print(f"  Dream{d} = FEU ch {lo:3d}-{hi:3d} = "
              f"{VIEW[lo]}_{CONNECTOR[lo]}:  ch{lo} -> {POSITION_MM[lo]:7.2f} mm, "
              f"ch{hi} -> {POSITION_MM[hi]:7.2f} mm")
    b = np.flatnonzero(np.abs(np.diff(POSITION_MM[:256])) > 2 * PITCH_MM)
    print(f"\nchannel-space discontinuities in the X view at ch {list(b)} "
          f"-- cluster in mm, not in channel number")
