# The det4 ↔ uRWELL alignment was a strip map, not a cable

**2026-08-01, run 53 (flat mount) and run 57 (rotated).** The first
det4 ↔ uRWELL alignment came out at 1.36 mm rms with the matched hits piled into
four blobs, which reads like a mismapping — and it is one, but not the one that
was suspected. **No cable is swapped.** The cabling recorded in banco's
`run_config_beam.py` is correct in every respect, including the `'inverted'`
plug orientation that was flagged there as unconfirmed. What was wrong is the
ad-hoc strip map inside `~/dylan/det4_urw_align.py`, which ignored that
orientation and numbered each connector forwards.

Fixing it takes the alignment from **24.5 mm to 0.46 mm median residual**, with
83 % of tracks inside 1 mm and a transform that is orthonormal to 3 parts in
1000. Measured map: `det4_sps_map.py`.

| | old map | measured map |
|---|---|---|
| median \|residual\| | 24.49 mm | **0.46 mm** |
| within 1 mm | 0.1 % | **82.6 %** |
| core (best 80 %) rms x / y | 8.0 / 21.4 mm | **0.34 / 0.30 mm** |
| det(A) | +1.165 | **+1.0034** |
| fitted roll | +89.4° | **+90.20°** |

n = 44 198 clean single-cluster coincidences, run 53, uRWELL track extrapolated
to z = 1120 mm.

---

## 1. What the map actually is

Each Dream connector is plugged **inverted**: FEU channel 0 is the connector's
*last* strip. The connectors themselves are in the recorded order.

```
FEU 3 channel c   ->  Dream d = c // 64,  local l = c % 64
view   = 'x' for d < 4, 'y' for d >= 4
conn   = 3 + (d % 4)                        # detector-side connector x_3..x_6 / y_3..y_6
pos_mm = (conn - 1) * 49.92 + (63 - l) * 0.78
```

Instrumented window 99.84–298.74 mm in both views, which is exactly what the
config comment claims. The old map had the same window and the same connector
order — it differed **only** in the direction of the 64 channels inside each
connector. That is a ±49 mm sawtooth: enough to scatter one beam spot into four,
not enough to look like nonsense.

**The trap that survives the fix.** Because the plugs are inverted, physically
adjacent strips across a connector boundary are **127 FEU channels apart**
(discontinuities at channels 63, 127, 191 and their Y-view counterparts). Any
clustering, span or multiplicity computed on the raw channel index splits a
cluster that straddles a boundary into two clusters 127 channels wide. Cluster
in millimetres. `det4_sps_map.cluster_positions()` does.

## 2. How it was measured — no det4 map assumed anywhere

**(a) Per-channel response.** For every one of the 512 FEU-3 channels, take the
uRWELL-front cluster position of the events in which that channel fired, in
time (the det4 drift window is 600–1850 ns, read off the hit-time spectrum) and
above 60 ADC, and subtract the accidental floor. A channel that measures a
coordinate parallel to uRW-x ramps in uRW-x and is flat in uRW-y. Result:

- Dreams 0–3 track **uRW-y**, Dreams 4–7 track **uRW-x**. The view assignment is
  right; X and Y are not interchanged.
- Inside a connector, position rises with local channel at **+0.7526 mm/ch**
  (0.78 mm pitch, diluted ~3 % by the beam profile).
- Between connectors, it *falls* by **48.85 mm per Dream index**.

Those two disagree in sign, which no forward map can produce. That alone proves
a mismapping and narrows it to two mirror-image possibilities: blocks reversed
with channels forward, or blocks forward with channels reversed.

**(b) The dead stripes break the mirror.** det4 amplifies over 38 % of its area
in ~12 irregularly spaced bands (`DET4_SPS_ASSESSMENT.md` §2), so its charge
profile is a fingerprint. There are only 20 physically possible maps — five
placements of four consecutive connectors, times block order, times channel
order. Scoring the run-53 beam-normalised hit rate against the June median-charge
profile:

| connectors | block order | channel order | corr | AUC |
|---|---|---|---|---|
| **3–6** | **forward** | **reversed** | **0.675** | **0.801** |
| 2–5 | forward | reversed | 0.554 | 0.839 |
| 3–6 | reversed | forward | 0.220 | 0.638 |
| 3–6 | forward | forward *(the old map)* | 0.002 | 0.499 |

The winner is the top of all 20, and it is the cabling as recorded. Its two
run-53 rate peaks land on the two strongest June bands — X 146–164 (the
chamber's highest charge) and X 178–216 ("the band to use") — with the dead
notch at 165–177 reading zero. Figure: `fingerprint_match.png`.

**(c) The alignment confirms it.** With the measured map the fitted transform is
`[[-0.004, -1.003], [+1.000, -0.004]]`, i.e. a proper rotation of +90.2° with
unit scale and no shear. Getting an orthonormal matrix out of a free 2×2 fit is
a real test, not a fit artefact — the old map returned det = 1.165 and 8 mm of
residual structure.

The `det4-X vs uRW-y` panel still shows **two** disjoint diagonal segments after
the fix. That is the physics: the X view only fires in the live bands. The
difference is that they are now two segments of *one* line rather than two
parallel lines 100 mm apart. Figure: `realign_clean.png`.

## 3. Things that fell out of it

**The roll is +90.2° — the chamber is mounted the way we meant to mount it,**
with the live bands horizontal (`SPS_MOUNT_2026-07-31.md` §2, the
`board_map_g_det4_rot90ccw` roll). The beam lands at detector-local X 140–235,
Y 163–229, straddling both target bands.

**The yaw on run 57 measures 25.4°.** With the map fixed, the alignment matrix
on the rotated mount is no longer orthonormal — its singular values are
1.1106 / 1.0032, and a yaw about the vertical foreshortens only the horizontal
detector axis, so yaw = arccos(1.0032/1.1106) = **25.4° ± ~0.5°**. That confirms
`DAQ_DETE_ROT_Y = 25.64` to better than a degree, from the data, and it can be
redone after any remount for the price of one run. (The same estimator on the
flat mount returns 0 ± 5° — arccos is degenerate near zero, so it can only say
"flat", not how flat.) Run 57's residual is 1.10 mm rather than 0.46: expected,
since a 25° track crosses ~14 mm of strips over the drift gap and this is a
plain centroid, not a micro-TPC fit.

**det4's z is ~1120 mm**, from scanning the assumed z and minimising the
residual. The minimum is shallow (0.459 mm at 1120, 0.471 at 1200), so this
confirms rather than replaces the `DET_Z_MM['mx17_E'] = 1155` placeholder —
it is right to within the method's ~100 mm resolution. Do not treat 1120 as a
survey.

**The HV scan results stand.** `eff_scan_full.py` and `spark_check_tilt_aware.py`
work entirely in channel space, so they are exposed to the boundary trap of §1 —
but only just. Rerunning their `nx <= 1 and y_span <= 30` cut in millimetres on
run 53 changes the accepted sample by **0.6 %** (4 802 good events wrongly cut,
5 316 bad ones wrongly kept, out of 870 k). Small because clusters are a few
strips wide and only rarely sit on one of the three boundaries. No efficiency or
spark number needs redoing; worth fixing before any tighter topology cut.

## 4. One real bug, in shared code

`common/Mx17StripMap.py`'s `Detector.map_hit()` looks the strip position up with
the **FEU** connector index instead of the detector-side one:

```python
pos = self.strip_map.lookup(axis, feu_connector, oriented_channel)
#                                 ^^^^^^^^^^^^^^ should be the number in det_key
```

Every bench MX17 has all 16 connectors cabled 1:1, so `det_key` number and
`feu_connector` are equal and the bug is invisible. mx17_E is the first partial
cabling — `'x_3': (3, 1)` — and there it puts the X view **99.84 mm too low** and
the Y view **99.84 mm too high**:

| FEU ch | `map_hit` | measured |
|---|---|---|
| 0 | x = 49.14 | x = 148.98 |
| 63 | x = 0.00 | x = 99.84 |
| 256 | y = 248.82 | y = 148.98 |
| 319 | y = 199.68 | y = 99.84 |

It gets the inverted channel order right; only the block placement is wrong. An
alignment fit absorbs a constant offset, so this would not have shown up as a
bad residual — it would have quietly reported det4-local coordinates 100 mm out,
which is exactly the number that decides whether the beam is on the 178–216 band.

**Not applied** — it is a one-line change in a library every June and n_TOF
analysis imports, and it is a no-op for all of them, but that is the user's call.

## 5. Files

| file | what |
|---|---|
| `det4_sps_map.py` | the measured map, importable, with a boundary-safe clusterer |
| `mapping_check/chanmap2.py` | per-channel uRWELL response — the measurement of §2a |
| `mapping_check/fingerprint3.py` | the 20-candidate fingerprint scan of §2b |
| `mapping_check/realign2.py` | old vs measured alignment, §1 table |
| `mapping_check/run57.py` | rotated mount, yaw measurement |
| `mapping_check/topo_impact.py` | the 0.6 % channel-space cut check |
| `mapping_check/extract2.py`, `extract57.py` | run on banco; write the npz the rest consume |

Data and figures: `/media/dylan/data/x17/sps_run53_det4_check/flat_ArCO2iso_95-3-2__run53-56/mapping_check/`.
