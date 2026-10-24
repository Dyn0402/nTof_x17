# det7 X shorts and the M1 / JZ pogo field (June bench, g_det7_long)

## Verdict

**The fault is in the pogo field, but a rigidly shifted or rotated connector does not explain it.**

- **Where it is: confirmed.** The M1 gerbers give each FEU channel a position on the card's 10 × 30 pogo grid (1 mm
  pitch). Each Mec8 connector fills a 5-column block, 5 channels per row. A vertical neighbour on the grid is therefore
  **±5 channels** and a horizontal neighbour is ±1. Every measured "5 apart" group is a vertical chain of grid
  neighbours, including the cross-connector group 68/73 ↔ 186/191, where Mec8 #1 and #2 meet between rows 14 and 15.
  The shorted channels also fire when the M3 track crosses the partner strip 5 strips away (0.446, against 0.345 at
  the mirror offset; unshorted pairs give 0.226 and 0.243). So strips are joined at pad level, not by crosstalk in
  the electronics.
- **Rigid misalignment: rejected.** A shift plus rotation of the pin array against the pads has a best fit that
  reproduces 78 of 108 measured short pairs but predicts 137 that are not there (F1 0.48). Moving the measured pattern
  to other places on the grid (mirrored, shifted by 15 rows) fits just as well (F1 0.35–0.44, against 0.42 for the
  real pattern on the same coarse grid). The fit is only using the general tendency of pins to bridge vertically, not
  this particular pattern. A uniform offset bridges every pin in a region. The data instead bridge pin by pin: one
  column chains and the next does not, and col 2 breaks between rows 2 and 3.
- **Noisy channels (rms 60–77) are not a pogo-geometry effect.** They are ~31 % of both the shorted and the unshorted
  channels and spread over all columns. Their rms level is set per connector (67 on conn 1, 73–76 on conns 2–4), which
  points at chip/cable/M1-level pickup ("the new M1 cards are noisy", logbook 07-03), not at pad contact.

## Measured vs model

![pogo field](figures/pogo_field_measured_vs_fit.png)

Left: links between grid neighbours with amplitude correlation > 0.9. On the clean card (connectors 5–8) no
neighbour pair exceeds 0.62. Right: the best rigid fit (green = reproduced, orange = predicted only, blue = missed).

| | value |
|---|---|
| measured neighbour short pairs (corr > 0.9), card 1 | 108 of 854 neighbour pairs (58 of 226 vertical pairs) |
| best rigid fit (carrier mode) | dx −0.55, dy +0.55 mm, θ +0.8°, pin radius 0.325 mm |
| reproduced / false / missed | 78 / 137 / 30 (F1 0.48) |
| best fit with the pins fixed to the M1 card (readout mode) | same offset, F1 0.46 |
| null: real pattern vs relocated copies (coarse grid) | 0.42 vs 0.44, 0.44, 0.42, 0.35, 0.37 |
| dead channels (6) | 34, 84, 94, 134, 194, 208: none predicted by any good fit |

The best offset puts each pin near the centre of four pads (half a pitch diagonally). Even this needs a pin contact
radius of 0.33 mm to bridge, which is large for a pogo tip. The vertical and horizontal pad gaps are 0.23 mm and the
diagonal ones 0.61 mm (`figures/pin_contact_map.png`).

**Fitted once for the whole card, not per connector.** DREAM connectors 1–4 (FEU 6 ch 0–255) are the four Mec8s of a
**single** M1 card with a single 300-pad pogo field (X1), so they share one seating. Connectors 5–8 are the other card
(clean). A per-connector fit would describe four seatings that do not exist. The local diagnostic `local_fit.py`
(the best uniform offset in each 3 × 3 window) shows the same thing at small scale. A uniform offset bridges every pin
in a window, while the measured shorts within a window are sparse, so most windows fit best with no offset at all.

## What the pattern does look like

- Mostly **vertical** bridges, i.e. along the board edge (the direction the card slides when it is tightened). Long
  chains: col 0 rows 1–8 and 13–18; cols 2–6 rows 0–5; cols 7–9 rows 25–29.
- One **dense 2-D block** at the corner of the field: cols 7–9 (outer radial edge), rows 0–8 and 11–14.
- The whole card is degraded, not just the shorted pins. Even the clean card-1 channels have rms 12–20 (card 2: 7.4)
  and fire at 0.34 when crossed (card 2: 0.52).

Pin-by-pin bridging that is densest at one corner fits a **non-rigid** contact fault better than a misaligned
connector. Candidates are pins leaning in the carrier, or conductive debris or residue across the 0.23 mm pad gaps.
Another is uneven compression from a card that is lifted or bowed at that corner, so some pins tilt or slide.
Logbook: 06-29 "two M1 cards on Detector 7, X1 and Y1, gave trouble while tightening". 06-30 "Stephan glued the X1 JZ
connector on detector 7 and seated, recovered most strips, ~10 still expected out". This bench run (06-26) is from
before the glueing. The data cannot tell these candidates apart. That is a stated limitation, and nothing here
constrains the condition of the chamber after 06-30.

**Other things that make groups 5 apart:** anything that joins grid neighbours along the edge direction. No
Mec8/cable effect can: on the Mec8 a channel's neighbours are ±1 and ±2. The M1 inner-layer routing is a possible
source too, but it would be the same on every card, and card 2 is clean.

## Inputs derived from the gerbers

- `m1_trace.py`: geometric copper tracing of the M1 card (6 layers plus plated holes; the net attributes are
  stripped near the pogo field). 300 pogo pads: 256 are wired to exactly one Mec8 signal pin each, 13 go to the Mec8
  ground pins and 31 are unconnected. Output: `m1_pogo_to_mec8.json`.
- `ro_roles.py`: the readout board (DFS3498A) has the same 4 × 300-pad fields. The 256 signal positions have
  fan-out traces, the 31 NC positions have none and the 13 GND positions go to ground, so the two boards' roles
  match. The fan-out alternates between the TrackX and TrackY routing layers in a checkerboard. That is routing, not
  view.
- `conventions.py`: Mec8 pin → DREAM channel. The data choose Mec8 k (ordered along the edge) → FEU connector k+1,
  local = 2·(signal pin index) + row. Of 48 conventions, this one puts 81 of 86 group MST edges on grid neighbours
  (worst: 24 of 86). The strip order of the readout fan-out was **not** traced. The nominal strip at each pad is taken
  from the M3-verified linear map.
- JZ / pogo-pin connector design files: **none on this machine** (searched the filesystem for pogo, JZ, z-ray and
  interposer). The pin tip size is unknown, so the contact radius was left free.

## Corrected mapping for the forward model

`det7_x_strip_map_measured.json`: for all 512 channels, the pogo position, the net (channels joined by neighbour
links with corr > 0.9), the strips each channel reads (named by the channel that nominally reads them) and the
noisy/dead/weak flags. There are 19 nets with 85 channels; nets taken at 0.7 are listed as `possible_net_0p7`.

**Can the waveform fit treat a shorted channel as the sum of its strips?** Yes, as a net, not channel by channel.
Inputs tied together at the pads share the charge, and the members of a net read identical amplitudes
(corr 0.9–1.0), so it is one measurement. Model each net as **one virtual channel**: a comb of its strips (3.9 mm
apart vertically, plus adjacent strips in the dense block), with a free gain, expected ≈ 1/n of the summed charge.
Drop the n − 1 duplicate channels from the likelihood. This keeps the position information that censoring throws
away (C20 found that censoring recovers ~40 % of the lost angles). **Untested:** the 1/n gain and whether the
partner strips' pulses arrive identically (they pass through one pad, so the timing should match). Check this on the
data before relying on it. Pickup on the noisy channels needs the per-group common mode from C20 either way. No
`wft/` code was changed.

## Files

| file | what |
|---|---|
| `gerber_flash.py` | minimal Gerber flash/draw reader with net attributes |
| `m1_trace.py` → `m1_pogo_to_mec8.json` | M1 pogo pad → Mec8 pin connectivity |
| `ro_roles.py` → `ro_pad_traces.json` | readout-board pad roles |
| `conventions.py` | Mec8 → FEU channel convention, chosen by group compactness |
| `corr_matrix.py` → `corr_feu6.npz` | FEU 6 amplitude correlation (census definition, 62.5 k events) |
| `fire_offset.py` → `fire_offset.npz` | firing vs M3 track offset (the strip-level short test) |
| `model.py` | pin/pad contact model (shift, rotation, pin radius; readout or carrier mode) |
| `fit_scan.py` → `scan_readout.npz`, `refine.py` → `refine.json` | rigid fit (coarse scan + refinement, both modes) |
| `null_test.py` → `null_test.json` | rigid fit on relocated copies of the pattern |
| `local_fit.py` → `local_fit.json` | best uniform offset per 3 × 3 window |
| `make_mapping.py` → `det7_x_strip_map_measured.json` | measured nets and strips per channel |
| `make_figures.py` → `figures/` | figures |

Hits are used only for QA here (which channels move together and fire). No position, angle or depth was
reconstructed from hit times.
