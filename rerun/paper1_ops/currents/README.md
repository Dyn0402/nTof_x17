# Resistive-electrode currents on the cosmic bench (Paper I §2 design note, §4 operation; 2026-11-02)

`hv_currents.py`: the CAEN monitor of every sub-run (`hv_monitor.csv`, imon in µA, a sample every few seconds),
per chamber, against the resist HV. Logs copied from EOS `cosmic_bench/{june_tests,det_*_old}/<run>/<subrun>/` to
`$WORK_DATA/claude-jobs/17c92c46/hv/` (13 runs, Jan–June 2026). Output `results/currents.{csv,json}`,
`results/figures/currents_vs_hv.png`.

## Scope (Dylan 2026-11-02)

Only the n_TOF batch of boards, from det2 (the first, bulked 12 June) on: runs from 6-17. Boards are identified by the
run-config **description**, not the chamber name: the names were renumbered between nights (on 6-17 `mx17_1` is the
Arnaud board = det2 and `mx17_2` the Stephan 15 June board = det3). The 6-19/6-20 weekend is excluded: its "det3"
curve is the 6-22 det2 curve point for point (same 1.9 µA floor), and that weekend has the known slot confusion.
The Jan–May prototypes remain in `currents.csv` (`use = False`) and are not quoted.

## Result

- **det3 (6-27 Saturday scan, one chamber on the bench):** I = 1.05 ± 0.05 µA × 2^((V − 490 V)/13.3 ± 0.3 V).
  The current doubles every 13.3 V while the gain doubles every 16.7 V: **I ∝ gain^1.25**, with no measurable
  constant part (b = 0.01 ± 0.04 µA). The drift field does not enter (drift 100 → 1100 V at 490 V: 1.03 → 1.11 µA).
  At the working point that is ~1 µA, about 10⁵ × the cosmic signal current (~24 Hz × ~10⁶ electrons ≈ 4 pA).
  So the steady current is a **gain-driven dark current in the amplification stage**, the same family as the
  fixed-patch noise bands (rate ∝ gain^1.4–1.6, flat in drift field: `rerun/bands/README.md`) and the discharges.
  Mechanism not identified (stated limitation).
- **Per board and night, at 490 V** (`currents.json: per_board_night_near_490V`):

| board | 6-17 | 6-22 | 6-23 | 6-24 | 6-26 | 6-27 |
|---|---|---|---|---|---|---|
| det2 | 1.03 | 2.39 (flat 1.9 µA floor below 480 V: an ohmic leak appeared) | | | | |
| det3 | 0.48 | 0.71 | 0.29 | | | 1.07–1.14 |
| det4 | | | 0.58 | 0.12 (495 V) | | |
| det6 | | | | | 0.45 (0.78 at 495 V overnight) | |
| det7 | | | | | 6.0–6.4 | |

  The same board varies ×4 from night to night (det3 0.29–1.14 µA). The lowest, 6-23, is the night the chambers
  were still below working gain (PAPER1_NUMBERS: charge rose ×4 through that night), as a gain-driven current
  predicts; the other nights are not explained individually. det7, the chamber with the X readout fault and the most discharges, draws ~10× the others.

## What this does not settle

- **The uniform-layer "hot spot" argument cannot be tested from the bench logs** (no uniform-layer board in the
  n_TOF batch); that comparison is Stephan's, from production tests.
- The board-from-description mapping is the run configs' own record; it is consistent across 6-17 → 6-27 except
  the excluded weekend.
- Spike counting (`spikes_per_h`) is limited by the ~3–6 s monitor sampling; discharges are counted from the
  waveforms instead (`rerun/paper1_ops`).
