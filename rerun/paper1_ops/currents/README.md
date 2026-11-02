# Resistive-electrode currents on the cosmic bench (Paper I §2 design note, §4 operation; 2026-11-02)

`hv_currents.py`: the CAEN monitor of every sub-run (`hv_monitor.csv`, imon in µA, a sample every few seconds),
per chamber, against the resist HV. Logs copied from EOS `cosmic_bench/{june_tests,det_*_old}/<run>/<subrun>/` to
`$WORK_DATA/claude-jobs/17c92c46/hv/` (13 runs, Jan–June 2026). Output `results/currents.{csv,json}`,
`results/figures/currents_vs_hv.png`.

## Result

- **det3 (6-27 Saturday scan, one chamber on the bench):** I = 1.05 ± 0.05 µA × 2^((V − 490 V)/13.3 ± 0.3 V).
  The current doubles every 13.3 V while the gain doubles every 16.7 V: **I ∝ gain^1.25**, with no measurable
  constant part (b = 0.01 ± 0.04 µA). The drift field does not enter (drift 100 → 1100 V at 490 V: 1.03 → 1.11 µA).
  At the working point that is ~1 µA, about 10⁵ × the cosmic signal current (~24 Hz × ~10⁶ electrons ≈ 4 pA).
  So the steady current is a **gain-driven dark current in the amplification stage**, the same family as the
  fixed-patch noise bands (rate ∝ gain^1.4–1.6, flat in drift field: `rerun/bands/README.md`) and the discharges.
  Mechanism not identified (stated limitation).
- det3 at 490 V differs from night to night: 0.29 µA (6-23), 0.71 (6-22), 1.13–1.14 (6-27).
- Other strip chambers at ~490 V: det4 0.54, det6 0.45, det2 2.2, det7 6.0 µA (det7 is the chamber with the X
  readout fault and the most discharges).
- Prototypes: the January board with a **plain resistive layer on top of the strips** (`strip_resit_with_plein_on_top`,
  the only one with a log) drew 0.49 µA at 530 V over 14.5 h, lower than the June strip chambers at 490 V; the
  April "strip with silver paste" board drew 2.5–5.0 µA, nearly linear in V (a leakage path, R ≈ 0.1–0.2 GΩ);
  the May strip board without silver paste 0.03–0.28 µA.

## What this does not settle

- **The uniform-layer "hot spot" argument cannot be tested from the bench logs:** no uniform-layer chamber has a
  current log, and the one plain-on-strips prototype does not show a higher current. That comparison is Stephan's,
  from production tests.
- Run-config chamber names on 6-19/6-20 and 6-22 carry the known slot remap (MEMORY: overnight 6-22), so per-name
  currents from those nights (det2, and the 6-20 "det3") may belong to the other chamber. The det3 statements above
  use the 6-27 runs, where det3 was alone.
- Spike counting (`spikes_per_h`) is limited by the ~3–6 s monitor sampling; discharges are counted from the
  waveforms instead (`rerun/paper1_ops`).
