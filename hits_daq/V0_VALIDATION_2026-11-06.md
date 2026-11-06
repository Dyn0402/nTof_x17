# V0 validation: the emulated DREAM hit finder against production `combined_hits` (2026-11-06)

**Answer: V0 reproduces the production hits closely enough to stand in for them.** The one change needed was the gate
noise. Production's 5-sample boxcar gate is compared with a per-channel `rms_gate` measured on the pedestal run, which
is 0.68 × the per-sample noise (time-correlated noise), not the white-noise 1/√5 = 0.447 that `emulate.dream` used
until today. `emulate.GATE_K = 0.68` is now the default, and V0 accepts a value per strip.

## Method

`validate_v0.py` on the Arm D window pickles (raw ADC, pedestal, noise and the production hits of each window's
channels), det3 Run 1 (`sat_det3_p0of4`) and Run 2 (`p2_det3_p0of12`), 4000 planes each. Per channel, production's
gate σ is recovered from its own hits as `rms_gate = boxcar peak / significance` (the `significance` branch of
`WaveformAnalyzer.cpp`) on the **even** planes. Everything is scored on the **odd** planes (2000 per run). The time
residual is per strip after removing each plane's median (production times carry the ftst clock phase). The ladder
ratio is u(V0) / u(production) from the same amplitude-weighted line, |w| > 0.002.

## Result

| | Run 1 (sat_det3) | Run 2 (p2) |
|---|---|---|
| measured gate factor rms_gate/rms (median; per channel 5–95 %) | 0.680 (0.645–0.721) | 0.678 (0.642–0.714) |
| strips, both / production only / V0 only (per-channel gate) | 14467 / 6 / 196 | 14986 / 2 / 238 |
| same, old gate 0.447 | 14473 / 0 / 1414 | 14988 / 0 / 1515 |
| time residual, MAD | 2.3 ns | 1.9 ns |
| amplitude ratio, median (MAD) | 1.0004 (0.013) | 0.9998 (0.012) |
| ladder ratio V0/production, median (MAD), n planes | 1.004 (0.022), 1105 | 1.003 (0.015), 1218 |

- The ~1.5 % extra strips in V0 are all marginal: gate significance 5.06–5.66σ (10–90 %), amplitude 4–5 σ. That is the
  ±3 % within-channel scatter of the recovered rms_gate at a 5σ cut.
- The ladder ratio (the quantity every tier is built on) agrees to 0.3–0.4 %, against the 17–37 % compression the
  section is about.

## Consequence

The λ scan and the local H0/H2a numbers in `HANDOFF.md` (2026-11-05) ran V0 with the old 0.447 gate, so about 10 %
more marginal strips than production fired. Those numbers are preliminary and will be redone. Every caller (`tiers.py`,
`h2c.py`, `resim.py`) uses the default and picks up 0.68 with no change. The analysis DAG had not been submitted.
Results: `results/v0_validation_{sat,p2}.json`.
