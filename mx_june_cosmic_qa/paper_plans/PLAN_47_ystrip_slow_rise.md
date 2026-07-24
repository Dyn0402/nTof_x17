# PLAN 47 — Y-plane slow-rise response: characterization + impact on charge sharing/unsharing

**Status: OPEN (2026-07-24).** Found during the waveform-analyzer audit
(window-truncation survey, scripts `47_window_truncation_survey.py` /
`47b_pulse_shape_and_leadtrunc.py`; full context in the 7-24 session notes and
`~/CLionProjects/mm_strip_reconstruction` commit rewriting the waveform analyzer).

## The observation

Fleet-wide waveform survey (10 FEUs, spark events separated, ped-sub + CNS,
5σ candidate waveforms, ~1.2k events/file):

| FEU / plane | drift | tail>thr @ s31 | peak in last 3 samples |
|---|---|---|---|
| det3wk X (f7) | 1000 V | 9.0 % | 1.7 % |
| det3wk **Y (f8)** | 1000 V | **35.7 %** | 8.1 % |
| det2/3 6-22 X (f6) | 1000 V | 6.3 % | 1.8 % |
| det2/3 6-22 **Y (f8)** | 1000 V | **35.1 %** | 9.6 % |
| det6 X (f3) | 700 V | 20.5 % | 11.0 % |
| det6 **Y (f4)** | 700 V | **41.8 %** | 14.9 % |
| det7 X (f6) | 700 V | 22.9 % | 5.9 % |
| det7 **Y (f8)** | 700 V | **45.5 %** | 11.1 % |
| det4 X (f6) | 900 V | 7.1 % | 3.5 % |
| det4 **Y (f8)** | 900 V | **17.5 %** | 6.0 % |

On **every detector** the Y plane has 2–6× more waveforms still above threshold
at the last sample than the X plane. Inspection shows these are mostly *real*
pulses that keep **slowly rising to the end of the 1920 ns window** (rise from
quiet baseline, then a long slow climb — not clipped ordinary tails, not
baseline artifacts: only ~4 % of the tail-truncated Y population is
baseline-shift-like). The *average clean-pulse shape* is nearly identical X vs
Y (92 % of charge within peak+5 samples), so this is a distinct slow component
on top of a normal fast component, present preferentially on Y.

**Working hypothesis (Dylan):** the resistive strips run in the **Y
direction** — the slow component is RC charge spreading/evacuation along the
resistive strips showing up on the Y readout.

Consequences already known: ~8–10 % of Y hits at 1000 V have their peak in the
last 3 samples (amplitude/time-of-max are floor estimates), estimated average
charge loss to the window end ~4 % on Y vs ~1 % on X. The new
`trunc_right`/`trunc_left` hit branches (waveform analyzer rewrite, 7-24) tag
these hits.

## Follow-up A — full characterization for the resistive-detector paper

1. Two-component pulse-shape decomposition per plane (fast + slow amplitude,
   slow rise constant) vs detector (design A/B vs C/D boards), HV, drift field.
2. Slow-component amplitude vs position **along** the resistive strip (distance
   to the HV/evacuation end) — the RC hypothesis predicts a strong dependence;
   use M3-tracked events for position.
3. Correlate with the known board-C/D X-plane low charge-sharing (c1≈0.25) and
   det7's Y-plane saturation band — same RC physics family?
4. Check the July beam bench detectors (20 ns / 400-sample DAQ windows record
   the full slow component — measure its total charge and time constant there).

## Follow-up B — impact on charge sharing/unsharing → tracking

The unsharing chain (scripts 26/27/28, `CSHARE` c1/c2 constants) and the
position estimators use **per-strip amplitudes**; the micro-TPC time fits use
per-strip 30 %-rise times:

1. If the slow component's fraction varies strip-to-strip (e.g. with distance
   from the avalanche along the resistive strip), Y amplitudes are
   depth/position-dependently biased → measured c1/c2 on Y (0.43–0.52) partly
   reflect slow-component leakage, not just capacitive sharing. Re-measure Y
   sharing constants from (a) peak amplitude, (b) full integral (now unbiased
   after the analyzer rewrite), (c) fast-component-only amplitude, and compare.
2. Unsharing kernel: test whether deconvolving with a two-component response
   (instead of the static c1/c2 kernel) tightens the Y cluster width and the
   hybrid-tracker Y angle (σ68) on det3 (`sat_det3` golden chain).
3. Timing: quantify whether the slow component biases the 30 %-rise time on Y
   skirt strips (ties into the known mesh-end-skirt lateness, script 24).
4. Cheap first check: re-run script 31 micro-TPC metrics with `trunc_right` Y
   hits excluded/down-weighted and see if Y residuals/angles move.

## Provenance

Survey scripts: `47_window_truncation_survey.py` (fleet table above),
`47b_pulse_shape_and_leadtrunc.py` (average shapes, charge-loss estimate,
lead-truncated classification). Data: local `~/x17/cosmic_bench/...`
decoded_root files listed in the scripts. Window-truncation *hit-level*
diagnostic at production fields: script 43 (window ceiling only bites at drift
≤500 V scan points — confirmed at waveform level; the bench is **not**
mis-timed).
