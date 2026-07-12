# TRACK_PLAN_07 — absolute timing and neutron time-of-flight per track

**Attach a neutron energy to every reconstructed track. The chain: PS pulse →
gamma flash (t=0 of the pulse) → DREAM event timestamps → in-window hit time
→ track t0 → ToF over 19.5 m → E_n. Bench PLAN_42 established the detector
end of this: hit `time` is an absolute, ftst-phase-corrected 30 %-CF time
with σ_t ≈ 33 ns per detector. Deliverable: `E_n` (+ uncertainty) columns on
the PLAN_03/04 track tables.**

## The time chain

1. **Pulse anchor.** Identify the gamma-flash event of each beam pulse
   (PLAN_01 taxonomy; rule from `ntof_may_analysis/
   dream_timber_time_sync_flash.py`). The flash is prompt photons: its
   arrival at EAR2 ≈ t_pulse + L/c (19.5 m → 65 ns). Within the flash event,
   define t_flash from the earliest reliable hit time — but the flash
   saturates the detector, so calibrate a robust flash-time estimator
   (median of the first unsaturated CF times, or the CM-step onset from
   waveforms) and measure its jitter on many pulses.
2. **Event-to-event bridging.** Later DREAM events in the same pulse are
   separate triggers: their offset from the flash event is
   Δ(trigger_timestamp_ns) + (in-window time − in-window time reference).
   VERIFY `trigger_timestamp_ns` granularity and monotonic behaviour first
   (PLAN_01 gotcha); bench used it only at seconds scale — its ns-level
   fidelity at beam must be demonstrated (histogram Δtimestamp between the
   flash and the NEXT event across many pulses; compare against the PS RF
   structure if visible). If it is coarse, the fallback is TIMBER matching
   per event (the May machinery matches flash events to PS pulses to <600 µs;
   in-pulse relative timing then rests entirely on trigger_timestamp_ns —
   this is THE risk item of this plan; flag its resolution as the first
   milestone).
3. **Track t0.** From PLAN_03: t0 = earliest-strip absolute time of the pair
   = mesh-arrival time of the closest ionization ≈ particle crossing time.
   Corrections: inter-plane walk (PLAN_06), and the geometry term (track
   crossing point vs strip — bench budget 17 ns). Detector σ_t: remeasure in
   situ via inter-plane Δt0/√2 (works with zero external reference; bench
   33 ns, walk-corrected 29 ns; possibly better at 20 ns sampling).
4. **Neutron energy.** The neutron that induced the event left the target at
   the reaction time t_r = t0 (charged-particle flight inside the setup
   <1 ns). ToF = t_r − t_pulse − (flash-lag corrections); E_n from the
   relativistic inversion already implemented in
   `neutron_energy_vs_flight_time.py` (19.5 m EAR2 path). Campaign band:
   0.2–2 MeV ↔ ~1.0–3.2 µs after flash.

## Resolution budget (compute, don't assume)

σ_ToF² = σ_det² (33–39 ns) ⊕ σ_flash-estimator² ⊕ σ_bridge²
(trigger_timestamp granularity) ⊕ σ_geom² (~17 ns). At ToF = 1–3 µs, a 50 ns
total gives ΔE/E ≈ 2·ΔT/T ≈ 3–10 % — fine for the 0.2–2 MeV band; the
budget is dominated by whichever bridging term step 2 reveals. Produce the
budget table per run era and attach σ_E per track.

## Deliverables

- `flash_table.parquet`: per pulse — flash eventId, t_flash, PS/TIMBER match
  (reuse `dream_timber_time_sync_flash.py` wholesale), pulse intensity.
- `E_n`, `sigma_En`, `t_since_flash_ns` columns on tracks/pairs tables.
- One validation figure per run: hit-rate and track-rate vs t_since_flash —
  the mid-window turn-off (July QA window 1.15–3.5 µs) and the DREAM
  post-flash recovery must be visible and stable; overlay the E_n bands.

## Acceptance

- t_since_flash of thermal/epithermal features lines up with the known EAR2
  ToF spectrum shape (compare with the May zero_suppress/beam QA windows).
- Inter-plane Δt0/√2 per detector ≈ 30–40 ns; absolute per-pulse closure:
  two tracks in the SAME window from the same reaction (pairs!) must agree
  in t0 within the detector budget — pairs are their own timing validation.
- E_n distribution stable across subruns at fixed conditions.

## Gotchas

- The flash blinds the detector: candidate windows shortly after the flash
  have depressed efficiency (May: DREAM saturation recovery; July analysis
  dirs `flash_recovery*` study exactly this — read their findings before
  interpreting early-window rates).
- The 65 ns photon flight and any fixed electronics latency are common-mode
  offsets — they calibrate out against the known gamma-flash line, but only
  if treated consistently per era.
- Do not use `time_of_max` for timing (walk-prone); `time` (30 % CF) is the
  calibrated quantity (bench PLAN_42).
