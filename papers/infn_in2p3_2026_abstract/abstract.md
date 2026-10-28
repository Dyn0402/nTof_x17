# Abstract draft: 2nd INFN–IN2P3 bi-national conference on detector R&D

Drafted 2026-10-27, from the MPGD2026 proceedings abstract (`../mpgd26_proceedings/main.tex`), updated to the
numbers in `../paper1` and `rerun/model_audit/FINDINGS.md`. **Not submitted.**

## The conference

- <https://agenda.infn.it/event/52777/overview>. LNF Frascati, 2–4 Dec 2026. Chair: A. Quaranta.
- Scope (verbatim): "The conference focuses on the detector R&D activities currently underway at IN2P3 (France)
  and INFN (Italy)." Detector R&D and technology only; no physics or applications track.
- Five half-day sessions: photosensors; **gas detectors**; solid-state; cryogenic & quantum sensors;
  electronics & laboratory infrastructures. Poster session 3 Dec (A0 portrait).
- Call for abstracts closed **19 Oct 2026**. No word limit is posted. Registration is open to 15 Nov and is
  mandatory for entry to the lab.
- **Eligibility:** CEA Saclay (Irfu) is not IN2P3. Either an INFN/IN2P3 co-author (n_TOF) or a word with the
  organisers, done together with asking about a late submission.

## Changes from the MPGD abstract, and why

- "µTPC mode" → "compact TPCs" (Dylan's request).
- Physics framing cut to one line of motivation: the measurement has ended, but physics results are not ready.
- **Withdrawn:** "close to the ~1° physical limit". Model audit F3 puts the diffusion floor at 0.30–0.35°, and the
  measured resolution is 3–6× above it. F4 says most of the gap is a mechanical bow of the chambers (M3 curvature
  confirms it on 10/10 views), and that is now stated as a detector lesson.
- Angular resolution is now 1.15–2.51° per view over five chambers (paper 1). It is still `\requote` there:
  re-derive it under the unified reconstruction (ws_unify, F22) before submitting.
- The 93 % efficiency predates the unified reconstruction (gate3 moved detection +1.1…+5.5 pt): re-check.
- Unchanged: 176 µm at SPS H4 (0.30 × pitch; F14 says the 3-pitch method stands); no dead time after a spark;
  the 20–30 % ladder compression / ~4° too steep from hit times; flash recovery 2 µs (chamber) and 5 ms (DREAM).

## Draft v2 (2026-10-28, ~140 words)

**Title:** A resistive-strip Micromegas TPC for the X17 search at n_TOF: design, track reconstruction and
performance

**Session:** Gas detectors

The X17 search at the CERN n_TOF neutron facility requires tracking e⁺e⁻ pairs from neutron capture on ³He. To
test the anomaly, their opening angle must be measured with as little material as possible. We developed a
resistive-strip bulk Micromegas that works as a compact TPC: one 30 mm drift gap gives a three-dimensional track
segment. Its 40 × 40 cm² area is read out through a pad matrix with a 0.78 mm pitch, capacitively coupled to two
orthogonal strip views.

The resistive layer presents unique challenges for TPC operation. Charge spreads to the neighbouring
strips with an RC delay, and this distorts the arrival time measured on each strip: angles taken from those times
read about 4° too steep. We solve this with a novel reconstruction that fits the strip waveforms directly.

The chamber reaches an angular resolution of 1.2°, 93 % efficiency and a spatial resolution of 180 µm. At
n_TOF, the reconstructed tracks image the ³He target capsule, demonstrating the detector's efficacy as a TPC
tracker.

Sources: det3 `fe` gate (`rerun/model_audit/followups/fe_unify2/results/gate_fe.md`: s68 1.08° X / 0.97° Y vs
M3, 92.7 % detected within 5 mm, interior); 180 µm = det4 at SPS H4 (176 µm, 0.30 × pitch; det3 was not at SPS);
capsule imaging = band crossing at −31 / −35 mm (chambers A / C) vs −28 mm Geant4 (`COORDINATES.md`).
