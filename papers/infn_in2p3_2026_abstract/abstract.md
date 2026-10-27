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

## Draft (~270 words)

**Title:** Large-area resistive Micromegas as compact TPCs: waveform reconstruction, performance, and operation
in the n_TOF γ flash

**Session:** Gas detectors (alternative: electronics, with the flash result first)

Four 40 × 40 cm² resistive-strip bulk Micromegas, built at CEA Saclay, were operated as compact TPCs. Each one
reconstructs a three-dimensional track segment from a single 30 mm drift gap. They were built for an e⁺e⁻ pair
measurement at the CERN n_TOF EAR2 neutron facility, which ended in August 2026. A capacitively coupled pad
matrix gives two orthogonal views of 512 strips each, and DREAM front ends sample every strip every 20 ns.

On a resistive anode, part of the charge reaches the neighbouring strips through the resistive layer, delayed
and spread out by its RC time constant. We show that this biases any per-strip time estimate: threshold,
constant-fraction and matched-filter timing all compress the drift-time ladder by 20–30 %, and angles read
about 4° too steep. Instead, a forward model fits the waveforms of all strips in a cluster at once, using the
measured neighbour-sharing kernel and the single-strip impulse response, and the bias disappears.

On a cosmic bench with four reference Micromegas, the chambers reach an angular resolution of 1.15–2.51° per
view and up to 93 % efficiency. In the SPS H4 beam the spatial resolution is 176 µm (0.3 × the strip pitch),
with no dead time after discharges. Diffusion alone would allow about 0.3°; most of the difference comes from a
mechanical bow of the chambers, a constraint on flatness for large-area compact TPCs. In the n_TOF γ flash the
chambers recover within 2 µs, but the charge-sensitive front end stays saturated for 5 ms. This defines what a
flash-tolerant readout must achieve.

If it must be shorter, cut the readout/DREAM sentence first.
