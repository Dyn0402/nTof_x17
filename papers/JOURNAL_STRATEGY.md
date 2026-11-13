# Journal strategy for the three MX17 papers (decided 2026-11-01, split 2026-11-13)

**Decision (Dylan, 2026-11-01; Paper II split into IIa and IIb 2026-11-13, `paper_split_decision/DECISIONS_2026-11-13.md`):**

| paper | journal | form |
|---|---|---|
| **I. Detector & performance** (`paper1/`, reorganised copy `paper1_v2/`) | **JINST** | long, appendix-heavy: results in the body, the mechanics in appendices |
| **IIa. Reconstruction** (`paper2/`) | **NIM A** (`elsarticle`) | short and focused: the waveform forward model, its calibration and the fleet |
| **IIb. The single-time bias** (`tpc_mixing_review/`, "Paper III" internally) | **JINST** | a measurement (C/N^2 + b, its decomposition, the remedies) with a literature survey |

I and IIa go to arXiv on the same day as companions with cross-referenced numbers; Paper I depends only on IIa.
**IIb follows a few weeks later**, so its open items do not set the date of I + IIa. Public titles are chosen at
submission; internally the papers stay I / II / III (D10). This replaces the 2026-10-08 plan to send both papers to
the *same* journal (PAPER_PLAN.md header, C4) and the 11-09 plan to keep the review as a separate third paper.

## Why each paper fits its journal

- **Paper I** is a detector-description and characterisation paper: five chambers, two setups, efficiency,
  gas, timing, tracking, bow, det4 stripes. JINST has no page limit and routinely publishes long
  methods appendices. In NIM A it would be squeezed, or draw "please shorten" comments.
- **Paper IIa** is a focused methods paper: a physical waveform forward model (RC kernel, footprint, diffusion,
  per-chamber calibration from the data alone) reconstructs angle and depth from the samples, with one table
  comparing it with the hit chain (H0, H2c, W). It cites IIb for why hit times fail. That suits NIM A.
- **Paper IIb** makes the general claim. A single time per readout channel aggregates charge from several drift
  depths (diffusion, the prompt footprint and, on resistive strips, the RC spreading), so the drift ladder is
  compressed by 20–30 % for every estimator, as an end effect tan_rec/tan_true - 1 = C/N^2 + b. It owns the
  compression numbers (C = 7.0 / 11.7 on chamber A), the decomposition, the hit-level remedies, what a hit readout
  must record, and the survey of the published strip-µTPC biases. It is not a "delayed copies" effect: that was the
  retired copy kernel's effective fit (CLAUDE.md, `paper2/hits_daq/P0_XBIAS_2026-11-05.md`). JINST fits a
  measurement-plus-survey paper of this length; EPJ C is no longer considered.

## Career reasoning

- NIM A and JINST have similar standing and impact factors (~1–1.5). No hiring committee ranks one over the
  other, and readers find all of them on arXiv (physics.ins-det). The choice of journal matters less than speed
  and authorship.
- **Dylan is first author on IIa and IIb (D08).** The general claim now lives in IIb: it applies to every strip
  µTPC that uses hit times, including the ATLAS NSW-style VMM readouts, and its readout-requirements section gives
  readout designers a reason to cite it. IIa is the method that makes the claim measurable.
- What actually drives impact: I + IIa out soon, IIb shortly after, first authorship that shows the contribution,
  talks that point to them (MPGD26), and later the X17 physics paper.

## Ruled out: short NIM A plus long JINST of the same result

Publishing the same headline twice (once short, once long) reads as duplicate publication or salami-slicing.
The I/IIa/IIb split does the legitimate version: each paper makes a distinct claim (performance, method, bias
law), and each paper's gory details go in its own appendices. IIa shows one motivating compression figure and
cites IIb for the numbers; it does not repeat them.

- Paper IIa's appendices or supplementary material: calibration protocol (R9), cost table (R10).
- Paper IIb's appendices: the white-noise toy (R6), the coupling tests, the remedy definitions, the survey table.
- Paper I's appendices: per-chamber calibration, bow mechanics, loss categories (A1), saturation model, M3 reference
  resolution, SPS working-region caveats.

## Appendix guidelines for Paper I

- The body stands on its own. Every headline number has its figure and a one-sentence account of how it was obtained in
  the body, and the sentence points to its appendix.
- Each appendix answers one question a referee would ask ("is the bow real or a reconstruction bias?"), not
  "everything we did".
- Nothing a result depends on lives only in an appendix. Caveats that qualify an abstract number stay in the body.
- Target a body of ~12–15 pages and 4–6 short appendices. If the appendices outgrow the body, check whether the material
  belongs in Paper II.
- State at 2026-11-01: body ~2,600 words, A1 ~800. `sections/08b_bow.tex` (~1,060 words) is the first candidate to
  split: tilt and mesh depth plus one figure stay in §8, while the fit, the transfer gate and the skeptic checks move
  to an appendix.

## Still to check (C4)

- What the n_TOF / X17 collaboration expects for sign-off and journal.
- Whether CERN's open-access agreements cover IOP (JINST) and Elsevier (NIM A) for the authors' affiliations.
- Switch the LaTeX classes: `jinst` for Papers I and IIb, `elsarticle` for IIa. Check the current `\documentclass` of
    each `main.tex` (all four use plain `article` at 2026-11-13; Paper I is shared with Stephan on Overleaf).
