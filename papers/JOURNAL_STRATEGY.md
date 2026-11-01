# Journal strategy for the two MX17 papers (decided 2026-11-01)

**Decision (Dylan, 2026-11-01):**

| paper | journal | form |
|---|---|---|
| **I. Detector & performance** (`paper1/`) | **JINST** | long, appendix-heavy: results in the body, the mechanics in appendices |
| **II. Reconstruction** (`paper2/`) | **NIM A** | short and focused: one general claim, details in short appendices or supplementary material |

Both still go to arXiv on the same day as companions with cross-referenced numbers. This replaces the
2026-10-08 plan to send both papers to the *same* journal (PAPER_PLAN.md header, C4).

## Why each paper fits its journal

- **Paper I** is a detector-description and characterisation paper: five chambers, two setups, efficiency,
  gas, timing, tracking, bow, det4 stripes. JINST has no page limit and routinely publishes long
  methods appendices. In NIM A it would be squeezed, or draw "please shorten" comments.
- **Paper II** makes one sharp, general claim. On resistive strips a strip's hit time includes delayed
  copies of its neighbours' charge, so the drift ladder is compressed by 20–30 % for every estimator. A
  waveform forward model fixes this, and the paper states what a hit readout must record. That's a
  focused methods paper with a headline, which suits NIM A.

## Career reasoning

- NIM A and JINST have similar standing and impact factors (~1–1.5). No hiring committee ranks one over the
  other, and readers find both on arXiv (physics.ins-det). The choice of journal matters less than speed
  and authorship.
- **Paper II is the reputation-building paper.** It applies to every resistive-strip µTPC that uses hit
  times, including the ATLAS NSW-style VMM readouts, and its readout-requirements section gives readout
  designers a reason to cite it. It's clearly Dylan's method. If Dylan is first author on only one paper,
  it should be this one.
- A higher-visibility option for Paper II is EPJ C, which publishes instrumentation papers. Its referees
  expect the general relevance stated up front. Consider it only if R3/R4 (the forward fit on hits, and
  the readout requirements) come out strong. The default is NIM A.
- What actually drives impact: both papers out soon, first authorship that shows the contribution, talks
  that point to them (MPGD26), and later the X17 physics paper.

## Ruled out: short NIM A plus long JINST of the same result

Publishing the same headline twice (once short, once long) reads as duplicate publication or salami-slicing.
The I/II split already does the legitimate version: the two papers make distinct claims, and each paper's
gory details go in its own appendices.

- Paper II's appendices or supplementary material: calibration protocol (R9), physics-floor toy (R6), cost table (R10).
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
- Switch the LaTeX classes: `jinst` for Paper I and `elsarticle` for Paper II. Not done yet, because `paper1/main.tex`
  and `paper2/main.tex` still use plain `article`.
