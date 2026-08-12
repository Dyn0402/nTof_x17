# Running order for a 15-minute slot — MPGD 2026, Prague, 3 September 2026

**Speaker:** Dylan Neff, on behalf of the n_TOF / X17 collaboration.
**Slot:** 15 min + 5 min questions.
**Framing decided 2026-08-10:** a *status report with analysis ongoing*. No claim
about physics reach or yield (STATUS_PLAN.md §5 framing option (c)) — slide D9 has
been removed from the deck for that reason.

## The problem

The deck is **43 main-flow slides** (28 when this was written; the facility slide
became a three-frame build on 2026-08-10 and a **five-frame** one on 2026-08-11,
and all five are in the backup column below; the setup section then landed as a
build-up and a plan, which is most of the growth), then a `Backup` divider and
**12 backup slides** — 56 printed pages in all. Two of those backup slides are new
as of 2026-08-11: the n_TOF target's layers and its cooling, which are reference
material, not talk material. A 15-minute talk on figure-heavy detector
material runs at roughly 50 s/slide once you include the two or three slides you
will linger on, so the budget is **16–17 slides including title and summary**. The
Status section was deliberately built long to cut from; this is the cut.

## Proposed order — 16 slides, ~14 min

| # | Slide | Budget | Why it survives |
|---|---|---|---|
| 1 | Title | 20 s | — |
| 2 | **A possible new boson at 17 MeV** *(merge of slides 3 + 4)* | 60 s | One motivation slide. Keep the ATOMKI angular-correlation figure, drop the three bullets to one line. The ⁴He panel is already called out as the mirror of our channel. |
| 3 | **n + ³He → the n_TOF search channel** | 60 s | Carries the physics *and* the facility. Fold "EAR2: 20 m vertical, highest instantaneous flux" into the caption. |
| — | *(optional)* **What the detector has to separate** — the ideal pair kinematics | 50 s | Added 2026-08-10. Not counted in the 16. Promote it here if you want the "why a tracker, not two counters" argument made explicitly; the right-hand panel makes it in one glance. Demote to backup if you are over time. |
| 4 | Chamber design | 60 s | The MPGD audience came for this. Needs the zoomed board-peel figure. |
| 5 | Micro-TPC operation | 50 s | How one gap gives a 3-D segment. |
| 6 | Resistive strips share charge — the fit has to know it | 70 s | **Load-bearing for this audience.** The one genuinely MPGD-methodological result in the talk. Linger here. |
| 7 | A cosmic muon, reconstructed | 40 s | The money shot. Let it sit on screen while you talk. |
| 8 | Efficiency: 93% on the best chamber | 60 s | Refreshed 2026-08-10. |
| 9 | Sub-degree angle, sub-mm position | 60 s | Refreshed; the deconvolved tile pre-empts the telescope question. |
| 10 | **The setup, as one build** *(collapse of the 3-D sequence)* | 80 s | **This section grew after this document was first written** — the two old setup slides were replaced by an **eight-frame** in-house Geant4 build sequence (`setup3d_1..8`, deck slides **16–23**), which is much better material but many times the budget. At 15 min, show **frame 3 (…and a pair leaves, slide 18)** and **frame 8 (the full setup, slide 23)** and narrate the rest over them. The sequence's own header comment nominates frames 4 and 2 as the first to drop; frames 5–7 (the layer-by-layer build) are the next. Everything not shown goes to backup — there is already a "setup to scale" backup slide for it, and the plan slide (25) now carries the distances on the page. |
| 11 | Where things stand (D0) | 50 s | The section's thesis. |
| 12 | What the γ flash does to a DREAM channel (D1) | 60 s | Sets up everything after it. |
| 13 | **The chamber is fine. The front end is not.** (D1b) | 70 s | The slide that makes this section an MPGD talk rather than a DAQ post-mortem. Linger. |
| 14 | **Dead time is set by charge, not by voltage** (D5) | 70 s | The one new result. It subsumes D2/D3/D4 — quote "~10² nC per pulse per chamber" in passing and let this slide carry it. |
| 15 | We chose the operating point off the dead-time map (D6) | 60 s | What we did about it. Ends the section on agency, not on a complaint. |
| 16 | **Summary** *(merge of D10 + Summary)* | 60 s | One closing slide. Keep the σ(p)/p ≲ 30 % requirement line and the "next" line from D10. |

Total ≈ 13 min 40 s of slides. That leaves headroom, which you will need.

## What moves to backup (not deleted — you will be asked about most of it)

| Slide | Where it goes |
|---|---|
| Outline | **Cut.** At 15 min an outline slide costs 15 s and tells the audience nothing they cannot infer. First thing to cut if you are over. |
| n_TOF facility (**now slides 5–9**, a five-frame build) | Backup, **all five as one unit** — and there is now a **sixth, already-in-backup slide** carrying the same figure with its full caption (the provenance and the drawn-not-measured list, moved off the main flow on 2026-08-11 so the pictures could use the full slide height). Its content compresses to one caption line on slide 3. On 2026-08-10 the single facility slide became a build-up of one in-house render; on 2026-08-11 the CERN photograph of the hall was set to the **left** of the render on every frame, and the build was cut finer to five: target → the neutrons leaving at 90° → the middle of the line (collimator, floor, shielding, the pipe ending) → back into a pipe on up to the dump → the measuring station. The deck has no fragment mechanism, so a build is consecutive slides. **This grows the printed deck from 28 to 32 main-flow slides but does not change the 16-slide cut above** — the whole unit was already demoted, and it is still demoted. If you find 60 s for the facility after all, **show frame 5 only** (`ear2_onfig_5_station.png`) and narrate the build over it; the frames are subsets of one picture, so nothing is lost but the reveal. If someone asks where a number or a drawn detail comes from, **the answer is the backup slide**, not these. And if the question is about the **spallation target** specifically &mdash; what it is made of, how it is cooled, why the beam hits it at 10&deg; &mdash; the answer is the two target backup slides (**46 and 47**), which carry the full geometry and the whole nitrogen circuit from the design paper. |
| Characterized on the Saclay cosmic bench (slide 11) | **Depends on Alexandra's P2 talk — flag left in place.** If she covers the bench, cut to a one-line back-reference on slide 8. If not, it is the first slide to *restore* if you find time. |
| The target, standalone (slide 14) | Backup, as the capsule-geometry detail slide. |
| Trigger and calorimetry, 3-panel (slide 16) | Backup. |
| How much charge (D2), Charge vs gain (D3), Dead time vs gain (D4) | Backup. All three are the abscissa and the ladder that D5 joins; D5 states the conclusion. **D4 was on the original "load-bearing if time is tight" list** — if you would rather show the dead-time-vs-HV map than the charge join, swap 14 for D4, but not both. |
| What we record, and when (D7) | Backup. |
| The tracks point back at the capsule (D8) | **Judgement call.** It was on the load-bearing list, and "reconstruction transfers from bench to beam" is a real result. But it is PRELIMINARY on one arm / one sub-run / a transferred calibration whose own document says nothing is quotable. At 15 min with a caveat that long, I would keep it in backup and mention the result in one sentence on slide 11. Restore it to main flow only if the in-situ calibration lands before 3 September. |

## One thing to review before trusting this

The 3-D setup sequence is **nine frames on slides 16–24**, plus a backup slide,
and since 2026-08-11 a tenth **plan** slide (25) that redraws the same setup and
the same event orthographically, at 1:1, with the standoff and the layer radii
dimensioned. It repeats no content, so at 15 min it is the cleanest thing in the
section to drop; it is also the slide to keep if the audience is the kind that
asks "how far apart, exactly?", since it answers that off the page.
It was finished and documented on 2026-08-10 (this paragraph updated then): the
geometry is imported at run time from the simulation's own
`plot_geometry.py`/`SimConfig.hh`, the neutron and the pair are real Geant4
events, and slide 18 says in its caption that the pair is one real event while
the neutron that made it is drawn rather than simulated alongside it. Provenance
and the two drawn-but-not-measured items are in `mpgd26/README.md` ("The n_TOF
setup, built up"), `mpgd26/report.html` and `NOTES.md`; regeneration is one
command, `make_ntof.py`. All ten were re-run on 2026-08-11 when the sim's
Micromegas active area was remeasured (38 × 34 → 39.9 × 36.0 cm).

**Slides 26–27 are photographs of the real station, and they are placeholders**
(added 2026-08-11): top-down into the assembly, and one arm from outside. They
are not in the 15-minute cut and not yet worked into the argument — see the
comment above them in `index.html`.

The `~10⁻⁸` radiative-branch ratio that appeared in two of those captions has
been removed from the main flow per the no-yield decision and now appears once,
in backup, with its reference. **Still read the eight slides before the talk** —
the wording is first-draft.

## Deviations from the original plan worth knowing

The deck's own note suggested a minimum set of **D1, D4, D6, D8, D9** if time is
tight. This order keeps D1 and D6, replaces D4 with D5 (the join is the stronger
statement and implies D4), drops D9 by decision, and demotes D8 for the
preliminary-calibration reason above. It also adds **D1b**, which post-dates that
note and is the best slide in the section for this audience.

## Open dependencies

- **Alexandra's P2 talk order** — governs slide 11 (still unknown as of 2026-08-10).
- **The angular-resolution figure** is still hits-basis (1.66°) while the stat tile
  advertises the waveform-first 1.0–1.1°. Slide 9 currently states both honestly,
  but regenerating the figure on the waveform-first basis would let the tile lead
  with 1.1°. See NOTES.md.
- **Slide 9's residual figure** is from a run never reprocessed after the
  2026-07-25 fix; HANDOFF_resolution.md §5 has the fix and it improves the number.

---

*Written 2026-08-10, after Dylan confirmed the slot length and the no-yield framing.*
