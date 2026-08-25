# Running order for a 15-minute slot — MPGD 2026, Prague, 3 September 2026

**Speaker:** Dylan Neff, on behalf of the n_TOF / X17 collaboration.
**Slot:** 15 min + 5 min questions.
**Framing decided 2026-08-10:** a *status report with analysis ongoing*. No claim
about physics reach or yield (STATUS_PLAN.md §5 framing option (c)) — slide D9 has
been removed from the deck for that reason.

## How the deck is numbered (2026-08-21, revised 2026-08-23)

The footer number is **not** a page number — it is the ONLY on-screen counter
now (2026-08-23: the separate JS "x / N" overlay was removed as a duplicate;
the printed footer is what the room sees, live or on paper).

* A **build** — one drawing or one argument that grows over several sections —
  costs **one** number and its frames are `N.1, N.2, …` (`slide bstart` /
  `slide bcont` + `data-frame` in `index.html`). The whole 3-D setup sequence,
  from the ³He capsule alone through to the top-down plan, is one such build:
  now **18.1–18.10** (see below — it moved).
* The **talk is 29 slides**, ending on the Summary. That slide carries
  `data-total-end`, and it is what sets the denominator (typed into the CSS as
  a literal `/ 29`, not computed — see the comment on `.slide::after`).
* **Backup keeps counting past it against the same denominator** — the first
  backup slide reads `30 / 29` and the last reads `76 / 29`. That is
  deliberate: the audience sees how far through the *talk* you are, and the
  backup numbers stay usable as addresses when someone asks for one.

**2026-08-23: four section-transition divider slides were added** ("The
Micromegas μTPC", "Characterization", "The n_TOF Search", "Status" — each a
kicker word plus a one-sentence primer, before the section they introduce).
Every slide from "The Micromegas μTPC design" onward shifted by however many
dividers now precede it (+1 through the reconstruction/characterization run,
+2 through the setup section, +3 through most of Status, +4 from the Summary
on). Slide 17 ("How we got here") also grew from one slide into a 3-frame
build, 21.1–21.3. **The row-by-row table below was NOT re-walked after either
change** — its slide numbers are very likely stale by 1-4 relative to what the
deck now shows; the footer in the live deck (or the PDF) is the authority, not
this table, until someone does that pass.

So "slide 15" below means the number in the footer *as of 2026-08-21* — not
necessarily what that content is numbered today. The printed PDF is 101 pages
for 76 numbered slides, 29 of them the talk.

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
| 2 | **A possible new boson at 17 MeV** *(**merged in the deck 2026-08-16** — was slides 3 + 4)* | 60 s | One motivation slide, and now one slide in `index.html` too: the ATOMKI angular-correlation figure at full width under three one-line bullets, the third of which is the ⁴He mirror-channel bridge. Nothing left to cut here. **Set larger on 2026-08-18** with the rest of the opening slides (`.slide.lead`, body type +25 %), and the figure column widened to match. |
| 3 | **Capture on ³He, and the three ways ⁴He\* comes down** *(build, 5.1–5.3)* | 60 s | **Replaced "n + ³He → the n_TOF search channel" on 2026-08-17** — the five-beat story moved here from the setup section and the compact one-figure slide is gone. Three frames: the beam on the vessel → capture → the level scheme and the three channels. It ends on **"Detect the e⁺e⁻ pair!"**, which is the hand-over to the detector half of the talk. |
| 3b | **The opening angle is set by the boost** *(build, 6.1–6.3)* | 50 s | The other half of the same figure: why a *mass* becomes an *angle*, and the spectrum that falls out. This is the beat the old compact figure had to assert. **6.2's spectrum is a stack since 2026-08-17** — a small X17 yield drawn on the IPC background at a declared 4 %, so it looks like a measurement rather than two normalised shapes. At 15 minutes it is the first of the two to drop — but see the row below, because dropping it changes what that slide has to do. **A third frame since 2026-08-18**: a cartoon under the spectrum showing the pair crossing two Micromegas drift volumes, because the deck otherwise cut from a kinematic spectrum straight to an exploded chamber. It is the hand-over to the detector half of the talk, and it is where you say *"so what measures an angle?"*. The opening angle on it is drawn true; nothing else on it is to scale. **Both rows were re-flowed on 2026-08-18** onto a narrower canvas so each fills the slide top to bottom and comes out ~29 % larger — and **the bottom row was re-flowed again the same day**, once its subtitle, its summary line, the spectrum's two paragraphs and the slide's `.fig-label` all came off: another ~19 % on top. **6.3 no longer stacks anything.** The cartoon is drawn inside the story canvas, in the box beat 4 was using, so the frame is one full-width picture and the spectrum neither moves nor resizes when it appears — what changes is the argument beside it. **2026-08-20:** the cartoon is headed **6.** rather than 4. (it is the beat *after* the spectrum, not before it), the two chambers now sit at the real **90° to each other**, and the ionisation is drawn as discrete primary clusters with their drift lines, the in-gas track dropped to 30 %. *Know this before you show it:* legs 110° apart onto planes 90° apart forces the incidence to **10° on each chamber** — a minimum-angle pair arrives nearly square, and that is the shortest ladder a µTPC ever reads. It is a property of the station, not of the drawing (in 3-D the pair plane tips out of the page and most pairs land far more obliquely), and it is a fair question to be asked. |
| — | ~~*(optional)* **What the detector has to separate**~~ — **in backup since 2026-08-17** | — | It overlapped slide 3b: both said the pair is bounded at 109°, this one from Geant4 truth and 3b from the generators. Dylan's call — 3b keeps the argument in the flow, this one moved to *Backup · X17 kinematics*, where its single-lepton-energy panel and its "real capsule material degrades this" line answer the question they will be asked. |
| 4 | Chamber design | 60 s | The MPGD audience came for this. **Relaid out 2026-08-17**: the exploded render is landscape — a window on the chamber with the labels on the render — at **56 %** of the width, and the board-peel view takes the other 44 %. Both figures now carry an HTML heading (`.fig-head`) instead of a burned-in matplotlib title. **Zoomed 2026-08-18** to a 44 × 34 mm window at an unchanged frame width, so the strips and pads are ~1.4× bigger, and the muon and its drift lines are a third of their old width. |
| 5 | Micro-TPC operation | 50 s | How one gap gives a 3-D segment. **The right panel is the raw per-strip waveforms since 2026-08-17**, not the first-arrival ladder: the ladder is the estimator slides 6–7 exist to replace, and showing it here as "what the chamber measures" undercuts them. **2026-08-20:** one HTML line under it says the event is **simulated** — it has to, because every constant in it is real — and the burned-in "(measured)" and "measured response (det3)" came off with it. The drift-time colours are new: green → teal → blue → violet → crimson, same darkness, four hue moves instead of one. |
| 6 | **Resistive strips share charge — and delay it** *(build, 9.1–9.2)* | 70 s | **Load-bearing for this audience.** The one genuinely MPGD-methodological result in the talk. Linger here. **Rebuilt 2026-08-17**: frame 1 is the mechanism (the charge goes sideways through the film's own sheet resistance, so the copy is *late*), frame 2 adds **the kernels production actually uses**, X and Y. If you are asked about the amplitudes, the honest answer is on the slide's HTML comment and in `make_share.py`: c₁ is at its calibration floor on a cosmic fit, the H4 beam measures ~0.3, and the fitted **X/Y ratio** is the number worth defending. **2026-08-20:** the right-hand caption came off and the left one is set much larger — it is now the slide's only sentence. What the removed caption said is on the figure itself: each panel carries a plan view of the film showing the charge long along its strips and short across them, with the direction that plane samples marked. That inset is kY as geometry rather than as a fitted number, and it is the thing to point at. **2026-08-23:** frame 9.2 now carries a one-line muted credit under the kernel figure — T2K ND280's resistive Micromegas measure the same object and fit neighbouring channels simultaneously, which is the closest published method to our forward fit. If the room asks "has anyone else done this?", that line is the answer, and the extension to claim is that we solve the *drift-depth profile* inside the fit rather than a sharpened centroid. Full prior-art list, Dixit 2004 onward, in `wft/REFERENCES.md`. |
| — | ~~A cosmic muon, reconstructed~~ — **in backup since 2026-08-18** | — | Dylan's call. It is hits-basis — the cloud is threshold crossings turned into depths — and standing between the forward-fit slides and the efficiency numbers it read as the output of the chain it is not. If you want it back, it is *Backup · Reconstruction*, retitled so the caveat is on the page. A forward-model version can show the fitted segment and the depth profiles but **not the point cloud**; the comment where the slide used to sit says why. |
| 8 | Efficiency: 93% on the best chamber | 60 s | **Rebuilt 2026-08-18.** The loss budget is HTML now, not a PNG, so it reads from the back of the room; beside it the efficiency map (flatness) and the |r| tail (the near-miss story). **The map became a sliding kernel on 2026-08-18** — a 20 mm circle stepped 0.5 mm, at the same 5 mm match the bars use, on `g_det3_wknd` (21,948 rays) because a sliding map lives on statistics. If asked why not the 2 mm the brief said: a 2 mm circle holds two muons. The numbers moved on 2026-08-13 and the slide had not caught up — it is **93.3 % on 7,049 rays**, not 93.5 on 7,055. Say *discharge*, not spark. |
| 9 | Sub-degree angle, sub-mm position | 60 s | **Rebuilt 2026-08-18** — both old figures were 2026-07-14 *hits-basis*, and the slide carried three different numbers for one quantity. Now: reconstructed-vs-reference density (X and Y) and σ₆₈ against angle, both from the frozen waveform-first table. Three measurements, each named: angle from the bench, **position from the SPS beam** (176 ± 35 µm on det4, the fleet's worst chamber — it bounds the design), timing from the bench and already telescope-free. If asked why position is not from the bench: the bench residual is reference- and scattering-limited, and the M3 pointing number excludes scattering by construction. |
| 10 | **The setup, as one build** *(collapse of the 3-D sequence)* | 80 s | **This section grew after this document was first written** — the two old setup slides were replaced by an **eight-frame** in-house Geant4 build sequence (`setup3d_1..8`, deck frames **15.1–15.9**), which is much better material but many times the budget. At 15 min, show **frame 3 (…and a pair leaves, 15.3)** and **frame 9 (the full setup, 15.9)** and narrate the rest over them. The sequence's own header comment nominates frames 4 and 2 as the first to drop; frames 5–7 (the layer-by-layer build) are the next. Everything not shown goes to backup — there is already a "setup to scale" backup slide for it, and the plan view (15.10, the frame the build ends on) now carries the
distances on the page. **The whole sequence is one slide since 2026-08-21** —
ten frames, one number — so skipping frames costs the audience no jump in the
footer. |
| 11 | **How we got here** *(slide 26)* | 50 s | **Rebuilt 2026-08-19.** The nine-month timeline, miniaturized to four beam exposures and their names, with the daily event census of the physics run expanded out of its last bar by a zoom wedge — one figure (`make_campaign.py`), four stat tiles, one caption line. It replaces both the full-text timeline slide *and* the old P1 "six weeks of beam". Says **events recorded**, not "DREAM events recorded". |
| 12 | **Almost all of the X17 rate is in the MeV** *(slide 27)* | 55 s | **New 2026-08-19, and it is the section's premise.** Dylan's December rate calculation on a neutron *flight-time* axis: two decades carry **79 %** of the X17 rate and they arrive **0.45–4.5 µs after the flash**. Say the number, and say that n_TOF sorts the energies for you by time of flight — the rest of the section is what happens when you try to be there. **Linear in y since 2026-08-20** (Dylan): on a log axis the six decades under the peak carried the same visual weight as the peak, which is the opposite of the sentence. The interpolation is PCHIP now, so it cannot overshoot the highest measured point. |
| 13 | **Our read-out is DREAM — and the γ flash rails every channel** *(slide 28, a two-frame build since 2026-08-20)* | 55 s | **Frame 1 is the chip, before any complaint about it** (Dylan): Saclay, CLAS12, 512 channels per front-end unit, a CSA and shaper on each, 20 ns sampling. **Frame 2 is February 2026** — the first prototype on beam, saturated in a way that hides itself — and pops the railing trace up on the right. The **absence of noise** is the tell, and the panel that measures it is now backup. Every word of provenance is in the markup (`.figsrc`), so it can be cut on the day without re-rendering anything. |
| 14 | **The chamber is fine. The front end is not.** *(slide 29)* | 70 s | The slide that makes this section an MPGD talk rather than a DAQ post-mortem. Linger. **Rebuilt 2026-08-20** on Dylan’s scepticism about the numbers: both rows are now **detector A at 540 V** — the n_TOF digitiser from run 224709 (a *named* chamber at the production setpoint) against the run_57 DREAM recovery at the same voltage — so the old "different run, different gas, unknown chamber" caveats are gone. Beside it, **the waveform that proves the top row**: the 45 mV strip pulse back under threshold **2.0 µs** after its own peak, with a DREAM flash on the same axis. The ratio is **×2 435**. |
| 15 | **Every volt of gain costs milliseconds — and the milliseconds are the beam** *(a six-frame build since 2026-08-24)* | 80 s | **Rebuilt 2026-08-24** (Dylan). The charge ladder and the dead-time-vs-charge plot are both in backup now; the slide is the X17-rate-vs-flight-time drawing with **each HV setpoint’s measured recovery drawn as the wall it really is**, built **560 → 550 → 540 → 530 → 520 V and then back to 540**. Three panels: **the cosmic bench’s own detection efficiency** vs HV (the 27 June det3 scan, carried across the 95/5 → 90/10 gas boundary, drawn in both noise eras), three large numbers (voltage, **per cent of the X17 rate left**, efficiency), and charge vs the recovery it buys on the main panel’s own time axis. **Linger on frame 6** — it is the same drawing as frame 3 with every other voltage’s edge left on the axis, which is where you say we chose this. The two costs multiplied (optimum 550 V, 540 V at 95 % of it) was frame 7 until 2026-08-24 and is **backup only** now (Dylan: “kick 25.7 to backup”) — bring it up if the setpoint is questioned. Numbers: `ntof_july_analysis/hv_tradeoff/report.html`. **Three questions this panel invites.** *Which scan?* — 27 June saturday det3, 425–525 V, the only one reaching below the plateau; the 22 June one starts already flat and both are on backup D2c. *Why two curves?* — solid is the production noise where we ran, dashed the quieter July front end; the 22 V between them is the 23 July noise step, and it is what turns 540 V from 81 % into **69 %**. *Is 520 V measured?* — no, it maps to bench 417 V and its number is marked `~` and extrapolated. **The run_55 MIP-track ladder is NOT an efficiency** (doubles trigger, ~50 % geometric ceiling, a 3-strip cluster in both views) and is backup D2b. |
| 16 | **So the measurement we ran is the thermal one** *(slide 31)* | 45 s | **The pay-off, and the same drawing as slide 27 with one thing added.** The dead band goes on the axis, the whole MeV peak is inside it, and what is left is the thermal end at 4.5–14 ms — 10 % of the rate, and recordable. Nothing moves between the two frames, so this costs the audience no re-reading. |
| 17 | **The angle a track arrives with says where it came from** *(slide 32)* | 60 s | **New 2026-08-19** (Dylan), replacing the P5 imaging slide. A schematic of the test beside the test: a source on the beam axis at L = 235 mm can only reach the strip plane at *u* with tan θ = *u*/L, and the line on the data is that relation — geometry, nothing fitted. Arm A, 7,849 tracks that a SiPM segment *and* a plastic bar both confirm. Say **PRELIMINARY**: the angle *scale* is still bench-transferred and reads about a quarter shallow; the sign, the correlation and the zero crossing do not depend on it. **Do not quote a drift velocity off this slide** — that story came off the deck on Dylan’s instruction. |
| 18 | **Two opposing arms, seen from above, one capsule** *(slide 33)* | 50 s | **New 2026-08-19**, and the talk’s closing result. The same statement as a picture: A and C sit 235 mm either side of the beam, every track drawn from its strip-plane crossing back to its closest approach, and the two fans nose in on the capsule. 15,194 tracks, all pointing-coincident, drawn exactly as reconstructed. Ends on **“further analysis is in progress”**, which is where Dylan wanted the section to land. |
| 19 | **Summary — and what the data are for** | 60 s | **Reworked into a figure slide 2026-08-24.** Three one-line bullets (~1° on a track · the campaign, 41.8 M / 17.9 TB / 44 days · the flash saturates the front end, so what we recorded is the thermal window) over the outlook figure: **find the two-track events → histogram the opening angle.** The left panel is the station in plan view and *to scale*, so a 110° pair visibly cannot land in one chamber; the right is the spectrum that falls out, with its four features named on it. Say out loud that the **X17 yield on it is drawn, not predicted** — the panel says so too, but say it. The two-chamber background comes out **flat** because the acceptance rises about as fast as the physics falls, which is the encouraging part. Closes forward, on the pair search. |

Total ≈ 15 min of slides after the 2026-08-19 rebuilds — three slides more
than the old cut, because the Status section became an argument rather than a
list. The three cheapest drops, in order: **3b**, **15** (the charge slide, if
the room is not a front-end room), and **18** (the overhead picture, which 17
already establishes). That is back to ~13 min. Slide 13 is a build, so its
first frame can also be narrated over frame 2 if you are running long.

## What moves to backup (not deleted — you will be asked about most of it)

| Slide | Where it goes |
|---|---|
| Outline | **Cut.** At 15 min an outline slide costs 15 s and tells the audience nothing they cannot infer. First thing to cut if you are over. |
| n_TOF facility (**now slides 5–9**, a five-frame build) | Backup, **all five as one unit** — and there is now a **sixth, already-in-backup slide** carrying the same figure with its full caption (the provenance and the drawn-not-measured list, moved off the main flow on 2026-08-11 so the pictures could use the full slide height). Its content compresses to one caption line on slide 3. On 2026-08-10 the single facility slide became a build-up of one in-house render; on 2026-08-11 the CERN photograph of the hall was set to the **left** of the render on every frame, and the build was cut finer to five: target → the neutrons leaving at 90° → the middle of the line (collimator, floor, shielding, the pipe ending) → back into a pipe on up to the dump → the measuring station. **Back to four on 2026-08-18** (Dylan): the dump frame revealed no new bullet — its side column was byte-identical to the frame before it — so it is cut and the station is now 4.4. The render is still made; the section is in git. The deck has no fragment mechanism, so a build is consecutive slides. **This grows the printed deck from 28 to 32 main-flow slides but does not change the 16-slide cut above** — the whole unit was already demoted, and it is still demoted. If you find 60 s for the facility after all, **show frame 5 only** (`ear2_onfig_5_station.png`) and narrate the build over it; the frames are subsets of one picture, so nothing is lost but the reveal. If someone asks where a number or a drawn detail comes from, **the answer is the backup slide**, not these. And if the question is about the **spallation target** specifically &mdash; what it is made of, how it is cooled, why the beam hits it at 10&deg; &mdash; the answer is the two target backup slides (**46 and 47**), which carry the full geometry and the whole nitrogen circuit from the design paper. |
| Characterized on the Saclay cosmic bench (**slide 12, a five-frame build since 2026-08-18** — the GIF is gone; the five stills page one at a time with HTML pins calling out the 60 × 60 cm triggers, the 50 × 50 cm M3 references and the 40 × 40 cm chambers under test. Show **frame 5** alone if it stays but the time does not) | **Depends on Alexandra's P2 talk — flag left in place.** If she covers the bench, cut to a one-line back-reference on slide 8. If not, it is the first slide to *restore* if you find time. |
| The target, standalone (slide 14) | Backup, as the capsule-geometry detail slide. |
| Trigger and calorimetry, 3-panel (slide 16) | Backup. |
| How much charge (D2), Charge vs gain (D3), Dead time vs gain (D4) | Backup. All three are the abscissa and the ladder that D5 joins; D5 states the conclusion. **D4 was on the original "load-bearing if time is tight" list** — if you would rather show the dead-time-vs-HV map than the charge join, swap 14 for D4, but not both. |
| What we record, and when (D7) | Backup. |
| The tracks point back at the capsule (D8) | **Judgement call.** It was on the load-bearing list, and "reconstruction transfers from bench to beam" is a real result. But it is PRELIMINARY on one arm / one sub-run / a transferred calibration whose own document says nothing is quotable. At 15 min with a caveat that long, I would keep it in backup and mention the result in one sentence on slide 11. Restore it to main flow only if the in-situ calibration lands before 3 September. |

## Added 2026-08-13: the reconstruction-algorithm slides

Two new main-flow slides now sit between slide 6 (charge sharing) and slide 7
(the cosmic-muon event display): **"The forward fit: predict every waveform,
never invert one"** (the mechanism) and **"One muon through the forward fit"**
(one event, model vs data, with the resolution numbers). **The first of the two
was rebuilt on 2026-08-17** — its four paragraph bullets and its callout are
gone, replaced by the four-stage diagram of what the model does and, beside it,
the same decomposition **on real data**: four consecutive strips, each waveform
split into own / ±1 / ±2 charge in the three colours slide 6 has just taught.
Eight
`Backup · Reconstruction` slides carry the question-answering depth
(NOTES.md, same date, has the map and the figure provenance).

**This does not change the 16-slide budget: at 15 minutes show only "One muon
through the forward fit"** — give it slide 6's leftover seconds and narrate the
mechanism over the figure ("we predict every strip's waveform from three track
numbers plus a free depth-charge profile, sharing included, and χ² does the
rest"). The mechanism slide is for the 20-minute version, or for a room that
asks methodology questions early. If both are shown, budget 60 s + 50 s and
recover the time from the facility build (slide 10's own note already nominates
its droppable frames).

## Added 2026-08-15: the timeline and the post-LS3 slides

Two main-flow slides landed after this order was written, both at Dylan's
request: **"How we got here"** (a project timeline, opening the Status section)
and **"What a post-LS3 measurement needs"** (between D10 and the Summary).
NOTES.md, same date, has their provenance and says how to find them in
`index.html` without a page number.

**Neither is in the 16-slide cut above, and this document does not decide for
them** — that is Dylan's call, and it is a real trade because both are cheap in
seconds and expensive in what they displace:

- **The timeline is a ~40 s slide** and is the only thing in the deck that says
  why the Status section is about a DAQ problem at all — the flash was the
  constraint from the first exposure in November 2025, and the audience
  otherwise has to take that on trust. If it goes in at 15 minutes, take the
  time from slide 10 (the setup build already nominates its droppable frames).
- **The post-LS3 slide overlaps slide 16 (Summary) and slide 15's last bullet.**
  At 15 minutes, showing both the outlook slide *and* D10 is redundant: **pick
  one.** The outlook slide is the better close for a room that wants to know
  what a future front end has to do; D10 is the better close for a room that
  wants what this campaign established. At 20 minutes, show both, in that order.

Both slides also state that data taking **ended on 10 August**, which is why
the three stale "still running" lines elsewhere in the deck were corrected the
same day (NOTES.md).

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

**The two photographs of the real station are now ONE slide, a two-frame build
(25.1 / 25.2)** — top-down into the assembly, then one arm from outside, with a
one-line label each and no bullets (2026-08-18, Dylan: *"so I can get the second
image on right when I click"*). They are still not in the 15-minute cut and
still not worked into the argument; everything the old bullets said is speaker
material now, including the "SiPM gamma-flash recovery" silkscreen, which is
read off the full-resolution original and is not legible at slide size.

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
