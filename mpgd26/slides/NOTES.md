# mpgd26/slides — status and open items

## The Status section, rebuilt — 2026-08-19 (fourth batch)

Dylan: *"work on the end of the presentation"* — miniaturize the timeline and
fold the campaign numbers into it, then tell the high-energy/low-energy story
with the expected-statistics plot, and end on the tracks pointing at the target
with analysis in progress. Four questions were asked and answered before any
markup moved (three flash slides, the run_145 pair of pointing slides, Summary
only after them, and the thermal-trigger slide kept).

### What the section is now

It used to be a list of results, drafted 2026-08-09 as "more slides than the
talk can hold, build it long then cut". This is the cut, and the ordering
principle changed with it: the section is **one argument**.

| | |
|---|---|
| 26 | how we got here, and the dataset, on one canvas |
| 27 | where the X17 rate actually is — the MeV, first microseconds |
| 28 | what the flash does to a channel (and DREAM is introduced here) |
| 29 | the chamber is fine; the front end is not |
| 30 | the flash, weighed — and dead time follows the charge |
| 31 | back to the same axis: the MeV is inside the dead time |
| 32 | and the trigger really does live in the thermal window |
| 33 | the tracks image the capsule |
| 34 | all four arms, and further analysis in progress |
| 35 | summary |

Nothing was deleted. D0, D2–D5, D6, D7, D8, D10, the post-LS3 slide, P2, P7
and the full-text timeline all went to backup, in the order the questions get
asked. **The "Proposed · 13 Aug" divider is gone**: P1, P3, P4, P5, P6 and P8
were adopted into the flow, P2 and P7 into backup, so the block that had been
waiting for a decision since 13 August is resolved.

### 27 and 31 are one drawing with one thing added

`make_x17_rate.py` draws both frames from the same code path: same points, same
limits, same annotation positions. Only the dead band appears and the accent
moves from the MeV decades to the thermal bin. That is the whole reason to come
back to the figure three slides later — the audience re-reads the axis for
free, and the argument lands in the picture rather than in a sentence.

**The data** is Dylan's December 2025 calculation, `results_3He` from
`/media/dylan/data/x17/calculation_tables/`, copied into
`data/x17_rate_3He.txt` so the figure builds without the mount. Read the way
`X17CalculationParser` reads it, but positionally: the header is two comment
lines and the units line repeats column names.

| neutron energy | arrives over 19.5 m | X17 / day |
|---|---|---|
| 0.1–1 MeV | 1.4–4.5 µs | 15.2 |
| 1–10 MeV | 0.45–1.4 µs | 17.9 |
| **0.01–0.1 eV** | **4.5–14.1 ms** | **4.42** |
| everything else | — | 4.7 |

**79 % of the rate in two decades, and they arrive in the first 4.5 µs.** The
thermal bin is 10 %, and it is the only one that arrives after the front end is
back. The ratio is what the slide uses; the absolute per-day numbers are for a
nominal cell (40 mm of gas, 500 atm, 0.5 mm Al + 1.2 mm CF, against the built
Ø20 mm bore with 40 mm of gas in 0.6 mm Al + 0.9 mm CFRP — same gas column,
walls within ~30 %) and this talk still makes no reach claim.

**The dead band** is two measurements, both `STATUS_PLAN.md` §1.2–1.3: no track
has ever been reconstructed before 0.993 ms on beam (run_79), and the run_57
recovery map puts the slowest chamber at the production operating point at
8.9 ms. Hence a firm edge at 1 ms and a fading one to 9 ms.

**Bars became points on Dylan's instruction** — the interpolated plot he had
already developed, `plot_spectrum_vs_time` in the repository root's
`neutron_energy_vs_flight_time.py`: one marker per decade with the exact bin
width as an asymmetric error bar, and a faint log-log cubic spline through
them. The spline is a reading aid; ten points over six decades, so it can and
does overshoot between them, and nothing is derived from it. With points there
is no bar to recolour, so the highlighted window is a shaded band instead —
which is what the sentence is about anyway.

### 26: the timeline and the census, on one canvas

Two figures joined rather than stacked, for the width-limited-figure reason:
two pictures one above the other are each capped at about 60 % of the slide's
width, so as one figure they share it. The join is drawn — a zoom wedge from
the July–August bar down to the corners of the events panel, because the events
panel *is* that bar opened up. Without it this is two plots that happen to be
adjacent.

- **The Saclay bench month is off the strip.** It is the one bar that is not a
  beam exposure, and it was also the one crowding the four remaining labels
  onto two tiers. Removing it and going back to one line each is the same edit.
- **"Events recorded", not "DREAM events recorded"** — Dylan's instruction. The
  census counts triggers of our own read-out either way, and the acronym buys
  the audience nothing at this point in the talk. (DREAM is introduced two
  slides later, on 28, where it is load-bearing.)
- The full-text timeline is unchanged and is now the first backup slide,
  retitled *"How we got here, in full"* so it does not read as a duplicate of
  26. `make_campaign.py` imports `CAMPAIGNS` from `make_timeline.py` rather
  than copying it, so the two cannot drift apart.

### Measuring the hole, again — and a trap

Both new figures are saved **on their full canvas**, at the measured aspect of
the hole they go into: **2.947:1** for slide 26 (four stat tiles and a caption
under the figure) and **2.225:1** for 27/31 (caption only). Measured the usual
way — print the slide through headless Chrome with `.figure-solo{background:red}`
and the img hidden, then read the red box off the PDF.

The trap: `plotstyle.use()` sets `savefig.bbox = 'tight'`, and
**`fig.savefig(..., bbox_inches=None)` falls back to that rcParam** rather than
disabling it. The first attempt came out at 3.23:1 and 2.44:1. The fix is to
name the canvas explicitly, `bbox_inches=fig.bbox_inches, pad_inches=0.0` — and
once you do, nothing rescues an axis label that falls off the edge, so both
margins have to hold their own axes (slide 26 needs room on the right for the
cumulative axis, which the first version cut off).

`P.note` anchors at *x* = 0 and leans on the tight box for its left margin, so
both scripts carry a local `_note()` that anchors at the axes' left edge and
hard-wraps to a character count instead.

### Smaller things in this batch

- **28 now introduces DREAM** rather than assuming it: retitled *"Our read-out
  is DREAM — and the flash rails every channel of it"*, and the caption opens
  with what DREAM is (1,024 channels per chamber, a CSA and shaper on each,
  sampled at 20 ns). Its first appearance in the visible deck used to be the
  word in the old title.
- **Neither new figure carries a title block.** Both live on `figure-solo`
  slides whose own title bar says the same thing, and the height goes to the
  plot. The first version of 27 repeated its slide title verbatim.
- Captions on 28, 29, 30, 33 cut by roughly a third; P3 and P6 went from four
  bullets to three. P6 ends on *"further analysis is in progress"*, which is
  where Dylan wanted the section to land.
- The twelve slides moved to backup had their kickers relabelled `Status ·` →
  `Backup ·`, or they read as main flow in the printed deck.

## Eleven slides at once — 2026-08-18 (third batch)

One message, eleven edits. What follows is what changed and, where a request
could not be met literally, the arithmetic that says why.

### 4.4 removed

The EAR2 build ran 4.1–4.5; frame 4 was the wide upper pipe on its way to the
dump, and **its side column was byte-identical to frame 3's**. A build frame
that reveals no bullet is carrying a picture, not an argument. The render
(`ear2_onfig_4_dump.png`) still exists and is still regenerated; the section is
in git if the beam dump ever needs a slide. The old frame 5 is now
`data-frame="4"`.

### Slide 6: the figures again, and the stack finally gone

*"Rework again the figures to be taller… Keep the 4. and 5. titles, but remove
the 'in the rest frame…' and 'whatever the orientation…'. Also remove the text
'The edge at…' and 'X17 drawn at…'. Finally remove the explanatory text on the
bottom. For 6.3, remove the left diagram with the angles and replace it with the
MMs, keeping the spectrum in place."*

Four separate things came off, and each one bought height:

| removed | was | cost it was paying |
|---|---|---|
| beat 4's subtitle | *"in the rest frame the pair is always back-to-back"* | said in words what the rest-frame icon shows |
| beat 4's summary | *"Whatever the orientation, X17 stays open…"* | said in words what the five angle numbers show |
| beat 5's two paragraphs | *"The edge at 109° is the measurement…"*, *"X17 drawn at 4 %…"* | ~14 canvas units of the band |
| the slide's `.fig-label` | four lines of generator provenance | four lines of *slide* height, on a width-limited figure |

The fig-label going is what let the band be **re-measured and re-shaped**: the
hole is now **2.028 : 1** (red-box recipe again), so `STORY_PARTS['bottom']`
went `bare=(4.6, 62.0)` → `(0.9, 62.0)`. The top row keeps its label and keeps
2.16.

**Where the extra size actually came from, because it was not free.** The row
is width-limited, so growing the drawing means finding *width*, and beat 4 had
none: five orientation columns at pitch 12.6 already ran from x = 25 to x = 86.
The units came out of the **left block**, which was rest-frame icon → boost
arrow → columns, all on one line, 21.5 units of them. Stacking the icon over
the arrow (vertical space the row now has) returned ~9 units, which paid for
**pitch 12.6 → 15.0 and every element at 1.19×**: arms 4.4 → 5.2, icon r
1.6 → 2.0, the angle numbers 9.0 → 10.4 pt. Beat 5's panel went **34 × 26 →
34 × 38** into the height its two paragraphs had been using.

Two collisions had to be fixed on the way, both of them the θ\* = 0 column,
where the X17 pair really is back-to-back and one arm points *backwards*:

* at pitch 15 that backward arm arrives exactly where the *previous* column's
  angle number sat. The number moved **2.2 units above the vertex line** — no
  width, and the row has the height.
* the last column's `back-to-back` note, set flush left off the number, ran
  into the spectrum panel's left spine. It is **centred under its own column**
  now.

**6.3 is one picture again.** The cartoon is drawn *inside* the story canvas,
in beat 4's box (`scenes_x17._story_detect`, `make_x17.py --layout bot3`), at
0.87 of `draw_detect`'s length scale — `_utpc_arm` grew a `scale` argument
rather than acquiring a second implementation, so the 110° is still the
kinematic minimum drawn true. The spectrum does not move and does not resize
between 6.2 and 6.3; **what changes is the argument beside it.** That closes
the open question from the second batch: two stacked full-width pictures can
never be more than ~59 % as wide as one, and now nothing is stacked.
`--layout detect_solo` still writes the cartoon on a canvas of its own, for
report.html.

### Slide 8: dark colours, thicker traces

*"Please use only dark colors and make the lines a bit thicker. Very good plot
but I worry about it showing up on a projector."*

`style.microtpc_cmap`'s light-theme cut moved **0.78 → 0.45** on the plasma
ramp, which takes the hottest colour's relative luminance from **0.67 to 0.36**
— every trace is now dark against the white page and the blue → purple → red
ordering still reads as drift time. Traces went `lw = fs*0.055 → fs*0.085`.

It is **one cut for the whole figure**, not just the panel complained about:
the render on the left, the stacked waveforms on the right and the colour bar
under them all encode the same drift time, and darkening only the right-hand
panel would put two colour scales for one quantity on one slide.

### Slide 9: "resistive layer", not "film"

Changed on the slide (fig-heads, fig-labels, alt text), on the figure
(`make_share.py`'s margin label and axis label) and in `report.html`'s blurbs.
Variable names (`FILM`, `Z_FILM`) are untouched — renaming a data field to fix
a slide is the tail wagging the dog.

### Slide 12: the GIF is five slides now

*"Replace the gif with just the equivalent images as 12.1, 12.2, … then add
html arrows and labels pointing out what is added as the structure is built."*

The five stills are not new renders — they are the frames the GIF was already
made of (`animations/build_bench_{1..5}_*.png`, `make_anim.py`'s `build_bench`
job at one fixed camera). A GIF cannot be paged, cannot be held on the frame
being talked about, and prints as whatever frame the PDF exporter caught.

The labels are new markup, `.pin`, and the one thing worth knowing about them:
**they are positioned against the image, not against the slide.** The still is
portrait (1300 × 1650), so it is height-limited and its rendered *width* depends
on the projector's aspect ratio — a pin placed in slide coordinates would slide
off its target on any screen but this one. `.pinwrap` is an inline-block sized
by the image, so `left:52%` is 52 % of the *picture*, always. Each pin is
anchored on **both** sides — the target in per-cent, the label at a fixed em
offset *outside* the picture — and the leader is what stretches between them
(`flex:1`), because a leader set in em would be the wrong length at any other
size. A pin already made stays and fades (`.done`), the `.dim` grammar the
bullet builds use.

### Slide 13: the sliding kernel

*"Please use the sliding 2mm kernel efficiency plot… takes a 2mm circle,
calculates the matched reference (<5mm) / all reference within that circle, then
moves the circle 500um. It should be yellow for efficient… use the highest
statistics det3 data set."*

The code meant is `mx_june_wft/report/make_june_figs.py:sliding_map`;
`mpgd26/make_efficiency_map.py` is that definition with the deck's axes on it,
and it replaces the 40 × 40 **binned** `efficiency_map_2mm`. Two things could
not be done as asked:

* **the kernel cannot be 2 mm.** 21,948 rays over a 354 × 389 mm box is
  0.16 rays/mm²: a 2 mm circle holds **two muons**. A 12 mm one was tried and
  holds ~75, where one missed muon moves a pixel by 1.3 % and every individual
  miss paints its own 24 mm disc — the map comes out as a field of blue circles
  that is *entirely counting noise* and reads as structure. **20 mm** holds
  ~224, one miss is 0.45 %, and what is left on the map is the chamber.
* **the step is 0.5 mm exactly as asked**, and free: the map is an FFT
  convolution of a 0.5 mm histogram with a disc, not the reference
  implementation's 550,000 × 22,000 double loop.

A second cut was needed beyond the June chain's `min_rays = 30`: a circle
hanging half off the chamber has the statistics but not the *area*, and drew a
dark fringe all the way round the map that is a property of the active-area
boundary. A circle now also has to be **55 % full** at the run's mean ray
density.

**The map and the bars are different runs, deliberately.** Bars: `sat_det3`,
7,049 rays, 93.3 %. Map: `g_det3_wknd`, **21,948** rays, 93.1 % — the highest-
statistics det3 set on disk, and a sliding map lives or dies on statistics.
They agree to 0.15 points, which is the check that they describe the same
chamber. Do not quote a number off the map as if it came from the bars.

Yellow-for-efficient means **viridis**, which is the one figure in the deck that
does not use `plotstyle.efficiency_cmap()`. Matching the June report's own
sliding maps was worth more here than matching the loss bars beside it.

### Slide 15: the capsule's own gauge, in markup

*"Can we pull the 3He capsule pressure from the run and put it on this slide
formatted nicely as an html plot rather than just python plot?"*

`tools/make_pressure_svg.py` prints an inline `<svg>`; it is pasted between the
`BEGIN/END he3-pressure` markers on the slide. Same argument as the efficiency
slide's loss budget: a small matplotlib panel beside body text arrives in
matplotlib's font at whatever size the PNG was saved and reads smaller and
greyer than the type next to it. As SVG it inherits the deck's colours and type
scale, stays sharp at any projector resolution, and costs ~9 kB.

Data: `ntof_run_report/data/he3_pressure_5min.csv`, the five-minute reduction of
**1.08 M samples** (Keithley 2000 on the capsule transducer over GPIB, ~2 s
cadence). **504.8 bar at the 14 July mount → 494.7 at the 10 August dismount**,
−0.38 bar/day, with a clean day/night breathing cycle of about half a bar on
top — which is the thing the plot shows that no number can. The 8 July bench
stub and the end-of-run vent to 7.8 bar are both out of the trace, for the
reasons `ntof_run_report/figures_local.py:capsule_pressure` gives. One gotcha
worth recording: the trace is rebinned hourly for markup size, and the gap
threshold has to be read **against the bin width** — at the five-minute file's
1800 s it made every hourly point its own one-point polyline.

### Slides 15–24: the text cut, and what the figures did with the space

*"Pretty good on visuals, but have WAY too much text as is. We need to hugely
simplify the story, maybe one or two quick bullets per slide."*

Every side column is now one to three short bullets; every paragraph caption and
`.callout` in the setup section is gone. 15 → 16 → 17 **accumulate** (Dylan:
*"keep the first bullet in place"*, *"add another bullet in place"*), with the
carried bullets `.dim`, which is the build grammar. 18 resets, and 18–24 each
carry their own.

The nine slides then went `0.92fr / 1.08fr` → **`1.32fr / 0.68fr`**: the width
the text gave up is width the render takes, and since these figures are
width-limited it is height too. That needed `align-items:stretch` with it —
without it `.figtext` centres, `.figure` stays unbounded in height, and the
first thing the newly-wider render did was climb over the title rule.

### 25 + 26 → 25.1 / 25.2

*"Combine slide 25 and 26 into 25.1 and 25.2 (so I can get the second image on
right when I click), eliminating almost all text."* One overlay build, one
`.fig-label` each, no bullets. What came off is all speaker material now — and
the "SiPM gamma-flash recovery" silkscreen line was always speaker material,
since it is read off the original and is not legible at slide size.


## The opening slides fill the page — 2026-08-18 (second batch)

*"For the figures on slides 5 and 6, can you rework the python scripts to make
them taller such that they better fill the page? Also on the previous slides,
increase the text size to better fill the page."*

### The measurement that drives all of it

The figure hole on one of these slides is **1186 × 547 px** (measured, not
estimated: the slide printed to PDF with `.imgwrap{background:red}` and the red
box read off — the recipe is worth repeating whenever a figure looks small).
That is **2.11 : 1** for a slide with a one-line `.fig-label`, 2.17 : 1 with a
two-line one. Any figure flatter than that leaves slide height empty; the
top story row was **4.57 : 1** and used 46 % of the height, the bottom row
3.92 : 1 and 55 %.

The second half of the arithmetic is the one that matters for legibility. These
figures are **width-limited**: whatever the canvas is, it is drawn across the
same 12.4 in of slide. Type is set in POINTS and the drawing in canvas UNITS,
so the *only* lever on rendered size is **how many units the row spans** —
160 units across 12.4 in renders 9 pt type at 7 pt; 124 units renders it at
9 pt. Making a figure "taller" and making it "bigger" are therefore the same
operation: re-flow the content into a narrower, taller box.

### What changed in `scenes_x17.py`

`SW = 124.0`, a story canvas separate from the compact layout's `W = 160`, and
every beat re-flowed into it. Nothing was cut, and **not one font size
changed** — the whole figure simply comes out ~29 % larger:

| beat | before | after |
|---|---|---|
| 1 beam | vessel at scale 0.245, bubble beside it | scale **0.40**, the one naturally vertical object pays for the height |
| 2 capture | n + ³He → ⁴He\* left to right, 38 units wide | **stacked downwards**, half again as large in two-thirds the width — and the arrow now points the way the beam goes |
| 3 channels | 20.58 MeV drop over 15.4 units, channels 7.3 apart | drop over **21.4**, channels **10.4** apart |
| 4 boost | column pitch 15.0, β and γ on one line | pitch **12.6**, β/γ stacked (the one line reached into the first θ\* label) |
| 5 spectrum | axes 39 × 21.4 | **34 × 26**, both text lines wrapped |

Rows are now **2.16 : 1**, i.e. the shape of the hole. Two traps found while
doing it, both recorded in the code: `_story_measure` placed its axes against a
default `x1f=W` — 160, not `SW` — which put the spectrum a third of the way
into beat 4; and at column pitch below 12.6 the θ\* = 0 example's **backward**
arm is drawn through the previous column's angle number (the X17 pair really is
back-to-back there).

### What it cost frame 6.3, honestly

Two full-width pictures stacked in one figure box can never be more than ~59 %
as wide as one. Frames 6.1–6.2 now fill the page, so **6.3 shrinks both of them
to that**: the spectrum ends up about the size it was before this batch, and
the cartoon about a quarter smaller. The row weights were `1fr / 1.30fr`,
tuned for the old 3.92 : 1 band; they are now **`1fr / 0.715fr`** (weights go as
1/aspect), which is what makes the two equal width. *Redo them whenever either
figure's shape changes.*

If that is too small in the room the fix is not the weights, it is to stop
stacking. Two ways, both a small edit:

1. **cartoon alone on 6.3** — full width, ~28 % larger than it has ever been;
   the spectrum has been on screen for two frames by then. Costs a rewrite of
   that frame's `.fig-label`, which currently describes the generators.
2. **swap the cartoon for beat 4 inside the story canvas** — frame 3 becomes
   one page-filling picture, spectrum at full size on the right, cartoon where
   the boost examples were. Costs `draw_detect` being refactored to draw into a
   given box, and it breaks the build rule that nothing already drawn leaves.

### The opening slides (1–4)

A `.slide.lead` class, not a change to the `:root` variables — raising
`--fs-body` globally would reflow every slide in the deck, including the ones
the parallel session has. It redefines the type variables locally, so bullets,
captions, `.callout` and the outline items all scale together: body **+25 %**,
caption +16 %, section-title line +12 %, title-slide title 4.2 → 4.8 vw. The
outline also takes the extra *width* (34 → 44 em) and air (padding .55 → 1.45
em) or four lines just sit in the top-left corner. Slide 3's columns went
0.74/1.26 → 0.68/1.32: the bullets read in a narrower column now, and the
figure is width-limited, so every point the text gives up becomes panel height.

⚠️ **Any new frame of the EAR2 build must carry `lead` too** — the frames of one
build have to be identical apart from the beat being revealed, and a frame
missing the class would resize its bullets mid-build.

## Five slides, one batch — 2026-08-18

Dylan's second batch of the day, worked while a parallel session had slides 9
and 10. Everything below is main-flow; nothing between slides 9 and 11 was
touched except one number (see the last bullet).

### 6.3 — the transition to the Micromegas

*"Slide 6.2 needs a transition to the Micromegas. Maybe we add a 6.3 with a
cartoon at the bottom of the page showing e+e- coming out at an angle and going
through a cartoon of our Micromegas drift volume."*

New third frame on the opening-angle build, and a new figure —
`x17_story_bot_3_detect.png`, from `scenes_x17.draw_detect`
(`make_x17.py --layout detect --slides`). It makes **one** claim: a micro-TPC
turns one gas gap into a direction, so two of them give the opening angle. That
is the argument for the whole detector half of the talk, and until now the deck
cut from a kinematic spectrum straight to an exploded chamber.

What is true on the drawing and what is not:

- the **opening angle is the real 110°**, drawn true — put a protractor on the
  slide and it measures 110;
- the standoff (204 mm from a 23 mm capsule), the 30 mm gap and the 400 mm
  chamber are **not** to scale, and at scale the gap would be a hairline;
- the **chamber** carries the 21° tilt, not the track. A track square to the
  readout plane leaves all its charge at one depth and there is nothing for a
  micro-TPC to reconstruct — my first draft tilted the track instead, which
  silently turned the drawn opening angle into 62°.

⚠️ **This build deliberately does not reserve the cartoon's row on frames 1–2**,
which is the one place in the deck that breaks the `.fut` rule. Reserving it
costs those frames half their figure — the story band is width-limited, so a
hidden second row does not leave blank space, it shrinks the picture above.
Frames 1 and 2 are byte-identical to what they were before this batch.

### 7 — the chamber, zoomed, with a thinner muon

*"Make the track and drift lines significantly smaller. Maybe even zoom in on
this track horizontally to better show the detail on the strips/pads, while
keeping the same width?"*

`scenes_chamber.WIN_MM` 60 → **44 mm** across, depth untouched at 34, with
`make_chamber.VIEW`'s `view_angle` 17.8 → 16.6° so the frame width does not
change — **a magnification, not a crop**, and the deck's column weights did not
have to move. 56 strips on screen instead of 77. The muon's tube went 0.9 →
0.30 mm and its drift lines 0.28 → 0.10: at 0.9 the track was 1.2 strip pitches
across, drawn at the scale of the structure it is supposed to be crossing.

Fixed while there: the two `.fig-head`s on that slide were **overlapping the
title rule**. `align-items:center` on the `.cols` leaves each grid item at its
content height, so `.imgwrap`'s `flex:1` has nothing to resolve against and the
image takes its natural height, overflowing the row top and bottom. Stretch (the
default) fixes it. Worth knowing before adding `.fig-head` to another slide.

### 12 → backup, and what a forward-model version could show

*"Kick slide 12 to backup — don't know if we can generate something like this in
the forward model?"*

`event_display_3d.png` is **hits-basis** — every point is a threshold crossing
turned into a depth — and it stood between the forward-fit slides and the
efficiency numbers, where it read as the output of the chain it is not. It is
now `Backup · Reconstruction`, retitled *"A cosmic muon, as the hit-time chain
saw it"*, with the caveat stated rather than implied.

**The answer to the question: partly.** The fit returns, per plane, (p₀, w, t₀)
and an 18-bin non-negative charge-versus-depth profile. Combine the planes and
you have a genuine 3-D segment plus two depth profiles, so *"the fitted track
through the gap, with the measured charge along it, and the M3 line beside it"*
is a straightforward job on the frozen parquet. What cannot come out of it is
this picture's **point cloud**: the model's charge is q_x(z) and q_y(z), two 1-D
profiles, and their outer product is not a measurement of where the charge was.
A forward-model display would be more honest and less pretty. Not built.

### 13 (was 12) — efficiency: the budget is markup now

*"Grab the numbers used in the python plot and make a new plot in html directly.
Instead of 'spark' I think we want to use the term 'discharge'. For the
reconstructed more than 5mm away, we can add a footnote… we want to show the 2mm
kernel scan efficiency map along with the residual distribution with the tail."*

- the loss budget is **HTML** (`.bar-chart.loss`, new CSS). The percentages are
  still the analysis JSON's and the row colours still match
  `make_efficiency_breakdown.py`'s `ROWS`, so the standalone figure and the
  slide agree bar for bar. The bar *width* is the number, including the 0 %
  row, which is why the "Silent" track is empty.
- **"discharge" everywhere on the page.** The JSON key stays `spark_cat` —
  renaming a data field to fix a slide is the tail wagging the dog.
- the `>5 mm` footnote says what Dylan asked it to: part of the tail is ours (a
  fit that locked onto a second deposit, recoverable) and part is the reference,
  whose own pointing has tails a core σ does not describe.
- new `efficiency_map_2mm.png`, and the residual figure is now one panel — its
  old second panel re-plotted the same histogram's cumulative.

⚠️ **The numbers on that slide had been stale since 2026-08-13.** The markup
said 93.5 % on 7,055 rays; the current reduction is **93.3 % on 7,049**, with
4.0 / 2.4 / 0.3 / 0.00 below it. The fleet bars on the backup slide were stale
the same way and are refreshed (93.3 / 92.0 / 74.9 / 57.0 / 41.6), as is its
core-σ row (det4/det7 are 0.56 / 0.62 mm, not 0.64 / 0.67).

⚠️ **Read before quoting a level off the map.** 2 mm is the *tight* criterion —
86.5 % detector-wide against 93.3 % at 5 mm — and it is **not a 2 mm kernel**:
12 mm bins of a 2 mm criterion, because a literal 2 mm kernel holds ~0.05 cosmic
muons. The map is there to show *flatness*, not level.

### 14 (was 13) — resolution: three measurements, named

*"Use the spatial resolution as extracted from the SPS beam test, erring on the
side of caution … show the 2D correlation plot of reference angle vs
reconstructed angle for X and Y and then make a new angular resolution plot from
this … clear out all the superfluous text."*

Both old figures were **2026-07-14 hits-basis** from `engineer_package`, i.e.
the estimator `RECONSTRUCTION_BASIS.md` forbids for geometry. `angular_
resolution.png` showed 1.66° while the tile beside it said 1.7 "hybrid" and the
reconstruction slides claimed 1.0–1.1: three numbers for one quantity, on one
slide. Both are replaced by `make_resolution.py` (new), which computes from the
frozen waveform-first table through `mx_june_wft/03_angles.py`'s own accounting.

| number | where it is from | why not from somewhere else |
|---|---|---|
| **1.19° / 1.16°** σ₆₈ per plane | det3, `wft/events.parquet`, no `slope_reliable` gate | cross-checks against `wft/angles/angular_resolution.json`'s `s68_deg` |
| **0.18 mm** position | **det4 at SPS H4**, `sps_beam_test_26/analysis/spatial_resolution/` | the bench cannot do this honestly — its residual is reference- and scattering-limited, and the M3 pointing number excludes scattering by construction. At H4 the DUT sits *between* two reference planes and the back one carries three pitches, so the reference is **fitted out** |
| **33 ns** timing | the June bench, `42_time_resolution.py` | Dylan asked whether this could come from the SPS too. **It cannot and does not need to**: det4 at H4 was mounted flat, so there is no drift-time ladder in that data at all — and 33 ns is already telescope-free, being (t_X − t_Y)/√2 for the same electrons under one gap |

**The conservative band on 176 µm.** Quoting the fit's own ±10 µm would be
dishonest by omission — it is the error on an *intercept* extrapolated to zero
reference pitch from three points. `make_resolution.spatial_band()` adds, in
quadrature: the fit error inflated by √(χ²/dof) (±16), the 176–212 µm spread
across the five reference zones (±18), and the report's own ±25 for a
factor-two error on the assumed front-plane resolution. **176 ± 35 µm**, i.e.
141–211 on a 0.78 mm pitch. What no error bar covers is on the slide in words:
**det4 is the fleet's worst chamber**, so this bounds the design rather than
describing det3.

### One number changed outside this batch

Slide 11's caption said σ_θ = 1.15° / 1.14°, the MAD-based robust σ from an
older reduction. It now says **σ₆₈ = 1.19° / 1.16°**, matching slide 14 — two
estimators for one quantity on adjacent slides is a question you do not want.

## Slides 9 and 10 are figures now, not bullets — 2026-08-17

The reconstruction pair was rebuilt on Dylan's brief: *"for slide 9 and 10 we
have quite a lot of work to do … the right plot of slide 9 is not relevant …
motivate finding kernels that we use in the forward model … show the kernels we
use in both x and y for charge deposited on a strip and then the kernels for the
±1 and ±2 strips … this should lead naturally into a slide 10 which shows more
explicitly how we reconstruct the final waveforms to match data with charge
coming from neighbouring strips."*

**One new figure module, `mpgd26/make_share.py`, four figures**, all from the
frozen production bundle `calib_bundle_lp2_t0p` and from event 1663 of its
ref-pinned calibration cache — the same event the "One muon through the forward
fit" slide uses:

| figure | what it is | where |
|---|---|---|
| `share_cartoon` | the mechanism as a drawing: down the avalanche, **sideways through the film's own sheet resistance**, then down onto the strips | 9.1 |
| `share_kernels` | the kernels production uses, X above Y: own charge, ±1, ±2, at the model's own amplitudes | 9.2 |
| `share_build` | what the model does, four stages: depth slices → geometry → kernel → impulse response | 10, left |
| `share_decompose` | the same split on **real data**: four consecutive strips, each waveform stacked into own / ±1 / ±2 against the measurement | 10, right |

**The colour rule is the point.** Blue = a strip's own charge, vermillion = ±1,
purple = ±2, in all four figures and on both slides — asked for directly ("the
colors for different strip contributions are different"). A colour means one
thing across the section, which is what lets slide 10 be read without a legend
after slide 9 has taught it.

Slide 9 is a **two-frame build** (9.1 → 9.2): the mechanism, then the measured
kernels, so the motivation lands before the answer. The right column carries the
whole markup on both frames with `.fut` on the wrapper — blanking its fig-label
on frame 1 instead changes the grid row height by a line and moves the left
figure, which is exactly what the build mechanism exists to prevent.

### What came off, and why

- **`unsharing_depth_bias.png`** (old slide 9, right). Not relevant any more:
  "unsharing removes the depth-dependent angle bias" is a **hit-level repair**,
  a fix for the basis we abandoned on 2026-07-28. It spent a slide defending a
  chain the next two slides replace.
- **Two thirds of `charge_sharing_schematic.png`**. Panels (b) and (c) were the
  time-fit velocity-bias mechanism — the old basis again. Panel (a), the one
  Dylan said he liked, is redrawn as `share_cartoon` with the physical flow
  corrected: the first drawing sent the sideways arrows straight from the impact
  point to the neighbouring strips, which quietly says the charge is split at
  the moment it lands. It is not — the sideways trip through the resistance is
  what costs the time.
- **The old fig-label's numbers**: "~50 % of the avalanche charge is shared to
  the first neighbour (45 % X / 52 % Y, ~70 ns delay)". Those are hit-level
  neighbour-amplitude ratios from the June study. The **model's** numbers, which
  are what the fit runs on, are different — and quoting one set on the slide
  while fitting with the other is how a talk gets taken apart in the questions.
- **Slide 10's four paragraph bullets and its callout.** Each was a sentence the
  speaker says anyway, and the diagram now carries the same four stages. The two
  claims that lived *only* in the text are on the fig-labels: NNLS solves the
  charges exactly at every trial geometry, and v_drift enters once at the end.
- **`wft_depth_ladder.png`** left the main flow with them. Its content — one
  strip's pulse decomposed by DEPTH — is already the bottom-right panel of the
  "design matrix" backup slide, so nothing is lost; the asset is still built.

### ⚠️ Read before quoting an amplitude off slide 9.2

On the frozen bundle **c₁ = 0.051 sits ON the `C1_MIN` = 0.05 floor**
`wft.calibrate` imposes. A cosmic-angle fit cannot separate sharing from a wider
initial cloud plus a different v_drift (`WAVEFORM_FIRST_THREADING.md` §17.2), so
without the bound it walks c₁ to zero and hides the sharing in `sigma_p0` —
which is what happened on det7 (c₁ = 0.004, kY = 6.6). The **H4 beam test at
normal incidence** breaks the degeneracy and measures c₁ ≈ 0.28–0.30. So:

- the **shapes and the delays** on the figure are the model's own and are what
  the reconstruction runs with — no caveat needed;
- the **amplitudes are a lower bound** on this detector, not a measurement of
  the film's sharing. The slide says "calibrated on det3" and does not claim
  more;
- the **X/Y ratio** (kY = 2.9) is fitted and is the interesting number — and it
  has a clean physical reading: the film's strips run along y, so the charge
  spreads along y, and the Y view, which is pitched across that direction, sees
  ~3× the sharing X does;
- **c₂ > c₁** in this bundle. Do not tell a ladder story about it: the two are
  strongly correlated in the fit, and what the model needs right is their sum
  and their delay.

All of this is in `make_share.py`'s module docstring as well, which is where it
will be found by whoever regenerates the figures next.

## Micro-TPC operation: waveforms, and no type bands — 2026-08-17

Dylan: *"remove both the figure caption and footer to make the visualization
larger. On the right side, instead of the hit points fit to a line we should
have a version with the actual waveforms."*

- The right panel is now the **stacked per-strip waveforms** (`make_microtpc.py
  --right waveforms`, a mode the script already had). That is the honest order
  of the argument: the waveforms are what the DAQ records, the ladder is one
  estimator built on top of them — and it is the estimator the next two slides
  exist to replace.
- `compose(bare=True)` writes a deck copy with **no title band and no caption
  paragraph**. Together they were 36 % of the figure's height. The operating
  point is burned onto the render instead, in three short lines, kept short
  because the muon enters the frame at ~27 % of the panel width and a long line
  runs into it.
- Two bugs fixed on the way: the waveform panel's x-axis label was being drawn
  **on top of the colour bar** (it had been since the mode was written), and the
  window is now trimmed to the last sample above 2 % of the peak instead of
  showing 400 ns of dead baseline.
- **The slide's `.caption` said "882 ns full-gap transit" next to
  "v_drift = 36.6 µm/ns". Those two numbers are inconsistent** — 30 mm at
  36.6 µm/ns is 820 ns; 882 ns is the Magboltz 34 µm/ns. The caption is gone and
  the figure computes the number from v_drift, so it cannot drift out of step
  again. The same mix-up was in `make_report.py`'s prose and is fixed there too.

## The chamber's planes go deeper, and the board-peel view lost its title — 2026-08-17

Two more passes on slide 7, both Dylan's:

- **"Make the planes extend further back into the page, keeping the current
  perspective."** `WIN_MM` went `(60, 18)` → **`(60, 34)`** mm. At 18 mm the
  depth direction had barely more extent than the exploded gaps between the
  layers, so the eye read a stack of *edges*; at 34 mm the far edge of the stack
  reaches the top-right of the frame without the near corner of the laminate
  leaving the bottom of it. 48 mm clips. The camera is untouched, as asked.
- **"Remove the title in the python plot and put it into the slide manually.
  Also, let's remove the footer on it."** `plot_mx17_model.fig_peel` grew a
  `bare=True` mode (`--only peel_slide`) that drops the title band and the
  "deeper into the board" arrow and writes `mx17_board_peel_zoom_slide.png`;
  the deck copies it in as `mx17_board_peel_slide.png`. MX17_Geant's own design
  document keeps the titled figure.
- Both figures on the slide now carry a **`.fig-head`** — a new CSS class, a
  heading above a figure in the deck's own type. The rule from here: a figure
  that will go on a slide has **no burned-in matplotlib title**; the title is
  HTML, at the deck's size and weight, and the band it used to occupy goes to
  the picture.

## …and its readout side is now the as-built board — 2026-08-17 (same day)

Dylan's read of the landscape version: zoom in so the strip structure is
visible, give some width back to the board-peel view, and **"make sure we match
the most up to date info from MX17_Geant"**. The readout side was re-sourced
from `~/CLionProjects/MX17_Geant` — `shared/MX17ModuleGeometry.hh` (mirrored in
`scripts/model/mx17_model.py`) and the gerbers, which is the same header
`MX17_Full_Geant` builds from — and four things were wrong or missing:

| | was | now |
|---|---|---|
| **L4 pads** | not drawn at all | 0.68 mm square pads on the 0.78 mm grid — the layer the drift charge actually lands on (`meshes.pad_grid`, new) |
| **Resistive strips** | purple, on the 0.78 mm readout pitch | **black** (`#1c1c1c`), 550 µm wide with 250 µm gaps → **its own 0.80 mm pitch**, which is why the pads show through in the board-peel figure |
| **Layer order** | X strips above Y | the board's: film → L4 pads → **L5 Y strips (along x)** → **L6 X strips (along y)**. A "Y strip" *measures* y, so it runs along x |
| **Readout PCB** | 7 mm of drawn slab under 1.6 mm strip layers | 2 mm drawn and labelled **1.70 mm**, which is the whole board |

Strip copper is 0.5 mm on the 0.78 mm pitch (gerber), not the 0.56 mm the figure
had been using, and the four readout layers now carry the **colours and the L-
numbers of the board-peel figure beside them**, so a colour means the same layer
in both pictures on that slide. The gas label lost its mixture ratio — "30 mm,
Argon/Isobutane".

**The zoom, the explode spacing and the frame shape are one setting.** The
window went 120 × 30 → **60 × 18 mm** (~77 strips on screen instead of 154, i.e.
twice the pixels per strip) and `EXPLODE` went 19 → **7.5 mm** with it: the
drawn stack height and the window width set the figure's aspect between them, so
zooming in without closing the gaps turns the landscape figure straight back
into a portrait one. The frame is 2400 × 1980 and the slide columns went 63/37 →
**56/44**, which is the width the board-peel view got back.

The one drawn thickness that is honest is the 30 mm drift gap, and it now takes
about half the height — that is the detector, not a drawing error. Everything in
the plane (pitch, width, pad size, gaps) is real.

## The exploded chamber is landscape — 2026-08-17

**Slide 7 "Chamber design" now gives the exploded render 63 % of the width, and
the render was rebuilt to want it** (Dylan: "the form factor needs to be
different … a rectangular subset of the detector to use the width of the
screen"). Three changes, and they only work together:

1. **The window on the chamber is a rectangle, not a square** —
   `scenes_chamber.WIN_MM = (120, 30)` mm, still at the real 0.78 mm pitch, so
   the layers run the width of the frame. `build()` takes an `(x, y)` pair (a
   scalar still works), and the strip count is now derived per direction from
   the extent the strips are pitched across, so the pitch on the page stays real
   in both views: 154 strips across the length for the Y view, 38 across the
   depth for X. The mesh keeps a fixed drawn pitch, so a wider window gets *more*
   mesh bars rather than stretched ones.
2. **The labels are on the render, down its left side** (`annotate.side_labels`,
   the treatment the EAR2 and target figures already use) instead of in a gutter
   column to the right of it. The anchors moved to each layer's **left** front
   corner — anchoring right, as the portrait version did, would have sent every
   leader across the picture. The label text was cut at the same time ("Drift
   cathode (mylar)" → "Drift cathode", "(spark protection, charge spreading)" →
   "charge spreading"): the text now sits *inside* the frame, so every character
   of it is width taken away from the picture.
3. **The frame is landscape** (2400 × 1740) and the camera's focal point is
   pushed to −x so the stack sits right of centre. The empty left band is the
   layout, not slack — it is where the labels go.

**Net effect: the layers are ~2.5× bigger on the slide.** The old figure was
portrait, so on a slide that gives it width and not height it was height-limited
and then spent a third of what was left on the label gutter.

Two things this touched that are easy to miss:

- **`make_anim`'s `turn_chamber`** would have spun a wide slab in a portrait
  frame around an off-centre focal point. It now uses `make_chamber.ANIM_VIEW` /
  `ANIM_SIZE` — centred focal point, landscape frame.
- **The slide's column weights and the figure's aspect have to change together**
  (`grid-template-columns:0.63fr 0.37fr`). Either one alone puts a white band
  back. The board-peel view on the right is square and unchanged.

The report keeps the fully titled and captioned version; the deck copy
(`chamber_exploded_slide` → `slides/assets/img/chamber_exploded.png`) drops the
title/subtitle/caption bands, because the slide has its own title and the
caption is a thing to say.

## The spectrum became a stack, and slide 7 went to backup — 2026-08-17

Two edits from the same read of the new slide 6.

**Slide 6.2's spectrum is now a stack.** It used to be the two channels
overlaid, each normalised to unit peak — the honest way to compare two *shapes*,
but not what a measurement looks like. It now draws the IPC background with a
small X17 yield sitting on top of it, the filled area between the curves being
the excess.

- The ratio is the thing the experiment sets out to measure, so the figure must
  not appear to know it. It is a declared parameter — `scenes_x17.SIG_FRAC`,
  **4 % of the IPC yield over the plotted window** — and the panel says so in
  words, in muted type, under the plot: *"X17 drawn at 4 % of the IPC yield —
  illustrative: that ratio is what we set out to measure."* The `.fig-label` and
  the compilation's footer caption say it too.
- 4 % puts the bump **~80 % above the local background** at its peak. Picked by
  eye against 2 % (too subtle from the back of a room) and 6 %.
- **The window now starts at 40°** (`SPEC_XLIM`), which is why the bump is
  legible at all: the IPC forward peak is **eight times** the yield at 109°, so
  including it flattens everything the panel is about. This is what ATOMKI do
  and for the same reason. The forward sweep is beat 4's argument, made there as
  kinematics — the spectrum does not have to make it a second time in a form
  nobody can read.
- **`x17_signature` was deliberately left alone** — the compact panel is still
  the unit-peak shape comparison, and it is the one place in the package where
  the two channels are drawn at equal area. If you ever put it back in the deck,
  the two figures now say different things on purpose.

Everything downstream of beat 5 was regenerated: `--layout both --capsule` for
`figures/`, then `--layout build --capsule --slides` for the deck copies. The
two bottom frames are still 4800 × 1224 apiece, so the build still does not
jump.

**Slide 7 went to backup** ("kick slide 7 to backup"), which closes the
redundancy the move created the day before. "What the detector has to separate"
(`ideal_pair_spectrum.png`) is now **Backup · X17 kinematics**, sitting next to
the two ATOMKI backup slides. The main flow states the 109° bound once, on
slide 6. In backup the slide earns its place twice over: its left panel
(single-lepton kinetic energy) is made nowhere else in the deck, and its caption
is the honest one about real capsule material degrading the edge — which is
exactly the question it will be asked. **The deck is still 77 slides in 84
pages**; only the main-flow/backup split moved.

## …and then moved to slide 5, as two builds — 2026-08-17 (same day)

**The two ³He-story slides are now slides 5 and 6, and each is a build.** They
replaced "n + ³He → the n_TOF search channel" (the compact `x17_signature.png`
slide), which is gone; the setup section no longer carries the story at all.
**The deck is 77 slides in 84 pages.**

- **5.1 → 5.3** beam on the vessel → capture → the level scheme and the three
  channels. **5.3 ends on "Detect the e⁺e⁻ pair!"** — Dylan's reason for the
  move: that line is the hand-over into the Micromegas half of the talk.
- **6.1 → 6.2** the boost, then the spectrum it produces.

**The frames come from `scenes_x17`, not from CSS or hand-made crops.**
`draw_story` gained `upto=N`: it draws the row on its own full canvas with only
the first N beats on it, so the frames are strict subsets of one picture (all
three top frames 4800 × 1050, both bottom frames 4800 × 1224) and a beat lands
in its final position the moment it appears. One command:

```
../.venv/bin/python make_x17.py --layout build --capsule --slides
```

The `.fig-label` is identical on every frame of a slide, for the same reason
the EAR2 build's bullets are — re-wording it between frames moves it.

**⚠️ This creates a redundancy downstream, and it is Dylan's to resolve:** the
next slide, "What the detector has to separate" (`ideal_pair_spectrum.png`),
makes slide 6's argument again from Geant4 truth. Two slides now state that the
opening angle is bounded below at 109°. `RUNNING_ORDER.md` says show one, not
both; nothing has been deleted. — **Resolved the same day: that slide went to
backup, see the entry above.**

`x17_signature.png` joins `x17_story_capsule.png` as an asset no slide
references any more. Both stay in `assets/img/` and in the report.

## Slide 16 split into two — 2026-08-17

**The five-beat ³He story is now two slides instead of one** (Dylan asked to
explore it): "Capture on ³He, and the three ways ⁴He\* comes down" (beats 1–3)
and "The opening angle is set by the boost" (beats 4–5). **The deck is 78
slides in 82 pages.**

**Nothing was redrawn.** `scenes_x17` has had `STORY_PARTS` 'top'/'bottom'
since the figure was written — the two rows are the same drawing seen through
different canvas bands, and `make_report.py` already documented the split as an
option. The deck simply started using it. The assets are the `--no-title`
variants, since the slide's title bar now carries what each row's headline said:

```
../.venv/bin/python make_x17.py --layout story1 --capsule --no-title --slides
../.venv/bin/python make_x17.py --layout story2 --no-title --slides
```

`--slides` is new (2026-08-17): `make_x17.py` now writes the deck copy into
`slides/assets/img/` itself, the way `make_ear2`/`make_target`/`make_timeline`
do, instead of the hand copy that put `x17_story_capsule.png` there in August.

**Known cost, measured, not a bug to fix in CSS: each row fills only ~55–60 %
of the slide height.** The rows are 4.6:1 and 3.9:1 against a ~2.3:1 content
box, so the figure is width-bound and the band below it is empty. Rearranging
the beats *within* a slide does not help — two stacked rows halve the height
available per row and hand back what the extra width buys (every 2-row
arrangement scores ×1.0 or worse; `1 over 2` helps the first beat ×1.22 and
costs the other two ×0.50). The only real lever is a **narrower set of beats**
— beat 4's five orientation columns are what make that row so wide — and that
is a `scenes_x17` change and a content decision, not a layout tweak.

`x17_story_capsule.png` is no longer referenced by any slide; it stays in
`assets/img/` and in the report, and is what to go back to if the split is
rejected (the HTML comment above the two slides says how).

## The edit queue, and the two ATOMKI slides merged — 2026-08-16

**`SLIDE_EDITS_TODO.md` is new and is now the document deck edits work from.**
It holds the queue and, at the top, the principle Dylan stated for it: a physics
talk speaks with its figures, so **slide text comes down hard**. NOTES.md stays
what it is — the log of what was done and where every figure came from; the
to-do file is what is still to do.

**First edit under that principle: the two motivation slides are one.** Pages 3
("A possible new boson at 17 MeV" — three bullets and a `17 MeV` stat tile) and
4 ("The ATOMKI evidence" — `atomki_angular_correlations.png` under a five-line
caption) are now a single page 3, with the three Krasznahorkay references as one
muted `.fig-label` line. **The deck is 81 pages.**

**Layout, after Dylan's read the same day: text left, figure right**
(`cols-2 figtext`, 0.74fr / 1.26fr), not bullets-over-figure as first built.

- The caption's mirror-channel sentence was the only load-bearing prose on the
  old page 4 — it is the bridge into the n_TOF channel slide — so it survives in
  the third bullet, which Dylan reworded into the question the talk actually
  asks: "**Can we make an independent measurement** — the same ⁴He* state,
  entered as n + ³He?". Its full original wording is preserved in the HTML
  comment above the slide.
- Bullet 1 carries the **years, 2016–2022** (Dylan asked for the experiment's
  year; the three panels span three papers, so it is a range — if he meant the
  ⁸Be discovery alone it should read 2016).
- Dropped: the stat tile (the title says 17 MeV) and the prose about the fits
  and the IPC background — that is a sentence to *speak* over the figure.
- Each bullet is capped at ~2 lines in the narrower column; the figure is
  1474 × 864 with small axis labels, so anything longer starts costing it.
- **This was already the plan** — `RUNNING_ORDER.md`'s row 2 has proposed this
  merge since 2026-08-10. It is now done in the deck, not just in the cut, and
  that row has been updated to say so.
- Still open from the 2026-08-08 comment that sat above these slides: the
  ATOMKI beam orientation is still unsourced (deliberately not asserted), and
  redrawing the borrowed spectrometer schematic in the house style is still an
  idea. Both stay with the backup ATOMKI slides.

## The EAR2 build, and overlay builds in general — 2026-08-16

**The five EAR2 frames are now one slide, numbered 4.1–4.5**, and their text is
cut to **six short lines and no callout** (seven lines and a callout on the
first pass; Dylan cut it again on reading it). Two reusable mechanisms went in
with it; both are documented in the `<style>` block and queued for the setup
build in `SLIDE_EDITS_TODO.md`.

**The γ-flash hand-off is deliberately not on the slide.** It was a callout on
frame 5 — "the flux we came for is also the problem" — and Dylan removed it:
*not time for this yet*. It is the bridge into the Status section, so it is now
something to **say**, not something the slide prints. **Frame 4 therefore adds
no new line** (its render adds the upper pipe and the dump; the bullet that
said so is gone), which is fine for a build and is stated in its comment.

**Text that does not move.** Every frame carries the whole bullet list; the
lines not yet narrated are `.fut` (`visibility:hidden` — hidden but still
holding their space) and the narrated ones `.dim`. So each line sits, from
frame 1, exactly where it will sit on frame 5, and the build only ever adds
below. **The bullet strings are therefore identical on all five frames** — the
old version re-worded and shortened them as they piled up, which is precisely
what makes text jump. Edit one, edit all five. `.side.top` top-aligns the
column so the first line starts at the top rather than centred.

**A build spends one slide number.** First frame `slide bstart`, the rest
`slide bcont`, each with `data-frame="n"`. The on-screen counter and the
progress bar count the build once (the script at the foot of `index.html`), and
`make_pdf.sh` injects the matching `counter-reset` so the print agrees. **The
deck is 77 slides in 81 printed pages.**

- The frame number is an *attribute*, not a second CSS counter, and that is not
  a style preference: `counter-reset: frame 1` on the first frame does not
  reach its following siblings once `.deck` has reset the same counter, so
  Chrome numbered the frames 1,1,2,3,4 in the printed footer while the
  JS counter said 1–5. `attr()` has no scope to get wrong.
- A CSS comment closed early during this edit and silently killed the rule
  after it (the numbers came out 4.1, 5.2, 6.3 …). If the numbering looks
  half-applied, check for a stray `*/` before assuming a specificity problem.

**What the text cut removed:** 7.4 m, 15.0–18.0 m, 70 → 21.8 mm, 18.16 m,
23.66 m, 24.73 m, and the materials. **Every one of them is already labelled on
the render** — the bullets were reading the picture out loud. The sourced
numbers themselves have not changed and the backup slide still carries the
full provenance. The text column went 1.04fr → 0.80fr, the render took the
width.

## Two new main-flow slides — 2026-08-15 (Dylan asked for both)

**A project timeline and a post-LS3 outlook slide are now in the main flow**,
not in the proposed block below. **No page number is quoted here on purpose** —
every structural edit renumbers the deck. Find them by the `index.html` comments
`17pre: T1 project timeline` and `27b: T2 post-LS3`, which do not move.

- **`How we got here`** opens the Status section, immediately before
  D0. One figure, `assets/img/project_timeline.png`, from the new
  **`mpgd26/make_timeline.py`** (`../.venv/bin/python make_timeline.py`; it also
  writes a `.pdf` beside it, which is the one to open if you need to read the
  small type). The figure's spine is a true date axis, the panel strip under it
  is five equal columns — done that way because the campaigns are weeks apart
  and their descriptions are paragraphs, so nothing readable fits beside a bar.
  **The script's docstring names the source of every count on the figure**; the
  slide caption quotes the same numbers, so the two have to move together.
  The window starts at **November 2025, when Dylan joined the project** — this
  is deliberately not the whole history of the experiment.
- **`What a post-LS3 measurement needs`** sits between D10 ("What we
  take away") and the Summary. Everything on it is quoted from
  `ntof_run_report/make_report.py` §10 and the packages that section cites.

**Both slides predate `SLIDE_EDITS_TODO.md` and do not obey it.** The timeline
carries a four-line caption and the post-LS3 slide carries three bullets, a
two-row table and a callout — under the cut-the-text principle both are
candidates for the standing text-reduction sweep. The timeline's caption is the
easier cut: the figure's own panel strip already says most of it, and what is
left ("the flash was there on the first day; every campaign since was a
response to it") is one line, or is spoken.

**What is deliberately NOT on the post-LS3 slide:** the Asimov significance
projection in `MX17_Full_Geant/docs/slides/HANDOFF_SLIDES.md` (4.9σ stat-only
for a 30-day post-LS3 run in 0.2–2 MeV; 6.4σ after the nose-first geometry
fix). That handoff's own caveat is that the IPC angular-*shape* systematic, not
statistics, will set the real CL, and the deck's 2026-08-10 framing decision is
that this talk makes no claim about physics reach. **If Dylan reverses that
decision, the number goes on this slide with its caveat and nowhere else** —
the HTML comment above the slide says so too.

**Three stale "still running" claims were corrected at the same time**, because
the timeline slide states that the beam came off on 10 August and they
contradicted it: D0's first bullet and its `flag` line, D6's first stat tile
(was "~1.7 TB … still running", now the campaign's 17.9 TB), and the Summary's
third bullet. This is the edit NOTES flagged as required on 2026-08-13.

**Not yet done:** neither slide is in `RUNNING_ORDER.md`'s 16-slide cut — see
the note added there the same day.

## Proposed results slides — 2026-08-13 (drafted by Claude for Dylan's review)

**Nine new sections sit at the very end of the deck behind a "Proposed · 13 Aug"
divider** (printed pages 72–80): a campaign-statistics slide (P1) + a
beam-availability backup (P2), a thermal-trigger slide (P3), a merged
γ-flash-charge slide (P4, a D2+D5 one-slider for the 15-minute cut), the
run_145 capsule image (P5) and four-arm pointing (P6), a "from tracks to
pairs" outlook (P7), and an updated Summary (P8). **They are options, not
adopted** — the main flow and backup above them are untouched. Each slide's
HTML comment says where it would slot and what it would absorb; the companion
review note (`mpgd26-deck-review` on the notes site) has the full argument and
a proposed revised 16-slide running order. New figures were copied into
`assets/img/` as `campaign_{events,beam}.png` (from `ntof_run_report/figures/`),
`trigger_data_vs_sim.png` (ibid, = the 30_trigger emulation-vs-Geant4 figure),
`run145_{image,pointing}.png` (ibid, from the run_145 note),
`run145_wall3d_all.png` and `doubletrack_ev1054_3d.png` (from
`/media/dylan/data/x17/…` — paths in the slide comments). **The one edit these
force even if none are adopted: D0, D6 and the old Summary still say the
campaign is running; data taking ended 2026-08-10.** To drop the whole block,
delete everything between the "PROPOSED ADDITIONS" comment and the click-zone
divs, and the seven new images if unused. `mpgd26_talk_draft.pdf` regenerated
at 80 pages; all nine new pages QA'd with pdftoppm.

MPGD2026 talk sketch, built as a single self-contained `index.html` (no build
step, no external network dependency — open it directly in a browser).

**View it:** open `index.html` in any browser, or `index.html#9` to jump to
slide 9. Arrow keys / Space / click the left-right edges to navigate.

**PDF backup:** `./make_pdf.sh` regenerates `mpgd26_talk_draft.pdf`. Don't use
the browser's own File → Print → Save as PDF — Chrome's headless print
pipeline blanks whichever slide is first whenever the output has more than one
page (confirmed positional, not content-specific, by swapping slide order).
`make_pdf.sh` sidesteps it by printing each slide to its own single-page PDF
and merging with `pdfunite`.

**Site mirror:** the deck is published as a self-contained note at
<https://dylan-neff.web.cern.ch/notes/mpgd26-talk-slides.html> (snapshot of
2026-08-13; images downscaled to ≤1600 px and WebP-inlined, 55 MB → 9.5 MB;
linked from the notes listing and the hub, deliberately *not* from the x17
page). **It does not update itself** — after editing the deck, refresh it with
`mpgd26/tools/mirror_slides_to_site.py` (usage in its docstring; `--force`
keeps the URL and the original listing date).

## The reconstruction-algorithm slides — 2026-08-13

**Ten new slides translate `docs/wft_reference/` (the waveform-first forward-fit
reference document) into the deck**: two in the main flow right after "Resistive
strips share charge" (9b "The forward fit: predict every waveform, never invert
one" — mechanism; 9c "One muon through the forward fit" — the proof on one
event), and eight `Backup · Reconstruction` slides ordered as the questions
come: hit times → design matrix/NNLS → the 3-D search → seeding/candidates →
calibration & the (kernel, v) degeneracy → "why not deconvolution" → the 1°
physics floor → the failure gallery.

**At 15 minutes show only 9c** and narrate the mechanism over it; 9b is for the
20-minute version (see RUNNING_ORDER.md, updated the same day).

**Figure provenance.** All `assets/img/wft_*.png` except the two deconvolution
figures were regenerated 2026-08-13 from `docs/wft_reference/figsrc/` against
the **frozen production products** (`calib_bundle_lp2_t0p` + `events.parquet`,
sat_det3) — the document's original spine (`calib_bundle_lp_sp0free` +
`events_lp.parquet`) no longer exists on disk after the campaign, and
`figsrc/wftdoc.py` / `f_valid.py` were re-pointed accordingly (committed, with a
comment). Regeneration:

```bash
cd docs/wft_reference/figsrc
WFT_DOC_FIGDIR=<dir> ../../../.venv/bin/python f_<set>.py   # f_hits f_model f_fit f_seed f_calib f_valid f_gallery
cp <dir>/<name>.png mpgd26/slides/assets/img/wft_<name>.png
```

The two deconvolution figures are the 2026-08-12 explain set
(`mx_june_wft/15_explain_figures.py`, numbers in `EXPLAIN_2026-08-12.md`),
copied from `<Analysis>/sat_det3/wft/explain/`.

Things to know before touching these slides:

- **`f_fit.py` crashes after its first four figures** ("too many values to
  unpack", 2026-08-13, under the re-pointed paths) — `errors` / `saturation` /
  `timing` did not regenerate. None are used on a slide. Fix before any future
  full-document rebuild.
- **`kernels.png` is deliberately not used**: its "lp — the one in use" panel
  title contradicts the freeze discovery that production loads the *delay*
  branch (`share_mode: null`, FREEZE_MPGD26_2026-08-12.md §2). The slides
  describe the kernel structurally ("measured sharing kernel") and take no side.
- **Numbers were re-read from the regenerated figures, not copied from the
  document**, because the frozen bundle differs from the doc's spine: σ_θ
  1.15°/1.14° and implied-v spreads are off `wft_angles.png` /
  `wft_implied_v.png` as regenerated; the degeneracy slide describes the
  *current* map (valley minimum at v ≈ 40.6 vs the pinned 36.6). The doc's
  own quoted hyper values (c₁ = 0.306 etc.) belong to the superseded bundle —
  do not re-import them onto slides.
- The 13–15 % ladder compression on the backup slide vs 20–30 % on main slide 9
  is not a contradiction: best offline estimators vs the production threshold
  time on zero-suppressed waveforms. The backup caption says so.

## Design reboot — 2026-08-12

**The deck was restyled wholesale for a scientific-conference aesthetic** (Dylan:
the tech look was wrong for the room, and HTML is risky to hand a conference —
so the **PDF from `make_pdf.sh` is now the deliverable**, with the HTML as the
authoring/presenting format). Slide *content* and *structure* are untouched: the
whole look lives in the `<style>` block, and the only markup edits were the
Backup divider (now `class="slide divider"`), a `dense` class on four
overstuffed backup slides (see below), and two column-ratio tweaks on the target
slides.

**The theme is "Modern", chosen by Dylan the same day from four rendered
options** (journal didone serif + mx17 purple / Times "preprint" + navy /
modern all-sans / slate banded-header TDR). The comparison snapshots are in
`theme_preview/` (`compare.html` + 20 screenshots) — a record of the decision,
regenerable from git history, delete at will. The three variant decks themselves
were deleted after the choice was merged into `index.html`. What the system is
now:

- **All-sans, Metropolis-inspired**: Noto Sans Display titles (bold, dark rule
  under), Noto Sans body — **local system fonts**, so the file stays
  offline-self-contained. White paper, dark teal-black ink (`#23373b`), **one
  orange accent** (`#eb811b`), hairline rules, booktabs-style `spec-table`s, a
  running footer (`D. Neff · MPGD 2026, Prague · …` + slide number) on every
  slide but the title. The copper caution accent survives; the figures' mx17
  purple now reads as a *figure* colour rather than a chrome colour, which
  keeps the two layers distinguishable.
- **Slide numbers are a CSS counter**, and `make_pdf.sh` injects a per-slide
  `counter-reset` so they survive its one-page-at-a-time printing. If you add or
  reorder slides, numbers update themselves in both paths.
- **Photographs get a hairline frame automatically** (`img[src$=".jpg"]`);
  renders and plots sit borderless on the paper. Don't re-add panel boxes — a
  height-limited figure in a bordered box reads as dead space.
- ⚠️ **`class="slide dense"`** shrinks `spec-table`/`callout`/`bullets`/caption
  on the four reference slides that carry more than a page of content (the EAR2
  documentation slide, both Target #3 slides, and "Why that is a measurement…").
  **The two target tables overflowed the page in the old design too** — checked
  against git HEAD before fixing, so this was a pre-existing fit bug, not a
  regression. If a new slide overflows, `dense` is the intended lever.
- The old body font stack resolved to Liberation Sans, which is ~7 % narrower
  than Noto Sans — that is why the densest slides needed refitting, and why
  eyeballing "it fit before" is not a check after any font change. The full-deck
  QA loop is: `./make_pdf.sh`, render pages with `pdftoppm`, look.

`mpgd26_talk_draft.pdf` regenerated on the new design, 56 pages.

## X17 theory backup slides — 2026-08-12

**Three theory backup slides added** (pages 50–52, right after the two ATOMKI
backup slides): (1) the three anomalies + spin/parity argument, (2) why
protophobic + the surviving ε windows, (3) the case against + who tests next.
Dylan asked for "the Feng paper" — that is Feng et al., **UC Irvine** (not
Berkeley), arXiv:1604.07411/1608.03591, and the constraint experiment is
**NA64** (not NA63); both corrected. Every number was verified against arXiv
abstracts/full texts on 2026-08-12; the four things that could *not* be
verified (MEG-II's final journal ref, PADME's point limit, the 6.8σ vs >5σ
attribution, the obsolete muon-g−2 claim) are listed in the HTML comment above
the slides and deliberately kept off them. They do not use `dense` — they were
tried with it and came out under-filled. Not in the 15-minute cut; if only one
goes up in questions, it is the middle one ("how is this not already
excluded?").

**Two more theory slides on 2026-08-12** (Dylan: "epsilon explained clearly in
general" + "visualizations for these constraints"): a "What ε is" explainer and
"The coupling windows, drawn", both figures from a new `mpgd26/make_couplings.py`
(the only writer of `assets/img/x17_{epsilon,couplings}.png`). The chart repeats
the table slide's verified numbers — the script's docstring says to change both
together. Presenting order inside the block: anomalies → ε explainer →
protophobic table → windows chart → case-against.

Current: **61 slides** (title, outline, motivation incl. the **5-frame EAR2
beam-line build-up, 2026-08-11**, detectors, **n_TOF setup — the 9-frame 3-D
build-up, 2026-08-10**, **Status — 11 slides, drafted 2026-08-09**, summary,
backup incl. the two imon teaching slides, **3 X17-theory slides (2026-08-12)**,
3 status backups, the old measured-drawing setup slide and the **EAR2 render's
documentation slide**).

**The Status section is deliberately over-built.** It is written as a menu to
cut from, not as a running order — see [`STATUS_PLAN.md`](STATUS_PLAN.md) for
what each slide is for. Its own "load-bearing if time is tight" set was D1, D4,
D6, D8, D9; that has since been superseded — **D9 was removed by decision**
(no yield claim, see below) and [`RUNNING_ORDER.md`](RUNNING_ORDER.md) is now the
authority on what to show in the 15-minute slot.

## Where this stands — 2026-08-10

**The slot is 15 min + 5 min**, Prague, 3 September 2026, speaker Dylan Neff.
The deck is far over length by design; [`RUNNING_ORDER.md`](RUNNING_ORDER.md)
holds the proposed cut. Read that first.

**Done this pass:** title byline · ⁴He-channel bridge on the evidence slide ·
D9 removed and the section reframed as a status report · efficiency numbers and
figures refreshed (three chambers moved 14–21 points — the old values were wrong,
not merely stale) · the spatial-resolution caveat withdrawn as wrong-premised and
replaced with a deconvolved number · the ideal e⁺e⁻ kinematics figure built from
Geant4 truth · the zoomed board-peel figure · the strip pitch corrected package-wide
(0.7785 → 0.78 mm) · the EAR2 beam line rendered in house, EAR1 dropped, and the
beam-dump identification verified against the literature.

**Also done, second pass, all from Dylan's review:** the EAR2 render split into a
build-up with his geometry changes (pillars gone, capsule floating) · the
board-peel strip layers redrawn as **strips with the vias suppressed** and the
pillars off the resist band · and **the imon systematic MEASURED AND CLOSED**,
with a two-slide teaching backup.

**Also done, third pass, 2026-08-11, from Dylan's review of the EAR2 render:**
**the beam pipe now really ends ~1 m above the EAR2 floor**, in a flanged
circular end, with the neutrons crossing the hall in air. That was a **factual
correction, not a styling change**: the previous render ran the pipe to the dump
and then cut it away as a drawing device, and the pipe does not run there. His
question *"where is the first collimator?"* is answered on frame 2 and in the
facility section below — **7.4–8.4 m above the target, ~10 m below the EAR2
floor**, 1 m of iron with a 200 mm bore, invisible in the render because it falls
inside the break.

**Fourth pass, 2026-08-11, same review round:** **the ceiling and the beam dump
came out of the drawing entirely** and the upper break went with them, on Dylan's
"cut off the full ceiling beam dump to make more space on top and put more
emphasis on the measuring station". `DRAWN_H` 8.67 m → 5.71 m, so everything left
is drawn **~1.5× larger**; the beam now leaves the top of the frame under a label
carrying the dump's height, and what is above the frame is stated in
`ASSUMPTIONS` and in the slide caption. The build went **back to three frames** —
the dump had its own frame for a few hours and then had nothing left to reveal.
**And the CERN photograph of the hall is back**, cropped to its right-hand panel
and set beside the render (`ear2_hall_photo.jpg`). ⚠️ It shows the vertical line
**closed from the floor to the ceiling** with detectors on a chamber in the beam
— a different configuration from the one the render draws. The backup slide's
caption says exactly that and claims nothing about which section is removable.
See "Left:" for the open question. (The photograph has since been promoted from
illustration to **source**: the shape of the line inside the hall is scaled off
it — see the fifth pass below.)

**The imon result is the big one.** The CAEN readback is a ~1 s averager, so
`mean − median` really is the time-average current: **the deck now says "142 nC"
with no inequality and no correction factor**, and four independent estimators of
the charge agree to ±3 %. Every lower-bound hedge is gone from the slides. Detail:
[`HANDOFF_imon.md`](HANDOFF_imon.md) and §8 of
`ntof_july_analysis/flash_charge/HANDOFF_FLASH_CHARGE_2026-08-09.md`.

**Fifth pass, 2026-08-11, same review round — the figure got more real and the
slide got taller.** Six changes:

1. **The facility slides carry no caption.** Both pictures on them are portrait,
   so height is the scarce resource, and the six-line caption was costing them a
   fifth of it. It moved, in full, onto **one new backup slide** under the same
   figure ("The EAR2 render, and what in it is drawn rather than measured"). The
   main slides keep a one-line `.fig-label` credit under each picture, which is
   the minimum attribution and is not a caption. **The photograph is now on the
   left**, the render in the middle, the text on the right.
2. **The column widths are proportional to the two images' aspect ratios**
   (0.62 / 1.00 / 1.04 for photo / render / text). That is not cosmetic: get it
   wrong and both images are *width*-limited, stop growing, and leave a band of
   dead space under them — which is exactly what the first caption-free attempt
   looked like. Both pictures now run 505 px of a 720 px frame.
3. **The slides use the on-figure-labelled render** (`ear2_onfig_<n>_<tag>.png`,
   labels on the drawing's own background, left *and* right) instead of the
   gutter variant. Dylan asked for that variant to compare; with no caption to
   pay for, a gutter is dead width, so it won. Both variants are built by every
   `make_ear2.py` run and swapping is three `img src` edits. The **backup slide
   keeps the gutter variant** — under a caption the figure is short, and
   on-figure labels are the first thing to stop being readable at that size.
4. **The line inside the hall is modelled from the photograph**: the white
   segmented polyethylene shielding disc and collar on the floor, the lead-disk
   vacuum chamber at the shaft bore, a **reducer** down to a narrow tube, and the
   flanged exit window. ⚠️ Radii are scaled off the photo against its ~1.2 m
   white disc and the step heights are chosen to fit — a reading of a photograph,
   not of a drawing, and `ASSUMPTIONS` says so in those words.
5. **There is a pipe above the experimental space again — a different pipe.**
   Dylan: *"the top pipe toward the beam dump included and cut off at some
   point."* So `add_uppipe` draws the wide section from `H_UP0 = 20.45` m up
   through the top of the frame, cut off by the camera margin rather than capped.
   The line now reads the way the photograph reads: narrow tube → flanged end →
   **open space with the experiment in it** → wide tube away. Drawn at 160 mm
   bore rather than the photo's own proportions, because at full size it
   *outweighed the measuring station*, which is the opposite of what removing the
   beam dump was for. `DRAWN_H` 5.71 → 5.91 m.
6. **"What is this pink diagonal thing in the collimator?"** — it was the
   *break's own section marker*, `COL['section']` on a 24°-tilted disc at the
   collimator's 680 mm radius, i.e. a third of it hanging outside the block. It
   was never hardware. Fixed by drawing that one face **flat and inset**, at the
   block's own bottom rather than 120 mm up inside it, and desaturating
   `COL['section']`. The tilt survives on the pipe's face below, where the disc
   stays inside its own silhouette.

**Sixth pass, 2026-08-11 — the build got finer and the station got rebuilt.** All
from Dylan's review of the fifth:

1. **Five frames, not three.** Target → **the neutrons leave at 90°** → the whole
   middle of the line (second collimator, floor and its shielding, lead disks,
   the pipe *ending*) → **back into a pipe, on up to the dump** → the measuring
   station. That needed `pipe` split into `pipe_lo`/`pipe_hi` and `neutrons` into
   `neutrons_lo`/`neutrons_hi`, so each frame can show the shaft without the hall
   and the raw flux without the collimated pencil. Deck 49 → **51 slides**,
   main flow 30 → **32**.
2. **The chambers are chambers now.** Dylan: *"I'm also not sure what the green is
   supposed to be there"* — fair, a 400 mm square of PCB colour seen obliquely is
   a green box. Each is built with `meshes.rect_chamber`: aluminium frame,
   readout board **with strips on it**, 30 mm of lit drift gas facing the sample,
   entrance window as a tint. ⚠️ **24 strips are drawn, not 512** — the real
   0.78 mm pitch is 0.26 px at this scale, i.e. a flat grey wash; declared in
   `ASSUMPTIONS`.
3. **Frame 5 exists in two versions, because Dylan asked to compare**: all four
   chambers (`ear2_onfig_5_station_4arm.png` — two solid, two the camera looks
   through drawn as bare frames) or two in section
   (`ear2_onfig_5_station.png`). `STATION_ARMS` picks; `make_ear2.py` writes both
   every run. **The section pair is the default and is on the slide** — see the
   seventh pass. ⚠️ In the four-arm version the near two must be drawn **uncut**:
   the cutaway plane passes through the beam axis and would *delete* them, not
   fade them, which is why the first four-arm attempt came out pixel-identical to
   the two-arm one. `frame_ring` is a hollow band, so drawing them solid frames
   the sample instead of hiding it.
4. **"Measuring station" became two labels**, `sample` and `detectors` — the thing
   the beam hits and the things that watch it are different objects and the slide
   should not make the audience work that out.
5. **The sample is centred between the two pipes.** `H_UP0` 20.45 → **20.74 m**,
   which puts 19.95 m exactly halfway between the end of the lower pipe (19.16 m)
   and the start of the upper one, and the upper pipe now shows **only its
   entrance flange and one ring** before the frame cuts it. `DRAWN_H` 6.11 m.
6. **The beam is faded across the sample** (`add_neutrons_hi(fade_sample=True)`,
   frame 5 only). At full strength it is a grey line down the middle of a
   translucent capsule and reads as the capsule being *behind* the beam. ⚠️ This
   is the **one exception** to "every frame is a strict subset of one picture" —
   documented in `scenes_ear2.py`'s docstring and in the slide comment.
7. **`Lead target` moved to the right column** (Dylan): its anchor is on the beam
   axis and `protons` comes in from the left, so with both on the left their
   leader lines crossed right under the target.

**Seventh pass, 2026-08-11 — legibility, all of it from Dylan looking at the
renders.** No slide was added or removed; every change is inside the drawing.

1. **The neutrons leaving the target are darker.** `COL['neutron']`
   `#8b96a3` → **`#55657a`**, and the five arrows are drawn at 12 mm instead of
   9 mm. Dylan could not find them: the old grey is the same *value* as the
   amber-tinted pipe interior it has to be seen against. ⚠️ The obvious fix —
   making them white — is the wrong one, and was rejected on purpose: the same
   neutrons cross the open hall higher up on a near-white background, where white
   would lose them completely. A darker slate is the only single colour that
   works on both of this figure's backgrounds.
2. **One opacity for the beam envelope, above the collimator and below it**
   (`ENV_ALPHA = 0.16`). It was 0.17 below and **0.34** above, on the reasoning
   that a 22 mm pencil needs more tint than a 317 mm cone. What that produced was
   two different-*coloured* beams in one figure — the same amber at twice the
   opacity, on a thin tube seen through both its walls, comes out olive-brown,
   which is what Dylan asked about. The pencil is carried by its tracks instead,
   which is what they are drawn oversized for.
3. **The break is drawn flat.** `BREAK_TILT` 24° → **0**. The tilt was meant to
   make a break read as a drawing cut rather than as a pipe that stops; from a
   camera 6° above the horizon a 350 mm disc tipped 24° projects to a thin dark
   **diagonal streak**, and Dylan asked what the line across the top of the beam
   pipe was. Flat, it caps the tube in the section accent. Same fix the
   collimator's own break face got in the sixth pass, same reason.
4. **The PE-shielding leader crosses to the left of the pipe.** Its label is in
   the left column and its anchor was at +x, so the leader ran across the beam
   pipe to reach a part that is present on *both* sides of it.
5. ⚠️ **Two chambers in section, not the four-arm pinwheel — and this is a drawn
   arrangement, not the station.** `STATION_ARMS = 2` is now the default, and the
   pair's azimuths come from `_section_angles()`, which is defined **relative to
   the camera**: just past edge-on, on the far side, so the cutaway slices them
   and every surviving piece is behind the sample. `SECTION_TILT = 22°` sets how
   much board face shows. **The price is that the two drawn azimuths are 136°
   apart, not the real 90°** — declared in `ASSUMPTIONS`, and
   `ear2_onfig_5_station_4arm.png` keeps the true pinwheel on record. Why it was
   worth it: drawn as the pinwheel, the two solid arms come out 45° to the screen
   and close on the drawn capsule until they touch it, so the sample is a sliver
   between two green rectangles.
6. **The station's slide labels are generic**, `sample` and `detectors`, not
   `³He sample` / `Micromegas trackers`. This is slide 9 and the experiment is
   introduced on slide 16: the specific words made the facility slide look like
   the setup slide. The standalone figure's `LABELS` still carry the full
   description, including the four-vs-two caveat.
7. ⚠️ **Nothing at the station is drawn 1:1 any more.** `CAPSULE_SCALE` 7 →
   **5.5** and a new **`PLATE_DRAW = 1.35`** on the chambers (Dylan: *"make the
   capsule slightly smaller and the detectors a bit larger — I don't care much
   about how the detectors actually look here or their accuracy, this is mostly
   just a generic example diagram"*). `PLATE_DRAW` scales board, gap and frame
   **together**, so each is still a chamber and only its size is wrong; the
   `STANDOFF` (`PLATE_R = 330 mm`) is still real. The chambers used to be the one
   true-size thing at the station, so the claim *"the four chamber plates are
   true size (400 mm)"* had to come **out of the backup slide's caption** — it is
   now the opposite statement. If you ever restore true scale, that caption is
   the thing to put back.

**Two new backup slides: the target, in full** (2026-08-11, Dylan — *"add a
backup slide with full details on how this target works (cooling, layers, etc).
Would be good for me in the future. Feel free to design a more detailed
visualization"*). Slides **46 and 47**, and they are the **source of record for
the target in this deck**:

1. *"The n_TOF spallation target, layer by layer — Target #3"* — the assembly cut
   open along the beam, plus a spec table (core, lead grade, plate gaps, vessel,
   both windows, the moderators, the beam).
2. *"How 2.7 kW leaves a block of lead that creeps at 135 °C"* — one anti-creep
   plate exploded off its slice, plus the nitrogen circuit end to end, the duty
   figures, the wedge trick, the temperatures and the creep numbers.

New files: **`scenes_target.py`** and **`make_target.py`** (`../.venv/bin/python
make_target.py`, ~40 s, two views). ⚠️ **Every shared dimension is imported from
`scenes_ear2`, never re-typed** — the facility render on slides 5–9 draws the same
object, and a divergence between the two would be worse than either being wrong.
That is also why `MOD_XY` was added to `scenes_ear2` while building these: at the
detail scale it was obvious the moderator can was drawn *overhanging* the vessel
it is bolted to, and there was nowhere for the gas outlets to be.

⚠️ **Three rendering traps recorded in `scenes_target.py`, all found the hard way:**

- **The beam-axis cutaway is wrong for a plate.** It slices a 9.85 mm part
  diagonally into a wedge — right for a tube, wrong for a face. The cooling view
  is an *exploded* plate instead, with no clip plane at all.
- **Lead, aluminium and a milled groove are three greys within 0.2 in value.** At
  this light rig they all saturate to the same pale grey. The groove interiors are
  pushed to a much darker slate, and the plate brighter, than the shared palette.
- **The plates in the layers view use PBR, uniquely in these two figures.** The
  cutaway leaves all six slices and five plates on *one flat plane*, where only
  shade separates them, and matte they merge. The metallic highlight is what makes
  9.85 mm of aluminium visible between 50 mm of lead — at the cost of some
  apparent thickness, which is why the plate's real thickness is shown in the
  other view.

And one thing that **cannot** be drawn on the cooling view and should not be
added: the **10° beam-to-target angle** is a yaw about the *vertical* axis, so it
foreshortens to nothing on a view of the plate face. It is in the layers figure,
where the horizontal proton arrow and the yawed stack are both in frame.

**Deck is now 56 printed slides** — 43 of main flow, a `Backup` divider, then 12
backup slides. Fine: it is a menu, and `RUNNING_ORDER.md` holds the 16-slide cut.
`mpgd26_talk_draft.pdf` is current at 56 pages.

⚠️ **Slide numbers in this file are a moving target.** The deck went 51 → 53 → 56
in two days and the setup section was renumbered when it landed, so any "slide N"
written here may be a day stale — the two target slides quoted as **20 and 21**
when they were written are now **46 and 47**. Identify a slide by its **title** and
re-derive the number; the check is one command:

```
cd mpgd26/slides && grep -n 'class="title' index.html | nl
```

**Left:**

- **Review the nine-frame setup build-up** (slides 16–24) — the largest unreviewed
  block in the deck. See "Still to do" below.
- **Slide 9's two superseded figures** — regenerating from `sat_det3` on the
  waveform-first basis gives a *better* resolution number.
- **The bench-slide flag** — needs Alexandra's talk order.
- ~~**Re-run `./make_pdf.sh`.**~~ Done 2026-08-11 — **56 pages**: the plan slide
  (25) plus the two photograph placeholders (26–27, PDF 30–31). Re-run it after
  the nine-frame review below, since that will change slides 16–25.
- **Work the two photographs in, or drop them** (slides 26–27) — they are parked
  after the plan as placeholders, with the open questions in the comment above
  them in `index.html`. The pairing that pays is the top-down photo *beside* the
  plan drawing rather than after it; both want a crop, and the "gamma-flash
  recovery" silkscreen bullet needs an inset or it is a claim about something
  the audience cannot see.
- **Confirm the beam-pipe configuration** — narrower than it was, but still open.
  The photograph beside the render shows the vertical line **closed all the way
  from the floor to the ceiling**, with detectors on a chamber in the beam. The
  render has a lower pipe ending ~1 m above the floor (Dylan's own description of
  the hall), open experimental space with the capsule in it, and a separate wide
  section from 20.45 m carrying the beam on to the dump. That upper section came
  from Dylan on 2026-08-11, so the two pictures now differ in **one** thing rather
  than two: whether the gap in the middle exists in the photograph's
  configuration. The presumption is that the section over the experimental space
  is removable and the photograph is an in-vacuum experiment, but **nobody has
  confirmed it**, so the backup caption states what each picture shows and stops
  there. Three numbers would close it: the height at which the lower pipe's exit
  flange sits, the height at which the upper section starts, and whether the
  middle section is a fixed part of the line.
- **Two things in the render are scaled off a photograph and could be replaced by
  a drawing if one exists** — the in-hall diameters and step heights (the lead-disk
  chamber, the reducer, the narrow tube, the upper pipe), and the white shielding
  on the floor, whose material is assumed to be polyethylene because it looks like
  it. Both are declared in `scenes_ear2.ASSUMPTIONS` and in the backup caption. If
  an EAR2 layout drawing turns up, these are the numbers to correct.
- **One question for Dylan:** which CAEN board is card 5 of the crate at
  128.141.177.244? It is the only thing left for the readback's absolute nA
  accuracy — integral conservation is closed regardless and nothing in the talk
  depends on it. The model is in neither repo, and the live crate was deliberately
  not probed during production running.

## Images are intentionally not tracked in git

The repo's top-level `.gitignore` blanket-excludes `*.png`/`*.gif` (see the
comment there — 250 MB was purged from history on 2026-08-04, policy is
"regenerate from the analysis scripts, don't commit the bulk output"). That
rule silently catches every image in `assets/img/` too, including several
that have **no one-command regeneration path** — unlike `mpgd26/figures/`,
this folder mixes mpgd26-native renders with curated external material. If
you clone this repo fresh, `assets/img/` will be empty. Regenerate it with
the steps below, or ask about carving out a `.gitignore` exception for
`mpgd26/slides/assets/` if that's preferable to re-deriving everything.

| File | Source / how to regenerate |
|---|---|
| `x17_signature.png`, `x17_story_capsule.png` | `mpgd26/make_x17.py` (see `mpgd26/README.md`) |
| `x17_story_top_{1,2,3}_*.png`, `x17_story_bot_{1,2}_*.png` (slides 5–6) | **`mpgd26/make_x17.py --layout build --capsule --slides`** — five frames of two overlay builds, each row drawn on its own full canvas with only the first N beats on it, so a beat lands in its final position and nothing already on the slide moves. **Re-flowed 2026-08-18 onto a 124-unit canvas (`scenes_x17.SW`) so each row is 2.16 : 1** — the shape of the figure hole on the slide — which makes the whole row render ~29 % larger with no font size changed. Do not widen `SW` back to 160 without re-flowing the beats |
| `chamber_exploded.png` | `mpgd26/make_chamber.py` |
| `microtpc.png` | **`mpgd26/make_microtpc.py --right waveforms`** (2026-08-17: the deck now takes the WAVEFORM variant, and the bare `compose(bare=True)` copy — no title band, no caption paragraph — is written straight into `assets/img/microtpc.png` by the script) |
| `share_cartoon.png`, `share_kernels.png`, `share_build.png`, `share_decompose.png` | **`mpgd26/make_share.py`** (new 2026-08-17; copies itself into `assets/img/`). Reads the frozen det3 bundle `calib_bundle_lp2_t0p` and event 1663 of its ref-pinned calibration cache off `/media/dylan/data`, so it needs the data disk mounted. ⚠️ **Read the module docstring before quoting any kernel amplitude** — c₁ sits on its calibration floor on a cosmic fit |
| `x17_story_bot_3_detect.png` | **`mpgd26/make_x17.py --layout detect --slides`** (new 2026-08-18, `scenes_x17.draw_detect`). The opening angle on it is the real 110°; the standoff, the gap and the chamber size are not to scale, and the 21° tilt is on the **chamber**, not the track |
| `efficiency_map_2mm.png`, `efficiency_residual_tail.png`, `efficiency_breakdown.png` | **`mpgd26/make_efficiency_breakdown.py`** (writes straight into `assets/img/`, no `figures/` copy — these are deck assets, not report figures). The map re-reads the 40 × 40 CSV that `mx_june_wft/report/make_maps_2mm.py` writes; everything else comes from `efficiency_breakdown.json`. `efficiency_breakdown.png` is **no longer on a slide** — the deck draws those bars in HTML — but is still generated so the handoff copy cannot go stale behind it |
| `angle_correlation.png`, `angle_resolution.png` | **`mpgd26/make_resolution.py`** (new 2026-08-18; deck assets only, plus `mpgd26/data/angle_resolution.json` with every number the slide quotes). Loads `wft/events.parquet` + the M3 rays, so it needs the data disk mounted and takes ~30 s. Replaces the 2026-07-14 hits-basis `angular_resolution.png` / `spatial_residuals.png`, **both now unused by the deck** |
| `mx17_board_peel_slide.png` | **`cd ~/CLionProjects/MX17_Geant && ~/PycharmProjects/nTof_x17/.venv/bin/python scripts/model/plot_mx17_model.py --only peel_slide`**, then `cp design/figures/mx17_board_peel_zoom_slide.png .../assets/img/mx17_board_peel_slide.png`. Same `fig_peel(zoom=True)` with `bare=True`: **no title band and no bottom arrow**, because the slide carries both in HTML type (2026-08-17). The titled `mx17_board_peel_zoom.png` below is unchanged and is what MX17_Geant's own design document uses |
| `build_bench.gif` | `mpgd26/make_anim.py --only build_bench` |
| `setup3d_1_capsule.png` … `setup3d_9_full.png` (slides 16–24) | **`mpgd26/make_ntof.py`, then `cp ../figures/ntof_build_<n>_<tag>_light.png assets/img/setup3d_<n>_<tag>.png`.** One command for all nine. Geometry is imported at run time from `~/CLionProjects/MX17_Full_Geant/scripts/plot_geometry.py` (itself written against `SimConfig.hh`), so the frames track the simulation; the e⁺e⁻ pair is a real Geant4 event picked by `mpgd26/tools/extract_ntof_event.py` into `mpgd26/data/ntof_event.json` (**which is tracked** — regenerating the frames does not need the sim). **The neutron is drawn, not transported** — a straight line up the beam axis to the pair vertex; the real history is selected and stored but deliberately not drawn (it belonged to a different event), and the beam envelope is measured from the neutron run's sampled primaries. Written with an alpha channel so they sit on the slide gradient. |
| `ntof_plan.png` (slide 25) | **`mpgd26/make_ntof_plan.py --bare`, then `cp ../figures/ntof_plan_bare_light.png assets/img/ntof_plan.png`.** A matplotlib **drawing**, not a PyVista render: orthographic plan down the beam, 1:1 in both axes. Geometry and event come from `scenes_ntof`, i.e. the same `plot_geometry.py` and the same `data/ntof_event.json` as the nine 3-D frames, and every number on it — including the dimension-chain labels — is computed from that geometry rather than typed in. The same command without `--bare` writes the titled version `figures/ntof_plan_light.{png,pdf}` that `report.html` cites. ⚠️ The beam axis is projected away, so the drawn opening angle (122°) is not the space angle (110°); the figure says both. |
| `photo_station_topdown.jpg`, `photo_arm_outside.jpg` (slides 26–27) | **`mpgd26/make_photos.py`** — slide-sized copies of the two photographs taken in EAR2 on 2026-08-10. The full-resolution originals are in `mpgd26/photos/` under the camera's own filenames; **do not go back to `~/Downloads` for them**, that is where they came from and it is not a durable location. No crop is applied (both are portrait phone frames); when the crop is decided, put the box in `PHOTOS` rather than cropping the original. ⚠️ **Placeholders** — parked after the plan, not yet worked into the argument; the open questions are in the comment above the slides in `index.html`. |
| `setup_1_mm.png`, `setup_2_sipm.png`, `setup_3_plastic.png`, `setup_4_full.png`, `setup_topdown.png` | copied from `~/CLionProjects/MX17_Full_Geant/scripts/mx17_buildup_clean_*.png` (`plot_buildup.py --style clean`) and `mx17_mm_layout_topdown.png` (`plot_mm_layout.py`). **Demoted to backup 2026-08-10** when the 3-D build-up took over the main flow — kept because they are the versions that carry the survey distances, i.e. what to show if someone asks "how far apart, exactly?" |
| `mx17_board_peel.png` | **`cd ~/CLionProjects/MX17_Geant && ~/PycharmProjects/nTof_x17/.venv/bin/python scripts/model/plot_mx17_model.py --only peel`** (~4 min), then copy from `design/figures/`. Refreshed 2026-08-10 to pick up two depth-key label fixes: the L5/L6 strip directions were **swapped** (the Y-measuring strips run along *x*, gerber-verified) and the resist thickness still read "100 µm" after `AsBuiltSpec` dropped to 10 µm — now read from `mx17_model.PASTE` so it cannot go stale again |
| `mx17_board_peel_zoom.png` | **`cd ~/CLionProjects/MX17_Geant && ~/PycharmProjects/nTof_x17/.venv/bin/python scripts/model/plot_mx17_model.py --only peel_zoom`**, then `cp design/figures/mx17_board_peel_zoom.png ~/PycharmProjects/nTof_x17/mpgd26/slides/assets/img/`. Same `fig_peel()` as the full-board figure via a new `zoom=True` mode (added 2026-08-10) — 25 × 25 mm of board (matching the figure's own title), split into four 6.25 mm bands so each peeled band shows 8 pad columns, with burned-in 5 × 0.78 mm readout-pitch and 5 × 0.80 mm resist-pitch calipers plus a 2 mm scale bar. **Revised 2026-08-10 (second pass, Dylan's review): the bulk pillars are gone from the resist band, and the L5/L6 bands now draw continuous strips on the gerber's own 0.78 mm grid with the vias suppressed, because the literal dot-and-stub artwork made the strip direction unreadable.** That makes those two bands **schematic, not copper**, so the title no longer says "real gerber copper" — it now names per band what is artwork (the pads) and what is schematic (resist ①, strips ③④, vias suppressed), and the band captions read "along x / along y — schematic". **How the dot-and-stub interconnect actually completes is still unresolved** (`HANDOFF_board_peel.md` §1) — the schematic asserts direction and pitch, not the connection. `--only peel_zoom` is deliberately **not** in the default all-figures run, and the full-board `peel` output is unchanged (it keeps the pillars and the literal artwork; both changes are gated on `zoom`). The pitch is 0.78 mm exactly, not 0.7785 mm; **the slide-7 caption was corrected to match on 2026-08-10** and the error was traced out of the whole package (see the pitch section below) — see [`HANDOFF_board_peel.md`](HANDOFF_board_peel.md) |
| `charge_sharing_schematic.png`, `unsharing_depth_bias.png`, `event_display_3d.png`, `angular_resolution.png`, `spatial_residuals.png`, `time_resolution.png` | `pdftocairo -singlefile -png -r 300` from the matching PDF in `mx_june_cosmic_qa/engineer_package/figures/` (e.g. `10-det3A-spatial-residuals.pdf` → `spatial_residuals.png`) — those source PDFs **are** tracked in git. **All still 2026-07-14 hit-chain vintage**, i.e. the same staleness the efficiency figure had; `angular_resolution.png` in particular is superseded by the waveform-first σ_θ (1.08°/1.11° on det3, not 1.66°) |
| `efficiency_breakdown.png`, `efficiency_residual_tail.png` | **`mpgd26/make_efficiency_breakdown.py`** — one command, reads only `mx_june_wft/02_efficiency.py`'s JSON reductions, writes straight into `assets/img/`. Regenerated 2026-08-10 on the waveform-first chain, `sat_det3`, 7,055 reference muons: **93.5 %** within 5 mm. Replaces the `pdftocairo` route from `engineer_package/figures/21-det3A-efficiency-breakdown.pdf`, which was 2026-07-14 hit-chain vintage on a *different* det3 run (`g_det3_wknd`, 92.9 %) **and** carried the stale "88.8 %" annotation — that literal was hardcoded in the engineer-package script's annotation string and is now derived there too, so the inconsistency cannot recur on either path. Full old-vs-new accounting and the slide markup: [`HANDOFF_efficiency.md`](HANDOFF_efficiency.md) |
| `ear2_onfig_{1_target,2_neutrons,3_collimation,4_dump,5_station}.png` plus `ear2_onfig_5_station_4arm.png` (**on slides 5–9**) and `ear2_beamline_{1_target,…,5_station}.png` (frame 5 is **on the backup slide**) | **`mpgd26/make_ear2.py`** — one command for everything: `cd mpgd26 && ../.venv/bin/python make_ear2.py` (~1 min 40 s), written straight into `assets/img/`. Two label layouts of the same render, both built every run: `ear2_onfig_*` puts the labels **on the drawing's own background, left and right** (wider canvas, 1740×2050, `annotate.side_labels`, `ONFIG_SIDES` assigns the sides by hand); `ear2_beamline_*` puts them in a **gutter column** down one side (`annotate.column_labels`). Swapping which the slides use is five `img src` edits — see the fifth pass at the top of this file for why each is where it is. It also writes `figures/ear2_build_<n>_<tag>_light{,_labelled}.png` with a live-text `.pdf`, `figures/ear2_onfig_<n>_<tag>.pdf`, and the standalone-figure aliases `figures/ear2_beamline_light{,_labelled}.png` that README.md and report.html cite. In-house PyVista render of the EAR2 vertical beam line, **a five-frame build-up**: target → the neutrons leaving at 90° → the middle of the line (collimator, floor, shielding, the pipe ending) → back into a pipe on up to the dump → the measuring station. The three frames are **strict subsets of one drawing**: same camera, lens, light rig, canvas and drawn scale, and each label column is solved once from the full set and then filtered, so **nothing moves and no label shifts between frames**. Parts and per-frame labels are `scenes_ear2.STAGE_PARTS`. Sourced heights, apertures and materials are in `scenes_ear2.py`'s docstring from C. Weiß et al. (n_TOF), *NIM A* **799** (2015) 90 and J. A. Pavon-Rodriguez et al. (n_TOF), *EPJ A* **61** (2025), arXiv:2505.00042 — **no sourced number has changed in any of the reworks.** Frame 5 is written **twice**: **two chambers in section** (the default and what the slide uses) and the true four-arm pinwheel (`STATION_ARMS`, `_4arm` suffix) — Dylan asked to compare, and chose the section pair because the pinwheel closes on the drawn capsule from both sides. ⚠️ Six things to know before editing. (1) **The lower beam pipe ends for real at `H_PIPE_END = 19.16` m**, ~1 m above the EAR2 floor, with the beam crossing the experimental space in air (Dylan, 2026-08-11 — 19.16 m is his recollection, not a drawing, hence the figure's `≈ 1 m above the floor`). (2) **A separate wide pipe starts at `H_UP0 = 20.45` m** and carries the beam on to the dump, cut off by the top of the frame; do not merge it with the lower one and do not delete it. (3) **The shape of the line inside the hall is scaled off the photograph** beside it, not off a drawing — shielding, stepped diameters, reducer, and the lead disks drawn at 0.36 m of the documented 0.57 m; all of that is in `ASSUMPTIONS`. (4) **The drawing stops ~1.35 m above the station**, so the bunker ceiling, its roof and the beam dump are real but above the frame. (5) **The chambers draw 24 strips, not 512** — the real pitch is 0.26 px here. (6) **The two drawn chambers are placed relative to the CAMERA** (`_section_angles`), so their drawn azimuths are 136° apart and the real station's 90° pinwheel is not what frame 5 shows — that is in `ASSUMPTIONS`, and the `_4arm` render keeps the true arrangement on record. In *that* render the near two chambers must be drawn UNCUT; run through `cut()` the cutaway deletes them outright, which silently makes the four-arm variant identical to the two-arm one. (7) **`BREAK_TILT` is 0** — a tilted section face reads as a stray diagonal line at this camera elevation. The one remaining departure from as-built geometry is the station's **support frame, not drawn** — declared in `scenes_ear2.ASSUMPTIONS`, in the on-figure label and in the backup slide's caption, and now also visible in the photograph beside it. Slide-copy labels must keep every line to **~21 characters** — `annotate` does not wrap, a longer line is silently clipped at the edge of the PNG, and the on-figure variant reuses `LABELS_SLIDE` for exactly that reason. **Replaced the borrowed schematic on slide 5.** |
| ~~`ntof_facility_schematic.png`~~ | cropped from Fig. 1 of G. Tagliente et al. (n_TOF Collaboration), *EPJ Web Conf.* 292, 12002 (2024), page 4 of `https://cds.cern.ch/record/2939795/files/fulltext.pdf` — **no longer used on any slide** (2026-08-10): its EAR2 content is in `ear2_beamline.png` and the only thing it added was the 185 m horizontal line to EAR1, which the deck no longer discusses |
| `ear2_hall_photo.jpg` | **crop of `ear1_ear2_photo.jpg`** (below), its right-hand panel, downsized: `cd mpgd26/slides/assets/img && ../../../../.venv/bin/python -c "from PIL import Image; Image.open('ear1_ear2_photo.jpg').crop((1960,0,3640,3264)).resize((860,1671), Image.LANCZOS).save('ear2_hall_photo.jpg', quality=86, optimize=True, progressive=True)"`. **Restored on 2026-08-11** at Dylan's request, and moved to the **left** of the render the same day at his request. It is the only thing in the deck that shows the real EAR2 hall and the **aluminium-profile support frame the render deliberately omits**, and since 2026-08-11 it is also the **source** for the shape of the line inside the hall (the white shielding, the stepped diameters, the reducer, the wide upper pipe — scaled against the ~1.2 m white disc at its foot). ⚠️ In it **the line is closed all the way from the floor to the ceiling**, with detector snouts on a chamber in the beam — a different configuration from the one the render draws. The backup slide's caption states both and claims nothing about which section is removable; **do not delete that sentence.** Credit **© CERN** is on all four slides that use it (a one-line `.fig-label` on the main three, in the caption on the backup) — keep it there. |
| `ear1_ear2_photo.jpg` | CERN Document Server, `https://cds.cern.ch/record/2148416/files/n-TOF-EAR1-EAR2.jpg` (OPEN-PHO-EXP-2016-006, © CERN) — this one **is** tracked (not a `.png`). Two panels; the deck uses only the right-hand one, via the `ear2_hall_photo.jpg` crop above. Kept as the uncropped source, and it is the photograph the beam-dump question originally came from. |
| `atomki_angular_correlations.png` | extracted (`pdfimages`) from `~/Downloads/Neff n_TOF Analysis Meeting X17 Update 3_24.pdf`, page 4; original data: Krasznahorkay et al., *PRL* 116, 042501 (2016) / *PRC* 104, 044003 (2021) / *PRC* 106, L061601 (2022) |
| `atomki_spectrometer_schematic.png` | cropped from Fig. 4 of J. Gulyás et al., arXiv:1504.00489 (*NIM A* 808, 21 (2016)), page 5 |
| `ideal_pair_spectrum.png` | `mpgd26/make_pair_kinematics.py` — **one command, reads a committed 394-line histogram reduction** (`mpgd26/data/ideal_pair_kinematics.csv`), so it rebuilds offline with no lxplus. The reduction itself comes from the Geant4 `pairs_thermal_trig_2cm` campaign's generator-truth vertices (`--reduce`, needs the 64 MB npz); provenance and the IPC model caveat are in the script's docstring. **The two curves are normalised separately and are 50/50 by construction — shapes, not a branching ratio.** |
| `status_*.png` (6 files) | `mpgd26/make_status_plots.py` — **one command, and it reads only committed reductions plus a small local mirror**; see that file's docstring for the four inputs and `~/.cache/mpgd26_status/` for where two of them are staged |
| `status_imon_response.png` | **`.venv/bin/python ntof_july_analysis/flash_charge/make_imon_figure.py`** (seconds; regenerate its input first with `ntof_july_analysis/flash_charge/imon_response.py --src /media/dylan/data/x17/beam_july`, ~2 min). Added 2026-08-10: the direct measurement of the HV-monitor's imon impulse response, which **closes the open systematic** the main charge slide and its backup used to carry — the readback is a ~1 s averager, so `mean − median` is the time-average current and the charge numbers are measurements, not lower bounds. Deliberately **not** in `make_status_plots.py` (that file was being edited concurrently); it reads only the committed reduction `flash_charge/results/imon_response_run_79.json` + the fold CSVs, uses `mpgd26/plotstyle.py` and adds **no new hues**, so folding it in later is one entry in that file's `FIGURES` dict. Measured on **run_79** (2026-07-26), not run_158 — same production setpoint, and the only production-point `hv_monitor.csv` in the local mirror; det C agrees with run_158's det C to 5 %. Full method, the timestamp-drift trap it had to survive, and the slide markup: `ntof_july_analysis/flash_charge/HANDOFF_FLASH_CHARGE_2026-08-09.md` §8 and [`HANDOFF_imon.md`](HANDOFF_imon.md) |
| `target_pointing_fans.png` | copied from `…/analysis/wft/run_79/stat090_0000/mx17_A/wall_segment_tour/wall_segment_tour_all.png`, made by `ntof_tracking/run79_wall_segment_gif.py` |
| `target_pointing_slope.png` | copied from the same analysis dir, `figures/01_target_pointing_A.png` (not currently used on any slide — kept as the alternative to the fans figure) |

## Open items, by slide

**Setup build-up (16–27), added 2026-08-10** — one figure in nine states with
the step's explanation beside it, since 2026-08-11 a tenth slide that draws the
same thing as a plan, and after that two **placeholder photographs** of the real
station. Two decisions worth knowing before editing:

* **It is four acts at four cameras**, because the capsule is 23 mm and the
  setup is 1.2 m — and then because the layers are stacked *radially*. Slides
  16–18 are the close-up act (`micro`), slide 19 is the chambers arriving
  around the vessel (`close`), slides 20–22 are one fixed camera (`hero`), and
  slides 23–24 are the same setup **from straight down** (`over`, 89°,
  and drawn *bare* — only the active volumes keep their colour), because in any
  three-quarter view the trigger wall stands in front of the plastics and the
  liquid and a leg arriving in them cannot be seen. Within each act the frames
  are interchangeable stills. Slide 19 repeats slide 20's layer at a larger
  scale, so **cut it first** if the section runs long, then slide 17.
* **Only the charged pair is drawn.** The gammas in the event are real but
  neutral, so they crossed the picture as rays with no visible cause; the beam
  is a plain grey column with a slim direction dart on the axis inside it (an
  cut off at the vessel's **nose** once any detector is placed — past the target
  it reads as a pole holding the apparatus up, and alongside it a column wider
  than a 23 mm vessel makes the vessel look hazy. **The neutron is drawn,
  not transported** — a straight line up the beam axis to the vertex; the real
  history is still in the JSON but belonged to a different event. The legs are
  cut at the first layer not yet placed, so they grow with the apparatus. All
  fixed 2026-08-10.
* **The event is selected on how it leaves the vessel.** Both legs must exit
  through the *barrel* wall, transversely, rather than through a domed end
  where they would cross several times the wall thickness at a glancing angle;
  and the extractor prefers events landing in the two arms the figure draws
  solid. `--prefer-arms` must be kept in step with `scenes_ntof.NEAR_ARMS`.
* **The pair is one real Geant4 event; the neutron that made it is drawn.** It
  cannot be otherwise: the radiative branch that makes the ⁴He* is ~10⁻⁸ of
  ³He(n,p)t, so no simulated neutron history contains one. Slide 18 says this in
  its caption. **Do not quietly drop that caption** — it is the honest part, and
  for this audience it is also the interesting part.
* **Every build frame names the layer it just added**, and only that one — the
  label moves outward with the build instead of accumulating
  (`make_ntof.LAYER_LABEL` / `LAYER_POS`, anchored from the geometry by
  `scenes_ntof.layer_anchor`). **One label, a leader to each solid arm**: a layer
  is four objects and two are drawn solid, so a single line implies the label is
  about that one — and it is a line per drawn *object*, so the plastics (two bars
  per arm) carry four and the text drops the "2 ×" the lines already say.
  Top-left on the build frames, **bottom** on the close-up, where
  the chambers fill the frame and the only empty space is the see-through one.
  Sizes on the figure are in **cm** (read across a room); the bullets keep mm.
  It repeats the first bullet's size on purpose, so a frame lifted out of the
  deck still says what it is. Added 2026-08-11.
* **The overhead act draws the capsule whole and solid** (`VIEWS['over']`,
  `cut=False`). The cutaway plane contains that view direction, so from above it
  deletes the half of the vessel nearest the bottom of the frame instead of
  sectioning it — and the capsule is the one thing `BARE` does *not* whisper,
  because 23 mm on a 1.2 m frame faded to 10 % is a smudge at the exact point
  the picture converges on.
* **Slide 25 closes the section with a plan** (`make_ntof_plan.py --bare`,
  added 2026-08-11) — the one figure here that is **not a render**. Every frame
  before it has perspective, so the four arms sit at four different distances
  from the lens and every length is foreshortened by its own amount; the plan is
  orthographic and 1:1 in both axes, so the **204.5 mm standoff** (the dashed
  circle all four windows are tangent to), the layer radii (330 / 410 / 487 mm,
  as a dimension chain on arm B — the one arm no leg crosses, and the numbers
  are that arm's; the outer two layers shift a few mm per arm) and the **23 mm target**
  can be measured off the slide instead of being asserted in a caption. It also
  shows two things a three-quarter camera cannot: the ~16 mm pinwheel offsets,
  and that two of the four liquid vessels are laid on their side with their PMTs
  pointing sideways. What it gives up is the beam axis — the legs rise ~135 mm
  along it, so the opening angle **as drawn is 122° against a 110° space
  angle**, and the figure quotes both. Same geometry module and same event JSON
  as the renders. It repeats no content, so it is a clean place to stop if the
  section overruns.
* **The chambers grew on 2026-08-11**, and the frame did not. The sim's MM
  active area went from an unsourced 38 × 34 cm to the measured **39.9 × 36.0**
  (the short axis is the one *along the beam*: ~19 mm at each end of the strip
  plane is passivated), and every figure here imports it, so all nine renders and
  the plan were re-run. The **frame** is now drawn as the fixed **440 mm outer
  square** it really is (`scenes_ntof.FRAME_HU`) instead of a constant 30 mm
  cheek — which used to give the same 440 around the old size but gives 490
  around the new one, and 490 does not fit: the pinwheel has only 15.5–17.3 mm
  of tangential shift, and 440 is exactly what it clears. The plan view is where
  this shows, since it is the only figure with all four arms in one plane; drawn
  the old way the neighbouring arms interpenetrate by ~10 mm on the page. The
  MM bullet on slide 20 and the layer label on the figures were updated with it.
* **Two photographs, slides 26–27, are placeholders** (`make_photos.py`, added
  2026-08-11). Top-down into the station, and one arm from outside. They are
  parked here rather than placed: the section has already said everything they
  show, and the pairing that would actually pay is the top-down photo *beside*
  the plan drawing. Open questions are in the comment above them in `index.html`.

Not done: only the light theme is rendered. `make_ntof.py --theme dark` works if
the deck ever goes dark; `make_ntof_plan.py` is light-only by construction
(`plotstyle` has no dark variant, as the deck's CSS has none).

**Title** — **done 2026-08-10.** Dylan Neff, MPGD 2026 Prague, 3 September 2026.

**Running order** — **the slot is 15 min + 5 min**, so the deck's 28 main-flow
slides do not fit. A concrete 16-slide cut, with what moves to backup and why,
is in [`RUNNING_ORDER.md`](RUNNING_ORDER.md). Read that before doing any more
slide work — some of the open items below are on slides that cut proposes to
demote, which changes how much they are worth.

**Motivation / ATOMKI** (comment above slide 3): condensed to 2 main-flow
slides + backup already done. Still open — beam orientation at ATOMKI
unconfirmed (deliberately left off every slide); PyVista remake of the
5-telescope schematic in house style not started; highlighting the
³H(p,e⁺e⁻)⁴He channel specifically as the bridge into the EAR2 slide not
done (evidence slide still gives ⁸Be/⁴He/¹²C equal weight).

**n_TOF / EAR2 facility** (slides 5–9, plus one backup) — **done 2026-08-10, revised four times on 2026-08-11.** EAR1 is out of
the discussion and the borrowed *schematic* is gone: the slides are the in-house
render (`mpgd26/make_ear2.py` + `scenes_ear2.py`) with the cropped CERN
photograph of the hall beside it. The Tagliente Fig. 1 schematic was **folded in
rather than kept as a second panel** — everything it said about EAR2 is in the
render, and the one thing it added was the 185 m line to EAR1.

**The five main-flow slides carry no caption; one backup slide carries it in
full.** Both pictures are portrait, so a six-line caption was eating a fifth of
the height they needed. Everything that was in it — the provenance, the citations
and every drawn-rather-than-measured item — is on the backup slide *"The EAR2
render, and what in it is drawn rather than measured"*, under the same figure. The
main slides keep a one-line `.fig-label` credit under each picture. ⚠️ **If you
change what the render draws, change that backup caption with it** — it is now
the only place in the deck where the disclosures live. Column widths on the main
slides are **proportional to the two images' aspect ratios** (0.62 / 1.00 / 1.04);
that is load-bearing, not cosmetic — see the fifth pass at the top of this file.

**It is five consecutive slides, one drawing built up** (2026-08-10 second pass,
then four rounds on 2026-08-11, all on Dylan's review): frame 1 the lead target
and the protons, frame 2 **the neutrons leaving at 90°** and filling the pipe with
the break annotated, frame 3 the whole middle of the line — second collimator,
EAR2 floor and its shielding, lead disks, the pipe **ending** a metre above the
floor — frame 4 **back into a pipe, on up to the dump**, frame 5 the measuring
station. Same camera, same scale, same labels-in-place — the deck has no fragment
mechanism (`make_pdf.sh` prints one page per `<section>`), so a build *is*
consecutive slides. The bullets accumulate, with the already-narrated ones set
`li.dim` and **shortened as they pile up** so frame 5 still fits, and the figure
markup is **deliberately identical on all five** so neither picture can move
between them. The **photograph is on all five frames** and is not part of the
build. ⚠️ One thing in the *render* is not a strict subset and it is deliberate:
the beam is drawn at a whisper across the 0.5 m the sample occupies, on frame 5
only — see the station paragraph below.
**At 15 minutes all five go to backup as one unit** — this does not change the
16-slide cut in [`RUNNING_ORDER.md`](RUNNING_ORDER.md), and if you show only
one, show frame 5.

**Three legibility fixes, all from Dylan looking at the output** (2026-08-11
seventh pass), each of which is a trap worth knowing before you touch the palette
or the break:

- **`COL['neutron']` is dark on purpose** (`#8b96a3` → `#55657a`, arrows 9 → 12 mm).
  The old grey has the same *value* as the amber-tinted pipe interior it sits in,
  so the five arrows leaving the target were invisible. ⚠️ The obvious fix —
  white — is wrong: the same neutrons cross the open hall higher up on a
  near-white background. A darker slate is the one colour that works on both.
- **One `ENV_ALPHA = 0.16` for the beam envelope everywhere.** It used to be 0.34
  above the collimator against 0.17 below, and the same amber at twice the opacity
  on a thin tube seen through both its walls came out **olive-brown** — two
  different-coloured beams in one figure. The thin pencil is carried by its
  tracks, which are drawn far wider than the beam really is for exactly this
  reason.
- **`BREAK_TILT = 0`.** The 24° tilt existed so a break would read as a drawing
  cut rather than as a pipe that stops. From a camera 6° above the horizon a
  350 mm disc tipped 24° projects to a thin dark **diagonal streak** across the
  pipe, which is what it actually read as. The collimator's own break face was
  flattened for the same reason a few hours earlier; this finished the job.

⚠️ **The target is now Target #3, drawn from its design paper** (2026-08-11,
second answer). Dylan asked whether the shape was known or guessed; the first
answer was that the lead body was sourced but was the **Target #2 cylinder** —
a real object, retired at the end of 2018, and *not the one our data are on* —
with none of the EAR2-facing assembly drawn at all. He asked for it as accurate as
feasible, so it was rebuilt from

> R. Esposito et al. (n_TOF), *Design of the third-generation lead-based neutron
> spallation target for the neutron time-of-flight facility at CERN*,
> **Phys. Rev. Accel. Beams 24 (2021) 093001**, arXiv:2106.11242 — Sec. III A the
> core, Sec. III B the vessel and the two moderators, Sec. II B the 4 cm water,
> Sec. II C the 5 cm lead plate.

Sourced and drawn to scale: **six lead slices** on the proton axis, **600 × 600 mm**
in cross-section, **five 50 mm + one 150 mm** with the thick one at the
**downstream** end against the EAR1 moderator (high-purity UNS L50006, ≥ 99.98 wt%);
**9.85 ± 0.05 mm** aluminium (EN AW-6082 T6) **anti-creep plates** between them,
which carry the **nitrogen** cooling channels — the slice gaps are held to
45–195 µm; an **AISI 316L vessel** at 0.5 bar N₂ with the proton window locally
thinned to **3 mm**; the **4 mm** stainless **neutron window** electron-beam welded
to its top; the **50 mm lead plate** that window supports, which buys back the
factor-6 prompt-γ increase the better EAR2 coupling cost; the **EAR2 moderator**
outside the vessel, bolted to it and resting on that plate, holding the **40 mm
water layer** FLUKA picked as the optimum for the EAR2 resolution function
(aluminium EN AW-5083 H112, two independent circuits, demineralised **or** borated
water); and the **hemispherical aluminium vacuum window** above it (that one from
Pavon-Rodriguez Sec. 3 — the previous target coupled to EAR2 through a *polygonal*
window instead).

That last stack is why the rebuild was worth doing rather than just correcting the
caption: **it is the part of the target this figure is about and it was missing.**
`H_PIPE_START` moved **0.30 → 0.60 m** to clear it, its flanges with it, and the
neutron envelope now starts at the top of the water rather than inside the lead.
A **twelfth label**, `water moderator + lead plate`, was added to frame 1.

⚠️ **Four target details are drawn, not sourced**, all wall thicknesses or extents
the paper does not give: the vessel wall (`VES_GAP`), the moderator cans' walls and
plan size (`MOD_WALL`), how far the EAR1 moderator extends (`EAR1_MOD_X` — the
paper says only that it is the larger of the two), and the vacuum window's radius
(`R_VACWIN`, drawn at the pipe bore). All four are in `ASSUMPTIONS` and on the
backup slide. The **EAR1 moderator is drawn as aluminium with no water fill** on
purpose: filled to match the EAR2 can it is a 0.3 m³ block of blue that outweighs
the 40 mm layer the figure is about.

**The PE-shielding leader crosses to the left of the pipe** — its label is in the
left column and its anchor was at +x, so the leader ran over the beam pipe to
reach a part that is present on both sides of it anyway.

**The measuring station, rebuilt** (2026-08-11, Dylan: *"I'm also not sure what
the green is supposed to be there"*). It was four 400 mm slabs in PCB colour with
translucent gas in front, and seen obliquely that is a green box and nothing else.
Each chamber is now built with `meshes.rect_chamber` and drawn as one: aluminium
frame, readout board **with strips on it**, 30 mm of lit drift gas facing the
sample, and the entrance window as a tint. The strips are what makes it read as a
detector at a glance — ⚠️ **24 are drawn, not 512**, because the real 0.78 mm
pitch is 0.26 px at this scale and comes out a flat grey wash; that is in
`ASSUMPTIONS`. `"Measuring station"` also became **two labels**, and on the slide copy they are deliberately **generic** — `sample` and `detectors`, not ³He and Micromegas. This is slide 9; the experiment is introduced on slide 16, and the specific words made the facility slide look like the setup slide. The standalone figure's `LABELS` still carry the full description. And the beam is **faded across the sample** on this frame only
(`add_neutrons_hi(fade_sample=True)`): at full strength it is a grey line down the
middle of a translucent capsule and reads as the capsule being *behind* the beam.

**Frame 5 exists in two versions and both are built every run** (Dylan asked to
compare): `ear2_onfig_5_station.png` shows **two chambers in section** and is the
default and what the slide uses; `ear2_onfig_5_station_4arm.png` shows the true
**four-arm pinwheel** — two solid, two the camera looks through drawn as bare
frames. `scenes_ear2.STATION_ARMS` picks; swapping is one `img src` edit.

⚠️ **The section pair is a drawn arrangement, and it is not the station's.**
`_section_angles()` places the two relative to the **camera** — just past edge-on,
on the far side, so the cutaway slices them and every surviving piece sits behind
the sample — which makes their drawn azimuths **136° apart rather than the real
90°**. That is declared in `ASSUMPTIONS` and the `_4arm` render keeps the truth on
record. It is worth the departure because the pinwheel's own geometry, at this
camera, brings the two solid arms in at 45° to the screen and closes them on the
drawn capsule until they touch it: the sample ends up a sliver between two green
rectangles, which was the original complaint about this part of the figure and is
not fixed by drawing the chambers in more detail. `SECTION_TILT = 22°` sets how
much board face shows; at 0° you get two 42 mm bars that read as nothing.

⚠️ In the **four-arm** render the near two must be drawn **uncut**: the cutaway
plane passes through the beam axis, so run through `cut()` they are not faded but
*deleted*, which is why the first four-arm attempt came out pixel-identical to the
two-arm one. `frame_ring` is a hollow band, so drawing them solid frames the
sample rather than hiding it.

**The sample is centred between the two pipes** (2026-08-11, Dylan): `H_UP0` went
20.45 → **20.74 m**, which puts 19.95 m exactly halfway between the end of the
lower pipe (19.16 m) and the start of the upper one, and the upper pipe shows
**only its entrance flange and one ring** before the top of the frame cuts it.
`DRAWN_H` is 6.11 m. And `Lead target` moved to the **right** label column,
because with it and `protons` both on the left their leaders crossed under the
target.

**It was four frames for a few hours** (2026-08-11): the dump got its own frame,
because collimation and the dump had been sharing frame 2 and that buried the
collimators — Dylan's reaction to that version was *"where is the first
collimator, I actually don't know?"*. Then the dump came out of the drawing
altogether, so the dump frame had nothing left to reveal and it went back to
three. Collimation still has its own frame, which was the point, and the
measuring station still appears **only on the last frame**: everything before it
is beam line, and the last thing to land is us.

**The drawing stops just above the station** (2026-08-11): the bunker ceiling, its
break and the beam dump drawn in full used to occupy a third act at the top. It
was 2.7 m of drawn height and the widest object in the picture, all of it
shielding, and it left the station small. Removing it took `DRAWN_H` from 8.67 m
to 5.71 m (5.91 m once the upper pipe needed room), so at the same canvas
everything left is drawn **~1.5× larger**, and the upper break went with it, which
Dylan did not want either. Nothing is asserted away: the beam leaves the top of
the frame inside the wide pipe that really carries it there, under a label giving
the dump's entrance height, and both `ASSUMPTIONS` and the backup caption say what
is above the frame. `FLOOR_W` was narrowed 2400 → 1900 mm in the same pass,
because at 1.5× a 2.4 m floor slab reaches almost the full frame width and
competes with the station. The dump's own dimensions survive as a comment in
`scenes_ear2.py` and in the slide caption; the geometry is in git history.

**The photograph is back, on the left of the render** (2026-08-11, Dylan asked for
it, and asked for it on the left): `ear2_hall_photo.jpg`, the right-hand panel of
the CERN photo cropped and downsized. It earns its place three times over — it
shows the real hall, it shows the **aluminium-profile support frame the render
deliberately omits** (the honest way to have that omission both ways), and it is
now the **source** for the shape of the line inside the hall. ⚠️ **In it the line
is closed all the way from the floor to the ceiling**, with detector snouts on a
chamber in the beam. Ours ends the lower pipe about a metre above the floor and
puts the capsule in air above it, with a separate wide section above taking the
beam on to the dump. The backup caption states both and claims nothing about which
section is removable — **do not delete that sentence**, and do not "reconcile" the
two pictures by guessing. Open question in "Left:" above.

**The line inside the hall is modelled from that photograph** (2026-08-11, Dylan:
*"can we try to add a bit more realism to the bottom beam pipe, with the white
polyethelyne somewhat modeled and a rough attempt at tapering the pipe"*). What is
drawn now, bottom to top: the white **segmented polyethylene shielding disc**
lying on the floor around the pipe penetration with a **collar** above it
(`add_shield`, `pie_segments`), the **lead-disk vacuum chamber** at the shaft's
317 mm bore, a **reducer** (`conical_shell`), the **narrow tube** at 156 mm, and
the flanged **exit window**. ⚠️ Radii are scaled off the photo against its ~1.2 m
white disc and the step heights are chosen to fit between the floor and the end of
the pipe: this is a reading of a *photograph*, not of a drawing, and
`ASSUMPTIONS` says so in those words. Two consequences worth knowing. The
shielding needed **its own colour** (`COL['pe']`, a warm white) because on
`COL['bpe']` it vanished into the floor slab it stands on — so `COL['floor']` went
darker in the same pass. And the **lead disks are drawn as a 0.36 m stack against
the documented 0.57 m** effective height, because the drawn hall segment is 1 m
tall and also has to carry the collar, the reducer and the exit flange; the figure
claims *that* there is further collimation in the hall and roughly where, not how
much.

**There is a pipe above the experimental space, and it is a different pipe**
(2026-08-11, Dylan: *"the top pipe toward the beam dump included and cut off at
some point"*). `add_uppipe` draws the wide section from `H_UP0 = 20.45` m up past
the top of the frame, **cut off by the camera margin rather than capped** — which
reads as "continues" in a way no drawn cap does. So the line reads the way the
photograph reads: narrow tube → flanged end → **open space with the experiment in
it** → wide tube away. It is drawn at a **160 mm bore rather than the photo's own
proportions**, because at full size it outweighed the measuring station, which is
the opposite of what removing the beam dump was for. The height at which it
begins is drawn, not sourced.

**"What is this pink diagonal thing in the collimator?"** (Dylan, 2026-08-11) — it
was the **break's own section marker**: `COL['section']` on a 24°-tilted disc at
the collimator's 680 mm radius, so a third of it hung outside the block and read
as a fin bolted to it. It was never hardware, and it was in the wrong place too —
drawn at the act boundary, it floated 120 mm up inside the block instead of
sitting on the block's own bottom. Now **flat, inset and at the block's bottom**,
with `COL['section']` desaturated. The tilt survives on the pipe's face below,
where the disc stays inside its own silhouette.

**The beam pipe really does end, ~1 m above the EAR2 floor** (Dylan,
2026-08-11) — and the space above it is open hall for experiments to stand in,
which is where our station is. The 2026-08-10 version of this render ran the
pipe continuously from the collimator, past the station and up through the roof
into the dump, and then **cut it away** above the floor as a drawing device with
a section marker and a three-line disclaimer. That was **wrong about the
facility**, not just about the drawing, so the pipe now terminates for real, in
a flat circular end with a flange (`H_PIPE_END = 19.16`, and `H_PIPE_TOP` is
gone), the neutrons cross the experimental space **in air**, and the disclaimer
went with it. The `section` accent is once again used for **only** the break in
the vertical scale, which is what it was for. **Do not join the lower pipe to the
upper one** — the wide section above 20.45 m is a *separate* pipe with open space
under it, and merging the two is precisely the mistake this correction undid.
19.16 m itself is not from a drawing — it is Dylan's recollection of the hall,
which is why the figure says `≈ 1 m above the floor` and not a height.

**One departure from the as-built geometry remains, and the deck says so.** The
station's support frame is not drawn — it really hangs in an aluminium-profile
frame standing on the floor — so the capsule appears to float. That was Dylan's
call, for legibility: drawn as built, the capsule and its four chambers sit
behind four grey uprights. It is disclosed three times over: the station
label's `(its support frame is not drawn)`, `scenes_ear2.ASSUMPTIONS` (hence the
standalone figure's caption), and the slide caption. **Do not quietly drop any
of those.** Smaller things are declared in `ASSUMPTIONS` only, because nothing on
the slide turns on them: the exit and entrance windows are schematic (no material
or thickness is claimed), and the in-hall diameters and step heights are the
photograph reading described above.

**Where the first collimator is** — 7.4–8.4 m above the target, i.e. **~10 m
below the EAR2 floor**, down in the shaft: 1 m of iron with a 200 mm bore
(Weiß Table 2). It is invisible in the render because it falls inside the lower
break, which is exactly why the break's label now names it and frame 2's bullet
spells it out. The second collimator is 15.04–18.04 m, 2 m of iron plus 1 m of
borated PE, bore 70 → 21.8 mm, and the sweeping magnet (10.4 m) and the 8-slot
filter station (11.4 m) are in that same break.

**The structure at the top of the pipe is the beam dump, and it is now
verified** — the old "tentatively the beam dump, UNVERIFIED" caveat is
discharged. C. Weiß et al. (n_TOF), *NIM A* **799** (2015) 90, §2: *"The beam
dump for this new vertical flight path is installed on the roof of the
bunker."* §2.4 and Fig. 5 give the three layers the render draws — a borated-PE
core 400³ mm with a 340 × 250 mm entrance bore, iron 1600³ mm, concrete
3200 × 3200 × 2400 mm — and Table 1 puts its entrance at 24.73 m above the
target. Independently confirmed by J. A. Pavon-Rodriguez et al. (n_TOF),
*EPJ A* **61** (2025), arXiv:2505.00042 §5 (beam *"stopped at the beam dump
downstream (not shown), placed above the ceiling"*) and by the CERN Bulletin of
29 July 2014. **Do not write "19.5 m" for the flight path** — no source says
it. What the literature actually quotes: ~20 m nominal, EAR2 floor 18.16 m,
ceiling 23.66 m, nominal sample position 19.76 m, and the current reference
flight path **19.95 m** (Pavon-Rodriguez, Figs. 10–14), which is the number the
render and the slide use.

**Chamber design** (comment above slide 7): ✅ **done 2026-08-10** —
`mx17_board_peel_zoom.png` is built and in `assets/img/`. The slide markup
change (swap the right-hand `img`) and the **caption pitch correction
(0.7785 → 0.78 mm)** are written up in
[`HANDOFF_board_peel.md`](HANDOFF_board_peel.md); `index.html` was
deliberately not edited. The HTML comment above slide 7 can go once the swap
is made.

**Cosmic bench** slide: flagged in-slide as pending the running-order
decision with Alexandra's P2 talk.

**Efficiency** — **done 2026-08-10, markup ready in
[`HANDOFF_efficiency.md`](HANDOFF_efficiency.md), not yet applied to
`index.html`.** All five numbers were re-derived on the current
(waveform-first) chain: det3 92.9→**93.5**, det2 91.3→**91.9**, det6
57.8→**75.4**, det7 43.1→**56.9**, det4 20.7→**41.9 %**. The three
spark/gain-limited chambers move by 14–21 points, so the old fleet bars and
the old caption are both wrong, not merely stale — det4 in particular now
*detects* 95.8 % of crossings, not "~70 %". The breakdown figure is
regenerated and self-consistent (annotation derived, no literals); the
residual pairing is a new tail-focused figure, `efficiency_residual_tail.png`,
rather than a crop of `spatial_residuals.png`, because that image is
hit-chain vintage and would not match the refreshed breakdown's basis.
Both blocks of markup from the handoff are now **applied** in `index.html`
(main slide → det3 alone, fleet bars → new backup slide).

**Resolution** — **the old caveat here was wrong and has been withdrawn
(2026-08-10).** It said the tile was reference-limited because the M3 detectors
"only have ~500 µm resolution"; that is the **per-plane** figure. What matters is
the four-plane fit's pointing interpolated to the DUT plane, measured on the same
run at **0.21 / 0.24 mm (X/Y)** — the DUT sits ~50 mm from the telescope
centroid, where interpolation is best. The reference is 10–21 % of the residual
*variance*, so the quadrature subtraction is stable to < 0.01 mm and the tile now
carries the deconvolved **0.57 / 0.69 mm**. Do not call this measurement
reference-limited on stage. Derivation and sources:
[`HANDOFF_resolution.md`](HANDOFF_resolution.md), which also documents two
pre-existing repo errors it uncovered (a "0.40 mm M3 pointing floor" quoted in
three places that is actually the *deconvolved DUT* value, and a radial-vs-per-axis
units mismatch in `m3_self_resolution/analyze.py`).

*Still open on this slide, and worth doing before 3 September:* both its figures
are superseded. `spatial_residuals.png` is hits-basis from `g_det3_wknd`, a run
never reprocessed after the 2026-07-25 significance-floor fix — regenerating from
`sat_det3` on the waveform-first basis (the basis the rest of the deck now uses)
gives **≈ 0.50 / 0.47 mm**, better than the slide claims. And
`angular_resolution.png` is hits-basis at 1.66° while the tile advertises the
waveform-first 1.0–1.1°; the tile states both honestly today, but regenerating the
figure would let it lead with 1.1°.

**Status** section — **built 2026-08-09, 11 main-flow slides + 3 backups.**
Everything on them is sourced; `STATUS_PLAN.md` is the map. What is still open:

- **Slide D9 (the physics reach) is DECIDED and REMOVED — 2026-08-10.** Dylan
  chose `STATUS_PLAN.md` §5 framing **option (c): say nothing about yield**,
  because there will not be enough analysis by 3 September to support a reach
  statement. The talk is framed as a status report with analysis ongoing. The
  slide's markup is in git history (last present in `d3e4993`) and its content is
  unchanged in `STATUS_PLAN.md` §5 if it ever needs to come back. Two knock-ons
  applied: D0's "physics reach is far below projection" bullet became an
  "analysis is ongoing" bullet, and D9's **σ(p)/p ≲ 30 % tracker requirement**
  survived onto the outlook slide — it is a detector statement, not a yield
  statement. The thermal-window rate table stays in **backup**; having it ready
  for a question is not "saying it".
- **The charge measurement's one systematic is still open** — it assumes the CAEN
  imon readback preserves the time-average of a short current burst. If it does
  not, every charge number is a lower bound. Closing it is item 4 of
  `ntof_july_analysis/flash_charge/HANDOFF_FLASH_CHARGE_2026-08-09.md`. Note the
  "cheapest route is a spec lookup" line is optimistic: **the board model is
  recorded nowhere in any of the repos** (the Python wrapper is generic and the
  crate sits behind a CFE server at 128.141.177.244), so the lookup needs either
  Dylan's knowledge or a crate-map query — and the crate must **not** be touched
  while production data taking is live. A stronger route that needs no hardware
  and no new data: phase-fold imon against the beam-intensity log's per-pulse
  timestamps to measure the monitor's impulse response directly. **That is now in
  flight** (third attempt — the first two were killed by infrastructure failures;
  the time reconstruction had been validated to ~1 ms jitter). It will also deliver
  a deliberately over-detailed **teaching slide** explaining the method, which
  Dylan asked for so he can follow the argument before it is pared back for an
  audience. Watch for `HANDOFF_imon.md`.
- **The target-imaging slide is on PRELIMINARY reconstruction** — one arm, one
  sub-run, 3 of 13 file tags of run_79, on a transferred bench calibration whose
  own document says "nothing here is quotable". Either do the in-situ
  calibration or keep the caveat visible. None of the ~1.7 TB of production
  statistics has been reconstructed yet; a four-arm back-projection over even a
  slice of it is what would turn "the fans converge" into an actual image.
- **It is too long** — and now quantified: the slot is 15 + 5 min against 28
  main-flow slides. The cut is proposed in [`RUNNING_ORDER.md`](RUNNING_ORDER.md),
  which keeps D1, D1b, D5 and D6 in the main flow, demotes D2/D3/D4/D7 to backup,
  and argues for demoting D8 too on the preliminary-calibration grounds below.

**Target slide** (n+³He physics) — **done 2026-08-10.** The ideal-case e⁺e⁻
kinematics figure did not exist anywhere in the repos; it does now, as a Geant4
campaign product rather than a hand-drawn curve — `ideal_pair_spectrum.png` from
`mpgd26/make_pair_kinematics.py` (provenance row above). It is on a new slide 6b,
"What the detector has to separate", after the n+³He channel slide, because the
kinematic confinement of a two-body 17 MeV decay above 109° is the cleanest
argument in the deck for *why this needs a tracker*. **At 15 minutes it is a cut
candidate** — `RUNNING_ORDER.md` treats it as promotable/demotable at Dylan's
discretion. Two things to say if asked: the two curves are shapes normalised
separately (50/50 by construction, **not** a branching ratio), and the IPC curve
is the standard dN/dM_ee ∝ 1/M_ee virtual-photon stand-in with no E0/M1/E2
multipole correlation.

## The strip pitch was wrong everywhere — corrected 2026-08-10

The deck (and the mpgd26 package, and `report.html`) said **0.7785 mm**. That is an
off-by-one: 398.58 mm is the first-to-last strip *centre* span, i.e. **511**
pitches, and dividing it by the strip *count* 512 gives 0.77852. The design pitch
is a round **0.78 mm** and the active area is 512 × 0.78 = **399.36 mm** —
gerber-verified, strip dots at ±0.39, ±1.17 mm, spacing exactly 0.780. Found while
building the zoom figure, which now carries a burned-in 0.78 mm caliper that would
have contradicted the old caption on the same slide.

Corrected in: the slide-7 caption, `scenes_chamber.py` (both the docstring and
`STRIP_PITCH_MM`, which computed it wrongly — now `/(N_STRIPS - 1)`),
`scenes_microtpc.py`, `make_chamber.py`, `make_report.py`, `README.md`, and
`report.html` (patched in place rather than regenerated, so as not to discard
hand edits in it).

**Not affected:** no analysis code uses 0.7785 — grepped across `common/`,
`wft/`, `mx_june_wft/`, `mx_june_cosmic_qa/`, `ntof_tracking/`,
`cosmic_bench_analysis/`. The 398.6 in `common/mx17_active_area.py` is a fiducial
box edge, which is legitimately the strip-centre span. **So no reconstructed
position is wrong** — this was a documentation error only.

*Loose end:* `chamber_exploded.png` and `microtpc.png` were rendered with the old
constant, so their drawn geometry is 0.2 % off. That is invisible at slide size and
the code is now right, so they will self-correct on the next re-render. Not worth a
re-render on its own.

## The plan slide, answered: two questions it invites — 2026-08-12

Both came from Dylan reading the finished figure, and both are the sort of thing
that gets asked from the floor. Numbers below are from `data/ntof_event.json`
itself (event #208), not estimates. **This is ONE event** — do not generalise
either answer into a statement about the setup.

### "110° in space but 122° as drawn — is that scattering in the capsule?"

No. Both numbers come from the *same* primary directions at the vertex: the
generator recorded 110.1°, and recomputing from the direction vectors the figure
actually draws gives 110.2°. Nothing about the material enters.

It is pure projection, and the intuition that projecting can only widen an angle
is not right — it can go either way:

```
cos θ_3D = a⊥·b⊥ + a_y b_y            (unit vectors, y = beam)
cos θ_2D = a⊥·b⊥ / (|a⊥| |b⊥|)
```

In this event **both legs leave going up** (+y components +0.284 and +0.407;
polar 73.5° and 66.0° from the beam), so `a_y b_y = +0.116` — the one term
pulling cos θ *up*, i.e. making the space angle **less** obtuse than the drawn
one. Drop it and renormalise by `|a⊥||b⊥| < 1` and you get 121.7°. Had the legs
gone to opposite sides of the drawing plane, projection would have **shrunk** the
angle instead. Recorded in the `make_ntof_plan.py` docstring too.

### "14.8 of 19.6 MeV seen — where did the other 4.7 go? The MM PCB?"

The PCB is the biggest *dead-material* sink but it is not most of it. Full
budget, by tracking each primary's KE loss region by region (sums exactly to the
19.558 MeV the pair started with):

| Where the energy went | e⁻ (arm D) | e⁺ (arm C) | total |
|---|---|---|---|
| **Seen** — deposits in sensitive volumes | 8.39 | 6.46 | **14.84** |
| ³He capsule itself (gas + Al + CFRP wall, ~13 mm crossed) | 0.63 | 0.50 | 1.13 |
| MM stack dead material | 0.61 | 0.52 | 1.13 |
| air, containers, plastic wrapping (~460 mm of path) | 0.06 | 0.08 | 0.14 |
| radiated out of the plastic/LS and never re-absorbed | 1.06 | 1.27 | 2.33 |
| | | | **19.56** |

Inside that MM row, per leg (e⁻ arm D / e⁺ arm C), read off a 0.2 mm dE/dx
profile through the stack rather than from volume names:

| | e⁻ | e⁺ |
|---|---|---|
| **readout board laminate** (1.7 mm of Cu/FR4/kapton, w = 35.0–36.8 mm) | **0.507** | **0.421** |
| Rohacell backing (5 mm, w = 36.8–41.8) | 0.048 | 0.048 |
| back mylar/Al foil (w ≈ 41.8, 40 µm) | 0.003 | 0.003 |
| mylar + Al window, drift cathode | 0.012 | 0.014 |
| **the 30 mm of drift gas** | **0.046** | **0.032** |
| gas behind the mesh (w = 30–35, dE/dx ≈ 0) | ~0 | ~0 |

So the chamber's whole material cost to a 10 MeV electron is the **readout
board** — 0.93 MeV of the 1.13, about a fifth of the total deficit — and the gas
it actually measures in costs 0.08 MeV. Everything else is foils.

⚠️ **The 8 mm Al support plate is NOT in this budget, and must not be put in it**
(Dylan, 2026-08-12). It is built by `addRing` in
`MX17_Geant/shared/MX17ModuleGeometry.hh` — a plate with a **402 mm square
through-aperture concentric with the active area** (`AsBuiltSpec`:
`plate_mm = 8.0`, `plateAp_mm = 402.0`), i.e. a frame around the outside with
nothing over the active area. These legs cross at |u| ≤ 83 mm and |v| ≤ 92 mm,
deep inside the ±201 mm aperture, and the profile shows air (0.1–0.2 keV/mm)
from w = 42 mm out. A track that did cross 8 mm of Al would lose ~3.7 MeV —
i.e. if the plate ever appears in an energy budget, something is wrong.
An earlier draft of this note credited it with 0.14 / 0.11 MeV; that energy is
the Rohacell and the foil behind it.

**The dominant term is escaping photons, ≈2.3 MeV — half the deficit.** Mostly
arm C: the e⁺ spent 5.97 MeV in the 25 mm plastic, arrived at the liquid with
~1.2 MeV, stopped in its first 2.2 mm, and arm C's LS recorded only 0.077 MeV.
The two 511 keV annihilation gammas evidently left as well — and note they are
*extra* energy, not part of the 19.56, so they do not close the gap either.

Rebuild the table: the region-by-region KE decomposition is a few lines over
`ev['legs'][i]['ke']` and `['layers']`; `layers` labels every step, and `'world'`
means any non-sensitive volume.

## The SiPM dead bars were mirrored — corrected 2026-08-12

Dylan, off the top-down slide: "the SiPM bars seem to be shifted in the wrong
direction — there should be one unread on the left and three on the right,
looking from the top and behind the MMs." Correct, and the deck had it backwards.

The 16-of-20 read-out window is shifted **one bar toward the MM**, and the MM is
pinwheel-shifted along **−u**, so the window goes to −u: **live bars 1–16**, dead
**{0}** on the MM side (left, seen from behind the wall) and **{17,18,19}** on the
far side (right). `DetectorConstruction.cc` and `mx_july_beam_qa/mx17_geom.py`
both had that sign; `MX17_Full_Geant/scripts/plot_geometry.py` — which every
figure in this package imports — had `+shift` instead of `−shift`, so it drew
bars 3–18 and mirrored the dead ones. **Drawing-only: the simulation is
unaffected.** One-line fix in `plot_geometry.py`, dated comment there and a full
write-up in `GEOMETRY_COORDINATE_CONVENTION.md` §6.

Regenerated and re-copied: `setup3d_6_sipm`, `_7_plastic`, `_8_plastic_top`,
`_9_full` (slides 26–29), `ntof_plan.png` (slide 30), the backup
`setup_2_sipm` / `_3_plastic` / `_4_full` (slide 51, from `plot_buildup.py
--style clean`), and the `build_ntof` / `turn_ntof` animations in `report.html`.
`setup_topdown.png` does not draw individual bars and was left alone.

⚠️ **Still unverified against hardware.** Sim, analysis, figures and the written
convention now agree, but they are all the *same* convention — no measurement has
confirmed which side the real dead bars are on. The check is to project
reconstructed MM tracks onto the wall and see which group fires; logged as an
open item in `GEOMETRY_COORDINATE_CONVENTION.md` §6, with the σ ≈ 47 mm pointing
blur caveat. Nothing on the slides depends on it — the wall is drawn, never used
for a number — so this is not a blocker for 3 September.

## Dylan's review of the two new renders — 2026-08-10, in flight

Both were reviewed and both got a verdict of broadly good with specific changes.
Sessions are working on these now; if they are not reflected in the figures on
disk, they did not land.

~~**EAR2 beam line** — split into **three progressive frames** that are strict
subsets of one final picture (1: lead target alone · 2: + the broken ~20 m flight
path, pipe and beam dump · 3: the full picture), same camera and scale throughout.
In the full frame: the beam pipe should **end not far above the floor**, the
pillar-like supports around the station come **out**, and the capsule should
**float above the top of the beam**.~~ **DONE 2026-08-10, then SUPERSEDED
2026-08-11** by a second round of review on the result — see below. Its guess
that "the real pipe continues through the roof to the dump" was **wrong**, and
the resulting cut-away-pipe disclaimer is gone.

**EAR2 beam line, round two — 2026-08-11. DONE.** Four changes, all landed:

1. **The beam pipe genuinely ends** about a metre above the EAR2 floor, and the
   hall above it is open space for experiments — so it is drawn ending, in a flat
   circle with a flange, and the neutrons cross the hall in air through a bore in
   the roof slab. This **replaced a factual error**, not a styling choice.
2. **One extra frame**, so the sequence is target → collimation and the beam
   arriving in EAR2 → the beam dump → the measuring station. The station appears
   **only on the last frame**.
3. **The first collimator is named**, on the break label and in frame 2's bullet:
   7.4–8.4 m above the target, ~10 m *below* the EAR2 floor, 1 m of iron with a
   200 mm bore. It is inside the lower break, which is why the drawing alone
   invited the question.
4. A slide-copy label was **clipped** by the fix in 3 (`annotate` does not wrap);
   every line in `LABELS_SLIDE` is now ≤ ~21 characters and `make_ear2.py` says so.

~~**Board peel zoom** — drop the **pillars** from the resistive-strip band, and on
the two strip layers **suppress the vias and draw the strips themselves** as plain
horizontal / vertical lines, because as gerber artwork "it isn't immediately obvious
what is going on". ⚠️ That makes two of the four bands **schematic rather than
gerber copper**, so the figure's "real gerber copper" title has to be narrowed and
the pre-existing note about the unresolved dot-and-stub interconnect must stay.~~
**DONE 2026-08-10** — asset regenerated, title narrowed to name artwork vs
schematic per band, interconnect note kept. `index.html` untouched; if the
fig-label should change, the exact replacement text is in
[`HANDOFF_board_peel.md`](HANDOFF_board_peel.md) §4.

## Still to do

- **Review the nine-frame setup build-up.** `scenes_ntof.py` / `make_ntof.py` →
  `setup3d_1..9_*.png`, now on deck slides 16–24. It **is** in the main flow, but
  it was written by a session that crashed before reporting, so **nobody has
  reviewed the wording** and there is no handoff note for it. The renders and the
  provenance are sound (geometry imported from the simulation's own
  `plot_geometry.py`, real Geant4 neutron and pair, and slide 18's caption admits
  the pair frame is two events spliced at the vertex). Two captions asserting the
  `~10⁻⁸` radiative branch have already been removed per the no-yield decision.
  **This is the largest unreviewed block in the deck** — and at 15 minutes it has
  to collapse to about two frames anyway, so review it with the cut in hand.
- **Slide 9's two figures are superseded** — see the Resolution item above.
  `spatial_residuals.png` is hits-basis from a run never reprocessed after the
  2026-07-25 fix, and regenerating from `sat_det3` on the waveform-first basis
  gives a *better* number (≈ 0.50/0.47 mm); `angular_resolution.png` is hits-basis
  at 1.66° while the tile advertises 1.0–1.1°.
- **The bench slide's running-order flag** — waiting on whether Alexandra's P2 talk
  covers the cosmic bench. Unknown as of 2026-08-10.
- ~~**The imon systematic**~~ — **CLOSED 2026-08-10.** Measured, not looked up: the
  readback is a ~1 s averager, the charges are measurements. Two teaching backup
  slides added. One residual, which is a *different* question from the one that was
  open and which nothing in the talk depends on: the absolute nA scale of the
  readback is uncalibrated, and that needs the CAEN card model.
  **Caveat worth knowing:** run_157/158/159 are not in the local mirror and the DAQ
  and lxplus were unreachable, so this was measured on **run_79** — the same
  production setpoint, and its det C charge (97–98 nC) agrees with run_158's det C
  (97–101 nC). Worth re-running when the August tree is back on disk.

### Done, for the record

- **EAR2 beam-line scene** (2026-08-10, revised three times on 2026-08-11):
  `scenes_ear2.py` + `make_ear2.py` → `assets/img/ear2_onfig_{1_target,
  2_collimation,3_full}.png` on slides 5–7 and `ear2_beamline_{…}.png` on the new
  backup slide, EAR1 dropped, beam dump identification verified. **Split into a
  three-frame build-up per Dylan's review** (it was four for a few hours),
  together with the legibility changes he asked for (support columns removed,
  capsule floating), one **factual correction — the lower beam pipe ends ~1 m
  above the EAR2 floor** and the beam crosses the experimental space in air — and
  then the **ceiling and beam dump removed from the drawing**, which took
  `DRAWN_H` from 8.67 m to 5.71 m and made the station ~1.5× larger. Third round:
  the **line inside the hall modelled from the photograph** (polyethylene
  shielding, stepped diameters, reducer), the **wide upper pipe** to the dump
  restored as a separate section cut off by the frame, the **break's section
  marker fixed** (Dylan's "pink diagonal thing"), the **caption moved to a backup
  slide** so the pictures get the height, and a **second label layout** built
  (`ear2_onfig_*`, labels on the figure's own background) which is what the main
  slides now use. The 08-10 "pipe cut away above the floor" drawing device and its
  disclaimer are both gone; the support frame is the only as-built departure left,
  disclosed on the figure and in the backup caption, and now also visible in the
  restored **CERN photograph of the hall** beside the render (see the facility
  section above for the configuration caveat that photograph carries).
  Note `RUNNING_ORDER.md` demotes these slides to backup **as a unit** at 15
  minutes, so their value is as backup figures and as the replacement for the
  deck's weakest borrowed image.
- **Zoomed board-peel figure** (2026-08-10): built, swapped onto the chamber-design
  slide, and the caption's pitch corrected. **Second pass done the same day** per
  Dylan's review — pillars removed from the resist band, L5/L6 vias suppressed and
  the strips drawn as continuous lines, and the "real gerber copper" title narrowed
  because those two bands are now schematic.
- **The sharing kernel behind slides 9, 9b, 9c and the reconstruction backup was
  wrong, and is now corrected** (2026-08-18). Every kernel figure in the deck was
  drawn from `calib_bundle_lp2_t0p`, which carries **c₂/c₁ = 1.14** — the ±2 copy
  larger than the ±1 copy, which cannot happen on a resistive film, since the ±2
  strip is reached only through the ±1 strip. Measured properly at H4 by the
  cross-relation (no deconvolution, no regulariser): **0.45 ± 0.02**, invariant
  over a 2.6× range of drift field; near-vertical det3 cosmics give 0.63 ± 0.09.
  det3 was refit with the ratio pinned (`calib_bundle_r06`, `c2_over_c1` = 0.6,
  a new gated branch in `wft/model.py`), and everything calibration-derived in the
  deck was regenerated on it:
  - `make_share.py` → `share_{cartoon,kernels,build,decompose}.png`. Its
    `_kernels()` became `_amps()`, which mirrors `build_matrix`: on a
    ratio-slaved bundle the stored `c2` is **0.0** and reading it directly draws
    no ±2 copy at all. `main()` now refuses to draw c₂ > c₁.
  - `docs/wft_reference/figsrc` via the new **`WFT_DOC_BUNDLE`** env override
    (the document itself still defaults to the frozen products, on purpose) →
    `wft_{model_vs_data,design_matrix,nnls_profile,chi2_surface,global_start,
    template_build,degeneracy,seeding,candidates,gallery}.png`.
  Text that moved with the figures: **delays 146/291 → 166/333 ns** (τ_s refitted
  with the ratio pinned), 9b's "half the pulse" → **40 %**, and 9c's denominator
  ("the full 7,093-event run" → **6,852 / 6,850 reference-matched**; 7,093 is the
  reconstructed count, not the resolution's denominator).
  ~~**Not regenerated, deliberately:** the ensemble figures … read
  `events.parquet` … the correction is free in resolution~~ — **superseded
  2026-08-19, and both halves of it were wrong.**

  **2026-08-19 — the mix is gone; det3's whole chain re-run on r06.** Reco →
  w0/kw re-measured from that reco → alignment → efficiency (×3) → angles →
  maps → digest, all on `calib_bundle_r06`, then promoted with
  `mx_june_wft/23_promote_r06.py` (frozen tree parked once in
  `pre_r06_backup_20260819/`, `--revert` restores). det7 and det2 the same.
  Regenerated: `wft_angles`, `wft_implied_v`, `wft_compression`, both
  `wft_deconv_*` (from `15_explain_figures.py --bundle …r06`),
  `angle_correlation` and `angle_resolution` (`make_resolution.py`).

  - **`wft_implied_v` and `wft_compression` never needed the event table.**
    `f_hits.py` runs off the bundle and the calibration cache, so they were
    regenerable via `WFT_DOC_BUNDLE` all along. Only `wft_angles` (`f_valid.py`)
    and the two `wft_deconv_*` actually read the reco.
  - **The correction is NOT free in resolution.** The "< 0.6 σ" was 220 held-out
    bench events. On 6,850, paired: σ₆₈(Y) **1.165 → 1.226°** (+0.061 ± 0.013),
    almost all of it head-on (|θ| < 5° band 1.22 → 1.43). X is unchanged.
    Efficiency, position and alignment do not move at all.

  **Numbers that moved on slides:** σ₆₈ **1.19 / 1.16 → 1.19 / 1.23°** (9c
  caption, the stat tile, both alt texts, the provenance cross-check comment);
  the per-|θ|-bin range 0.94–1.48 → **0.98–1.71**; implied-v flatness ~1 →
  **1.1–1.4 µm/ns**; the reconstruction-backup slide's gated σ_θ 1.15/1.14 →
  **1.16/1.13** and its pull width 1.19 → **1.16** (X).

  **Why ship a worse number:** the ordering is the physics — c₂ > c₁ is not an
  RC cascade, and the H4 beam measures 0.45 ± 0.02 model-free. It is a trade and
  the deck now states it as one. Gate table, per-detector χ² cost and the det2
  discovery: `mx_june_wft/R06_GATE_2026-08-19.md`. Worked example:
  `mpgd26/walkthrough/` and
  <https://dylan-neff.web.cern.ch/notes/forward-fit-det3.html>.
