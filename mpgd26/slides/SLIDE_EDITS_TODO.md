# Slide edits — the working to-do list

**This is the document we work from for MPGD 2026 deck edits** (started
2026-08-16). Everything that changes `slides/index.html` from here on gets an
entry here first: what to change, why, and — once done — what was done.

Related documents, and what each is for:

| | |
|---|---|
| `STATUS_PLAN.md` | the Status section's argument and its plan |
| `RUNNING_ORDER.md` | the 15-minute cut — *which* slides are shown |
| `NOTES.md` | the running log: provenance, figure sources, decisions |
| **this file** | *what still needs editing*, and the principles the edits follow |

`NOTES.md` stays the log — record the provenance of anything new there. This
file is the queue.

---

## Core principle: talk with the figures, not with the text

Physics conferences are a *visual* medium. The audience reads a slide or listens
to the speaker, never both, so every line of text on a slide is a line that
competes with what is being said. **Dramatically reduce slide text.**

Working rules:

1. **A slide is a figure with a title.** The default slide is one visualization
   filling the frame. Text earns its place; it is not the default state.
2. **At most three short bullets, and only when the figure cannot carry the
   point.** A bullet is one line — if it wraps to two, cut it, don't shrink it.
3. **No paragraph captions in the main flow.** The long explanatory captions in
   this deck were written to make the HTML readable as a document. They are good
   *notes* and bad *slides*: move them to speaker notes or a backup slide, and
   leave a one-line label on the figure.
4. **References stay, but stay light** — one small muted line under the figure,
   never a highlighted block.
5. **Numbers go in stat tiles or on the figure**, not in prose.
6. **What is being said is not on the slide.** If a sentence is one you will
   speak anyway, it does not need to be printed.

Backup slides are exempt — they exist to answer questions and can be as dense as
they need to be.

---

## Queue

### Done

- [x] **The Status section rebuilt into one argument** (2026-08-19, fourth
      batch). Full write-up in `NOTES.md`; the four things worth remembering:
      **(a) two frames of one drawing beat two drawings.** Slides 27 and 31 are
      the same figure with the dead band added and the accent moved. Nothing
      shifts between them, so returning to it three slides later costs the
      audience nothing and the argument lands in the picture.
      **(b) joining two figures is only worth it if the join is drawn.** The
      timeline and the event census share a canvas because the events panel is
      the timeline's last bar opened up — and there is a wedge saying so. Two
      plots that merely sit above each other should have stayed two slides.
      **(c) `bbox_inches=None` does not turn off a tight bounding box** — it
      falls back to `savefig.bbox`, which `plotstyle.use()` sets to `'tight'`.
      Name the canvas: `bbox_inches=fig.bbox_inches, pad_inches=0.0`. And once
      you do, every axis label has to fit inside the margins you chose.
      **(d) an acronym has to be introduced where it is load-bearing.** DREAM
      came off slide 26 (Dylan: "just say Events recorded") and onto 28, whose
      title and first caption clause now say what it is. It had never actually
      been defined in the deck.
      Cut slides went to backup in the order the questions get asked, and the
      **"Proposed · 13 Aug" block is retired** — six of its slides into the
      flow, two into backup.

- [x] **Eleven edits in one pass** (2026-08-18, third batch). Full write-up in
      `NOTES.md`; the short version, and the four decisions in it worth
      remembering:
      **(a) 4.4 cut** — a build frame whose side column is byte-identical to the
      previous frame's is carrying a picture, not an argument.
      **(b) slide 6** — subtitle, summary, both spectrum paragraphs and the
      `.fig-label` all off; the row re-measured at 2.028 : 1 and re-flowed;
      beat 4's left block **stacked** to buy the width that pays for pitch
      12.6 → 15.0 and 1.19× on everything. **6.3 is one full-width picture**:
      the micro-TPC cartoon stands in beat 4's box and the spectrum does not
      move. That retires the "what the stack costs" item from the second batch.
      **(c) slide 13** — the binned 2 mm map is now a **sliding 20 mm circle
      stepped 0.5 mm** on the highest-statistics det3 run. 2 mm was asked for
      and is impossible: it holds two muons.
      **(d) slides 15–24** — one or two bullets each, 15→17 accumulating with
      `.dim`; the figure column went 0.92fr → **1.32fr** with
      `align-items:stretch`.

- [x] **`.pin` — a way to point at part of a picture** (2026-08-18, built for
      slide 12). Absolutely-positioned label + leader + arrowhead, anchored on
      **both** sides: the target in per-cent of the *image* (via `.pinwrap`,
      an inline-block sized by the image), the label at a fixed em offset
      outside it, and the leader stretching between. Use it wherever a render
      needs calling out; do not burn the label into the render, and do not
      position a pin in slide coordinates — the picture's rendered width
      changes with the projector.

- [x] **Slides 5–6 fill the page** (2026-08-18, Dylan: *"rework the python
      scripts to make them taller"*). The figure hole on one of these slides is
      **1186 × 547 px** — measure it, do not guess: print the slide with
      `.imgwrap{background:red}` and read the red box off the PDF. That is
      2.11 : 1, and the two story rows were 4.57 : 1 and 3.92 : 1, using 46 %
      and 55 % of the height.
      **A third standing rule, and the one that matters most for these figures:
      a slide figure is width-limited, so the only lever on how big its type
      and its drawing come out is how many canvas units it spans.** 160 units
      across 12.4 in renders 9 pt type at 7 pt; 124 units renders it at 9 pt.
      Making a figure taller and making it bigger are the same operation:
      re-flow the content into a narrower, taller box. Hence
      `scenes_x17.SW = 124`, every beat re-flowed (beat 2 now reads *downwards*,
      the vessel is drawn at 0.40 instead of 0.245, the boost columns are at
      pitch 12.6 with β/γ stacked), rows at **2.16 : 1**, everything ~29 %
      larger, and **no font size changed**.
      ⚠️ **It cost frame 6.3.** Two full-width pictures stacked in one figure
      box can never be more than ~59 % as wide as one, so the frame that shows
      the spectrum *and* the cartoon now shrinks both to that; the row weights
      were re-derived (`1fr / 0.715fr`, i.e. 1/aspect) so at least they are
      equal width. If it is too small in the room, stop stacking — cartoon
      alone on 6.3, or swap the cartoon for beat 4 inside the story canvas.
      Both options are written out in `NOTES.md`.

- [x] **Bigger type on slides 1–4** (2026-08-18, Dylan). A `.slide.lead` class
      that redefines `--fs-body` (+25 %), `--fs-caption` (+16 %),
      `--fs-title-sm` (+12 %) and `--fs-kicker` locally, so bullets, captions
      and the outline items all scale together. **A class, not a change to
      `:root`** — the deck's other 78 slides are set for pages that are already
      full, and one of them was open in another session. The outline also takes
      the extra width (34 → 44 em) and air; slide 3 gave the figure column six
      more points of width, since its panels' axis labels are the smallest type
      in the deck. ⚠️ **Every frame of the EAR2 build carries `lead`** — a
      frame that missed it would resize its bullets mid-build.

- [x] **A transition to the Micromegas: new frame 6.3** (2026-08-18). A cartoon
      under the spectrum — the pair leaving at the 110° kinematic minimum, each
      leg crossing a Micromegas drift volume, the ionisation drifting to the
      readout plane — carrying **one sentence**: one gas gap gives a direction,
      two give the opening angle. New figure `scenes_x17.draw_detect`
      (`make_x17.py --layout detect`). ⚠️ **The one build in the deck that does
      not reserve its future row**, on purpose: reserving it costs frames 1–2
      half their figure, because the story band is width-limited and a hidden
      row shrinks the picture above rather than leaving space below. The
      `.fut` rule is about text not jumping while it is read; a figure zooming
      out once, on the frame where attention has already moved below it, is a
      reveal.

- [x] **The chamber, zoomed, with a muon that is not a rod** (2026-08-18).
      `WIN_MM` 60 → **44 mm** across (depth untouched) with `view_angle`
      17.8 → 16.6° so the frame width is unchanged — a magnification, not a
      crop. Track tube 0.9 → 0.30 mm, drift lines 0.28 → 0.10: at 0.9 the muon
      was 1.2 strip pitches across, drawn at the scale of the structure it
      crosses. Fixed on the way past: `align-items:center` on that slide's
      `.cols` was pushing both `.fig-head`s up through the title rule — with
      centre alignment a grid item is only as tall as its content, so
      `.imgwrap`'s `flex:1` has nothing to resolve against. **Use stretch (the
      default) on any `.cols` whose columns carry a `.fig-head`.**

- [x] **The 3-D event display goes to backup** (2026-08-18, Dylan). It is
      hits-basis and it sat between the forward-fit slides and the efficiency
      numbers. Retitled in backup so the caveat is on the page. The question it
      came with — *can the forward model make one?* — is answered in the
      comment where it used to sit: the segment and the depth profiles yes, the
      **point cloud no**, because the model's charge is q_x(z) and q_y(z) and
      their outer product is not a measurement.

- [x] **Efficiency: the loss budget is markup** (2026-08-18, Dylan). Five bars
      in HTML (`.bar-chart.loss`) instead of a matplotlib PNG whose
      sentence-length labels arrived smaller and greyer than the deck's own
      body text. Same JSON, same colours as the figure's `ROWS`. "Spark" →
      **"discharge"** on the page (the JSON key stays `spark_cat`). New
      footnote on the >5 mm slice, and the right column is the r < 2 mm map
      plus the |r| tail. ⚠️ **The slide's numbers had been stale since
      2026-08-13** — 93.3 % on 7,049, not 93.5 on 7,055; the backup fleet bars
      were stale the same way and are refreshed.
      **A second standing rule falls out of this: size a matplotlib figure near
      the size it will be displayed at.** The tail figure went 7.4 → 5.6 in
      wide because at a fifth of the slide width its tick labels were 4 px.

- [x] **Resolution: three measurements, each named** (2026-08-18, Dylan). Both
      old figures were 2026-07-14 *hits-basis*, and the slide showed three
      different numbers for one quantity (1.66° on the figure, 1.7° on the
      tile, 1.0–1.1° two slides earlier). Now the reconstructed-vs-reference
      density for X and Y, and σ₆₈ against angle — new module
      `make_resolution.py`, computed through `03_angles.py`'s own accounting.
      Position comes from **the SPS beam** (176 ± 35 µm, det4) because the
      bench cannot give it honestly; timing stays on the bench because it is
      already telescope-free. The ±35 is deliberately pessimistic and is
      assembled in `spatial_band()`. Slide 11's σ was moved to the same
      estimator so the deck quotes one number for one quantity.

- [x] **The exploded chamber goes deeper, and the board-peel view loses its
      title** (2026-08-17, third and fourth passes). `WIN_MM` 18 → **34 mm**
      along the strips, so the layers read as *planes* and not ribbons — camera
      untouched, as asked. `plot_mx17_model.fig_peel` grew `bare=True`
      (`--only peel_slide`): no title band, no "deeper into the board" arrow,
      new asset `mx17_board_peel_slide.png`. Both figures now carry a
      **`.fig-head`** — a new CSS class for a heading *above* a figure, in the
      deck's own type. **New standing rule: a figure destined for a slide has no
      burned-in matplotlib title.** The title is HTML; the band goes to the
      picture.

- [x] **Micro-TPC operation: the waveforms, and no type bands** (2026-08-17).
      The right panel is the raw per-strip waveforms rather than the
      first-arrival ladder — the ladder is the estimator the next two slides
      replace. `compose(bare=True)` drops the title band and the caption
      paragraph (36 % of the height between them) and the slide's own
      `.caption` came off with them. It had said **"882 ns full-gap transit"
      beside "v_drift = 36.6 µm/ns", which are inconsistent** (820 ns is the
      measured value's answer; 882 is Magboltz's). The figure computes it now.

- [x] **Slides 9 and 10 rebuilt around figures** (2026-08-17) — the big one.
      Slide 9 is a two-frame build: the sharing mechanism, then **the kernels
      production uses**, X and Y. Slide 10 is the four-stage diagram of the
      model beside **the same split on real data** — four consecutive strips,
      each waveform stacked into own / ±1 / ±2. One new module,
      `make_share.py`, four figures, **one colour rule across all of them**.
      Dropped: `unsharing_depth_bias` (a hit-level repair for an abandoned
      basis), two thirds of `charge_sharing_schematic` (the same), slide 10's
      four bullets and its callout, and a fig-label quoting hit-level sharing
      numbers while the fit used different ones. `NOTES.md` has the full entry
      **including the caveat to read before quoting a kernel amplitude**.

- [x] **Merge the two ATOMKI motivation slides into one** (2026-08-16).
      Old slide 3 ("A possible new boson at 17 MeV", three bullets + a 17 MeV
      stat tile) and old slide 4 ("The ATOMKI evidence", the angular-correlation
      figure + a five-line caption) are now one slide: the bullets, cut to three
      short lines, over the bump figure, with a single muted reference line.
      - The ⁴He-is-our-channel bridge, which was the long caption's real payload,
        survives as the last bullet — it is what sets up the next slide.
      - Dropped: the "17 MeV" stat tile (the title already says it) and the
        caption prose about IPC background and fit colours (speak it).
      - The old slide-4 caption text is preserved verbatim in an HTML comment
        above the merged slide, so nothing is lost if we want it in backup.
      - **Revised the same day** on Dylan's read: **text left, figure right**;
        bullet 1 carries the years 2016–2022; bullet 3 is now the question the
        talk asks — "Can we make an independent measurement — the same ⁴He*
        state, entered as n + ³He?"

- [x] **The EAR2 facility build: text cut, and it is now one slide** (2026-08-16).
      Four paragraph-length bullets became **six short lines**, identical on
      every frame — seven plus a callout on the first pass, cut again on Dylan's
      read. The γ-flash hand-off callout came off with it ("not time for this
      yet"): it is spoken now, not printed, and **frame 4 adds no new line** as a
      result. Every number the old bullets repeated —
      7.4 m, 15.0–18.0 m, 70 → 21.8 mm, 18.16 m, 23.66 m, 24.73 m — is already
      labelled **on the render**, so the text was saying the picture's lines out
      loud. The text column went 1.04fr → 0.80fr and the render took the width.
- [x] **Overlay builds** (2026-08-16). Two mechanisms landed with that edit and
      both are reusable — see the comment block in `index.html`'s `<style>`:
      - **The text does not move.** Every frame carries the *whole* text and
        hides what has not been narrated yet with `.fut` (`visibility:hidden`,
        so the line keeps its space). Lines appear below; nothing above shifts.
        The text column is top-aligned (`.side.top`) so the first bullet starts
        at the top of the frame instead of centred in it.
      - **A build costs one slide number.** `slide bstart` + `slide bcont` and
        a `data-frame="n"`: the frames number **4.1 … 4.5**, count once in the
        on-screen `x / N`, hold the progress bar still, and print the same way
        (`make_pdf.sh` mirrors the rule). The deck is **77 slides in 81 pages**.

- [x] **Slide 16 split into two** (2026-08-17, exploratory — Dylan's call
      whether it stays). Beats 1–3 on one slide, beats 4–5 on the next, on the
      `--no-title` variants of the layouts `scenes_x17` already had. Deck is 78
      slides in 82 pages. **Open question for the review:** each row is
      width-bound and fills ~55–60 % of the slide height, and no rearrangement
      of the same beats recovers it (measured) — living with the white band,
      redrawing beat 4 narrower, or going back to the single slide are the three
      options.

- [x] **…and moved to slide 5, as two builds** (2026-08-17). The two story
      slides replaced the compact "n + ³He → the n_TOF search channel" slide and
      became **5.1–5.3** and **6.1–6.2**, revealing a beat at a time.
      `draw_story(upto=N)` draws each frame on the row's own full canvas, so the
      frames are strict subsets and nothing moves. **5.3 ends on "Detect the
      e⁺e⁻ pair!"** — the hand-over into the Micromegas section.

- [x] **The spectrum on 6.2 is a stack** (2026-08-17). Was the two channels
      overlaid at unit peak; now a small X17 yield sitting **on** the IPC
      background, which is what the measurement will look like. The ratio it
      has to assert is a declared parameter — `scenes_x17.SIG_FRAC` = **4 % of
      the IPC yield**, printed on the panel in words and marked illustrative —
      and the window starts at 40° so the forward IPC peak, eight times the
      yield at 109°, does not flatten the bump. `x17_signature` keeps the
      unit-peak comparison on purpose.

- [x] **The exploded chamber is landscape** (2026-08-17). "Chamber design" gives
      it **63 %** of the width and the render was rebuilt to want it: a
      **120 × 30 mm** rectangular window on the chamber instead of a 30 mm
      square (`scenes_chamber.WIN_MM`), labels **on the render down its left
      side** next to their own layer instead of in a gutter column, and the
      label text cut to fit ("Drift cathode", "charge spreading"). The layers
      come out ~2.5× bigger. The board-peel view keeps the right-hand 37 %,
      unchanged. ⚠️ **The column weights and the figure's aspect are one
      decision** — change either alone and a white band comes back.
      - **Second pass the same day:** zoomed to a **60 × 18 mm** window so
        the strip structure resolves (`EXPLODE` 19 → 7.5 with it — the two
        set the aspect between them), columns 63/37 → **56/44**, and the
        readout side re-sourced from **MX17_Geant**: the **L4 pad layer**
        was missing entirely, the resistive strips are **black on their own
        0.80 mm pitch** (550/250 µm, not the 0.78 mm readout pitch), L5/L6
        were in the wrong order, and the PCB is 1.70 mm, not a slab. The
        four readout layers now carry the board-peel figure's colours and
        L-numbers, so the two figures on the slide agree.

- [x] **Slide 7 kicked to backup** (2026-08-17) — the redundancy the move
      created, closed. "What the detector has to separate" is now
      **Backup · X17 kinematics** next to the ATOMKI backup pair. The flow
      states the 109° bound once, on slide 6; the backup slide keeps the
      single-lepton-energy panel (made nowhere else) and the honest caption
      about capsule material, which is the question it answers. Deck unchanged
      at 77 slides / 84 pages.

### Next up — the rest of the X17 introduction

- [x] ~~**Slide "n + ³He → the n_TOF search channel"**~~ — gone: the builds
      replaced it (2026-08-17).
- [ ] **The nine-frame 3-D setup build (pages 16–24)** — the obvious next user
      of the overlay mechanism: nine frames for one slide, and its text has not
      been cut yet. Doing it would take the deck from 77 slides to 69.
- [x] ~~**Ideal pair kinematics slide**~~ — its paragraph caption is no longer a
      main-flow problem: the slide is in backup, where dense is allowed and the
      caption is the answer to the question. Left verbatim (2026-08-17).

### Available now — the ³He story, one beat per file

`make_x17.py --layout beats --capsule` writes each beat of slide 16's
compilation (`x17_story_capsule`) as its own PNG **and PDF** in `figures/`:

| file | beat |
|---|---|
| `x17_beat1_beam[_capsule]` | a neutron beam on ³He (the vessel, with `--capsule`) |
| `x17_beat2_capture` | capture makes ⁴He\* |
| `x17_beat3_channels` | three ways to shed it |
| `x17_beat4_boost` | the boost is what makes the difference |
| `x17_beat5_spectrum` | so this is what we look for |

They are the *same drawing cropped*, not redrawn — an edit to a beat lands in
the compilation and in its own file together. Each keeps its row's full height
so beats used one after another sit in the same register (`--tight` trims to
the ink instead). **This is what slide 16 needs to become a build** — the
overlay mechanism above supplies the numbering and the still text.

### The corrected sharing kernel (2026-08-18) — DONE except where noted

Package: `mpgd26/walkthrough/`; note:
<https://dylan-neff.web.cern.ch/notes/forward-fit-det3.html>. The deck's kernel
was `calib_bundle_lp2_t0p`, which carries c2/c1 = 1.14 — a ±2 copy *larger* than
the ±1 copy, which cannot happen, since the ±2 strip is reached only through the
±1 strip. Everything calibration-derived now runs on `calib_bundle_r06`
(ratio pinned at 0.6; H4 measures 0.45 ± 0.02 head-on and model-free).

- [x] **`make_share.py` repointed** at `calib_bundle_r06`, and `_kernels()`
      replaced by `_amps()`, which mirrors `wft.model.build_matrix`: on these
      bundles the stored `c2` is 0.0 and the real value is slaved from
      `c2_over_c1`, so the old code would have drawn **no ±2 copy at all**.
      `main()` now refuses to draw a kernel with c2 > c1. All four figures
      regenerated (cartoon, kernels, build, decompose).
- [x] **Slide 9 text follows the refit:** the delays are **166 / 333 ns**, not
      146 / 291 — τ_s moved when the ratio was pinned and everything else
      refitted. Both build frames, plus the alt text. Amplitudes are now
      quotable: Y 15 % / 9 %, X 5 % / 3 %, kY = 2.9.
- [x] **Slides 9b/9c figures regenerated** from `docs/wft_reference/figsrc` via
      the new `WFT_DOC_BUNDLE` env override (the doc still defaults to the
      frozen products, deliberately). Also the calibration-derived backup
      figures: design_matrix, nnls_profile, chi2_surface, global_start,
      template_build, degeneracy, seeding, candidates, gallery.
- [x] **9b's fig-label:** "half the pulse stops being this strip's charge" →
      **40 %**, which is what the regenerated decomposition measures (20 % on
      the core, 42 % three strips out).
- [x] **9c's denominator:** was "the full 7,093-event run"; 7,093 is the
      *reconstructed* count and the resolution is measured on the **6,852 (X) /
      6,850 (Y)** that also have an M3 reference. Fixed on slide 9c and on the
      backup validation slide.
- [x] **Ensemble figures done too (2026-08-19) — the section is no longer
      mixed.** det3's full chain was re-run locally on `calib_bundle_r06` and
      promoted (`23_promote_r06.py`, frozen tree parked in
      `pre_r06_backup_20260819/`), so `wft_angles` and the two `wft_deconv_*`
      come from the r06 reco. `wft_implied_v` and `wft_compression` turned out
      never to need the reco at all — `f_hits.py` runs off the bundle and the
      calibration cache — so the earlier "they read `events.parquet`" was wrong
      for those two. **No condor campaign was needed for the golden key**:
      w0/kw are a post-hoc map from the fitted `w`, so one local reco pass
      (11 min) plus `bench/apply_w0.py` does it.
- [ ] *(optional)* Walkthrough figures 1, 5 and 7 — raw 0/±1/±2 waveforms, one
      slice → five waveforms, per-strip own/±1/±2 decomposition — as
      `Backup · Reconstruction` candidates.

~~**No physics number on the deck moved.**~~ — **true on 2026-08-18, false
after the full-statistics gate on 08-19.** Efficiency, position and alignment
really are unchanged (within 5 mm 93.276 → 93.266 %, core σ_r 0.4473 → 0.4468 mm,
alignment offsets by < 10 µm). **The Y angle resolution is not**: σ₆₈ **1.16 →
1.23°**, paired +0.061 ± 0.013 on 6,850 events, and it is concentrated head-on
(|θ| < 5° band 1.22 → 1.43). The earlier "free in resolution" came from 220
held-out bench events and does not survive full statistics.

**It is still the right kernel to ship** — c₂ > c₁ is not a physical RC cascade
and the beam measures 0.45 ± 0.02 — but it is a **trade**, and the slides now
say so rather than implying the correction was free. Numbers updated: σ₆₈
1.19/1.16 → **1.19/1.23°** (caption, stat tile, two alt texts, the provenance
cross-check), per-bin range 0.94–1.48 → **0.98–1.71**, implied-v ~1 →
**1.1–1.4 µm/ns**, backup slide σ_θ 1.15/1.14 → **1.16/1.13**, pull width 1.19 →
**1.16** (X).

**Also found, and it widens the job:** det2 is inverted at **c₂/c₁ = 1.53** and
was missed — `make_manifest.GOLDEN['mx17_2']` is `longer_run` (`o22_long_det2`),
not `long_run` (`g_det2`, 0.74), and `o22_long_det2` is the det2 key the
efficiency fleet figure uses. Three detectors needed the refit, not two.
Full record: `mx_june_wft/R06_GATE_2026-08-19.md`.

### Text-reduction sweep (the standing job)

- [x] ~~The Status section (D0–D10)~~ — **done 2026-08-19** with the rebuild;
      the argument was settled and the text cut in the same pass, which is why
      it was left until last.
- [ ] Walk the remaining main-flow slides in order and apply the rules above.
      The setup build (16–24) had its bullets cut on 2026-08-18; what is left
      is the reconstruction run, slides 7–14.
- [ ] Once a slide's caption is cut, put what it said into the speaker's script,
      not into the void. There is no speaker-notes mechanism in `index.html`
      yet; **decide whether to add one** (an HTML comment per slide is the cheap
      version, and is already the deck's habit).

---

## How the edits get made

Directly in `slides/index.html` (the interleaved rationale comments are part of
the file's value — keep writing them). `slides/edit/` has the browser
click-and-type editor if you would rather edit in the rendered page; it splices
byte spans, so it will not reflow the file.

After a structural change: `slides/make_pdf.sh` still has to split the right
number of pages, and `tools/mirror_slides_to_site.py` still has to resolve every
asset reference.
