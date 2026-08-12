# Handoff — zoomed readout-board "peel" figure

Status: **figure done**, `index.html` **not touched** (as instructed).
New asset: `assets/img/mx17_board_peel_zoom.png`
Provenance row added to `NOTES.md`.

---

## 1. Caption numbers — VERIFY BEFORE YOU PROJECT THIS

The slide caption currently claims:

> 470×470 mm PCB, 398.6 mm metallised, 512 strips/view at 0.7785 mm pitch,
> 30 mm drift gap, 150 µm amplification gap

Checked against the model/gerber source of truth
(`~/CLionProjects/MX17_Geant/scripts/model/mx17_model.py`, which mirrors
`shared/MX17ModuleGeometry.hh`, plus `design/GEOMETRY_FROM_CAD.md` and
`design/NEEDED_INPUTS.md`):

| caption claim | source value | verdict |
|---|---|---|
| 470×470 mm PCB | `PCB_FACE = 470.0` | ✅ correct |
| 512 strips/view | "exactly regular 512 × 512 grid", gerber-verified flash-by-flash | ✅ correct |
| 30 mm drift gap | `DRIFT = 30.0` | ✅ correct |
| 150 µm amplification gap | `AMP = 0.150` | ✅ correct |
| **0.7785 mm pitch** | **0.78 mm exactly** | ❌ **wrong — see below** |
| **398.6 mm metallised** | 398.6 mm is the strip-**centre span**; active envelope is **399.36 mm** | ⚠️ **mislabelled** |

### The pitch is 0.78 mm, not 0.7785 mm

`design/GEOMETRY_FROM_CAD.md:151` and `design/NEEDED_INPUTS.md:182`:

> 0.68 mm pads on 0.78 mm pitch, 512 × 512 over a 399.36 mm active area
> … an exactly regular 512 × 512 grid on a 0.78 mm pitch spanning ±199.29 mm,
> verified flash-by-flash against the gerbers.

The arithmetic shows exactly how 0.7785 arose:

```
512 × 0.78 = 399.36 mm   ← the active area (512 whole pitch cells)
511 × 0.78 = 398.58 mm   ← first-to-last strip CENTRE span  (= ±199.29)
398.6 / 512 = 0.77852    ← 0.7785, the caption's number
398.6 / 511 = 0.78004    ← recovers the true pitch
```

So **0.7785 mm is an off-by-one artefact**: it comes from dividing the
strip-centre span (398.6 = 511 pitches) by the strip *count* (512) instead of by
the number of *gaps* (511). The design pitch is a round 0.78 mm.

This is precisely the sort of thing an MPGD audience checks — and the zoom
figure now has a burned-in 0.78 mm caliper right next to the pads, so a
0.7785 in the caption would visibly contradict the figure on the same slide.

**Recommended caption fix:** "398.6 mm metallised … 0.7785 mm pitch" →
**"399.36 mm active, 512 strips/view on a 0.78 mm pitch"**. If you want to keep
398.6, call it what it is ("398.6 mm between outer strip centres"), but do not
pair it with a per-strip pitch derived by dividing it by 512.

Independent confirmation, straight out of the gerber rather than the docs: the
Ø0.5 mm strip dots in `DFS3498A_L3-TrackY.gbr` sit at x = …, −1.17, −0.39,
+0.39, +1.17, … i.e. **spacing exactly 0.780 mm in both x and y**. So 0.78 mm is
measured, not just declared.

### Bonus fact the zoom now makes visible

The ESL resistive strips are **550 µm wide on 250 µm gaps = 0.80 mm pitch**,
*deliberately not* the 0.78 mm readout pitch (`NEEDED_INPUTS.md:221`, confirmed
2026-08-06). The 0.80 / 0.78 mismatch beats a slow moiré between resist stripes
and pads (`NEEDED_INPUTS.md:266`). At the old full-board magnification this was
invisible; in the zoom the two pitches are separately countable, so the figure
can now carry that point. Both are labelled, with a caliper each.

### Two label errors found and fixed in the figure itself

Neither is in your caption, but both were in the *figure's* depth key, i.e. they
have been on the slide (unreadably small) all along:

1. **L5/L6 strip directions were swapped.** The key said "L5 Y strips — traces
   run along y". It is the other way round: the Y-*measuring* strips sit at
   constant y and run along **x**. Confirmed in the gerbers — `L3-TrackY`'s
   connector stubs are horizontal (dy = 0), `L4-TrackX`'s vertical (dx = 0).
   This is the easy mistake to make from the file names, and I nearly shipped it
   backwards too. Now correct in both the full-board and zoom figures.
2. **Stale resist thickness.** The key read "Geant4: 100 µm slab" long after
   `AsBuiltSpec` dropped to 10 µm (corrected 2026-08-08 in
   `mx17_model.py:PASTE`). It now reads the constant instead of a literal, so it
   cannot go stale again.

### One thing the figure deliberately does NOT claim

The L5/L6 artwork in the gerbers is **Ø0.5 mm dots on the 0.78 mm grid joined by
0.1 mm, 0.39 mm-long stubs** — and the stubs are staggered, present on only
about half to two-thirds of the cells (`NEEDED_INPUTS.md:182` notes the same
"~2/3 of the cells, so it is not periodic"). *(Update 2026-08-10: the zoom figure
now draws those two bands as continuous strips with the vias suppressed — see
**§4b**. That is a labelled schematic and it changes nothing below: the
interconnect is still unresolved.)* I did **not** resolve how the strip
interconnect is actually completed (checkerboard via assignment to X vs Y is the
obvious guess, but I did not verify it). So the captions say "Ø0.5 mm dots,
joined along x / along y" and stop there — they do not assert continuous strip
traces. **If anyone asks how the strips are actually connected, that is a real
open question, not something to improvise an answer to.**

---

## 2. What was built

`fig_peel()` in `~/CLionProjects/MX17_Geant/scripts/model/plot_mx17_model.py`
gained a `zoom=True` mode plus a new `--only peel_zoom` choice. Nothing was
rewritten destructively: the full-board figure keeps its own parameter branch,
its filename, and its place in the default all-figures run, and `--only peel`
was rebuilt end to end to confirm it still works.

Its **layout is untouched**; the only intended difference is the two corrected
depth-key labels described in §1 (L5/L6 directions, 10 µm resist). Those shift
`tight_layout` by one pixel of figure height, so a naive before/after pixel diff
will show everything as "changed" — compare the rendered figures, not the bytes.

Regeneration command (also in `NOTES.md`):

```bash
cd ~/CLionProjects/MX17_Geant && ~/PycharmProjects/nTof_x17/.venv/bin/python \
  scripts/model/plot_mx17_model.py --only peel_zoom
cp design/figures/mx17_board_peel_zoom.png \
  ~/PycharmProjects/nTof_x17/mpgd26/slides/assets/img/
```

MX17_Geant has no venv of its own; the nTof_x17 venv runs it fine. Nothing was
rebuilt on the C++ side.

**`assets/img/mx17_board_peel.png` was also refreshed** — same figure, but it
picks up the two label corrections in §1 (L5/L6 directions, 10 µm resist). Worth
knowing if you diff the assets; regenerate with `--only peel` (slow, ~4 min,
it has no window pre-filter).

Design of the zoom, arrived at by rendering and looking (four iterations):

- **Region 25 × 25 mm at the board centre**, so each of the four peel bands is
  6.25 mm = exactly **8 pad columns**, over 32 rows. That is ~3.6× the linear
  magnification of the full-board figure — enough periods to read a pitch off,
  few enough that each one is separately resolved at 10+ m. Centre of the board
  is a plainly periodic region: no connector fan-out or edge features.
- **Square, not a wide strip.** This is the change that matters most and it is
  about the slide, not the board. The full-board figure is ~2.3:1, but it sits in
  one half of a two-column 16:9 layout — a *portrait* slot. Dropped in at 860 px
  of column width the old figure is only 371 px tall, using well under half the
  available height; that is a large part of why it is illegible. The zoom is
  ~0.88:1 and fills the slot.
- **No separate depth-key column.** It cost ~27 % of the width, and a four-line
  paragraph is not readable from 10 m anyway. Each band now carries a short
  caption on a white plate inside the picture, staggered between two heights with
  leader lines (a caption box is wider than its 6.25 mm band, so at one common
  height neighbours overwrite each other — that was iteration 2's bug). The
  depth-order arrow became a single horizontal "deeper into the board" arrow.
- Copper is drawn from the same production gerbers as the full-board figure, so
  nothing is schematic except the ESL strips, which have no gerber artwork and
  are built from the confirmed 550/250 µm spec.
- Gerber traces are drawn at **true width** (`lw_scale` is computed from the
  points-per-mm of the page rather than hardcoded), so the copper is not
  understated. At this magnification true scale is already legible.
- Two burned-in calipers — **5 × 0.78 mm readout pitch** and **5 × 0.80 mm ESL
  resist pitch** — plus a 2 mm scale bar. Spanning 5 periods rather than 1
  because a single 0.78 mm caliper is too short to read from the back of a room.
  Burned into the render, not axis-derived, so they survive rescaling.
- Fonts sized for projection: at a half-column on a 1920-wide projection the
  figure renders ~860 px across, so 1 pt ≈ 1 px and nothing is below ~15 pt.
  Verified by downsampling the render to 860 px and reading it at that size.

Also, as a side effect: `--only peel_zoom` renders in **~18 s** instead of
~4 min, because `draw_gerber_clipped` gained an optional `window=` pre-filter.
Previously every band built all ~262 k board flashes as matplotlib artists and
relied on the clip path to hide 99.9 % of them. `window=None` is the default and
the full-board figure passes nothing, so its render path is unaffected (still
~4 min). The clip path, not the filter, still decides what is visible, so the
filter cannot change the picture — only how long it takes to draw.

---

## 3. Suggested markup change (I did NOT edit index.html)

**Recommendation: a straight swap of the right-hand image. One line.**

I had assumed side-by-side full-board + zoom, but that is wrong here: slide 7 is
**already** two-up — `chamber_exploded.png` on the left, the peel on the right,
in `<div class="cols cols-2">`. There is no room for a third panel, and no need
for one: the exploded chamber already supplies the "this is a big square
detector" context that the full-board peel was doing, and the zoom's own title
states *"25 × 25 mm of the 470 mm board"*, so the scale relationship is explicit
inside the figure. The full-board peel in that slot contributes an unreadable
texture; the zoom contributes the actual point of the slide.

The full-board version remains built and in `assets/img/` if you want it as a
backup slide.

### The edit

In the `<!-- 7: B1 chamber design -->` section, second `.figure` block, change
the `src` (and ideally the `fig-label`, which currently just restates the layer
order that the figure now labels itself):

```html
<div class="figure">
  <div class="imgwrap"><img src="assets/img/mx17_board_peel_zoom.png" alt="Close-up of a 25 by 25 mm patch of the MX17 readout board with four layers peeled back: ESL resistive strips, L4 readout pads, L5 Y strips, L6 X strips"></div>
  <div class="fig-label">A 25&nbsp;mm patch of the board, peeled back four layers &mdash; ESL resist (0.80&nbsp;mm) over pads and X/Y strips (0.78&nbsp;mm)</div>
</div>
```

### And the caption fix on the same slide

```html
<div class="caption">470&times;470&nbsp;mm PCB (399.36&nbsp;mm active) &middot; 512 strips/view on a 0.78&nbsp;mm pitch &middot; 30&nbsp;mm drift gap &middot; 150&nbsp;&micro;m amplification gap</div>
```

Do these two together. Shipping the zoom while the caption still says
0.7785 mm puts a burned-in "0.78 mm" caliper directly above a contradicting
number, on the one slide where this audience is most likely to be reading
closely.

The HTML comment block above slide 7 (the TODO describing this task) can be
deleted at the same time.

*(Status 2026-08-10: the caption fix and the image swap are both **applied** in
`index.html`. §3 is history; the live open item is §4 below.)*

---

## 4. Second pass — 2026-08-10, after Dylan's review

Two changes to `mx17_board_peel_zoom.png` only. Regenerated with the same
command as §2 and copied into `assets/img/`. **Both are gated on `zoom`, so the
full-board `--only peel` figure is byte-for-byte the same picture as before**
(re-rendered end to end to confirm).

### 4a. Pillars removed from the resist band

The bulk pillars (`3498A_bulk.gbr`, Ø0.6 mm on ~4.68 mm pitch) are no longer
drawn on band ①. At this magnification only about five of them land in the
6.25 mm band, they are not what the figure is about — the two pitches and the
strip directions are — and as pale dots on black resist stripes they read as
noise. The full-board figure still draws them, and its depth key still explains
them ("White dots: bulk pillars Ø0.6 @ 4.68 mm"), so nothing about the pillars
has been lost from the package.

### 4b. L5/L6: vias suppressed, strips drawn — and this is a SCHEMATIC

Dylan's words: as raw artwork "it isn't immediately obvious what is going on
here". He is right — Ø0.5 mm dots joined by short staggered stubs render as a
dot field, and the one thing those two bands exist to communicate (this layer
measures Y and runs along x; that one measures X and runs along y) was invisible.

So at zoom the two strip bands now draw **continuous strips, vias suppressed**:

- horizontal strips on band ③ (L5, the Y-*measuring* layer, running along x) and
  vertical strips on band ④ (L6, X-measuring, running along y) — directions per
  the gerber-verified correction in §1;
- on the gerber's **own grid**, dot centres at `0.39 + n × 0.78 mm`, re-measured
  from `DFS3498A_L3-TrackY.gbr` / `L4-TrackX.gbr` for this change;
- **0.5 mm wide = the via dot diameter**, so the strip is exactly as wide as the
  copper the artwork actually shows, and the 0.28 mm gaps keep the 0.78 mm pitch
  countable.

**This is a schematic, not copper — and the figure now says so on its face.**
The old title line "(real gerber copper)" was true when every band was literal
artwork and is not true now, so the title reads:

> MX17 readout board, four layers peeled back
> 25 × 25 mm of the 470 mm board — pads drawn from the gerber artwork;
> resist ① and X/Y strips ③④ schematic, vias suppressed

and the two band captions read "③ L5 Y strips / along x — schematic" and
"④ L6 X strips / along y — schematic". Do not put a blanket "gerber" claim back
on this figure. (Band ① was always schematic — the ESL strips have no gerber
artwork — it just was not labelled as such before; now it is. The pads visible
in ① and ② are still real `L2-pads.gbr` artwork.)

### ⚠️ What §4b does and does not assert — READ §1 BEFORE ANSWERING A QUESTION

The strips assert **direction and pitch**. They do **not** assert the
interconnect. **§1's last subsection stands unchanged and is not superseded by
this pass:** the artwork is dots plus 0.1 mm × 0.39 mm stubs present on only
~2/3 of the cells, and *how the strip interconnect is actually completed remains
an open question* — checkerboard via assignment to X vs Y is the obvious guess
and it has **not** been verified. If someone asks how the strips are connected,
the honest answer is still "that is not resolved", and the figure being drawn
with continuous lines is not evidence either way. Drawing it this way was a
legibility decision made with that gap open and labelled.

The code carries the same warning at the point of the change
(`fig_peel`, the `zoom and i >= 2` branch in
`~/CLionProjects/MX17_Geant/scripts/model/plot_mx17_model.py`), so a future
reader cannot mistake the schematic for artwork.

### Verified by looking

Rendered and inspected at full size and downsampled to 860 px — the width the
slide's half-column actually gives it. At 860 px the horizontal-vs-vertical
contrast between bands ③ and ④ is immediate, which is the whole point of the
change; the calipers, the 2 mm bar, the staggered label plates and the ~15 pt
floor all survive. Everything listed as worth keeping in §2 (25 × 25 mm in four
6.25 mm bands, both burned-in calipers, scale bar, square aspect, leader lines,
font floor) is unchanged.

### Markup: nothing is required, one optional line

`index.html` was **not** touched. The asset was overwritten in place under the
same name, so slide 7 picks it up with no edit, and the existing `fig-label` and
`alt` text are both still accurate.

If you want the slide text itself to carry the schematic caveat rather than
leaving it to the figure title, this is the exact replacement for the
`fig-label` on slide 7's second `.figure`:

```html
          <div class="fig-label">A 25&nbsp;mm patch of the board, peeled back four layers &mdash; ESL resist (0.80&nbsp;mm) over pads and X/Y strips (0.78&nbsp;mm); strip layers drawn schematically</div>
```

My recommendation is to leave the label alone: the figure's own title already
states it, and the label is doing scale-and-pitch work that the extra clause
lengthens without clarifying.
