# Handoff — the HV build: recovery time as a cut on the X17 rate

*Written 2026-08-23. **APPLIED 2026-08-24** on Dylan's go-ahead. What follows
is what was done, kept as the record of why; `NOTES.md` has the short version.
Re-running step 1 is safe and idempotent.*

> **Two changes since it was applied, both 2026-08-24. Read these before the
> body, which still describes the as-applied state.**
>
> 1. **The build is six frames, not seven** (slide 25.1–25.6). Dylan: *"kick
>    25.7 to backup."* The figure-of-merit frame is backup only now; the
>    backup slide already existed as a duplicate and is the surviving copy.
>    `hv_window_7_trade.png` is still generated — nothing to re-render.
> 2. **The top-left panel is the cosmic bench's own efficiency now**, not
>    run_55's MIP-track ladder. Dylan went looking for the bench curve on that
>    panel and rightly did not recognise it. The curve is the **27 June
>    saturday det3 scan**, both passes, 425–525 V — the only bench scan that
>    reaches below 450 V. ~~(49 % at 425 V, 81 % from 455)~~ ⛔ **those numbers
>    are wrong — see item 3.** Both noise-era placements are drawn: solid
>    production, dashed July, 22 V apart. ~~**540 V is worth 81 % on the July
>    placement and 69 % on the production one**~~ ⛔ **93 % on production,
>    95 % on July.** The 520 V frame is extrapolated and says so. The old
>    ladder is backup D2b.
>
>    *Superseding a claim made in this file on 2026-08-24:* "the bench scan
>    never went below 450 V, so we have no measurement of a low-voltage
>    turn-on" — **wrong**, that was true only of the 22 June scan. The 27 June
>    one measures it directly, with the same efficiency definition.

> 3. **THE EFFICIENCY CURVE WAS WRONG, AND IS RE-DERIVED (2026-08-28).**
>    Everything item 2 says about the *numbers* is superseded. See
>    "The 2026-08-28 re-derivation" below — it is the first thing to read.

Dylan, 2026-08-23: *"the recovery-vs-charge plot is very nice — it shows the
flash recovery is roughly linearly related to the total charge — but make it a
backup slide, and incorporate the recovery time into the X17-stats-vs-time-of-
flight plot instead. Take the HTML text off the slide to make vertical space.
Then a very short, simplified recovery-vs-charge plot just to show it is
linear, and a series of X17-stats plots showing each HV point: 560 V first, to
show it eliminates almost all of the spectrum, then vertical lines for 550 V in
steps of 10 V down to 520 V, highlighting that we ran at 540 V."*

---

## Which slide this is

**Identify it by its title, not by its number.** When Dylan wrote the brief the
slide was 21; by the time the figures were built the deck had grown and it was
25. The target is:

> **Far too much charge — and every volt of gain costs milliseconds**
> (`.title-sm`, kicker *Status · The environment*, two figures:
> `status_charge_ladder.png` + `status_deadtime_detA.png`)

and the plot being replaced is the right one, `status_deadtime_detA.png`.

## The figures

`make_hv_window.py` (new, in `mpgd26/`). Regenerate with

```sh
../.venv/bin/python make_hv_window.py --variant b --shape wide --contact
../.venv/bin/python make_hv_window.py --numbers          # the arithmetic alone
```

It writes `figures/hv_window_{variant}_{shape}_{i}_{volts}.png|pdf` and a
`…_contact.png` with all five frames stacked, for looking at the build as a
build. **It does not copy anything into `slides/assets/img` unless you pass
`--slides`** — deliberately, so it cannot collide with the deck edit in flight.

**Variant B is the one Dylan picked** (2026-08-24). Its top row is three
things, and the layout follows from one constraint: *the strip's x axis is the
main panel's own*, so the strip has to span the full canvas — which is what
makes the lit point stand directly above the wall it produces. Every recovery
time the chamber can reach lands in the last decade and a half of a six-decade
axis, so the strip's left two thirds are empty, and that is where the rest goes.

* **top-left, an inset** — **relative track efficiency** against amplification
  voltage: n_TOF's own resist scan (run_55, chamber A, the production gas),
  tracks per trigger relative to the best the scan ever reached, with the June
  cosmic bench's efficiency plateau carried across the gas boundary and shaded
  behind it. It is opaque and sits over the strip's empty end; nothing overlaps
  the strip's data, which begins at 0.70 of the canvas width.
* **top-middle, the three numbers** — the voltage, the per cent of the X17 rate
  left, and the efficiency, large; the recovery time small under them, because
  the strip already puts it on an axis and the main panel already draws it as a
  wall. Numbers are right-aligned in one column and their descriptors
  left-aligned in the next, so **nothing re-flows between frames**.
* **top-right, the strip** — charge (nC) against the recovery time it buys,
  sharing the main panel's time axis, with a dotted plumb line from the lit
  point down to the wall. **Compressed** (2026-08-24): its charge axis is read
  on the *right*, labelled ticks from 10², label *charge [nC]*. The floor
  itself is **28 nC, not 100** — 520 / 530 / 540 V put 35 / 74 / 93 nC on the
  chamber, and a hard 10² floor drops three of the five build points off the
  panel. **And it stops where its data starts:** sharing an axis needs a time
  to map to the same figure position in both panels, not a panel that spans
  the canvas, so the strip is cut back to just before its first point and its
  box moves right by exactly that fraction. Alignment checked through the
  transforms: **0.0000 px**. The `col` shape keeps a full-width strip, having
  no efficiency panel to hand the space to.
* **main panel** — the X17 rate against neutron flight time (the same drawing
  as the *Almost all of the X17 rate is in the MeV* slide: same limits, same
  points, same 79 % callout in the same place), with the frame's recovery time
  drawn as the edge of a red blind band. Points whose decade is more than half
  behind the edge are dimmed.
*(Variant A keeps the older arrangement: a linear recovery-vs-charge strip, a
separate yield strip, and a four-line scoreboard in the top right.)*

Frames, in build order — file names are `hv_window_b_wide_{n}_{tag}.png`.

**Frame 6 is the return to 540 V** (Dylan, 2026-08-24): the same drawing as
frame 3, except that by then every other voltage's edge is on the axis behind
it, so the frame says *of all of these* rather than *this one*. It is where the
argument lands and where the discussion happens. Frame 7, the closing one, is
the two costs multiplied — **not** a frame of the build (different geometry,
because it is a different statement about the same numbers), so it can be cut
for time or promoted to its own slide.

| # | frame | blind for | X17 rate left | efficiency (bench, production) |
|---|---|---|---|---|
| 1 | `…_1_560.png` | 13.9 ms | **0.1 %** | 81 % |
| 2 | `…_2_550.png` | 7.7 ms | 5.5 % | 78 % |
| 3 | `…_3_540.png` | 5.0 ms | **9.5 %** | **69 %** |
| 4 | `…_4_530.png` | 2.4 ms | 11.7 % | 53 % |
| 5 | `…_5_520.png` | 0.9 ms | 13.0 % | ~39 % (extrapolated) |
| 6 | `…_6_540.png` | **back to 540 V**, with every other edge left on the axis | | |
| 7 | `…_7_trade.png` | — the product peaks at **550 V**, and 540 V delivers 95 % of it — | | |

Each spent edge stays on the axis as a dotted line labelled with its voltage,
so the wall visibly walks left as the build runs; 540 V's line stays in ink and
bold once it has been passed.

### Variants and shapes

* `--variant b` **— Dylan's pick, 2026-08-24.** The strip shares the main
  panel's **time** axis, so the lit point stands directly above the wall it
  makes and the link is a plumb line rather than a number said twice. What it
  gives up is the "recovery is *proportional* to charge" reading — on a shared
  log time axis that becomes a slope-1 line over a factor 20 rather than a
  straight line through the origin. The emptiness it creates is not wasted any
  more: it holds the efficiency panel and the three numbers.
* `--variant a` — the strip is recovery [ms] against charge [nC] on **linear
  axes**, and proportionality is something the eye reads in one second. Kept
  because that is a real advantage, and because it is the layout to fall back
  to if the efficiency panel is ever cut.
* `--shape wide` **(recommended)** — 1.961 : 1, sized for a `.figure-solo`
  hole on a slide with a kicker, a `.title-sm` and **no caption and no
  `.figsrc`**, which is the layout this brief asks for. Measured 2026-08-23 by
  the probe recipe in `NOTES.md` § *Measuring the hole*.
* `--shape col` — 0.930 : 1, for the right column of a `.cols-2` slide if the
  charge ladder stays beside it. It works, but the column is too narrow for a
  six-decade log axis, the scoreboard has to become one line squeezed between
  the panels, **and there is no room for the efficiency inset** — so in the
  column shape the efficiency is not on the slide at all. Use it only if the
  ladder cannot move.

## Where the yield half comes from

The second strip and the closing frame are not deck arithmetic — they are an
analysis, and it has its own report:

**`ntof_july_analysis/hv_tradeoff/`** (`hv_tradeoff.py`, `make_report.py`,
`report.html`). The deck imports its numbers rather than re-deriving them, so
the slide and the report cannot disagree. In one paragraph:

* The bench ran **Ar/iso 95/5** and n_TOF ran **90/10**, so the bench curve
  cannot be read on the n_TOF voltage axis without a gain map. The repository's
  own `garfield_sim/results/hv_equivalence.json` says the gas costs
  **+72.6 V**; site pressure (Saclay → CERN) gives **−4.7 V** back; and the
  electronics — **200 fC CSA range on the bench against 600 fC at n_TOF**, plus
  the pedestal noise the threshold rides on — costs **+12.8 V** in the run_55
  configuration and **+34.8 V** after the 23 July noise step.
* So **n_TOF 540 V is worth bench 459 V** in the run_55 configuration, the
  bench's 91–92.5 % plateau maps to **531–561 V**, and the bench's own
  discharge onset maps to **565 V** — a second ceiling, a few volts above where
  the dead time had already ended the argument.
* **The map is worth about ±20 V**, not ±2: three Garfield determinations of
  the same ratio, divided by either the measured or the simulated gain slope,
  span 63–103 V. Everything on the slide is drawn as a band for that reason.
* **Two traps, both in the report's own words.** The n_TOF ladder is *not* an
  efficiency (doubles trigger, ~50 % geometric ceiling, a 3-strip cluster
  requirement) — only its shape is used. And its 8–12 ms window sits *inside*
  the recovery above 550 V, so building the trade on it would be circular; the
  strip and the product both use 16–28 ms.

## The markup — as applied

Two steps, both done. Step 1 is idempotent; re-run it after any figure change.

### Step 1 — the assets

```sh
cd mpgd26
../.venv/bin/python make_hv_window.py --variant b --shape wide --slides
# -> slides/assets/img/hv_window_{1_560,2_550,3_540,4_530,5_520,6_540,7_trade}.png

# the three backup figures come from the analysis package, which does not
# know about the deck, so they are copied by hand
cp ../ntof_july_analysis/hv_tradeoff/figures/gas_map.png      slides/assets/img/hv_gas_map.png
cp ../ntof_july_analysis/hv_tradeoff/figures/bench_mapped.png slides/assets/img/hv_bench_mapped.png
cp ../ntof_july_analysis/hv_tradeoff/figures/ntof_ladders.png slides/assets/img/hv_ladders.png
```

### Step 2 — the main-flow slide

Replace the whole `<section>` with **seven** sections — `bstart` + six
`bcont`, the pattern the bench build uses — so the seven frames cost **one**
slide number, `N.1`–`N.7`:

```html
    <section class="slide bstart" data-frame="1">
      <div class="kicker">Status &middot; The environment</div>
      <div class="title-sm">Every volt of gain costs milliseconds &mdash; and the milliseconds are the beam</div>
      <div class="figure-solo"><img src="assets/img/hv_window_1_560.png" alt="Three panels. Top left, relative track efficiency against amplification voltage with the 560 volt point starred at 100 per cent. Middle, three large numbers: 560 volts, 0.1 per cent of the X17 rate left, 100 per cent relative efficiency. Top right, avalanche charge against the recovery time it buys, on the same time axis as the panel below. Bottom, the X17 rate against neutron flight time with a red blind band covering everything up to 13.9 milliseconds, which is nearly the whole axis."></div>
    </section>

    <section class="slide bcont" data-frame="2">
      <div class="kicker">Status &middot; The environment</div>
      <div class="title-sm">Every volt of gain costs milliseconds &mdash; and the milliseconds are the beam</div>
      <div class="figure-solo"><img src="assets/img/hv_window_2_550.png" alt="The same three panels at 550 volts: the starred point has moved down both panels, the blind band has retreated to 7.7 milliseconds, and the numbers read 5.5 per cent of the rate left and 43 per cent relative efficiency."></div>
    </section>

    <section class="slide bcont" data-frame="3">
      <div class="kicker">Status &middot; The environment</div>
      <div class="title-sm">Every volt of gain costs milliseconds &mdash; and the milliseconds are the beam</div>
      <div class="figure-solo"><img src="assets/img/hv_window_3_540.png" alt="The same three panels at 540 volts, where we ran: blind for 5.0 milliseconds, the thermal point clear of the band, 9.5 per cent of the rate left and 29 per cent relative efficiency."></div>
    </section>

    <section class="slide bcont" data-frame="4">
      <div class="kicker">Status &middot; The environment</div>
      <div class="title-sm">Every volt of gain costs milliseconds &mdash; and the milliseconds are the beam</div>
      <div class="figure-solo"><img src="assets/img/hv_window_4_530.png" alt="The same three panels at 530 volts: blind for 2.4 milliseconds, 11.7 per cent of the rate left, but only 10 per cent relative efficiency. The 540 volt edge stays on the axis behind it."></div>
    </section>

    <section class="slide bcont" data-frame="5">
      <div class="kicker">Status &middot; The environment</div>
      <div class="title-sm">Every volt of gain costs milliseconds &mdash; and the milliseconds are the beam</div>
      <div class="figure-solo"><img src="assets/img/hv_window_5_520.png" alt="The same three panels at 520 volts: blind for 0.9 milliseconds, 13.0 per cent of the rate left, 11 per cent relative efficiency. The whole MeV peak is still behind the band at every voltage."></div>
    </section>

    <section class="slide bcont" data-frame="6">
      <div class="kicker">Status &middot; The environment</div>
      <div class="title-sm">Every volt of gain costs milliseconds &mdash; and the milliseconds are the beam</div>
      <div class="figure-solo"><img src="assets/img/hv_window_6_540.png" alt="The 540 volt frame again, now with every other voltage in the build left on the axis behind it as a dotted vertical line, so the setpoint is shown against all the alternatives at once."></div>
    </section>

    <section class="slide bcont" data-frame="7">
      <div class="kicker">Status &middot; The environment</div>
      <div class="title-sm">Every volt of gain costs milliseconds &mdash; and the milliseconds are the beam</div>
      <div class="figure-solo"><img src="assets/img/hv_window_7_trade.png" alt="Three curves against amplification voltage: the X17 rate that arrives after the chamber is alive falling from left to right, the tracks the chamber reconstructs rising, and their product, which is flat between 540 and 555 volts and falls away on both sides."></div>
    </section>
```

**The seventh frame is optional and the sixth is not.** Frame 6 is the return
to 540 V and it is where the argument lands; frame 7 states the same thing
arithmetically and can be cut for time, or promoted to its own slide if you
want to dwell on it.

**The title is a proposal.** The old one led with the charge, and the charge
ladder is not on this slide any more.

**No `.caption` and no `.figsrc`** — that is what buys the vertical space, and
it is what the 1.961 : 1 figure was sized against. If you add either back, the
hole becomes 2.05 : 1 (one caption line) or 2.38 : 1 (caption + provenance) and
the figure has to be re-rendered at that `figsize`, not squeezed.

If you want one provenance line anyway, this is it — nothing is burned into
the canvas:

```html
      <div class="figsrc"><b>Efficiency:</b> run_55 resist scan, det A, MIP-track rate per trigger 16&ndash;28 ms after the flash, relative to the best the scan reached &mdash; a rate with a geometric ceiling, not an absolute efficiency. Shading: the June cosmic bench&rsquo;s own plateau, carried across the gas boundary. <b>Charge and recovery:</b> run_57, det A, one sub-run per 2 V. <b>Rate:</b> the December 2025 &sup3;He calculation on the relativistic flight time over 19.5 m. <b>A nominal per-day rate, not a yield projection.</b></div>
```

### Step 3 — the backup slides

**Five** of them (the map slide split in two when it went in — see below), in
`Backup &middot; The environment`. The first is the slide this build replaces;
the rest are the evidence under the efficiency panel, and they are what to jump
to when the question comes.

**Two things changed at insertion time, both from looking at the rendered
slide.** The gas map and the mapped bench curve were going to share a
`cols-2` slide; at 2.5 : 1 and 2.2 : 1 against a 1.05 : 1 column hole they came
out postage-stamp-sized with half the slide empty, so they are **two
figure-solo slides** instead. And all three analysis figures carried a
burned-in matplotlib title that repeated the slide's own — the deck's rule is
that a figure never does that — so `make_report.py` grew a **`--deck`** mode
that renders them title-less straight into `assets/img/`. The report keeps its
titled copies.

**B1 — the existing section, verbatim.** Both figures, the caption and the
`.figsrc`, unchanged, retitled:

> **The charge, and what it costs — the full version**

It carries `status_deadtime_detA.png`, the plot this build replaces, with its
power law, its ±2 V points and its three-decade MeV gap. The answer to *"how do
you know the recovery scales with the charge?"* is that plot, not a sentence.

**B2 — how a bench curve at 95/5 becomes an n_TOF curve at 90/10:**

```html
    <section class="slide">
      <div class="kicker">Backup &middot; The environment</div>
      <div class="title-sm">Reading the cosmic bench on the n_TOF voltage axis</div>
      <div class="cols cols-2">
        <div class="figure">
          <div class="fig-head">The gas costs 73 volts <span>&mdash; and the sim is worth &plusmn;20 of it</span></div>
          <div class="imgwrap bare"><img src="assets/img/hv_gas_map.png" alt="Left: simulated gas gain against voltage for Ar/iso 95/5 and 90/10, with an arrow marking the 73 volt shift at equal gain. Right: five determinations of that shift as a bar chart, spanning 63 to 103 volts."></div>
        </div>
        <div class="figure">
          <div class="fig-head">The bench curve, moved <span>&mdash; det3 <i>is</i> chamber A</span></div>
          <div class="imgwrap bare"><img src="assets/img/hv_bench_mapped.png" alt="Reconstruction efficiency and discharge fraction from the June bench scan plotted against the equivalent n_TOF voltage, with the operating point marked and a second copy shifted further right for the production noise configuration."></div>
        </div>
      </div>
      <div class="caption">Three terms carry a bench voltage onto the n_TOF axis: the <b>gas</b> (+72.6 V), the <b>site pressure</b> (&minus;4.7 V, thinner air at CERN), and the <b>front end</b> (+12.8 V in July, +34.8 V after the 23 July noise step, because the CSA range is 600 fC against the bench&rsquo;s 200 fC). So <b>n_TOF 540 V is worth bench 459 V</b>, the bench&rsquo;s 91&ndash;92.5 % plateau maps to <b>531&ndash;561 V</b>, and its discharge onset to <b>565 V</b>.</div>
      <div class="figsrc">Map: <code>garfield_sim/results/hv_equivalence.json</code> via <code>ntof_july_analysis/gain_map.GainMap</code>, per-mixture ln&nbsp;G fits at two site pressures, inverted to match gain. Bench: <code>mx17_det2_det3_overnight_6-22-26</code>, M3-referenced, drift 1000 V. Noise: <code>ntof_pedestal_qa/</code> and the 22 June bench pedestal. Full ledger and its uncertainties: <code>ntof_july_analysis/hv_tradeoff/report.html</code>.</div>
    </section>
```

**B3 — the two ladders, measured:**

```html
    <section class="slide">
      <div class="kicker">Backup &middot; The environment</div>
      <div class="title-sm">Both halves of the trade, on the same chamber</div>
      <div class="figure-solo"><img src="assets/img/hv_ladders.png" alt="MIP track rate per trigger against amplification voltage in two time windows, with the post-flash recovery time on a second logarithmic axis and the thermal arrival window shaded."></div>
      <div class="caption">Yield from <b>run_55</b> (18 July), recovery from <b>run_57</b> &mdash; chamber A at drift 600 V in both, two days apart, so the two halves join without a conditions jump. <b>Use the 16&ndash;28 ms window:</b> above 550 V the recovery reaches into the 8&ndash;12 ms one, so those points are suppressed by the very quantity being traded against.</div>
      <div class="figsrc"><b>The track rate is not an efficiency.</b> Its denominator is a scintillator-doubles trigger whose geometric ceiling per arm is ~50 %, and its numerator needs a 3&ndash;20 strip, &le;25 mm MIP-like cluster in both views &mdash; a cluster loses strips over threshold as the gain falls, so it turns on far later than <i>detection</i> does. Only its shape is used, never its normalisation. <code>mx_july_beam_qa/calib/25_hv_scan_summary.json</code>.</div>
    </section>
```

**B4 — optional, the trade as its own backup slide** if frame 7 is cut from the
main flow: the same `hv_window_7_trade.png` with the caption *"the product
peaks at 550 V and 540 V delivers 95 % of it; the low side is flat and the high
side is a cliff."*

## One decision this build forces

The **540 V frame does the same job as the current** *So the measurement we ran
is the thermal one* **slide** (`x17_rate_2_window.png`) — same axis, same
points, a dead band drawn on it — but does it with a *measured per-voltage*
edge instead of the 1 ms firm / 9 ms fading band, and with the surviving
fraction stated. Showing both is showing one drawing three times.

Suggestion: let the build end on 540 V's frame and **drop the thermal-window
slide**, moving its one number (10 % of the rate) into the scoreboard, which
already says 9.5 %. Alternatively keep the thermal slide as the pay-off and end
the build at 550 V. Either way, not both at full weight — that is Dylan's call,
not mine, and it is why the build stops short of touching that section.

## Numbers, and what they rest on

| quantity | value | where from |
|---|---|---|
| recovery at 520 / 530 / 540 / 550 / 560 V | 0.86 / 2.40 / 4.99 / 7.73 / 13.88 ms | run_57 flash-random probe, det A |
| charge at the same points | 35 / 74 / 93 / 160 / 277 nC per pulse | resistive-layer HV supply current |
| linear fit | 0.032 ms/nC + 0.9 ms, R² = 0.92 | over all 31 sub-runs, 520–580 V |
| X17 rate left | 13.0 / 11.7 / 9.5 / 5.5 / 0.1 % | December 2025 ³He calculation |
| tracks kept (of the best) | 11 / 10 / 29 / 43 / 100 % | run_55, det A, 16–28 ms window |
| gas 95/5 → 90/10 | +72.6 V (bracket 63–103) | `garfield_sim/results/hv_equivalence.json` |
| bench 540 V-equivalent | 459 V (run_55) / 437 V (production) | gas + pressure + electronics ledger |
| optimum of the product | 550 V; 540 V is at 95 % of it | `hv_tradeoff.figure_of_merit('b2')` |

The surviving fraction splits the decade the edge lands in **log-uniformly in
time**, which is the same reading of the table the plotted markers already
make. It is an interpolation inside one bin; at 540 V it moves the answer by
under a percentage point either way, which is why the figure rounds to 0.1 %
and the number is never quoted finer.

**One thing this figure deliberately does not do:** it does not multiply the
surviving fraction by a track yield to find an optimum voltage. run_55's resist
scan has det A's track rate in the 6–14 ms window at 1.5 / 2.0 / 3.1 / 8.1 /
12.3 % for 520 / 530 / 540 / 550 / 560 V — but *that window is itself inside
the recovery* at the top of the scan, so the two axes are entangled: at 560 V
the chamber is blind for 13.9 ms and its 6–14 ms rate is the highest of the
scan, which is proof enough. The gain trade stays a sentence the speaker says
(the deck already quotes the ~4×), not a curve.


---

# The 2026-08-28 re-derivation — read this before quoting any number here

**Dylan, 2026-08-28:** *"the top efficiencies now look very wrong. I expect
that at top efficiency I will get close to 93 %, as I see with the high stats
run that I base the rest of the slides on."*  He was right.

## The answer in five lines

* The panel's curve came from CSVs written on **29 June 2026** that carried
  **none** of July's basis changes. They plateaued at **81 %**.
* Same chamber, same night, same 490 V, its own long run: **93.3 %**
  (`long_run_resist_490V_drift_1000V/mx17_3/wft/efficiency/`). A 12.5-point
  internal contradiction inside one run directory.
* Re-derived, the plateau is **93–95 %** (455–500 V mean **93.5 %**), which is
  det3's published headline.
* **The frame numbers all change: 560/550/540/530/520 V go
  81/78/69/53/~39 % → 94/92/93/90/~90 %.**
* **And there is no turn-on.** 425 V reads **89.6 %**, not 49 %.

## What was wrong, largest term first

1. **The M3 reference recipe.** The old scan is on chi2<5 & NClus>=3 —
   **2,206** rays on the 490 V sub-run against **938** on the golden
   chi2<1.0 & NClus=4 (`qa_config`, 2026-07-13). A looser reference *points
   worse*, and this efficiency is a 5 mm match to it, so the surplus rays land
   off track and are booked as detector inefficiency. This is most of the 12.5
   points.
2. **No significance floor, and pre-reprocessing hits.** The matched-filter
   reprocessing of the raw waveforms (2026-07-24, ~+40 % hits) and the relative
   significance floor that makes it usable (`cm.apply_significance_floor`,
   rel = 0.10, DET3_RECO_FIX 2026-07-25). Without the floor, coherent noise
   inflates strip multiplicity and pushes ordinary muons over the >50-strip
   discharge veto.
3. **The active box.** The old script took its fixed box from the sub-run with
   the most reconstructed points — which is the **highest-HV** one, where >50 %
   of events are discharges. The discharge cloud blows the box out past the
   real active area (396.8 mm in y against a 362 mm active height), so rays
   fall in corners the chamber cannot see. Symptom: `has_any` reads **95 %**
   there against **99.99 %** on the long run. The re-derivation takes the box
   from the **long run** — the same box the published breakdown uses, which is
   what makes the operating-point scan point directly comparable to it.

## How it was re-derived, and how you know it is right

`mx_june_cosmic_qa/10b_hv_scan_efficiency.py` runs every sub-run through the
**`mx_june_wft/02_efficiency.py` accounting** — the one that produces the
deck's 93.5 %.

```bash
cd mx_june_cosmic_qa
../.venv/bin/python 10b_hv_scan_efficiency.py sat_det3_1   # 27 June pass 1
../.venv/bin/python 10b_hv_scan_efficiency.py sat_det3_2   # 27 June pass 2
../.venv/bin/python 10b_hv_scan_efficiency.py o22_det3     # 22 June, det A
../.venv/bin/python 10b_hv_scan_efficiency.py o22_det2     # 22 June, det B
```

Three independent checks, all passed:

| check | result |
|---|---|
| **Closure.** `10b_hv_scan_efficiency.py sat_det3_1 --closure` scores the long run through this exact code and prints it beside the published `efficiency_breakdown_hits.txt` | **exact**: 93.131 % against 93.13 %, and every one of the five categories matches (no_hit 0.014, hit_no_reco 0.295, spark 2.669, reco_far 3.891, reco_near 93.131) |
| **The two interleaved passes.** They share no voltage, so they are one curve sampled twice | agree to 1–2 points throughout: 465→94.1 against 460→94.8 and 470→94.9; 485→93.7 against 480→93.8 and 490→93.7 |
| **The independent 22 June scan**, same chamber, other slot, five days earlier | **now agrees**, 91–94 % against 93–95 % |
| **det B on the same 22 June run**, which *does* have a published breakdown (`o22_long_det2`) | its 490 V scan point reads **92.6 %** against the published **92.1 %** |

That last one retires a story this deck was telling. The two scans used to
differ by ~10 points, and the gap was explained by the top slot doubling the M3
lever arm into the same fixed 5 mm box. **The gap is gone, so the explanation
is withdrawn.** The lever arm is real and still visible where it belongs — the
core residual is 0.34–0.41 mm in the bottom slot against 0.44–0.59 mm in the
top — it simply never cost efficiency at a 5 mm match.

## The corrected curve

27 June saturday scan, both passes, bench volts (Ar/iso 95/5):

| V | 425 | 435 | 445 | 455 | 460 | 465 | 470 | 475 | 480 | 485 | 490 | 495 | 500 | 505 | 510 | 515 | 520 | 525 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| within 5 mm % | 89.6 | 93.1 | 91.8 | 93.4 | 94.8 | 94.1 | 94.9 | 91.6 | 93.8 | 93.7 | 93.7 | 92.7 | 92.4 | 88.1 | 87.0 | 79.5 | 59.7 | 47.7 |
| discharge % of firing | 1.5 | 2.0 | 2.0 | 1.9 | 3.2 | 2.9 | 3.4 | 4.4 | 5.7 | 6.1 | 7.3 | 9.6 | 11.4 | 13.9 | 18.4 | 26.4 | 44.7 | 53.7 |

Statistical errors are ~0.9 points and the plateau scatter is a little wider
than that, so **do not quote a single point to better than ~1.5 points.**

## The thing to decide before you build the slide

**There is no efficiency turn-on anywhere in this scan, and that is a real
result, not a gap in the data.** The chamber is still 89.6 % efficient at
425 V, 65 V below its operating point — where the measured gain ladder says the
gain is ~18x lower.

It is not an artefact of the new chain either. `mesh_ladder.csv` comes from
these same sub-runs, and at 425 V it puts the peak strip at **69 ADC in the
weakest 2 % of events** — about **10 sigma** over the 6.85 ADC bench pedestal,
and roughly 2x the DAQ's own 5-sigma threshold. There is simply no threshold
problem at 425 V. The old curve's rise from 49 % to 81 % across 425–455 V was
the **pre-reprocessing analyzer's amplitude threshold turning on**, not the
chamber. This is the same mechanism that moved det4/det6/det7 by 14–21 points
in the July reprocessing (`HANDOFF_efficiency.md` §3.1), and det3 at 425 V is,
in signal terms, a low-gain chamber.

**What this does to the build's argument.** It does not break it — it sharpens
it. The title *"Every volt of gain costs milliseconds"* becomes literally true:
the volts cost dead time and buy **essentially nothing** in efficiency across
the whole 520–560 V window. The trade-off is now one-sided in favour of running
lower, and the reason we could not go lower was never efficiency.

**Three consequences worth a decision, all yours:**

1. **The top-left panel is now a flat line.** Honest, and much less
   interesting to look at than a turn-on. Options: keep it (it *is* the
   answer, and the star still moves along the charge panel), re-scale its y
   axis to 80–100 % so the ~4-point sag and the high-V collapse are visible,
   or drop the panel from the main flow and keep D2c as the place the bench
   efficiency lives. **My suggestion: re-scale to 80–100 %** — it keeps the
   panel informative without pretending to a turn-on.
2. **The extrapolation below 425 V is now nearly horizontal** (0.0011 per V
   through the three lowest points, against 0.0141 before). It is a *weak*
   statement — the fit's slope times the extrapolated span is smaller than the
   points' own scatter — and it says only "no turn-on yet by 425 V". It must
   not be read as a measured plateau below 425 V; the chamber has to turn off
   eventually and this scan never reaches low enough to see it. Still drawn
   dashed and labelled.
3. **The bench curve and the run_55 MIP-track ladder now disagree much more
   loudly.** Over 560→540 V the ladder falls 100→29 % where the bench now says
   detection went 94→93 %. The deck already says why (the ladder is a
   reconstructability proxy on a doubles trigger with a 3-strip cluster
   requirement, and only its *shape* is ever used) and that caveat is now
   carrying much more weight. If anyone puts the two on one axis, the
   distinction has to be said out loud.

## The gain curve (added 2026-08-28, afternoon)

Flat efficiency is the *absence* of a measurement — it cannot say whether the
chamber is coasting on a large gain margin or sitting one volt above threshold.
`mx_june_cosmic_qa/10c_hv_scan_gain.py` puts the missing axis under it, from
**the same sub-runs**: the threshold-free peak-strip waveform maximum on
M3-selected muons (`mx17_sim_wft/hv_slope/peaks.parquet`, golden recipe, no
amplitude cut anywhere — a 5σ cut would truncate the low tail and fake a
shallower rise).

* **d ln A / dV = 0.419 ± 0.004 per 10 V** → gain ×2 every **16.5 V**,
  **×25 between 425 and 500 V**. Ten fits (4 quantiles × 2 views + the
  neighbour sum) agree to ±2.5 %, so the spectrum is being *rescaled*, not
  reshaped.
* **Relative only** — no ADC→electron calibration exists for the June bench CSA
  range. Restricted to the 460–490 V head window this is the HV-slope test's
  data/sim = 1.52, unchanged.
* Above ~505 V the peak strip is on the 12-bit rail (3872 ADC); those points are
  drawn open and never fitted.

**This is what makes the flat plateau a statement rather than an absence.** At
425 V det3 is **14× down in gain** and still 89.6 % efficient; the weakest 2 %
of muons there peak at 69 ADC = 9.3 σ = **1.9× the DAQ's own 5σ threshold**
(37 ADC). And gain is *still climbing exponentially* through the voltages where
efficiency collapses — 500→525 V takes the discharge fraction 11 → 54 %. The
high-V edge of the window is a discharge limit, full stop.

Two figures, written untitled into `assets/img/` by `10c ... --slides`:
`hv_gain_ladder.png` (four parallel lines, the rail and the 5σ line) and
`hv_gain_and_efficiency.png` (gain over efficiency over discharge fraction, one
voltage axis). **Neither is wired into a slide** — the second is the obvious
companion to open decision 1, since it is what would make a re-scaled flat
panel worth looking at, but that is Dylan's call.

Two of this package's own comments were wrong and are corrected in `10c`: the
±1 neighbours carry **~1.1× the peak amplitude together** (not ~30 %, so the
neighbour sum buys 15 V of headroom, not 3×), and the bench pedestal is
**7.41 ADC**, not 6.85.

## How tight a match? (added 2026-08-28, after the gain curve)

`10b ... --radii` scores the *same cached events, same box, same denominator* at
0.5 / 1 / 2 / 3 / 5 mm. Plateau (455–500 V): **26.0 / 61.7 / 86.5 / 91.4 /
93.5 %**.

**A 1 mm curve is not an efficiency curve.** The median matched residual is
0.75–0.80 mm across the plateau, so 1 mm cuts inside the residual distribution
and the 61.7 % it returns is mostly a containment fraction — quoting it as
detection efficiency would understate det3 by 32 points. Scaling the 5 mm curve
by the Rayleigh containment implied by each sub-run's own median residual
reproduces the 1 mm points to 1–4 points everywhere, i.e. **the tight cut
carries no extra information about detection**. The published 5 mm cut sits at
~6× the residual scale, which is what makes it a detection measurement.

**What the tight cut does add, and it is deck-relevant:** resolution keeps
improving after detection has plateaued. The median residual falls
**0.97 → 0.79 mm between 425 and 460 V** and then stops, and the 1 mm / 0.5 mm
curves rise over exactly that span while the 5 mm curve is already flat. Against
the gain ladder: **×3.7 in gain bought 0.18 mm; the next ×7 bought none.** So
the *low* edge of the usable window is set by resolution, not detection — a
better reason to stay above ~460 V than the deck's old (artefactual) turn-on.

Caveats: this is the hits-chain centroid (for a resolution *number* use `wft`,
`RECONSTRUCTION_BASIS.md`); the residual includes the M3 pointing error, 0.224 mm
at z = 702; and the robust core width 0.40–0.59 mm is consistent with the
published 0.47 mm core — the median radial residual is larger because it carries
the non-Gaussian tail, and it is the median that sets 1 mm containment.

Figure: `hv_eff_radii.png` (assets/img), `efficiency_vs_hv_radii.{csv,png}` in
the Analysis dir. **Not wired into a slide.**

## Files touched

| File | Change |
|---|---|
| `mx_june_cosmic_qa/10b_hv_scan_efficiency.py` | **new** — the scan on the 02_efficiency accounting, with `--closure`; `--compare` draws the archived 29 June curve against it |
| `mx_june_cosmic_qa/10c_hv_scan_gain.py` | **new** — the gain curve on the same sub-runs; `--check` refuses to plot a reduction that no longer matches the disk |
| `mx_june_cosmic_qa/10d_hv_scan_wft_reco.py` | **new** — the frozen r06 forward fit over the 18 mesh sub-runs; retargets `OUT_BASE` per sub-run (the config freezes it at construction, so without that every sub-run overwrites the golden long-run table) |
| `mx_june_cosmic_qa/10f_hv_scan_occupancy.py` | **new** — one threshold-free read pass: `q_win`, 5σ cells, strips, time span |
| `mx_june_cosmic_qa/10e_hv_scan_charge_angle.py` | **new** — the reduction, four figures and a generated `report.html` |
| `mx_june_cosmic_qa/10g_hv_scan_strip_matrix.py` | **new** — per-strip significance profile (±10 strips, in units of each strip's own noise) around the peak strip, sub-threshold values kept |
| `mx_june_cosmic_qa/10h_hv_scan_multiplicity.py` | **new** — strips over threshold with the M3 angle divided out: the *a*/*b* decomposition, holes, the threshold-scaling prediction, the over-threshold charge fraction |
| `.../hv_scan/mx17_3/strip_matrix.parquet` | **new** — 23,774 event-planes × 21 strips |
| `.../hv_scan/mx17_3/strip_multiplicity_vs_hv.csv`, `strip_profile_vs_hv.csv` | **new** — the multiplicity ladder and the folded per-offset profile |
| `.../hv_scan/mx17_3/{charge_angle_vs_hv.csv,.meta.json,local_gain_slope.csv,occupancy_raw.parquet}` + `charge_vs_hv.png`, `occupancy_vs_hv.png`, `angres_vs_hv.png`, `local_gain_slope.png`, `charge_angle_summary.png`, `report.html` | **new** |
| `.../<subrun>/mx17_3/wft/events_hvscan.parquet` (×18) | **new** — the per-sub-run forward-fit tables |
| `mx_june_cosmic_qa/10b ... --radii` | **new** — the 0.5–5 mm match-radius family; any `--r` other than 5 mm now writes `efficiency_vs_hv_r<R>mm.csv` **beside** the published file, never over it |
| `.../hv_scan/mx17_3/efficiency_vs_hv_radii.{csv,png}`, `hv_scan{,2}/.../efficiency_vs_hv_r1mm.*` | **new** |
| `.../hv_scan/mx17_3/gain_vs_hv.{csv,meta.json,png}`, `gain_and_efficiency.png`, `efficiency_before_after.png` | **new** |
| `.../mx17_det3_saturday_scan_6-27-26/hv_scan{,2}/mx17_3/efficiency_vs_hv.csv` | **new**, replacing `efficiency_vs_hv_scan{,2}.csv` |
| `.../mx17_det2_det3_overnight_6-22-26/hv_scan/mx17_{2,3}/efficiency_vs_hv.csv` | regenerated on the same chain |
| `.../hv_scan{,2}/mx17_3/_superseded_20260629/` | the 29 June originals, parked with a README saying why |
| `ntof_july_analysis/hv_tradeoff/hv_tradeoff.py` | new paths, provenance, and the withdrawn lever-arm explanation |
| `ntof_july_analysis/hv_tradeoff/make_report.py` | `fig_bench_mapped` description and labels; **the verdict paragraph rewritten** — it used to say the noise step pushed 540 V "onto the shoulder of the turn-on, not the plateau", 69 % against 81 %, and called that ~12-point gap the cost of the decision. Both eras now put 540 V on the plateau (92.8 % production, 94.6 % July) and the 540→560 V difference is ~1 point either way, inside the scan's own scatter. **The 540 V decision cost no measurable detection efficiency**; the figure-of-merit product is unaffected (it uses the n_TOF MIP ladder, not this curve) |
| `mpgd26/make_flash_slides.py` output `status_eff_recovery.png` | regenerated for both chambers; det A's plateau 90 → 88–94 %, its 520 V spark fraction 49 → 36 % |
| `mpgd26/make_hv_window.py` | `_eff_panel` docstring; label placement fixed (the 520 V star's label collided with the "extrap." tag once it moved to 90 %) |
| `mpgd26/slides/index.html` | five frame alt texts, the D2c alt text, the speaker-note block, the D2c figsrc |
| figures | `hv_bench_mapped`, `hv_gas_map`, `hv_ladders`, `hv_window_{1..7}`, `status_eff_recovery`, and `hv_tradeoff/report.html` |

The old CSVs are **kept** under `_superseded_20260629/` rather than deleted, so
the previous curve can still be reproduced if a figure elsewhere turns out to
have been built on it.

## Past the peak strip: total charge, occupancy, angle (added 2026-08-28, evening)

10c measured the tallest *sample* of the tallest *strip*. That estimator dies at
500 V, and it says nothing about how much of the track the chamber records or
what the extra gain does to the angle. Three new pieces answer that, all on the
same M3-golden fiducial population as 10b/10c, all on the waveform basis:

* `10d_hv_scan_wft_reco.py` — the frozen r06 forward fit over all 18 sub-runs
  (11,074 events). It **censors saturated samples**, so it keeps measuring
  where the peak sample cannot.
* `10f_hv_scan_occupancy.py` — one threshold-free read pass: the plain sum of
  every sample over ±10 strips (`q_win`, no threshold, no model), the same
  restricted to 5σ cells, the strips and the time samples that carry them.
* `10e_hv_scan_charge_angle.py` — the reduction, figures and `report.html`.

**The peak-sample gain curve was right.** Over 425–505 V the deconvolved charge
gives **0.4185 ± 0.0023 per 10 V** and the model-free window sum
**0.4280 ± 0.0024**, against 10c's peak sample **0.419 ± 0.004**. Three
estimators with nothing in common but the events. What saturation cost 10c was
*reach*, not accuracy — and the ratio (total charge)/(peak sample) is flat to
±5 % up to 500 V and then triples, which is the clipping made visible.

**But "×2 every 16.5 V" is an average, not a constant.** The local slope runs
~0.37 per 10 V over 425–450 V and ~0.50 near 490–500 V (±0.05 point to point;
the two interleaved passes agree to 3–5 % where their ranges overlap, so this is
curvature, not a pass offset). Ordinary α(E) behaviour — but the doubling
voltage must not be carried outside the range it was fitted in.
`local_gain_slope.{csv,png}`.

**Occupancy: the depth never changes, the threshold crossing does.** The fit
recovers a charge column of 26.0–28.5 mm (x) and 30.2–34.1 mm (y) against the
30 mm gap, **flat across a factor 29 in charge** — at 425 V, 14× down in gain,
the whole drift column is still there. What grows is what crosses threshold:
5σ strips **5 → 12** per plane, time over 5σ **912 → 1490 ns** (twice the 820 ns
gap transit, so that span is the resistive/shaping tail emerging from the noise,
not more track). The cluster core at 10 % of the peak stays **5–6 strips**
throughout — *rescaled, not reshaped*, transversely as well as in amplitude.

**Angular resolution does deteriorate, and at the top.** s68 against M3 over
every fitted plane: best **1.02° at 455 V (x)** and **1.06° at 460 V (y)**,
against 1.15/1.21° at 425 V and **1.38/1.31° at 515 V**. The 3-D opening angle
is the cleaner single number: **1.58° at 460 V → 2.06° at 515 V, +30 %.**
So the angular optimum sits at the *bottom* of the efficiency plateau, and the
top of the plateau costs 15–30 % in angle before any discharge is counted.

**A trap this analysis walked into and out of, worth keeping:** the fit-quality
gate (χ²/dof < 300) fails on saturated events, and saturated events are the
high-gain ones. Gated, the resolution appears to *improve* to 0.93° at 525 V
while the surviving fraction falls from 100 % to 19 %. Every angle number here
is therefore quoted over **every fitted plane**, with the gated curve drawn
beside it precisely so nobody quotes it. Same reason the charge ladder is
measured on the ungated (spark-free) set.

Not measured above ~505 V: the window sum clips too (tens of railed cells per
track), the 12-strip clusters start to reach the edge of 10f's ±10-strip
window, and the spark veto removes 46 % of fiducial rays at 525 V. Those points
sit in the grey band and are in no fit.

Figures (`--slides` writes untitled copies to `assets/img/`):
`hv_total_charge.png`, `hv_occupancy.png`, `hv_angres.png`,
`hv_local_slope.png`, `hv_charge_angle_summary.png`. **None wired into a
slide.** The summary panel is the one that would go with open decision 1 if the
flat efficiency panel is ever re-scaled.


## Strips over threshold: is low gain losing the small deposits? (added 2026-08-28, night)

The question behind it: *the strip count rises 5 → 12 with voltage — is that the
threshold eating into a fixed cluster, or is charge genuinely going missing at
low gain?* They have opposite consequences, and the strip count on its own
cannot separate them, because it is threshold-limited by construction. `10g`
therefore records the amplitude of **every** strip in a ±10-strip window in
units of that strip's own noise, sub-threshold values included; `10h` reduces
it. Four handles, deliberately independent:

**1. The over-threshold charge fraction — the direct answer.** `q_5s / q_win`
from 10f, referred to its own 470–490 V plateau: **85.9 % (x) and 78.9 % (y) at
425 V**, closing to within 1 % by **470 V (x) / 475 V (y)**. So at the bottom of
the ladder **14 % (x) / 21 % (y) of the charge the detector collected never
crosses 5 σ**. That is the proposed mechanism, measured.

*(The raw ratio plateaus a few per cent above 1 — the denominator is a whole-
window sum and carries the shaped pulse's undershoot and whatever the
64-channel common-mode median removed from a wide signal, both
signal-proportional and negative. Harmless for the charge **slope**, which is
why q_win and the deconvolved q_sum agreed to 2 %, but it means the ratio is
only a detected fraction after normalising to its own plateau.)*

**2. Angle normalisation.** `w_geo = gap·|tan θ_ref|/pitch` from the **M3
telescope**, so the normaliser cannot move with the detector's own gain (median
4.2 strips in x, 3.5 in y; p90 ≈ 9 and 8). Fit `n_lit = a(V) + b(V)·w_geo`:

| | 425 V | best | 525 V |
|---|---|---|---|
| *a* (footprint at normal incidence), x | 2.93 | — | 10.74 |
| *a*, y | 3.68 | — | 12.11 |
| *b* (lit per crossed strip), x | 0.49 | 0.72 @ 460 V | 0.40 |
| *b*, y | 0.40 | 0.52 @ 445 V | 0.40 |

The rise is **almost all in *a*** (+7.8 x, +8.4 y), and it survives angle
matching — every `w_geo` band rises by about the same *number* of strips rather
than in proportion to its width. **So the count grows because the transverse
tail crosses threshold, not because more track is recovered.** *b* never reaches
1 at any voltage: a wider track spreads the same charge over more strips, so its
marginal strips are the faintest. Above ~505 V both are distorted by the
±10-strip window (up to 34 % of events touch its edge).

**3. Holes — dark strips strictly inside the lit span.** Needs no angle
normalisation at all. x: **3.2 % of the span at 425 V → 0.3 % at 465–470 V**,
and **13.9 % of clusters carry at least one hole at 425 V against 2.5 %**. They
are concentrated in the widest footprints — **12.5 % for w_geo > 9 strips** —
exactly what dilution predicts. The holes at the bottom sit at **4.0 σ**: strips
that *just* missed, not empty ones.

**4. The scaling prediction.** Take the 465 V events, scale each strip's
**signal** by the measured charge ratio (the ~2 σ noise-max floor held fixed —
a max over 32 samples is not zero, and scaling it too invents hits), re-apply
5 σ, recombine over the same `w_geo` bands as the measurement. It reproduces the
measured multiplicity **to within 0.36 strips over 435–490 V**. Above 495 V it
overshoots by up to 1.7 strips — that is the *measurement* being limited (rail,
window edge), not the model.

**Verdict.** Both are true, and the split matters: **the charge deficit is real
(14–21 % at 425 V) but it is at the edges of a cluster whose shape is fixed.**
The profile in `strip_profile_vs_hv.png` is parallel in log across the whole
ladder until it reaches the noise floor — rescaled, not reshaped — and the strip
count is simply where the 5 σ line cuts it. Per-offset turn-on: |k| = 2 crosses
50 % near 445–460 V, |k| = 3 near 470–485 V, |k| = 4 near 510–520 V.

**This is the low-voltage half of the angular optimum.** At 425 V a seventh of
the x clusters are broken and a sixth of the charge is invisible; that is why
the angle is worse at 425 V than at 455–460 V, where the deficit has closed and
saturation has not yet started. The optimum is where the two curves cross.

Cross-check: `n_lit` from 10g equals 10f's `n_strip_5s` on **22,148 of 22,148**
event-planes, max difference 0 — two separate read passes, same window, same
threshold.

Caveat that no amount of reduction removes: this is all **at 5 σ on the June
bench noise**. A different threshold moves *a*(V) bodily and moves the voltage
at which the deficit closes. Only the shape result — profile rescales, does not
reshape — is threshold-free.

Figures: `multiplicity_holes.png`, `strip_profile_vs_hv.png`,
`multiplicity_vs_hv.png`, `multiplicity_prediction.png`; slide copies
`hv_multiplicity_holes.png`, `hv_strip_profile.png`, `hv_multiplicity.png`,
`hv_multiplicity_prediction.png`. **None wired into a slide.** The section is
folded into the same `report.html` as the charge/occupancy/angle work.

## The top-left panel is the GAIN now, not the efficiency (2026-08-28, late)

Dylan, after the strips-over-threshold study: *"rework the figure … instead we
should show here a scaled gain curve, also reporting this in the center
numbers. We should show the full charge collected, and aim for the peak strip
in the median event to be saturated (just barely saturating is probably ideal
gain). We'll call this 100 % optimal gain (can show over 100 % and indicate
probably too much if we end up there on the scale). Then we can show
percentages of this optimal gain instead of efficiency."*

**Why the swap is right.** The efficiency was re-derived that morning and came
out flat at 93–95 % with no turn-on. That correction stands and it matters —
but a panel that does not move cannot be the left half of a trade whose right
half walks 13.9 → 0.9 ms. The gain does move, it is what the milliseconds are
being paid for, and the same 27 June scan measures it best.

### What 100 % is

The mesh voltage at which the peak strip of the **median** track just fills the
12-bit sample. `peak_amp` is the tallest **sample** of the tallest **strip** of
the event — the max strip, which is the thing that clips first, not a per-strip
average. Measured, not modelled: `frac_sat` goes **0.39 at 495 V to 0.66 at
500 V in both views**, so the 0.5 crossing is bracketed by two points 5 V apart,
at **bench 497.0 V** (x 497.0, y 497.0 — they agree to 0.04 V).

**Checked again 2026-08-29**, when Dylan said he remembered ~500 V off the gain
plot. The clipping ladder, spark-free, at the 3550 ADC level:

| bench V | 475 | 480 | 485 | 490 | 495 | **497** | 500 | 505 | 510 |
|---|---|---|---|---|---|---|---|---|---|
| max strip railed | 10 % | 12 % | 17 % | 27 % | 39 % | **50 %** | 66 % | 80 % | 94 % |

5 % of tracks by 469 V, a quarter by 489 V, half by 497 V, 90 % by 509 V.

**Both readings are right, and they are different statements.** *Half the events
have the max strip clipped* at **497 V**. The *median amplitude* only reaches the
nominal 3871.5 ADC rail near **500 V**, because at the 50 % point half the sample
is still below it — and 500 V is exactly where the p50 marker visibly sits on the
rail line in `gain_vs_hv.png`, which is what the eye reads. Anchoring at 500 V
instead would lower every percentage on the scale by **~13 %**.

The 50 % point is insensitive to where the clipping line is drawn: **496.4 /
497.0 / 497.1 V** at 0.88 / 0.92 / 0.95 of the rail. A cut at 0.98 gives 508.6 V,
but that stops being a clipping test — per-channel pedestal subtraction spreads
the railed population over ~3700–3900 ADC, so it asks whether a channel's rail
landed high. The spark veto is not what sets it either: 496.8 V on the full
fiducial set against 497.0 spark-free.

What is **plotted** is the *total collected charge*, not the peak sample: the
deconvolved forward fit censors railed samples and so keeps measuring where the
peak sample cannot. Charge and peak amplitude are proportional to ±5 % across
this range, so normalising a charge ladder to a saturation voltage is
self-consistent. The model-free window sum gives the same percentages to 5 %.

### The map — and this is where the first version was wrong

Dylan, reading the finished figure: *"in my head 560 corresponded to 490, did I
have this wrong?"* **He did not.** Only the **gas and the site pressure** move
the *voltage*, and that is **−67.85 V**: n_TOF 560 V is read off the bench
ladder at **bench 492.1 V**. Checked straight out of the blessed map — the gas
term is +72.70/72.62/72.44/72.34/72.20/71.84 V at 95/5 440/460/480/490/500/520,
flat to ±0.4 V, and the pressure term −4.84 to −4.34 over the same span.

The readout change (CSA 200 → 600 fC) then **divides the ADC by three**. It is a
factor, not a voltage:

> pct(W) = Q_bench(W − 67.85) / (3 × Q_bench(497))

and no slope enters it anywhere.

**What the first version did, and why it was wrong.** It folded the factor three
into the voltage axis as ln 3 / slope = +26.3 V, giving a single shift of
**+94.1 V**, and evaluated the ladder there. That is exact only for a perfectly
exponential ladder. This one is **curved** — local slope 0.33 per 10 V near
435–445 V, 0.52 near 485–505 — so the slide read the wrong part of it:
**−13 % at n_TOF 520 V, +6 % at 560 V.** The displayed integers moved by at
most one point, but the span across the build did not: 4.7× before, **5.7×**
now.

`adc_shift()` is kept, because saying *"n_TOF 560 V makes the ADC that bench
466 V makes"* is a useful sentence. It is no longer used to evaluate anything.

**Three bench-equivalents of one n_TOF setpoint, all correct, all different:**

| question | shift | 560 V is bench |
|---|---|---|
| same **gas gain** — what the curve is read at | +67.9 | **492.1** |
| same **ADC counts** (+ the 600 fC CSA) | +94.1 | 465.9 |
| same **signal-to-noise** (+ the 23 July noise step) — the *efficiency* map | +102.7 | 457.3 |

The 8.6 V between the last two is exactly `ln(9.80/6.85)`. A rail sits at a
fixed ADC count however noisy the channel is, so the noise term has no business
in a saturation statement.

### Full scale of *what*? It is worth a factor 3 — and Dylan chose

Bench 497 V fills the **200 fC** DREAM the scan was taken with. n_TOF ran
**600 fC** — 3× less ADC per electron — so filling *that* needs three times the
avalanche, at bench ~518 V, **n_TOF 586 V**.

Dylan, 2026-08-29, looking at the report figure: *"on the plot of collected
charge vs voltage the 100 % level is at something like 518 V instead of 497 V?
Why is this? Can we put 100 % at 497 V instead? Then also make it linear
y-axis"*. Done, and it is the right lead for three reasons:

* it is the scan's **own measured** saturation point;
* it leaves **nothing on the panel extrapolated** — bench 425–505 V covers
  n_TOF 493–573 V and the crossing at 565 V is inside it, where 586 V is 13 V
  past the last trustworthy bench point;
* the 600 fC setting was **forced by the gamma flash** (668 pC on one strip,
  1113× the DREAM range), not chosen for tracking, so referring a *tracking*
  gain to it asks for 3× more avalanche than a MIP measurement needs.

Both scales are in `results()['gain_scale']` (`pct` and `ntof600.pct`) and both
columns are in the report table. `bench_gain_on_ntof_axis(ref='ntof600')` draws
the other one. **Say which one a number came from; never mix them.**

### The answer

| n_TOF V | 520 | 530 | **540** | 550 | 560 |
|---|---|---|---|---|---|
| % of optimal gain (forward fit) | 13.6 | 20.5 | **31.6** | 47.7 | 78.0 |
| % (window sum) | 12.9 | 19.7 | 30.8 | 46.7 | 76.4 |
| % if referred to the 600 fC range | 4.5 | 6.8 | 10.5 | 15.9 | 26.0 |
| bench V it is read at | 452 | 462 | 472 | 482 | 492 |

**100 % is at n_TOF 565 ± 20 V** — just past the top of the build. We ran at
about a **third** of the gain that fills the readout, and 560 V is at 78 %.

**Nothing on the panel is extrapolated**, which is a consequence of where 100 %
was put. The ±20 V is the gas map's own bracket: it slides the whole curve and
takes 565 with it, without touching the ratios between setpoints — and the
ratios are all the percentages compare.

**Linear y axis** (same request). It works now and would not have before: with
100 % at 565 V the drawn range is 14–100 % over the build, a factor 7, where the
600 fC scale ran 4.5–26 and needed a log axis to be anything but a hockey stick.
Linear also puts the run-away back on the canvas — the curve is flat-ish to
550 V and then goes vertical, which is the slide's whole argument.

### The caveat to have ready

**"100 % of optimal gain" is a readout limit, not a physics optimum.** It says
the ADC would be full, nothing more. The same scan's angular resolution against
M3, over every fitted plane, is best at **bench 445–460 V (1.02–1.06°)**, is
already 1.11–1.15° at bench 497 V where the median track saturates, and
1.31–1.38° by 515 V. A resolution is a threshold quantity, so the ledger that
carries it across is the **signal-to-noise** one (+102.7 V), putting that
optimum at **n_TOF 548–563 V** — at or a little above where we ran, and well
inside the map's own ±20 V. So the scale says *how much of the readout's range
we were using*, not *how far from best we were*.

Two more, in `hv_tradeoff/report.html`'s caveat list: the percentages are gain
**ratios** (no ADC→electron calibration exists for the June bench range), and
the flash charge climbs faster with voltage than the cosmic gain does (0.51
against 0.42 per 10 V over 520–560 V) — inside the known scatter between a
MIP's avalanche and a gamma flash read off a supply current, but not zero.

### Files touched

| File | Change |
|---|---|
| `ntof_july_analysis/hv_tradeoff/hv_tradeoff.py` | **new** `bench_charge_ladder`, `saturating_voltage`, `adc_shift`, `bench_gain_on_ntof_axis`; `results()['gain_scale']`; the ladder printed by `main()`. `bench_gain_on_ntof_axis` **rewritten the same evening** — it returns a dict now, evaluates the ladder at the gas-equivalent voltage and divides by 3 rather than sliding by one shift, and hands back the extrapolated tail separately so the caller can dash it |
| `ntof_july_analysis/hv_tradeoff/make_report.py` | **new** `fig_gain_scale` and report section 4 (old 4 and 5 renumbered to 5 and 6), three new caveat bullets |
| `ntof_july_analysis/hv_tradeoff/{report.html,results.json,figures/gain_scale.png}` | regenerated |
| `mpgd26/make_hv_window.py` | **new** `_gain_panel` and `gain_at`; `_eff_panel`/`eff_at` kept as backup behind `--panel eff`; scoreboard row 3 in all three readouts; module docstring |
| `mpgd26/slides/index.html` | five frame alt texts and the speaker-note block for slides 50–55 |
| `mpgd26/figures/hv_window_*`, `mpgd26/slides/assets/img/hv_window_{1..7}.png` | regenerated |

**The `.pptx` is not updated by any of this** — the PNGs have to be swapped in
by hand on the laptop, as always.
