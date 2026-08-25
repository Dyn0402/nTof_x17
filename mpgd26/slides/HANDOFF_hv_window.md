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
>    reaches below the plateau (49 % at 425 V, 81 % from 455). Both noise-era
>    placements are drawn: solid production, dashed July, 22 V apart. **540 V
>    is worth 81 % on the July placement and 69 % on the production one**; the
>    520 V frame is extrapolated and says so. The old ladder is backup D2b.
>
>    *Superseding a claim made in this file on 2026-08-24:* "the bench scan
>    never went below 450 V, so we have no measurement of a low-voltage
>    turn-on" — **wrong**, that was true only of the 22 June scan. The 27 June
>    one measures it directly, with the same efficiency definition.

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
