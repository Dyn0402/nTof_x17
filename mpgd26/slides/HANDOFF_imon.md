# HANDOFF — the imon slide, and the two slides it changes

**Written 2026-08-10.** Everything here is markup to paste; `index.html` was not
touched (you own it). Analysis, all numbers and every caveat:
`ntof_july_analysis/flash_charge/HANDOFF_FLASH_CHARGE_2026-08-09.md` **§8**.

---

## 0. The verdict, so you can decide before reading the markup

**The deck can say "142 nC" as a measurement. No inequality, no correction
factor.**

The charge numbers came from `Q = [mean(imon) − median(imon)] / f_pulse` on the
resistive-layer HV supply's current readback. That is only the charge if the
readback reports the **time-average** of a current burst much shorter than the
1 s sample spacing. If it instead reported something closer to an instantaneous
sample it would mostly miss the ~10 ms bursts, and every charge would have been a
lower bound by an unknown factor.

I measured the readback's impulse response directly — phase-folded imon against
the individual logged beam pulses — and it is a **~1 s averager**: one pulse
produces a response that rises from 0.3 s, peaks at 88 nA about 1.1 s later, and
is back at zero by 2.3 s, with an **area equal to the charge**. A hard,
timestamp-free bound says the averaging window is **≥ 0.47 s** against a ~10 ms
burst. So the estimator is right, and the four independent ways of extracting the
number agree to **±3 %**.

Two things to know about the honesty of that statement:

* It was measured on **run_79** (2026-07-26), not run_158 — the August tree was
  not reachable from this laptop. run_79 is the *same production setpoint* and its
  det C charge (97–98 nC) matches run_158's det C (97–101 nC), so it is the same
  measurement. Worth re-running on run_158/run_157 when the mirror is back.
* The **trap was real and had to be beaten**: the HV logger writes whole-second
  timestamps while its loop actually runs at 1.0162 s, so folding on the raw
  labels smears the response by a full 1 s box — which would have manufactured
  exactly this reassuring answer. Two independent defences (§2 below) carry the
  verdict; the strongest one uses no timestamps at all.

---

## 1. NEW SLIDE — the explainer. Deliberately over-detailed; cut later.

Put it **immediately after** the existing backup slide "How the charge per pulse
is measured" (it is the natural continuation), or promote it into the main Status
flow next to "How much charge are we talking about?" if the systematic comes up
in questions.

Figure: `assets/img/status_imon_response.png` — already written, regenerate with
`.venv/bin/python ntof_july_analysis/flash_charge/make_imon_figure.py`.

```html
    <!-- Backup — the imon systematic, measured and closed 2026-08-10.
         Analysis: ntof_july_analysis/flash_charge/imon_response.py; every number
         in HANDOFF_FLASH_CHARGE_2026-08-09.md §8.  This slide is deliberately
         long — the teaching version.  If time is short, keep the callout, the
         figure and the last two table rows. -->
    <section class="slide">
      <div class="kicker">Backup &middot; How the charge number survives scrutiny</div>
      <div class="title-sm">Does the current readback actually see a millisecond burst?</div>
      <div class="callout" style="margin-top:0;"><b>Yes — it averages over ≥ 0.47 s, so it cannot miss one.</b> The charge numbers are measurements, not lower bounds, and there is no correction factor. Four independent estimators agree to <b>±3 %</b>.</div>
      <div class="figure-solo"><img src="assets/img/status_imon_response.png" alt="The HV monitor's measured response to one beam pulse: a 1 s wide hump peaking at 88 nA, on three chambers, with a flat drift-cathode null"></div>
      <div class="caption"><b>What the left panel is.</b> Every imon sample is tagged with how long it came after the nearest beam pulse, and then averaged in bins of that delay — a phase fold. What comes out is the monitor's response to one known impulse of charge. <b>The area under it is the charge per pulse: 98 nC.</b> The dashed step is the same fold on the logger's raw whole-second timestamps — visibly wider, and rising <i>before</i> the pulse, which is impossible: that is the timestamp smear, and beating it is most of the work. <b>Right panel:</b> the same shape on three chambers once each is divided by its own area — the shape belongs to the <i>monitor</i>, not to a detector — plus a drift-cathode channel on the same crate and the same logger, which stays flat to the last digit.</div>
    </section>
```

### 1b. Companion slide — the reasoning, if you want it on a slide rather than in your head

This is the part that genuinely does not fit on the slide above. Keep it as a
second backup, or keep it only as speaker notes.

```html
    <!-- Backup — why the imon measurement is believable.  The adversarial
         version: what the failure mode would have looked like, and the two
         defences against the timestamp trap. -->
    <section class="slide">
      <div class="kicker">Backup &middot; How the charge number survives scrutiny</div>
      <div class="title-sm">Why that is a measurement and not a comforting story</div>
      <div class="content">
        <ul class="bullets">
          <li><b>What imon physically is.</b> The resistive layer is what the avalanche ions land on. Every electron the gas multiplies has to be resupplied through that one HV channel, so the channel's <b>average current is the avalanche charge per second</b> — divided by the pulse rate, the charge per pulse. It sits entirely outside the readout, which is the whole point: the readout is the thing that saturates.</li>
          <li><b>Why mean &minus; median.</b> The readback samples at ~1 Hz and beam pulses come every ~3.3 s, so <i>most</i> samples sit at the standing leakage current. The <b>median is therefore the leakage at that exact voltage</b> — and leakage is strongly HV-dependent, which is why a single subtracted constant would not do and why an HV scan is usable at all. The <b>mean</b> includes the beam. The difference is beam-induced current; divided by the pulse rate, it is charge per pulse.</li>
          <li><b>Where the doubt came from.</b> The burst lasts milliseconds; the samples are a second apart. A monitor that reported an <i>instantaneous</i> current would sit at baseline almost always and catch a burst once in ~350 samples — and every charge we quote would have been a lower bound by a factor nobody had measured.</li>
          <li><b>The two numbers that settle it, using no timestamps at all.</b> <b>27 %</b> of samples sit above baseline, not 0.3 %. And the largest single-sample excess ever recorded on that channel is <b>0.216 µA</b>, where an instantaneous read of the same charge in 10 ms would have to read <b>10.1 µA</b> — 47× larger. Turn that round: a sample that contains a whole burst and reads 0.216 µA <i>must</i> be averaging over at least Q/ΔI = <b>0.47 s</b>. Both discriminate by two orders of magnitude, in the same direction.</li>
          <li><b>And the trap we had to beat.</b> The HV logger writes whole-second timestamps but its loop runs at <b>1.0162 s</b>, so the true read time drifts through its labelled second and a raw-label fold is smeared by a 1 s box — which would fabricate a wide response from a narrow one. The drift is also the cure: the pattern of dropped seconds over-constrains the logger's own clock, and solving it recovers <b>95 % of the sample times to 2 ms</b>. The sharp fold is what the figure shows.</li>
        </ul>
        <table class="spec-table">
          <tr><td class="k">Null #1</td><td>randomise the pulse times and the fold goes flat: &chi;&sup2;/ndf <b>1.03 vs 63</b></td></tr>
          <tr><td class="k">Null #2</td><td>the <b>drift-cathode</b> channel — same crate, same logger, no avalanche current — is constant to the last digit</td></tr>
          <tr><td class="k">Causality</td><td>zero response <b>before</b> the pulse, to &plusmn;2 nA, on the reconstructed time base</td></tr>
          <tr><td class="k">Linearity</td><td>n_TOF's two intensity bands (414 vs 853 &times;10<sup>10</sup> p) give the same charge per proton to <b>4 %</b> — no amplitude-dependent loss, and every logged pulse really does deliver charge</td></tr>
          <tr><td class="k">Clock offset</td><td>the intensity log is on CERN timing, the HV log on the DAQ host clock. A &plusmn;1 h lag scan bounds the offset at <b>~1 s</b> — and an offset <i>shifts</i> this curve, it cannot widen it</td></tr>
        </table>
      </div>
      <div class="caption">Measured on run_79 at the production setpoint (the same setpoint as the run the headline numbers come from, and the only one with the monitor log on disk here); its det C charge agrees with the headline run's det C to 5 %. Full method and caveats: <code>ntof_july_analysis/flash_charge/HANDOFF_FLASH_CHARGE_2026-08-09.md</code> §8.</div>
    </section>
```

---

## 2. REPLACEMENT — the main slide "How much charge are we talking about?"

Only the `<div class="caption">` changes. The three stats are unchanged: **142 nC
stands as a measurement.**

**Current:**

```html
      <div class="caption">Measured from the resistive-layer supply current — the one handle on this that does not go through the saturated readout. Method and validation on the next-but-one slide.</div>
```

**Replace with:**

```html
      <div class="caption">Measured from the resistive-layer supply current — the one handle on this that does not go through the saturated readout. The readback's response to a single beam pulse has now been measured directly, so this is a number and not a lower bound: it averages over <b>&ge; 0.47 s</b> against a millisecond burst, and four independent estimators of the charge agree to <b>&plusmn;3 %</b>. Method and validation in backup.</div>
```

*(If you keep the old phrase "on the next-but-one slide", check it still points at
the right slide — the backup section moved on 2026-08-10.)*

---

## 3. REPLACEMENT — the backup slide "How the charge per pulse is measured"

Its last table row currently states this systematic as **open**. That row is now
wrong.

**Current (last row):**

```html
          <tr><td class="k">Open systematic</td><td>assumes the supply's current readback preserves the time-average of a burst shorter than the sample spacing. 28 % of samples sit above baseline on both clean chambers, consistent with ~1 s smoothing (which conserves the integral). <b>If not, every charge quoted is a lower bound.</b></td></tr>
```

**Replace with:**

```html
          <tr><td class="k">Readback response</td><td><b>measured, not assumed</b> (2026-08-10): phase-folding imon against the individual beam pulses gives the monitor's response to one impulse of charge — it rises over 0.3 s, peaks at 88 nA, and is back to zero by 2.3 s, with an <b>area equal to the charge</b>. A timestamp-free bound puts the averaging window at <b>&ge; 0.47 s</b> against a ~10 ms burst, so <b>the time-average is preserved and the charges are measurements, not lower bounds</b>. Four estimators agree to &plusmn;3 %. See the imon backup slide.</td></tr>
```

**Optional extra row**, if you want the remaining honest caveat visible — it is a
different question from the one that was open, and nothing in the talk depends on
it:

```html
          <tr><td class="k">Still not calibrated</td><td>this shows the readback conserves the <i>integral</i> of a short burst; it does not check the absolute accuracy of its nA scale, which is a board-datasheet question. The exact CAEN card model is recorded in neither repo and the live crate was deliberately not probed.</td></tr>
```

---

## 4. One question for you

**Which CAEN board is card 5 of the crate at 128.141.177.244?** It is the last
thing that would let us put a datasheet number on the readback's absolute nA
accuracy (the integral-conservation question is closed either way, so nothing is
blocked). The model appears in neither `nTof_x17` nor `nTof_x17_DAQ`, and I did
not probe the crate — four chambers are on production physics and its CFE server
has crashed mid-scan before (`nTof_x17_DAQ/docs/incident_2026-07-05_hv_cfe_crash.md`).

---

## 5. If you want to check any of this yourself

```bash
cd ~/PycharmProjects/nTof_x17
.venv/bin/python ntof_july_analysis/flash_charge/imon_response.py \
    --src /media/dylan/data/x17/beam_july --run run_79     # ~2 min, prints everything
.venv/bin/python ntof_july_analysis/flash_charge/make_imon_figure.py
```

The first command prints the whole chain — cadence, time-base reconstruction
quality, clock lag scan, the timestamp-free counts, the response metrics per
chamber, the intensity-band linearity and the nulls — and writes
`results/imon_response_run_79.json` plus the fold CSVs the figure reads.
