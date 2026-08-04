# det4 flat at high voltage — run 56 / meshscan_m70V

Analysis of the one sub-run where det4 sat **perpendicular to the beam at its
highest flat voltage**: `run_56/meshscan_m70V`, 2026-08-01 15:47:25–15:59:34,
Ar/CO₂/iso 95/3/2, drift held 700 V, resist stepping 590 V → 625 V inside the
sub-run. Written 2026-08-03.

Scripts (all in this directory, run in order):

```bash
../../.venv/bin/python pair_m70V.py               # det4 hits <-> uRWELL tracks
../../.venv/bin/python align_eff_m70V.py          # alignment + efficiency maps
../../.venv/bin/python extract_waveforms_m70V.py  # waveform windows, clean events
../../.venv/bin/python kernel_fit_m70V.py --plateau 625V   # the sharing kernel
../../.venv/bin/python tilt_m70V.py --plateau 625V         # is it really flat?
```

Outputs in `/media/dylan/data/x17/sps_run53_det4_check/staging/run_56_m70V/`.

---

## 0. Configuration — three things the record had wrong

Taken from the machine record, not from `run_config.json`:

| | value | source |
|---|---|---|
| samples × period | **64 × 60 ns** (3.84 µs) | `run_56/run_config.json` `dream_daq_info` |
| ZS threshold | **5 σ**, TPC mode | `..._03_thr.prg` header, *"Threshold value: 5.000000 sigmas"* |
| pedestal subtraction | on-FEU → `--zs-baseline 1` | `dream_daq_info.pedestal_subtraction` |
| Dream peaking | 180 ns → matched filter 5 samples | `P2B_Beam.cfg`, `(0xd023>>4)&0xF = 2` |

Three corrections to `RUN_TIMELINE.md`:

1. **run_56 ran at 64 samples, not 32.** §3 says 32 normally and 64 only for
   drift scans; the flat resist ladder was also at 64.
2. **det4 was at 5 σ during run_56, not 2 σ.** §3's threshold history is wrong
   before 16:20. The `_thr.prg` copied into the sub-run directory settles it.
3. **`dream_daq.log` exists on EOS for runs 54, 55 and 56**, not just run_61.
   Real sub-run boundaries differ from the QA-PNG-mtime table in §5 by 10–15
   min (m70V is 15:47:25–15:59:34).

---

## 1. Alignment and efficiency

627k events paired, 3.14M det4 hits, 70,106 clean single-cluster uRWELL tracks.

| plateau | clean tracks | roll | det(A) | median \|residual\| |
|---|---:|---:|---:|---:|
| 590 V | 31,724 | +90.27° | 1.0089 | 0.51 mm |
| 625 V | 36,989 | +90.29° | 1.0094 | 0.51 mm |

Consistent with the independent run_53 mapping (+90.20°, 0.46 mm), so the
inverted-connector strip map holds at this voltage too.

| plateau | whole spot | inside June live bands | between bands |
|---|---:|---:|---:|
| 590 V | 23.7 % | 34.1 % | 0.1 % |
| 625 V | 25.9 % | 37.3 % | 0.1 % |

Best band X 146–164 mm reaches **60.2 %** at 625 V (53.3 % at 590 V). The dead
stripes are still dead to 0.1 % — no voltage fills them, as the pre-trip
assessment concluded. Efficiency is still climbing at 625 V.

`m70V_efficiency.png`.

---

## 2. det4 is NOT flat — it is tilted ~0.4°

This was checked because the X-view sharing came out asymmetric, and it
changes what the rest of the analysis is allowed to claim.

**The uRWELL track slopes do not answer this.** They measure where the *beam*
points in the *uRWELL* frame (mean 0.18°, rms 0.61°). det4's own mounting
angle is a separate quantity, and the 2-D affine alignment is blind to it — a
tilt only rescales the projected footprint.

Measured instead from the **charge centroid walk**: at normal incidence the
ionisation column lands at one transverse position, so the charge-weighted
centroid must not move as the column drifts in. It does:

| view | centroid walk | tan θ | θ (at v = 34 µm/ns) |
|---|---:|---:|---:|
| **X** (striped coord.) | **−0.239 µm/ns** | −0.0070 | **0.40°** |
| Y | +0.026 µm/ns | +0.0008 | 0.04° |

The X walk is clean and monotonic — 0.4 mm over 1.4 µs (`tilt_625V.png`).

**It is a detector tilt, not a beam tilt.** Splitting events by their
uRWELL-measured track slope, the walk stays at −0.23 to −0.29 µm/ns in every
bin; the slope dependence is +0.015 µm/ns per degree against the +0.593
that pure geometry demands — consistent with zero. The intercept at zero track
slope is −0.233 µm/ns, i.e. a fixed **mount tilt of 0.39° in X, 0.04° in Y**.

θ scales as 1/v_drift, and v is not measured for this gas at this field
(the 34 µm/ns is the June Ar/iso value at 1000 V, a stand-in). The range
v = 20–40 µm/ns gives θ = 0.34–0.68°. So: **0.4°, uncertain to about a factor
1.7, in the striped coordinate only.**

Small — but over the 30 mm drift it displaces late-arriving charge by ~0.2 mm
(¼ strip), which is exactly enough to make the X-view ±2 sharing asymmetric.
**Consequence: quote the kernel from the Y view. The X view is contaminated.**

---

## 3. The charge-spreading kernel

### What the flat geometry buys, and what the zero suppression costs

At normal incidence every 60 ns slice of drift charge lands at the same
transverse position (`w = 0` in `wft/model.py`), so any charge on a
neighbouring strip is either **prompt** (direct transverse diffusion of the
same column) or **delayed** (through the resistive layer). Two components,
separated in time — which is the degeneracy a cosmic fit cannot break.

The zero suppression is the limiting systematic, and it was characterised
before anything was fitted:

* ZS is **per channel** at 5 σ ≈ 41 ADC. The ±2 strip is present only 29–47 %
  of the time overall; beyond \|d\| = 3 what is left is the accidental floor
  (5–8 %). Fixed by requiring a **strong central strip (900–3000 ADC)**, where
  ±2 is present 99.5 % of the time — the tail is then essentially uncensored.
* The kept sample window **grows with amplitude**: 5 samples (300 ns) at
  threshold, 17–25 samples (1.0–1.5 µs) above ~100 ADC. So the central strip
  carries a real shape and weak neighbours carry a usable peak time.
* Saturated channels (>3000 ADC) show a repeated-constant readout pathology in
  28.8 % of cases. Excluded; 0.26 % of all kept channels overall.

### The measurement

The mean waveforms (`kernel_fit_625V.png`) show the neighbours peaking
**with** the central strip but carrying a long tail the central strip does not
have. So the sharing is not "a copy delayed by `tau_s`" — it is an
RC-**dispersed** copy. That is the `share_lp` branch of `wft/model.py`
(`_lp_copies`), and this is a direct measurement of it rather than an
inference from a track fit.

Fitted per offset, using the measured central-strip waveform as the basis (so
no shaper template, no drift-ladder model and no v_drift enter):

```
W_d(t) = alpha_d * W_0(t)  +  beta_d * (W_0 (*) K_tau^|d|)(t)
```

**Y view, 625 V** (the trustworthy one — tilt 0.04°, ±2 uncensored, symmetric):

| offset | α (prompt diffusion) | β (dispersed share) |
|---|---:|---:|
| +1 | 0.192 | 0.275 |
| −1 | 0.188 | 0.286 |
| +2 | 0.037 | 0.110 |
| −2 | 0.031 | 0.111 |

| parameter | value |
|---|---|
| `tau_s` (RC dispersion) | **298 ns** |
| `c1` | **0.281** |
| `c2` | **0.111** |
| prompt diffusion onto ±1 | 0.190 |

The ±1 and ±2 sides agree to 2–4 %, which is the internal consistency check
the tilted X view fails.

### Reconciling with the bench's "29 % at τ ≈ 47 ns"

`RECONSTRUCTION_BASIS.md` and the analyzer README quote ~29 % sharing to ±1 at
τ ≈ 47 ns, inferred from the forward-model fit. This measures **c1 = 0.281**
directly — agreement to a few percent, from completely independent data and
method. That is a real validation of the bench number.

The time constants look discrepant only because they parameterise different
things. The measured *peak-time shift* of the ±1 strip is **+29 ns (X), +36 ns
(Y)** — right on the bench's 47 ns delay. The 298 ns is the RC *dispersion*
constant of the tail, not a delay. Both describe the same tail: the shared
charge is spread over hundreds of ns, so it barely moves the peak while
building a long tail. **The plain-delay parameterisation and the `share_lp`
one are fitting the same physics; `share_lp` is the better description**, and
its constant is ~300 ns, not 47.

### The premise: is the kernel gain-independent?

590 V vs 625 V is a 35 V (≈6 % gain) step within the same sub-run, same gas,
same mount, same pedestal set:

| | 590 V | 625 V | change |
|---|---:|---:|---:|
| `tau_s` (Y) | 296 ns | 298 ns | +0.7 % |
| `c1` (Y) | 0.278 | 0.281 | +1.1 % |
| `c2` (Y) | 0.114 | 0.111 | −2.4 % |

**The kernel does not move with gain at the 1–2 % level.** That is the first
direct support for the premise that these parameters belong to the resistive
layer and readout rather than to the operating point. It is *not* yet a test
of gas independence — that needs the drift/gas lever in §5.

### X/Y asymmetry

From the (tilt-contaminated) X fit, `kY = c1_Y/c1_X = 1.12` and
`kTauY = 0.83`. Both are real effects — the two views sit at different depths
under the resistive layer — but the X numbers carry the tilt systematic, so
treat these as indicative, not calibration-grade.

---

## 4. What this does and does not license

**Does:** det4's own kernel at this condition, with c1 measured directly and
agreeing with the bench inference; the `share_lp` structure confirmed and its
time constant measured; gain independence demonstrated over 6 %.

**Does not:** transfer to det2/det3/det6/det7. `CLAUDE.md` is explicit that a
kernel is per detector — surface resistivity varies chamber to chamber. What
transfers is the *structure* (RC dispersion, not delay) and the *finding* that
the kernel is gain-independent, which is what lets a bench kernel be reused
across that detector's own run conditions.

**Open systematic:** the ZS acceptance panels in `kernel_fit_625V.png` show
the central strip's own window closing ~400 ns after its peak, so the basis
waveform is itself censored in the tail. `tau_s` and `c2` are therefore
lower-confidence than `c1`; τ carries roughly a ±30 % systematic from where
the fit window is cut. `c1` is robust because it is dominated by the
well-sampled core.

---

## 5. Next — and there is now better data for it

`run_63/operating_00…03` (2026-08-02 23:53 → 08-03 01:24+) is **flat, at
operating voltage, for two hours** — resist held 770 V, drift stepping
(675 V, 625 V, …), Ar/CF₄/iso 88/10/2, 64 samples, **ZS at 4 σ** (one step
looser than run_56). That is the dataset this analysis wanted:

1. **The drift lever.** Fixed resist with drift stepping varies diffusion
   while leaving the resistive layer alone — the clean separation of `sigma_p0`
   / `Dp` (gas) from `c1` / `c2` / `tau_s` (layer), and the actual test of the
   gas-independence premise.
2. **Different gas** (CF₄ vs CO₂), so it tests transfer across the one axis
   run_56 cannot.
3. **Two hours instead of seven minutes**, and a looser threshold, so the
   dispersed tail is less censored — directly attacking the dominant
   systematic above.

Also worth doing: measure v_drift for this gas so the 0.4° tilt firms up, and
re-run the X kernel with the tilt in the model rather than dropping the view.
