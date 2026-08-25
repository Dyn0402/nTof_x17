# Prior art for the waveform-first forward model

Who to credit when `wft/` is written up. Compiled 2026-08-23, on Dylan's ask:
"I assume you pulled the idea from existing literature — find who did something
similar so I credit the correct people."

## Provenance, honestly

**The model in `wft/model.py` was not taken from a paper.** There is no hidden
citation. Its shape came from our own data: the `lp` (RC-dispersed) neighbour
kernel was adopted over the original delayed-copy form because the delayed copy
could not fit the observed tail without an unphysical `sigma_p0` — see
`M70V_FLAT_ANALYSIS.md §3` and `RAW_RUN71_REANALYSIS §4`, which are the sources
the module docstring already names.

That makes this an independent re-derivation, not a borrowing. It does **not**
make it novel. Every ingredient below is established in the literature, some of
it for twenty years, and the credit is genuinely owed. Write it up as "following
X, in the form of Y, extended by us in direction Z" — not as new.

## The bibliography, by which part of the model it covers

### 1. Charge dispersion on a resistive anode — the founding idea

**M.S. Dixit, J. Dubeau, J.-P. Martin, K. Sachs**, *Position sensing from charge
dispersion in micro-pattern gas detectors with a resistive anode*, **NIM A 518
(2004) 721–727**, arXiv:physics/0307152.

The seminal paper, and the one to cite first. It establishes the entire premise:
the resistive layer plus the readout plane form a distributed 2-D RC network;
localised avalanche charge disperses across it with an RC time constant set by
the sheet resistivity and the anode–readout capacitance density; and — the point
that matters for us — the signal on a *non-collecting* neighbour is real,
physical, and carries position information rather than being contamination.

They also state the problem `wft/` exists to solve, in 2004: conventional cluster
reconstruction cannot be used on a resistive anode, because the pulse *shape* on
a given electrode depends on where the track is relative to it.

Companions:

- **M.S. Dixit, A. Rankin**, *Simulating the charge dispersion phenomena in
  micro pattern gas detectors with a resistive anode*, **NIM A 566 (2006)
  281–285**, arXiv:physics/0605121. The analytic model function for the
  dispersed signal, folding initial ionisation clustering, drift, diffusion,
  intrinsic pulse shape and electronics. This is the closest thing in the older
  literature to our per-plane *template*.
- **A. Bellerive, K. Boudjemline, R. Carnegie, M. Dixit, J. Miyamoto et al.**,
  *Spatial resolution of a Micromegas-TPC using the charge dispersion signal*,
  eConf **C050318** (2005) 0829, arXiv:physics/0510085.
- **K. Boudjemline, M.S. Dixit, J.-P. Martin, K. Sachs**, *Spatial resolution of
  a GEM readout TPC using the charge dispersion signal*, **NIM A 574 (2007)
  22–27**, arXiv:physics/0610232.

### 2. T2K ND280 — the closest published *method*

**D. Attié et al.** (T2K ND280 TPC upgrade), *Characterization of charge
spreading and gain of encapsulated resistive Micromegas detectors for the
upgrade of the T2K Near Detector Time Projection Chambers*, **NIM A 1056 (2023)
168534**, arXiv:2303.04481.

**Cite this most prominently.** The method is nearly ours. They build a signal
model that is *charge spreading ⊗ electronics response* and fit the waveforms of
a 3×3 pad matrix **simultaneously** by χ², requiring at least three waveforms to
constrain the position, and extract the RC constant and the gain together. Their
description of the neighbour signal — smaller in amplitude, delayed, longer in
time, with the delay growing with distance from the leading pad — is a
description of our `share_mode` kernel.

The difference is what the fit is *for*. They fit transverse position and
calibration constants on X-ray point deposits. We fit a whole drift-depth charge
profile by NNLS and pull `(p0, w, t0)` out of it — i.e. we use it as a µTPC.

Companions from the same programme:

- **D. Attié et al.**, *Characterization of resistive Micromegas detectors for
  the upgrade of the T2K Near Detector Time Projection Chambers*, **NIM A 1025
  (2022) 166109**, arXiv:2106.12634.
- **D. Attié et al.**, *Performances of a resistive Micromegas module for the
  Time Projection Chambers of the T2K Near Detector upgrade*, **NIM A 957 (2020)
  163286**, arXiv:1907.07060. Has the ~80 %-on-the-leading-pad and 2–3-pad
  cluster numbers, and the µs-scale spreading delay.

### 3. Resistive *strips* — the geometry-correct citation for our detectors

**J. Galan et al.**, *Signal propagation and spark mitigation in resistive strip
read-outs*, **JINST 7 (2012) C04009**, arXiv:1110.6640.
Also **arXiv:1304.2057**, *Characterization and simulation of resistive-MPGDs
with resistive strip and layer topologies*.

Sections 1 and 2 are resistive-*anode* pad work. Ours are resistive *strips*, and
this is the paper for that: resistive strips facing metallic strips create an
inter-capacitance and therefore an electric transmission line, solved by
subdividing the strip into differential elements with Kirchhoff continuity at
each node.

This is the physical justification for two things in `model.py`:

- the one-pole low-pass **cascaded once per strip step** (`lp` share mode), and
- the invariant that **`c2 < c1`**, because the ±2 strip is reached only through
  the ±1. See `wft.calib.check_kernel_ordering` and
  `mx_june_wft/RETIRE_C2GTC1_2026-08-21.md` — the gate that refuses an inverted
  bundle is Galan's transmission line expressed as an assertion.

### 4. The underlying signal theory

**W. Riegler**, *Electric fields, weighting fields, signals and charge diffusion
in detectors including resistive materials*, **JINST 11 (2016) P11002**,
arXiv:1602.07949. Also *Studying signals in particle detectors with resistive
electrodes*, arXiv:2304.01883.

The extended Ramo–Shockley theorem for conductive media: in a geometry with
finite-conductivity material the weighting field itself becomes time-dependent,
found by solving Poisson in the Laplace domain with ε(x) → ε(x) + σ(x)/s. This is
the first-principles ground for treating the neighbour's signal as
induced-and-dispersed rather than as collected charge — which is exactly the
assumption that lets the kernel live *inside* the forward model.

### 5. The µTPC half — and why we cite it as a foil

- **T. Alexopoulos et al.**, *A spark-resistant bulk-micromegas chamber for
  high-rate applications*, **NIM A 640 (2011) 110–118** — resistive strips for
  Micromegas.
- **T. Alexopoulos et al.**, *Performance studies of resistive-strip bulk
  micromegas detectors in view of the ATLAS New Small Wheel upgrade*, **NIM A
  937 (2019) 125–140**.
- **ATLAS Collaboration**, *Muon spectrometer Phase-I upgrade: the New Small
  Wheel project*, arXiv:1810.01394.

The per-strip-time µTPC: strip position gives one coordinate, the strip's own
arrival time times the drift velocity gives the other, and a local fit in the
drift gap gives the angle.

Cite these **precisely because `RECONSTRUCTION_BASIS.md` is an argument against
doing it that way on a resistive detector.** Our measured result is that the
aggregate per-strip hit time compresses the drift ladder by 20–30 % and reads
~4° too steep, estimator-independently. That is the gap our extension fills, and
it is the honest way to position the work: not "we invented a forward model" but
"the µTPC and the charge-dispersion literature had not been joined, and on a
resistive detector they must be."

## What appears to be ours

No paper found puts the neighbour-sharing kernel inside a **global forward fit
that recovers track angle and drift depth**. The split in the literature is
clean:

- charge-dispersion people fit neighbours to sharpen the **centroid** (a position);
- µTPC people use **per-strip times** to get the angle.

Joining them — a per-depth-slice NNLS charge profile, folded through the
geometric strip integral, then the resistive kernel, then the measured impulse
response, with `(p0, w, t0)` searched — is the part we should claim.

**Caveat on that claim.** This rests on roughly a dozen web searches on
2026-08-23, not a systematic review. Before asserting novelty in print, run a
targeted check of recent µRWELL literature and of the MPGD work around
IDEA/FCC-ee, and check whether any of the T2K ND280 track-level (as opposed to
X-ray-level) analyses fit the drift profile rather than the centroid.

## Suggested citation sentence

> The forward model follows the charge-dispersion picture of Dixit *et al.* [1],
> in the resistive-strip transmission-line form of Galan *et al.* [2], and is
> fitted simultaneously across neighbouring channels as in the T2K ERAM analysis
> [3]. We extend it by solving for the drift-depth charge profile, giving a µTPC
> measurement that is immune to the neighbour-induced time bias which corrupts
> per-strip-time µTPC on resistive detectors [4].

[1] NIM A 518 (2004) 721 · [2] JINST 7 (2012) C04009 · [3] NIM A 1056 (2023)
168534 · [4] this work, `RECONSTRUCTION_BASIS.md`.

## Where this is referenced from

- `wft/model.py` — module docstring.
- `RECONSTRUCTION_BASIS.md` — "What replaces it".
- `mpgd26/slides/index.html` — slide 9.2 carries a one-line T2K credit under the
  measured-kernel figure.
