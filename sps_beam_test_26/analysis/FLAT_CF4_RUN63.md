# det4 flat at the operating point, new gas — run 63 / operating_03

The dataset the run_56 analysis asked for: det4 **flat**, at its **operating
point**, in the **new gas**, with the zero suppression one step looser.
Written 2026-08-03.

```bash
../../.venv/bin/python decode_dataset.py run63_flat --subrun operating_03
../../.venv/bin/python pair_dataset.py    run63_flat
../../.venv/bin/python flat_align_eff.py  run63_flat
../../.venv/bin/python tilt_m70V.py       --wf .../wf_run63_flat.npz --plateau flat700
../../.venv/bin/python kernel_fit_m70V.py --wf .../wf_run63_flat.npz --plateau flat700
```

Conditions live in `datasets.py` and nowhere else.

---

## 0. run_63 is TWO conditions — it straddles a zone access

The H4 TAX beam stopper (`XTAX_022_023:POSITION_MEAS`) dates it to the second.
That variable is logged **only by the mx17-daq NXCALS client**
(`ssh daq:~/beam_july/slow_control/sps_spill/h4_tax_2026-08-03.csv`); banco's
mirror does not carry it, which is why earlier campaign docs claim accesses
cannot be dated.

| state | window | |
|---|---|---|
| moving | 00:37:16 – 00:40:10 | closing |
| **blocked** | **00:40:11 – 00:57:55** | the access, 17.7 min |
| moving | 00:57:56 – 01:00:49 | opening |
| open | **01:00:50** → | beam back |

det4 was rotated from 25.64° back to **flat** during it. So:

| | sub-runs | mount | usable |
|---|---|---|---|
| `run63_rot25` | operating_00, _01 | **25.64°** | drift ladder 675→325 V at fixed resist |
| `run63_flat` | operating_02 tail, _03 | **flat** | 53.4 min at the operating point |

The "beam dip" first seen inside `operating_01` **is the stopper**, not a
machine fault. Recorded in `datasets.py`.

> **Correction to the previous session's note.** It concluded "run_63 is not
> flat" from a `det(A) = 1.1132` measured on `operating_01` alone — which is
> entirely pre-access and therefore genuinely 25.64°. The measurement was
> right; generalising it to all of run_63 was not.

---

## 1. Flatness, alignment, geometry

| | run_56 m70V (flat) | **run_63 operating_03** | run_63 operating_01 (25.64°) |
|---|---:|---:|---:|
| det(A) | 1.0090 | **1.0100** | 1.1132 |
| row scales | 1.0055 / 1.0035 | **1.0067 / 1.0033** | 0.9993 / 1.1141 |
| roll | +90.28° | **+90.03°** | +89.99° |
| median \|residual\| | 0.51 mm | **0.59 mm** | 2.52 mm |
| fitted z | — | **1100 mm** (real minimum) | unconstrained |

`operating_03` is flat, to the same precision as run_56. `run_config.json`'s
z = 1155 mm is stale as usual; the fit prefers 1100 mm with a genuine minimum.

**The ~0.4° residual tilt is real and persistent.** From the charge-centroid
walk:

| | run_56 (flat, CO₂) | run_63 (flat, CF₄) |
|---|---:|---:|
| X view | −0.239 µm/ns → **0.40°** | −0.232 µm/ns → **0.39°** |
| Y view | +0.026 µm/ns → 0.04° | −0.036 µm/ns → 0.06° |

Same number, two days apart, across a full remount cycle
(flat → 25.64° → 15.465° → 25.64° → flat) **and** a gas change. It is a
standing det4-vs-uRWELL pitch misalignment in the striped coordinate, not a
mounting accident, and it should be carried as a known offset.

## 2. Efficiency

| | drift | resist | clean tracks | fired | within 5 mm | in live bands |
|---|---:|---:|---:|---:|---:|---:|
| run_56, 625 V, CO₂ | 700 | 625 | 36,989 | 44.6 % | 25.9 % | 37.3 % |
| **run_63, operating, CF₄** | 700 | **770** | **84,738** | **66.1 %** | **38.5 %** | **56.2 %** |

In-band efficiency 37.3 % → **56.2 %**. Both the higher resist voltage and the
CF₄ mixture contribute; this dataset cannot separate them.

## 3. The sharing kernel, and the premise

Same fit as run_56: the neighbour waveform decomposed against the measured
central-strip waveform as `W_d = alpha_d W_0 + beta_d (W_0 * K_tau^|d|)`.

**run_63 flat, Y view** (the clean one — tilt 0.06°, ±2 acceptance 0.75):

| offset | α (prompt) | β (dispersed) |
|---|---:|---:|
| +1 | 0.2080 | 0.2327 |
| −1 | 0.2092 | 0.2333 |
| +2 | 0.0408 | 0.0855 |
| −2 | 0.0410 | 0.0822 |

±1 sides agree to 0.3 %, ±2 to 4 %. That is the best internal consistency of
any measurement in this campaign.

### Across gas, voltage and threshold

| | run_56 (CO₂ 95/3/2, resist 625 V, ZS 5σ) | run_63 (CF₄ 88/10/2, resist 770 V, ZS 4σ) | change |
|---|---:|---:|---:|
| `c1` X | 0.2506 | 0.2525 | **+0.8 %** |
| `c1` Y | 0.2805 | 0.2330 | −17 % |
| `c2` X | 0.0222 | 0.0306 | +38 % |
| `c2` Y | 0.1108 | 0.0838 | −24 % |
| `tau_s` X | 357 ns | 282 ns | −21 % |
| `tau_s` Y | 298 ns | 215 ns | −28 % |
| prompt α(±1) Y | 0.190 | 0.209 | +10 % |

**`c1` = 0.23–0.28 across two gases, a 145 V resist step and two ZS
thresholds.** That is the headline, and it supports the premise: the ±1
sharing amplitude is a property of the resistive layer and readout, not of the
gas or the operating point. Combined with the run_56 internal gain scan
(590→625 V moved `c1` by 1.1 %), the evidence is consistent and it agrees with
the bench's independently-inferred ~29 %.

**`tau_s` and `c2` are not yet transferable claims.** Both moved 20–30 %, and
that is the same size as the censoring systematic already flagged: the ZS
acceptance panels show the **central strip's own window closing ~400 ns after
its peak**, so the basis waveform is truncated in exactly the region the tail
fit depends on. Going 5σ → 4σ improved the *neighbour* acceptance (±2 up to
0.75 from 0.6) but not the basis, so the systematic did not go away — and the
fit visibly undershoots the late tail beyond +500 ns in both runs, meaning the
single RC cascade is an approximation to a heavier tail.

So: `c1` is measured. `tau_s` is "a few hundred ns" and `c2` is O(0.03–0.11)
per view, both dominated by acceptance, not statistics.

## 4. What is still missing

- ~~**No drift lever in the flat data.** Every drift scan (passes A–D, drift
  675→125 V at fixed resist 770 V) ran *before* the access, so the whole
  diffusion lever sits at 25.64°.~~ **RETRACTED 2026-08-05 (late).** Two flat
  CF₄ drift scans ran *after* the access, in runs never listed in any analysis
  table: **run_68** (`det4_drift_scan_700_100_64smp.csv`, 700→100 V, **64
  samples**, but only ~40 MB of det4) and **run_70**
  (`det4_drift_scan_600_100.csv`, 600→100 V, 2.7 GB, 32 samples so
  window-railed for v(E)). `EXTRACTION_2026-08-05b.md` §8. The follow-on
  sentence still stands: separating `sigma_p0`/`Dp` from `c1`/`c2` needs a flat
  drift scan or the forward fit on the 25.64° ladder — and the forward fit was
  tried and did not resolve it (`EXTRACTION_2026-08-05b.md` §2), which makes
  run_68/run_70 more interesting than they looked.
- **The basis truncation is the binding systematic.** A short run with det4 in
  RAW (non-ZS) mode, or at 3σ, would record the central strip's full tail and
  turn `tau_s` and `c2` into real measurements. That is a cheap ask while the
  beam is up.
- `operating_02`'s post-access tail (23 min more flat beam) is not staged.

## 5. Data

Staged at `/media/dylan/data/x17/sps_run53_det4_check/staging/run_63/`:
`operating_03/` (5 file groups, 4.3 GB FEU3), `pair_run63_flat.npz`
(1.74 M events, 18.5 M det4 hits), `wf_run63_flat.npz` (27 k selected events),
`eff_run63_flat.png`, `flat_tilt_flat700.png`, `flat_kernel_fit_flat700.png`.
