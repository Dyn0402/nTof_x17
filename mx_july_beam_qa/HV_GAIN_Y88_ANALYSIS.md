# Plastic PMT gain curves + Y-88 absolute scale → setting the equalized HV

**Goal:** document the two plastic HV scans with their gain curves in the
**current (FIFO) readout**, anchor them to the Y-88 absolute energy scale, and
turn that into a recipe for setting the equalized operating voltages so the
plastic signal lands where we want it. Also: flag the large Y-88-vs-triples-MIP
disagreement for a later recheck.

Scripts: `23d_hv_gain_absolute.py` (this analysis), `22`/`23` (Y-88 edges/scale),
`19c` (FIFO/BNC-T ratio). Outputs: `calib/plastic_hv_gain_absolute.json`,
`figures/21_y88/hv_gain_curves.png`, `figures/21_y88/hv_absolute_scale.png`.

## 1. The two HV scans and the readout change

| scan | run | plastic readout | date | what it gives |
|------|-----|-----------------|------|---------------|
| scan 1 | **224466** | **BNC-T split** (old) | 07-16 | old gain curve + old equalization |
| scan 2 | **224489** | **linear FIFO** (current) | 07-17 PM | **current gain curve** |

Both scans stepped all 8 plastic PMTs together, 1200–1600 V, and we read the
gain as the sideband-subtracted wall-tagged **coincident-median** amplitude (a
clean gain proxy; `12` cache). The FIFO recovers **×1.13–1.65 per PMT** over the
BNC-T (`19c`, matched-HV medians); at equal HV the two curves differ by exactly
that factor (Fig. `hv_gain_curves.png`, squares = FIFO above circles = BNC-T),
so the gain *shape* (power law) is unchanged — only the scale.

**Use the FIFO (224489) curve to set HV.** The BNC-T curve is kept only for
history / to define the FIFO factor.

## 2. The Y-88 absolute anchor

The coincident median has no absolute energy meaning. The Y-88 scan supplies it:
a **known 698.63 keVee Compton edge**, measured on BNC-T at the gain-equalized
bias (the Y-88 runs 224476–79 used those biases — confirmed from the 224489
pre-scan HV log). Converting to the FIFO config with the per-PMT FIFO factor
pins the absolute gain, per PMT, in the current readout:

```
amp_mV(E_keVee, V) = (E_keVee / 698.63) * edge699_fifo_mv * (V / V_equalized)^n
```

with `n`, `edge699_fifo_mv`, `V_equalized` per PMT in
`calib/plastic_hv_gain_absolute.json`. At the current (gain-equalized) biases the
699 keVee edge sits at **25–45 mV** and the gain is **35–64 mV/MeVee** across the
eight PMTs (Fig. `hv_absolute_scale.png`, dots = current bias).

## 3. Setting the equalized HV (scale-together)

Because the gain is now absolute, we can (a) equalize all PMTs to a common
amplitude for a reference energy and (b) slide the whole set to place the signal.
The table below is the per-PMT bias that puts the **699 keVee edge at a common
target amplitude** — the target is the single "scale-together" knob.

| PMT | n | FIFO | 30 mV | 40 mV | 50 mV | 60 mV | 70 mV |
|-----|---|------|-------|-------|-------|-------|-------|
| PSSA1 | 6.9 | 1.65 | 1230 | 1282 | 1324 | 1360 | 1390 |
| PSSA2 | 5.1 | 1.56 | 1168 | 1236 | 1292 | 1339 | 1380 |
| PSSB1 | 5.2 | 1.23 | 1429 | 1510 | 1577 | 1633 | 1683 |
| PSSB2 | 7.1 | 1.60 | 1241 | 1292 | 1333 | 1368 | 1398 |
| PSSC1 | 3.8 | 1.26 | 1201 | 1295 | 1374 | 1441 | 1501 |
| PSSC2 | 6.6 | 1.53 | 1304 | 1362 | 1409 | 1448 | 1482 |
| PSSD1 | 5.8 | 1.13 | 1322 | 1389 | 1444 | 1491 | 1531 |
| PSSD2 | 6.4 | 1.33 | 1439 | 1505 | 1559 | 1604 | 1643 |

Notes for choosing the target:
- The plastic **trigger threshold is ~4.9 mV** and **full scale ~2000 mV**.
- At target T, the 699 keVee edge is at T mV on every PMT; a signal of energy E
  scales as `T * E/699` mV. (What the physics signal energy is — a MIP, or an
  X17 deposit — is your call; §4 flags that the "MIP" energy is unsettled.)
- PSSB1 is the weakest tube — it needs the highest bias (e.g. 1577 V for the
  50 mV target); PSSA/PSSB2 the least. Keep an eye on the CAEN per-channel max.
- Higher target = more head-room above threshold but less below saturation, and
  higher bias (more dark current / spark risk). ~40–50 mV at 699 keVee looks
  like a sensible window (699 edge well clear of threshold, huge margin to
  saturation) — but pick against the real physics signal energy.

## 4. ⚠️ Y-88 vs triples-MIP: a ~40× disagreement (unresolved)

The Y-88 absolute scale (a known 699 keVee line) says the plastic gain is
**35–64 mV/MeVee**. The earlier triples "plastic MIP" (`19d`, `pss_mip_calib`)
assumed the MIP deposits **5.05 MeV** (2.5 cm PVT) and got **0.65–2.05 mV/MeV** —
a **~40× disagreement**. Put the other way: the Y-88 line places the triples MIP
peak at **~130 keVee**, not 5 MeV.

The Y-88 number is trustworthy — it is a known gamma Compton edge measured
directly. The triples MIP is the suspect: either the "MIP peak" it picked is not
a through-going 5 MeV MIP (mis-identified feature), or the triple-coincidence
sample was statistics-starved and the peak is spurious. **Action:** recheck the
triples MIP with a long run (more triple-coincidence statistics) before trusting
any 5 MeV-based plastic energy scale. Until then, use the Y-88 scale for absolute
plastic energy.

## 5. HV / config facts established (from the DAQ logs)

- Y-88 runs 224476–79 (07-17 AM) used the **gain-equalized biases** (AL 1303,
  AR 1242, BL 1376, BR 1279, CL 1180, CR 1307, DL 1303, DR 1417 V; `run_config_beam.py`),
  on the **BNC-T** readout — confirmed: the 224489 pre-scan HV log shows exactly
  these values standing before the evening scan ramped.
- Nothing logged the Y-88 window itself (local DAQ idle 08:08–12:11; its monitor
  covers only the Micromegas HV), but the before/after logs bracket it.
- The 224466 (BNC-T) LIQ reprocessing is running so the plastic MIP triples can
  be redone directly in BNC-T (and the first BNC-T liquid scale extracted).
