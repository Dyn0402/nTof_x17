# Scintillator calibration — split into two deep-dive decks (rev 2)

Two self-contained beamer decks, one per detector type, + a LIQ appendix on the
plastic deck. Deep-dive / pedagogical. Figures in `figures/`; numbers from
`calib/*.json`. Build with `pdflatex`.

## Corrections folded into this revision (Dylan, 7-20)
1. **SiPM "MIP" = wall × plastic-behind coincidence, NOT same-bar top/bottom.**
   Top+bottom coincidence only says the bar was hit at both ends (a gamma fires
   both ends too) — that's a HIT spectrum, not a MIP. The real MIP is the
   wall×plastic telescope. All top/bottom-as-MIP framing removed.
2. **Add run dates/times to the slides** so the morning→evening gain shift is
   legible at a glance (timeline table below).
3. **Plastic triple-coincidence MIP is NOT trusted — it is garbage and needs
   work.** State this plainly; it is not a calibration, it's a flagged failure.
4. **Y-88 spectra in lin–lin only** (no log-y). Current `21_y88` / `22` figures
   are linear-x but log-y → regenerate with linear y for the deck.
5. **The long-run recheck already exists:** run **224503**, script `23e`. It uses
   the proper wall×plastic MIP. Result: clean per-arm SiPM wall MIP, but the
   plastic triple MIP is STILL swamped (no peak at predicted ~120–217 mV) →
   statistics was not the fix; the method needs rework.

## Run timeline (put a compact version on slide 2 of each deck)
| run | what | readout | date/time (local) |
|-----|------|---------|-------------------|
| 224466 | plastic HV scan | BNC-T | 2026-07-16 |
| 224476 | Y-88 arm A | BNC-T | 2026-07-17 10:25:57–10:38:55 |
| 224477 | Y-88 arm B | BNC-T | 2026-07-17 10:46:45–10:59:41 |
| 224478 | Y-88 arm C | BNC-T | 2026-07-17 11:12:43–11:26:38 |
| 224479 | Y-88 arm D | BNC-T | 2026-07-17 11:29:19–11:42:16 |
| 224489 | beam (HV scan 2, FIFO, 1st LIQ) | FIFO | 2026-07-17 17:39 (PM) |
| 224503 | long beam run (MIP recheck) | FIFO | 2026-07-17 later / overnight (exact TBD) |

The shift-relevant gap: Y-88 walls at **10:25–11:42** vs beam MIP at **17:39**,
same day → ~6–8 h. `224503` extends it further. **Next step (Dylan):** pull a
beam run from BEFORE ~10:25 to anchor the MIP/gain baseline on the early side.

---

## DECK 1 — SiPM (wall) absolute calibration  →  `sipm_calib_slides.tex`

**Thesis:** the walls have two independent absolute handles — the Y-88 699 keVee
Compton edge (source, AM) and the through-going **wall×plastic MIP** (224489 +
long run 224503). They agree; the MIP sits at ~0.9 MeV; the one wrinkle is a real
~40 % SiPM gain rise AM→PM on B/C/D.

| # | Frame title | Figure(s) | Key numbers / point |
|---|---|---|---|
| 1 | Title | — | "SiPM Wall Absolute Calibration: Y-88 edge ⋈ wall×plastic MIP" |
| 2 | What a SiPM wall is + run timeline | `11_diagrams/concept_wall_plastic.png` + timeline table | 20 bars/arm, top+bottom SiPM; organic scint → Compton edge is the landmark; dates on-slide |
| 3 | **What a MIP is here (and isn't)** | `mx17_geom` schematic / `11_diagrams/concept_wall_plastic_liq.png` | MIP = charged track through wall AND the plastic behind (telescope). **Same-bar top+bottom coincidence = a HIT, not a MIP** — gammas fire both ends |
| 4 | Method A — beam MIP via wall×plastic coincidence | `19_triples/wall_in_triples.png` (224489) | wall amp of wall hits with a plastic partner, sideband-subtracted; clean Landau |
| 5 | Beam MIP spectra, per channel | `06_07_geometry_mip/mip_wall_spectra_linear.png` | Landau MPV 29–39 mV on B/C (lin–lin) |
| 6 | Method B — Y-88 699 keVee Compton edge | `21_y88/edges_run224476.png` (wall panels, **lin–lin**) | erfc step + linear bkg; bootstrap errors; convention stated |
| 7 | Y-88 wall edges uniform & stable across all 4 AM runs | `21_y88/wall_edge_stability.png` | 26–30 mV every arm × run (`y88_wall_edge_matrix.json`) → morning gain flat & uniform |
| 8 | **Head-to-head: Y-88 edge vs wall×plastic MIP** | `21_y88/y88_vs_beam_mip.png` | edge/MIP median 0.747 → E_MIP ≈ 0.94 MeV. ⚠ NOTE: current JSON used the 224489 top/bottom proxy → **redo with the wall×plastic MIP** (number robust, label must change) |
| 9 | ⚠ Discrepancy: A agrees, B/C/D MIP ~1.4× high | annotate slide 8 | A 0.98; B/C/D 0.63–0.70. Y-88 edges uniform ⇒ a *time* change AM→PM, not nonuniformity → **~40 % SiPM gain rise; confirm bias/temp records** |
| 10 | Long run 224503: proper wall×plastic MIP, high stats | `21_y88/mip_224503.png` | per-arm SiPM MIP clean; confirms scale & the A-vs-B/C/D split (plastic swamped — teaser) |
| 11 | Result: absolute scale, mV/keVee per wall channel | table (`y88_energy_calib.json` WAL rows, **lin**) | ~33–39 mV/MeVee arm A — the deliverable |
| 12 | Trigger thresholds — top+bottom SUM at 0.5× the sum peak | `21_y88/sipm_sums.png` | per-group threshold 18–37 mV (`sipm_sum_thresholds.json`). NB the SUM peak is a through-bar-hit landmark (flux MIP-dominated), used only to set a trigger level |
| 13 | Summary & open items | — | MIP≈0.9 MeV, scale delivered; open: AM→PM gain-drift confirm, redo y88-vs-MIP with wall×plastic MIP, top/bottom slot assumption, 4/20 bars unread; **next: pre-Y88 baseline run** |

---

## DECK 2 — Plastic (PSS) absolute calibration  →  `plastic_calib_slides.tex`

**Thesis:** the hard case. The **only trusted plastic scale is Y-88** (699 keVee
known line, raised HV). The triples-MIP "calibration" is **not trusted — it is
garbage and needs work**: Y-88 places its "MIP" at ~130 keVee vs the assumed
5.05 MeV, a ~40× disagreement, and the long run 224503 did NOT rescue it.

| # | Frame title | Figure(s) | Key numbers / point |
|---|---|---|---|
| 1 | Title | — | "Back-Plastic Absolute Calibration: Y-88 works, the triples MIP does not" |
| 2 | What a back plastic is + run timeline | `11_diagrams/concept_wall_plastic_liq.png` + timeline | at nominal HV the 699 edge is ~1 mV (below ~4.9 mV threshold) → HV RAISED to calibrate; no photopeak |
| 3 | Method A — Y-88 edges on plastics (THE trusted scale) | `21_y88/edges_run224476.png` (PSS panels, **lin–lin**), `energy_calib.png` | both 699 & 1612 keVee as two independent smeared steps; ratio 2.4 ≈ 2.31 confirms assignment; 24–39 mV/MeVee |
| 4 | Y-88 plastic result & consistency | `21_y88/energy_calib.png` | per-PMT mV/MeVee; edges equal to ±15 % ⇒ channels are gain-equalized |
| 5 | Method B — the triples MIP attempt (WAL×PSS×LIQ, 224489) | `19_triples/pss_mip_spectra.png`, `pss_mip_aligned.png` | linear-space MPV per HV step, transported to nominal via power law; 5.05 MeV assumed |
| 6 | **⚠ The triples MIP is NOT trusted — garbage, needs work** | annotate `21_y88/energy_calib.png` | Y-88 35–64 mV/MeVee vs triples 0.65–2.05 mV/MeV → triples "MIP" really at ~130 keVee, **~40× off**. Likely a mis-identified/low-quality feature, NOT a 5 MeV MIP |
| 7 | Long-run recheck already done — still fails | `21_y88/mip_224503.png` | run 224503 (`23e`) with the proper wall×plastic+LIQ triple: plastic MIP STILL swamped, no peak at predicted ~120–217 mV → **statistics was not the problem; the method needs rework** |
| 8 | What we DO trust: gain curves in the FIFO readout | `21_y88/hv_gain_curves.png`, `19_triples/fifo_ratio.png` | 224466 BNC-T vs 224489 FIFO; power law n=3.8–7.1; FIFO ×1.13–1.65 per PMT (shape unchanged) |
| 9 | Y-88 absolute anchor → mV/MeVee vs HV | `21_y88/hv_absolute_scale.png` | 699 edge 25–45 mV, 35–64 mV/MeVee at current bias (`plastic_hv_gain_absolute.json`) |
| 10 | Do the three equalizations agree? | `21_y88/equalization_compare.png` | BNC-T median / FIFO median / Y-88 edge biases agree ~±30 V (`equalization_compare.json`) |
| 11 | **Recommended plastic HV** (Y-88-edge equalization, FIFO) | table (`HV_GAIN_Y88_ANALYSIS.md` §3) | per-PMT V for 30–70 mV target at 699 edge; PSSB1 weakest |
| 12 | Caveat: response nonlinearity | `21_y88/nonlinearity_concept.png`, `nonlinearity_data.png` | measured 1612/699 ratio 1.93–2.5 vs 2.31; how to handle |
| 13 | Summary & open items | — | Y-88 is the plastic scale; **triples MIP needs rework (not statistics); missing raised-HV values for 224476–79** to transport to nominal |

### Appendix (plastic deck) — LIQ (liquid scintillator) first absolute scale
| # | Frame title | Figure(s) | Key numbers / point |
|---|---|---|---|
| A1 | LIQ: first absolute energy scale | `19_triples/liq_spectra.png` | new with the Mucciola LIQ PSA reprocessing; only LIQA/LIQD cabled |
| A2 | Y-88 699 keVee edge on all four liquids | `21_y88/edges_run*.png` (LIQ panels, **lin–lin**) | clean 699 bump 22–26 mV → 32–37 mV/MeVee, consistent ~10 % arm-to-arm (`y88_energy_calib.json` LIQ rows) |
| A3 | LIQ gain-vs-position gradient (triples) | `19_triples/liq_position_map.png` | 1.5–2× gain gradient toward the PMT; source sits mid-vessel (single position, so edge anchors mid-gradient) |
| A4 | LIQ timing / caveats | `19_triples/liq_timing.png` | wall–LIQ / plastic–LIQ dt offsets; modest stats (6–110 k hits); one source position only |

---

## Build actions before compiling
- **Regenerate** `21_y88/spectra_run*.png` and `edges_run*.png` with **linear y-axis**
  (drop `set_yscale('log')` in `21_y88_spectra.py:102/119/134` and
  `22_y88_edges.py:298`) — keeps the already-linear amplitude axis, makes it lin–lin.
- **Reframe/redo** the wall Y-88-vs-MIP comparison figure to use the wall×plastic MIP
  (from `19` triples / `23e`) rather than the 224489 top/bottom proxy in `23b`.
- Lift working frames from the existing `y88_calib_slides.tex` where they map.
