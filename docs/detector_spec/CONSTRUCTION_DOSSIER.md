# MX17 chambers: how they were designed and built (dossier for Paper I, Section 3)

Compiled 2026-11-25 for the Section 3 rewrite of Paper I v2. This file collects everything we could find about the
design and construction of the five chambers (A–E), says where each fact comes from, and marks where the sources
disagree. Section 3 of the paper (`papers/paper1_v2/sections/03_design.tex`) and its construction appendix
(`A0_construction.tex`) are condensed from it.

**This repository is public.** Prices, phone numbers and personal remarks from the sources are left out on purpose.

Status marks used below: **[S]** Stephan wrote or said it, **[D]** Dylan's account (Paper I v2 feedback note,
2026-11-25), **[G]** production Gerber files, **[Q]** CERN MPT offer 3498-A and its stack-up sheet, **[L]** our
logbooks and bench run configurations, **[M]** measured in our data, **[?]** not confirmed, **[≠]** the sources
disagree.

## 0. Sources

| source | what it holds | where |
|---|---|---|
| Stephan, "Mx det.docx" (email 2026-11-24, "Detector Paper Draft") | his first notes for Section 3: design choices, the 2D-pixel pattern, why strips, pastes, bulk process, mesh, HV test | CERN mailbox; text copy outside the repo |
| Stephan, "MMX17 V1.4.2.pptx" (14/03/2025, sent by Thomas 2025-09-04) | the original design: specification, pixel pattern, stack options A/B, HV contact, multiplexing cards | CERN mailbox |
| Stephan, "MX17 Detector part for test report.docx" (2026-02-27) | outline: concept, readout PCB, multiplex card, bulk, mechanics, µTPC mode, field degrader | CERN mailbox |
| CERN MPT offer 3498-A (Rui De Oliveira, 2025-03-27) and the Stephan–Rui–Bertrand thread (Mar–Sep 2025) | board specification (6 layers, 530 µm, ~270 000 vias), multiplexing cards, the job split CERN/Saclay | CERN mailbox (forwarded 2026-11-24) |
| stack-up 3498-A.xlsx | the layer stack of the readout board | `stackup_3498A/` (this folder) |
| Production Gerbers (2025-05-23) | pads, strips, vias, pillar artwork, resist outline | `MX17_Geant/design/gerbers/readout_pcb` |
| Feb 2026 n_TOF logbook, May 2026 logbook and status, March 2026 status talk | prototype ("Detector 0") behaviour, repairs at GDD, floating mesh, injection circuit | `~/PycharmProjects/MX17_Documentation/` |
| Bench run configurations (`run_config.json`, field `description`) | who bulked each board and when, and what was seen | `$cosmic_bench/<dir>/<run>/run_config.json` |
| Cosmic_Bench_DAQ_Control git history (Oct–Nov 2025) and `define_detectors.py` | the small RD5 test boards: plain, strip and grid resistive patterns in ESL and Saral pastes | `~/PycharmProjects/Cosmic_Bench_DAQ_Control` |
| Thomas, emails 2025-10-30 and 2025-11-26 | test-beam planning, mesh-wrinkle tests, boards received | CERN mailbox |
| Dylan's stream of consciousness | the history and the failure modes as he saw them | Claude Hub note "Paper I v2 Feedback", §3 |
| Our measurements | active area, passivated bands, currents, bow, per-chamber defects | Paper I v2 §4.5, §10, App. B, E |

Not reachable from this session: Dylan's own Google Docs (the Drive connector is Alexandra's account) and Granola
(no account). Email search on the hub is by subject only, so threads without a telling subject may be missing.

## 1. Timeline

| when | what | source |
|---|---|---|
| before 2025 | The n_TOF X17 proposal planned µRWELL detectors for the TPC. Beam tests of CERN µRWELL prototypes did not work out ("logistical issues" in the March 2026 talk). | [D], March talk |
| spring 2025 | Pierfrancesco Mastinu asks Thomas Papaevangelou (CEA Saclay) for Micromegas; Stephan Aune designs them. | [D] |
| 2025-03-14 | Design deck "MMX17 – Multiplex Micromegas for X17", v1: 400 × 400 mm², 0.78 mm pixels joined into 512 X + 512 Y strips, 30 mm drift, plain resistive layer of ~5 MΩ/□ with an HV contact on four lines around the active area, one DREAM card (512 channels) per detector through multiplexing cards. | [S] deck |
| 2025-03-19 | Stephan asks CERN MPT (Rui De Oliveira) for: design advice, Gerbers of the readout board, the insolation mask and the multiplexing boards, production of ~8 boards (two without Kapton) and the multiplexing cards. Saclay does the resistive screen printing (on the Kapton already pressed on the boards), the bulk and the integration and test. | [Q] |
| 2025-03-27 | Offer 3498-A: 8 boards, 475 × 475 mm, 400 × 400 mm² active, 6 copper layers, 530 µm total with ~68 µm copper, ~270 000 vias, Ni/Au on top for contact connection, no copper on the bottom, 8 weeks lead time. Item 2: M1/M2/M4 multiplexing cards, ~450 × 50 mm, 4 layers. | [Q] |
| 2025-05-23 | Date in the production Gerbers. | [G] |
| Jul–Sep 2025 | Bertrand Mehl lays out the M1 interface card (MEC8 connectors moved off the GFZ screws), then the M2/M4 versions; drift-field PCBs (28) quoted; Gerbers approved by Stephan 2025-08-28. | [Q] |
| Sep 2025 | Dylan joins (works in earnest from the November beam test). | [D] |
| Oct–Nov 2025 | Small RD5 test boards (128.5 × 154 mm) with the resistive pattern in plain, strip and grid form, in ESL and Saral pastes (`rd5_plein_esl`, `rd5_strip_esl`, `rd5_grid_esl`, `rd5_plein_saral`, ...), tested with Fe-55 and cosmics at Saclay (23–28 Oct) and in the DRD1 SPS test beam (8–22 Nov), next to EIC and µRWELL test detectors. No MX17 prototype was ready for this beam (Thomas, 30 Oct: "if the detector cannot be ready, we can schedule a few days test with one of the EIC detectors"). | [L], email |
| 2025-11-26 | Thomas: all readout boards received; Stephan is running systematic tests to isolate the parameters that cause wrinkles in the mesh. | email |
| Dec 2025 | Plan: chambers for a physics run in February. | [D] |
| Jan 2026 | Production starts at Saclay; February becomes a beam test with one prototype. Prototype ("mx17_1", "Detector 0") on the cosmic bench 27–28 Jan. | [D], [L] |
| 31 Jan – Mar 2026 | February n_TOF beam test with Detector 0 (30 mm drift, Al frame, resistive strips). A second board had its resistive strips disconnected from the HV contact (GDD, 4 Feb). The gamma flash saturates the DREAM; most of the test goes to it (gases, drift gaps of 6 and 30 mm, fields). Detector 0 on the bench: efficient except where resistive strips are broken (no HV) and where an M4 connector edge did not make contact; angular resolution ~2° (hit-based). | [L], March talk |
| Feb 2026 | After the first flash, the mesh is made floating instead of grounded (to try to inject charge against the flash). Not foreseen in the design: Kapton tape all around the active area, one layer under the mesh (to cut the mesh on it) and one on top after cutting; ground brought to the mesh through a circuit. | [S], [L] |
| 2026-04-24 | A board bulked with the resistive strips **horizontal** (along x). "Waves on mesh. Resist strips broken." (bench, 25 Apr, Ar/CF4). This is the detector Dylan remembers with strips "the wrong way". It is none of A–E. | [L] |
| 2026-05-05 | A board bulked with strips **vertical**, ESL paste, **no silver paste** ("Det-3" of the May test). | [L] |
| 6–18 May 2026 | May n_TOF test with two chambers: Det-3 (Al frame, 16 mm adjustable drift, 4 M1 cards, efficient) and Det-4 (30 mm drift, M2 ×2 multiplexing, only a small strip ~50 % efficient). Det-3 sparked; Stephan glued over suspect areas at GDD. Mesh-injection circuit killed by 400–800 V discharges. | [L] |
| 2026-06-12 | Board bulked by Arnaud: "giant pillars on parts of the detector" → **chamber B** (det2). Dylan: pillars too large because the mask shifted during the UV exposure. | [L], [D] |
| 2026-06-15 | Board bulked by Stephan → **chamber A** (det3). | [L] |
| 2026-06-22 | Batch of three bulked by Stephan, including board "C" → **chamber E** (det4) ("a few bubbles, the pillars underneath still there, just no caps") and board "B" → **chamber D** (det7) ("one or two bubbles ... just no caps"). | [L] |
| 2026-06-24 (?) | Board "D" → **chamber C** (det6): lamination redone several times after wrinkles in the first layer; "in the end a column of waves in the mesh and maybe a spot with no pillar caps". | [L] |
| 17–28 Jun 2026 | All five on the Saclay cosmic bench (the data of this paper). | [L] |
| Jul–Aug 2026 | A–D at n_TOF EAR2 (last beam run 224715, 10 Aug); E in the SPS H4 beam. | [L] |

Board letters in the run configurations ("board B/C/D") are **not** the chamber letters of the paper. Only chamber
letters go into the paper.

What we could not establish: which physical boards were re-used between the February, May and June builds (Dylan:
"all of our boards underwent multiple iterations of serigraphy, bulking, debulking during 2026"), and whether the May
Det-3 board is the June chamber A.

## 2. Design choices, with the reasons

1. **One readout plane with 2D pixels ("2Dp") rather than a 2D resistive readout ("2Dr").** [S] With a single tracking
   plane the readout must give X and Y from one avalanche. The usual 2Dr solution reads X and Y from two copper strip
   layers below the resistive layer, through ~75 µm of insulator (Kapton and glue), with a narrow (~100 µm) top strip
   and a wide lower strip. It has the least copper. Its drawback is the X/Y charge share when the resistive layer is
   made of strips: the copper strips perpendicular to the resistive strips get many hits (~10?) and the parallel ones
   few (~3?) [S, his question marks]. The 2Dp solution puts square copper pads on top, each joined by its own via to an
   X or a Y strip below. It needs more copper (to fill the vias; the plain copper is about the same) and one via per
   pad: 512 × 512 = 262 144 vias. Rui De Oliveira (CERN MPT) validated the fabrication. Its advantage is the same
   response in X and Y, which was first meant for a plain resistive layer. No time for R&D on 2Dr strip sizes.
   Measured: X and Y collect the same charge to < 3 % (Paper I §8.1).
2. **The pixel pattern ("2Dp-spread").** [S] In the CAST-type 2Dp ("2Dp-diag"), X and Y strips run along the
   diagonals. Each pad then has two identical neighbours, and diagonal neighbours of the same strip can be linked on
   the pad layer, which halves the vias, but the edges of a square are hard to fill. In 2Dp-spread the strips run along
   x and y, and no two adjacent pads belong to the same strip: an Xn pad has two Xn−1, two Xn+1, two Ym and two Ym+1
   neighbours. This maximises the cluster size and makes 512 vertical and 512 horizontal strips easy to route to the
   edges. In the Gerbers this is a checkerboard of X and Y pads [G].
3. **Resistive Micromegas.** Protection against discharges at a pulsed, high-intensity facility (Paper I §1). [D]
4. **Resistive strips rather than a plain resistive layer.** [S][D] The plain layer was the first choice (X/Y
   symmetry). The aim is at least 1 MΩ/□ between the HV contact and any point of the resistive layer under the mesh.
   With the HV contact all around the active area, the plain layer came too close to the HV contact where the mesh
   first sees it: even with some passivation the resistance between HV and mesh was too low, so current leaked and
   flowed through hot spots. A plain pattern with ~15 mm strips all around for the HV connection was planned, but the
   screen for it came from the supplier with the wrong pattern and there was no time to re-order. **[S: "explain ???"
   — the reason is still to be written.]** A striped screen made for another project (TomoMu) was used instead: 100 µm
   wide strips on a 488 µm pitch [S, "tbc"] over 50 × 50 cm², taped down to a 40 × 40 cm² printable area. CLAS12 buries
   its resistive strips over 15 mm to reach the same value, ATLAS NSW over up to ~100 mm [S]. The strip ends are
   covered by photoresist during the bulk process over 20 mm on each side [S]. Dylan: strips were already used in the
   ATLAS NSW and CLAS12 and in muon-tomography projects, so they were better understood [D, "fact check"]. Measured:
   the passivated bands are 17.9–20.4 mm over the five chambers (18.0 and 18.7 mm on A) [M].
5. **Strip direction.** [S][D] The resistive strips are passivated in one direction only (top and bottom, the y ends),
   to keep the horizontal (x) direction fully efficient: the chambers sit side by side at n_TOF. In x there is no HV
   bus on the first and last resistive strip, so the strips run the full 40 cm with nothing to passivate there.
6. **Number of DREAM channels** [S, "fifth choice"]. The design read 512 channels per chamber through ×2 or ×4
   multiplexing cards (M2, M4) and a full-readout card (M1). The data of this paper were taken with full readout
   (1024 channels per chamber) [?: check M1 on all five on the bench].
7. **The overall constraint** [S][D]: four (goal five) chambers in about one year. No R&D or long tests were possible,
   so every choice had to be made quickly from known techniques.

## 3. The chamber, layer by layer (target side first)

| layer | value | status | source / note |
|---|---|---|---|
| entrance window | Mylar 60 µm; aluminised? (one side, ~100 nm, assumed) | CAD; Al [?] | spec.py |
| gas gap | 5.0 mm, flushed by the chamber gas (assumed) | CAD | spec.py |
| drift cathode | Kapton 50 µm + Cu; "double copper drift" in Stephan's outline | Cu thickness and sides [?] | spec.py; [S] test-report outline says "Double copper drift + mylar window" |
| drift gap | 30.000 mm design (Al frame is the spacer); 28.2 mm measured on A | CAD, [M] | §5.1 |
| field cage | 4 PCBs 408.4 × 1.62 mm; divider 3 × 1.30 GΩ; chamber B no ring chain | CAD, [M] | spec.py |
| mesh | ASADA 45/19 (45 µm opening, 19 µm wire), AISI 316 stainless steel, **calendered to 20 µm**, 1.2 × 20 m rolls | [S] | replaces the P2 placeholder (48 µm opening, 38 µm, uncalendered) |
| mesh tension | 12 N/cm (nominal ~8 N/cm gave waves; 2 N/cm was fine but hard on the CLAS12 stretching bench) | [S] | |
| amplification gap | 150 µm: two Dynamask layers (2 × 75 µm) laminated, the mesh on them, one more Dynamask layer (75 µm) on top | [S], confirmed by the pillar Gerber | |
| pillars | [≠] Stephan: Ø 500 µm on a 5 mm pitch. Bulk Gerber: Ø 0.6 mm on a 4.68 mm grid (85 × 85). | [S] vs [G] | ask which is as built |
| resistive strips | [≠] Stephan: TomoMu screen, 100 µm wide on 488 µm (tbc). Our model and Table 1: 550 µm on 800 µm (Dylan from memory). | [S] vs [D] | ask; affects Table 1 and the Fig. 3 caption |
| resistive paste | ESL ("hard") on A–E; Saral ("soft") tried first | [S][D] | §4 below |
| strip thickness, sheet resistance | unknown; 1.4–2.6 MΩ/□ inferred from the charge spread; Stephan will send the bulk-lab resistance values | [?] | |
| insulator under the strips | [≠] stack-up: 25 µm UBE polyimide + 25 µm EMC prepreg (50 µm). Our response model: 50 µm Kapton + 25 µm glue (70.5 µm effective). | [Q] vs model | asked of Bertrand 2026-11-24 |
| L3 top copper | 9 + 17 = 26 µm, Ni/Au; connectors and the resistive-layer contact; **no copper inside the active area** | [Q][G] | |
| L4 pads | 17 + 17 µm; 0.68 mm squares on 0.78 mm (512 × 512) | [Q][G] | |
| dielectric | 100 µm prepreg | [Q] | |
| L5 Y strips | 17 µm; Ø 0.50 mm lands + 98 µm bus | [Q][G] | |
| dielectric | 100 µm EMC prepreg | [Q] | |
| L6 X strips | 17 µm | [Q][G] | |
| dielectric | 100 µm prepreg | [Q] | |
| L7 | 17 + 17 µm, called "GND" in the sheet; in the Gerber only 0.65 mm via lands under the pads | [Q][G] [≠] | asked of Bertrand |
| dielectric | 25 µm EMC prepreg + 25 µm UBE polyimide | [Q] | |
| L8 bottom | no copper | [Q] | |
| **board total** | [≠] **528 µm** by the stack-up (offer: 530 µm). Mechanical CAD and Geant4: 1.70 mm. | [Q] vs CAD | the largest open item in the material budget; asked of Bertrand |
| vias | Ø 0.25 mm, L3→L4 and L4→L7, one per pad, a land on every layer crossed; filled? [?] | [G] | |
| behind the board | Rohacell 5 mm (grade?); back Mylar 25 µm (assumed); Al plate 8 mm with a 402 × 402 mm aperture (nothing behind the active area) | CAD / [?] | "Readout on rohacell" [S] |
| front end | M1 cards on the board edge (pogo pins onto the connector pads, MEC8 outputs), DREAM | CAD | |

The 1.70 mm against 0.53 mm question matters only behind the measurement (backscatter), not for the material in front
of the drift volume.

## 4. Production at Saclay, and what went wrong

### Process (Stephan's list, with the problems found)

1. **Resistive screen printing** on the Kapton already pressed onto the board.
   - Saral ("RS soft paste"): did not hold at the contact with the copper HV pads. The printed strip broke at the step
     down from the copper HV contact to the board [D]. Silver glue at the contact was needed [S].
   - ESL ("hard paste"): held without silver glue. Brittle, and very hard to remove if the board has to be re-bulked
     (Saral comes off easily) [S][D]. ESL could be sanded (800 paper, both directions) to remove resistive spikes,
     which spark, and to help the pillars attach [S].
   - Repair: paint the strip/copper interfaces with silver paste and cover with Kapton tape [D]. It worked only some
     of the time, and silver is known to migrate in the field and grow conductive "pillars" towards the mesh until it
     shorts, so its long-term behaviour is doubtful [D].
2. **Bulk**: lamination of 2 × 75 µm Dynamask, the stretched mesh, 1 × 75 µm Dynamask; UV exposure through the
   pillar mask (maximised, 20 min, so that the first layer on the Kapton polymerises under the mesh); development
   (chemical spray from below, rinse in de-ionised water); curing at 145 °C in 10 °C steps [S].
   - **Waves in the mesh** appear when the bulk is heated to cure the photoresist. Several tests on the tension: nominal
     ~8 N/cm gives waves; ~2 N/cm works but is hard to do on the CLAS12 stretching bench; 12 N/cm works but puts more
     stress on the pillars (some lost on some detectors). All five chambers were made at 12 N/cm with the stepwise cure.
   - **Photoresist waves** at lamination can make pillars fail. Before the mesh is added, the process can be stopped
     and restarted; after, a photoresist wave only loses the top pillar caps, and the gap is kept by the pillars under
     the mesh [S]. (The June bench notes "bubbles ... pillars underneath still there, just no caps" are this.)
   - **Pillars**: smaller is better for dead area, but they attach less well. Ø 500 µm on a 5 mm pitch [S].
     Development time was kept short to avoid losing pillars. The drawback: the last detectors were not completely
     developed, leaving photoresist residue under the mesh in bands (the spray develops in bands), with ~50 %
     efficiency there [S]. Dylan: over-development can also weaken and break pillars [D].
   - **Mesh detachment** [D][S]: patches of ~1–10 cm diameter where the mesh is not held, seen by eye as a "wavy"
     mesh; caused by the mesh tension against the board with some pillars failing to hold the two. Not fully
     understood; still present on the last chambers.
3. **Floating mesh** (after February) [S]: Kapton tape all around the active area, under and over the cut mesh; ground
   brought to the mesh through a circuit (failed on one detector).
4. **HV test in the clean room** (class 100 000) [S]: goal 900 V with I < 1 µA; thermal camera to find hot spots;
   washing to remove photoresist residue.
5. **Integration** in the clean room (dust).

### What each chamber came out as

| chamber | bulked | what was seen at production | what we measure |
|---|---|---|---|
| A (det3) | 15 June, Stephan | — | uniform; 92.9 % / ~95 % efficient, ~1° |
| B (det2) | 12 June, Arnaud | "giant pillars on parts of the detector" (mask shifted in the UV exposure [D]) | uniform; 91.3 % |
| C (det6) | ~24 June, Stephan | lamination redone after wrinkles; a column of waves in the mesh, maybe a spot without pillar caps | discharges over the whole area |
| D (det7) | 22 June, Stephan (batch of 3) | one or two bubbles, pillar caps missing there | discharges; poorly mounted X readout card |
| E (det4) | 22 June, Stephan (batch of 3) | a few bubbles, pillar caps missing there | broken resistive strips: 11 non-amplifying bands, 62 % of the area (App. E) |

Dylan's summary [D]: one ideal chamber (A), one that works well (B), three with problems of varying size, all three
probably with some broken resistive strips (obvious on E). The process works, as A shows, but the Saclay bulk lab has
not yet mastered it: each step has some probability of failing, and the yield was about one good chamber in five.

## 5. Disagreements and open questions (to Stephan unless noted)

1. Resistive strip width and pitch: 100 µm on 488 µm (Stephan, tbc) or 550 µm on 800 µm (our tables, from memory)?
2. Pillars: Ø 500 µm on 5 mm (Stephan) or Ø 0.6 mm on 4.68 mm (bulk Gerber)?
3. Readout board: 0.53 mm (stack-up) or 1.70 mm (CAD)? Bertrand asked 2026-11-24.
4. Insulator above the pads: 25 + 25 µm (stack-up) or 50 + 25 µm (our model)?
5. Drift cathode: copper on one side or both ("double copper drift"), and how thick?
6. Why the plain-pattern screen was wrong ("explain ???" in his note).
7. Resistance values of the strips (Stephan: "next week, from the bulk-lab data").
8. Board history: which boards were re-printed or re-bulked, and how many times, before June.
9. Field cage: ring strips, pitch; chamber B's missing chain.
10. Rohacell grade; back foil.
11. The Nov 2025 RD5 tests: is there a written result (plain vs strip, ESL vs Saral)? Only the run configurations are in
    our repositories.
12. [Dylan] Can the µRWELL history be cited (a note or talk), or should the paper only say the tracker was redesigned?
