# infn_in2p3_26 — poster for the INFN-IN2P3 detector R&D conference

**INFN-IN2P3 bi-national conference on detector R&D**, 2nd edition, 2–4 Dec 2026,
LNF Frascati (ed. 36, B. Touschek). <https://agenda.infn.it/event/52777/>
Chair: Alberto Quaranta (CSN5 president). Dylan is registered; the poster
contribution was requested on 30 Oct and followed up on 17 Nov (Quaranta thread,
CERN mail). **Not yet confirmed as added to the programme.** The submitted abstract
is `papers/infn_in2p3_2026_abstract/` (session: gas detectors). That folder notes an
eligibility point: CEA Irfu is not IN2P3, so list an INFN/IN2P3 co-author (n_TOF)
or the INFN LNL affiliation.

## What the conference is (read from Indico, 2026-11-17)

- Aim: bring the INFN and IN2P3 detector R&D communities together, "identify
  synergies, and promote collaborations". First edition Paris, Nov 2025.
- Five half-day sessions: photosensors, **gas detectors** (Thu 3 Dec 14:40–16:20),
  solid-state, cryogenic & quantum sensors, electronics & DAQ / platforms.
  The gas session's talk list was not public yet.
- **Poster sessions:** Wed 2 Dec 16:20–17:20 (with the coffee break) and
  Thu 3 Dec 18:10 (with the cocktail). **A0 portrait recommended. The best posters
  are awarded by the scientific committee.**
- ⚠ Dylan's INFN oral exam (call 29230) is **Wed 2 Dec 15:30**, which overlaps
  poster session I. Plan to be at the poster for session II; ask whether the poster
  can be hung on the morning of the 2nd regardless.

## The audience, and the angle

Detector builders, not X17 people. LNF is the home of the µ-RWELL, and a lot of
the room runs resistive MPGDs (NSW Micromegas, µ-RWELL, ERAM). Many of them use
µTPC mode. So the poster is **about the detector and the reconstruction**, with
X17 as the reason and n_TOF as the proof.

**The hook: "your µTPC hit times are lying, even on a metallic readout."** A single
time per strip mixes charge from different drift depths. Drift diffusion, together
with the prompt induction footprint, puts deep charge onto neighbours whose own
charge arrived earlier, and a leading-edge or peak time is pulled toward the earliest
charge. The cathode end is pulled in more than the mesh end, so the time ladder is
compressed and tracks read ~2–4° too steep. On our chambers the X view, which has no
resistive spreading, is compressed ×1.17–1.20. A **metallic readout in the same gas
would show almost all of that** (model with the resistive term off: ×1.17–1.24). The
resistive layer then roughly doubles the excess in the view across the strips (Y
×1.27–1.53). Every estimator tried shows it (leading edge, CFD, peak, matched
filter), so no threshold tuning fixes it. The room runs µTPC on NSW Micromegas and
µ-RWELL, so the question to put to them is whether they have corrected for this.
Sources: `papers/paper2/hits_daq/P0_XBIAS_2026-11-05.md` (the table to plot),
`RECONSTRUCTION_BASIS.md`. **Do not use the retired "29 % delayed copies at 47 ns"
explanation** (repo `CLAUDE.md`). The submitted abstract still says "charge spreads
… with an RC delay … about 4° too steep"; the poster should state the corrected
mechanism.

The fix is a forward model of the whole strip × sample picture (`wft/`): it fits the
mixing instead of reading one time per strip. The proof is the n_TOF beam, where the
reconstructed tracks image the ³He capsule.

## Poster layout (A0 portrait, 841 × 1189 mm)

| # | Panel | Content | Figure source |
|---|---|---|---|
| 0 | Header | Title, authors (CEA Irfu / n_TOF; INFN LNL incoming), QR to the online version | — |
| 1 | Why build it | X17 → ³He(n,e⁺e⁻), the opening angle, why low mass means a TPC. Kept small. | `mpgd26/make_x17.py --layout beats` |
| 2 | The chamber | 40 × 40 cm², 30 mm drift, bulk resistive strips, 0.78 mm pads coupled to two strip views, DREAM readout | `mpgd26/make_chamber.py` (exploded view), `make_microtpc.py` |
| 3 | **One time per strip compresses the track** (the large centre panel) | Cartoon of the mechanism (diffusion pulls deep charge early); the P0 bar chart, data / full model / metallic / point for X and Y on four chambers; cluster vs. reference line against depth (hits walk away, waveforms do not) | new plot from `papers/paper2/hits_daq/results/p0_*.json`; `RECONSTRUCTION_BASIS.md` depth table |
| 4 | Fit the waveforms, not the hits | Model diagram; angle resolution and efficiency from Paper I | `make_share.py` (model diagram); numbers and figures from `papers/paper1` (`figures.toml`) |
| 5 | In the n_TOF beam | EAR2 setup; capsule image from track pointing; the γ-flash: chamber back in 2 µs, DREAM in 5 ms | `make_ntof.py`, `make_run145_pointing.py`, `make_flash_slides.py` |
| 6 | Questions for you | Short list aimed at the room: does your µTPC see the same compression? c2 < c1 kernel trap; who has a flash-tolerant readout? | text |

Small extras: a QR code to the 3D pair event display
(<https://dylan-neff.web.cern.ch/files/x17_pair_24576.gif>), and possibly a QR to an
interactive "slide τ and watch the ladder compress" toy on the site.

## Build plan

- Generated by `make_poster.py` → `poster.pptx` (A0 portrait) + `poster.pdf`.
  pptx so Dylan can nudge things by hand; generated so numbers and figures refresh
  together.
- Figures are re-rendered from `mpgd26/` builders at poster resolution (≥ 300 dpi at
  their printed size). Most `mpgd26/figures/` outputs are not on this Windows
  machine; several builders need the desktop Ubuntu data (see
  `mpgd26/slides/HANDOFF_offline_rebuild.md`). The fallback is crops from
  `ntof_athens_26/slides/neff_x17_ntof_athens_26.pdf`.

## Numbers to reconcile before printing

- **Paper I (`papers/paper1/main.tex`) is the source of truth for the numbers.** It gives
  intrinsic resolution 176 µm (0.30 × pitch, SPS beam) and cosmic angular resolution
  0.90–2.41° (chamber A 0.99/0.90° in X/Y). The abstract (`papers/infn_in2p3_2026_abstract/`)
  says 1.2°, 93 % and 180 µm. Paper I is shared with Stephan on Overleaf: read it, don't
  edit it. State the conditions for each number on the poster (cosmics or beam, which
  detector, which bundle).
- Use the `calib_bundle_r06` (c2 < c1) numbers only. Anything older is retired.
- Credit prior art (Dixit 2004, Galan 2012, the T2K ERAM fit by Attié 2023;
  `wft/REFERENCES.md`). This is a Saclay-heavy audience.

## Timeline

| when | what |
|---|---|
| by 21 Nov | Draft v1 (all panels, placeholder figures) |
| by 25 Nov | Figures at print resolution; numbers reconciled |
| by 27 Nov | Final PDF; print (check LNF or CERN printshop lead time) |
| 2–3 Dec | Poster sessions |
