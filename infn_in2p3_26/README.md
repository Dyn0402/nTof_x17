# infn_in2p3_26 — poster for the INFN-IN2P3 detector R&D conference

**INFN-IN2P3 bi-national conference on detector R&D**, 2nd edition, 2–4 Dec 2026,
LNF Frascati (ed. 36, B. Touschek). <https://agenda.infn.it/event/52777/>
Chair: Alberto Quaranta (CSN5 president). Dylan is registered; the poster
contribution was requested on 30 Oct and followed up on 17 Nov (Quaranta thread,
CERN mail). **Not yet confirmed as added to the programme.** The submitted abstract
is `papers/infn_in2p3_2026_abstract/` (session: gas detectors). That folder notes an
eligibility point: CEA Irfu is not IN2P3, so list an INFN/IN2P3 co-author (n_TOF)
or the INFN LNL affiliation.

## Mock-up v0 (2026-11-18): detector-first, no Paper II/III findings

Dylan, 11-18: too early to show the Paper II/III findings (the hit-time compression) before they are published. So the
"your hit times are lying" hook and panel 3 of the layout below are **dropped**. The poster is now the detector for a
general X17 search (Paper I material), with the forward fit shown as a method (data against model, no bias numbers) and
n_TOF as the first use. Top left is a 3D version of Paper I Fig. 1(a).

- `make_hero.py` -> `assets/hero_x17_box.png`: pyvista render, four TPCs boxed around a target, one pair.
- `make_poster.py --pdf` -> `poster.html` (true A0, sizes in mm), `poster.pdf`, `poster_preview.png`. HTML rather than
  pptx for the mock-up; the `ASSETS` dict in the script says where each figure comes from.
- v1 (11-18, after Dylan's reply): navy header; panel 6 gains the 3D pair event display (run_145 trigger 24576, marked
  PRELIMINARY) beside the capsule track fan; panel 7 is now "the ATOMKI measurement on a proton beam" with a second
  render (`make_hero.py --scene proton`: beam pipe, foil target, pair at 140 deg, a cosmic that misses the target).
  Its claims come from `~/PycharmProjects/x17_facility_search/lnl/README.md` and `ntof_italy_26/figures/lnl_numbers.json`:
  ~40 % against ~2.5 % geometric acceptance, 17 d (as built) and 1 d (big plastics) to 3 sigma at 1 uA. Check the
  wording with Dylan before printing, in particular the cosmic-bump remark and the comparison with ATOMKI.
- v2 (11-18, Dylan's second round): the proton render (redundant with panel 1) and the 3D pair event display (not
  revealing, and the cherry-picked pair does not come from the capsule) are **out**. Panel 6 is the rendered n_TOF station
  (`mpgd26/make_ntof.py --only full --view hero --no-labels --size 2400 2040`, output gitignored there, copied to
  `assets/ntof_station.png`) beside the full capsule-pointing figure. Panel 7 is text plus an acceptance bar. Panel 1's
  labels and leader lines are drawn by `make_hero.py`. Panel 4 is `make_waterfall.py`: event 7884 as per-strip
  waveforms in 3D with the model at the fit (method only, no hits comparison).
- v3 (11-18, after Dylan: "the format is not ideal"): redesigned from a study of 25 flash-talk posters (SQM26, QM25,
  QM23), written up in `POSTER_LESSONS.md`. Full-width hero band (box render + take-home sentence), two columns read
  top to bottom (why / chamber / waveform fit | performance / n_TOF / summary), assertion headings, a Summary and
  next-step box. `make_poster_figs.py` redraws the air-scattering and angle-resolution plots at print size (SVG,
  21-28 pt) from the paper's json/csv; the exploded chamber is cropped (`chamber_bare.png`) and its layer labels are
  HTML. The readout-board render and the two-track figure are out (two-track survives as a tile).
- Numbers on it are from the `paper1_v2` draft abstract and are not reconciled. Authors, logos, QR codes and panel 7
  (LNL / upgrades) are placeholders. The submitted abstract promises the "4 deg too steep" result, which the poster now
  leaves out.

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
