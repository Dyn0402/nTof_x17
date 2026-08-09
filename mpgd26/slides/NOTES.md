# mpgd26/slides — status and open items

MPGD2026 talk sketch, built as a single self-contained `index.html` (no build
step, no external network dependency — open it directly in a browser).

**View it:** open `index.html` in any browser, or `index.html#9` to jump to
slide 9. Arrow keys / Space / click the left-right edges to navigate.

**PDF backup:** `./make_pdf.sh` regenerates `mpgd26_talk_draft.pdf`. Don't use
the browser's own File → Print → Save as PDF — Chrome's headless print
pipeline blanks whichever slide is first whenever the output has more than one
page (confirmed positional, not content-specific, by swapping slide order).
`make_pdf.sh` sidesteps it by printing each slide to its own single-page PDF
and merging with `pdfunite`.

Current: **34 slides** (title, outline, motivation, detectors, n_TOF setup,
**Status — 11 slides, drafted 2026-08-09**, summary, backup incl. 3 new status
backups).

**The Status section is deliberately over-built.** It is written as a menu to
cut from, not as a running order — see [`STATUS_PLAN.md`](STATUS_PLAN.md) for
what each slide is for and which ones the plan considers load-bearing (D1, D4,
D6, D8, D9 if time is tight). One slide, D9, carries a `flag` because its
framing needs a decision with the collaboration before it is shown.

## Images are intentionally not tracked in git

The repo's top-level `.gitignore` blanket-excludes `*.png`/`*.gif` (see the
comment there — 250 MB was purged from history on 2026-08-04, policy is
"regenerate from the analysis scripts, don't commit the bulk output"). That
rule silently catches every image in `assets/img/` too, including several
that have **no one-command regeneration path** — unlike `mpgd26/figures/`,
this folder mixes mpgd26-native renders with curated external material. If
you clone this repo fresh, `assets/img/` will be empty. Regenerate it with
the steps below, or ask about carving out a `.gitignore` exception for
`mpgd26/slides/assets/` if that's preferable to re-deriving everything.

| File | Source / how to regenerate |
|---|---|
| `x17_signature.png`, `x17_story_capsule.png` | `mpgd26/make_x17.py` (see `mpgd26/README.md`) |
| `chamber_exploded.png` | `mpgd26/make_chamber.py` |
| `microtpc.png` | `mpgd26/make_microtpc.py` |
| `build_bench.gif` | `mpgd26/make_anim.py --only build_bench` |
| `setup_1_mm.png`, `setup_2_sipm.png`, `setup_3_plastic.png`, `setup_4_full.png`, `setup_topdown.png` | copied from `~/CLionProjects/MX17_Full_Geant/scripts/mx17_buildup_clean_*.png` and `mx17_mm_layout_topdown.png` — generating script in that repo not investigated |
| `mx17_board_peel.png` | copied from `~/CLionProjects/MX17_Geant/design/figures/mx17_board_peel.png`, made by `~/CLionProjects/MX17_Geant/scripts/model/plot_mx17_model.py` |
| `charge_sharing_schematic.png`, `unsharing_depth_bias.png`, `event_display_3d.png`, `angular_resolution.png`, `spatial_residuals.png`, `time_resolution.png`, `efficiency_breakdown.png` | `pdftocairo -singlefile -png -r 300` from the matching PDF in `mx_june_cosmic_qa/engineer_package/figures/` (e.g. `21-det3A-efficiency-breakdown.pdf` → `efficiency_breakdown.png`) — those source PDFs **are** tracked in git |
| `ntof_facility_schematic.png` | cropped from Fig. 1 of G. Tagliente et al. (n_TOF Collaboration), *EPJ Web Conf.* 292, 12002 (2024), page 4 of `https://cds.cern.ch/record/2939795/files/fulltext.pdf` |
| `ear1_ear2_photo.jpg` | CERN Document Server, `https://cds.cern.ch/record/2148416/files/n-TOF-EAR1-EAR2.jpg` (OPEN-PHO-EXP-2016-006, © CERN) — this one **is** tracked (not a `.png`) |
| `atomki_angular_correlations.png` | extracted (`pdfimages`) from `~/Downloads/Neff n_TOF Analysis Meeting X17 Update 3_24.pdf`, page 4; original data: Krasznahorkay et al., *PRL* 116, 042501 (2016) / *PRC* 104, 044003 (2021) / *PRC* 106, L061601 (2022) |
| `atomki_spectrometer_schematic.png` | cropped from Fig. 4 of J. Gulyás et al., arXiv:1504.00489 (*NIM A* 808, 21 (2016)), page 5 |
| `status_*.png` (6 files) | `mpgd26/make_status_plots.py` — **one command, and it reads only committed reductions plus a small local mirror**; see that file's docstring for the four inputs and `~/.cache/mpgd26_status/` for where two of them are staged |
| `target_pointing_fans.png` | copied from `…/analysis/wft/run_79/stat090_0000/mx17_A/wall_segment_tour/wall_segment_tour_all.png`, made by `ntof_tracking/run79_wall_segment_gif.py` |
| `target_pointing_slope.png` | copied from the same analysis dir, `figures/01_target_pointing_A.png` (not currently used on any slide — kept as the alternative to the fans figure) |

## Open items, by slide

**Title** — `[Your name]`, `[venue, date]` are placeholders.

**Motivation / ATOMKI** (comment above slide 3): condensed to 2 main-flow
slides + backup already done. Still open — beam orientation at ATOMKI
unconfirmed (deliberately left off every slide); PyVista remake of the
5-telescope schematic in house style not started; highlighting the
³H(p,e⁺e⁻)⁴He channel specifically as the bridge into the EAR2 slide not
done (evidence slide still gives ⁸Be/⁴He/¹²C equal weight).

**n_TOF / EAR2 facility** (comment above slide 5): drop EAR1 from the
discussion entirely; replace the photo panel with a PyVista scene of the
EAR2 vertical beamline (target → upward neutrons → capsule → the structure
at the top of the real photo, *tentatively* the beam dump, unverified);
fold Fig. 1 into that same scene once it exists.

**Chamber design** (comment above slide 7): `mx17_board_peel.png` needs a
version zoomed into one region so the per-layer patterns (resist strip
pitch, pad grid, X/Y strip pitch) are actually legible on a projected
slide — regenerate from `plot_mx17_model.py` rather than crop the PNG.

**Cosmic bench** slide: flagged in-slide as pending the running-order
decision with Alexandra's P2 talk.

**Efficiency** (comment above the efficiency slide): move the 5-chamber
fleet comparison to backup, keep this slide on det3 alone; pair the
breakdown with the residual distribution (the log-scale tail panels in
`spatial_residuals.png` explain the "off-track" slice); **every number on
this slide needs refreshing from a new analysis before the conference** —
same source figure already had one stale-annotation bug caught and not yet
fixed at the source.

**Resolution** (comment above the resolution slide): the "0.6–0.7 mm"
stat is against the M3 reference telescope, which itself is only
~500 µm — that's close to the floor of what this method can measure, not
a clean DUT number. Needs an explicit caveat on the slide, or a
deconvolved number, before this goes in front of an audience. Possibly
superseded by a det4 measurement at the SPS test beam.

**Status** section — **built 2026-08-09, 11 main-flow slides + 3 backups.**
Everything on them is sourced; `STATUS_PLAN.md` is the map. What is still open:

- **The framing of slide D9** (the physics reach) needs a collaboration
  decision. Three options are written out in `STATUS_PLAN.md` §5; the deck
  currently ships option (b) — state the branching ratio and the Al capture as
  facts about the reaction and the capsule, put the rate table in backup.
- **The charge measurement's one systematic is open** — it assumes the CAEN
  imon readback preserves the time-average of a short current burst. If it does
  not, every charge number is a lower bound. Closing it is item 4 of
  `ntof_july_analysis/flash_charge/HANDOFF_FLASH_CHARGE_2026-08-09.md`, and the
  cheapest route is a spec lookup, not an experiment.
- **The target-imaging slide is on PRELIMINARY reconstruction** — one arm, one
  sub-run, 3 of 13 file tags of run_79, on a transferred bench calibration whose
  own document says "nothing here is quotable". Either do the in-situ
  calibration or keep the caveat visible. None of the ~1.7 TB of production
  statistics has been reconstructed yet; a four-arm back-projection over even a
  slice of it is what would turn "the fans converge" into an actual image.
- **It is too long.** Cut to taste — that is what it was built for.

**Target slide** (n+³He physics): no "ideal e⁺e⁻ energy spectrum" plot was
found in the repos for the ideal-case kinematics Dylan originally wanted to
show — either locate one or generate it fresh.
