# Literature survey: depth mixing of charge in TPC time signals (2026-11-08)

**Question (Dylan):** why does nobody else seem to have the charge-spread / time-signal convolution problem?
**Answer:** strip micro-TPCs do, under other names. Long TPCs cannot show it (Λ is 7-110× smaller) and most directional
detectors have no reference that would reveal it. What survives as ours: the per-track slope ratio against a telescope, the
decomposition into diffusion / footprint / strip ends / RC, and a per-event forward fit of a depth-resolved charge profile.

| File | What |
|---|---|
| `../LITERATURE_TPC_SURVEY.md` | **The report**: table per family, Λ definition, four hypotheses, safe Paper II wording |
| `../LITERATURE_MIXING.md` | The earlier (11-06) µTPC-only review this extends |
| `strip_microtpcs.md`, `pad_tpcs.md`, `lartpc.md`, `short_track_tpcs.md` | Per-family notes with verbatim quotes, page numbers, derived numbers marked, gaps listed |
| `survey_data.py` | The survey as data: one record per experiment (detector, use, what they did and concluded, relevance, quotes, gaps) |
| `make_deck.py` | Slide note generator; `python make_deck.py`, then publish (command in its docstring) |
| `note/mx17-tpc-literature.html` | Built deck. Live: https://dylan-neff.web.cern.ch/notes/mx17-tpc-literature.html |
| `pad_tpc_toy.py` | Geometry-only scaling toy behind the pad-TPC argument: scaling only, under-predicts MX17 |

## Method and how far to trust it
Four parallel strands on 2026-11-08, primary sources only for key claims. Quotes were string-matched against `pdftotext` output
of the PDFs; failures and search-snippet-only items are flagged in the notes. **(derived)** in the notes means our arithmetic; no
source states Λ or the lever-arm argument. Source PDFs and extracted text (986 MB): `/eos/user/d/dneff/litsurvey_sources_2026-11-08.tgz`.

## Corrections made along the way
- Bench gas behind 1.19 / 1.37 is Ar/iC₄H₁₀ 95/5 + ~1 % water, not Ar/CF₄/iso.
- MX17 Λ₀.₃ = 0.076 on one definition (0.04-0.08, 0.07, 9.5 % in the notes are the same formula with other inputs).
- The strip notes pair 1.19/1.37 with tan θ 0.3-0.6; the sample is |tan θ| ≲ 0.3, median 0.11. Binned by angle the tension goes away.

## Not read (all absences are negative results of a limited search)
Lösel PhD, Klitzner PhD, Alexopoulos 2019, Iodice, ALEPH and PEP-4 NIM papers, Blum-Rolandi-Riegler, Bradt, Mauss, Giovinazzo
2020, later D3, SpecMAT, DMTPC, LAr theses and DUNE/SBN notes, internal ATLAS and BESIII notes. ATLAS bias figures exist as images only.

## To change in Paper II text
“Delayed copies on resistive strips” (`../../OUTLINE_PLAN_2026-11-05.md`), “whatever the estimator” (charge-weighted untested),
“~4° too steep” (near normal only; sample median 2.5° X, 3.1° Y), “20-30 %” (16 % X, 27 % Y).
