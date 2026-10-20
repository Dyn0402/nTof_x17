# MPGD2026 proceedings (JINST)

Dylan Neff, "Searching for X17 at n_TOF with large-area Micromegas detectors in µTPC mode",
oral talk, MPGD2026, Prague, 3 Sept 2026. **Deadline: 17 January 2027**, submitted via jinst.sissa.it
(Submit → "submit a proceedings contribution" → MPGD2026).

- Limit: **5 pages** for an oral talk, not counting authors/affiliations, acknowledgements or references.
  The current draft's body runs over pages 2–6.
- `main.tex` follows the official JINST template; `jinstpub.sty` is SISSA's own copy (same as the Overleaf template).
- `figures/` are copies of the talk's assets (`mpgd26/slides/assets/img/`). Their provenance is in
  `mpgd26/slides/NOTES.md`.
- `\askDylan{…}` (cyan) and `% TODO Dylan` mark open items. Remove all of them before submitting.
- `draft_preview.pdf` (gitignored, local only) was compiled locally with Times (`mathptmx`) in place of `newtx`, because this machine's TeX Live 2023
  cannot load the current newtx. Overleaf compiles the real thing, so expect small differences in line breaks.

Overleaf project (Dylan's copy of the JINST template): https://www.overleaf.com/project/6ad72207b122b98dea142d04 ,
linked 2026-10-20 with `papers/overleaf.sh` (remote `overleaf-mpgd26_proceedings`). Day to day:

    papers/overleaf.sh pull mpgd26_proceedings   # bring Overleaf edits in (commit your own first)
    papers/overleaf.sh push mpgd26_proceedings   # send repo commits to Overleaf

The template's own files (main.tex, biblio.bib) were replaced at the project's head and stay in its history;
its jinstpub.sty, JHEP.bst and line numbers were adopted. Set `main.tex` as the main document in Overleaf if it isn't.
