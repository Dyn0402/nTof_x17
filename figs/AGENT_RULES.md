# Rules for every agent working on Paper I v2 figures (2026-11-10)

Context: `papers/paper1` is the current draft of a JINST detector paper on compact resistive-strip Micromegas TPCs
(cosmic-bench data on five chambers, SPS beam data on one). It is being reorganised into `papers/paper1_v2`.
Several agents work in this checkout at the same time. The audit that motivates the rework is
`papers/paper1_audit/report.html` (text extract: run `python3 -c "import re,html;s=open('papers/paper1_audit/report.html').read();print(html.unescape(re.sub('<[^>]+>',' ',re.sub('<style.*?</style>','',s,flags=re.S))))"`).

## Hard rules
1. **New files only.** Write scripts to `papers/paper1_v2/figs/make_<name>.py` and outputs to
   `papers/paper1_v2/figs/out/<name>.pdf` (+ `.png`, + a small `.json` of the numbers you quote). Do NOT edit, move or
   delete any existing tracked file anywhere in the repo. If you need a change in an existing script, import from it, or
   copy it into `papers/paper1_v2/figs/` and change the copy.
2. **No git state changes** (no add, commit, checkout, stash, pull, reset). No `ssh` to any host (lxplus would send
   the user a 2FA prompt). No condor submission.
3. **The dataset is final.** No new runs or measurements exist or can be suggested. Existing data, simulation and
   literature only.
4. **Never take a position, angle or depth from `combined_hits` times.** Use the waveform-reconstruction tables
   (`wft`, "pass 1", `Analysis_pass1`, `Analysis_pass1_twotrack`). See `RECONSTRUCTION_BASIS.md`.
5. **Compute budget:** a shared 8-core, 15 GB laptop. At most 2 processes, under 2 GB RAM, nothing longer than
   ~20 minutes. Python is `.venv/bin/python` from the repo root.
6. **Data:** local cache `~/work_data` (many producers expect `WORK_DATA=~/work_data`). EOS is readable with xrdcp
   (the Kerberos ticket is valid): `root://eosexperiment.cern.ch//eos/experiment/ntof/data/x17/...`. Fetch only what
   you need (under 1 GB) into `~/work_data/claude-jobs/v2_<yourname>/`.

## Figure style
- `papers/paper_style.py`: `ps.use()` before plotting, `ps.det('det3')` for chamber colours, `ps.save(fig, path)`
  (writes pdf+png, rewrites internal names). Sizes `ps.FULL`, `ps.WIDE`, `ps.TALL`; full text width is 6.1 in.
- Public names only in anything a reader sees: chambers **A–E** (det3=A, det2=B, det6=C, det7=D, det4=E), "Run 1" and
  "Run 2" for chamber A's two long runs, "reference" (never "M3"), no "pass 1", no "MX17", no run keys, no file names.
  Dates as "27 June", not "6-27".
- X view solid lines and filled markers, Y view dashed and open. Data are markers, models are lines.
- Panel labels **(a)**, **(b)** … bold, placed as in the existing figures (look at one of `papers/paper1/figures/*.pdf`).
- Type at document size; nothing smaller than about 7 pt when the figure is 6.1 in wide. No title on the figure:
  the caption carries it.
- **Check your own output**: `pdftoppm -r 110 -png -singlefile out.pdf /tmp/x` and Read the PNG. Fix overlapping
  labels, clipped text, unreadable legends before you report.

## What to report (your final message; under 500 words)
- Files written, and the exact command that regenerates each figure.
- What each panel shows, and the headline numbers with the file they come from.
- A draft caption in plain JINST style (lower-case "figure", no internal names).
- Anything you could not do, any number you are unsure of, any disagreement you found between sources. Say
  "not found" rather than guessing. If two sources disagree, give both and say which you trust and why.
