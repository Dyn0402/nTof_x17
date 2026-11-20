# Writing in Dylan's style (papers)

Dylan, 2026-11-20: Claude's paper prose "comes out too complicated" next to his own, which is simpler and more
pedagogical. This file is what was learned from his writing, to be read before writing or editing any paper text.

Sources read: his thesis (chapters 2, 3, 5, 6), the sPHENIX Vernier scan note (beam width, simulation), "Separating
Time Correlated Signal from Noise via Moving Average", the Saclay cosmic bench manual, the 18L/180A teaching manuals,
and his own Overleaf edits to Sections 1 to 4 of Paper I v2 (the word-level diffs of those edits are the best evidence:
`git log overleaf-paper1_v2/main --grep='Update on Overleaf'`, then `git show --word-diff`). Local copies of the
Overleaf projects: `$WORK_DATA/writing_style_corpus/` (regenerable with `git clone https://git.overleaf.com/<id>`).

## What his prose does

1. **It walks the reader through.** A section starts by saying what is measured and why it matters, then how it is
   done, then what is seen, then what it means. A sentence that announces the next step is welcome ("We first
   demonstrate ...", "To quantify this ...", "Now consider ..."). Intuition comes before the number.
2. **Sentences are complete and literal.** Each one names its subject, the quantity and what happens to it:
   "the uncertainty on the opening angle is 2.4°", "an angle error of 1° translates to 4.0 mm of uncertainty at the
   target", "the reference ray pointing resolution is 210-280 µm". Not "the opening angle is known to 2.4°", not
   "1° is 4.0 mm at the target".
3. **One claim per sentence.** A colon or semicolon that joins two claims becomes two sentences. Parentheses are
   fine for a short clarification or an example value.
4. **Connecting words carry the logic**: therefore, then, however, since, so, in this case, for this reason, in
   addition, here, this means that. He adds them where Claude's text left the link implicit.
5. **"We" for what we do** (we measure, we find, we use, we call, we expect). Passive is fine for procedure.
6. **Terms are defined in a full sentence before use, and the noun is repeated** rather than replaced by "it",
   "one", "the first", "the latter".
7. **Figures are introduced by name**: "Figure 5 shows ... as a function of ...", then what is observed in it, then
   what that means. Captions describe literally what is plotted, panel by panel, colour by colour.
8. **Plain verbs and adjectives**: shows, indicates, increases, decreases, is consistent with, agrees with, is small
   relative to, uniform, rotated, detected, discusses. "As a function of", not "against", in running text.
9. **Honest hedges**: can, typically, roughly, appears to, it is possible that. Limits are stated plainly ("we have
   no measurement of ...").
10. **Units and symbols are written out** (95/5 %, 210-280 µm, σ_θ = 1°).

## What he removes from Claude's text

- **Aphorisms and compressed contrasts**: "A point is not a particle.", "Water slows the drift; oxygen removes
  electrons from it.", "It does, in two ways that share nothing.", "one curve, one number". Say the thing in full.
- **Shorthand verbs**: "reads on 61 channels" (reduces to), "hardly counts beside it" (is small relative to),
  "is enough:" (deflects ... by), "closes to 2 %" (are consistent to 2 %), "can be had", "rests on", "is not a
  detail", "at work".
- **Elliptical sentences** that make the reader supply the subject or the comparison.
- **Asides and extra numbers** that are not needed for the argument (a second unit conversion, a toy cross-check,
  a list of three variants). When rewriting existing text, keep the results but move a secondary number into
  parentheses or a table; propose cuts to Dylan rather than making them.
- **Stacked qualifiers** in one sentence. Split it, or put the condition first in its own sentence.

## Rules when rewriting existing paper text in this style

- Change the prose only. Every number, unit, `\cref`, `\cite`, `\label`, equation, table, figure include,
  `\sectionstate`, `\intent`, `\askDylan`/`\askStephan`/`\analysis`/`\requote` box and `%` comment stays as it is.
- Add no claim and drop no result. A sentence may be split, reordered or given an explanatory lead-in.
- Section and subsection titles stay (the feedback note is keyed on them); suggest plainer ones separately.
- US spelling ("percent", "center"), "figure 3"/"section 5" lower case through `\cref`.
- It is fine for the text to get somewhat longer. It should not get denser.

## Example (Paper I v2, section 5)

Before: "No record of the gas on the bench exists beyond its nominal mixture, so the drift velocity has to come from
the data. It does, in two ways that share nothing, and the comparison of the two measures the drift gap."

After: "The drift velocity depends strongly on the gas, and the only record of the gas on the bench is its nominal
mixture. For this reason we measure the drift velocity from the data themselves. We do this with two independent
methods, and comparing the two also gives a measurement of the drift gap."

## Status and scope (Dylan, 2026-11-21)

After the Section 4 review of Paper I v2 Dylan said he is "very happy" with the writing since this guide was made, and
asked that this style and voice be used always: in papers, in notes for his CERN site, and when explaining things to him
in a session. The global instructions (`~/.claude/CLAUDE.md`, "Writing: Dylan's style") carry the short version for
every repo. This file stays the full guide.

What the Section 4 review added:

- **One name per thing, across the whole paper.** He chose "broken resistive strips" (not "non-amplifying stripes") and
  "poorly mounted readout card" (not "fault of the X readout") and wanted each used everywhere.
- **A term of art is worth introducing only if it is used several times.** If it is, define it in a full sentence at
  its first use ("A forward model is a calculation that predicts ...") and then use that term consistently.
- **Give the reason for a number or a choice** when it is one clause away (why the interior margin is 8 mm at the y sides).
- **How he likes a review:** fix plain typos directly, then give numbered suggestions, each with the proposed wording.
  He answers by number. Check a factual claim against the data or code before proposing wording that depends on it.
