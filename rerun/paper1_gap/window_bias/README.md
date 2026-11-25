# Drift gap: the end of the record and attachment (2026-11-25)

**Question (Dylan):** for the chambers whose drift column ends close to, or after, the end of the recorded waveform, is
the fitted drift time T (and so the gap v_ang x T of Paper I Sec. 5) biased?

**Answer.** Not from the record end, as long as the record ends after the column. The fit recovers T within 5 ns in toys at
every margin >= 0. Cutting the real stacks of A and B and of D's long run to 120 ns after the column moves T by 1-7 ns.
Chamber E's 6-24 long run is the quoted row with the shortest record (116 ns after the column, against ~500 ns of
electronics response). It is fine: its gap moves by < 0.2 mm, and E shows no attachment (tau 22 us). A record that ends
before the column end does bias T. That is chamber A at 177 V/cm, which is already quoted as a lower limit, and the
300 V row, which is not used.

**What does bias T is attachment, and a short record amplifies it.** The uniform-column fit returns T too short when the
column loses charge with drift time. Chamber C's 26 June rows prefer tau ~ 2 us (1.5-2.4 sigma per row). With the loss
fitted, C's gap would be 30.1 mm instead of 29.4 mm. D's HV-scan rows lean the same way (+0.25 mm). Paper I quotes the
uniform-column gap and states the +0.7 mm as a limit (askDylan box in Sec. 5).

**Bug found and fixed: D's long run had no fit.** `g_det7_long|in30` in `x17rerun/paper1_gas/fit_v1` is the fit's
starting point (T = 820, dt = 100, sig0 0.4, rms NaN). Its stack was empty: 141 masked X channels below 270 left the +2
offset at 58 % < 60 %. D's quoted water (0.48 %) was 29.44 mm / 820 ns. The same failure hit D's short run, a third of the
long run, and the 485 V `|all` arm. Fixed as follows:
- **D's long and short runs:** refitted with the S4 det7 recipe (X peak on ch >= 256, all events) and put in place of
  the failed fits, in `x17rerun/paper1_gas/fit_v2` (EOS). Long run T = 798.6 +- 17.2 ns, the S4 re-check had 797 +- 17.
- **The other two failed fits:** dropped.
- **The fitter:** `x_width.fit_x` now raises on an empty stack.
- **Consequences:** D water 0.48 -> 0.44 %. D gap 30.2 -> 29.4 mm (chi2 over its rows 12.6 -> 4.4 / 6). D drying 0.68 -> 0.35
  becomes 0.67 -> 0.33 %. Correlation of the C and D excesses 0.975 -> 0.976.

**Clock note.** The last charge of the column reaches the strips at stack time dt + 60 + T. predict_x's index 0 is
-300 ns and the template grid starts at -360 ns. Fig. two_velocities (a) draws its triangles 60 ns early; this is
cosmetic.

Report: `report.html` (figures in `figures/`). Scripts, in order:
- `inventory.py <root>`: margins per row; refits every stack and reproduces the stored T.
- `toy.py <root>`: the fit's model, cut and refitted, with and without attachment.
- `datacut.py <root>`: real stacks cut earlier.
- `attfit.py <root>`: fit with attachment free, plus closure.
- `attboot.py <root>`: bootstrap of that fit.
- `make_report.py`: builds the report.

`<root>` = a copy of cosmic_bench/ with `x17rerun/paper1_gas/{fit_v2, Analysis/**/headon.pkl}` and the arms' template
bundles (`Analysis_pass1/<row>/wft/calib_bundle`). On the desktop it is `$WORK_DATA/claude-jobs/a21e7df1/gapwin`.

Downstream:
- `rerun/paper1_gas/pg_gas.py <root>/x17rerun/paper1_gas/fit_v2` then `make_report.py` gives gas.json and summary.json.
- `pg_figs.py drying` remakes the drying figure.
- `rerun/paper1_gap/audit/gaps_all/make_gaps.py` remakes the gaps.
- `papers/paper1_v2/figs/make_gas_vE_all.py` remakes `gas_vE_all`.
