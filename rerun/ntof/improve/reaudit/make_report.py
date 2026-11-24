#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_report.py -- report.html of the n_TOF tracking re-audit of 2026-11-24.

    .venv/bin/python rerun/ntof/improve/reaudit/make_report.py

The text and the tables are in this file; the numbers were measured by the tests archived in tests/<id>/ (scripts and
small result tables; the caches and full outputs were in $WORK_DATA/claude-jobs/eb8200b3/ and are regenerable).  The
tables that have a machine-readable source are read from it (tests/W2a_map_r3/r3_validation.csv), so re-running the
map prototype and this script updates them together.  Figures are linked relatively (figures/*.png, git-ignored; a
copy with figures is on EOS, see README.md).
"""
from __future__ import annotations

import csv
import html
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------------------------------------------------
def esc(s):
    return html.escape(str(s), quote=False)


def table(head, rows, cls=''):
    h = ''.join(f'<th>{c}</th>' for c in head)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def fig(name, caption):
    return (f'<figure><a href="figures/{name}"><img src="figures/{name}" alt="{esc(name)}"></a>'
            f'<figcaption>{caption}</figcaption></figure>')


def ul(items):
    return '<ul>' + ''.join(f'<li>{i}</li>' for i in items) + '</ul>'


def r3_rows():
    """Held-out clean sigma per chamber and view from the prototype's validation table (x1e-3)."""
    src = HERE / 'tests' / 'W2a_map_r3' / 'r3_validation.csv'
    v = {}
    for r in csv.DictReader(open(src)):
        if r['section'] == 'cosmic_heldout' and r['metric'] == 'sigma_clean' and r['region'] == 'all':
            v[(r['arm'], r['view'], r['model'])] = (float(r['value']) * 1e3, float(r['err'] or 0) * 1e3)
    names = [('none', 'No map'), ('free50', 'Current algorithm (free 50 mm cells), held out'),
             ('r2_file_insample', 'Shipped r2 file (in sample: trained on these lines)'),
             ('T6_pot10_s3', 'Polynomial potential of order 10 + scale of order 3'),
             ('T6_cell33_s3', 'Free 33 mm tilt cells + scale of order 3'),
             ('spl_free_common', 'Spline, two free tilt fields, one common scale'),
             ('spl_pot_common', 'Spline, tilt from one potential, one common scale'),
             ('spl_pot_sep', '<b>Spline, tilt from one potential, one scale per view (r3 candidate)</b>')]
    rows = []
    for key, label in names:
        cells = [label]
        for arm in 'AC':
            for view in 'xy':
                x = v.get((arm, view, key))
                cells.append(f'{x[0]:.1f} ± {x[1]:.1f}' if x else '')
        rows.append(cells)
    gain = ['Gain of r3 over the current algorithm']
    for arm in 'AC':
        for view in 'xy':
            a, b = v[(arm, view, 'spl_pot_sep')][0], v[(arm, view, 'free50')][0]
            gain.append(f'{100 * (a / b - 1):+.0f} %'.replace('-', '−'))
    rows.append(gain)
    return rows


# ---------------------------------------------------------------------------------------------------------------------
CSS = """
body{font:15px/1.5 system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1c2330;background:#fff;
     max-width:1040px;margin:0 auto;padding:18px 18px 60px}
h1{font-size:25px;margin:8px 0 4px} h2{font-size:20px;margin:34px 0 8px;border-bottom:1px solid #d5dae3;padding-bottom:4px}
h3{font-size:16.5px;margin:22px 0 6px} p{margin:8px 0} .sub{color:#5a6475;margin:0 0 14px}
.verdict{background:#f1f6fd;border-left:4px solid #2f6fba;padding:10px 16px;margin:14px 0}
.warn{background:#fdf5ea;border-left:4px solid #c9822b;padding:10px 16px;margin:14px 0}
table{border-collapse:collapse;margin:10px 0 14px;font-size:13.5px;display:block;overflow-x:auto}
th,td{border:1px solid #d5dae3;padding:4px 8px;text-align:left;vertical-align:top} th{background:#f3f5f9}
figure{margin:14px 0} img{max-width:100%;border:1px solid #e3e7ee} figcaption{font-size:13.5px;color:#444e5e;margin-top:4px}
code{background:#f3f5f9;padding:1px 4px;border-radius:3px;font-size:13px} ul{margin:6px 0 10px 0;padding-left:22px}
li{margin:3px 0} .small{font-size:13px;color:#5a6475}
@media (prefers-color-scheme: dark){body{background:#14181f;color:#e3e7ee} th{background:#222935}
 th,td{border-color:#39414f} .verdict{background:#1b2636} .warn{background:#2d2518} code{background:#222935}
 h2{border-color:#39414f} .sub,.small,figcaption{color:#a3adbd} img{border-color:#39414f;background:#fff}}
"""

S = []


def add(*parts):
    S.extend(parts)


add('<h1>n_TOF tracking: re-audit before the test campaign</h1>',
    '<p class="sub">2026-11-24. Baseline: the rebuilt analysis view <code>stage3_r2_final</code> (pass 1 with the audit '
    'fixes of 22 and 23 November). Existing data only. Tests, scripts and tables: <code>tests/</code> next to this file.</p>')

# ------------------------------------------------------------------------------------------------------------- verdict
add('<div class="verdict">',
    '<p><b>Verdict.</b> The work of the last days is consistent and the rebuilt track sample is healthy. '
    'However, the test campaign was not ready to run as prepared, and the largest gain in angle resolution that we '
    'found does not need a new reconstruction at all.</p>',
    ul([
        '<b>The studies agree with each other and with the products.</b> The code of the test campaign with every '
        'switch off reproduces pass 1 track by track. The sample of chambers A and C is uniform over the campaign.',
        '<b>The test campaign had a blocking error.</b> Its stage-3 chain would have failed on every sub-run. This is '
        'fixed and checked. The comparison script that the campaign needs did not exist, and it now does.',
        '<b>The campaign needs changes before it is submitted.</b> It costs about twice the estimate, chamber D needs a '
        'longer job flavour, no beam-off cosmic sub-run is in it, several decided items are in no configuration, and '
        'one switch (the seed ranking) should be corrected first.',
        '<b>A new angle map is the largest measured gain.</b> A smooth spline map fitted to the same cosmic lines is '
        '16 to 26 % better than the current map on held-out lines, in chambers A, C and D. It is a table-level step.',
        '<b>The drift geometry found on the bench is present at n_TOF.</b> The tilt of the drift lines is the gradient '
        'of one potential, as an electrostatic origin requires. The correction has to be measured at n_TOF, because the '
        'cathode changes shape with every mounting. The cosmic lines measure it down to the noise of a single track.',
        '<b>A second reconstruction pass would gain efficiency and purity, not resolution.</b> We found no '
        'reconstruction-level change that improves the angle of a track.',
    ]),
    '</div>')

add('<h3>Headline numbers</h3>',
    table(['Quantity', 'Value', 'Source'], [
        ['<code>ref</code> (all switches off) against pass 1, 8 acquisitions of A and C',
         '20,251 tracks in both, 100 % matched, largest difference 1.3 × 10⁻⁶ mm and 1.0 × 10⁻⁷ in tan', 'T2'],
        ['Held-out width of the cosmic-line residual, current map → spline map (× 10⁻³, x / y view)',
         'A 21.2 / 19.6 → 15.7 / 16.5; C 24.3 / 28.7 → 18.2 / 22.0; D 34.8 / 33.9 → 25.6 / 26.0', 'W2a'],
        ['Late gated tracks (≥ 25 ms) with every switch on, against pass 1',
         'A +3.7 %, C +8.1 %, D −6.6 %', 'T2'],
        ['Cost of the eight-configuration test', 'about 10,000 to 12,000 core-hours (estimate was 5,000)', 'T1, T2'],
        ['Gain of a per-track timing correction (held out, cosmic lines)', '0.4 to 3.3 %', 'T9'],
        ['Common timing error of the x and y views of one track', 'correlation −0.002 ± 0.008 (A), −0.018 ± 0.008 (C)', 'T9'],
        ['Chamber D, late gated tracks at x_local 0 to 60 mm relative to the neighbouring regions',
         '9 to 13 % for the whole campaign', 'W2b'],
    ]),
    '<p class="small">An interruption is recorded here for completeness. The session login expired at about 00:55, '
    'so the tests ran from 08:30 on 24 November and not overnight. The single-switch configurations of the campaign '
    'were not run.</p>')

# ------------------------------------------------------------------------------------------------------------- part 1
add('<h2>1. Does everything line up?</h2>',
    '<p>We checked the work of the last three days in four ways. We reviewed the phase-4 code against the roadmap and '
    'against the three code reviews of the audit. We reconstructed a few acquisitions with every switch off and '
    'compared them with pass 1. We compared the first outputs with every switch on with pass 1. Finally, we made a '
    'time series of the track sample over the whole campaign, sub-run by sub-run.</p>')

add('<h3>1.1 The reference reproduces pass 1</h3>',
    '<p>Configuration <code>ref</code> is the code of the test campaign (<code>code_e86028b5</code>) with every switch '
    'off. We ran it on the four A and four C acquisitions of the sizing cluster (condor cluster 4729151, 8 jobs, all '
    'finished with exit code 0). The tracks agree with <code>stage3_r2_final</code> one to one. There are 20,251 tracks '
    'in both sets, of which 6,513 are gated, and every gated track is matched. The largest differences are '
    '1.3 × 10⁻⁶ mm in position, 1.0 × 10⁻⁷ in tan and 1.05 × 10⁻⁴ ns in t0, which is rounding on different worker '
    'nodes. This means that any difference seen in a test configuration comes from its switch.</p>')

add('<h3>1.2 Problems found in the test campaign</h3>',
    table(['#', 'Problem', 'Consequence', 'Status'], [
        ['1', '<code>stage3_chain.sh</code> called stage 3 on a test output that has no chamber-B files. '
              '<code>ntof_stage3</code> stops when a file of any chamber is missing.',
         'No table could have been built from the 896 jobs.',
         '<b>Fixed.</b> <code>ntof_stage3.py --allow-missing</code>, used by the chain. The fixed chain reproduces '
         'pass 1 on the <code>ref</code> outputs.'],
        ['2', 'The comparison script did not exist.', 'No number could have been produced.',
         '<b>Added:</b> <code>phase4/compare.py</code>.'],
        ['3', 'The ringing correction marked the first event of each new waveform reader as corrected whenever an '
              'earlier event had been corrected (<code>deflect.py</code>).',
         'One event per acquisition and readout board lost its hits below 5σ on a plane that was not corrected.',
         '<b>Fixed.</b> The flag is now set per reader. The copy on lxplus must be redeployed.'],
        ['4', 'The seed ranking by charge (4.4) is applied before the veto of wide noise bands, and a wide band has a '
              'large summed charge.',
         'A noise band can take one of the five seed slots and then be dropped. This is the first suspect for the '
         'real chamber-D tracks that disappear with every switch on.',
         'Open. The review asked for a ranking by significance.'],
        ['5', 'The job flavour <code>workday</code> (8 h) is too short for chamber D. The large D acquisitions took '
              '5.4 to 8.3 h in pass 1 and 6.9 to 7.6 h in the sizing cluster.',
         'Some D jobs would be killed.', 'Open. Use <code>tomorrow</code> for D.'],
        ['6', 'The cost was sized on configuration <code>all</code>, in which the ringing correction removes many '
              'early fits. A <code>ref</code> job takes about 3 times longer on A and C.',
         'About 10,000 to 12,000 core-hours instead of 5,000. This is within our budget.', 'Open (information).'],
        ['7', 'No beam-off cosmic sub-run is in the test.',
         'No configuration can be compared with the cosmic-line angle truth. The angle map is trained on cosmic runs.',
         'Open. Add two or three sub-runs, for example of run_149.'],
        ['8', 'Chamber D is not run on run_104, although the canon names a bundle for it.',
         'The early-run slope cut of D cannot be tested.', 'Open.'],
        ['9', 'The depth profiles (4.11) are stored only in <code>all</code>.',
         'They change no fit, so they should be on in every configuration.', 'Open.'],
        ['10', 'The sizing cluster ran D on <code>bundles_osc/D_E2_beam</code>. The campaign uses the canon bundle '
               '<code>D_E2_beam_L1</code>.', 'The sizing D jobs are not the campaign configuration.', 'Information.'],
    ]),
    '<p>The following items were decided or recommended in the last days, but no configuration of the campaign tests '
    'them.</p>',
    table(['Item', 'Where it was decided', 'Level'], [
        ['The n_TOF x–y time tolerance (A 151, C 296, D 535 ns) in the bundles. The bundles carry the bench values '
         '(197, 456, 726 ns).', 'x/y matching study, recommendation 3', 'reconstruction'],
        ['The per-track waveform flag that the ringing rule of 23 November needs. <code>--ringing</code> does not '
         'write it.', 'roadmap 2.8', 'reconstruction'],
        ['The second cause of a lost second track: both views are fitted, but the x–y association builds one track.',
         't0 study (one third of the losses)', 'reconstruction'],
        ['An upper bound of the column end. The present bound (1.6 T) is beyond the profile grid and never rejects.',
         'reco review 4, t0 study', 'reconstruction'],
        ['The charge rule q ≥ 600 ADC. It is decided, but <code>LOW_CHARGE_RULES</code> is still <code>none</code>.',
         'roadmap 2.8', 'table'],
        ['The late quality reference (4.6). It changes a flag column that is not part of <code>gated</code>, and no '
         'configuration sets it.', 'roadmap 4.6', 'table'],
        ['The matching-score column, the per-track wall time, the partly censored x windows of C.',
         'roadmap 4.1, t0 study, finding 23', 'table'],
    ]),
    '<p>The full table, with about 60 rows, is <code>tests/T1_phase4_review/completeness_table.md</code>. The code '
    'findings with file and line are in <code>tests/T1_phase4_review/q1_code_findings.md</code>.</p>')

add('<h3>1.3 First look at every switch on</h3>',
    '<p>The sizing cluster ran configuration <code>all</code> on eight A and C acquisitions and four D acquisitions. '
    'We compared it with pass 1 on the same acquisitions. Since every switch is on, the attribution to one switch is '
    'an inference from the track properties and not a measurement. The single-switch configurations are still needed.</p>',
    table(['Quantity (<code>all</code> / pass 1)', 'A', 'C', 'D'], [
        ['Gated tracks', '2,975 / 3,520', '2,838 / 2,993', '1,451 / 1,539'],
        ['Gated tracks at 1 to 3 ms', '49 / 274', '249 / 290', '7 / 6'],
        ['Late gated tracks (≥ 25 ms)', '1,481 / 1,428', '1,270 / 1,175', '875 / 937'],
        ['Wall fraction: matched / only in <code>all</code> / only in pass 1', '0.86 / 0.50 / 0.21',
         '0.94 / 0.55 / 0.37', '0.85 / 0.47 / 0.56'],
        ['Split children at the t0 limit (number of children)', '0.36 (31) / 0.44 (69)', '0.25 (95) / 0.36 (137)',
         '0.23 / 0.33'],
        ['Late pointing, fraction within 15 mm of the source', '0.609 / 0.611', '0.575 / 0.575', '0.571 / 0.580'],
        ['Same-chamber pair fraction', '0.013 / 0.040', '0.024 / 0.026', '0.018 / 0.011'],
    ]),
    ul([
        '<b>Head-on rescue (4.5) adds real tracks.</b> Of the tracks that only <code>all</code> has from 16 ms on, 65 % '
        '(A) and 47 % (C) are at |tan x| &lt; 0.08, and the wall confirms them as often as the matched tracks '
        '(0.95 and 0.99). This is the gain of 3.7 % (A) and 8.1 % (C) in late tracks.',
        '<b>The ringing correction (4.10) removes early fakes.</b> On A, 825 tracks before 12 ms disappear and the wall '
        'confirms only 0.18 to 0.23 of them. The fall of the A pair fraction from 0.040 to 0.013 is all early. The '
        'correction also moves 71 % of the matched A tracks at 16 to 25 ms by small amounts (95 % within 0.38 mm).',
        '<b>The t0 prior on split children (4.3) lowers the railing</b>, but the samples are small.',
        '<b>The local seeding floor (4.12) shows no visible gain here.</b> Only 1 % of the added late tracks are second '
        'tracks. The sample holds only 158 (A) and 103 (C) two-track events from 12 ms on, so this switch has to be '
        'judged with the injection test, which gave 76 → 87 % (A) and 71 → 84 % (C).',
        '<b>Chamber D loses real tracks.</b> D loses 131 late tracks that the wall confirms at 0.985 and gains 69 that '
        'are worse (wall 0.957). Between 21 and 30 of the lost tracks are removed by the flags that the added hot '
        'channels raise. The other 88 have no matching track at all, so a reconstruction switch removed them. '
        'For this reason D should not receive every switch as it stands.',
    ]))

add('<h3>1.4 The sample over the campaign</h3>',
    '<p>We made one row per sub-run and chamber from <code>stage3_r2_final</code> and looked for sub-runs and eras that '
    'deviate. The known conditions all appear: the access of 27 July, the dead connector of chamber A in run_79 and '
    'run_81, the latency epochs of D. The fraction of tracks with no time since the flash is 0.68 %, against 14.7 % '
    'before the fix of the two-slim error.</p>',
    ul([
        '<b>Chambers A and C are uniform.</b> The v(t) correction holds their angle scale to 1 to 2 % while the drift '
        'velocity changes by 14 % on C. No step of tens of ns exists in t0 anywhere. The horizontal source crossing is '
        'constant from run_84 on (A −10.81 mm, C −11.14 mm, with a spread of 0.3 mm between runs).',
        '<b>Chamber D has an efficiency hole at x_local 0 to 60 mm for the whole campaign.</b> Late gated tracks reach '
        '9 to 13 % of the level of the neighbouring regions. From a linear interpolation, about 26 % of D\'s late '
        'sample is missing there. This region is where D sees the source. The occupancy change at run_133 is not dead '
        'strips coming back: the four masked strips stay dead, and five live channels of the same connector became '
        'hotter after a new pedestal was loaded on 4 August. The fix is a fiducial exclusion or an efficiency that '
        'changes at run_133. No new mask is needed.',
        '<b>The angle scale of D does not follow v(t).</b> The first reading of the stability check (0.35) came from a '
        'change of which tracks are in the sample. Against the pointing truth the dependence is 0.04 ± 0.08.',
        '<b>Chamber B has a noisy channel group</b> (two x channels and y channels 13 to 63) until about run_117. '
        'It holds 17.8 % of B\'s late tracks through run_106, and these are fakes. It touches about 1 % of the B–D '
        'lines. The fix is a table-level exclusion of that box through run_132.',
        '<b>Unexplained and small:</b> a t0 step of 2 to 6 ns and a velocity jump at run_150, a common dip of 5 to 10 % '
        'in the track rate in runs 139 to 147, and a loss of D tracks at x_local 60 to 120 mm in runs 135 to 150 that '
        'the beam-off runs do not show.',
    ]),
    fig('D_xprofile.png', 'Chamber D: profile of the late gated tracks along x_local before and after run_133. The '
                           'region from 0 to 60 mm is depleted in both periods.'),
    fig('stability_crossing.png', 'Horizontal source crossing per run for chambers A, C and D.'))

add('<h3>1.5 Is the beam angle scale off by 1 to 5 %?</h3>',
    '<p>Two tests measured the angle scale of the beam tracks as the slope of the measured horizontal tangent against '
    'the tangent expected from the track position and the source. They find 0.990 (A), 0.965 (C) and 0.946 ± 0.009 (D). '
    'The cosmic lines give D a scale of 0.997 ± 0.004. We therefore applied the same estimator to the Geant4 capsule '
    'electrons, where the truth is known.</p>',
    ul([
        'On Geant4 truth, with no detector, the estimator gives 0.989, 0.986, 0.972 (A, C, D) with the nose library and '
        '0.931, 0.924, 0.987 with the EAR2-beam library. The statistical errors are 0.011 to 0.022.',
        'In Geant4 the deficit comes mostly from scattering and from the cut |tan| &lt; 0.5. The size of the source '
        'contributes less than 1 %.',
        'So the deficit seen in the data is consistent with an effect of the estimator. It is not evidence of a '
        'detector scale error, and we do not propose a scale correction for D. The cosmic lines remain the reference '
        'of the angle scale.',
        'Geant4 cannot give the bias to 1 %. The two libraries differ by about 5 %, the order of the chambers does not '
        'match the data, and the Geant4 pointing core is 30 to 60 % wider than the data.',
    ]),
    '<p class="small">Two side findings on the Geant4 digitisation chain, not followed up: the pass-1 reconstruction of '
    'the digitised events reads tangents 3.4 ± 1.0 % (nose) and 9 ± 2 % (EAR2) steeper than the truth line, and the y '
    'view of chamber C in the EAR2 product is not correlated with the truth (0.02).</p>')

# ------------------------------------------------------------------------------------------------------------- part 2
add('<h2>2. The drift geometry at n_TOF</h2>',
    '<p>On the cosmic bench, the drift lines of chamber A are tilted and bent by two surfaces, the mesh and the '
    'cathode. The bench audit of the same night confirmed this picture and found that the cathode is a foil whose '
    'shape changes with every mounting, while the dish of the mesh stays '
    '(<code>rerun/paper1_gap/audit/AUDIT_2026-11-24.md</code>). For n_TOF this has one direct consequence: a map made '
    'on the bench is not a property of the chamber, so the correction must be measured in the n_TOF mounting. '
    'We tested three ways to measure it with the data we have.</p>')

add('<h3>2.1 The distortion at n_TOF is electrostatic</h3>',
    ul([
        '<b>The tilt field is close to curl-free.</b> The x view and the y view each measure one component of the tilt '
        'of the drift lines. If the tilt comes from the shape of the electrodes, the two components are the gradient '
        'of one potential. The curl of the measured field is 1.03 × 10⁻⁴ per mm on A and 1.06 × 10⁻⁴ on C. The noise '
        'gives 0.88 and 0.97 × 10⁻⁴, and a shuffled map gives 3.0 × 10⁻⁴. A small curl above the noise remains '
        '(χ² 114 and 155 for 49 points).',
        '<b>Requiring one potential costs nothing.</b> A spline map whose tilt is the gradient of one potential is '
        'never worse on held-out lines than a map with two free tilt fields (differences −0.30 to +0.06 × 10⁻³).',
        '<b>The scale is common to the two views on C, but not on A.</b> On C the per-cell scales of the two views '
        'correlate at 0.86 with slope 0.96 ± 0.03. On A the correlation is 0.28. One gap map therefore does not '
        'describe the scale of A.',
        '<b>The bench bow block over-corrects chamber A at n_TOF</b> and is right for C (bench audit). This is expected '
        'if the dish of A is mostly the sag of a horizontal board, since the chambers stand vertically at n_TOF. '
        'The in-situ map is fitted on top of the bow block, so it absorbs this.',
    ]),
    fig('curl_scale_A.png', 'Chamber A, free 50 mm cells: the scale of the y view as a function of the scale of the '
                             'x view, and the two cross derivatives of the tilt field.'),
    fig('curl_scale_C.png', 'Chamber C, the same quantities.'))

add('<h3>2.2 Cosmic lines: a spline map reaches the noise of a single track</h3>',
    '<p>The current map has one tilt and one scale per 50 mm cell and view. We replaced the cells with penalised '
    'cubic B-splines, with knots every 20 mm for the tilt and every 40 mm for the scale. The smoothing is chosen by '
    'cross-validation over runs. Table 1 gives the width of the residual of clean cosmic lines that the map has not '
    'seen (the map is trained on one run parity and tested on the other).</p>',
    table(['Model (clean single-view width, × 10⁻³ in tan)', 'A x', 'A y', 'C x', 'C y'], r3_rows()),
    ul([
        '<b>Chamber D.</b> Held out by run parity on the B–D lines: no map 35.5 / 36.3, current algorithm 34.8 / 33.9, '
        'spline 25.6 / 26.0. The current 100 mm cells barely beat no map when they are tested on other runs.',
        '<b>The map is now limited by the noise of a track, not by the number of lines.</b> The learning curve '
        'extrapolates to 15.5 / 16.3 (A) and 18.2 / 21.8 (C), and twice the lines would gain 0.1 to 0.3.',
        '<b>The edges gain most.</b> Beyond 125 mm from the centre the width falls from 24.7 / 21.7 to 17.1 / 18.4 '
        'on A and from 33.0 / 31.7 to 20.0 / 22.5 on C.',
        '<b>Much of the error of the current map is noise in its per-cell scales.</b> Keeping the 50 mm tilt cells and '
        'forcing only the scale to be smooth already gives 18.7 / 18.6 on A.',
    ]),
    fig('r3_sigma_vs_model.png', 'Held-out width of the cosmic-line residual for each model, chambers A, C and D.'),
    fig('r3_learning_curve.png', 'Held-out width as a function of the number of training lines.'),
    fig('r3_fields_A.png', 'Chamber A, spline map: the potential of the tilt, the two tilt components and the scales.'),
    fig('r3_fields_C.png', 'Chamber C, spline map.'),
    fig('r3_fields_D.png', 'Chamber D, spline map (B–D lines and capsule pointing together).'),
    '<p>On the beam tracks the map changes the pointing core very little, because the core is set by the scattering of '
    'the electrons. What changes is the bias from cell to cell. With the current map and with the spline map, the rms '
    'of the residual centre over cells is 0.0086 → 0.0059 (A), 0.0089 → 0.0082 (C) and 0.0164 → 0.0065 (D, on tracks '
    'the fit did not see). On D the fraction of late tracks within 15 mm of the source rises from 0.554 to 0.590.</p>',
    '<p>The candidate is <code>tests/W2a_map_r3/angle_map_r3_candidate.json</code>, with its evaluator '
    '<code>apply_r3.py</code> (pure numpy). It is not applied anywhere.</p>')

add('<h3>2.3 Capsule pointing: an independent confirmation, and an input for D</h3>',
    '<p>A late track from the capsule has an expected horizontal tangent from its position alone. With about one '
    'million late tracks per chamber, the pointing measures the distortion per 25 mm cell with an error of 0.001 to '
    '0.0015. At one position the expected tangent is nearly fixed, so the pointing measures one combination of tilt '
    'and scale and cannot separate them.</p>',
    ul([
        '<b>The pointing confirms the map.</b> The spline map fitted to the cosmic lines alone predicts the pointing '
        'cells with a correlation of 0.93 (A), 0.70 (C) and 0.94 (D).',
        '<b>It does not improve the map of A or C.</b> Adding the pointing cells to the spline fit changes the held-out '
        'cosmic width by less than 0.3 × 10⁻³.',
        '<b>It helps D.</b> On D the cell-to-cell rms on unseen beam tracks falls from 0.0082 to 0.0065 when the '
        'pointing joins the B–D lines. The candidate uses both for D.',
        '<b>It covers a band only.</b> The pointing works where the expected |tan| is between about 0.10 and 0.44. '
        'The edges in x are outside the gate.',
        '<b>Electrons and muons differ.</b> The pointing residual depends on the track charge (0.008 to 0.010 rms '
        'between charge tertiles).',
    ]),
    fig('pointing_validation.png', 'Width of the cosmic-line residual with no map, with the cosmic map trained on the '
                                    'other run parity, and with corrections built from the pointing.'))

add('<h3>2.4 Drift-time maps: measurable everywhere, but not a substitute for an angle truth</h3>',
    '<p>Each track carries its own drift-time column, so a map of the column duration needs no reference. We used '
    'the 90 % point of the charge profile of late tracks.</p>',
    ul([
        '<b>The maps are stable.</b> The first and second halves of the campaign correlate at 0.97 (A), 0.99 (C), '
        '0.99 (D) and 0.90 (B). The x and y views agree, and beam-off tracks give the same maps.',
        '<b>On A the map behaves like a gap map.</b> The duration varies by about 5 % rms, and its relation to the '
        'cosmic scale has the slope that a gap change predicts (−0.37 ± 0.08 and −0.31 ± 0.05 for −0.34). '
        'The correlation is only −0.3, so the duration explains a small part of the scale.',
        '<b>On C and D the map is too large to be a gap.</b> C tilts by 10 % per 100 mm along both axes and has a '
        'rectangular block that is more than 15 % long. D tilts by 13 % per 100 mm along x. Read as a gap, these give '
        '27 to 33 mm and 28 to 34 mm.',
        '<b>The scale gradient of C along y is not a drift-velocity effect.</b> The duration falls by 7.8 % per '
        '100 mm along y, so a velocity change would make the scale rise. The scale falls.',
        '<b>As a correction it is weaker than the map.</b> The bench audit predicted the tilt from the drift-time map '
        'at n_TOF. This recovers 7.5 % on A x, about 2 % on A y and C, and 12 to 13 % on D. The spline map gives 24 to '
        '35 % against no map.',
    ]),
    fig('tmaps_xy.png', 'Column duration relative to the chamber median, per 25 mm cell, for chambers A, C, D and B '
                         '(late beam tracks, mean of the x and y views).'),
    '<p>One possible reading of C and D is that the 90 % point of the charge profile also responds to electron '
    'attachment, since B, C and D carried water and A was dry. This is an inference and we did not test it.</p>')

add('<h3>2.5 What a track\'s own timing tells</h3>',
    '<p>The two views of a chamber see the same electrons, but the fit treats them separately and leaves the column '
    'duration free. We tested on the cosmic lines whether per-track timing predicts the angle error.</p>',
    ul([
        'A view whose column is longer than the local median reads a shallower slope (Spearman −0.08 on A, −0.16 on C). '
        'The shift is 2 to 3 % of the slope for steep tracks.',
        'A correction built from it gains 0.4 to 3.3 % on held-out lines, and nothing measurable on the beam.',
        'The relative errors of the x and y views of one track are not correlated (−0.002 ± 0.008 on A, '
        '−0.018 ± 0.008 on C). A joint x–y column in the fit would therefore gain less than 0.5 %.',
    ]),
    '<p>For this reason we propose no timing correction and no joint x–y fit.</p>')

add('<h3>2.6 The three n_TOF candidates of the bench audit</h3>',
    '<p>The bench audit lists three candidates for n_TOF (its HANDOFF section, item 3). Our tests answer each of them. '
    'The widths below are held out on the cosmic lines. The two audits quote different widths for "no map", because '
    'the bench audit uses all lines and we use clean lines, so we compare the relative gains.</p>',
    table(['Candidate of the bench audit', 'Its measured gain', 'With the spline map (r3)', 'Reading'], [
        ['A drift-time correction for chamber D',
         'D z view 35.9 → 31.4 × 10⁻³ (−13 %), y view 0 to 3 %',
         'D 35.5 / 36.3 → 25.6 / 26.0 × 10⁻³ (−28 % in both views)',
         'The spline map on the B–D lines and the pointing gives more, in both views. The drift-time route stays '
         'useful as a check that needs no reference.'],
        ['The bench fine map of chamber A, with no dish, against the in-situ map',
         'A x 17.6 %, A y 1.9 % (8.5 % with fitted amplitudes)',
         'A x −35 %, A y −24 % against no map',
         'The in-situ spline map is better in both views. The bench map is an independent confirmation of the shape.'],
        ['The gap map of each chamber as an input to the v(t) angle scale',
         'not measured there',
         'The drift-time map correlates with the cosmic scale at only −0.3 on A and C, and on C and D it is not a gap map.',
         'We do not recommend it. The spline scale measures the scale directly.'],
    ]),
    '<p>The spline map is at the noise of a single track on A and C, so a term added on top of it has nothing left '
    'to gain there. The mapping between the bench frame and the n_TOF local frame, which test T6 could not find, is '
    'derived in <code>rerun/paper1_gap/audit/results/g_RESULT.md</code>, section 1. One point of the bench audit '
    'remains open here too: the gap-gradient field of A from beam-off muons and from late beam tracks disagree along '
    'y (r = 0.04), although the drift-time maps themselves agree between the two samples (r = 0.96 on 50 mm cells).</p>')

# ------------------------------------------------------------------------------------------------------------- part 3
add('<h2>3. The plan we recommend</h2>',
    '<p>The plan separates what needs no new reconstruction from what does. The table-level steps come first, because '
    'they hold the largest measured gain and because they do not depend on the decision about a second pass.</p>')

add('<h3>Step 1. Table level (no condor)</h3>',
    table(['#', 'Change', 'Measured effect', 'Note'], [
        ['1', 'Replace the 50 mm cell map with the spline map (r3) for A, C and D.',
         'Held-out width −26 / −16 % (A), −25 / −23 % (C), −26 / −23 % (D).',
         'Wire <code>apply_r3.py</code> into <code>map_view.py</code>. Refit after any new reconstruction (minutes).'],
        ['2', 'Chamber D: exclude x_local 0 to 60 mm from efficiency and shape analyses, or use an efficiency that '
              'changes at run_133.', 'The region is 87 to 91 % inefficient.', 'D sees the source there.'],
        ['3', 'Chamber B: exclude the noisy box (x −52 to −44, y −200 to −160 mm) through run_132.',
         '17.8 % of B\'s late tracks through run_106 are fakes there.', 'Costs 0.05 % of B\'s tracks.'],
        ['4', 'Switch on the decided charge rule, the late quality reference, the matching score and the per-track '
              'wall time.', 'Decided or recommended earlier.', 'The full ringing rule also needs the waveform flag.'],
        ['5', 'No scale correction for D.', 'The deficit is consistent with the estimator (section 1.5).', ''],
        ['6', 'For D, keep the pass-1 bundle and flag the windows dominated by oscillating channels, instead of adding '
              'those channels to the hot list.', 'The hot list removes 21 to 30 real late tracks and gains at most 8.',
         'To be confirmed by the single configuration <code>osc</code>.'],
    ]))

add('<h3>Step 2. Before the test campaign is submitted</h3>',
    ul([
        'Rank the seeds by significance and apply the cap after the wide-band veto (4.4).',
        'Make <code>--ringing</code> write the per-track waveform flag, and store the depth profiles in every '
        'configuration.',
        'Add a configuration with the n_TOF x–y time tolerance in the bundles.',
        'Include chamber D on run_104, and two or three beam-off cosmic sub-runs for the angle truth.',
        'Give D the single switches before <code>all</code>. D lost real tracks with every switch on.',
        'Use <code>tomorrow</code> for D. Plan about 10,000 to 12,000 core-hours.',
        'Redeploy the code on lxplus, because two fixes were made after <code>code_e86028b5</code>.',
        'Free AFS space first. The quota is at 90 %, and <code>log_pass1_2026-11</code> uses 93 % of its name slots.',
        'Re-run the injection test on the new code for 4.3 and 4.12, and the Geant4 digitisation with any switch that '
        'is adopted, so that data and Geant4 go through the same seeder.',
    ]),
    '<p>Table 2 gives the quantity that decides each switch. Writing it down before the campaign runs avoids choosing '
    'the measure after seeing the result.</p>',
    table(['Switch', 'Adopt if', 'Reject if'], [
        ['<code>ref</code>', 'It reproduces <code>stage3_r2_final</code> within tolerance (done for 8 acquisitions).', ''],
        ['4.5 head-on rescue', 'The added tracks are confirmed by the wall as often as the matched tracks, and the '
                                'cosmic-line residual at |tan| &lt; 0.1 does not widen.', 'The added tracks point worse.'],
        ['4.10 ringing correction', 'Triggers from 20 ms on are identical to <code>ref</code>, and the early purity rises.',
         'Late tracks change.'],
        ['4.3 children t0 prior', 'The railing of split children falls and the injected pairs keep their opening angle.', ''],
        ['4.12 local floor', 'Both tracks of injected pairs are found more often, and the added single tracks are '
                              'confirmed by the wall.', 'The number of fakes rises on D.'],
        ['4.4 seed rank', 'No real late track is lost on any chamber, D included.', 'Real tracks disappear.'],
        ['4.9 D oscillating channels', 'The purity of D at 20 to 30 ms rises without a loss of wall-confirmed tracks.',
         'Real tracks are lost (the first look says they are).'],
    ]))

add('<h3>Step 3. The decision on a second pass</h3>',
    '<p>From what we measured, a second pass would bring the following.</p>',
    ul([
        'About 4 to 8 % more late single tracks on A and C, near normal incidence (head-on rescue).',
        'A cleaner sample before 12 ms (ringing correction). The clean tracks before 12 ms are about 9 % of all clean '
        'tracks of A, so this is a gain of statistics of that order at most.',
        'Fewer split children at the t0 limit (0.36 to 0.44 → 0.25 to 0.36).',
        'More second tracks of separated in-time pairs (injection: 76 → 87 % on A, 71 → 84 % on C).',
        'The stored depth profiles and the waveform flag, which no table-level step can supply.',
    ]),
    '<p>It would not improve the angle of a track. The angle map is at the noise of a single track, per-track timing '
    'gains at most 3 %, and a joint x–y fit gains nothing. The drift geometry needs nothing from a new '
    'reconstruction: the tangent is recomputed from the fitted speed in stage 3, so the whole correction is a table '
    'step.</p>')

# ------------------------------------------------------------------------------------------------------------- limits
add('<h2>4. What this audit does not rule out</h2>',
    ul([
        '<b>The single switches were not run.</b> Every attribution in section 1.3 is an inference from one '
        'configuration with all switches on, on 12 acquisitions of four sub-runs.',
        '<b>The gain of the spline map is measured on cosmic muons.</b> On beam electrons the pointing core is set by '
        'scattering, so the gain shows only as a smaller bias from cell to cell. We have no direct measurement of the '
        'effect on an opening angle.',
        '<b>The spline fields have structure that we did not explain</b>: a diagonal ridge on A, a scale band at the '
        'bottom of C, and scales down to −0.28 (A) and −0.36 (C) near the edges. Outside the region with training '
        'lines the map is clamped. On D this region covers 66 % of the chamber, and 4.4 % of D\'s beam tracks lie '
        'outside it.',
        '<b>The noise of a track could not be split into its sources.</b> The residuals of A and C against the same '
        'line are anticorrelated (−0.08), so the simple decomposition fails.',
        '<b>The drift-time maps of C and D are not understood.</b> They are stable and real, but they are not a gap.',
        '<b>The scale of the two views of A differs</b>, which one gap map cannot give.',
        '<b>The beam angle scale is known only through the cosmic lines.</b> The pointing estimator is biased by 0 to '
        '8 % in Geant4, depending on the source model.',
        '<b>The cause of D\'s hot channels after 4 August is an inference.</b> The per-channel thresholds were not read.',
        '<b>Chamber B has no angle</b>, so nothing here applies to it beyond positions.',
    ]))

add('<h2>5. Tests and files</h2>',
    table(['Test', 'Question', 'Directory under <code>tests/</code>'], [
        ['T1', 'Review of the phase-4 code and of the campaign design', '<code>T1_phase4_review</code>'],
        ['T2', 'Does <code>ref</code> reproduce pass 1? First comparison of <code>all</code>', '<code>T2_ref_compare</code>'],
        ['T4', 'Drift-time maps of the four chambers', '<code>T4_time_maps</code>'],
        ['T5', 'Angle map from capsule pointing', '<code>T5_pointing_map</code>'],
        ['T6', 'Is the distortion a potential plus a common scale?', '<code>T6_surface_map</code>'],
        ['T7', 'Stability of the sample over the campaign', '<code>T7_stability</code>'],
        ['T9', 'Does per-track timing predict the angle error?', '<code>T9_track_timing</code>'],
        ['W2a', 'The spline map candidate (r3)', '<code>W2a_map_r3</code>'],
        ['W2b', 'Chamber D and B items of the stability check', '<code>W2b_chamber_D</code>'],
        ['W3', 'The pointing scale estimator on Geant4 truth', '<code>W3_g4_scale</code>'],
    ]),
    '<p class="small">Each directory holds the scripts and the small result tables. The caches were on the desktop '
    'work disk and are regenerable from <code>stage3_r2_final</code> with the scripts.</p>')

OUT = HERE / 'report.html'
OUT.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width, initial-scale=1">'
               '<title>n_TOF tracking re-audit, 2026-11-24</title>'
               f'<style>{CSS}</style></head><body>' + '\n'.join(S) + '</body></html>')
print(f'wrote {OUT} ({OUT.stat().st_size / 1024:.0f} kB)')
