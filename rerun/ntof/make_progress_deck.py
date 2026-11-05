#!/usr/bin/env python3
"""make_progress_deck.py -- n_TOF pass 1 progress and next steps as a figure-first slide note.

Numbers come from the pre-freeze studies' findings (plan section 6, tests/*/FINDINGS.md) and are
set in the DATA block below with their source; edit that block when a study is re-run.

    python rerun/ntof/make_progress_deck.py [--out PATH]
    python ~/PycharmProjects/dylan-cern-site/scripts/add-note.py PATH --slug ntof-pass1-progress --force --deploy
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.expanduser(os.environ.get(
    'SLIDEDOC_DIR', '~/PycharmProjects/dylan-cern-site/scripts')))
import slidedoc as sd                                                  # noqa: E402
from slidedoc import (BLUE, ORANGE, RED, GOLD, GREY, GREEN, INK, MUT,  # noqa: E402
                      DBLUE, DGREEN, DINK, DMUT)

PLAN = 'rerun/ntof/NTOF_PASS_PLAN_2026-11-02.md'

# ---- DATA (source: plan section 6, 11-04/05 block) ----------------------------------------------
N_ACQ, N_FEU, N_FEU_EV, N_TRIG, N_JOBS = 3937, 31496, 246.2, 30.8, 15748
# t0 pin, A-C cosmic line, run_149 12 sub-runs (~1,850 lines), 68 % residual angle in degrees: (free, pinned, err of diff)
T0 = dict(A=(2.88, 2.62, 0.08), C=(3.33, 3.23, 0.09))
# Q7 beam overlay truth: both members found and correctly paired, % (gate, cost+charge); n overlays
Q7 = dict(A=(26, 61, 695), C=(25, 57, 783))
# bundles: arm -> (v um/ns, kw x/y, Dp, t0, note)
BUNDLES = [
    ('A', '38.6', '0.976 / 0.965', '0.0245', 'pinned σ 15 ns', 'line muons, A–C cosmic line'),
    ('C', '28.1', '1.009 / 1.000', '0.0136', 'pinned σ 15 ns', 'drifts ~1.5 % between sub-runs'),
    ('D', '31.0 ± 4 %', '0.880 / 0.881', '0.0157', 'free', 'beam: 360 ns oscillation, angles limited'),
    ('B', '42.6 (prior)', 'no angle', '–', 'free', 'positions only'),
]


def cover(D):
    nums = ''.join([
        sd.bignum(f'{N_ACQ:,}', 'acquisitions re-decoded and re-hit', DBLUE,
                  f'{N_FEU:,} FEU files, 0 errors; {N_FEU_EV:.0f} M FEU-events (~{N_TRIG} M triggers).',
                  tip='Condor 4612805 (decode, fa80c06) and 4616842 (rr1_cm0 hits).', size=64),
        sd.bignum('closed', 'pre-freeze studies', DGREEN,
                  'Seeding, v, kw, t0, tan_max, pairing, masks, Dp: each settled on data, not by default.',
                  tip='Plan section 5 and the 11-04 block of section 6.', size=64),
        sd.bignum(f'{N_JOBS:,}', 'condor jobs, ready to launch', '#f0a36b',
                  'One per acquisition and arm. Output bit-identical to the desktop on the dry run.',
                  tip='rerun/ntof/condor/: ntof_stage.sh, ntof.sub, make_ntof_jobs.py', size=64)])
    body = (sd.kicker('n_TOF beam data · full reprocessing · pass 1 · 5 Nov 2026')
            + '<h1 style="font-size:80px;font-weight:600;line-height:1.08;letter-spacing:-2px;width:1640px">'
              'n_TOF pass 1: every input is ready and every calibration choice is made. '
              'Only the bundles and the submission are left</h1>'
            + '<div style="flex:1"></div>'
            + f'<div style="display:flex;gap:56px">{nums}</div>')
    D.slide('cover', body, f'''
<p><b>What this is.</b> The reprocessing of every n_TOF trigger of ours with the reconstruction that locked on the bench and at SPS (code <code>6a0bb1f</code>, <code>Analysis_pass1</code>), calibrated from nothing on n_TOF's own conditions. Plan, decisions and every result: <code>{PLAN}</code>.</p>
<p><b>Dataset.</b> Final: 36 production runs by run title, last beam run 224715. Nothing here asks for new data; open questions are stated limitations.</p>''',
            dark=True, short='Answer')


def pipeline(D):
    done, todo = GREEN, GOLD
    steps1 = [
        dict(label='Registry', sub='run title, era, HV, latency', color=done, tip='make_ntof_registry.py; ours identified by run title only.'),
        dict(label='Re-decode', sub=f'{N_ACQ:,} acq · 0 errors', color=done, tip='condor 4612805, decoded_root_fa80c06'),
        dict(label='Hits', sub='rr1_cm0, one provenance', color=done, tip='condor 4616842; ~184 GB on EOS for runs ≥ 100'),
        dict(label='Channel layer', sub='8 masks: A–D × E1/E2', color=done, tip='E1 = before the 27 July access, E2 = after.'),
        dict(label='Condor tooling', sub='1 job / (acq, arm)', color=done, tip='Dry run: 4/4 ok, 582/582 candidates identical to desktop.'),
    ]
    steps2 = [
        dict(label='Calibration studies', sub='v · kw · t0 · Dp · tan_max · Q7', color=done, tip='Next slides.'),
        dict(label='Pass bundles', sub='(A,B,C,D) × (E1,E2) × (beam, cosmic latency)', color=todo, tip='ntof_bundle.py --mask-era --tan-max 0.6'),
        dict(label='N-G1', sub='run_145 + run_149 vs is2_v2', color=todo, tip='Plan section 3. is2_v2 is a reference, not a rerunnable baseline.'),
        dict(label='Full submission', sub='15,748 jobs, ledger', color=todo, tip='Archive the code first; lxstore status before and after.'),
        dict(label='Stage 3', sub='track tables, y up', color=todo, tip='build_tracks / campaign_tracks on the new rows.'),
    ]
    leg = sd.legend([('done', done, 'box'), ('next', todo, 'box')])
    body = (sd.title('Inputs are done; the work left is a short, ordered chain',
                     'Pass 1 pipeline, status on 5 Nov 2026')
            + sd.flow(steps1, size=34) + sd.flow(steps2, size=34) + leg
            + sd.p('Hover the boxes for the condor cluster or file behind each step.', 22, MUT))
    D.slide('pipeline', body, '''
<p>The registry is built from the run titles and run configs, so it carries latency, samples, HV, noise era and access epoch per sub-run. The noise floor doubled on 23 July (RdClk_Div 6→4): run_68 sits inside the bracket and is excluded or checked directly.</p>
<p>The same code path as the bench and SPS pass is used, with no WFT_* switches; the ledger must prove every row ran on the archived code and that every output exists (a few EOS FUSE I/O flakes are retried twice).</p>''',
            short='Status')


def bars_slide(D, sid, ttl, sub, data, ymax, ylabel, fmt, labels, leg_items, notes, tipf, short, err=None):
    P = sd.Plot(900, 600, x=(0, 5), y=(0, ymax), ylabel=ylabel, margin=(40, 30, 92, 104))
    P.yticks([(v, f'{v:g}') for v in range(0, int(ymax) + 1, max(1, int(ymax) // 5))])
    xt = []
    xs = [(1.0, 1.75), (3.25, 4.0)]
    for (arm, vals), (xa, xb) in zip(data.items(), xs):
        P.vbar(xa, vals[0], 110, GREY, tip=tipf(arm, 0), label=fmt(vals[0]))
        P.vbar(xb, vals[1], 110, BLUE if arm == 'A' else ORANGE, tip=tipf(arm, 1), label=fmt(vals[1]))
        xt.append(((xa + xb) / 2, f'chamber {arm}'))
    P.xticks(xt)
    P.text(2.5, ymax * 0.93, labels, 30, INK, 'middle', 600)
    body = (sd.title(ttl, sub) + P.svg(ttl) + sd.legend(leg_items))
    D.slide(sid, body, notes, short=short)


def t0_slide(D):
    P = sd.Plot(900, 600, x=(0, 5), y=(0, 4), ylabel='A–C line residual angle, 68 % (deg)',
                margin=(40, 30, 92, 104))
    P.yticks([(v, str(v)) for v in range(0, 5)])
    xs = [(1.0, 1.75), (3.25, 4.0)]
    xt = []
    for (arm, (free, pin, e)), (xa, xb) in zip(T0.items(), xs):
        P.vbar(xa, free, 110, GREY, tip=f'{arm}, free t0: {free:.2f}°', label=f'{free:.2f}°')
        P.vbar(xb, pin, 110, BLUE if arm == 'A' else ORANGE,
               tip=f'{arm}, pinned t0 (σ 15 ns): {pin:.2f}°\nΔ = {pin - free:+.2f} ± {e:.2f}°', label=f'{pin:.2f}°')
        xt.append(((xa + xb) / 2, f'chamber {arm}'))
    P.xticks(xt)
    P.text(2.5, 3.7, 'Pinning t0 sharpens A by 0.25 ± 0.08°', 30, INK, 'middle', 600)
    cp = sd.card(
        sd.p('The trigger knows the particle time 3–5× better than a free fit does (≈13 ns vs 47–72 ns).', 26)
        + sd.p('The beam trigger has its own latency, +11…+26 ns after the cosmic one.', 26), w=640)
    body = (sd.title('t0 is pinned to the trigger, with one latency per trigger class',
                     'Cosmic A–C line, 12 run_149 sub-runs, ~1,850 lines, held-out; pinned vs free')
            + sd.row(P.svg('t0 pin'), cp, gap=44)
            + sd.legend([('free t0', GREY, 'box'), ('pinned (A)', BLUE, 'box'), ('pinned (C)', ORANGE, 'box')]))
    D.slide('t0pin', body, '''
<p>Scored on the cosmic A–C line: the residual angle between each chamber's direction and the line through both mesh crossings, with kw measured on even sub-runs and scored on odd ones. C improves by 0.10 ± 0.09°, so neutral; A by 0.25 ± 0.08°, with A-x tan −11 %. Line events rise 6 %.</p>
<p>Beam latency files: <code>bundles/latency_{A,C}_r145beam.json</code> (A 6.9 / 20.0 ns, C 52.4 / 52.5 ns for x / y). On beam the paired capsule-pointing distance changes by −0.006 ± 0.008 mm (A) and −0.06 ± 0.02 mm (C).</p>
<p>Open: the ftst slope on beam is about −7 ns per class and the pin assumes −10; at ftst 5 that is at most 15 ns.</p>''',
            foot='Source: rerun/ntof/tests/t0pin/, plan section 6 (11-04).', short='t0 pin')


def q7_slide(D):
    bars_slide(
        D, 'q7', 'Cost pairing plus a charge check finds both tracks 2.3× as often',
        'Beam overlay truth: two real single beam tracks superimposed; both found and correctly paired',
        {k: v[:2] for k, v in Q7.items()}, 80, 'both members correct (%)', lambda v: f'{v:g} %',
        'Old gate → cost + charge', [('old gate pairing', GREY, 'box'), ('cost + charge (A)', BLUE, 'box'),
                                      ('cost + charge (C)', ORANGE, 'box')],
        '''
<p>Q7 compares <code>xy_pairing select</code> gate vs cost and <code>split_corrob</code> off vs charge, scored on overlays of real beam single tracks (A 695, C 783 overlays, steep donors included). Accepted ghosts per overlay fall 0.24 → 0.11; the shallow member of a shallow+steep pair is right 36 → 71 / 68 % of the time.</p>
<p>Nothing else moves: single tracks, the A–C line (1,647 → 1,645 lines, same residuals) and the capsule-pointing excess (A 1463 ± 69 → 1446, C 1206 ± 61 → 1201). With cost + charge the in-acceptance efficiency is flat in tan_max, tracks above 0.6 carry no pointing excess, and tan_max 1.0 or a column window only adds line mis-fits and +30–60 % CPU. <b>tan_max stays 0.6 raw.</b> Applied in <code>ntof_bundle.py</code>.</p>''',
        lambda arm, i: f'chamber {arm}, {"old gate" if i == 0 else "cost + charge"}: {Q7[arm][i]} % of {Q7[arm][2]} overlays',
        'Pairing', None)


def bundles_slide(D):
    rows = [[f'<b>{a}</b>', v, kw, dp, t0, n] for a, v, kw, dp, t0, n in BUNDLES]
    tips = ['v from the A–C line muons by column duration; kw closes on held-out cos_0001 (1.003 / 0.994).',
            'v adopted 28 mm-gap; kw 1.009 / 1.000 after one more step (<1 %).',
            'kw from B–D cosmic lines through centroids, ±0.03 syst; gap 29.44 ± 1.12 mm.',
            'Its column is not a box; v only moves B efficiency via the raw-tan gate.']
    tb = sd.table(['Chamber', 'v (µm/ns)', 'kw (x / y)', 'Dp (90/10)', 't0', 'Note'], rows, size=28,
                  widths=[160, 220, 260, 220, 240, 560], align=['left', 'right', 'right', 'right', 'left', 'left'], tips=tips)
    body = (sd.title('The angle scale at n_TOF is set by the timing v, with kw ≈ 1',
                     'Calibration values going into the pass bundles; each measured in situ on run_149 cosmics')
            + tb
            + sd.callout('The bench kw (1.02–1.08) does <b>not</b> carry to n_TOF. Bow, v map, w0 stay bench-only.', BLUE, 28)
            + sd.callout('Gas is Ar/iso 90/10, not the bench’s 95/5, so Dp is new: A 0.0245 ± 0.0014, C 0.0136 and D 0.0157 from Magboltz.', GOLD, 28))
    D.slide('bundles', body, '''
<p>v is the timing v = gap / column duration, never a waveform χ²(v). A–C line muons: A 38.4 ± 0.7 (x) / 38.9 ± 0.8 (y), C 27.7 ± 0.7 / 28.6 ± 0.7 µm/ns, in agreement with is2's geometric in-situ v (38.0 / 28.7) by an independent route.</p>
<p>Model checks: the template holds (A, C within 0.02 of peak; systematic 0.1° / 0.15 mm); the saturation rail stays at 3550 (a per-sample rail is the real fix, proposed in <code>model_checks/FINDINGS.md</code>, not applied). The channel layer is neutral on the A–C line (15 of 1,909 lines lost) and on beam removes 4.4 % of A's tracks, almost all non-pointing.</p>''',
            foot='Source: plan section 6; tests/t0pin, tests/model_checks, tests/d_b.', short='Calibration')


def last(D):
    nxt = sd.flow([
        dict(label='1 · Bundles', sub='A, B, C, D × E1/E2 × beam/cosmic latency; make --mask-era required', color=GOLD),
        dict(label='2 · Re-check', sub='A–C line and capsule pointing under the pin; re-measure kw if the scale moves', color=GOLD),
        dict(label='3 · N-G1', sub='run_145 + run_149 against is2_v2', color=GOLD),
        dict(label='4 · Submit', sub='15,748 jobs, ledger, then stage 3', color=GOLD)],
        dark=True, size=28)
    lim = (sd.p('What this does not rule out', 34, DINK, 600)
           + sd.p('• <b>Chamber D on beam:</b> a ~360 ns oscillation on 82–90 % of triggers means its beam v and angles are not measurable. A hum flag or sinusoid subtraction is untested.', 26, DMUT)
           + sd.p('• <b>Chamber B:</b> positions only; no angle.', 26, DMUT)
           + sd.p('• <b>Chamber C:</b> its angle scale moves ~1.5 % between sub-runs; a per-sub-run v is open.', 26, DMUT)
           + sd.p('• Q7 and the pin were scored on A–C cosmics and run_145 overlays; other beam runs are checked only at N-G1.', 26, DMUT))
    body = (sd.kicker('Next steps') + '<h2 style="font-size:60px;font-weight:600;line-height:1.1">Four steps to the full pass</h2>'
            + nxt + '<div style="flex:1"></div>' + lim)
    D.slide('next', body, '''
<p>Not blocking: the ftst slope on beam (−7 vs −10 ns/class), the per-sample saturation rail, and C's sub-run variation. One Q7 acquisition (run_145 <code>_006</code> C, column window) is still running as condor 4617508; it cannot change the decision.</p>
<p>Commands: <code>ntof_bundle.py</code> for the bundles, <code>make_ntof_jobs.py</code> then <code>condor_submit ntof.sub</code> for the pass. Archive the code first and run <code>lxstore status</code> before and after.</p>''',
            dark=True, short='Next')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.environ.get('WORK_DATA', '/tmp'), 'ntof_progress_deck', 'ntof-pass1-progress.html'))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    D = sd.Deck('n_TOF pass 1: progress and next steps', 'Status of the n_TOF beam data full reprocessing, 5 Nov 2026')
    cover(D); pipeline(D); t0_slide(D); q7_slide(D); bundles_slide(D); last(D)
    D.write(a.out, note_meta=dict(
        title='n_TOF pass 1: progress and next steps',
        summary='Every input of the n_TOF reprocessing is ready and every calibration choice is made; bundles and submission are next.',
        tags=['x17', 'ntof', 'reconstruction'], date='2026-11-05'))
    print(a.out)


if __name__ == '__main__':
    main()
