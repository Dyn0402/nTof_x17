#!/usr/bin/env python3
"""collect_figures.py -- gather every figure that might serve Paper II into papers/paper2/collection/.

Layout: collection/<section>/<file>, plus collection/INDEX.md and collection/index.json (the catalogue the
slide note, make_collection_deck.py, reads). Re-run after adding rows to ITEMS; copies are overwritten.

Rating: A paper-ready or near, B usable but needs a remake (old model / old pass / wrong framing), C context only.
Basis: pass1 | r06 | rnd (July R&D model, pre-audit) | hits (superseded for position/angle) | toy | schematic.
Items with src=None are gaps: figures the paper needs and nobody has made.

Run:  .venv/bin/python papers/paper2/collect_figures.py
"""
import base64
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / 'collection'
NOTES = Path.home() / 'PycharmProjects/dylan-cern-site/notes'
WF = REPO / 'mx_june_cosmic_qa/waveform_first_threading'
SC = REPO / 'mx_june_cosmic_qa/stephan_cosmics/figures'
R = REPO / 'rerun'
SPSA = REPO / 'sps_beam_test_26/analysis'

SECTIONS = {
    '1_kernel': 'Signal formation: the sharing kernel',
    '2_hits_fail': 'Why hit times fail',
    '3_model': 'The forward model and its calibration',
    '4_floor': 'The physics floor',
    '5_hits_vs_waveforms': 'Hits against waveforms, same events',
    '6_readout': 'What a hit readout must record',
    '7_known_angle': 'A known angle in the beam',
    '8_fleet': 'Fleet performance',
    '0_context': 'Context: detector, displays, supporting studies',
    '9_gaps': 'Gaps: figures Paper II needs that do not exist yet',
}

# (section, name, src, caption, basis, rating, producer)
ITEMS = [
    # ---- 1 kernel
    ('1_kernel', 'kernel_forms', R / 'paper1_spsfigs/figures/kernel_forms.pdf',
     'Head-on kernel, SPS run71 RAW (no zero suppression), X and Y: ±1..3 responses and time forms at 96/156/243 V/cm. ±1/lead 0.377 X, 0.390 Y; field-independent.',
     'pass1 2026-11-01', 'A', 'rerun/paper1_spsfigs/make_figs.py --only kernel_forms'),
    ('1_kernel', 'angled_kernel', R / 'paper1_spsfigs/figures/angled_kernel.pdf',
     'Kernel survives the 25.4° rotation (run63 flat700 vs rot25): X identical, Y carries the rotation.',
     'pass1 2026-11-01', 'A', 'rerun/paper1_spsfigs/make_figs.py --only angled_kernel'),
    ('1_kernel', 'sharing_schematic', SC / '03-charge-sharing-schematic.pdf',
     'Three-panel schematic: sharing cartoon, why a deep strip fires early, the S-shaped ladder distortion. Cartoon shows c1≈0.5, c2 0.05–0.15: check against measured c2<c1.',
     'schematic 2026-07-12', 'B', 'mx_june_cosmic_qa/engineer_package'),
    ('1_kernel', 'modelfree_ratio', Path('/media/dylan/data/x17/cosmic_bench/plane_ratio/figures/modelfree_ratio.png'),
     'Model-free ±2/±1 charge ratio by plane, all five chambers: X ≈ 0.15–0.19, Y 0.36–0.54.',
     'r06-era cosmics, Oct 2026', 'B', 'sps_beam_test_26/analysis/plane_ratio/make_report.py'),
    ('1_kernel', 'angled_stacks', SPSA / 'angled_kernel/figures/stacks.png',
     'Time stacks of centre/±1/±2 of the lead strip, X and Y, flat 233 V/cm vs 25.6° 142 V/cm.',
     'ZS 2026-08-18', 'B', 'sps_beam_test_26/analysis/angled_kernel/make_figures.py'),
    ('1_kernel', 'angled_ladder_bars', SPSA / 'angled_kernel/figures/geometry.png',
     'Y view, flat vs rotated: ±2/±1 0.062 → 0.28–0.30, cluster extent 3 → 6–8 strips, delay asymmetry −8 → −104…−120 ns. (File is named geometry.png but is NOT a geometry schematic.)',
     'ZS 2026-08-18', 'B', 'angled_kernel/make_figures.py'),
    ('1_kernel', 'angled_width', SPSA / 'angled_kernel/figures/width.png',
     'Lateral rms vs drift depth at four fields: field-independent where diffusion would scale as E^-1/2.',
     'ZS 2026-08-18', 'B', 'angled_kernel/make_figures.py'),
    ('1_kernel', 'det3_charge_sharing_hits', SC / '18-det3A-charge-sharing-measured.pdf',
     'Old hits-amplitude kernel on det3 vertical tracks (X 0.449/0.055, Y 0.519/0.152). Superseded by kernel_forms; kept as the "before".',
     'hits 2026-07-18, superseded', 'C', 'engineer_package/make_hybrid_figures.py'),
    ('1_kernel', 'share_kernels_mpgd26', REPO / 'papers/mpgd26_proceedings/figures/share_kernels.png',
     'Production kernels per plane from the det3 bundle with the resistive-layer inset. Layout good, numbers are the R&D-era bundle.',
     'rnd ~2026-08-17', 'C', 'mpgd26/make_share.py --only kernels'),
    # ---- 2 hits fail
    ('2_hits_fail', 'estimator_residual_curves', 'WF:001',
     'Per-strip time residual vs reference drift time, x and y, for matched-filter t50 and leading edge 20 %: +40 ns late at the mesh, −100…−130 ns early at the cathode. The ladder compression, estimator-independent. det3.',
     'rnd 2026-07-25 (hits timing vs M3)', 'A-', 'waveform_first_threading/05_estimator_compare.py: REMAKE on pass 1, all chambers (manifest: ladder_compression)'),
    ('2_hits_fail', 'event_waveforms', SC.parent / 'event_displays/event_display_waveforms.pdf',
     'Raw strip-waveform matrices (X and Y) of one det3 cosmic plus three single-strip pulses: later when deeper.',
     'waveform 2026-07-18 display', 'A-', 'engineer_package/make_event_displays.py'),
    ('2_hits_fail', 'event_raw_x_y', 'WF:000',
     'One det3 muon as strip × sample matrices, X and Y, with the M3 reference track dashed.',
     'rnd 2026-07-26', 'B', 'waveform_first_threading/02_explore.py'),
    ('2_hits_fail', 'unsharing_depth_bias', SC / '09-det3A-unsharing-depth-bias.pdf',
     'Signed hit-to-reference distance vs depth, 500 det3 muons: production ladder (+19 µm/mm) vs unshared (+2 µm/mm). The hits ladder fans out with depth.',
     'hits 2026-07-18', 'A-', 'engineer_package/make_hybrid_figures.py'),
    ('2_hits_fail', 'angle_bias_schematic', SC / '04-angle-bias-schematic.pdf',
     'Why a hit-time slope is v(tanθ ± w/z_vis): cluster width = z tanθ + w. The z_vis logic is superseded by the column-duration view.',
     'schematic 2026-07-12', 'B', 'engineer_package'),
    ('2_hits_fail', 'event_3d_hits_vs_m3', SC / '07-det3A-event-3d-display.pdf',
     '3-D event display: X and Y hits against the M3 reference track (±0.2 mm band).',
     'hits 2026-07-18', 'B', 'engineer_package/make_event_displays_3d.py'),
    ('2_hits_fail', 'forward_fit_walkthrough', 'NOTE:forward-fit-det3',
     'One muon through the fit step by step: slices, kernel, five waveforms, NNLS charge solve, "does it work" (10 images from the published note).',
     'rnd Sept 2026', 'B', 'forward-fit-det3 make_figures.py'),
    # ---- 3 model
    ('3_model', 'mixing_diagram', None, None, None, None, None),  # placeholder removed below
    ('3_model', 'templates_perplane', 'WF:002',
     'Measured impulse templates, X vs Y, and the Y−X difference (Y undershoot 4× deeper).',
     'rnd 2026-07-26', 'A-', 'waveform_first_threading/11_templates_perplane.py'),
    ('3_model', 'event_fit', 'WF:003',
     'One X-plane event: data, forward-model fit and residual in σ (fit 10.9° vs M3 10.4°).',
     'rnd 2026-07-26', 'A-', 'waveform_first_threading/10_display_calibrated.py'),
    ('3_model', 'vtension_resolution', 'WF:005',
     'Profiled χ²(v) and the resolution of the v tension: 36.7 (fit) vs 34 (geometry).',
     'rnd 2026-07-26', 'B', 'waveform_first_threading/27_vtension_figure.py'),
    ('3_model', 'share_decompose', REPO / 'papers/mpgd26_proceedings/figures/share_decompose.png',
     'Four consecutive Y strips of one muon: fitted waveform split into own / ±1 / ±2 contributions vs data (20–42 % from neighbours). Best "what the model does" picture; remake on pass 1.',
     'rnd ~2026-08-17', 'B', 'mpgd26/make_share.py --only decompose'),
    ('3_model', 'bow_profiles', R / 'paper1_res/figures/bow_profiles.pdf',
     'Chamber bow estimator on pass-1 golden rows (calibration-constant structure the model must carry).',
     'pass1 2026-11-04', 'B', 'rerun/paper1_res/bow_pass1.py'),
    ('3_model', 'bow_closure', R / 'paper1_res/figures/bow_closure.pdf',
     'Bow closure on pass-1 golden rows.', 'pass1 2026-11-04', 'B', 'rerun/paper1_res/bow_pass1.py'),
    ('3_model', 'model_slides_diagrams', 'NOTE:mx17-detector-model-slides',
     'NOT copied (inline SVG): slides "From a track to the strip waveforms in six steps", "Drift", "Y: RC line", "Electronics template". Extract from cloud_basics/make_model_deck.py when redrawing the forward-model diagram.',
     'schematic 2026-10-21', 'A (as diagrams)', 'sps_beam_test_26/analysis/cloud_basics/make_model_deck.py'),
    ('3_model', 'v_vs_E_gas', 'WF:006',
     'det3 forward-fit v(E) against refined Magboltz grids (Paper I gas topic; shows v comes from the fit, not a prior).',
     'rnd 2026-07-26', 'C', 'waveform_first_threading/20_vE_figure.py'),
    # ---- 4 floor
    ('4_floor', 'floor_vs_angle_det3', R / 'model_audit/ws_floor/figures/floor_vs_angle_det3.png',
     'det3 s68 vs angle: information limit 0.18°, noise-only 0.01°, perfect model 0.34°, 60 ns bins degrade past 20°, 20 ns flat; measured ~1° on top.',
     'toy 2026-10-27, bench', 'A', 'rerun/model_audit/ws_floor/make_figure.py'),
    ('4_floor', 'floor_breakdown_det3', R / 'model_audit/ws_floor/figures/floor_breakdown_det3.png',
     'One-effect-off breakdown at 5/20/35°: without diffusion 0.02–0.09°, so diffusion is the floor; noise negligible.',
     'toy 2026-10-27', 'A', 'ws_floor/make_figure.py'),
    ('4_floor', 'floor_fleet', R / 'model_audit/ws_floor/figures/floor_fleet.png',
     'Floor vs measured X/Y for all five chambers (measured = pre-pass-1 audit tables).',
     'toy 2026-10-27', 'B', 'ws_floor/make_figure.py'),
    ('4_floor', 'floor_bias_grid_det3', R / 'model_audit/ws_floor/figures/floor_bias_grid_det3.png',
     'The 60 ns charge grid reads steep tracks too shallow (−0.23/−0.51° at 30/40°); 20 ns is flat.',
     'toy 2026-10-27', 'A', 'ws_floor/make_figure.py'),
    ('4_floor', 'angle_floor_deck', 'NOTE:mx17-angle-floor',
     'NOT copied (inline SVG): the 60 ns jump mechanism, realism ladder (31–46 % of measured), "47–75 % of the variance is in nothing we simulated". Producer: rerun/model_audit/followups/fd_floor_realism/make_floor_deck.py.',
     'toy + pass1 refresh 2026-11-03', 'B', 'fd_floor_realism/make_floor_deck.py'),
    # ---- 5 tiers
    ('5_hits_vs_waveforms', 'fleet_H0_vs_W', R / 'paper1_floor/figures/fleet.pdf',
     'Five chambers, same interior_good events: H0 hits vs W waveforms. Angle s68 (H0 2.0–2.6° vs W 0.73–1.56°), position core σ, efficiency within 5 mm. The only H0-vs-W figure on pass 1.',
     'pass1 2026-10-31', 'A', 'rerun/paper1_floor/hits_arm.py'),
    ('5_hits_vs_waveforms', 'three_way_comparison', 'WF:004',
     'Per-event angle residual histograms and cumulative max line deviation over 29 mm, same 2000 det3 events: production ladder vs SOTA hybrid vs forward fit. July prototype of the tier table.',
     'rnd 2026-07-26', 'A (flagship layout; remake with the H0…W tiers on pass 1)', 'waveform_first_threading/09_three_way.py'),
    ('5_hits_vs_waveforms', 'hybrid_angle_resolution', SC / '16-det3A-hybrid-angular-resolution-vs-angle.pdf',
     'Hybrid (hits + unsharing) angular resolution vs angle and its head-on blind spot. SUPERSEDED; baseline for H1/H2 only.',
     'hits 2026-07-18, superseded', 'C', 'engineer_package/make_hybrid_figures.py'),
    ('5_hits_vs_waveforms', 'tier_diagram_note', 'NOTE:mx17-detector-paper-status',
     'NOT copied (inline SVG): slide "Four hit tiers against the waveform fit, on the same events" (H0/H1/H2a/H2b/H2c/W/V diagram). Reusable diagram; bars are July numbers.',
     'rnd 2026-07', 'B', 'mx_june_cosmic_qa/make_paper_plan_deck.py'),
    # ---- 6 readout: nothing
    # ---- 7 known angle
    ('7_known_angle', 'known_angle', R / 'paper1_spsfigs/figures/known_angle.pdf',
     'det4 at 25.4°: inclined-view |tan| against the stage angle, three v treatments. 0.981 ± 0.03 of the stage tan with the column-duration v (13 rows), 0.927 with the registry v.',
     'pass1 2026-11-01', 'A', 'rerun/paper1_spsfigs/make_figs.py --only known_angle'),
    ('7_known_angle', 'sps_resolution', R / 'paper1_spsfigs/figures/sps_resolution.pdf',
     'det4 SPS residuals X/Y, 3-zone decomposition (Paper I figure; an intrinsic-resolution anchor for the telescope-free comparison).',
     'pass1 2026-11-01', 'B', 'rerun/paper1_spsfigs/make_figs.py --only sps_resolution'),
    # ---- 8 fleet
    ('8_fleet', 'tracking_perf_all', R / 'paper1_res/figures/tracking_perf_all.pdf',
     'σθ vs |θ_ref| per chamber, X solid/Y dashed: det3 0.87–1.0° flat to 20°, det2 1.0–1.15°, det4 up to 2.9°.',
     'pass1 2026-11-04', 'A', 'rerun/paper1_res/angle_resolution.py'),
    ('8_fleet', 'angle_resolution', R / 'paper1_res/figures/angle_resolution.pdf',
     'Angle resolution per calibration row.', 'pass1 2026-11-04', 'B', 'rerun/paper1_res/angle_resolution.py'),
    ('8_fleet', 'bench_position_res', R / 'paper1_floor/figures/bench_position_res.pdf',
     'Bench position σ vs M3: core σ vs reference χ² and vs assumed pointing.', 'pass1 2026-11-04', 'B', 'rerun/paper1_floor/position_ms_fig.py'),
    # ---- 0 context
    ('0_context', 'microtpc_principle', SC / '02-microtpc-principle-schematic.pdf',
     'Micro-TPC principle schematic (also Paper I).', 'schematic 2026-07-12', 'B', 'engineer_package'),
    ('0_context', 'det3_loss_displays', REPO / 'papers/paper1/figures/loss_gallery.pdf',
     'Pass-1 waveform event displays by loss class (Paper I appendix): how to read a strip × time display; seeder failure classes.',
     'pass1 2026-10-31', 'B', 'rerun/paper1_figs/a1_displays.py'),
    ('0_context', 'xwidth_note', 'NOTE:x-width-excess',
     'NOT copied (inline SVG, 15): the X-width excess: saturation, det3 remainder, closure toy. The model\'s main known mismatch.',
     'rnd 2026-10-17', 'C', 'sps_beam_test_26/analysis/cloud_basics/make_xwidth_deck.py'),
    # ---- 9 gaps (src None)
    ('9_gaps', 'ladder_compression_fleet', None,
     'Ladder compression on pass 1 for all five chambers, with CFD / peak / leading-edge / matched-filter estimators and the implied-v-vs-angle line. det3 July exists only inside the report PDF.',
     '', '', 'manifest placeholder ladder_compression; R1'),
    ('9_gaps', 'forward_model_diagram', None,
     'A forward-model diagram on the production (physical RC) model. mpgd26 share_cartoon/share_build exist only embedded in the talk note and use the R&D kernel.',
     '', '', 'mpgd26/make_share.py --only cartoon|build; cloud_basics model deck'),
    ('9_gaps', 'toy_floor_production', None,
     'Floor/toy closure figure in paper style on pass 1 (ws_floor is the content; the manifest note "electronics 0.02°, centroid jitter ~1°" was WITHDRAWN 2026-10-27).',
     '', '', 'rerun/model_audit/ws_floor'),
    ('9_gaps', 'tiers_H1_H2abc', None,
     'Tiers H1 (best per-strip estimator), H2a (slope remap), H2b (unsharing hybrid), H2c (forward fit on hit lists) on pass 1. Nothing computed; H2c is new work (R3).',
     '', '', 'R1–R3, R5'),
    ('9_gaps', 'v_flatness_bias_vs_angle', None,
     'Bias and implied-v vs angle, H0 vs W, on pass 1 (hits_arm.py has the H0 data; only s68, position, efficiency are plotted).',
     '', '', 'rerun/paper1_floor/hits_arm.py'),
    ('9_gaps', 'threshold_neighbours', None,
     'Angular resolution vs hit threshold (3–7σ) and ±1/±2 neighbour readout, amplitude on/off (R4). No data; VMM emulation (V1) deferred.',
     '', '', 'R4'),
    ('9_gaps', 'known_angle_geometry', None,
     'A real schematic of the rotated det4 in the beam. The manifest PNG is a Y-view bar chart. Mount angle quoted as 25.4/25.6/25.64° in different places.',
     '', '', 'mpgd26/scenes_sps.py is a candidate'),
    ('9_gaps', 'seeder_performance_pass1', None,
     'Seeder performance on pass 1 (the min-3 result and the 2-strip head-on rule exist only as note slides from 10-07 and 10-28).',
     '', '', 'two-track-limit and det3-loss-categories notes'),
    ('9_gaps', 'x_asymmetry_figure', None,
     'X asymmetry (the symmetric model\'s main known mismatch): measured one-sided ±1 copy 0.13×, 114 ns late; cost −0.05° overall, ∓0.19° head-on. Numbers only (wft/xasym_maps.json).',
     '', '', 'cloud_basics/xasym_toy.py'),
]
ITEMS = [i for i in ITEMS if i[2] is not None or i[3] is not None]


def wf_images():
    d = OUT / '_extracted'
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(['pdfimages', '-png', str(WF / 'report/main.pdf'), str(d / 'r')], check=True)
    return {i: d / f'r-{2 * i:03d}.png' for i in range(10)}


def note_images(slug):
    h = (NOTES / f'{slug}.html').read_text()
    out = []
    for k, (ext, b64) in enumerate(re.findall(r'data:image/(png|jpeg);base64,([A-Za-z0-9+/=]+)', h)):
        out.append((ext if ext == 'png' else 'jpg', base64.b64decode(b64)))
    return out


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    wf = wf_images()
    cat = []
    for sec, name, src, cap, basis, rating, prod in ITEMS:
        rec = dict(section=sec, name=name, caption=cap, basis=basis, rating=rating, producer=prod, files=[], src=str(src))
        (OUT / sec).mkdir(exist_ok=True)
        if src is None:
            pass
        elif isinstance(src, str) and src.startswith('WF:'):
            s = wf[int(src[3:])]
            dst = OUT / sec / f'{name}.png'
            shutil.copy(s, dst)
            rec['files'].append(dst.name)
            rec['src'] = f'{(WF / "report/main.pdf").relative_to(REPO)} (embedded image {src[3:]}; standalone PNG no longer on disk)'
        elif isinstance(src, str) and src.startswith('NOTE:'):
            slug = src[5:]
            rec['src'] = f'published note {slug}'
            if name == 'forward_fit_walkthrough':
                for k, (ext, b) in enumerate(note_images(slug), 1):
                    dst = OUT / sec / f'{name}_{k:02d}.{ext}'
                    dst.write_bytes(b)
                    rec['files'].append(dst.name)
        else:
            if not Path(src).exists():
                print('MISSING', src, file=sys.stderr)
                rec['caption'] += ' [SOURCE FILE MISSING]'
            else:
                dst = OUT / sec / f'{name}{Path(src).suffix}'
                shutil.copy(src, dst)
                rec['files'].append(dst.name)
                try:
                    rec['src'] = str(Path(src).relative_to(REPO))
                except ValueError:
                    rec['src'] = str(src)
        cat.append(rec)
    shutil.rmtree(OUT / '_extracted', ignore_errors=True)
    (OUT / 'index.json').write_text(json.dumps(dict(sections=SECTIONS, items=cat), indent=1))
    L = ['# Paper II figure collection', '',
         'Generated by `papers/paper2/collect_figures.py`. Rating A paper-ready, B needs a remake, C context. '
         'Basis `rnd` = July R&D model (pre-audit), `hits` = superseded for position/angle.', '']
    for sec, title in sorted(SECTIONS.items()):
        rows = [c for c in cat if c['section'] == sec]
        L += [f'## {sec.split("_")[0]}. {title}', '']
        if not rows:
            L += ['*Nothing exists.*', '']
            continue
        L += ['| item | files | rating | basis | what it shows | source |', '|---|---|---|---|---|---|']
        for c in rows:
            L.append(f"| {c['name']} | {', '.join(c['files']) or '—'} | {c['rating']} | {c['basis']} | {c['caption']} | `{c['src']}` |")
        L.append('')
    (OUT / 'INDEX.md').write_text('\n'.join(L))
    n = sum(len(c['files']) for c in cat)
    print(f'{len(cat)} items, {n} files ->', OUT)


if __name__ == '__main__':
    main()
