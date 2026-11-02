#!/usr/bin/env python3
"""make_principle.py -- Paper I, the micro-TPC principle figure (fig.microtpc_principle).  Dylan 2026-11-02.

    .venv/bin/python papers/paper1/figs/make_principle.py --event 7884 \
        --bench <X17_BENCH_ROOT staging> --ana <Analysis_pass1 copy> --rays <paper1_eff budget rays_det3.parquet> \
        [--cache <dir>] [--out papers/paper1/figs/out/microtpc_principle]

(a) a geometry sketch: a track crossing the 30 mm drift gap, its ionisation drifting to the mesh, the avalanche charge
    spread by the resistive strips over +-1-2 readout strips (the RC sharing the forward model carries), the pads and
    the X / Y strips they are routed to;
(b) one real det3 cosmic muon of the golden row sat_det3 (pass 1), X view, as strip x drift time: the data window the
    reconstruction fits, and next to it the forward model at the pass-1 fit (the same window, the same colour scale),
    with the fitted track and the M3 reference drawn on the data.

The event is replayed through the production reconstruction exactly as wft.reco runs it (hits -> seeds -> windows ->
fit_plane_candidates with the bundle's t0 prior; the recipe of rerun/paper1_floor/boot_extract.py), so the model shown
is the pass's own model: the replay's w must equal the pass table's (printed; the run stops if it does not).  Nothing
here uses hit times for geometry (RECONSTRUCTION_BASIS.md).

Inputs (local copies of EOS, the X17_BENCH_ROOT layout of rerun/inputs.py):
  --bench  <root>/det3/mx17_det3_saturday_scan_6-27-26/{run_config.json, long_run_.../{decoded_root (rr1 fa80c06),
           combined_hits_root (rr1_cm0), m3_tracking_root_v2}}
  --ana    <root> holding mx17_det3_saturday_scan_6-27-26/long_run_.../mx17_3/wft/{events.parquet, calib_bundle}
  --rays   the effbudget rays of the row (M3 mesh crossing and tangent per event)
Event choice (7884): detected, |tan_x(M3)| = 0.176 (10.0 deg), |tan_y| < 0.01, residual < 0.05 mm in both views,
no dropped / saturated / band strips, one track, chi2 and charge within 15 % of the row's medians, 70 mm from every edge.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
KEY = 'sat_det3'
REL = ('mx17_det3_saturday_scan_6-27-26', 'long_run_resist_490V_drift_1000V', 'mx17_3')


def replay(event, bench, ana):
    """{view: dict(W, M, pos, ch, p0, w, t0, q, chi2, dof, w_table, censor)} for one event, from the production code."""
    os.environ['X17_BENCH_ROOT'] = bench
    os.environ['X17_ANALYSIS_ROOT'] = ana
    sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]
    import pandas as pd
    import qa_config
    from wft import io as wio, model as wm, reco, seed as wseed
    cfg = qa_config.RUNS[KEY]
    wdir = os.path.join(cfg.OUT_BASE, 'wft')
    reco._worker_init(os.path.join(wdir, 'calib_bundle'))
    cal = reco._CAL
    tab = pd.read_parquet(os.path.join(wdir, 'events.parquet'))
    tab = tab[tab.event_id == event]
    tw = {(str(r.acq_tag), p): float(getattr(r, f'{p}_w')) for r in tab.itertuples() for p in 'xy'}
    pos_maps = wio.strip_position_map(cfg)
    hits = reco._load_hits(cfg)
    hits, _pg = reco.pedestal_guard(hits, cfg)
    dead = (getattr(cal, 'dead', None) or {}) if wseed.DEAD_SEED else {}
    seeds = wseed.seeds_from_hits(hits, pos_maps, cfg.MX17_FEU_X, cfg.MX17_FEU_Y, hot=cal.hot, dead=dead)
    wanted = {event} & set(seeds)
    if wm.BAND_NSIG > 0:
        hits = hits.set_index('eventId')
        hits = hits[hits.index.isin(wanted)]
    else:
        hits = None
    out, evtime = {}, {}
    for payloads in reco._stream_windows(cfg, pos_maps, seeds, wanted, 3, verbose=False, dead=dead, hits=hits,
                                         hot=cal.hot, evtime=evtime):
        for eid, wins, sds, _nh, _sp, ftst in payloads:
            ftst = ftst if isinstance(ftst, dict) else {}
            tag = evtime.get(int(eid), (-1, ''))[1]
            for p in ('x', 'y'):
                if not wins.get(p):
                    continue
                tp = reco.t0_prior_for(cal, p, ftst.get(p))
                best, _r = reco.fit_plane_candidates(wins[p], p, cal, seeds=sds.get(p), return_all=True, t0_prior=tp)
                if best is None:
                    continue
                P = wins[p][best._win]
                Wg, nz, pos, st = wm.prep_plane(P, p)
                M = wm.model_waveforms(p, pos, best.p0, best.w, best.t0, best._q, cal.hyper)
                rec = dict(W=Wg, M=M, pos=pos, ch=np.asarray(P['ch'], int), p0=best.p0, w=best.w, t0=best.t0,
                           q=np.asarray(best._q), chi2=best.chi2, dof=best.dof, w_table=tw.get((tag, p)),
                           censor=np.asarray(st, bool), tag=tag, tan=best.tan_theta)
                if rec['w_table'] is not None and abs(rec['w'] - rec['w_table']) < 1e-6:
                    out[p] = rec                       # the acquisition the pass took the event from
                else:
                    out.setdefault(p, rec)
    for p, r in out.items():
        print(f'[replay] {p}: w {r["w"]:.6f} table {r["w_table"]}  p0 {r["p0"]:.2f}  t0 {r["t0"]:.1f}  '
              f'chi2/dof {r["chi2"]:.0f}/{r["dof"]}', flush=True)
    return dict(views=out, v_drift=float(cal.v_drift), meta=dict(event=event, key=KEY))


# --------------------------------------------------------------------------------------------------------------------
# (a) the sketch
# --------------------------------------------------------------------------------------------------------------------
def sketch(ax, ps, es):
    """Drift gap drawn at half scale (z [mm] -> z/2); the readout layers far thicker than they are (not to scale)."""
    from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon
    navy, acc, muted, ink = ps.DET_COLOR['det3'], ps.ACCENT, ps.MUTED, ps.INK
    S = 0.5                                                              # drift-gap drawing scale
    xl, xr = -7.0, 7.0
    ax.set_xlim(-8.6, 19.0)
    ax.set_ylim(-14.6, 18.0)
    ax.axis('off')
    fs = 7.0
    # drift volume, cathode, mesh
    ax.add_patch(Rectangle((xl, 0), xr - xl, 30 * S, facecolor='#f4f6fa', edgecolor='none', zorder=0))
    ax.plot([xl, xr], [30 * S, 30 * S], color=ink, lw=1.6)
    ax.text(xr + 0.4, 30 * S, 'cathode', va='center', fontsize=fs, color=ink)
    ax.plot([xl, xr], [0, 0], color=ink, lw=1.0, ls=(0, (1, 1.2)))
    ax.text(xr + 0.4, 0.75, 'mesh', va='center', fontsize=fs, color=ink)
    ax.annotate('', xy=(-7.6, 30 * S), xytext=(-7.6, 0), arrowprops=dict(arrowstyle='<->', color=muted, lw=0.8))
    ax.text(-7.85, 15 * S, 'drift gap 30 mm', rotation=90, ha='right', va='center', fontsize=fs, color=muted)
    # the track at 10 deg (true proportion in x, z drawn scaled)
    tan = np.tan(np.radians(10.0))
    x0 = -2.4
    zt = np.array([33.0, 0.0])
    ax.plot(x0 + tan * zt, zt * S, color=ink, lw=1.6, zorder=4)
    ax.text(x0 + tan * 33 + 0.4, 33 * S, 'muon', fontsize=fs, color=ink, va='center')
    rng = np.random.default_rng(3)
    for z in np.sort(rng.uniform(0.8, 29.4, 15)):
        x = x0 + tan * z
        ax.plot([x], [z * S], 'o', ms=2.6, color=acc, zorder=5)
        ax.add_patch(FancyArrowPatch((x, z * S - 0.35), (x, 0.25), arrowstyle='-|>', mutation_scale=5, lw=0.55,
                                     color=navy, alpha=0.55, zorder=3))
    ax.text(x0 + tan * 30 + 0.9, 13.6, 'ionisation drifts\nto the mesh at v:\n' r'$t = t_0 + z/v$',
            fontsize=fs, color=navy, va='top')
    # readout stack, top to bottom (schematic)
    rows = [(-1.1, 1.1, '#efe7f7', 'amplification gap, 150 µm'),
            (-2.1, 0.7, None, 'resistive strips (+HV)'),
            (-3.3, 0.9, '#f3e3b5', 'insulator (kapton)'),
            (-4.5, 0.8, None, 'pads, 0.78 mm pitch (L4)'),
            (-5.8, 0.8, None, 'Y strips (L5)'),
            (-7.1, 0.8, None, 'X strips (L6)')]
    for y, h, col, lab in rows:
        if col is not None:
            ax.add_patch(Rectangle((xl, y), xr - xl, h, facecolor=col, edgecolor='none', zorder=1))
        ax.text(xr + 0.4, y + h / 2, lab, va='center', fontsize=fs, color=ink)
    xm0, xm1 = x0, x0 + tan * 30
    ax.add_patch(Polygon([[xm0 - 0.15, 0], [xm1 + 0.15, 0], [xm1 + 0.45, -1.1], [xm0 - 0.45, -1.1]], closed=True,
                         facecolor=acc, alpha=0.4, edgecolor='none', zorder=2))
    for xs in np.arange(xl + 0.1, xr - 0.4, 0.8):
        ax.add_patch(Rectangle((xs, -2.1), 0.55, 0.7, facecolor='#555555', edgecolor='none', zorder=2))
    pads = np.arange(xl + 0.05, xr - 0.6, PITCH_MM)
    for xp in pads:
        ax.add_patch(Rectangle((xp, -4.5), 0.68, 0.8, facecolor='#b87333', edgecolor='none', zorder=2))
        ax.add_patch(Rectangle((xp + 0.12, -7.1), 0.44, 0.8, facecolor='#b87333', edgecolor='none', zorder=2))
    ax.add_patch(Rectangle((xl, -5.8), xr - xl, 0.8, facecolor='#d9a066', edgecolor='none', zorder=2))
    for i, xp in enumerate(pads[1::3]):
        ax.plot([xp + 0.34] * 2, [-4.5, -6.3 if i % 2 else -5.0], color='#7a4b20', lw=0.7, zorder=3)
    # charge spread: what each X strip sees (own + +-1 + +-2, the RC sharing of the forward model)
    xc = 0.5 * (xm0 + xm1)
    ic = int(np.argmin(np.abs(pads + 0.34 - xc)))
    base = -14.0
    for di, sh in {0: 1.0, 1: 0.29, -1: 0.29, 2: 0.05, -2: 0.05}.items():
        if 0 <= ic + di < len(pads):
            ax.add_patch(Rectangle((pads[ic + di] + 0.04, base), 0.6, 5.2 * sh, facecolor=navy, edgecolor='none'))
    ax.plot([xl, xr], [base, base], color=ink, lw=0.6)
    ax.annotate('', xy=(xc, base + 5.6), xytext=(xc, -7.3), arrowprops=dict(arrowstyle='-|>', color=muted, lw=0.6))
    ax.text(xr + 0.4, base + 2.6, 'signal per X strip:\nits own charge, plus\n29 % on ±1 (47 ns later)\n'
            'and 5 % on ±2, spread\nby the resistive strips', va='center', fontsize=6.4, color=ink)


PITCH_MM = 398.58 / 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--event', type=int, default=7884)
    ap.add_argument('--bench'); ap.add_argument('--ana'); ap.add_argument('--rays')
    ap.add_argument('--cache', default=os.path.join(os.environ.get('WORK_DATA', '/tmp'), 'paper1_principle'))
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'microtpc_principle'))
    ap.add_argument('--view', default='x')
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    cp = os.path.join(a.cache, f'principle_{a.event}.pkl')
    if not os.path.exists(cp):
        rep = replay(a.event, a.bench, a.ana)
        if a.rays:
            import pandas as pd
            R = pd.read_parquet(a.rays).drop_duplicates('event_id').set_index('event_id')
            rep['m3'] = {k: float(R.loc[a.event, k]) for k in ('mesh_x', 'tan_x', 'mesh_y', 'tan_y')}
        pickle.dump(rep, open(cp, 'wb'))
    rep = pickle.load(open(cp, 'rb'))
    r = rep['views'][a.view]
    if r['w_table'] is None or abs(r['w'] - r['w_table']) > 1e-6:
        sys.exit(f'replay does not reproduce the pass table: w {r["w"]} vs {r["w_table"]}')

    sys.path.insert(0, os.path.join(REPO, 'papers')); sys.path.insert(0, HERE)
    import paper_style as ps
    import evd_style as es
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ps.use()
    v_mm = rep['v_drift'] * 1e-3
    fig = plt.figure(figsize=(ps.FULL[0], 3.7))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.38], wspace=0.2, left=0.0, right=0.905, bottom=0.12,
                             top=0.93)
    axa = fig.add_subplot(outer[0])
    sketch(axa, ps, es)
    axa.set_title('(a)', loc='left', x=0.0)
    inner = outer[1].subgridspec(1, 2, wspace=0.07)
    axd = fig.add_subplot(inner[0]); axm = fig.add_subplot(inner[1], sharey=axd)
    cen = r['p0'] + r['w'] * 0.5 * 30.0 / v_mm
    zoom = 6.5
    vmax = max(float(np.percentile(r['W'], 99.5)), 60.0)
    pc, _ = es.cells(axd, r['W'], r['pos'], cen, zoom, vmax=vmax)
    es.cells(axm, r['M'], r['pos'], cen, zoom, vmax=vmax)
    m3 = rep.get('m3')
    for ax in (axd, axm):
        es.frame(ax, cen, zoom, a.view.upper())
        ax.set_ylim(120, 1620)              # time increasing upwards here, like the depth in (a)
        es.track(ax, r['p0'], r['w'], r['t0'], v_mm, es.FIT_KW, label='model fit')
    if m3 is not None:
        es.m3_track(axd, m3[f'mesh_{a.view}'], m3[f'tan_{a.view}'], r['t0'], v_mm, label='M3 track')
    axd.set_ylabel('drift time [ns]')
    axm.set_xlabel('')
    axd.yaxis.set_label_coords(-0.42, 0.5)
    axd.xaxis.set_label_coords(1.035, -0.085)        # one label centred under the pair
    plt.setp(axm.get_yticklabels(), visible=False)
    axd.set_title('(b) data', loc='left')
    axm.set_title('model at the fit', loc='left')
    axd.legend(loc='upper left', fontsize=6.3, frameon=True, framealpha=0.9, edgecolor='none', handlelength=2.2)
    cax = fig.add_axes([0.915, 0.12, 0.011, 0.81])
    cb = fig.colorbar(pc, cax=cax)
    cb.set_label('amplitude / window maximum', fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    tan_tab = None
    if a.ana:
        import pandas as pd
        T = pd.read_parquet(os.path.join(a.ana, *REL, 'wft', 'events.parquet'))
        T = T[(T.event_id == a.event) & (T.acq_tag == r['tag'])]
        tan_tab = float(T[f'{a.view}_tan_theta'].iloc[0]) if len(T) else None
    info = dict(event=a.event, view=a.view, tan_fit_table=tan_tab,
                theta_fit_deg=float(np.degrees(np.arctan(tan_tab))) if tan_tab is not None else None,
                p0=float(r['p0']), w=float(r['w']), t0=float(r['t0']), chi2=float(r['chi2']), dof=int(r['dof']),
                v_drift=rep['v_drift'], m3=m3,
                theta_m3_deg=float(np.degrees(np.arctan(m3[f'tan_{a.view}']))) if m3 else None, acq_tag=r['tag'])
    json.dump(info, open(a.out + '.json', 'w'), indent=1)
    import pandas as pd
    q = pd.DataFrame(dict(depth_bin=np.arange(len(r['q'])), q=r['q']))
    ps.save(fig, a.out, data=q)
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
