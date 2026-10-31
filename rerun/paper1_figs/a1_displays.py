#!/usr/bin/env python3
"""a1_displays.py <key> --out DIR [--n 6] [--root PAPER1_EFF] [--ana ANALYSIS_PASS1] [--gen rr1] [--paper CLS=EID,...]
-- appendix A1 event displays on pass 1, any chamber (Paper I).  Read-only; runs on lxplus / condor (EOS through FUSE).

For one golden row: the per-ray classes of a1_recat.py (<root>/a1/<rel>/recat.parquet), the M3 rays of the budget
(<root>/budget/<rel>/rays_<det>.parquet), the pass's fits (<ana>/<rel>/wft/events.parquet) and the rr1 decoded
waveforms (registry eos_run_dir / subrun / decoded_root_fa80c06, read with wft.io.FeuReader: pedestal + CM exactly as
the reconstruction reads them).  Per class: n events (fixed seed 17; all of them if fewer), each as a strip-position x
drift-time display of both views (samples >= 4 % of the view's 99.5 % quantile, M3 extrapolated through the gap dashed,
the production fit solid) -> DIR/<det>/<class>.png (contact sheet, for the eyeball audit) and, per event, the numbers
that put it in its class -> DIR/<det>/picks.json.  --paper writes the chosen event of a class as a single print
figure DIR/<det>/paper_<class>.pdf.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [REPO, os.path.join(REPO, 'rerun'), os.path.join(REPO, 'mx_june_cosmic_qa'),
                os.path.join(REPO, 'cosmic_bench_analysis')]
import inputs  # noqa: E402

ROOT = '/eos/experiment/ntof/data/x17/cosmic_bench/x17rerun/paper1_eff'
ANA = '/eos/experiment/ntof/data/x17/cosmic_bench/Analysis_pass1'
SHORT = {'mx17_2': 'det2', 'mx17_3': 'det3', 'mx17_4': 'det4', 'mx17_6': 'det6', 'mx17_7': 'det7'}
TS = 60.0
S_SAT = 3550.0
ZOOM = {'band': 60, 'multi:m3_picked': 60, 'multi:other_picked': 45, 'multi:m3_not_found': 22, 'spark:discharge': 80,
        'spark:muon+other': 110, 'spark:shower': 45, 'far:DELTA': 18, 'far:FIT_SAT': 14, 'far:FIT': 16,
        'fit_failed': 10, 'far:EDGE': 20, 'far:POS_TAIL': 16, 'far:MULTI': 30, 'cut:other': 30}


def posmap(row):
    import qa_config
    from wft.io import strip_position_map
    key = row['key']
    if key not in qa_config.RUNS:
        qa_config.RUNS[key] = qa_config._Config(key, row['run'], row['subrun'], feus=[int(row['feu_x']), int(row['feu_y'])],
                                                det_z=float(row['det_z']), det_name=row['det'], zero_suppressed=False,
                                                base_path=row['eos_run_dir'].rsplit('/', 1)[0] + '/')
    cfg = qa_config.RUNS[key]

    class _Cfg:                       # the registered config, with the run_config read from EOS (not the desktop tree)
        run_config_path = os.path.join(row['eos_run_dir'], 'run_config.json')
        MAP_CSV_PATH, DET_NAME, MX17_FEU_X, MX17_FEU_Y = cfg.MAP_CSV_PATH, cfg.DET_NAME, cfg.MX17_FEU_X, cfg.MX17_FEU_Y
    pm = strip_position_map(_Cfg)
    return dict(x=pm[cfg.MX17_FEU_X], y=pm[cfg.MX17_FEU_Y]), cfg


def read_waveforms(row, gen, want):
    """{(acq_tag, eid): {'x': W, 'y': W}} for the wanted (acq_tag, eid) pairs: data files only (the pedestal files in
    decoded_root reuse event ids), each event from the acquisition the pass took it from (event ids restart per file)."""
    from wft import io as wio
    dec = inputs.GENERATIONS[gen]['decoded_root']
    out = {}
    for v, feu in (('x', int(row['feu_x'])), ('y', int(row['feu_y']))):
        fs = sorted(f for f in glob.glob(os.path.join(row['eos_run_dir'], row['subrun'], dec, f'*_{feu:02d}.root'))
                    if '_datrun_' in os.path.basename(f))
        for f in fs:
            tag = wio.file_tag(f)
            ids = {e for (t, e) in want if t == tag}
            if not ids:
                continue
            r = wio.FeuReader(f)
            have = ids & set(r.event_ids.tolist())
            for e, _ft, W in r.iter_events(have):
                out.setdefault((tag, int(e)), {})[v] = W.astype(np.float32)
            print(f'  {v} {os.path.basename(f)}: {len(have)} events', flush=True)
    return out


def panel(ax, W, P, cen, zoom, ref, fit, v_mm_ns, mark=False, tmax=1920):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    ok = np.isfinite(P); o = np.argsort(np.where(ok, P, np.inf))[:ok.sum()]; P, W = P[o], W[o]
    sel = np.abs(P - cen) < zoom
    vmax = max(float(np.percentile(W[sel], 99.5)) if sel.any() else 100.0, 60.0)
    ax.set_facecolor('#0b1020')
    rects, vals, clip = [], [], []
    for i in np.flatnonzero(sel):
        for k in range(W.shape[1]):
            val = float(W[i, k])
            if mark and val >= S_SAT:
                clip.append(Rectangle((P[i] - 0.389, k * TS), 0.778, TS))
            if val < 0.04 * vmax:
                continue
            rects.append(Rectangle((P[i] - 0.389, k * TS), 0.778, TS)); vals.append(min(val / vmax, 1.0))
    pc = PatchCollection(rects, cmap='viridis', linewidths=0); pc.set_array(np.array(vals)); pc.set_clim(0, 1)
    ax.add_collection(pc)
    if clip:
        ax.add_collection(PatchCollection(clip, facecolor='none', edgecolor='#ff4d4d', linewidths=0.6))
    mesh, tan, t0 = ref
    if np.isfinite(mesh):
        ts = np.array([t0, t0 + 30.0 / v_mm_ns])
        ax.plot(mesh + tan * v_mm_ns * (ts - t0), ts, color='#ffb000', lw=1.3, ls=(0, (5, 3)), label='M3')
    if fit is not None:
        p0, w, tf = fit
        ts = np.array([tf, tf + 30.0 / v_mm_ns])
        ax.plot(p0 + w * (ts - tf), ts, color='white', lw=1.1, label='fit')
    ax.set_xlim(cen - zoom, cen + zoom); ax.set_ylim(tmax, 0); ax.tick_params(labelsize=6)
    return pc


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument('key'); ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--root', default=ROOT); ap.add_argument('--ana', default=ANA); ap.add_argument('--gen', default='rr1')
    ap.add_argument('--paper', default='', help='CLS=EID,... single print figures')
    ap.add_argument('--paper-all', action='store_true', help='a print figure for every picked event')
    a = ap.parse_args()
    row = dict(inputs.row_for(a.key)); row['key'] = a.key
    det, short = row['det'], SHORT[row['det']]
    rel = os.path.join(row['run'], row['subrun'], det)
    C = pd.read_parquet(os.path.join(a.root, 'a1', rel, 'recat.parquet'))
    C = C.reset_index() if 'event_id' not in C.columns else C
    R = pd.read_parquet(os.path.join(a.root, 'budget', rel, f'rays_{short}.parquet')).drop_duplicates('event_id').set_index('event_id')
    E = pd.read_parquet(os.path.join(a.ana, rel, 'wft', 'events.parquet'))
    dup = E.event_id.duplicated(keep=False)
    E = E[~dup].set_index('event_id')
    B = json.load(open(os.path.join(a.ana, rel, 'wft', 'calib_bundle', 'bundle.json')))
    v_mm = float(B['v_drift']) * 1e-3
    rng = np.random.default_rng(17)
    picks = {}
    for cls, g in C[C.recat != 'detected'].groupby('recat'):
        ids = np.sort(g.event_id.values)
        ids = [int(i) for i in ids if i in E.index]          # the acquisition comes from the pass table
        picks[cls] = sorted(rng.choice(ids, size=min(a.n, len(ids)), replace=False).tolist()) if ids else []
    paper = dict(p.split('=') for p in a.paper.split(',') if p)
    for cls, e in paper.items():
        if int(e) not in picks.get(cls, []):
            picks.setdefault(cls, []).append(int(e))
    ACQ = {int(e): str(E.loc[e, 'acq_tag']) for es in picks.values() for e in es if e in E.index}
    want = {(ACQ[e], e) for es in picks.values() for e in es if e in ACQ}
    print(f'{a.key}: {sum(len(v) for v in picks.values())} events in {len(picks)} classes', flush=True)
    WF = read_waveforms(row, a.gen, want)
    PM, _ = posmap(row)
    od = os.path.join(a.out, short); os.makedirs(od, exist_ok=True)
    info = {}

    def draw(axx, axy, e, zoom):
        r = R.loc[e] if e in R.index else None
        f = E.loc[e] if e in E.index else None
        w = WF.get((ACQ.get(e), e), {})
        pcs = None
        for v, ax in (('x', axx), ('y', axy)):
            if v not in w:
                ax.text(0.5, 0.5, 'no waveform', transform=ax.transAxes, ha='center', color='w', fontsize=7); continue
            mesh = float(r[f'mesh_{v}']) if r is not None else np.nan
            tan = float(r[f'tan_{v}']) if r is not None else 0.0
            t0 = float(f[f'{v}_t0']) if f is not None and np.isfinite(f[f'{v}_t0']) else (257.5 if v == 'x' else 302.5)
            cen = mesh + 15 * tan if np.isfinite(mesh) else float(PM[v][np.nanargmax(w[v].max(1))])
            fit = (float(f[f'{v}_p0']), float(f[f'{v}_w']), float(f[f'{v}_t0'])) if f is not None and bool(f[f'{v}_ok']) else None
            pcs = panel(ax, w[v], PM[v].astype(float), cen, zoom, (mesh, tan, t0), fit, v_mm, mark=cls.endswith('SAT'))
            ax.set_xlabel(f'{v} [mm]', fontsize=6)
        return pcs

    for cls, es in picks.items():
        if not es:
            continue
        zoom = ZOOM.get(cls, 30)
        n = len(es)
        fig, axs = plt.subplots(2, n, figsize=(2.3 * n, 4.6), squeeze=False)
        for j, e in enumerate(es):
            draw(axs[0, j], axs[1, j], e, zoom)
            f = E.loc[e] if e in E.index else None
            r = R.loc[e] if e in R.index else None
            axs[0, j].set_title(f'{e}', fontsize=7)
            info.setdefault(cls, []).append(dict(
                event_id=e, spark=bool(f['spark']) if f is not None else None,
                x_ok=bool(f['x_ok']) if f is not None else None, y_ok=bool(f['y_ok']) if f is not None else None,
                x_nd=int(f['x_n_dropped']) if f is not None else None, y_nd=int(f['y_n_dropped']) if f is not None else None,
                n_hits=int(f['n_hits']) if f is not None else None,
                res_x=float(r['res_x']) if r is not None else None, res_y=float(r['res_y']) if r is not None else None))
        frac = 100.0 * len(C[C.recat == cls]) / len(C)
        fig.suptitle(f'{short} {cls}: {frac:.2f} % of interior rays ({(C.recat == cls).sum()} rays); top x, bottom y; '
                     'orange dashed M3, white fit', fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(od, cls.replace(':', '_').replace('+', '_') + '.png'), dpi=80)
        plt.close(fig)
        for e in (es if a.paper_all else ([int(paper[cls])] if cls in paper else [])):
            fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.75), sharey=True, constrained_layout=True)
            pc = draw(axs[0], axs[1], e, zoom)
            axs[0].set_ylabel('drift time [ns]')
            fig.suptitle(f'{short} event {e}: {cls}, {frac:.2f} % of interior M3 tracks', fontsize=9, x=0.02, ha='left')
            if pc is not None:
                cb = fig.colorbar(pc, ax=axs, shrink=0.9, pad=0.01); cb.set_label('ADC / view max', fontsize=8)
            axs[0].legend(fontsize=6.5, loc='lower left', framealpha=0.6)
            fig.savefig(os.path.join(od, f"paper_{cls.replace(':', '_').replace('+', '_')}_{e}.pdf")); plt.close(fig)
    json.dump(dict(key=a.key, picks=picks, info=info, n_interior=int(len(C)),
                   counts=C.recat.value_counts().to_dict()), open(os.path.join(od, 'picks.json'), 'w'), indent=1, default=float)
    print('done', od)


if __name__ == '__main__':
    main()
