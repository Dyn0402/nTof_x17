#!/usr/bin/env python3
"""lx_sparks.py <key> --fe-dir DIR --bundle DIR --out DIR [--jobs 16] [--min-cand 30] [--n-ctl 400] [--n-wf 160]

GD (spark detection after the FE unification), the lxplus part, one golden row.  Run through rerun/stage_job.py with
--parts run_config,m3,hits,decoded on the code the FE gate ran (code_1f8defd), so that every fit here is the `fe` arm's.

1. per M3-matched event of the gate range (event ids <= the fe table's last), hit-level features per view from the
   seeding hits (combined_hits_root_rr1_cm0): raw count, count after the 10 % relative floor without / with the flat-hit
   drop (= budget 'mult' / the veto count), saturated hits, flat hits, max significance -> hitfeat.parquet
2. candidates = events whose floored count (either rule) exceeds --min-cand, plus --n-ctl random controls: waveform
   features per view from decoded_root (pedestal + block CM as wft.io.FeuReader): strips over SNR 6, strips under
   -10 sigma, strips on the raw ADC rail, band strips (wft.model.band_rows), the block common-mode step, the veto count
   after the band re-seed -> wffeat.parquet; a random subset of windows for displays -> wf_samples.npz
3. the candidates (and 200 ordinary events, an identity check against the fe table) reconstructed by the ONE
   algorithm with the spark veto off (seeds_from_hits spark_veto=None) -> nospark/events.parquet
Nothing is decided here; the scoring is gd_score.py on the desktop.
"""
import argparse, functools, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
if os.path.isdir(os.path.join(REPO, 'wft')):   # on lxplus the script sits outside the code dir: stage_job set the path
    sys.path[:0] = [REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]

RAIL = 4000          # raw ADC (12 bit, clip 4095): a sample at or above this is on the rail
STEP_ADC = 50.0      # a 64-channel block whose common mode moves by more than this between the first samples and its extreme


def hit_features(H, feu_x, feu_y, rel):
    """Per event and view: counts before/after the relative floor, with/without the flat-hit drop."""
    from wft.seed import apply_significance_floor
    flat = H.trunc_left.to_numpy() & H.trunc_right.to_numpy()
    out = {}
    fl_all = apply_significance_floor(H, rel)                 # budget 'mult' (no flat drop)
    fl_seed = apply_significance_floor(H[~flat], rel)          # the seeding / veto count
    for v, feu in (('x', feu_x), ('y', feu_y)):
        g = H[H.feu == feu].groupby('eventId')
        out[f'{v}_nraw'] = g.size()
        out[f'{v}_nsat'] = g.saturated.sum()
        out[f'{v}_nflat'] = H[(H.feu == feu) & flat].groupby('eventId').size()
        out[f'{v}_maxsig'] = g.significance.max()
        out[f'{v}_nfl'] = fl_all[fl_all.feu == feu].groupby('eventId').size()
        out[f'{v}_nseed'] = fl_seed[fl_seed.feu == feu].groupby('eventId').size()
    F = pd.DataFrame(out).fillna(0)
    for c in F.columns:
        if not c.endswith('maxsig'):
            F[c] = F[c].astype(int)
    F['mult'] = F.x_nfl + F.y_nfl
    F['n_hits'] = F.x_nseed + F.y_nseed
    F.index.name = 'event_id'
    return F


def raw_events(rdr, wanted):
    """(eventId, raw (ns, 512) float32) for the wanted events of one FeuReader (complete windows only)."""
    from wft.io import CHUNK
    idx = np.where(np.isin(rdr.event_ids, np.fromiter(wanted, dtype=np.int64)))[0]
    for lo in range(0, len(idx), CHUNK):
        block = idx[lo:lo + CHUNK]
        arr = rdr.tree.arrays(['eventId', 'amplitude'], entry_start=int(block[0]), entry_stop=int(block[-1]) + 1,
                              library='np')
        for i in block:
            a = arr['amplitude'][i - int(block[0])]
            if len(a) != rdr.n_sample * 512:
                continue
            yield int(arr['eventId'][i - int(block[0])]), a.reshape(-1, 512).astype(np.float32)


def wf_features(raw, ped, noise, pos):
    """One view of one event: raw (ns, 512) -> features + the CM-subtracted window (512, ns) as wft.io gives it."""
    from wft import model as wm
    from wft.io import CNS_BLOCK
    ns = raw.shape[0]; nblk = 512 // CNS_BLOCK
    sub = raw - ped[None, :]
    cms = np.median(sub.reshape(ns, nblk, CNS_BLOCK), axis=2)          # (ns, 8)
    W = (sub - np.repeat(cms, CNS_BLOCK, axis=1)).T                     # (512, ns)
    n = np.maximum(noise, 3.0)
    ok = np.isfinite(pos)
    snr = W.max(1) / n
    on = ok & (snr > 6)
    rail = ((raw >= RAIL).sum(0) >= 2) & ok
    base = cms[:3].mean(0)
    step = np.abs(cms - base[None, :]).max(0)                           # per block
    band = wm.band_rows(W, noise) & ok
    f = dict(n_on6=int(on.sum()), n_on20=int((ok & (snr > 20)).sum()), n_neg10=int((ok & (W.min(1) / n < -10)).sum()),
             n_rail=int(rail.sum()), n_band=int(band.sum()), cm_step_max=float(step.max()),
             n_blk_step=int((step > STEP_ADC).sum()), cm_mean_max=float(np.abs(cms - base[None, :]).mean(1).max()))
    # the extent of the fired region: a discharge covers a large contiguous part of the plane
    if on.any():
        p = np.sort(pos[on]); f['span_mm'] = float(p[-1] - p[0])
        g = np.diff(p); f['n_gaps12'] = int((g > 12).sum())
    else:
        f['span_mm'] = 0.0; f['n_gaps12'] = 0
    return f, W, band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('key'); ap.add_argument('--fe-dir', required=True); ap.add_argument('--bundle', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--jobs', type=int, default=16)
    ap.add_argument('--min-cand', type=int, default=30); ap.add_argument('--n-ctl', type=int, default=400)
    ap.add_argument('--n-wf', type=int, default=160); ap.add_argument('--n-id', type=int, default=200)
    ap.add_argument('--no-reco', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    from qa_config import get_config, setup_paths
    setup_paths()
    import uproot
    from wft import seed as wseed, reco as wr, io as wio
    from wft.calib import CalibrationBundle
    from wft.cli import matched_event_ids
    cfg = get_config(a.key)
    os.makedirs(a.out, exist_ok=True)
    E = pd.read_parquet(os.path.join(a.fe_dir, 'events.parquet'))
    emax = int(E.event_id.max())
    filt = {e for e in matched_event_ids(cfg) if e <= emax}
    feu_x, feu_y = cfg.MX17_FEU_X, cfg.MX17_FEU_Y
    files = sorted(f for f in os.listdir(cfg.combined_hits_dir) if f.endswith('.root') and '_datrun_' in f)
    cols = ['eventId', 'feu', 'channel', 'amplitude', 'significance', 'saturated', 'trunc_left', 'trunc_right']
    H = uproot.concatenate([f'{cfg.combined_hits_dir}{f}:hits' for f in files], expressions=cols, library='pd')
    H = H[H.feu.isin((feu_x, feu_y)) & H.eventId.isin(filt)].reset_index(drop=True)
    F = hit_features(H, feu_x, feu_y, wseed.SIG_REL_FLOOR)
    F = F.reindex(sorted(filt), fill_value=0)
    F.index.name = 'event_id'
    F['in_fe'] = F.index.isin(set(E.event_id))
    F.to_parquet(os.path.join(a.out, 'hitfeat.parquet'))
    print(f'[gd] {a.key}: {len(F):,} matched events <= {emax}; mult>50 {int((F.mult > 50).sum())}, '
          f'n_hits>50 {int((F.n_hits > 50).sum())}  ({time.time() - t0:.0f} s)', flush=True)

    rng = np.random.default_rng(17)
    cand = set(F.index[(F.mult > a.min_cand) | (F.n_hits > a.min_cand)].tolist())
    rest = np.array(sorted(set(F.index) - cand))
    ctl = set(rng.choice(rest, size=min(a.n_ctl, len(rest)), replace=False).tolist())
    idchk = set(rng.choice(sorted(set(rest) & set(E.event_id)), size=min(a.n_id, len(rest)), replace=False).tolist())
    want = cand | ctl
    pos_maps = wio.strip_position_map(cfg)
    cal = CalibrationBundle.load(a.bundle)
    dead = (getattr(cal, 'dead', None) or {}) if wseed.DEAD_SEED else {}
    flat = H.trunc_left.to_numpy() & H.trunc_right.to_numpy()
    Hs = H[~flat][['eventId', 'feu', 'channel', 'amplitude', 'significance']].set_index('eventId')
    Hs = Hs[Hs.index.isin(want)]
    pick = set(rng.choice(sorted(cand), size=min(a.n_wf, len(cand)), replace=False).tolist()) | \
        set(rng.choice(sorted(ctl), size=min(a.n_wf // 5, len(ctl)), replace=False).tolist())
    rows, samples = [], {}
    by_tag = {}
    for v, feu in (('x', feu_x), ('y', feu_y)):
        for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu):
            by_tag.setdefault(wio.file_tag(f), {})[v] = f
    for tag in sorted(by_tag):
        if len(by_tag[tag]) < 2:
            continue
        tag_x = by_tag[tag]['x']
        rd = {v: wio.FeuReader(f) for v, f in by_tag[tag].items()}
        got = {}
        for v, r in rd.items():
            w = want & set(r.event_ids.tolist())
            for eid, raw in raw_events(r, w):
                f, W, band = wf_features(raw, r.ped, r.noise, pos_maps[feu_x if v == 'x' else feu_y])
                got.setdefault(eid, {})[v] = (f, W, band)
        for eid, pl in got.items():
            if len(pl) < 2:
                continue
            row = dict(event_id=eid, cand=eid in cand)
            for v in 'xy':
                row.update({f'{v}_{k}': x for k, x in pl[v][0].items()})
            bch = {feu_x: set(np.flatnonzero(pl['x'][2]).tolist()), feu_y: set(np.flatnonzero(pl['y'][2]).tolist())}
            hev = Hs.loc[[eid]].reset_index() if eid in Hs.index else None
            rs = wr.band_reseed(eid, hev, bch, pos_maps, feu_x, feu_y, cal.hot, dead,
                                seeder=functools.partial(wseed.seeds_from_hits, spark_veto=None))
            row['n_hits_reseed'] = int(rs['n_hits']) if rs is not None else int(F.n_hits.get(eid, 0))
            row['band_hits'] = int(rs['band_reseeded']) if rs is not None else 0
            rows.append(row)
            if eid in pick:
                samples[eid] = (pl['x'][1].astype(np.float16), pl['y'][1].astype(np.float16))
        print(f'[gd]   {os.path.basename(tag_x)}: {len(got)} events ({time.time() - t0:.0f} s)', flush=True)
    WF = pd.DataFrame(rows).set_index('event_id').sort_index()
    WF.to_parquet(os.path.join(a.out, 'wffeat.parquet'))
    ids = sorted(samples)
    np.savez_compressed(os.path.join(a.out, 'wf_samples.npz'), event_id=np.array(ids),
                        x=np.stack([samples[e][0] for e in ids]), y=np.stack([samples[e][1] for e in ids]),
                        pos_x=pos_maps[feu_x], pos_y=pos_maps[feu_y])
    json.dump(dict(key=a.key, emax=emax, n_matched=len(F), n_cand=len(cand), n_ctl=len(ctl), n_idcheck=len(idchk),
                   min_cand=a.min_cand, rail=RAIL, step_adc=STEP_ADC), open(os.path.join(a.out, 'gd_meta.json'), 'w'))
    print(f'[gd] waveform features {len(WF):,} events ({time.time() - t0:.0f} s)', flush=True)
    if a.no_reco:
        return
    # 3. the candidates with the spark veto off (the rest of the algorithm, band re-seed included, unchanged)
    wseed.seeds_from_hits = functools.partial(wseed.seeds_from_hits, spark_veto=None)
    out = os.path.join(a.out, 'nospark', 'events.parquet')
    wr.reconstruct_run(cfg, cal, out, event_filter=cand | idchk, jobs=a.jobs, bundle_path=a.bundle)
    json.dump(sorted(int(e) for e in idchk), open(os.path.join(a.out, 'idcheck.json'), 'w'))
    print(f'[gd] done ({time.time() - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
