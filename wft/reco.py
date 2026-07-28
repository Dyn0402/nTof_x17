"""
Per-event reconstruction and the batch driver.

One row per event, one set of columns per plane. The row carries the fit
(position at the mesh, transverse speed, angle), its errors, the charge-profile
summary, and the quality flags — plus enough provenance to know which
calibration produced it.

Reference-free by construction: the seed comes from the detector's own hits
(``wft.seed``), the starting point from the brightest strip and the earliest
half-maximum crossing, and the slope from a wide scan. The M3 reference is
never an input to a fit — otherwise alignment and efficiency would be circular.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np

from .calib import CalibrationBundle
from . import model as wm

# ---------------------------------------------------------------- quality
TAN_MIN_SLOPE = 0.08       # below this the timing carries no slope information
FLOOR_TAN = 0.018          # ~1.0 deg: the measured per-event physics floor
FLOOR_P0_MM = 0.33         # measured charge-centroid jitter per 60 ns bin
CHI2DOF_BAD = 300.0        # showers / multi-track / spark

# slope search: +-0.021 mm/ns covers |tan| ~ 0.57 at v = 36.6 um/ns
W_SCAN_HALF = 0.021        # covers |tan| ~ 0.57 at v = 36.6 um/ns
W_SCAN_STEP = 0.0021
P0_SCAN_HALF = 2.5         # mm around the window's charge centroid
P0_SCAN_STEP = 0.5
T0_SCAN_HALF = 120.0
T0_SCAN_STEP = 40.0

RECO_COLUMNS = [
    'event_id', 'n_hits', 'spark',
    # per plane p in (x, y): p0, w, t0, tan, errors, chi2, dof, profile, flags
]


@dataclass
class PlaneFit:
    p0: float                # track position at the mesh [mm]
    w: float                 # transverse speed [mm/ns]; tan = w / v_drift
    t0: float                # arrival time of charge from the mesh [ns]
    tan_theta: float
    theta_deg: float
    chi2: float
    dof: int
    p0_err: float
    w_err: float
    tan_err: float
    q_sum: float             # total fitted charge
    q_u50: float             # median charge arrival time after t0 [ns]
    q_u90: float
    q_uend: float            # last depth bin above 5 % of the profile peak [ns]
    n_strips: int            # strips in the fit window
    n_seed: int              # strips in the seed cluster
    n_dropped: int
    slope_reliable: bool
    quality_ok: bool
    n_candidates: int = 1        # candidate clusters fitted for this plane


def _profile_summary(q: np.ndarray) -> tuple:
    """(total, median arrival, 90 % arrival, column end) from the NNLS charge
    profile, in ns after t0. Deliberately raw quantiles: the gap/column
    estimators that need a specific definition build it downstream."""
    q = np.asarray(q, float)
    tot = float(q.sum())
    if tot <= 0:
        return 0.0, np.nan, np.nan, np.nan
    u = wm.UK[:len(q)]
    c = np.cumsum(q) / tot
    u50 = float(np.interp(0.5, c, u))
    u90 = float(np.interp(0.9, c, u))
    live = np.where(q > 0.05 * q.max())[0]
    uend = float(u[live[-1]] + 0.5 * wm.DT) if len(live) else np.nan
    return tot, u50, u90, uend


def _errors(P, plane, r, hyper, dp=0.05, dw=2e-4) -> tuple:
    """1-sigma (p0, w) from the chi2 curvature, scaled by sqrt(chi2/dof) so that
    model imperfection is absorbed rather than ignored."""
    W, noise, pos, sat = wm.prep_plane(P, plane)

    def chi(p0v, wv):
        return wm.chi2_plane(plane, W, noise, pos, sat, p0v, wv, r['t0'],
                             hyper, snap_t0=False)[0]

    try:
        c0 = r['chi2']
        d2p = (chi(r['p0'] + dp, r['w']) - 2 * c0 + chi(r['p0'] - dp, r['w'])) / dp ** 2
        d2w = (chi(r['p0'], r['w'] + dw) - 2 * c0 + chi(r['p0'], r['w'] - dw)) / dw ** 2
        scale = max(r['chi2'] / max(r['dof'], 1), 1.0)
        ep = float(np.sqrt(2 * scale / d2p)) if d2p > 0 else np.nan
        ew = float(np.sqrt(2 * scale / d2w)) if d2w > 0 else np.nan
        return ep, ew
    except Exception:
        return np.nan, np.nan


def _global_start(P, plane, p0_seed, t0_seed, hyper):
    """Reference-free global search for the fit's starting point.

    The R&D fits were seeded at the M3 reference (position AND angle), which is
    not available in production and would make alignment/efficiency circular.
    Seeding instead from the brightest strip and a local search lands in the
    wrong basin for ~17 % of planes (measured against the reference-seeded fits:
    those failures sit at 5 deg error with a *higher* chi2, i.e. genuinely
    missed minima, not disagreements). The production hit ladder is no help
    either — it is compressed ~40 %, so an inclined track's seed starts outside
    the basin.

    So: scan. (p0, t0) first at zero slope, then (p0, w) at the best t0. ~310
    chi2 evaluations, all of them NNLS-profiled, ~0.4 s.
    """
    W, noise, pos, sat = wm.prep_plane(P, plane)

    def chi(p0, w, t0):
        return wm.chi2_plane(plane, W, noise, pos, sat, p0, w, t0, hyper)[0]

    # charge-weighted centre of the window as the p0 scan centre
    amp = np.maximum(W.max(axis=1), 0.0)
    p_c = float((pos * amp).sum() / amp.sum()) if amp.sum() > 0 else p0_seed
    p0s = p_c + np.arange(-P0_SCAN_HALF, P0_SCAN_HALF + 1e-9, P0_SCAN_STEP)
    t0s = np.arange(t0_seed - T0_SCAN_HALF, t0_seed + T0_SCAN_HALF + 1e-9,
                    T0_SCAN_STEP)

    best = (np.inf, p0_seed, t0_seed)
    for t0 in t0s:
        for p0 in p0s:
            c = chi(p0, 0.0, t0)
            if c < best[0]:
                best = (c, float(p0), float(t0))
    t0b = best[2]

    ws = np.arange(-W_SCAN_HALF, W_SCAN_HALF + 1e-9, W_SCAN_STEP)
    best2 = (np.inf, best[1], 0.0)
    for p0 in p0s:
        for w in ws:
            c = chi(p0, w, t0b)
            if c < best2[0]:
                best2 = (c, float(p0), float(w))
    return best2[1], best2[2], t0b


def fit_plane(P, plane: str, cal: CalibrationBundle, hyper: Optional[dict] = None,
              n_seed: int = 0, n_dropped: int = 0) -> Optional[PlaneFit]:
    """Fit one plane's window. P: dict/PlaneWindow-like with W, pos, noise, ch."""
    hyper = hyper or cal.hyper
    W = np.asarray(P['W'])
    if W.shape[1] != wm.NSAMP:
        wm.set_nsamp(W.shape[1])
    p0_seed, _w0, t0_seed = wm.init_guess(P, plane)
    p0_seed, w_seed, t0_seed = _global_start(P, plane, p0_seed, t0_seed, hyper)
    r = wm.fit_plane_raw(P, plane, p0_seed, w_seed, t0_seed, hyper=hyper)
    if r is None or not np.isfinite(r['chi2']):
        return None
    tan = r['w'] * 1e3 / cal.v_drift
    ep, ew = _errors(P, plane, r, hyper)
    q_sum, q_u50, q_u90, q_uend = _profile_summary(r['q'])
    return PlaneFit(
        p0=float(r['p0']), w=float(r['w']), t0=float(r['t0']),
        tan_theta=float(tan), theta_deg=float(np.degrees(np.arctan(tan))),
        chi2=float(r['chi2']), dof=int(r['dof']),
        p0_err=float(np.hypot(ep, FLOOR_P0_MM)) if np.isfinite(ep) else FLOOR_P0_MM,
        w_err=float(ew) if np.isfinite(ew) else np.nan,
        tan_err=float(np.hypot(ew * 1e3 / cal.v_drift, FLOOR_TAN))
        if np.isfinite(ew) else FLOOR_TAN,
        q_sum=q_sum, q_u50=q_u50, q_u90=q_u90, q_uend=q_uend,
        n_strips=int(W.shape[0]), n_seed=int(n_seed), n_dropped=int(n_dropped),
        slope_reliable=bool(abs(tan) >= TAN_MIN_SLOPE),
        quality_ok=bool(r['chi2'] / max(r['dof'], 1) < CHI2DOF_BAD))


# --- candidate-cluster selection -------------------------------------------
# A track's charge column crosses the drift gap, so it lasts a few hundred ns
# and its transverse speed is bounded. Coherent noise and stray deposits do not
# satisfy both. Among candidates that do, take the one whose charge the model
# explains best (chi2 improvement over "no signal").
U_MIN_NS = 250.0
U_MAX_NS = 1100.0
TAN_MAX = 0.6


def _candidate_score(P, plane, fit: PlaneFit) -> tuple:
    """(plausible, dchi2) for one candidate cluster's fit."""
    W, noise, pos, sat = wm.prep_plane(P, plane)
    chi_null = float(((W / noise[:, None]) ** 2)[~sat].sum())
    u = fit.q_uend
    plausible = (np.isfinite(u) and U_MIN_NS <= u <= U_MAX_NS
                 and abs(fit.tan_theta) < TAN_MAX)
    return bool(plausible), float(chi_null - fit.chi2)


def fit_plane_candidates(windows: list, plane: str, cal: CalibrationBundle,
                         seeds: Optional[list] = None, return_all: bool = False):
    """Fit every candidate cluster of one plane and keep the muon's.

    'Largest cluster wins' is wrong for ~5 % of events, and when it is wrong the
    true track is a median 37 mm outside the fit window — so the failures are
    catastrophic, not marginal. Measured on those failures (det3, 224 events):

        rule                        median |p0 - ref|   within 5 mm
        most strips (old)                 76.6 mm            19 %
        most charge                       47.3 mm            32 %
        best chi2 improvement              3.4 mm            51 %
        plausible + best improvement       1.6 mm            55 %
        (best available candidate)         0.4 mm            95 %

    The right cluster is nearly always among the candidates; this rule finds it
    half the time, which is a large net gain and still leaves headroom.
    """
    best = None
    best_key = None
    n_ok = 0
    ranked = []
    for i, P in enumerate(windows):
        s = (seeds or [None] * len(windows))[i]
        try:
            fit = fit_plane(P, plane, cal,
                            n_seed=getattr(s, 'n_strips', 0) if s else 0,
                            n_dropped=getattr(s, 'n_dropped', 0) if s else 0)
        except Exception:
            fit = None
        if fit is None:
            continue
        n_ok += 1
        plausible, dchi2 = _candidate_score(P, plane, fit)
        fit._plausible, fit._dchi2 = plausible, dchi2
        key = (1 if plausible else 0, dchi2)
        ranked.append((key, fit))
        if best_key is None or key > best_key:
            best, best_key = fit, key
    if best is not None:
        best.n_candidates = n_ok
    ranked.sort(key=lambda kv: kv[0], reverse=True)
    for _k, f in ranked:
        f.n_candidates = n_ok
    return (best, [f for _k, f in ranked]) if return_all else best


DT_XY_TOL_NS = 120.0     # how far t0x - t0y may sit from the measured offset


def select_pair(cand_fits: Dict[str, list], ftst_diff: Optional[int],
                cal: CalibrationBundle) -> Dict[str, Optional[PlaneFit]]:
    """Choose one cluster per plane using the fact that the muon fired both.

    A muon's X and Y charge arrives at the same time, so ``t0x - t0y`` must sit
    at the measured FEU offset (``dt_xy``, keyed by the ftst difference).
    Coherent noise or a stray deposit in one plane has no reason to be
    time-coincident with the track in the other, which is information that
    single-plane selection cannot use.

    Falls back to the per-plane rule when a plane has only one candidate.
    """
    out = {}
    for plane in ('x', 'y'):
        fits = [f for f in cand_fits.get(plane, []) if f is not None]
        out[plane] = fits[0] if fits else None
    if not (len(cand_fits.get('x', [])) > 1 or len(cand_fits.get('y', [])) > 1):
        return out
    dt = cal.dt_xy.get(int(ftst_diff), -18.8) if ftst_diff is not None else -18.8
    best, best_key = None, None
    for fx in cand_fits.get('x', []) or [None]:
        for fy in cand_fits.get('y', []) or [None]:
            if fx is None or fy is None:
                continue
            coincident = abs((fx.t0 - fy.t0) - dt) <= DT_XY_TOL_NS
            plaus = int(getattr(fx, '_plausible', True)) + int(getattr(fy, '_plausible', True))
            key = (int(coincident), plaus,
                   getattr(fx, '_dchi2', 0.0) + getattr(fy, '_dchi2', 0.0))
            if best_key is None or key > best_key:
                best, best_key = (fx, fy), key
    if best is not None:
        out['x'], out['y'] = best
    return out


def fit_event(windows: Dict[str, object], cal: CalibrationBundle,
              seeds: Optional[dict] = None) -> Dict[str, Optional[PlaneFit]]:
    out = {}
    for plane in ('x', 'y'):
        P = windows.get(plane)
        if P is None:
            out[plane] = None
            continue
        s = (seeds or {}).get(plane)
        out[plane] = fit_plane(P, plane, cal,
                               n_seed=getattr(s, 'n_strips', 0) if s else 0,
                               n_dropped=getattr(s, 'n_dropped', 0) if s else 0)
    return out


def row_from_fits(event_id: int, fits: Dict[str, Optional[PlaneFit]],
                  n_hits: int = 0, spark: bool = False) -> dict:
    row = {'event_id': int(event_id), 'n_hits': int(n_hits), 'spark': bool(spark)}
    for plane in ('x', 'y'):
        f = fits.get(plane)
        if f is None:
            row[f'{plane}_ok'] = False
            for k in ('p0', 'w', 't0', 'tan_theta', 'theta_deg', 'chi2',
                      'p0_err', 'w_err', 'tan_err', 'q_sum', 'q_u50', 'q_u90',
                      'q_uend'):
                row[f'{plane}_{k}'] = np.nan
            for k in ('dof', 'n_strips', 'n_seed', 'n_dropped', 'n_candidates'):
                row[f'{plane}_{k}'] = 0
            row[f'{plane}_slope_reliable'] = False
            row[f'{plane}_quality_ok'] = False
        else:
            row[f'{plane}_ok'] = True
            for k, v in asdict(f).items():
                row[f'{plane}_{k}'] = v
    return row


# --------------------------------------------------------------- the driver
def _worker_init(bundle_path):
    global _CAL
    _CAL = CalibrationBundle.load(bundle_path)
    wm.use_calibration(_CAL)


PAIR_SELECT = os.environ.get('WFT_PAIR_SELECT', '0') == '1'


def _worker_fit(payload):
    eid, wins, seeds, n_hits, spark, ftst_diff = payload
    fits, all_fits = {}, {}
    for plane in ('x', 'y'):
        cand = wins.get(plane)
        if not cand:
            fits[plane], all_fits[plane] = None, []
            continue
        try:
            best, ranked = fit_plane_candidates(cand, plane, _CAL,
                                                seeds=seeds.get(plane),
                                                return_all=True)
            fits[plane], all_fits[plane] = best, ranked
        except Exception:
            fits[plane], all_fits[plane] = None, []
    if PAIR_SELECT:
        try:
            fits = select_pair(all_fits, ftst_diff, _CAL)
        except Exception:
            pass
    return row_from_fits(eid, fits, n_hits, spark)


def reconstruct_run(cfg, cal: CalibrationBundle, out_path: str,
                    event_filter: Optional[set] = None, jobs: int = 12,
                    limit: Optional[int] = None, pad_strips: int = 3,
                    bundle_path: Optional[str] = None, verbose: bool = True):
    """Reconstruct one run/subrun into a parquet table.

    cfg           : qa_config run config (paths, FEUs, detector name)
    event_filter  : if given, only these event ids (e.g. events with an M3 ray)
    """
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    from . import io as wio
    from . import seed as wseed

    if bundle_path is None:
        bundle_path = os.path.join(os.path.dirname(out_path), 'calib_bundle')
        cal.save(bundle_path)

    pos_maps = wio.strip_position_map(cfg)
    feu_x, feu_y = cfg.MX17_FEU_X, cfg.MX17_FEU_Y

    if verbose:
        print(f'[wft] {cal.summary()}')
        print(f'[wft] hits -> seeds ...', flush=True)
    hits = _load_hits(cfg)
    seeds = wseed.seeds_from_hits(hits, pos_maps, feu_x, feu_y)
    del hits
    wanted = set(seeds)
    if event_filter is not None:
        wanted &= set(int(e) for e in event_filter)
    wanted = {e for e in wanted
              if not seeds[e]['spark'] and (seeds[e]['x'] or seeds[e]['y'])}
    if limit:
        wanted = set(sorted(wanted)[:limit])
    if verbose:
        print(f'[wft] {len(seeds):,} seeded events, {len(wanted):,} to reconstruct',
              flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_init,
                             initargs=(bundle_path,)) as pool:
        for payloads in _stream_windows(cfg, pos_maps, seeds, wanted, pad_strips,
                                        verbose=verbose):
            if not payloads:
                continue
            rows.extend(pool.map(_worker_fit, payloads, chunksize=8))
            if verbose:
                print(f'[wft]   {len(rows):,} events reconstructed', flush=True)

    df = pd.DataFrame(rows).sort_values('event_id').reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    meta = dict(n_events=len(df), calibration=bundle_path,
                bundle=dict(detector=cal.detector, run_key=cal.run_key,
                            v_drift=cal.v_drift, hyper=cal.hyper,
                            conditions=cal.conditions,
                            provenance=cal.provenance),
                run=dict(key=getattr(cfg, 'KEY', ''), run=cfg.RUN,
                         sub_run=cfg.SUB_RUN, detector=cfg.DET_NAME,
                         feu_x=feu_x, feu_y=feu_y),
                selection=dict(sig_rel_floor=wseed.SIG_REL_FLOOR,
                               gap_mm=wseed.GAP_THRESHOLD_MM,
                               spark_veto=wseed.SPARK_VETO_HITS,
                               pad_strips=pad_strips,
                               event_filter=bool(event_filter)))
    with open(out_path.replace('.parquet', '.meta.json'), 'w') as f:
        json.dump(meta, f, indent=1, default=str)
    if verbose:
        print(f'[wft] wrote {out_path} ({len(df):,} events)')
    return df


def _load_hits(cfg):
    """Combined hits for the run — used ONLY for seeding (see wft.seed)."""
    import uproot
    import pandas as pd
    files = [f for f in os.listdir(cfg.combined_hits_dir)
             if f.endswith('.root') and '_datrun_' in f]
    df = uproot.concatenate(
        [f'{cfg.combined_hits_dir}{f}:hits' for f in files],
        expressions=['eventId', 'feu', 'channel', 'amplitude', 'significance'],
        library='pd')
    return df[df['feu'].isin(cfg.MX17_FEUS)]


def _stream_windows(cfg, pos_maps, seeds, wanted, pad_strips, verbose=True):
    """Yield lists of (eid, windows, seedinfo, n_hits, spark) per file pair."""
    from . import io as wio
    fx = wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, cfg.MX17_FEU_X)
    fy = wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, cfg.MX17_FEU_Y)
    by_tag = {}
    for f in fx:
        by_tag.setdefault(wio.file_tag(f), {})['x'] = f
    for f in fy:
        by_tag.setdefault(wio.file_tag(f), {})['y'] = f
    for tag in sorted(by_tag):
        pair = by_tag[tag]
        if 'x' not in pair or 'y' not in pair:
            if verbose:
                print(f'[wft]   {tag}: missing a plane, skipped')
            continue
        rx = wio.FeuReader(pair['x'])
        ry = wio.FeuReader(pair['y'])
        want = wanted & (set(rx.event_ids.tolist()) | set(ry.event_ids.tolist()))
        if not want:
            continue
        buf = {}
        for plane, rdr, feu in (('x', rx, cfg.MX17_FEU_X), ('y', ry, cfg.MX17_FEU_Y)):
            for eid, ftst, wfm in rdr.iter_events(want):
                cl = seeds[eid][plane]
                if not cl:
                    continue
                cl = cl if isinstance(cl, list) else [cl]
                wins, used = [], []
                for s in cl:
                    win = wio.extract_window(wfm, rdr.noise, pos_maps[feu],
                                             s.channels, pad_strips)
                    if win is None:
                        continue
                    wins.append(dict(W=win.W, pos=win.pos, noise=win.noise,
                                     ch=win.ch))
                    used.append(s)
                if not wins:
                    continue
                rec = buf.setdefault(eid, {'w': {}, 's': {}})
                rec['w'][plane] = wins
                rec['s'][plane] = used
                rec['ftst_' + plane] = ftst
        payloads = []
        for eid, rec in buf.items():
            fd = (rec['ftst_x'] - rec['ftst_y']
                  if 'ftst_x' in rec and 'ftst_y' in rec else None)
            payloads.append((eid, rec['w'], rec['s'], seeds[eid]['n_hits'],
                             seeds[eid]['spark'], fd))
        if verbose:
            print(f'[wft]   {tag}: {len(payloads):,} events windowed', flush=True)
        yield payloads
