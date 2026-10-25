#!/usr/bin/env python3
"""display.py — event displays for the outlier study.

    extract(det, eids)          -> {eid: {'x': W[512, ns], 'y': ..., 'ftst_x': .., 'ftst_y': ..}} from decoded_root
                                   (pedestal + plain CM, wft.io.FeuReader; full planes, channel order), cached in
                                   ~/x17/cosmic_bench/outliers/wf_<det>.pkl
    draw(det, row, wf, ax_row)  -> one event: per view a strip-position x time image of the WHOLE plane around the
                                   track (+-ZOOM mm), the M3 reference line and the fitted line in (position, time),
                                   and the strip-max profile over the full chamber (where else the plane has charge)
    python display.py <det> <eid> [<eid> ...] --out dir    (one PNG per event)
"""
import argparse, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, 'mx_june_cosmic_qa'), os.path.join(REPO, 'cosmic_bench_analysis')]
from olib import BASE, load  # noqa: E402

KEYS = {'det3': 'sat_det3', 'det4': 'g_det4', 'det7': 'g_det7_long', 'det6': 'g_det6_long', 'det2': 'o22_long_det2'}
V_BUNDLE = {'det2': 39.3, 'det3': 35.8, 'det4': 33.0, 'det6': 24.17, 'det7': 36.8}
TS = 60.0          # ns per sample (DREAM bench, wft.model.SNS)
ZOOM = float(os.environ.get("OL_ZOOM", 30.0))


def _cfg(det):
    from qa_config import get_config
    return get_config(KEYS[det])


def extract(det, eids):
    from wft import io as wio
    cache = os.path.join(BASE, f'wf_{det}.pkl')
    have = pickle.load(open(cache, 'rb')) if os.path.exists(cache) else {}
    want = set(int(e) for e in eids) - set(have)
    if want:
        cfg = _cfg(det)
        for v, feu in (('x', cfg.MX17_FEU_X), ('y', cfg.MX17_FEU_Y)):
            for f in wio.subrun_files(cfg.BASE_PATH, cfg.RUN, cfg.SUB_RUN, feu):
                r = wio.FeuReader(f)
                if not np.isin(r.event_ids, list(want)).any():
                    continue
                for eid, ft, W in r.iter_events(want):
                    e = have.setdefault(eid, {})
                    e[v] = W.astype(np.float16); e[f'ftst_{v}'] = ft; e[f'noise_{v}'] = r.noise.astype(np.float16)
        pickle.dump(have, open(cache, 'wb'))
    return {int(e): have[int(e)] for e in eids if int(e) in have}


_PMAP = {}


def posmap(det):
    if det not in _PMAP:
        from wft.io import strip_position_map
        cfg = _cfg(det); pm = strip_position_map(cfg)
        _PMAP[det] = {'x': pm[cfg.MX17_FEU_X], 'y': pm[cfg.MX17_FEU_Y]}
    return _PMAP[det]


def draw(det, row, wf, axes, title_extra=''):
    """axes: 2 x 2 (view x [image, profile])."""
    pm = posmap(det); v_d = V_BUNDLE[det] * 1e-3            # mm/ns
    for i, v in enumerate('xy'):
        axI, axP = axes[i]
        if v not in wf:
            axI.set_title(f'{v}: no waveform'); continue
        W = wf[v].astype(np.float32); pos = pm[v]; ok = np.isfinite(pos)
        o = np.argsort(np.where(ok, pos, np.inf))[:ok.sum()]
        P, Wv = pos[o], W[o]
        ns = Wv.shape[1]; t = np.arange(ns) * TS
        pref = row.get(f'ref_mesh_{v}', np.nan); p0 = row.get(f'{v}_p0', np.nan)
        cen = pref if np.isfinite(pref) else (p0 if np.isfinite(p0) else P[np.argmax(Wv.max(1))])
        sel = np.abs(P - cen) < ZOOM
        if sel.sum() > 1:
            vmax = max(np.percentile(Wv[sel], 99.5), 30)
            axI.pcolormesh(P[sel], t, Wv[sel].T, cmap='viridis', vmin=0, vmax=vmax, shading='nearest')
        t0 = row.get(f'{v}_t0', np.nan)
        tt = np.linspace(0, ns * TS, 50)
        if np.isfinite(t0):
            tz = np.clip(tt - t0, 0, None)
            if np.isfinite(pref):
                axI.plot(pref + row[f'ref_tan_{v}'] * v_d * tz, tt, 'r--', lw=1.5, label=f"M3 tan {row[f'ref_tan_{v}']:+.3f}")
            if np.isfinite(p0) and row.get(f'{v}_ok', False):
                axI.plot(p0 + row[f'{v}_w'] * tz, tt, 'w-', lw=1.2, label=f"fit tan {row.get(f'tl_{v}', np.nan):+.3f}")
            axI.axhline(t0, color='w', lw=0.5, ls=':')
            axI.axhline(t0 + 30.0 / v_d, color='w', lw=0.5, ls=':')
        axI.set_xlim(cen - ZOOM, cen + ZOOM); axI.set_ylim(ns * TS, 0)
        axI.set_xlabel(f'{v} strip position [mm]'); axI.set_ylabel('time [ns]')
        axI.legend(fontsize=7, loc='lower right')
        axI.set_title(f"{v}: dθ {row.get(f'dl_{v}', np.nan):+.1f}°  res {row.get(f'rres_{v}', np.nan):+.1f} mm  "
                      f"χ²/dof {row.get(f'chi2dof_{v}', np.nan):.0f}  q {row.get(f'{v}_q_sum', np.nan):.0f}  "
                      f"nsat {row.get(f'h{v}_nsat', 0):.0f}  ndrop {row.get(f'{v}_n_dropped', 0):.0f}", fontsize=8)
        prof = Wv.max(1)
        axP.plot(P, prof, 'k-', lw=0.6)
        axP.axvline(pref, color='r', ls='--', lw=1); axP.axvline(p0, color='b', lw=0.8)
        axP.axvspan(cen - ZOOM, cen + ZOOM, color='y', alpha=0.15)
        axP.set_xlim(0, 400); axP.set_xlabel(f'{v} [mm], full plane'); axP.set_ylabel('strip max [ADC]')
    axes[0][0].figure.suptitle(
        f"{det} event {int(row['event_id'])}   M3 χ² {row.get('m3_chi2x', np.nan):.2f}/{row.get('m3_chi2y', np.nan):.2f}"
        f"  ref mesh ({row.get('ref_mesh_x', np.nan):.0f}, {row.get('ref_mesh_y', np.nan):.0f}) mm  {title_extra}", fontsize=10)


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument('det'); ap.add_argument('eids', nargs='+', type=int); ap.add_argument('--out', required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    d = load(a.det).set_index('event_id', drop=False)
    wf = extract(a.det, a.eids)
    for e in a.eids:
        fig, axs = plt.subplots(2, 2, figsize=(13, 8), gridspec_kw=dict(width_ratios=[1.3, 1]))
        draw(a.det, d.loc[e].to_dict(), wf.get(e, {}), axs)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f'{a.det}_{e}.png'), dpi=80); plt.close(fig)


if __name__ == '__main__':
    main()
