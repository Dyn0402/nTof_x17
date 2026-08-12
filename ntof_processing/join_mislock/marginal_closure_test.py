#!/usr/bin/env python3
"""Closure test for the marginal-overlap slim failures (run_79, 2026-08-11).

Two questions, both on a segment that does NOT need help
(run_79/stat090_0001 x 224572, fully contained, 61 min, fits at ~95.9%):

A. Is the -0.983 ms broad structure in the wide scan UNIVERSAL, i.e. also
   present under a successful fit?  If yes, its appearance in the failure
   logs is a selection effect, and it is not evidence about the DREAM data.

B. Truncate the healthy segment to the overlap lengths that failed in the
   campaign (14/17/19/26/29 min) and to ones that passed (33/43 min):
     - does the unseeded bootstrap fail below ~30 min the way the campaign
       segments did?
     - does a fit seeded from the NEIGHBOUR sub-run's constants
       (stat090_0000: K=1.106350e-4, T0=-252.60 ns, the validated values)
       with boot=False recover the full-segment truth on the same slice?
"""
import sys
import json
import time
import numpy as np
from pathlib import Path

REPO = Path('/home/dylan/PycharmProjects/nTof_x17')
sys.path.insert(0, str(REPO))

from ntof_processing.slim_pipeline import config as C          # noqa: E402
from ntof_processing.slim_pipeline import clockfit as cf       # noqa: E402
from ntof_processing.slim_pipeline.slim import (               # noqa: E402
    Segment, _bind_ntof, join_events, bunch_table, pass1_candidates)

OUT = Path(__file__).parent / 'marginal_closure_results.json'
V12 = Path('/media/dylan/data/x17/ntof_reproc/v12_liqpileup')

# neighbour constants: run_79/stat090_0000 x 224572 (validated in README)
NEIGH_K, NEIGH_T0 = 1.106350e-4, -252.60

res = {'segment': 'run_79/stat090_0001 x 224572'}


def wide_scan_full(ev_bunch, ev_t, cd_bunch, cd_t, bin_ns=cf.XC_BIN_NS,
                   burst_ms=cf.XC_BURST_MS, max_bunches=cf.XC_BUNCHES):
    """clockfit.xcorr_lag, but returning the whole (lags, acc) array."""
    nb = int(burst_ms * 1e6 / bin_ns)
    acc = np.zeros(2 * nb)
    used = 0
    for b in np.unique(ev_bunch)[:max_bunches]:
        te, tc = ev_t[ev_bunch == b], cd_t[cd_bunch == b]
        if te.size < 5 or tc.size < 5:
            continue
        a = np.bincount(np.clip((te / bin_ns).astype(int), 0, nb - 1),
                        minlength=nb).astype(float)
        c = np.bincount(np.clip((tc / bin_ns).astype(int), 0, nb - 1),
                        minlength=nb).astype(float)
        acc += np.fft.irfft(np.conj(np.fft.rfft(a, 2 * nb))
                            * np.fft.rfft(c, 2 * nb), 2 * nb)
        used += 1
    lags = np.arange(2 * nb) * bin_ns
    lags[nb:] -= 2 * nb * bin_ns
    o = np.argsort(lags)
    return lags[o], acc[o], used


def zscore_at(lags, acc, lag_ns, exclude_ns=20000.0):
    """Robust z of acc near lag_ns, floor from everything far from BOTH the
    true peak (lag ~ 0) and the probe lag."""
    far = (np.abs(lags) > exclude_ns) & (np.abs(lags - lag_ns) > exclude_ns)
    med = float(np.median(acc[far]))
    mad = float(np.median(np.abs(acc[far] - med))) * 1.4826
    near = np.abs(lags - lag_ns) <= 5000.0
    peak = float(acc[near].max())
    i = int(np.argmax(acc[near]))
    return (peak - med) / max(mad, 1e-9), float(lags[near][i])


t0 = time.time()
seg = Segment('run_79', 'stat090_0001', 224572, ntof_source=V12)
_bind_ntof(seg)
ev = join_events(seg)
btbl, keep = bunch_table(ev)
if not keep.all():
    ev = ev[keep].reset_index(drop=True)
phys = ~ev['is_flash'].to_numpy()
ev_b = ev['BunchNumber'].to_numpy().astype(np.int64)
ev_t = ev['t_since_flash_ns'].to_numpy().astype(np.float64)
bunches = np.unique(ev_b[phys])
print(f'[{time.time()-t0:.0f}s] joined: {phys.sum():,} physics events, '
      f'{bunches.size} bunches')

cd, offs, thr = pass1_candidates(seg, bunches)
print(f'[{time.time()-t0:.0f}s] candidates: {cd["t"].size:,}')

eb, et = ev_b[phys], ev_t[phys]
cb, ct, ca = cd['bunch'], cd['t'], cd['arm']

# ---------------------------------------------------------------- truth fit
K, T0, arm_off, ginfo = cf.fit_global(eb, et, cb, ct, ca)
corr_in, corr_cv, pb = cf.fit_perbunch(eb, et, cb, ct, ca, K, T0, arm_off)
qa = cf.efficiency(eb, et, cb, ct, ca, K, T0, arm_off, corr_in, C.ACCEPT_NS)
res['truth'] = dict(K=K, T0_ns=T0, eff=qa['efficiency'],
                    boot=ginfo['bootstrap']['snr'])
print(f"TRUTH: K={K:.6e} T0={T0:+.2f} eff={qa['efficiency']:.4%}")

# ------------------------------------------------- A. wide scan, full health
lags, acc, used = wide_scan_full(eb, et, cb, ct)
i = int(np.argmax(acc))
med = float(np.median(acc))
mad = float(np.median(np.abs(acc - med))) * 1.4826
res['wide_scan'] = dict(
    used_bunches=used,
    top_lag_ms=float(lags[i] / 1e6),
    top_z=float((acc[i] - med) / max(mad, 1e-9)))
for probe in (-0.983e6, -0.982e6, -0.986e6):
    z, at = zscore_at(lags, acc, probe)
    res['wide_scan'][f'z_at_{probe/1e6:+.3f}ms'] = dict(z=float(z),
                                                        lag_ms=at / 1e6)
    print(f'wide scan z near {probe/1e6:+.3f} ms: {z:.1f} (peak at '
          f'{at/1e6:+.4f} ms)')
# save the acc curve around -1.05..-0.90 ms for plotting later
w = (lags >= -1.05e6) & (lags <= -0.90e6)
res['wide_scan']['window'] = dict(lag_ns=lags[w].tolist(),
                                  acc=acc[w].tolist())

# --------------------------------------------------------- B. truncation
res['trunc'] = []
full_min = 61.0
for minutes in (14, 17, 19, 26, 29, 33, 43):
    nkeep = max(1, int(round(bunches.size * minutes / full_min)))
    sel = set(bunches[:nkeep].tolist())
    me = np.isin(eb, list(sel))
    mc = np.isin(cb, list(sel))
    row = dict(minutes=minutes, n_bunches=nkeep,
               n_events=int(me.sum()), n_candidates=int(mc.sum()))

    # unseeded, as the campaign ran it
    try:
        Kb, T0b, offb, gib = cf.fit_global(eb[me], et[me], cb[mc], ct[mc],
                                           ca[mc], log=lambda *a: None)
        row['boot'] = dict(ok=True, K=Kb, T0_ns=T0b,
                           snr=gib['bootstrap'].get('snr'),
                           sigma=gib['bootstrap'].get('sigma'))
    except RuntimeError as e:
        row['boot'] = dict(ok=False, err=str(e)[:180])

    # seeded from the NEIGHBOUR sub-run, no bootstrap
    try:
        Ks, T0s, offy, gis = cf.fit_global(eb[me], et[me], cb[mc], ct[mc],
                                           ca[mc], K=NEIGH_K, T0=NEIGH_T0,
                                           boot=False, log=lambda *a: None)
        ci, cv, _ = cf.fit_perbunch(eb[me], et[me], cb[mc], ct[mc], ca[mc],
                                    Ks, T0s, offy, log=lambda *a: None)
        q = cf.efficiency(eb[me], et[me], cb[mc], ct[mc], ca[mc],
                          Ks, T0s, offy, ci, C.ACCEPT_NS)
        row['seeded'] = dict(ok=True, K=Ks, T0_ns=T0s,
                             dT0_vs_truth_ns=T0s - T0,
                             dK_rel=(Ks - K) / K,
                             eff=q['efficiency'])
    except RuntimeError as e:
        row['seeded'] = dict(ok=False, err=str(e)[:180])

    res['trunc'].append(row)
    b = row['boot']
    s = row['seeded']
    print(f"{minutes:3d} min: boot {'OK ' if b['ok'] else 'FAIL'}"
          f"{'' if b['ok'] else ' (' + b['err'][:60] + ')'}  |  seeded "
          + (f"T0 {s['T0_ns']:+.2f} (d {s['dT0_vs_truth_ns']:+.2f} ns) "
             f"eff {s['eff']:.4%}" if s['ok'] else 'FAIL ' + s['err'][:60]))

OUT.write_text(json.dumps(res, indent=1))
print(f'[{time.time()-t0:.0f}s] wrote {OUT}')
