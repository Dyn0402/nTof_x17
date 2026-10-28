#!/usr/bin/env python3
"""sps_colsum.py <run> <subrun> --dataset D --tmpl TMPL.npz --out OUT.npz [--max-events N] [--half-win 16]

GC: model-free column duration T of det4 at H4 vs wall-clock time, one sub-run.  Same estimator as FA's colsum.py (strip
sum of a view = box of duration gap/v convolved with the electronics template, angle terms cancel), on the decoded FEU 03
files (decoded_root_fa80c06) with sps_build_cache.py's event selection and baseline: per view the brightest strip
(>= 300 ADC) and +-half_win strips in position; ZS kept samples minus the 256 ADC flat baseline, RAW minus the
pre-signal median.  Missing samples count as zero (ZS-suppressed = baseline).  Template: bench det4 tmpl_x (ws_spsv).
Wall clock: sub-run span from dream_daq.log, the 'timestamp' counter scaled to it (as sps_build_cache.py does).
Events are taken with a stride so that at most --max-events are fitted, spread over the whole sub-run.
"""
import argparse
import glob
import os
import sys

import numpy as np

CODE = os.environ.get('WFT_REPO') or '/eos/user/d/dneff/x17rerun/followups/fa/code_2ce1dcc'
sys.path[:0] = [os.path.join(CODE, 'rerun', 'model_audit'), os.path.join(CODE, 'rerun', 'model_audit', 'followups',
                                                                          'fa_gas_v_tilt')]
import sps_build_cache as sbc          # noqa: E402
import colsum                          # noqa: E402


class MaskedGridFit(colsum.GridFit):
    """colsum.GridFit with a per-event sample mask: samples read out on no strip of the window (ZS-suppressed on
    all of them, or a RAW dropped group) are left out of the fit, as ws_spsv/boxfit.py does."""

    def fit(self, y, ok=None):
        if ok is None or ok.all():
            return super().fit(y)
        B = self.B[:, :, ok, :]; yy = y[ok]
        M = np.einsum('ijsk,ijsl->ijkl', B, B)
        det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] ** 2
        good = det > 1e-9 * np.maximum(M[..., 0, 0] * M[..., 1, 1], 1e-30)
        Bty = np.einsum('ijsk,s->ijk', B, yy)
        d = np.where(good, det, 1.0)
        c0 = (M[..., 1, 1] * Bty[..., 0] - M[..., 0, 1] * Bty[..., 1]) / d
        c1 = (M[..., 0, 0] * Bty[..., 1] - M[..., 0, 1] * Bty[..., 0]) / d
        chi = yy @ yy - (c0 * Bty[..., 0] + c1 * Bty[..., 1])
        chi = np.where(good & (c0 > 0) & (c0 + c1 > 0), chi, np.inf)
        i, j = np.unravel_index(np.argmin(chi), chi.shape)
        if not np.isfinite(chi[i, j]):
            return (np.nan,) * 5
        T = colsum.T_GRID[j]
        if 0 < j < len(colsum.T_GRID) - 1 and np.all(np.isfinite(chi[i, j - 1:j + 2])):
            a0, a1, a2 = chi[i, j - 1], chi[i, j], chi[i, j + 1]
            den = a0 - 2 * a1 + a2
            if den > 0:
                T = T + 0.5 * (a0 - a2) / den * (colsum.T_GRID[1] - colsum.T_GRID[0])
        return (float(colsum.T0_GRID[i]), float(T), float(c0[i, j]), float(c1[i, j]),
                float(chi[i, j] / max(ok.sum() - 4, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('subrun'); ap.add_argument('--dataset', required=True)
    ap.add_argument('--tmpl', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--max-events', type=int, default=30000); ap.add_argument('--half-win', type=int, default=16)
    ap.add_argument('--qmin', type=float, default=300.0)
    a = ap.parse_args()
    import uproot
    D = sbc.datasets.get(a.dataset); ns = int(D['n_samples']); raw = bool(D.get('raw', False))
    colsum.DTS = 60.0
    colsum.T0_GRID = np.arange(0.0, 1500.0 + 1e-9, 10.0)
    colsum.T_GRID = np.arange(600.0, min(3600.0, ns * 60.0) + 1e-9, 10.0)
    z = np.load(a.tmpl); grid, tmpl = z['grid'], z['tmpl_x'] / z['tmpl_x'].max()
    gf = MaskedGridFit(colsum.Kern(grid, tmpl), ns)
    files = sorted(glob.glob(os.path.join(sbc.RUNS, a.run, a.subrun, 'decoded_root_fa80c06', '*_datrun_*_03.root')))
    t_lo, t_hi = sbc.subrun_span(a.run, a.subrun)
    ntot = sum(uproot.open(f)['nt'].num_entries for f in files)
    stride = max(1, ntot // a.max_events)
    ts0 = uproot.open(files[0])['nt'].arrays(['timestamp'], entry_stop=1, library='np')['timestamp'][0]
    tl = uproot.open(files[-1])['nt']
    ts1 = tl.arrays(['timestamp'], entry_start=tl.num_entries - 1, library='np')['timestamp'][0]
    scale = (t_hi - t_lo) / float(ts1 - ts0) if (t_hi and ts1 > ts0) else 1e-8
    print(f'[sps_colsum] {a.run}/{a.subrun}: {len(files)} files, {ntot} events, stride {stride}, '
          f'tick {scale * 1e9:.2f} ns, span {t_lo}..{t_hi} s', flush=True)
    rec = {k: [] for k in ('t_wall', 'ts', 'eid', 'file_idx')}
    for v in 'xy':
        for k in ('T', 't0', 'a', 'b', 'chi2', 'q', 'pk', 'n', 'cen', 'S', 'ok'):
            rec[f'{v}_{k}'] = []
    for fi, f in enumerate(files):
        t = uproot.open(f)['nt']
        for lo in range(0, t.num_entries, 20000):
            arr = t.arrays(['eventId', 'timestamp', 'sample', 'channel', 'amplitude'], entry_start=lo,
                           entry_stop=min(lo + 20000, t.num_entries), library='np')
            for j in range(0, len(arr['eventId']), 1):
                if (lo + j) % stride:
                    continue
                ch = arr['channel'][j].astype(int); sm = arr['sample'][j].astype(int)
                am = sbc._baseline(ch, sm, arr['amplitude'][j].astype(np.float32), raw)
                res = {}
                for view in 'xy':
                    k = sbc.VIEW[ch] == view
                    if not k.any():
                        break
                    pk = np.zeros(512); np.maximum.at(pk, ch[k], am[k])
                    c0 = int(np.argmax(pk))
                    if pk[c0] < a.qmin:
                        break
                    sel = k & (np.abs(sbc.POSITION_MM[ch] - sbc.POSITION_MM[c0]) <= a.half_win * sbc.PITCH_MM)
                    sel &= (sm >= 0) & (sm < ns)
                    if len(np.unique(ch[sel])) < 4:
                        break
                    S = np.zeros(ns); np.add.at(S, sm[sel], am[sel])
                    ok = np.zeros(ns, bool); ok[sm[sel]] = True     # samples read out on at least one strip
                    q = np.clip(am[sel], 0, None)
                    cen = float((q * sbc.POSITION_MM[ch[sel]]).sum() / max(q.sum(), 1e-9))
                    res[view] = (gf.fit(S, ok), float(np.clip(S, 0, None).sum()), float(pk[c0]),
                                 len(np.unique(ch[sel])), cen, S.astype(np.float32), ok)
                if len(res) < 2:
                    continue
                rec['t_wall'].append(t_lo + (float(arr['timestamp'][j]) - ts0) * scale)
                rec['ts'].append(int(arr['timestamp'][j])); rec['eid'].append(int(arr['eventId'][j]))
                rec['file_idx'].append(fi)
                for view in 'xy':
                    (t0, T, aa, bb, chi), q, pk, n, cen, S, ok = res[view]
                    for k, val in zip(('T', 't0', 'a', 'b', 'chi2', 'q', 'pk', 'n', 'cen', 'S', 'ok'),
                                      (T, t0, aa, bb, chi, q, pk, n, cen, S, ok)):
                        rec[f'{view}_{k}'].append(val)
        print(f'[sps_colsum] {os.path.basename(f)}: {len(rec["eid"])} events so far', flush=True)
    out = {k: np.asarray(v) for k, v in rec.items()}
    out['meta'] = np.array([f'run={a.run} subrun={a.subrun} dataset={a.dataset} ns={ns} raw={raw} stride={stride} '
                            f'tick_ns={scale * 1e9:.3f} half_win={a.half_win}'])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f'[sps_colsum] wrote {a.out} ({len(out["eid"])} events)', flush=True)


if __name__ == '__main__':
    main()
