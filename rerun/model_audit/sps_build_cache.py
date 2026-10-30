#!/usr/bin/env python3
"""sps_build_cache.py <run> <subrun> --dataset run63_rot25 --out DIR [--per-plateau 1000] [--half-win 16] [--tan 0.4800]

det4 H4 events from the decoded FEU 03 files on EOS (<runs>/<run>/<subrun>/decoded_root_fa80c06/*_03.root, ZS: sparse
sample/channel/amplitude, pedestal-subtracted on the FEU) in the bench calibration-cache format, one pickle per drift
plateau of the dataset.  Self-seeded (no uRWELL): per view, the brightest strip's +-half_win strips; both views
required, peak >= 300 ADC.  Plateau by wall clock: the 'timestamp' counter is calibrated to seconds from the sub-run's
span (dream_daq.log start of this and the next sub-run), events within 20 s of a plateau edge are dropped.
tan_x = tan_y = |tan(mount)| (magnitude; the inclined view and the sign are read off the data); ref_mesh = the own
charge centroid.  (ZS zeros are fitted as measured zeros; only RAW zeros are censored, sps_reco.py).
"""
import argparse, glob, os, pickle, re, sys
import numpy as np

REPO = os.environ.get('WFT_REPO') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path[:0] = [os.path.join(REPO, 'sps_beam_test_26', 'analysis'), os.path.join(REPO, 'sps_beam_test_26', 'det4_sps_assessment')]
import datasets                                              # noqa: E402
from det4_sps_map import POSITION_MM, VIEW, PITCH_MM         # noqa: E402

RUNS = '/eos/experiment/ntof/data/x17/p2_sps_july/runs'
NOISE_ADC = 10.0
#: ZS FEU data are decoded as 256 + (ADC - pedestal) on every kept sample (the flat ZS baseline; extract_det4_only.py,
#: flat_align_eff.py subtract it).  Before 2026-10-27 (ws_sps) this builder did not, so every kept sample of the ZS caches
#: (the model-audit F16/F20 SPS rows) sat 256 ADC above the suppressed zeros.
ZS_BASELINE = 256.0
#: RAW FEU data are decoded on the raw per-channel pedestals (~300-700 ADC), with the FEU's common-mode subtraction off.
#: Per event: (1) per channel the median of the present samples among the first RAW_PRE (pre-signal; the head-on column
#: starts at ~800 ns) is subtracted; (2) common mode: per sample, the median of each 64-channel connector block,
#: excluding the strips within RAW_CM_MASK of either view's leading strip and the oscillating channels RAW_CM_BAD
#: (extract_det4_only.py --cm masked); (3) step (1) again for the per-channel offset step (2) leaves.  The FEU's dropped
#: samples stay 0.  Without (2) (pass 1, 2026-10-30) a chip's 64 channels move together by ~260 ADC within the window
#: against 11 ADC noise, every strip reads as signal and the fit takes the common mode for charge
#: (rerun/pass1_review/REVIEW_2026-10-30.md P2).
RAW_PRE = 8
RAW_CM_MASK = 10
RAW_CM_BAD = (510, 372)


def _baseline(ch, sm, am, raw):
    """Pedestal-corrected amplitudes of one event's samples (see ZS_BASELINE / RAW_PRE)."""
    if not raw:
        out = am - ZS_BASELINE
    else:
        out = _pre_baseline(ch, sm, am)
        out = _pre_baseline(ch, sm, out - _common_mode(ch, sm, out))
    out[out == 0.0] = 1e-3       # an exact 0 means 'not read out' downstream (suppressed / dropped)
    return out


def _pre_baseline(ch, sm, am):
    """Per channel, subtract the median of its present samples among the first RAW_PRE (all of them if none)."""
    out = am.copy()
    for c in np.unique(ch):
        k = ch == c; pre = am[k & (sm < RAW_PRE)]
        out[k] = am[k] - (np.median(pre) if len(pre) else np.median(am[k]))
    return out


def _block_median(ch, sm, am, use):
    """Per sample and 64-channel connector block, the median of am over the channels in ``use`` (bool[512]), returned
    per sample of the input; 0 where the block has no such channel at that sample."""
    import warnings
    ns = int(sm.max()) + 1
    grid = np.full((512, ns), np.nan, np.float32)
    k = use[ch]
    grid[ch[k], sm[k]] = am[k]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nan_to_num(np.nanmedian(grid.reshape(8, 64, ns), axis=1))   # (block, sample)
    return med[ch // 64, sm]


def _common_mode(ch, sm, am):
    """The RAW common mode per sample of the input: the block median over the quiet channels, i.e. not RAW_CM_BAD and
    not within RAW_CM_MASK strips of either view's leading strip.  The leading strips are found after a first, unmasked
    block median (the common mode differs between blocks by ~100s of ADC, so the raw maximum can sit in the wrong one)."""
    use = np.ones(512, bool); use[list(RAW_CM_BAD)] = False
    first = am - _block_median(ch, sm, am, use)
    pk = np.full(512, -np.inf); np.maximum.at(pk, ch, first)
    pk[list(RAW_CM_BAD)] = -np.inf
    for view in ('x', 'y'):
        v = VIEW == view
        lead = np.flatnonzero(v)[np.argmax(pk[v])]
        use[v & (np.abs(POSITION_MM - POSITION_MM[lead]) <= RAW_CM_MASK * PITCH_MM)] = False
    return _block_median(ch, sm, am, use)


def subrun_spans(run, subrun):
    """[(t_start, t_end), ...] seconds since midnight of the run's first day, from dream_daq.log: one per time the sub-run
    was started (run_61 meshscan_m30V was taken twice, 13:46 and 16:08; pass1 10-30)."""
    import datetime
    lines = open(os.path.join(RUNS, run, 'dream_daq.log')).read().splitlines()
    day0 = None; out = []
    for ln in lines:
        m = re.match(r'(\d{4}-\d\d-\d\d) (\d\d):(\d\d):(\d\d),\d+ INFO: Subrun (started|finished): (\S+)', ln)
        if not m:
            continue
        d, h, mi, s, what, name = m.groups()
        if day0 is None:
            day0 = d
        dd = (datetime.date.fromisoformat(d) - datetime.date.fromisoformat(day0)).days
        t = dd * 86400 + int(h) * 3600 + int(mi) * 60 + int(s)
        if name == subrun and what == 'started':
            out.append([t, None])
        if name == subrun and what == 'finished' and out and out[-1][1] is None:
            out[-1][1] = t
    return [tuple(x) for x in out]


def subrun_span(run, subrun):
    """(t_start, t_end) of the sub-run's LAST start (the historical behaviour; build_cache uses subrun_spans)."""
    sp = subrun_spans(run, subrun)
    return list(sp[-1]) if sp else [None, None]


STEM_HM = re.compile(r'_datrun_\d{6}_(\d\d)H(\d\d)_\d{3}_\d\d\.root$')


def acquisitions(files, spans):
    """[(span, files)]: the decoded files grouped by acquisition (file-stem HHhMM), each paired with the sub-run start in
    dream_daq.log closest to its stem time (same day; the stem is the acquisition start to the minute)."""
    groups = {}
    for f in files:
        m = STEM_HM.search(os.path.basename(f))
        groups.setdefault((int(m.group(1)), int(m.group(2))) if m else None, []).append(f)
    if len(groups) == 1 or not spans:
        return [(spans[-1] if spans else (None, None), sorted(files))]
    out = []
    for hm, fs in sorted(groups.items()):
        t = hm[0] * 3600 + hm[1] * 60
        sp = min(spans, key=lambda x: min(abs((x[0] % 86400) - t), 86400 - abs((x[0] % 86400) - t)))
        out.append((sp, sorted(fs)))
    return out


def build_cache(run, subrun, dataset, per_plateau=1000, half_win=16, tan=None, edge_s=20.0, qmin=300.0, only=None, verbose=True):
    """{plateau: {eid: event}} for the decoded FEU 03 files of one sub-run (``only``: restrict to these plateaus)."""
    a = argparse.Namespace(run=run, subrun=subrun, dataset=dataset, per_plateau=per_plateau, half_win=half_win,
                           tan=float(np.tan(np.radians(25.64))) if tan is None else tan, edge_s=edge_s, qmin=qmin)
    import uproot
    D = datasets.get(a.dataset); nsmp = int(D['n_samples'])
    files = sorted(glob.glob(os.path.join(RUNS, a.run, a.subrun, 'decoded_root_fa80c06', '*_datrun_*_03.root')))
    plateaus = [(n, lo + a.edge_s, hi - a.edge_s) for n, lo, hi, dv, rv in D['plateaus'] if only is None or n in only]
    out = {}
    # timestamp calibration per acquisition (the DREAM counter restarts with each one): first event of its first file,
    # last event of its last file -> its own start/finish in dream_daq.log
    for (t_lo, t_hi), afiles in acquisitions(files, subrun_spans(a.run, a.subrun)):
        print(f'{len(afiles)} files; sub-run {a.subrun} {t_lo}..{t_hi} s', flush=True)
        ts0 = uproot.open(afiles[0])['nt'].arrays(['timestamp'], entry_stop=1, library='np')['timestamp'][0]
        tl = uproot.open(afiles[-1])['nt']; ts1 = tl.arrays(['timestamp'], entry_start=tl.num_entries - 1, library='np')['timestamp'][0]
        scale = (t_hi - t_lo) / float(ts1 - ts0)
        print(f'timestamp unit {scale * 1e9:.1f} ns  (span {ts1 - ts0} counts over {t_hi - t_lo} s)', flush=True)
        _fill(out, afiles, a, D, plateaus, t_lo, ts0, scale, nsmp, verbose, files.index(afiles[0]))
        if all(len(out.get(n, {})) >= a.per_plateau for n, _, _ in plateaus):
            break
    return out


def _fill(out, files, a, D, plateaus, t_lo, ts0, scale, nsmp, verbose, base):
    """Add one acquisition's events to out ({plateau: {eid: event}}); base = index of its first file in the sub-run's
    sorted file list (event keys eid + 1e7 * file index, as before the per-acquisition split)."""
    import uproot
    for f in files:
        t = uproot.open(f)['nt']
        for lo in range(0, t.num_entries, 20000):
            arr = t.arrays(['eventId', 'timestamp', 'ftst', 'sample', 'channel', 'amplitude'], entry_start=lo,
                           entry_stop=min(lo + 20000, t.num_entries), library='np')
            tw = t_lo + (arr['timestamp'].astype(float) - ts0) * scale
            for j in range(len(tw)):
                L = next((n for n, l0, l1 in plateaus if l0 <= tw[j] < l1), None)
                if L is None or len(out.get(L, {})) >= a.per_plateau:
                    continue
                ch = arr['channel'][j].astype(int); sm = arr['sample'][j].astype(int); am = arr['amplitude'][j].astype(np.float32)
                am = _baseline(ch, sm, am, bool(D.get('raw', False)))
                rec = dict(eid=int(arr['eventId'][j]), tan_x=a.tan, tan_y=a.tan, t_wall=float(tw[j]), file=os.path.basename(f),
                           timestamp=int(arr['timestamp'][j]))
                ok = True
                for view in ('x', 'y'):
                    k = VIEW[ch] == view
                    if not k.any():
                        ok = False; break
                    # brightest strip of the view, then +-half_win strips in position
                    pk = np.zeros(512); np.maximum.at(pk, ch[k], am[k])
                    c0 = int(np.argmax(pk))
                    if pk[c0] < a.qmin:
                        ok = False; break
                    p0 = POSITION_MM[c0]
                    sel = k & (np.abs(POSITION_MM[ch] - p0) <= a.half_win * PITCH_MM)
                    cc, ss, aa = ch[sel], sm[sel], am[sel]
                    chs = np.unique(cc)
                    if len(chs) < 4:
                        ok = False; break
                    W = np.zeros((len(chs), nsmp), np.float32); ci = np.searchsorted(chs, cc); m = (ss >= 0) & (ss < nsmp)
                    W[ci[m], ss[m]] = aa[m]
                    o = np.argsort(POSITION_MM[chs])
                    q = np.clip(W, 0, None).sum(1)
                    rec[view] = dict(ch=chs[o].astype(np.int16), pos=POSITION_MM[chs[o]].astype(np.float32), W=W[o],
                                     noise=np.full(len(chs), NOISE_ADC, np.float32))
                    rec[f'ref_mesh_{view}'] = float((q * POSITION_MM[chs]).sum() / max(q.sum(), 1e-9))
                    rec[f'ftst_{view}'] = int(arr['ftst'][j])
                if ok:
                    out.setdefault(L, {})[rec['eid'] + 10_000_000 * (base + files.index(f))] = rec
        if verbose:
            print(f'  {os.path.basename(f)}: ' + ', '.join(f'{k} {len(v)}' for k, v in sorted(out.items())), flush=True)
        if all(len(out.get(n, {})) >= a.per_plateau for n, _, _ in plateaus):
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('subrun'); ap.add_argument('--dataset', default='run63_rot25')
    ap.add_argument('--out', required=True); ap.add_argument('--per-plateau', type=int, default=1000)
    ap.add_argument('--half-win', type=int, default=16); ap.add_argument('--tan', type=float, default=float(np.tan(np.radians(25.64))))
    ap.add_argument('--edge-s', type=float, default=20.0); ap.add_argument('--qmin', type=float, default=300.0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    out = build_cache(a.run, a.subrun, a.dataset, a.per_plateau, a.half_win, a.tan, a.edge_s, a.qmin)
    dv = {x[0]: x[3] for x in datasets.get(a.dataset)['plateaus']}
    for L, evs in out.items():
        p = os.path.join(a.out, f'{a.dataset}_{a.subrun}_{L}.pkl'); pickle.dump(evs, open(p, 'wb'), protocol=4)
        print(f'{L}: {len(evs)} events, drift {dv[L]} V -> {p}')


if __name__ == '__main__':
    main()
