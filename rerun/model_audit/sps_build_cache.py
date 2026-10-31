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
#: Channels that may never SEED a window or enter the own-centroid ref: the oscillating RAW_CM_BAD.  Y ch 372
#: (158.34 mm) and ch 510 (250.38 mm) swing by +-400-900 ADC in some sub-runs (known since
#: sps_beam_test_26/analysis/RAW_RUN71_REANALYSIS_2026-08-04.md, never carried into this builder).  Before 2026-10-30
#: (rerun/sps_audit) the brightest-strip seed picked them and the whole +-16-strip window sat on the oscillation: 41-44 %
#: of the run71 RAW Y fits at 156-160.5 mm and 9-12 % at 248.5-252 mm, 24 % of run55 d700, 16 % of run57 d650/d700, 37 %
#: of run60 ov07.  The census-dead channels (det4_sps_h4.json 'dead') are NOT excluded: some carry as much charge as
#: their neighbours (X ch 2, 64 on run66), the fit censors them anyway and a window seeded on one still holds the track.
SEED_BAD = np.zeros(512, bool)
SEED_BAD[list(RAW_CM_BAD)] = True
#: Which channel oscillates changes with the sub-run: on ZS run55 d700 it is ch 373 (157.56 mm), the neighbour of 372,
#: and 25 % of that row's Y windows were seeded on it.  So each plateau is also probed: the first HOT_SCAN events are
#: seeded with SEED_BAD, and a channel that seeds more than HOT_FRAC of them AND more than HOT_RATIO x the median of its
#: +-4 position neighbours in the same view is excluded from seeding for that plateau (HOT_LAST records it).  A real
#: beam strip seeds ~1 % (Y) to ~7 % (X, inside a 12 mm live band) with neighbours at the same level, a band-edge strip
#: at most ~3x its neighbours; the oscillating strips seed 4-45 % against 0-15 / 3000 around them (rerun/sps_audit/README.md).
#: The probe reads the plateau's events of the sub-run, as many as the row keeps but at least HOT_SCAN_MIN (HOT_SCAN None),
#: in blocks of HOT_BLOCK events, and a channel hot in any block is hot for the plateau.  It was the first 2000 events as
#: one block until 2026-10-31: an oscillation that is on in part of the plateau only was missed or caught by chance
#: (run61 d210: ch 373 seeds 4.2 % of the row but < 3 % of its first 2000 events, a 1.3 % excess kept).  The minimum keeps
#: what the 2000-event probe saw where a sub-run gives the row only a few hundred events (run60 ov14: 373 oscillates in
#: overnight_14 after the row's events; excluding a channel that is quiet costs one censored strip, keeping an oscillator
#: costs junk windows) (rerun/sps_junk/README.md).
#: The neighbours are the +-4 position neighbours that are NOT under a dead amplification stripe or electronically dead
#: (rerun/channel_layer/det4_sps_h4.json 'dead_stripes', STRIPE).  Until 2026-10-31 the dead ones counted, with ~0 seeds:
#: the last live strip at a stripe edge inside a busy beam (X ch 172/173, 214 mm, on run55: the beam spot ends there) read
#: as > 10x its neighbours and was censored as hot in run55 d500-d700, and with it the beam's edge.
#: HOT_FRAC was 3 % of the 2000 events until 2026-10-31; it is 1 % of a 1000-event block (>= 10 seeds) since: on run55
#: d600 ch 373 is ON in 4-7 % of the events and seeds 2.3-2.6 % of each block against 0.1-0.4 % for its live neighbours.
#: The ratio to the live neighbours is what separates an oscillator from a beam strip; the floor only asks for statistics.
HOT_SCAN = None
HOT_SCAN_MIN = 2000
HOT_BLOCK = 1000
HOT_FRAC = 0.01
HOT_RATIO = 10.0
HOT_LAST = {}
#: Cross-talk of an oscillator onto its neighbours (rerun/sps_junk/README.md, 2026-10-31).  The oscillation is periodic
#: (a lobe every ~15 samples) and, in the events where it is on, appears on adjacent channels at ~0.3-0.7 of its height
#: and the same phase: on run55 d700 ch 375 (156.0 mm) exceeds 300 ADC in 80 % of the events in which ch 373 is on and in
#: 0.2 % of the others.  Those copies stay below the hot-seed rule (the oscillator itself is excluded, and the copies split
#: over 2-3 strips) but win the seed whenever the event's real Y signal is weaker, and put 2-13 % of a row's Y windows at
#: 155-160 mm.  So, per plateau, from the probe: an oscillator o (SEED_BAD or hot) is ON in an event when its peak exceeds
#: COUPLE_ON_ADC.  A channel within COUPLE_NB position neighbours of o (same view) is COUPLED to o when, at either lit
#: level of COUPLE_LIT_ADC (peak > 100 or > 300 ADC), it is lit in at least COUPLE_MIN of the ON events and its lit rate in
#: ON events is more than COUPLE_RATIO x its rate in the OFF events, divided by the same ratio of the strips away from
#: every oscillator (the beam's own
#: modulation between ON and OFF events, e.g. an oscillator that is on only while the beam is).  Two levels because the
#: copies differ: on run55 d700 the 375 copy is ON/OFF 99x at 100 ADC and 385x at 300; on run60 ov03 the 371 copy is 12x
#: at 100 ADC (that strip is lit at 100 ADC in 8 % of the OFF events too) but 32x at 300.  A real strip is lit by
#: the beam whether or not the oscillator is on, so it is not coupled.  In an event where o is ON its coupled channels
#: are treated like o: kept out of the seed and the own-centroid ref, blanked in the window and censored in the fit
#: (record 'censor_x' / 'censor_y'); in the other events they are ordinary strips.  The rule only applies where o
#: actually oscillates in the plateau: its ON rate must exceed COUPLE_OSC_RATIO x the median ON rate of its live
#: +-COUPLE_NB position neighbours (live_neighbours).  A SEED_BAD channel that is quiet in the plateau (ch 510 sits in the beam, 250 mm) is ON only
#: when a track crosses it, like its neighbours, and those tracks share charge with the neighbours: that is not coupling.
COUPLE_ON_ADC = 300.0
COUPLE_LIT_ADC = (100.0, 300.0)
COUPLE_NB = 4
COUPLE_MIN = 10
COUPLE_RATIO = 10.0
COUPLE_OSC_RATIO = 10.0
COUPLE_LAST = {}


def _stripe_mask():
    """bool[512]: the H4 channel layer's 'dead_stripes' (electronically dead + under a dead amplification stripe)."""
    import json
    m = json.load(open(os.path.join(REPO, 'rerun', 'channel_layer', 'det4_sps_h4.json')))
    out = np.zeros(512, bool)
    for p in ('x', 'y'):
        out[[int(c) for c in m[p]['dead_stripes']]] = True
    return out


STRIPE = _stripe_mask()


def live_neighbours(c, nb=4, skip=None):
    """The channels within +-nb position ranks of c in its view, not c and not in skip (default STRIPE)."""
    skip = STRIPE if skip is None else skip
    view = VIEW[c]; chs = np.flatnonzero(VIEW == view); o = chs[np.argsort(POSITION_MM[chs])]
    i = int(np.flatnonzero(o == c)[0])
    return [int(x) for x in np.r_[o[max(0, i - nb):i], o[i + 1:i + nb + 1]] if not skip[x]]


def hot_channels(seed_counts, n, skip=None):
    """bool[512]: channels whose seed count (array[512]) over n events is > HOT_FRAC n and > HOT_RATIO x the median
    of their live +-4 position neighbours in the same view (live_neighbours)."""
    cnt = np.asarray(seed_counts, float); out = np.zeros(512, bool)
    for c in np.flatnonzero(cnt > HOT_FRAC * n):
        if VIEW[c] not in ('x', 'y'):
            continue
        nb = live_neighbours(int(c), skip=skip)
        out[c] = cnt[c] > HOT_RATIO * max(np.median(cnt[nb]) if nb else 0.0, 1.0)
    return out


def hot_in_blocks(seeds, block=None):
    """bool[512]: hot_channels over consecutive blocks of the probe's events (seeds: [(seed_x, seed_y)] in time order);
    hot in any block of at least half a block (a shorter tail joins the block before it)."""
    block = block or HOT_BLOCK
    n = len(seeds); edges = list(range(0, n, block)) + [n]
    if len(edges) > 2 and edges[-1] - edges[-2] < block // 2:
        edges.pop(-2)
    out = np.zeros(512, bool)
    for lo, hi in zip(edges[:-1], edges[1:]):
        cnt = np.zeros(512)
        for sx, sy in seeds[lo:hi]:
            cnt[sx] += 1; cnt[sy] += 1
        out |= hot_channels(cnt, hi - lo)
    return out


def coupled_channels(peaks, osc):
    """{oscillator: [coupled channels]} from the probe's per-event peaks (array [n_events, 512], baseline-subtracted ADC)
    and the oscillators (bool[512] or channel list); the rule is described at COUPLE_*.  An oscillator with fewer than
    COUPLE_MIN ON or OFF events gives no coupling (nothing to compare)."""
    pk = np.asarray(peaks, float)
    oscs = np.flatnonzero(osc) if np.asarray(osc).dtype == bool else np.asarray(sorted(osc), int)
    out = {}; lits = [pk > t for t in COUPLE_LIT_ADC]
    order_of, rank_of = {}, {}
    for view in ('x', 'y'):
        chs = np.flatnonzero(VIEW == view); order_of[view] = chs[np.argsort(POSITION_MM[chs])]
        rank_of[view] = {int(c): i for i, c in enumerate(order_of[view])}
    # the strips of BOTH views away from every oscillator: the beam's own ON/OFF modulation is measured on them (the
    # other view always sees the beam)
    near = set()
    for o in oscs:
        v = VIEW[o]; i = rank_of[v][int(o)]
        near |= {int(c) for c in order_of[v][max(0, i - COUPLE_NB - 4):i + COUPLE_NB + 5]}
    far = np.array([c for c in range(512) if c not in near and VIEW[c] in ('x', 'y')], int)
    for view in ('x', 'y'):
        order, rank = order_of[view], rank_of[view]
        vo = [int(o) for o in oscs if VIEW[o] == view]
        for o in vo:
            on = pk[:, o] > COUPLE_ON_ADC; n1, n0 = int(on.sum()), int((~on).sum())
            if n1 < COUPLE_MIN or n0 < COUPLE_MIN:
                continue
            i = rank[o]
            nbs = [int(c) for c in order[max(0, i - COUPLE_NB):i + COUPLE_NB + 1] if int(c) not in vo]
            live = [c for c in live_neighbours(o, COUPLE_NB) if c not in vo]
            nb_rate = np.median((pk[:, live] > COUPLE_ON_ADC).mean(0)) if live else 0.0
            if on.mean() <= COUPLE_OSC_RATIO * max(nb_rate, 1.0 / len(pk)):
                continue                                   # not oscillating here: ON = a track crossing it
            got = []
            for lit in lits:
                f1, f0 = lit[on][:, far].mean(), lit[~on][:, far].mean()
                base = (f1 / f0) if f0 > 0 else (np.inf if f1 > 0 else 1.0)
                for c in nbs:
                    k1, k0 = int(lit[on, c].sum()), int(lit[~on, c].sum())
                    r = (k1 / n1) / (max(k0, 1) / n0)
                    if c not in got and k1 >= COUPLE_MIN and r > COUPLE_RATIO * max(base, 1.0):
                        got.append(c)
            if got:
                out[o] = got
    return out


#: entries read per uproot call.  A RAW entry carries every channel x sample (~32 k values, ~150 kB as read), so a
#: RAW caller lowers it (rerun/vt_all/vt_sps.py: 1000).
CHUNK = 20000


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


def build_cache(run, subrun, dataset, per_plateau=1000, half_win=16, tan=None, edge_s=20.0, qmin=300.0, only=None, verbose=True,
                hot_scan=HOT_SCAN, keep_peaks=False):
    """{plateau: {eid: event}} for the decoded FEU 03 files of one sub-run (``only``: restrict to these plateaus).
    Probe first (hot_scan events per plateau; None = per_plateau, 0 = no probe): its hot channels are kept out of the
    seed (HOT_LAST) and the channels coupled to an oscillator are censored in its ON events (COUPLE_LAST).
    keep_peaks: each record also carries 'peak' (float32[512], the event's per-channel peak; the probe's input)."""
    a = argparse.Namespace(run=run, subrun=subrun, dataset=dataset, per_plateau=per_plateau, half_win=half_win,
                           tan=float(np.tan(np.radians(25.64))) if tan is None else tan, edge_s=edge_s, qmin=qmin, seed_bad={},
                           couple={}, keep_peaks=keep_peaks)
    n_scan = max(per_plateau, HOT_SCAN_MIN) if hot_scan is None else hot_scan
    if n_scan and per_plateau > 0:
        probe = build_cache(run, subrun, dataset, n_scan, half_win, tan, edge_s, qmin, only, False, hot_scan=0,
                            keep_peaks=True)
        for L, evs in probe.items():
            cnt = np.zeros(512)
            for e in evs.values():
                cnt[e['seed_ch_x']] += 1; cnt[e['seed_ch_y']] += 1
            hot = hot_in_blocks([(e['seed_ch_x'], e['seed_ch_y']) for e in evs.values()])
            a.seed_bad[L] = SEED_BAD | hot
            HOT_LAST[(run, subrun, L)] = [int(c) for c in np.flatnonzero(hot)]
            if hot.any():
                print(f'{subrun} {L}: hot seed channels {HOT_LAST[(run, subrun, L)]} kept out of the seed '
                      f'({", ".join(f"{int(cnt[c])}/{len(evs)}" for c in np.flatnonzero(hot))})', flush=True)
            cp = coupled_channels(np.array([e['peak'] for e in evs.values()]), a.seed_bad[L]) if evs else {}
            a.couple[L] = {o: np.array(cs, int) for o, cs in cp.items()}
            COUPLE_LAST[(run, subrun, L)] = {int(o): [int(c) for c in cs] for o, cs in cp.items()}
            if cp:
                print(f'{subrun} {L}: channels coupled to an oscillator, censored in its ON events '
                      f'{COUPLE_LAST[(run, subrun, L)]}', flush=True)
    import uproot
    D = datasets.get(a.dataset); nsmp = int(D['n_samples'])
    files = sorted(glob.glob(os.path.join(RUNS, a.run, a.subrun, 'decoded_root_fa80c06', '*_datrun_*_03.root')))
    plateaus = [(n, lo + a.edge_s, hi - a.edge_s) for n, lo, hi, dv, rv in D['plateaus'] if only is None or n in only]
    out = {}
    # timestamp calibration per acquisition (the DREAM counter restarts with each one): first event of its first file,
    # last event of its last file -> its own start/finish in dream_daq.log
    for (t_lo, t_hi), afiles in acquisitions(files, subrun_spans(a.run, a.subrun)):
        print(f'{len(afiles)} files; sub-run {a.subrun} {t_lo}..{t_hi} s', flush=True)
        with uproot.open(afiles[0]) as F:
            ts0 = F['nt'].arrays(['timestamp'], entry_stop=1, library='np')['timestamp'][0]
        with uproot.open(afiles[-1]) as F:
            tl = F['nt']; ts1 = tl.arrays(['timestamp'], entry_start=tl.num_entries - 1, library='np')['timestamp'][0]
        scale = (t_hi - t_lo) / float(ts1 - ts0)
        print(f'timestamp unit {scale * 1e9:.1f} ns  (span {ts1 - ts0} counts over {t_hi - t_lo} s)', flush=True)
        _fill(out, afiles, a, D, plateaus, t_lo, ts0, scale, nsmp, verbose, files.index(afiles[0]))
        if all(len(out.get(n, {})) >= a.per_plateau for n, _, _ in plateaus):
            break
    return out


def event_record(ch, sm, am, raw, nsmp, half_win=16, qmin=300.0, tan=0.0, seed_bad=None, min_strips=4, couple=None,
                 keep_peaks=False):
    """One event's cache record ({'x': window, 'y': window, ref_mesh_x/y, seed_ch_x/y, tan_x/y, censor_x/y}) from its
    decoded samples, or None when a view is empty, its peak is below qmin or its window has < min_strips strips with
    samples (the pass: 300 ADC, 4; the resolution sample of rerun/sps_audit drops both).  Per view the brightest
    strip that is not in ``seed_bad`` (default SEED_BAD) seeds a +-half_win-strip window in POSITION (the plugs are
    inverted: det4_sps_map).  The window keeps every strip, bad ones included; the own-centroid ref leaves the bad
    (oscillating) ones out.  ``couple`` ({oscillator: channels}, coupled_channels): in this event, the channels coupled
    to an oscillator that is ON are bad as well, and listed in censor_x/censor_y for the fit (sps_reco)."""
    bad = SEED_BAD if seed_bad is None else seed_bad
    am = _baseline(ch, sm, am, raw)
    rec = dict(tan_x=tan, tan_y=tan)
    if couple or keep_peaks:
        pk_all = np.zeros(512, np.float32); np.maximum.at(pk_all, ch, am)
        if keep_peaks:
            rec['peak'] = pk_all
        if couple:
            on = [cs for o, cs in couple.items() if pk_all[o] > COUPLE_ON_ADC]
            if on:
                bad = bad.copy(); ev_cens = np.zeros(512, bool); ev_cens[np.concatenate(on)] = True
                ev_cens &= ~bad; bad |= ev_cens
                rec['_cens'] = ev_cens
    for view in ('x', 'y'):
        k = VIEW[ch] == view
        if not k.any():
            return None
        # brightest good strip of the view, then +-half_win strips in position
        pk = np.zeros(512); np.maximum.at(pk, ch[k], am[k])
        pk[bad] = 0.0
        c0 = int(np.argmax(pk))
        if pk[c0] < qmin:
            return None
        p0 = POSITION_MM[c0]
        sel = k & (np.abs(POSITION_MM[ch] - p0) <= half_win * PITCH_MM)
        cc, ss, aa = ch[sel], sm[sel], am[sel]
        chs = np.unique(cc)
        if len(chs) < min_strips:
            return None
        W = np.zeros((len(chs), nsmp), np.float32); ci = np.searchsorted(chs, cc); m = (ss >= 0) & (ss < nsmp)
        W[ci[m], ss[m]] = aa[m]
        # an oscillating / hot strip carries no charge information and the fit censors it (H4 dead list + sps_reco
        # extra_dead): its samples are blanked so that nothing downstream (wft.model.init_guess, the brightest strip of the
        # window; sps_cluster) reads the oscillation as charge.  Measured effect (sps_audit, 10-31): it moves 568 of 3000
        # run71 raw700 Y fits and 18 of run55 d700, and leaves the ZS rows' residual 156-160.5 mm excess (2-8 %) as it was:
        # that excess is the oscillation's NEIGHBOURS (ch 374-376, ~155.4 mm, low charge, random time), not this strip.
        W[bad[chs]] = 0.0
        o = np.argsort(POSITION_MM[chs])
        q = np.where(bad[chs], 0.0, np.clip(W, 0, None).sum(1))
        rec[view] = dict(ch=chs[o].astype(np.int16), pos=POSITION_MM[chs[o]].astype(np.float32), W=W[o],
                         noise=np.full(len(chs), NOISE_ADC, np.float32))
        rec[f'ref_mesh_{view}'] = float((q * POSITION_MM[chs]).sum() / max(q.sum(), 1e-9))
        rec[f'seed_ch_{view}'] = c0
        cens = rec.get('_cens')
        rec[f'censor_{view}'] = [int(c) for c in chs[cens[chs]]] if cens is not None else []
    rec.pop('_cens', None)
    return rec


def _fill(out, files, a, D, plateaus, t_lo, ts0, scale, nsmp, verbose, base):
    """Add one acquisition's events to out ({plateau: {eid: event}}); base = index of its first file in the sub-run's
    sorted file list (event keys eid + 1e7 * file index, as before the per-acquisition split)."""
    import uproot
    for f in files:
        t = uproot.open(f)['nt']
        for lo in range(0, t.num_entries, CHUNK):
            hi = min(lo + CHUNK, t.num_entries)
            # the timestamps first (8 bytes an entry): skip a chunk that falls in no plateau still being filled, without
            # reading its waveforms (a RAW entry is ~100 kB; vt_all 10-31)
            tw = t_lo + (t.arrays(['timestamp'], entry_start=lo, entry_stop=hi, library='np')['timestamp'].astype(float)
                         - ts0) * scale
            if not any(((tw >= l0) & (tw < l1)).any() for n, l0, l1 in plateaus if len(out.get(n, {})) < a.per_plateau):
                continue
            arr = t.arrays(['eventId', 'timestamp', 'ftst', 'sample', 'channel', 'amplitude'], entry_start=lo,
                           entry_stop=hi, library='np')
            for j in range(len(tw)):
                L = next((n for n, l0, l1 in plateaus if l0 <= tw[j] < l1), None)
                if L is None or len(out.get(L, {})) >= a.per_plateau:
                    continue
                ch = arr['channel'][j].astype(int); sm = arr['sample'][j].astype(int); am = arr['amplitude'][j].astype(np.float32)
                rec = event_record(ch, sm, am, bool(D.get('raw', False)), nsmp, a.half_win, a.qmin, a.tan,
                                   seed_bad=a.seed_bad.get(L), couple=a.couple.get(L), keep_peaks=a.keep_peaks)
                ok = rec is not None
                if ok:
                    rec.update(eid=int(arr['eventId'][j]), t_wall=float(tw[j]), file=os.path.basename(f),
                               timestamp=int(arr['timestamp'][j]), ftst_x=int(arr['ftst'][j]), ftst_y=int(arr['ftst'][j]))
                if ok:
                    out.setdefault(L, {})[rec['eid'] + 10_000_000 * (base + files.index(f))] = rec
        # close (unmap) the file: an open uproot file stays mapped, and 37 RAW files of ~600 MB took a condor job past
        # 17 GB (vt_all run71, 2026-10-30)
        t.file.close()
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
