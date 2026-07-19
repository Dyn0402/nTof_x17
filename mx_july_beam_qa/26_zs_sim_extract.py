#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
26_zs_sim_extract.py — simulate DREAM zero-suppression on run_55 no-ZS data.

Mechanism being simulated (from Tcm cfg + pedestal run 07-18-26_14-06-43,
taken 5 h before run_55):
  - Firmware ZS runs with pedestal subtraction ON: samples are normalized to
    the CmOffset (256) baseline using the static _ped.prg memory, then a
    channel crosses when sample > 256 + N*sigma_ch. The sigma_ch are the
    beam-off Std values in the _thr.aux; the deployed default is N = 5.00
    ("Sys PedRun Threshold").
  - On our RAW data the equivalent test is  max_raw_ch - ped_nat_ch >
    N*sigma_ch  with ped_nat measured from late-time (recovered) events of
    the run itself.  Using a STATIC ped_nat reproduces the real failure mode:
    post-flash baseline sag suppresses crossings.
  - ZsTyp=1 (tpc) + ZsChkSmp=4: sample-level readout; volume ≈ samples above
    threshold + 4 per crossing run + per-channel headers.

Per subrun outputs (cache/26_run55/<subrun>.npz):
  - per event × FEU: n channels kept and n samples above threshold, for each
    N in ZS_GRID; baseline shift (median raw - ped_nat median);
  - per MIP-track cluster (25's definition, x/y matched): n strips and
    n strips surviving each N (strip survival from the raw waveform maxima);
  - per-channel ped_nat and in-beam robust sigma (MAD) for comparison with
    the beam-off aux sigmas.

Run:  venv/bin/python mx_july_beam_qa/26_zs_sim_extract.py
"""

import glob
import os
import re
import sys

import numpy as np
import uproot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.Mx17StripMap import RunConfig  # noqa: E402

RUN_DIR = os.path.expanduser('~/x17/beam_july/runs/run_55')
PED_DIR = os.path.expanduser(
    '~/x17/beam_july/pedestals/pedestals_07-18-26_14-06-43/pedestals')
MAP_CSV = os.path.join(os.path.dirname(__file__), '..', 'mx17_m1_map.csv')
CACHE25 = os.path.join(os.path.dirname(__file__), 'cache', '25_run55')
CACHE = os.path.join(os.path.dirname(__file__), 'cache', '26_run55')

ZS_GRID = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0])
THR_HIT = 400.0        # 25's hit threshold (combined-hits clustering)
GAP_MM = 12.0
MIP_NMIN, MIP_NMAX, MIP_EXT = 3, 20, 25.0
SMP_SLACK = 2
NFEU = 8
NCH = 512
NSMP = 32
LATE_T_MS = 15.0       # events later than this define the natural pedestal
MAX_PED_EV = 300

DETS = ['mx17_A', 'mx17_B', 'mx17_C', 'mx17_D']


def parse_thr_aux(feu):
    """Beam-off per-channel (ped_norm, sigma) from the _thr.aux (last block)."""
    pat = re.compile(r'\s*(\d+)\s+D\d+\s+C\s*\d+\s+thr=\s*(\d+)\s.*'
                     r'Avr =\s*([\d.]+)\s+Std =\s*([\d.]+)')
    f = glob.glob(os.path.join(PED_DIR, f'*_{feu:02d}_thr.aux'))[0]
    sig = np.zeros(NCH)
    for line in open(f):
        m = pat.match(line)
        if m:
            sig[int(m.group(1))] = float(m.group(4))
    return sig


def load_feu_maxraw(subdir, feu, eid_ref):
    """max over samples of the raw waveform, (n_ev, NCH) int16, aligned to
    eid_ref; also the full raw array for pedestal/sample counting."""
    f = sorted(glob.glob(os.path.join(
        subdir, 'decoded_root', f'*_{feu:02d}.root')))[0]
    t = uproot.open(f)['nt']
    a = t.arrays(['eventId', 'channel', 'sample', 'amplitude'], library='np')
    eids = a['eventId'].astype(np.int64)
    n_ev = len(eid_ref)
    idx_of = {e: i for i, e in enumerate(eid_ref)}
    raw = np.zeros((n_ev, NCH, NSMP), np.int16)
    for k in range(len(eids)):
        i = idx_of.get(eids[k])
        if i is None:
            continue
        ch = a['channel'][k]
        smp = a['sample'][k]
        amp = a['amplitude'][k]
        raw[i][ch, smp] = amp
    return raw


def mip_cluster_strips(subdir, z25, pos_lut):
    """Re-run 25's clustering keeping strip channels of matched MIP clusters.

    Returns list of dicts: event index, det, plane, t_ms, cluster strips as
    (feu, ch) arrays, n, sum_amp.
    """
    fh = [f for f in glob.glob(os.path.join(
        subdir, 'combined_hits_root', '*combined_hits.root'))
        if '_pedestals_' not in f]
    h = uproot.open(fh[0])['hits'].arrays(
        ['eventId', 'feu', 'channel', 'amplitude', 'max_sample'],
        library='np')
    heid = h['eventId'].astype(np.int64)
    sel = h['amplitude'] > THR_HIT
    heid = heid[sel]
    hfeu = h['feu'][sel].astype(np.int16)
    hch = h['channel'][sel].astype(np.int16)
    hamp = h['amplitude'][sel].astype(float)
    hsmp = h['max_sample'][sel].astype(np.int32)
    lut = pos_lut[hfeu, hch]
    hdet, hplane, hpos = lut[:, 0], lut[:, 1], lut[:, 2]

    eid_ref = z25['eventId']
    t_ms = z25['t_ms']
    idx_of = {e: i for i, e in enumerate(eid_ref)}

    o = np.argsort(heid, kind='stable')
    heid, hfeu, hch, hamp, hsmp, hdet, hplane, hpos = (
        x[o] for x in (heid, hfeu, hch, hamp, hsmp, hdet, hplane, hpos))
    ue = np.unique(heid)
    lo = np.searchsorted(heid, ue)
    hi = np.append(lo[1:], len(heid))

    def clusters(pos, amp, smp, feu, ch):
        """gap clusters as index lists, dedup positions keep larger pulse"""
        if len(pos) == 0:
            return []
        o = np.argsort(pos, kind='stable')
        pos, amp, smp, feu, ch = pos[o], amp[o], smp[o], feu[o], ch[o]
        keep = np.ones(len(pos), bool)
        for i in range(1, len(pos)):
            if pos[i] == pos[i - 1]:
                keep[i if amp[i] < amp[i - 1] else i - 1] = False
        pos, amp, smp, feu, ch = (x[keep] for x in (pos, amp, smp, feu, ch))
        cuts = np.where(np.diff(pos) > GAP_MM)[0] + 1
        out = []
        for seg in np.split(np.arange(len(pos)), cuts):
            out.append(dict(
                n=len(seg), sum_amp=float(amp[seg].sum()),
                extent=float(pos[seg].max() - pos[seg].min()),
                smp_lo=int(smp[seg].min()), smp_hi=int(smp[seg].max()),
                feu=feu[seg], ch=ch[seg]))
        out.sort(key=lambda c: -c['sum_amp'])
        return out[:5]

    rows = []
    for e, l, r in zip(ue, lo, hi):
        i = idx_of.get(e)
        if i is None:
            continue
        for di in range(4):
            md = hdet[l:r] == di
            xs = clusters(hpos[l:r][md & (hplane[l:r] == 0)],
                          hamp[l:r][md & (hplane[l:r] == 0)],
                          hsmp[l:r][md & (hplane[l:r] == 0)],
                          hfeu[l:r][md & (hplane[l:r] == 0)],
                          hch[l:r][md & (hplane[l:r] == 0)])
            ys = clusters(hpos[l:r][md & (hplane[l:r] == 1)],
                          hamp[l:r][md & (hplane[l:r] == 1)],
                          hsmp[l:r][md & (hplane[l:r] == 1)],
                          hfeu[l:r][md & (hplane[l:r] == 1)],
                          hch[l:r][md & (hplane[l:r] == 1)])

            def mipok(c):
                return (MIP_NMIN <= c['n'] <= MIP_NMAX
                        and c['extent'] <= MIP_EXT)
            best = None
            for cx in xs:
                for cy in ys:
                    if not (mipok(cx) and mipok(cy)):
                        continue
                    ov = (min(cx['smp_hi'], cy['smp_hi'])
                          - max(cx['smp_lo'], cy['smp_lo']))
                    if ov < -SMP_SLACK:
                        continue
                    s = cx['sum_amp'] + cy['sum_amp']
                    if best is None or s > best[0]:
                        best = (s, cx, cy)
            if best is not None:
                for plane, c in [(0, best[1]), (1, best[2])]:
                    rows.append(dict(ev=i, det=di, plane=plane,
                                     t_ms=float(t_ms[i]), n=c['n'],
                                     sum_amp=c['sum_amp'],
                                     feu=c['feu'], ch=c['ch']))
    return rows


def process_subrun(subdir, sig_aux, pos_lut):
    name = os.path.basename(subdir.rstrip('/'))
    z25 = np.load(os.path.join(CACHE25, name + '.npz'))
    eid_ref = z25['eventId']
    t_ms = z25['t_ms']
    n_ev = len(eid_ref)
    late = (t_ms > LATE_T_MS) & ~z25['is_first']

    nZ = len(ZS_GRID)
    n_ch_kept = np.zeros((n_ev, NFEU, nZ), np.int16)
    n_smp_above = np.zeros((n_ev, NFEU, nZ), np.int32)
    base_shift = np.zeros((n_ev, NFEU), np.float32)
    ped_nat = np.zeros((NFEU, NCH), np.float32)
    sig_nat = np.zeros((NFEU, NCH), np.float32)
    maxraw_all = np.zeros((n_ev, NFEU, NCH), np.int16)

    n_ch_kept_nocm = np.zeros((n_ev, NFEU), np.int16)   # N=5, no CM (ref)

    for feu in range(1, NFEU + 1):
        raw = load_feu_maxraw(subdir, feu, eid_ref)  # (n_ev, NCH, NSMP)
        li = np.where(late)[0][:MAX_PED_EV]
        sub = raw[li].astype(np.float32)
        ped = np.median(sub, axis=(0, 2))
        ped_nat[feu - 1] = ped
        del sub
        sg = sig_aux[feu - 1].copy()
        sg[sg <= 0] = np.inf                        # dead in aux: never kept

        res = raw.astype(np.float32) - ped[None, :, None]
        base_shift[:, feu - 1] = np.median(res, axis=(1, 2))
        # no-CM reference at N=5 (the as-configured firmware would do this)
        n_ch_kept_nocm[:, feu - 1] = (
            res.max(axis=2) > 5.0 * sg[None, :]).sum(axis=1)
        # firmware-style common-mode correction: per Dream chip (64 ch),
        # per sample, median over channels — signal bias in blob events is
        # faithful to what the FEU would do
        r4 = res.reshape(n_ev, 8, 64, NSMP)
        cm = np.median(r4, axis=2)                  # (n_ev, dream, smp)
        res -= np.repeat(cm, 64, axis=1).reshape(n_ev, NCH, NSMP)
        del r4, cm
        # residual (post-CM) in-beam noise from late events
        mad = np.median(np.abs(res[li]), axis=(0, 2))
        sig_nat[feu - 1] = 1.4826 * mad
        mx = res.max(axis=2)                        # (n_ev, NCH) CM-corrected
        maxraw_all[:, feu - 1] = np.clip(mx, -32000, 32000).astype(np.int16)
        for k, N in enumerate(ZS_GRID):
            kept = mx > N * sg[None, :]
            n_ch_kept[:, feu - 1, k] = kept.sum(axis=1)
            above = (res > (N * sg)[None, :, None]).sum(axis=2)
            n_smp_above[:, feu - 1, k] = (above * kept).sum(axis=1)
        del raw, res

    # MIP-cluster strip survival
    rows = mip_cluster_strips(subdir, z25, pos_lut)
    cl_meta = np.zeros((len(rows), 6), np.float32)  # ev,det,plane,t,n,sum
    cl_surv = np.zeros((len(rows), nZ), np.int16)
    for j, rrow in enumerate(rows):
        i = rrow['ev']
        cl_meta[j] = (i, rrow['det'], rrow['plane'], rrow['t_ms'],
                      rrow['n'], rrow['sum_amp'])
        for k, N in enumerate(ZS_GRID):
            surv = 0
            for feu, ch in zip(rrow['feu'], rrow['ch']):
                sg = sig_aux[feu - 1][ch]
                if sg <= 0:
                    continue
                # maxraw_all is already the CM-corrected excess over pedestal
                if maxraw_all[i, feu - 1, ch] > N * sg:
                    surv += 1
            cl_surv[j, k] = surv

    np.savez_compressed(
        os.path.join(CACHE, name + '.npz'),
        zs_grid=ZS_GRID, t_ms=t_ms, is_first=z25['is_first'],
        resist_v=int(z25['resist_v']), cycle=int(z25['cycle']),
        n_ch_kept=n_ch_kept, n_smp_above=n_smp_above,
        n_ch_kept_nocm5=n_ch_kept_nocm,
        base_shift=base_shift, ped_nat=ped_nat, sig_nat=sig_nat,
        cl_meta=cl_meta, cl_surv=cl_surv)
    print(f'{name}: {n_ev} ev, {len(rows)} MIP clusters, '
          f'kept@3sig b1 median '
          f'{np.median(n_ch_kept[(t_ms>6)&(t_ms<14)][:, :, 2].sum(axis=1)):.0f}'
          f'/4096 ch')


def main():
    os.makedirs(CACHE, exist_ok=True)
    sig_aux = np.stack([parse_thr_aux(feu) for feu in range(1, NFEU + 1)])
    print('aux sigmas: p50 %.1f  frac>40 %.3f'
          % (np.median(sig_aux[sig_aux > 0]),
             (sig_aux > 40).mean()))

    rc = RunConfig(os.path.join(RUN_DIR, 'run_config.json'), MAP_CSV)
    pos_lut = np.full((9, NCH, 3), np.nan)
    for di, det in enumerate(DETS):
        d = rc.get_detector(det)
        for feu in sorted(set(v[0] for v in d.dream_feus.values())):
            for ch in range(NCH):
                x, y = d.map_hit(feu, ch)
                if x is not None:
                    pos_lut[feu, ch] = (di, 0, x)
                elif y is not None:
                    pos_lut[feu, ch] = (di, 1, y)

    subs = sorted(glob.glob(os.path.join(RUN_DIR, 'scintd_*')))
    # optional sharding: 26_zs_sim_extract.py <shard> <nshards>
    if len(sys.argv) == 3:
        shard, nsh = int(sys.argv[1]), int(sys.argv[2])
        subs = subs[shard::nsh]
    todo = [s for s in subs if not os.path.exists(
        os.path.join(CACHE, os.path.basename(s) + '.npz'))]
    print(f'{len(subs)} subruns, {len(todo)} to do')
    for s in todo:
        process_subrun(s, sig_aux, pos_lut)
    print('donzo')


if __name__ == '__main__':
    main()
