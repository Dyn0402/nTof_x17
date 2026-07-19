"""23e_mip_224503.py — SiPM-wall MIP (per arm) and plastic triple-coincidence MIP
on the long run 224503, to (a) recheck the arm-A-vs-B/C/D SiPM discrepancy with
high statistics and (b) test whether the real plastic MIP (~3.4 MeV, predicted
~120-217 mV on the Y-88 scale) shows up in triples — vs the spurious ~130 keVee
peak the old 19d triples found.

Standalone (does not need the HV-scan pipeline): 224503 is a single-HV beam run,
all bunches beam-on. Coincidence dt peaks measured on the data: wall-plastic
(t_wall - t_pss) ~ -31 ns, liq-plastic (t_liq - t_pss) ~ -38 ns.

Method (per arm, tagged/flag approach — the combinatorial rate is low enough per
20 ns window that a plastic hit rarely has a random partner):
  * late plastic hits (tof - tflash > 0.1 ms).
  * mark each plastic hit that has a WALL partner in the signal window
    |dt - dt_wp| <= W, and (separately) in a one-sided sideband; likewise a LIQ
    partner. Also mark each wall hit with a plastic partner (for the wall MIP).
  * SiPM wall MIP per channel = wall amp of wall hits with a plastic-signal
    partner, minus the sideband-tagged wall amp (scale = 2W / sideband_width).
  * plastic triple MIP = plastic amp with (wall & liq) signal tags, double
    sideband subtracted: SS - s_l*SB - s_w*BS + s_w*s_l*BB.
Amplitudes in mV (per-run DAQ factors). Linear bins to 400 mV (MIP ~150-260 mV).

Output: cache/23e_mip_run224503.npz, figures/21_y88/mip_224503.png
Usage: python 23e_mip_224503.py [n_bunches]   (default 20000 for speed)
"""

import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import hitcache
from adc_mv import mv_factors

BASE = Path(__file__).parent
RUN = Path.home() / 'x17' / 'beam_july' / 'data' / 'run224503.root'
OUT = BASE / 'figures' / '21_y88'
CACHE = BASE / 'cache'

LATE_TOF = 1e5
DT_WP, W_WP, SBWP_LO, SBWP_HI = -31.0, 10.0, 40.0, 140.0   # wall-plastic (ns)
DT_LQ, W_LQ, SBLQ_LO, SBLQ_HI = -38.0, 12.0, 40.0, 140.0   # liq-plastic (ns)
S_WP = (2 * W_WP) / (SBWP_HI - SBWP_LO)
S_LQ = (2 * W_LQ) / (SBLQ_HI - SBLQ_LO)
MV_EDGES = np.linspace(0, 400, 201)      # 2 mV bins
NBIN = len(MV_EDGES) - 1
# argv: <arm A|B|C|D | combine> [n_bunches]. One arm per process (the 224503
# per-arm working set is ~3 GB and CPython won't release it between arms, so
# each arm runs in a fresh process and its .npz is combined at the end).
ARG = sys.argv[1] if len(sys.argv) > 1 else 'combine'
N_BUNCH = int(sys.argv[2]) if len(sys.argv) > 2 else 4000


def tag_partners(kp, kx, tp, tx, dt0, w, sb_lo, sb_hi, n_pss,
                 xmv=None, xch=None):
    """Boolean (n_pss,) flags: does each plastic hit have an X-partner in the
    signal window |dt-dt0|<=w / one-sided sideband [sb_lo, sb_hi]? If xmv/xch
    are given (wall MIP), ALSO accumulate the X-hit amplitude per channel,
    incrementally per block (no index storage — keeps memory bounded)."""
    sig = np.zeros(n_pss, bool)
    sb = np.zeros(n_pss, bool)
    h_sig = np.zeros((8, NBIN)) if xmv is not None else None
    h_sb = np.zeros((8, NBIN)) if xmv is not None else None
    lo, hi = dt0 - w - 1, dt0 + sb_hi + 1
    for ri, oi in hitcache.iter_pairs(kp, kx, lo, hi, tp, tx):
        dt = tx[oi] - tp[ri] - dt0
        ms = np.abs(dt) <= w
        mb = (dt >= sb_lo) & (dt <= sb_hi)
        sig[ri[ms]] = True
        sb[ri[mb]] = True
        if xmv is not None:
            for mask, h in ((ms, h_sig), (mb, h_sb)):
                idx = oi[mask]
                ab = np.clip(np.digitize(xmv[idx], MV_EDGES) - 1, 0, NBIN - 1)
                h += np.bincount(xch[idx] * NBIN + ab,
                                 minlength=8 * NBIN).reshape(8, NBIN)
    return sig, sb, h_sig, h_sb


def process_arm(st, fac):
    print(f'  arm {st}: loading...', flush=True)
    gb = np.arange(N_BUNCH)                    # load only these bunches (memory)
    wal = hitcache.load(RUN, f'WAL{st}', ['BunchNumber', 'tof', 'detn', 'amp'], gb)
    pss = hitcache.load(RUN, f'PSS{st}', ['BunchNumber', 'tof', 'detn', 'amp', 'tflash'], gb)
    liq = hitcache.load(RUN, f'LIQ{st}', ['BunchNumber', 'tof', 'detn', 'amp'], gb)
    late = (pss['tof'] - pss['tflash']) > LATE_TOF
    pss = {k: v[late] for k, v in pss.items()}
    n_pss = len(pss['tof'])
    kp = hitcache.bunch_key(pss['BunchNumber'], pss['tof'])
    kw = hitcache.bunch_key(wal['BunchNumber'], wal['tof'])
    kl = hitcache.bunch_key(liq['BunchNumber'], liq['tof'])

    fw = fac[f'WAL{st}']
    ch_w = (wal['detn'] - 1).astype(int)
    wal_mv = wal['amp'] * fw[ch_w]
    w_sig, w_sb, wal_sig, wal_sb = tag_partners(
        kp, kw, pss['tof'], wal['tof'], DT_WP, W_WP, SBWP_LO, SBWP_HI, n_pss,
        xmv=wal_mv, xch=ch_w)
    l_sig, l_sb, _, _ = tag_partners(
        kp, kl, pss['tof'], liq['tof'], DT_LQ, W_LQ, SBLQ_LO, SBLQ_HI, n_pss)

    # --- plastic triple MIP (double sideband) per bar
    ch_p = (pss['detn'] - 1).astype(int)
    p_mv = pss['amp'] * np.array(fac[f'PSS{st}'])[ch_p]
    pss_net = np.zeros((2, NBIN))
    combos = [(w_sig, l_sig, 1.0), (w_sig, l_sb, -S_LQ),
              (w_sb, l_sig, -S_WP), (w_sb, l_sb, S_WP * S_LQ)]
    for b in range(2):
        for wf, lf, sc in combos:
            sel = wf & lf & (ch_p == b)
            pss_net[b] += sc * np.histogram(p_mv[sel], bins=MV_EDGES)[0]
    out = dict(wal_sig=wal_sig, wal_sb=wal_sb, pss_net=pss_net,
               n_pss=n_pss, n_wsig=int(w_sig.sum()))
    del wal, pss, liq, kp, kw, kl, wal_mv, p_mv, w_sig, w_sb, l_sig, l_sb, ch_p, ch_w
    gc.collect()
    return out


def run_one_arm(st):
    fac = mv_factors(RUN)
    print(f'run224503 arm {st}, first {N_BUNCH} bunches', flush=True)
    r = process_arm(st, fac)
    np.savez_compressed(CACHE / f'mip_run224503_{st}.npz', mv_edges=MV_EDGES,
                        wal_sig=r['wal_sig'], wal_sb=r['wal_sb'],
                        pss_net=r['pss_net'])
    print(f'arm {st}: {r["n_pss"]:,} late pss, {r["n_wsig"]:,} wall-sig tags '
          f'-> cache/mip_run224503_{st}.npz', flush=True)


def main():
    cen = 0.5 * (MV_EDGES[:-1] + MV_EDGES[1:])
    res = {}
    for st in 'ABCD':
        z = np.load(CACHE / f'mip_run224503_{st}.npz')
        res[st] = {k: z[k] for k in ('wal_sig', 'wal_sb', 'pss_net')}

    # ---- SiPM wall MIP peak per arm (subtracted, smoothed) ----
    kern = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2); kern /= kern.sum()

    def peak(sub, lo=8):
        sm = np.convolve(sub, kern, 'same')
        m = cen > lo
        return cen[m][np.argmax(sm[m])] if sm[m].max() > 0 else np.nan

    print('\n=== SiPM wall MIP peak per channel (mV), 224503 wall x plastic ===')
    wall_arm = {}
    for st in 'ABCD':
        pk = []
        for c in range(8):
            sub = res[st]['wal_sig'][c] - S_WP * res[st]['wal_sb'][c]
            pk.append(peak(sub))
        wall_arm[st] = np.nanmedian(pk)
        print(f'  WAL{st}: ' + ' '.join(f'{x:4.0f}' for x in pk)
              + f'   median {wall_arm[st]:.0f} mV')
    print(f'  --> per-arm median: A={wall_arm["A"]:.0f} B={wall_arm["B"]:.0f} '
          f'C={wall_arm["C"]:.0f} D={wall_arm["D"]:.0f} mV  '
          f'(B/C/D vs A ratio {np.median([wall_arm[a] for a in "BCD"])/wall_arm["A"]:.2f})')

    print('\n=== plastic triple-coincidence MIP peak (mV), 224503 ===')
    cal = json.loads((BASE / 'calib' / 'plastic_hv_gain_absolute.json').read_text())['pmts']
    for st in 'ABCD':
        for b in range(2):
            sub = res[st]['pss_net'][b]
            pmt = f'PSS{st}{b + 1}'
            pk = peak(sub, lo=30)
            pred = 3.4 * 1000 * cal[pmt]['mv_per_kevee_fifo']
            print(f'  {pmt}: triple-MIP peak {pk:5.0f} mV   (predicted ~{pred:.0f} mV)')

    figure(res, cen, wall_arm)
    print('\n-> cache/mip_run224503.npz & figures/21_y88/mip_224503.png')


def figure(res, cen, wall_arm):
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    # wall MIP per arm (channel-summed subtracted)
    for st in 'ABCD':
        sub = (res[st]['wal_sig'] - S_WP * res[st]['wal_sb']).sum(0)
        ax[0, 0].plot(cen, np.convolve(sub, np.ones(3) / 3, 'same'), label=f'WAL{st}')
    ax[0, 0].set_xlim(0, 120); ax[0, 0].set_xlabel('wall amp [mV]')
    ax[0, 0].set_ylabel('net coincidences'); ax[0, 0].legend()
    ax[0, 0].set_title('SiPM wall MIP per arm (x plastic, sideband-subtracted)')
    ax[0, 0].grid(alpha=0.3)
    # plastic triple MIP per arm (bar-summed)
    for st in 'ABCD':
        sub = res[st]['pss_net'].sum(0)
        ax[0, 1].plot(cen, np.convolve(sub, np.ones(3) / 3, 'same'), label=f'PSS{st}')
    ax[0, 1].axvspan(120, 217, color='gray', alpha=0.15, label='predicted MIP range')
    ax[0, 1].set_xlim(0, 400); ax[0, 1].set_xlabel('plastic amp [mV]')
    ax[0, 1].set_ylabel('net triples'); ax[0, 1].legend()
    ax[0, 1].set_title('Plastic triple-coincidence MIP (WALxPSSxLIQ)')
    ax[0, 1].grid(alpha=0.3)
    # per-arm wall MIP bar
    arms = list('ABCD')
    ax[1, 0].bar(arms, [wall_arm[a] for a in arms], color='steelblue')
    ax[1, 0].set_ylabel('wall MIP peak [mV]'); ax[1, 0].set_title('Wall MIP per arm')
    # plastic triple, individual bars for arm A
    for b in range(2):
        ax[1, 1].plot(cen, np.convolve(res['A']['pss_net'][b], np.ones(3) / 3, 'same'),
                      label=f'PSSA{b + 1}')
    ax[1, 1].set_xlim(0, 400); ax[1, 1].set_xlabel('plastic amp [mV]')
    ax[1, 1].legend(); ax[1, 1].set_title('Arm-A plastic triple MIP per bar')
    ax[1, 1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'mip_224503.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    if ARG in 'ABCD' and len(ARG) == 1:
        run_one_arm(ARG)
    else:
        main()
