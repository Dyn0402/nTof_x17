"""T1: does the measured template depend on the track angle or the pulse amplitude?

Same algorithm as wft.calibrate.measure_templates (2 brightest strips per event, t50 alignment, median), binned in
|tan| (M3) and in strip amplitude.  If the template were the electronics response alone, every bin would agree; if it
carries the strip's own drift segment, its width shrinks with |tan| as pitch / (v |tan|).
Input: the rr1 calibration caches (model_audit/ws_t0/caches/<det>.pkl, 1500 events each, waveform alignment).
Usage: python -I t1_tan_amp.py [det ...]   -> results/t1_<det>.json, figures/t1_<det>.png"""
import json, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.expanduser('~/x17/cosmic_bench/model_audit/ws_t0/caches/{}.pkl')
GRID = np.arange(-360, 1400, 10.0)
SAT = 3550.0
TAN_BINS = [(0.0, 0.08), (0.08, 0.15), (0.15, 0.22), (0.22, 0.30), (0.30, 0.40), (0.40, 0.70)]
AMP_BINS = [(500, 1000), (1000, 1800), (1800, 2600), (2600, 3500)]


def t50(w):
    ipk = int(np.argmax(w)); a = w[ipk]
    for k in range(1, ipk + 1):
        if w[k] >= 0.5 * a > w[k - 1]:
            return k - 1 + (0.5 * a - w[k - 1]) / (w[k] - w[k - 1])
    return np.nan


def pulses(events, plane, nbright=2):
    """(tan, amp, aligned normalised shape) for the brightest strips, measure_templates' cuts minus the tan cut."""
    out = []
    for ev in events.values():
        if plane not in ev:
            continue
        W = np.asarray(ev[plane]['W'], np.float32); ns = W.shape[1]
        for i in np.argsort(W.max(axis=1))[::-1][:nbright]:
            w = W[i]; a = float(w.max()); ipk = int(np.argmax(w))
            if a < 500 or a > SAT or ipk < 6 or ipk > ns - 12:
                continue
            c = t50(w)
            if np.isfinite(c):
                tt = (np.arange(ns) - c) * 60.0
                out.append((abs(ev[f'tan_{plane}']), a, np.interp(GRID, tt, w / a, left=np.nan, right=np.nan)))
    return out


def shape_stats(acc):
    t = np.nanmedian(np.array(acc), axis=0)
    t = t - np.nanmedian(t[GRID < -250]); t = np.nan_to_num(t); t /= t.max()
    ipk = int(np.argmax(t))
    r10 = GRID[np.argmax(t >= 0.1)]; r90 = GRID[np.argmax(t >= 0.9)]
    above = np.where(t >= 0.5)[0]
    # interpolated FWHM
    def cross(i0, i1):
        return GRID[i0] + (0.5 - t[i0]) / (t[i1] - t[i0]) * 10.0
    lo = cross(above[0] - 1, above[0]); hi = cross(above[-1], above[-1] + 1)
    return t, dict(n=len(acc), rise=float(r90 - r10), peak=float(GRID[ipk]), fwhm=float(hi - lo),
                   under=float(t.min()), tail800=float(np.interp(800, GRID, t)))


def boot_err(acc, key, nb=40, rng=np.random.default_rng(1)):
    acc = np.array(acc); vals = []
    for _ in range(nb):
        vals.append(shape_stats(acc[rng.integers(0, len(acc), len(acc))])[1][key])
    return float(np.std(vals))


def main(dets):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    os.makedirs(f'{HERE}/results', exist_ok=True); os.makedirs(f'{HERE}/figures', exist_ok=True)
    for det in dets:
        ev = pickle.load(open(CACHE.format(det), 'rb'))
        res = {}
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        for j, plane in enumerate('xy'):
            P = pulses(ev, plane)
            tn = np.array([p[0] for p in P]); am = np.array([p[1] for p in P]); S = [p[2] for p in P]
            res[plane] = {'tan': [], 'amp': []}
            for lo, hi in TAN_BINS:
                sel = [S[i] for i in np.where((tn >= lo) & (tn < hi))[0]]
                if len(sel) < 25:
                    continue
                t, st = shape_stats(sel)
                st.update(lo=lo, hi=hi, fwhm_err=boot_err(sel, 'fwhm'), rise_err=boot_err(sel, 'rise'))
                res[plane]['tan'].append(st)
                axs[0, j].plot(GRID, t, label=f'|tan| {lo:.2f}-{hi:.2f} n={st["n"]} FWHM {st["fwhm"]:.0f}')
            # amplitude bins on the inclined tracks only (the production selection)
            incl = tn >= 0.22
            for lo, hi in AMP_BINS:
                sel = [S[i] for i in np.where(incl & (am >= lo) & (am < hi))[0]]
                if len(sel) < 20:
                    continue
                t, st = shape_stats(sel)
                st.update(lo=lo, hi=hi, fwhm_err=boot_err(sel, 'fwhm'), rise_err=boot_err(sel, 'rise'))
                res[plane]['amp'].append(st)
                axs[1, j].plot(GRID, t, label=f'amp {lo}-{hi} n={st["n"]} FWHM {st["fwhm"]:.0f}')
            for i in (0, 1):
                axs[i, j].set_xlim(-300, 1300); axs[i, j].axhline(0, c='k', lw=.5)
                axs[i, j].legend(fontsize=7); axs[i, j].set_xlabel('ns after t50')
            axs[0, j].set_title(f'{det} {plane}: by M3 |tan|'); axs[1, j].set_title(f'{det} {plane}: by amplitude, |tan|>0.22')
        fig.tight_layout(); fig.savefig(f'{HERE}/figures/t1_{det}.png', dpi=90); plt.close(fig)
        json.dump(res, open(f'{HERE}/results/t1_{det}.json', 'w'), indent=1)
        for plane in 'xy':
            print(f'== {det} {plane}')
            for st in res[plane]['tan']:
                print(f"  tan {st['lo']:.2f}-{st['hi']:.2f} n={st['n']:4d} rise {st['rise']:4.0f}±{st['rise_err']:.0f} "
                      f"FWHM {st['fwhm']:4.0f}±{st['fwhm_err']:.0f} peak {st['peak']:4.0f} under {st['under']:+.3f} t800 {st['tail800']:+.3f}")
            for st in res[plane]['amp']:
                print(f"  amp {st['lo']:4d}-{st['hi']:4d} n={st['n']:4d} rise {st['rise']:4.0f}±{st['rise_err']:.0f} "
                      f"FWHM {st['fwhm']:4.0f}±{st['fwhm_err']:.0f} peak {st['peak']:4.0f} under {st['under']:+.3f} t800 {st['tail800']:+.3f}")


if __name__ == '__main__':
    main(sys.argv[1:] or ['det2', 'det3', 'det4', 'det6', 'det7'])
