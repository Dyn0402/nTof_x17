#!/usr/bin/env python3
"""Can a basis of templates binned by TAIL FRACTION fit the liquid pulses?

local_fit.py established that no single scaled template describes a liquid
pulse: the best reduced chi2 is ~70, i.e. the residual sits ~8x above the
noise. The PSD study says why -- the slow-component fraction is a continuum
(tail/total p16=0.13 to p84=0.24), not a single value and not two clean
classes. That is ordinary for a pulse-shape-discriminating scintillator seeing
a mix of particles and energies.

So the template basis should span the tail fraction, not the amplitude (which
is what we tried before, and amplitude is exactly the variable the fit already
scales out). This builds N templates at tail-fraction quantiles and measures
best-of-N against the same pulses, so the gain from each extra shape is
visible.

    python tail_basis.py <liq_pulses.npz> <outdir>
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_fit import fit_one, load_template, PRE, SHIPPED  # noqa: E402


def build(rows, psd, nbin):
    """One median template per tail-fraction quantile bin."""
    edges = np.percentile(psd, np.linspace(0, 100, nbin + 1))
    edges[0] -= 1e-9
    tm = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (psd > lo) & (psd <= hi)
        if m.sum() < 30:
            continue
        med = np.median(rows[m], axis=0)
        t = np.arange(len(med)) - int(np.argmax(med))
        tm.append((t, med / med.max()))
    return tm


def main():
    z = np.load(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for tree in ('LIQA', 'LIQB', 'LIQD'):
        if f'{tree}_rows' not in z.files:
            continue
        rows, amps, psd = z[f'{tree}_rows'], z[f'{tree}_amps'], z[f'{tree}_psd']
        sel = amps > 3000
        rows, psd = rows[sel], psd[sel]
        if len(rows) < 200:
            continue
        sigma = float(np.median(np.std(rows[:, :12], axis=1)))
        grid = np.arange(-4, 4.01, 0.25)
        # hold out half the pulses so the basis is not scored on its own data
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(rows))
        tr, te = idx[: len(idx) // 2], idx[len(idx) // 2:]

        print(f'\n{tree}: {len(rows)} pulses (amp>3000), '
              f'{len(tr)} to build / {len(te)} to score, sigma={sigma:.4f}')
        row = {}
        shipped = [load_template(SHIPPED / 'X17_LIQA_Signal_7.txt'),
                   load_template(SHIPPED / 'X17_LIQB_Signal_0.txt')]
        for name, tmpls in [('shipped pair', shipped)] + \
                [(f'tail-binned, {n} shapes', build(rows[tr], psd[tr], n))
                 for n in (1, 2, 3, 4, 6, 8)]:
            if not tmpls:
                continue
            c = np.array([min(fit_one(p, tt, ty, sigma, grid)
                              for tt, ty in tmpls)[0] for p in rows[te]])
            row[name] = float(np.median(c))
            print(f'   {name:26s} n_shapes={len(tmpls):2d}  '
                  f'chi2 p50={np.median(c):8.2f}  p90={np.percentile(c,90):9.2f}')
        results[tree] = row

    # what the basis looks like
    rows, amps, psd = z['LIQA_rows'], z['LIQA_amps'], z['LIQA_psd']
    sel = amps > 3000
    tm = build(rows[sel], psd[sel], 4)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for i, (t, y) in enumerate(tm):
        ax.plot(t, y, lw=1.0, label=f'tail-fraction bin {i + 1}')
    ax.set_yscale('log'); ax.set_ylim(1e-3, 1.4); ax.set_xlim(-15, 200)
    ax.set_xlabel('ns from peak'); ax.set_ylabel('normalised amplitude')
    ax.set_title('LIQA: templates binned by slow-component fraction')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out / 'liq_tail_basis.png', dpi=110)
    print(f'\nwrote {out}/liq_tail_basis.png')
    return 0


if __name__ == '__main__':
    sys.exit(main())
