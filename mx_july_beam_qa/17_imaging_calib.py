"""
17_imaging_calib.py — Image the plastics on the SiPM wall and turn the MIP
coincidences into a geometry-based calibration.

Premise (tested, not assumed): tracks come from the He-3 source at the origin.
A source track through the wall at transverse (u,v) hits at incidence angle θ with
cosθ = R_wall/|r|, so a MIP deposits E = dE/dx · t_scint/cosθ in the 3 mm bar.
The geometric mean √(A_top·A_bot) removes attenuation, so its MIP peak (MPV)
∝ gain · lightyield · E(u,v).  Therefore:
  * within a group (u, gain fixed) MPV(v) must follow 1/cosθ(u,v)  -> geometry check
  * MPV / (1/cosθ) per group  -> relative SiPM gain map (the calibration)
  * MPV / E_MeV(u,v)          -> absolute ADC/MeV under the MIP assumption

Also images each plastic bar onto the wall (u-group × v) and overlays the
source-projected plastic footprint.

Uses cache/16_vertical_<run>.npz (needs the geov array) + mx17_geom.py.
Usage: python 17_imaging_calib.py [run_stem]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import mx17_geom as G

RUN_STEM = sys.argv[1] if len(sys.argv) > 1 else 'run224404'
BASE = Path(__file__).parent
OUT = BASE / 'figures' / '17_imaging'
OUT.mkdir(parents=True, exist_ok=True)
V_EFF = 15.0

d = np.load(BASE / 'cache' / f'16_vertical_{RUN_STEM}.npz')
sc = float(d['sb_scale'])
A2 = d['a2_edges']; AC = np.sqrt(A2[:-1] * A2[1:])
VDT = d['vdt_edges']; VDTC = 0.5 * (VDT[:-1] + VDT[1:])
GU = G.group_u_centers()
KERN = np.exp(-0.5 * (np.arange(-5, 6) / 2.0) ** 2); KERN /= KERN.sum()


def geov(st):
    """sideband-subtracted geo-mean spectrum, (group, bar, vdt, amp)."""
    h = d[f'{st}_geov']
    return h[0] - sc * h[1]


def mpv(spec, win=4):
    """Sub-bin MIP peak (MPV) from a geo-mean spectrum: weighted parabola fit in
    log-amp over +-win bins around the smoothed maximum."""
    sm = np.convolve(np.clip(spec, 0, None), KERN, mode='same')
    m = AC > 150
    x, y = AC[m], sm[m]
    if y.max() <= 0:
        return np.nan
    i = int(np.argmax(y))
    lo, hi = max(i - win, 0), min(i + win + 1, len(x))
    lx, yy = np.log(x[lo:hi]), y[lo:hi]
    if len(lx) < 3 or yy.max() <= 0:
        return float(x[i])
    a, b, c = np.polyfit(lx, yy, 2, w=yy)      # weight by counts
    if a >= 0:
        return float(x[i])
    return float(np.exp(-b / (2 * a)))


def skew(st, g):
    """Cable-skew dt (=v-0 point) from the count-weighted dt centroid of group g."""
    prof = np.clip(geov(st)[g].sum(axis=(0, 2)), 0, None)   # over bar & amp
    return np.average(VDTC, weights=prof) if prof.sum() > 0 else 0.0


def v_cm(st, g):
    return (V_EFF / 2) * (VDTC - skew(st, g))


# ---- (1) imaging: u (bar-fraction vs group) and v (fine profile) -------------
DT_FINE = d['dt_edges']; DTFC = 0.5 * (DT_FINE[:-1] + DT_FINE[1:])   # 0.25 ns bins


def v_profile(st):
    """Fine sideband-subtracted v-distribution per group, mapped to a common
    v-grid and summed (uses the 0.25 ns h_dt histogram)."""
    hd = d[f'{st}_dt']                       # (2,4,NDT)
    sub = hd[0] - sc * hd[1]
    vgrid = np.linspace(-30, 30, 25)         # 2.5 cm bins (> 1.9 cm dt spacing)
    prof = np.zeros(len(vgrid) - 1)
    for g in range(4):
        sk = skew(st, g)
        vv = (V_EFF / 2) * (DTFC - sk)
        prof += np.histogram(vv, bins=vgrid, weights=np.clip(sub[g], 0, None))[0]
    k = np.array([0.25, 0.5, 0.25])
    prof = np.convolve(prof, k, mode='same')
    return vgrid, prof


def fig_imaging(st='C'):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    # (a) u-imaging: bar-2 fraction per group vs geometry crossover
    gb = d[f'{st}_geobar']
    M = (gb[0] - sc * gb[1]).sum(axis=2)     # (group, bar)
    frac2 = M[:, 1] / M.sum(axis=1)
    axes[0].plot(GU, frac2, 'o-', ms=8, color='#c2410c', label='data: bar-2 fraction')
    # geometry: wall-u where the plastic bar boundary projects
    pc = G.plastic_u_centers(st); boundary = pc.mean()
    u_cross = boundary * G.R_wall(st) / G.R_plastic(st)
    axes[0].axvline(u_cross, color='#2563eb', ls='--', lw=1.5,
                    label=f'geom. crossover u={u_cross:.1f} cm')
    axes[0].axhline(0.5, color='#999', lw=0.8)
    axes[0].set_xlabel('wall group u-center [cm]'); axes[0].set_ylabel('+u plastic-bar fraction')
    axes[0].set_title(f'arm {st}: U-imaging (plastic position)')
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.25)
    # (b) v-imaging: fine v-profile vs plastic v-projection
    vgrid, prof = v_profile(st)
    vc = 0.5 * (vgrid[:-1] + vgrid[1:])
    axes[1].step(vc, prof / prof.max(), where='mid', color='#c2410c', lw=1.5)
    pj = G.project_plastic_to_wall(st)
    for s in (-1, 1):
        axes[1].axvline(s * pj['v_half'], color='#2563eb', ls='--', lw=1.5,
                        label='plastic v-edge (proj)' if s == 1 else None)
        axes[1].axvline(s * 25, color='#16a34a', ls=':', lw=1.2,
                        label='wall bar end' if s == 1 else None)
    axes[1].set_xlabel('v = beam-axis position [cm]'); axes[1].set_ylabel('coincidences (norm)')
    axes[1].set_title(f'arm {st}: V-imaging (plastic height, res.-limited)')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25); axes[1].set_xlim(-30, 30)
    fig.suptitle(f'{RUN_STEM}: imaging the plastics onto the SiPM wall from the source')
    fig.tight_layout()
    fig.savefig(OUT / f'imaging_{st}.png', dpi=140)
    plt.close(fig)


def fig_calibration():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(4); w = 0.35
    for ax, st in zip(axes, 'BC'):
        gv = geov(st)
        mpv0, adcmev, edep = [], [], []
        for g in range(4):
            vv = v_cm(st, g); i0 = np.argmin(np.abs(vv))
            spec = gv[g, :, i0, :].sum(axis=0)
            pk = mpv(spec); e0 = G.mip_edep_MeV(st, GU[g], 0.0)
            mpv0.append(pk); adcmev.append(pk / e0); edep.append(e0)
        mpv0 = np.array(mpv0); adcmev = np.array(adcmev)
        relg = adcmev / adcmev.mean()
        ax.bar(x - w/2, mpv0, w, color='#93c5fd', label='raw MIP peak [ADC]')
        ax.bar(x + w/2, adcmev, w, color='#c2410c',
               label='geom-corrected ADC/MeV')
        for i in range(4):
            ax.text(x[i] + w/2, adcmev[i] + 20, f'{relg[i]:.2f}', ha='center', fontsize=8)
        ax.set_xticks(x, [f'g{g+1}\n{GU[g]:+.0f}cm\n1/cosθ={1/G.cos_theta(st,GU[g],0):.2f}'
                          for g in range(4)], fontsize=8)
        ax.set_title(f'arm {st}: gains within ±{100*np.abs(relg-1).max():.0f}%'
                     f'  (mean {adcmev.mean():.0f} ADC/MeV)')
        ax.legend(fontsize=8); ax.grid(alpha=0.2, axis='y')
    fig.suptitle(f'{RUN_STEM}: geometry-corrected wall calibration '
                 '(number above bar = relative SiPM gain)')
    fig.tight_layout()
    fig.savefig(OUT / 'calibration_bars.png', dpi=140)
    plt.close(fig)


# ---- (2) MPV(v) within each group vs 1/cosθ ----------------------------------
def fig_mpv_vs_geometry(st='C'):
    gv = geov(st)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for g in range(4):
        vv = v_cm(st, g)
        col = plt.get_cmap('viridis')(g / 3)
        # MPV per v bin (require enough counts)
        pk, vkeep = [], []
        for iv in range(len(VDTC)):
            spec = gv[g, :, iv, :].sum(axis=0)
            if np.clip(spec, 0, None).sum() < 2000:
                continue
            pk.append(mpv(spec)); vkeep.append(vv[iv])
        pk, vkeep = np.array(pk), np.array(vkeep)
        m = np.abs(vkeep) <= 13
        pk, vkeep = pk[m], vkeep[m]
        inv_cos = 1 / G.cos_theta(st, GU[g], vkeep)
        # normalize measured MPV to its value at v≈0 for shape comparison
        i0 = np.argmin(np.abs(vkeep))
        axes[0].plot(vkeep, pk / pk[i0], 'o-', color=col, ms=3, lw=1,
                     label=f'g{g+1} (u={GU[g]:+.0f})')
        axes[0].plot(vkeep, inv_cos / inv_cos[i0], '--', color=col, lw=1)
    axes[0].set_xlabel('v [cm]'); axes[0].set_ylabel('MPV(v) / MPV(0)')
    axes[0].set_title(f'arm {st}: MIP peak vs v  (points=data, dashed=1/cosθ)')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)

    # (2b) MPV vs 1/cosθ scatter, all groups+v pooled -> should be linear thru 0-ish
    allx, ally, allc = [], [], []
    for g in range(4):
        vv = v_cm(st, g)
        for iv in range(len(VDTC)):
            spec = gv[g, :, iv, :].sum(axis=0)
            if np.clip(spec, 0, None).sum() < 2000 or abs(vv[iv]) > 13:
                continue
            allx.append(1 / G.cos_theta(st, GU[g], vv[iv]))
            ally.append(mpv(spec)); allc.append(g)
    allx, ally, allc = np.array(allx), np.array(ally), np.array(allc)
    for g in range(4):
        m = allc == g
        axes[1].scatter(allx[m], ally[m], s=10, color=plt.get_cmap('viridis')(g / 3),
                        label=f'g{g+1}')
    axes[1].set_xlabel('1/cosθ  (geometry)'); axes[1].set_ylabel('MIP peak MPV [ADC]')
    axes[1].set_title('MPV vs geometry — slope per group = gain·LY·E$_0$')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)
    fig.suptitle(f'{RUN_STEM}: does the MIP peak track the geometric path length?')
    fig.tight_layout()
    fig.savefig(OUT / f'mpv_vs_geometry_{st}.png', dpi=140)
    plt.close(fig)


# ---- (3) calibration: relative gain and ADC/MeV per group --------------------
def report_calibration(st):
    gv = geov(st)
    print(f'\narm {st}: geometry-corrected calibration')
    print(f'{"group":6s} {"u[cm]":>6s} {"MPV0[ADC]":>10s} {"E0[MeV]":>8s} '
          f'{"ADC/MeV":>9s} {"rel.gain":>9s}')
    adcmev = []
    for g in range(4):
        vv = v_cm(st, g)
        i0 = np.argmin(np.abs(vv))
        spec = gv[g, :, i0, :].sum(axis=0)
        pk = mpv(spec)
        e0 = G.mip_edep_MeV(st, GU[g], 0.0)
        adcmev.append(pk / e0)
    adcmev = np.array(adcmev)
    for g in range(4):
        vv = v_cm(st, g); i0 = np.argmin(np.abs(vv))
        spec = gv[g, :, i0, :].sum(axis=0); pk = mpv(spec)
        e0 = G.mip_edep_MeV(st, GU[g], 0.0)
        print(f'g{g+1:<5d} {GU[g]:>6.1f} {pk:>10.0f} {e0:>8.3f} {pk/e0:>9.0f} '
              f'{adcmev[g]/adcmev.mean():>9.3f}')


if __name__ == '__main__':
    for st in 'BC':
        fig_imaging(st)
        report_calibration(st)
    fig_calibration()
    print(f'\nFigures -> {OUT}')
