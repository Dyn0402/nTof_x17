#!/usr/bin/env python3
"""
Cheap Monte Carlo of the Demokritos 2H(d,n)3He neutron field with the MX17
3He sphere + Micromegas box placed to scale.

Approximations (the "30% we leave for GEANT4"):
  - straight-line transport only: no scattering, no room return, no attenuation
  - d+d CM angular distribution parametrized as 1 + a2*P2 + a4*P4 (forward peaked)
  - constant Ed along the gas cell (energy loss in 1 atm D2 is ~tens of keV)
  - source = uniform line along the 3.7 cm cell, zero radius
Geometry (cm), beam along +z, cell exit (Pt stop) at z = 0:
  - gas cell: z in [-3.7, 0]
  - 3He sphere: radius R_SPH, center on axis at z = D_CAPS
  - Micromegas box: 50x50 cm planes forming a cube of half-size HALF
    centered on the sphere center; upstream plane has a beam-pipe hole
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LogNorm

# ----------------------- global parameters -----------------------
ED        = 2.0      # MeV deuteron energy  -> En(0deg) ~ 5.3 MeV
Q_DDN     = 3.269    # MeV, 2H(d,n)3He
M_D, M_N, M_HE3 = 1875.613, 939.565, 2808.391   # MeV
A2, A4    = 0.95, 0.30   # CM Legendre anisotropy (approx, Ed ~ 2 MeV)
S_TOT     = 5.0e7    # n/s total source strength (1 uA, anchored to 1e6 n/cm2/s @ 2 cm)
CELL_LEN  = 3.7      # cm
R_SPH     = 3.0      # cm  3He sphere radius
D_CAPS    = 4.0      # cm  sphere center distance from cell exit
HALF      = 25.0     # cm  box half-size (50x50 planes)
R_PIPE    = 2.0      # cm  beam pipe hole radius in upstream plane
N_MC      = 4_000_000
RNG       = np.random.default_rng(17)

def main():
    th, ph, en, z0 = sample_source(N_MC)
    d = direction(th, ph)
    hits = trace_to_box(z0, d)
    sphere = hits_sphere(z0, d)

    w = S_TOT / N_MC                       # n/s per MC particle
    print(f"d+d at Ed={ED} MeV: En(0)={en_lab(0):.2f}  En(90)={en_lab(np.pi/2):.2f} "
          f"En(180)={en_lab(np.pi):.2f} MeV")
    print(f"Source S = {S_TOT:.1e} n/s ({N_MC:.0e} MC particles)\n")
    print(f"{'surface':<22}{'n/s':>12}{'% of S':>9}{'peak n/cm2/s':>15}")
    tot = 0
    for name in ["downstream", "side+x", "side-x", "side+y", "side-y", "upstream"]:
        m = hits["plane"] == name
        n_s = w * m.sum()
        tot += n_s
        peak = peak_flux(hits, m, w)
        print(f"{name:<22}{n_s:12.3e}{100*m.sum()/N_MC:9.1f}{peak:15.3e}")
    print(f"{'beam-pipe hole':<22}{w*(hits['plane']=='hole').sum():12.3e}"
          f"{100*(hits['plane']=='hole').sum()/N_MC:9.1f}")
    print(f"{'TOTAL through box':<22}{tot:12.3e}{100*tot/S_TOT:9.1f}")
    print(f"\n{'3He sphere':<22}{w*sphere.sum():12.3e}{100*sphere.sum()/N_MC:9.1f}"
          f"   <En> on sphere = {en[sphere].mean():.2f} MeV "
          f"(spread {en[sphere].std():.2f})")
    print(f"{'sphere, En window':<22}", end="")
    core = sphere & (np.abs(en - en[sphere].mean()) < 0.25)
    print(f"{w*core.sum():12.3e}   (within +-250 keV of mean)")

    make_figures(th, en, z0, d, hits, sphere, w)

# ----------------------- physics -----------------------
def en_lab(theta_lab, Ed=ED):
    """Two-body 2H(d,n)3He lab neutron energy vs lab angle (non-relativistic)."""
    m1, m3, m4 = M_D, M_N, M_HE3
    A = np.sqrt(m1*m3*Ed)/(m3+m4)*np.cos(theta_lab)
    C = (m4*Q_DDN + (m4-m1)*Ed)/(m3+m4)
    r = A + np.sqrt(A*A + C)
    return r*r

def cm_to_lab(cos_cm, Ed=ED):
    """lab angle for given CM angle (massive two-body)."""
    m1, m3, m4 = M_D, M_N, M_HE3
    Et = Ed + Q_DDN
    g = np.sqrt(m1*m3*Ed/(m4*(m3+m4)*Et - m3*(m3+m4)*Ed + m1*m3*Ed)) * 0 + \
        np.sqrt(m1*m3*Ed)/np.sqrt(m4*((m3+m4)*Q_DDN + (m3+m4-m1)*Ed))
    sin_cm = np.sqrt(1-cos_cm**2)
    return np.arctan2(sin_cm, cos_cm + g)

def sample_source(n):
    """Sample CM angles from 1+a2*P2+a4*P4 by rejection, convert to lab."""
    out_cos = np.empty(n); filled = 0
    wmax = 1 + A2 + A4
    while filled < n:
        c = RNG.uniform(-1, 1, n)
        wgt = 1 + A2*0.5*(3*c*c-1) + A4*0.125*(35*c**4-30*c*c+3)
        keep = c[RNG.uniform(0, wmax, n) < wgt]
        take = min(len(keep), n-filled)
        out_cos[filled:filled+take] = keep[:take]
        filled += take
    th_lab = cm_to_lab(out_cos)
    ph = RNG.uniform(0, 2*np.pi, n)
    z0 = RNG.uniform(-CELL_LEN, 0.0, n)
    return th_lab, ph, en_lab(th_lab), z0

def direction(th, ph):
    return np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)], 1)

# ----------------------- geometry -----------------------
def trace_to_box(z0, d):
    """First intersection with the 6 box planes (box centered at z=D_CAPS)."""
    n = len(z0)
    src = np.zeros((n, 3)); src[:, 2] = z0
    zc = D_CAPS
    plane, px, py = np.full(n, "", dtype=object), np.zeros(n), np.zeros(n)
    best_t = np.full(n, np.inf)
    specs = [("downstream", 2, zc+HALF), ("upstream", 2, zc-HALF),
             ("side+x", 0, HALF), ("side-x", 0, -HALF),
             ("side+y", 1, HALF), ("side-y", 1, -HALF)]
    for name, ax, val in specs:
        t = (val - src[:, ax]) / d[:, ax]
        with np.errstate(invalid="ignore"):
            ok = t > 1e-9
        p = src + t[:, None]*d
        o1, o2 = [a for a in (0, 1, 2) if a != ax]
        lim1 = HALF if o1 != 2 else None
        in1 = (np.abs(p[:, o1]) <= HALF) if o1 != 2 else (np.abs(p[:, 2]-zc) <= HALF)
        in2 = (np.abs(p[:, o2]) <= HALF) if o2 != 2 else (np.abs(p[:, 2]-zc) <= HALF)
        ok &= in1 & in2 & (t < best_t)
        plane[ok] = name
        best_t[ok] = t[ok]
        px[ok], py[ok] = p[ok, o1], p[ok, o2]
    # beam-pipe hole in upstream plane
    up = plane == "upstream"
    hole = up & (px**2 + py**2 < R_PIPE**2)
    plane[hole] = "hole"
    return {"plane": plane, "u": px, "v": py}

def hits_sphere(z0, d):
    """Does the ray intersect the 3He sphere?"""
    oc = np.zeros((len(z0), 3)); oc[:, 2] = z0 - D_CAPS
    b = np.einsum("ij,ij->i", oc, d)
    disc = b*b - (np.einsum("ij,ij->i", oc, oc) - R_SPH**2)
    return (disc > 0) & (-b + np.sqrt(np.maximum(disc, 0)) > 0)

def peak_flux(hits, mask, w, bins=25):
    H, _, _ = np.histogram2d(hits["u"][mask], hits["v"][mask],
                             bins=bins, range=[[-HALF, HALF], [-HALF, HALF]])
    cell_area = (2*HALF/bins)**2
    return H.max()*w/cell_area

# ----------------------- figures -----------------------
def make_figures(th, en, z0, d, hits, sphere, w):
    fig = plt.figure(figsize=(13, 10))

    # (1) side view flux map, to scale, x-z plane
    ax1 = fig.add_subplot(2, 2, 1)
    g = np.linspace(-32, 42, 260)
    X, Z = np.meshgrid(g, g, indexing="ij")
    flux = np.zeros_like(X)
    zs = np.linspace(-CELL_LEN, 0, 12)
    for zsrc in zs:
        r2 = np.maximum(X**2 + (Z-zsrc)**2, 0.25)
        cth = np.nan_to_num((Z-zsrc)/np.sqrt(r2))
        thl = np.arccos(np.clip(cth, -1, 1))
        flux += np.nan_to_num(ang_weight_lab(thl))/r2
    flux *= S_TOT/(4*np.pi*len(zs)) * norm_factor()
    pc = ax1.pcolormesh(Z, X, flux, norm=LogNorm(vmin=1e2, vmax=1e7), cmap="inferno")
    fig.colorbar(pc, ax=ax1, label="n/cm$^2$/s (unscattered)")
    draw_geometry(ax1)
    ax1.set_xlabel("z [cm] (beam axis)"); ax1.set_ylabel("x [cm]")
    ax1.set_title(f"Side view to scale — d+d, Ed={ED} MeV, 1 µA")
    ax1.set_aspect("equal"); ax1.set_xlim(-32, 42); ax1.set_ylim(-32, 32)

    # (2) downstream plane hit map
    ax2 = fig.add_subplot(2, 2, 2)
    plane_map(ax2, hits, "downstream", w, fig)
    ax2.set_title("Flux on downstream MM plane (z=+29 cm)")

    # (3) side plane hit map
    ax3 = fig.add_subplot(2, 2, 3)
    plane_map(ax3, hits, "side+x", w, fig, sidemode=True)
    ax3.set_title("Flux on side MM plane (x=+25 cm)")

    # (4) En spectra
    ax4 = fig.add_subplot(2, 2, 4)
    b = np.linspace(0, en.max()*1.05, 120)
    ax4.hist(en[sphere], bins=b, weights=np.full(sphere.sum(), w),
             histtype="step", lw=2, label="entering 3He sphere")
    for name, ls in [("downstream", "-"), ("side+x", "--"), ("upstream", ":")]:
        m = hits["plane"] == name
        ax4.hist(en[m], bins=b, weights=np.full(m.sum(), w),
                 histtype="step", lw=1.2, ls=ls, label=name)
    ax4.set_xlabel("En [MeV]"); ax4.set_ylabel("n/s per bin")
    ax4.set_yscale("log"); ax4.legend(fontsize=8)
    ax4.set_title("Neutron energy by surface (kinematic spread)")

    fig.tight_layout()
    fig.savefig("/home/claude/demokritos/mx17_demokritos_flux.png", dpi=140)
    print("\nfigure saved: mx17_demokritos_flux.png")

def ang_weight_lab(th_lab):
    """Approximate lab angular weight by mapping through the CM dist."""
    cgrid = np.linspace(-1, 1, 2001)
    tlab = cm_to_lab(cgrid)
    wcm = 1 + A2*0.5*(3*cgrid**2-1) + A4*0.125*(35*cgrid**4-30*cgrid**2+3)
    # dOmega_cm/dOmega_lab jacobian folded numerically via histogram
    h, edges = np.histogram(tlab, bins=400, range=(0, np.pi), weights=wcm)
    hn, _ = np.histogram(tlab, bins=400, range=(0, np.pi))
    prof = h/np.maximum(hn, 1)
    sin = np.sin((edges[:-1]+edges[1:])/2)
    dens = prof  # relative intensity per solid angle (approx)
    idx = np.clip((th_lab/np.pi*400).astype(int), 0, 399)
    return dens[idx]

def norm_factor():
    thg = np.linspace(1e-3, np.pi-1e-3, 1000)
    wg = ang_weight_lab(thg)
    integ = np.trapezoid(wg*np.sin(thg), thg)*2*np.pi
    return 4*np.pi/integ

def draw_geometry(ax):
    ax.add_patch(Rectangle((-CELL_LEN, -0.5), CELL_LEN, 1.0, fc="none",
                           ec="cyan", lw=1.5))
    ax.add_patch(Rectangle((-30, -1.0), 26.3, 2.0, fc="none", ec="w", lw=0.8))
    ax.add_patch(Circle((D_CAPS, 0), R_SPH, fc="none", ec="lime", lw=2))
    ax.add_patch(Rectangle((D_CAPS-HALF, -HALF), 2*HALF, 2*HALF, fc="none",
                           ec="deepskyblue", lw=2, ls="--"))
    ax.annotate("gas cell", (-CELL_LEN, 1.5), color="cyan", fontsize=8)
    ax.annotate("3He", (D_CAPS-1.5, -R_SPH-3), color="lime", fontsize=9)
    ax.annotate("MM box", (D_CAPS-HALF+1, HALF-3.5), color="deepskyblue", fontsize=9)
    ax.annotate("beam pipe", (-29, 2.0), color="w", fontsize=8)

def plane_map(ax, hits, name, w, fig, sidemode=False):
    m = hits["plane"] == name
    bins = 50
    H, xe, ye = np.histogram2d(hits["u"][m], hits["v"][m], bins=bins,
                               range=[[-HALF, HALF], [-HALF, HALF]])
    cell = (2*HALF/bins)**2
    pc = ax.pcolormesh(xe, ye, (H.T*w/cell)+1e-2,
                       norm=LogNorm(vmin=1e1, vmax=1e5), cmap="viridis")
    fig.colorbar(pc, ax=ax, label="n/cm$^2$/s")
    if sidemode:
        ax.set_xlabel("y [cm]"); ax.set_ylabel("z' [cm] (box frame)")
    else:
        ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]")
    ax.set_aspect("equal")

main()
