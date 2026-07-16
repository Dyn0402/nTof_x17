"""
mx17_geom.py — MX17 (n_TOF EAR2 X17) geometry for wall/plastic imaging & MIP
calibration, transcribed from the Geant4 sim
(~/CLionProjects/MX17_Full_Geant, include/SimConfig.hh + src/DetectorConstruction.cc).

Frame (per arm, "local"):
  u = in-plane transverse (bar-across),  v = +Y = BEAM axis (bar length),
  w = radial outward from the beam axis (depth into the stack).
Origin = He-3 target centre = the "source".  Tracks are assumed to originate
there (small D20xL40 mm cylinder).

Radial build-up (w measured from the MM drift-mylar FRONT face):
  SiPM scint centre = sipm_front_from_mylar + sipm_container_depth/2
  plastic scint centre = sipm_front_from_mylar + sipm_container_depth
                          + gap_sipm_to_plastic + plastic_env_depth/2
Mylar front sits at mm_distance_{x,z} from the origin, so the radial distance of
a plane from the beam axis is mm_distance + (w above).
"""

import numpy as np

# ---- SimConfig defaults (cm) --------------------------------------------------
MM_DIST_X = 20.40          # +/-X arms (B, D) mylar front from origin
MM_DIST_Z = 20.45          # +/-Z arms (A, C)
SIPM_FRONT_FROM_MYLAR = 11.0
SIPM_CONTAINER_DEPTH = 3.3
SIPM_SCINT_THICK = 0.3     # active scint depth (3 mm)
SIPM_BAR_WIDTH = 2.5
SIPM_N_BARS = 20
SIPM_N_READOUT = 16
SIPM_READOUT_SHIFT_BARS = 1
SIPM_SIZE_V = 50.0         # bar length along beam

GAP_SIPM_TO_PLASTIC = 7.0
BSC_U = 20.0               # plastic bar width (u)
BSC_V = 30.0               # plastic bar length along beam (v)
BSC_THICK = 2.5
BSC_GAP = 0.3
BSC_TAPE_UM = 200.0
BSC_AL_UM = 20.0
PINWHEEL = {'D': 1.55, 'B': 1.575, 'A': 1.635, 'C': 1.73}   # tangential MM shift

# ---- derived radial distances (cm) -------------------------------------------
SIPM_SCINT_W = SIPM_FRONT_FROM_MYLAR + SIPM_CONTAINER_DEPTH / 2      # 12.65
SIPM_BACK_W = SIPM_FRONT_FROM_MYLAR + SIPM_CONTAINER_DEPTH           # 14.30
_plastic_env_half = BSC_THICK / 2 + BSC_AL_UM * 1e-4 + BSC_TAPE_UM * 1e-4
PLASTIC_W = SIPM_BACK_W + GAP_SIPM_TO_PLASTIC + _plastic_env_half    # ~22.57


def mm_dist(arm):
    return MM_DIST_X if arm in 'BD' else MM_DIST_Z


def R_wall(arm):
    """Radial distance of the SiPM scint plane from the beam axis [cm]."""
    return mm_dist(arm) + SIPM_SCINT_W


def R_plastic(arm):
    return mm_dist(arm) + PLASTIC_W


# ---- wall bar / group u-centres (structure frame, u=0 at mechanical centre) ---
def bar_u_centers():
    """u-centre of every instrumented bar (16), structure frame [cm]."""
    bar_half = (SIPM_N_READOUT - 1) / 2.0
    bar_center = (SIPM_N_BARS - 1) / 2.0 - SIPM_READOUT_SHIFT_BARS
    lo = int(round(bar_center - bar_half))
    hi = int(round(bar_center + bar_half))
    idx = np.arange(lo, hi + 1)                              # 1..16
    return (idx - (SIPM_N_BARS - 1) / 2.0) * SIPM_BAR_WIDTH  # [-21.25 .. +16.25]


def group_u_centers():
    """4 groups of 4 ganged bars -> u-centre of each group [cm], left->right."""
    bu = bar_u_centers()
    return np.array([bu[4 * g:4 * g + 4].mean() for g in range(4)])  # [-17.5,-7.5,2.5,12.5]


def plastic_u_centers(arm):
    """u-centre of the two plastic bars in the structure (wall) frame [cm].
    Plastics are centred on the pinwheel-shifted MM; the wall is on the
    un-shifted structure -> plastics offset by -pinwheel along +u."""
    off = -PINWHEEL[arm]                       # MM centre in structure frame
    half = BSC_U / 2 + BSC_GAP / 2             # 10.15
    return np.array([off - half, off + half])  # [-u bar, +u bar]


# ---- imaging & angle ----------------------------------------------------------
def project_plastic_to_wall(arm):
    """Where the plastic bars/edges project onto the wall, seen from the source."""
    s = R_wall(arm) / R_plastic(arm)
    uc = plastic_u_centers(arm) * s
    half_u = (BSC_U / 2) * s
    half_v = (BSC_V / 2) * s
    return {'scale': s, 'u_centers': uc, 'u_half': half_u, 'v_half': half_v,
            'u_edges': np.array([uc - half_u, uc + half_u]).T}


def cos_theta(arm, u_wall, v_wall):
    """cos of incidence angle on the wall for a source track hitting (u,v) at
    radial R_wall.  cosθ = R / |r|."""
    R = R_wall(arm)
    return R / np.sqrt(R ** 2 + np.asarray(u_wall) ** 2 + np.asarray(v_wall) ** 2)


def path_length_mm(arm, u_wall, v_wall):
    """MIP path length through the 3 mm scint at (u,v) [mm] = t / cosθ."""
    return (SIPM_SCINT_THICK * 10.0) / cos_theta(arm, u_wall, v_wall)


# ---- MIP energy deposit -------------------------------------------------------
# G4_PLASTIC_SC_VINYLTOLUENE: rho=1.032 g/cm3, MIP dE/dx_min ~= 2.00 MeV/cm.
MIP_DEDX_MEV_PER_CM = 2.00          # PVT minimum ionizing


def mip_edep_MeV(arm, u_wall, v_wall):
    """Deposited energy of a normal-incidence-corrected MIP at (u,v) [MeV]."""
    return MIP_DEDX_MEV_PER_CM * (SIPM_SCINT_THICK / cos_theta(arm, u_wall, v_wall))


if __name__ == '__main__':
    print('group u-centers [cm]:', np.round(group_u_centers(), 2))
    for arm in 'BCAD':
        pj = project_plastic_to_wall(arm)
        print(f'\narm {arm}: R_wall={R_wall(arm):.2f}  R_plastic={R_plastic(arm):.2f}  '
              f'proj scale={pj["scale"]:.3f}')
        print(f'  plastic u-centers (wall frame) = {np.round(plastic_u_centers(arm),2)} cm'
              f'  -> project to {np.round(pj["u_centers"],2)} cm (+-{pj["u_half"]:.1f})')
        print(f'  plastic v half-height 15 cm -> projects to +-{pj["v_half"]:.1f} cm on wall')
        gu = group_u_centers()
        print(f'  1/cosθ at group centers (v=0):',
              np.round(1 / cos_theta(arm, gu, 0), 3))
        print(f'  1/cosθ at (u=+12.5, v=+-11.5):',
              round(1 / cos_theta(arm, 12.5, 11.5), 3))
