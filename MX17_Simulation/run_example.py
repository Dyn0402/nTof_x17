"""
Example: running and comparing configurations of the Micromegas simulation
with scintillator trigger stack.
"""

import numpy as np
import matplotlib.pyplot as plt
from MX17_Simulator import (
    MicromegasSimulation, SimConfig,
    GaussianSpectrum, ExponentialSpectrum, IsotropicSpectrum,
    FlatSpectrum, HistogramSpectrum, calculate_solid_angle_coverage
)

cfg = SimConfig(
    # MM geometry
    detector_distance=22.0,         # cm
    detector_size=(38.0, 34.0),     # cm, (horiz/u, into-page/v)

    # Scintillator trigger stack (all four sides, from origin outward)
    mm_drift_gap=3.0,               # cm, active drift gap (30 mm)
    gap_mm_to_scint=2.77,           # cm, drift back → scint centre (PCB + air + tape)
    scint_wall_size=(48.0, 48.0),   # cm, (horiz/u, along-beam/v)
    scint_wall_thickness=0.3,       # cm, 3 mm PVT
    gap_scint_to_liq=3.44,          # cm, scint centre → LS-1 centre (air + CFRP/Al)
    liq_scint_size=(43.0, 43.0),    # cm, (horiz/u, along-beam/v)
    liq_scint_thickness=2.0,        # cm, 2 cm LAB layer

    # Resolution / timing
    spatial_resolution=0.05,        # cm
    time_resolution=30.0,           # ns
    time_spread=2000.0,             # ns
    coincidence_window=60.0,        # ns

    # Event statistics
    n_events=1_000_000,
    n_random=1e-2,
    n_background_pairs=1e-2,
    n_signal=2e-4,

    signal_spectrum=GaussianSpectrum(mean_deg=120.0, sigma_deg=15.0),
    background_spectrum=ExponentialSpectrum(scale_deg=40.0),

    merge_hits=True,
    merge_spatial_threshold=0.1,
    merge_time_threshold=50.0,
    allow_same_detector_pairs=True,
    seed=42,
)

sim = MicromegasSimulation(cfg)

frac = calculate_solid_angle_coverage(
    [d for d in sim.detectors if d.det_type == 'mm'], n_samples=1_000)
print(f"MM coverage: {frac*100:.2f}%  ({frac*4*np.pi:.4f} sr)")

# Main run — use n_workers>1 for speed; hits are not stored in parallel mode.
sim.run(n_workers=4)
sim.summary_stats()

# Figure 1: Summary — geometry + scint single (large), double/mm_any/mm_double (bottom)
fig1 = sim.plot_summary()

# Figure 2: All four trigger scenarios side-by-side
fig2 = sim.plot_trigger_comparison()

# Figure 3: MM double vs scint double — shows liq scint acceptance loss
fig3 = sim.plot_mm_vs_scint_comparison()

# Figure 4: Per-track scintillator acceptance bar chart
fig4 = sim.plot_scint_efficiency()

# Figure 5: True generated vs reconstructed (scint double + MM double overlaid)
fig5 = sim.plot_true_vs_reconstructed(trigger=['double', 'mm_double'])

# Figure 6: Same, shape-normalized
fig6 = sim.plot_true_vs_reconstructed(trigger=['double', 'mm_double'], normalized=True)

plt.show()
