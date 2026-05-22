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
    mm_drift_gap=3.0,               # cm, active drift gap
    gap_mm_to_scint=3.0,            # cm, gap between MM back and scint wall front
    scint_wall_size=(48.0, 48.0),   # cm, (horiz/u, into-page/v)
    scint_wall_thickness=0.3,       # cm
    gap_scint_to_liq=3.0,           # cm, gap between scint wall and liq scint 1 front
    liq_scint_size=(38.0, 38.0),    # cm, (horiz/u, into-page/v)
    liq_scint_thickness=1.5,        # cm

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

    signal_spectrum=GaussianSpectrum(mean_deg=120.0, sigma_deg=5.0),
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

# Populate self.hits for the MM hit-map panels in plot_summary.
# A few thousand events is enough to show the hit distributions.
sim.run_display_sample(n_sample=5_000)

# Figure 1: Summary (default = scint double trigger)
fig1 = sim.plot_summary(trigger='double')

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
