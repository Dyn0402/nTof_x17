"""
Example: running and comparing configurations of the Micromegas simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from MX17_Simulator import (
    MicromegasSimulation, SimConfig,
    GaussianSpectrum, ExponentialSpectrum, IsotropicSpectrum,
    FlatSpectrum, HistogramSpectrum, calculate_solid_angle_coverage
)

cfg = SimConfig(
    detector_distance=22.0,         # cm
    # detector_size=40.0,             # cm
    detector_size=38.0,             # cm
    spatial_resolution=0.05,        # cm
    time_resolution=30.0,           # ns
    time_spread=4000.0,             # ns — duration of one readout event
    coincidence_window=60.0,        # ns
    n_events=1000000,
    n_random=0e-1,                   # few uncorrelated random hits per event
    n_background_pairs=1e-2,
    n_signal=2e-4,
    signal_spectrum=GaussianSpectrum(mean_deg=120.0, sigma_deg=5.0),
    background_spectrum=ExponentialSpectrum(scale_deg=40.0),
    merge_hits=True,
    merge_spatial_threshold=0.1,    # cm
    merge_time_threshold=50.0,      # ns
    allow_same_detector_pairs=True,
    seed=42,
)

sim = MicromegasSimulation(cfg)

frac = calculate_solid_angle_coverage(sim.detectors, n_samples=1_000)
print(f"Coverage: {frac*100:.2f}%  ({frac*4*np.pi:.4f} sr)")

sim.run(n_workers=4)  # set n_workers=1 for serial (also stores self.hits for plotting)
sim.summary_stats()

# Figure 1: Summary with any-hit trigger
fig1 = sim.plot_summary(trigger='any')

# Figure 2: Side-by-side trigger comparison
fig2 = sim.plot_trigger_comparison()

# Figure 3: True generated spectra vs reconstructed (counts)
fig3 = sim.plot_true_vs_reconstructed()

# Figure 4: Same but shape-normalized to compare distributions directly
fig4 = sim.plot_true_vs_reconstructed(normalized=True)

plt.show()
