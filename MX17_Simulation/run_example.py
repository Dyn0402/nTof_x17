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

# ---------------------------------------------------------------------------
# Example 1: Basic run with default parameters
# ---------------------------------------------------------------------------

cfg = SimConfig(
    detector_distance=21.0,         # cm
    detector_size=40.0,             # cm
    spatial_resolution=0.05,        # cm
    time_resolution=30.0,           # ns
    time_spread=4000.0,             # ns — duration of one readout event
    coincidence_window=60.0,        # ns
    n_events=2000,                   # number of readout events
    n_random=100.0,                  # mean random particles per event (Poisson)
    n_background_pairs=10.0,         # mean background pairs per event (Poisson)
    n_signal=1.0,                   # mean signal pairs per event (Poisson)
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

sim.run()
sim.summary_stats()

fig = sim.plot_summary()
# plt.savefig('summary_default.png', dpi=150, bbox_inches='tight')

# ---------------------------------------------------------------------------
# Example 2: Compare merging on vs off
# ---------------------------------------------------------------------------

# fig2, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
#
# for ax, merge in zip(axes, [False, True]):
#     cfg2 = SimConfig(
#         n_events=100, n_random=0.0, n_background_pairs=0.0, n_signal=1.0,
#         merge_hits=merge, merge_spatial_threshold=1.0, merge_time_threshold=40.0,
#         allow_same_detector_pairs=True, spatial_resolution=0.05, time_resolution=20.0,
#         time_spread=4000.0, coincidence_window=50.0,
#         seed=7,
#     )
#     sim2 = MicromegasSimulation(cfg2)
#     sim2.run()
#     sim2.plot_angular_separation(ax=ax)
#     label = 'with merging' if merge else 'no merging'
#     ax.set_title(f'Angular separation — {label}\n'
#                  f'({len(sim2.hits)} hits after, {len(sim2.coincident_pairs)} pairs)')
#
# fig2.tight_layout()
# plt.savefig('merge_comparison.png', dpi=150, bbox_inches='tight')

plt.show()
