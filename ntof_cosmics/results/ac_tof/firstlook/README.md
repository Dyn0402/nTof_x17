# n_TOF arm-to-arm timing on beam-off cosmics: first look (2026-11-19, unreviewed; superseded by ../../../ac_tof_beamoff.py)

Throwaway scripts from a first look at σ_t in the n_TOF data (beam-off runs 224678–687, run_149), kept so the numbers can be
reproduced. They read the n_TOF scintillator trees with uproot (`tof` rounded to 1 ns).

- `ac_dt.py`: wall A–C bar coincidences (both ends of a bar in each arm). 2,579 coincidences, raw Δt s68 3.46 ns.
- `ac_dt3.py`: plus offsets per segment pair (2.93 ns) and the linear top−bottom correction (**2.70 ns**, core rms 2.5 ns):
  σ_t ≲ 1.9 ns per arm, an upper bound that still includes the muons' own time of flight.
- `ac_pl.py`: the plastics (amplitude > 300): 1,842 coincidences, s68 3.8 ns after per-PMT-pair offsets, flat-topped.

The real measurement is still to do: A–C through-goers from the Micromegas line (`nTof_x17/ntof_cosmics/inbeam_through_goers.py`),
the predicted L/c and position along the bars per event, time-walk correction, σ/√2 per arm. ~1,000 events: ±0.05–0.1 ns stat.,
floor ~0.2–0.3 ns from the 1 ns rounding: it can test σ_t ≥ 0.5 ns, not demonstrate 0.3 ns.
