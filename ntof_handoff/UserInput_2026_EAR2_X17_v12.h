.                                                DETECTOR SPECIFIC PARAMETERS (Lines may be commented with '#' sign!)
===================================================================================================================================================================================================================================
DETECTOR   DETECTOR   DETECTOR  STEP   TIMING    MIXED     EXPAND   SMOOTHING  TIME     G-FLASH    G-FLASH     G-FLASH     G-FLASH   BASELINE   BASELINE   AMPLITUDE   AMPLITUDE   AREA/AMP.   AREA/AMP.   SIGNAL WIDTH   SIGNAL WIDTH    NUMBER OF     PULSE SHAPE
  NAME      NUMBER     CLASS    SIZE   FILTER   POLARITY   PULSES    FILTER    LIMIT    OPTION    THRESHOLD    MIN_WIDTH   WINDOW     OPTION     FILTER     OPTION     THRESHOLD   LOW THR.    HIGH THR.   LOW THR.       HIGH THR.      PULSE SHAPES     ADDRESS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# X17 EAR2 2026 -- variant v12_liqpileup  =  v11_pssfit_width  +
#   LIQ*  STEP SIZE -> 1/3
#   LIQ*  SIGNAL WIDTH HIGH -> 5000/30
#
# The liquids, attacked at the right target this time.
#
# Three template replacements all failed, and the overnight raw-waveform study
# says why we were aiming at the wrong thing:
#
#  1. Only 8-24 % of liquid pulses are ISOLATED (LIQA 1014 of 6965 blocks, LIQB
#     812 of 10033, LIQD 1250 of 5175). The liquids are a PILEUP problem, and
#     every template we built was measured on the isolated minority.
#  2. On those isolated pulses a measured template really is 3-4x better than the
#     shipped pair (reduced chi2 224 -> 71 on LIQA, 81 -> 23 on LIQD, scored on
#     held-out pulses). It still made the PSA output worse -- because a longer,
#     more faithful template overlaps more of its neighbours in a population that
#     is mostly pileup.
#  3. Single-pulse fit quality is anyway floored by PHOTON STATISTICS: the fit
#     residual scales as sqrt(amplitude), flat to 10 % over a factor 25 in
#     amplitude (LIQD resid/sqrt(A) = 0.61/0.62/0.64/0.65/0.67). The slow
#     component is a countable number of photoelectrons, so it fluctuates
#     irreducibly. No template can fit shot noise, which is why binning the basis
#     by tail fraction bought only ~10 %.
#
# So: keep the shipped templates, and spend the change on pileup separation
# instead.
#
#    STEP SIZE 2/4 -> 1/3   the finest derivative window available, for a 6 ns
#                           FWHM pulse at 1 GS/s. The guide's first practical
#                           advice is that reducing STEP SIZE resolves pileup,
#                           and v7_step (which moved LIQ to 2/3) was the only
#                           change so far that raised liquid yield, +3..+6 %.
#    SIGNAL WIDTH HIGH 5000 -> 5000/30
#                           enables the fast/slow area split. `afast` and `aslow`
#                           are currently 0.0 % filled -- the PSA's pulse-shape
#                           discrimination observable has never been switched on
#                           for these detectors, and PSD is the entire reason one
#                           runs a liquid scintillator. 30 ns is placed just past
#                           the prompt peak (FWHM 6 ns) and inside the slow
#                           component, which the raw pulses show running to
#                           ~150 ns.
#
PKUP        0         PSA     300/6     0        0          3        50      100000      0          100          1         0          -1         300         0           300        0.0       2000          1            4000               0

SILI        1           PSA    150       0        0         1         0      1e5    0       5000.        0.          0        1/70      1e4         1          2500        200         2000         400            4000     	 	  0
SILI        2           PSA    150       0        0         1         0      1e5    0       5000.        0.          0        1/70      1e4         1          2500        200         2000         400            4000     	 	  0
SILI        3           PSA    150       0        0         1         0      1e5    0       5000.        0.          0        1/70      1e4         1          2500        200         2000         400            4000     	 	  0
SILI        4           PSA    150       0        0         1         0      1e5    0       5000.        0.          0        1/70      1e4         1          2500        200         2000         400            4000     	 	  0

WALA        0           PSA    8/7       0        0        1        50/5     40000        0        250/11400      0.          0        4/150      800          2           50        10          200          5/100         4000     	 	  3		X17_WALA_Signal_avg0.txt X17_WALA_Signal_avg1.txt X17_WALA_Signal_avg2.txt
WALB        0           PSA    8/7       0        0        1        50/5     40000        0        250/11400      0.          0        4/150      800          2           50        10          200          5/100         4000     	 	  3		X17_WALB_Signal_avg0.txt X17_WALB_Signal_avg1.txt X17_WALB_Signal_avg2.txt
WALC        0           PSA    8/7       0        0        1        50/5     40000        0        250/11400      0.          0        4/150      800          2           50        10          200          5/100         4000     	 	  3		X17_WALC_Signal_avg0.txt X17_WALC_Signal_avg1.txt X17_WALC_Signal_avg2.txt
WALD        0           PSA    8/7       0        0        1        50/5     40000        0        250/11400      0.          0        4/150      800          2           50        10          200          5/100         4000     	 	  3		X17_WALD_Signal_avg0.txt X17_WALD_Signal_avg1.txt X17_WALD_Signal_avg2.txt
PSSA        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          2            50        1           60           4            3000                3		X17_PSSA_Signal_avg0.txt X17_PSSA_Signal_avg1.txt X17_PSSA_Signal_avg2.txt
PSSB        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          2            50        1           60           4            3000                3		X17_PSSB_Signal_avg0.txt X17_PSSB_Signal_avg1.txt X17_PSSB_Signal_avg2.txt
PSSC        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          2            50        1           60           4            3000                3		X17_PSSC_Signal_avg0.txt X17_PSSC_Signal_avg1.txt X17_PSSC_Signal_avg2.txt
PSSD        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          2            50        1           60           4            3000                3		X17_PSSD_Signal_avg0.txt X17_PSSD_Signal_avg1.txt X17_PSSD_Signal_avg2.txt

LIQA        0           PSA    1/3        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000/30                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQB        0           PSA    1/3        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000/30                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQC        0           PSA    1/3        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000/30                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQD        0           PSA    1/3        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000/30                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
