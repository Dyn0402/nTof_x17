.                                                DETECTOR SPECIFIC PARAMETERS (Lines may be commented with '#' sign!)
===================================================================================================================================================================================================================================
DETECTOR   DETECTOR   DETECTOR  STEP   TIMING    MIXED     EXPAND   SMOOTHING  TIME     G-FLASH    G-FLASH     G-FLASH     G-FLASH   BASELINE   BASELINE   AMPLITUDE   AMPLITUDE   AREA/AMP.   AREA/AMP.   SIGNAL WIDTH   SIGNAL WIDTH    NUMBER OF     PULSE SHAPE
  NAME      NUMBER     CLASS    SIZE   FILTER   POLARITY   PULSES    FILTER    LIMIT    OPTION    THRESHOLD    MIN_WIDTH   WINDOW     OPTION     FILTER     OPTION     THRESHOLD   LOW THR.    HIGH THR.   LOW THR.       HIGH THR.      PULSE SHAPES     ADDRESS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# X17 EAR2 2026 -- variant v6_lowthr  =  v4_walshapes  +
#   LIQ*  AMPLITUDE THRESHOLD -> 25
#   PSS*  AMPLITUDE THRESHOLD -> 25
#   PSS*  AREA/AMP LOW -> 0.2
#
# The amplitude threshold is BINDING on the plastics and the liquids
# and NOT on the walls -- measured on the v4 output with
# ntof_processing/threshold_headroom.py:
#
#     tree   amp p1   <2x cut   <3x cut     (cut = 50)
#     WAL     68-82    1.7-3.6%  4.9-8.7%   spectrum dies BEFORE the cut
#     PSS     52-53   11.2-22.9% 28.1-42.7% spectrum piles UP against the cut
#     LIQ     53-54    8.4-28.2% 15.5-55.3% same
#
# So there is signal below 50 channels on PSS/LIQ and none on WAL (the DAQ
# zero-suppression is the wall's floor, not the PSA). Halve the plastic and
# liquid thresholds and open the AREA/AMP low edge, which also sits right at the
# PSS p1 (1.30-1.55 against a cut of 1.0). Walls deliberately untouched.
#
# Noise is the thing to watch: the PSS baseline RMS is ~20 channels, so 25 is
# ~1.2 sigma and the width/area conditions are doing the rejecting. If the
# singles-matcher EFFICIENCY rises this was signal; if only the false rate rises
# it was noise.
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
PSSA        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          1            25        0.2           60           10            3000                0
PSSB        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          1            25        0.2           60           10            3000                0
PSSC        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          1            25        0.2           60           10            3000                0
PSSD        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          1            25        0.2           60           10            3000                0

LIQA        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           25        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQB        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           25        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQC        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           25        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQD        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           25        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
