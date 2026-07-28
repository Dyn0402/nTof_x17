.                                                DETECTOR SPECIFIC PARAMETERS (Lines may be commented with '#' sign!)
===================================================================================================================================================================================================================================
DETECTOR   DETECTOR   DETECTOR  STEP   TIMING    MIXED     EXPAND   SMOOTHING  TIME     G-FLASH    G-FLASH     G-FLASH     G-FLASH   BASELINE   BASELINE   AMPLITUDE   AMPLITUDE   AREA/AMP.   AREA/AMP.   SIGNAL WIDTH   SIGNAL WIDTH    NUMBER OF     PULSE SHAPE
  NAME      NUMBER     CLASS    SIZE   FILTER   POLARITY   PULSES    FILTER    LIMIT    OPTION    THRESHOLD    MIN_WIDTH   WINDOW     OPTION     FILTER     OPTION     THRESHOLD   LOW THR.    HIGH THR.   LOW THR.       HIGH THR.      PULSE SHAPES     ADDRESS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# X17 EAR2 2026 -- variant v2_elim  =  v1_flash  +  loosened elimination.
#
# Measured on isolated late-time (1-6 ms) pulses of run 224572, ~11 k walls /
# 15 k plastics / 2.6 k liquids (ntof_processing/FINDINGS_2026-07-28_psa_optimization.md):
#
#   family  area/amp p1..p99   current AREA/AMP cut   fraction cut away
#   WAL       42 .. 110            10 .. 200                 0 %
#   PSS      1.3 ..  29             2 ..  20               ~25 %  <-- !
#   LIQ      1.7 ..  14             2 ..  10               ~19 %  <-- !
#
# The plastic and liquid AREA/AMP HIGH thresholds sit inside the bulk of the
# real pulse distribution.  The walls' own cut is 2x above their maximum, which
# is what a safe cut looks like.  The PSA guide is explicit that elimination
# should be loose and that final rejection belongs in the offline analysis.
#
#   PSS  AREA/AMP  2 .. 20  -> 1 .. 60      LIQ  AREA/AMP  2 .. 10 -> 1 .. 60
#   PSS  AMPLITUDE THR 100  -> 50           (walls use 50; do not truncate the
#                                            small-hit spectrum in its bulk)
#   PSS  G-FLASH WINDOW  0  -> 1000         (as LIQ; protect the flash region
#                                            from elimination so the baseline
#                                            machinery sees it as a pulse)
#
PKUP        0         PSA     300/6     0        0          3        50      100000      0          100          1         0          -1         300         0           300        0.0       2000          1            4000               0

SILI        1           PSA    150       0        0         1         0      1e5    0       5000.        0.          0        1/70      1e4         1          2500        200         2000         400            4000     	 	  0
SILI        2           PSA    150       0        0         1         0      1e5    0       5000.        0.          0        1/70      1e4         1          2500        200         2000         400            4000     	 	  0
SILI        3           PSA    150       0        0         1         0      1e5    0       5000.        0.          0        1/70      1e4         1          2500        200         2000         400            4000     	 	  0
SILI        4           PSA    150       0        0         1         0      1e5    0       5000.        0.          0        1/70      1e4         1          2500        200         2000         400            4000     	 	  0

WALA        0           PSA    8/7       0        0        1        50/5     40000        0        250/11400      0.          0        4/150      800          2           50        10          200          5/100         4000     	 	  3		X17_WALA_Signal_3.txt X17_WALC_Signal_0.txt X17_WALB_Signal_0.txt
WALB        0           PSA    8/7       0        0        1        50/5     40000        0        250/11400      0.          0        4/150      800          2           50        10          200          5/100         4000     	 	  3		X17_WALA_Signal_3.txt X17_WALC_Signal_0.txt X17_WALB_Signal_0.txt
WALC        0           PSA    8/7       0        0        1        50/5     40000        0        250/11400      0.          0        4/150      800          2           50        10          200          5/100         4000     	 	  3		X17_WALA_Signal_3.txt X17_WALC_Signal_0.txt X17_WALB_Signal_0.txt
WALD        0           PSA    8/7       0        0        1        50/5     40000        0        250/11400      0.          0        4/150      800          2           50        10          200          5/100         4000     	 	  3		X17_WALA_Signal_3.txt X17_WALC_Signal_0.txt X17_WALB_Signal_0.txt

PSSA        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          1            50        1           60           10            3000                0
PSSB        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          1            50        1           60           10            3000                0
PSSC        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          1            50        1           60           10            3000                0
PSSD        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          1            50        1           60           10            3000                0

LIQA        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQB        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQC        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQD        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
