.                                                DETECTOR SPECIFIC PARAMETERS (Lines may be commented with '#' sign!)
===================================================================================================================================================================================================================================
DETECTOR   DETECTOR   DETECTOR  STEP   TIMING    MIXED     EXPAND   SMOOTHING  TIME     G-FLASH    G-FLASH     G-FLASH     G-FLASH   BASELINE   BASELINE   AMPLITUDE   AMPLITUDE   AREA/AMP.   AREA/AMP.   SIGNAL WIDTH   SIGNAL WIDTH    NUMBER OF     PULSE SHAPE
  NAME      NUMBER     CLASS    SIZE   FILTER   POLARITY   PULSES    FILTER    LIMIT    OPTION    THRESHOLD    MIN_WIDTH   WINDOW     OPTION     FILTER     OPTION     THRESHOLD   LOW THR.    HIGH THR.   LOW THR.       HIGH THR.      PULSE SHAPES     ADDRESS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# X17 EAR2 2026 -- variant v9_liqaug  =  v4_walshapes  +
#   LIQ*  NUMBER OF PULSE SHAPES -> 3
#   LIQ*  pulse shapes -> 3 (X17_LIQA_Signal_7.txt, X17_LIQB_Signal_0.txt, X17_{tree}_Signal_avg2.txt)
#
# The liquid retry, done the other way round. Replacing the shipped
# liquid templates lost twice (551 ns in v3_shapes, 81 ns in v5_liqshort), and
# the length hypothesis is dead. The measured difference that survives is basis
# diversity: the shipped pair is a normal pulse (LIQA_Signal_7, FWHM 7 ns) AND a
# near-delta spike (LIQB_Signal_0, FWHM 1 ns), while every set I built spanned
# only 5-7 ns.
#
# So AUGMENT instead of replace: keep both shipped shapes and add one measured
# per-detector average as a third. If this wins, diversity was the story; if it
# is neutral, the liquids are limited by something other than the template and we
# stop spending variants on it.
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
PSSA        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          1            50        1           60           10            3000                0
PSSB        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          1            50        1           60           10            3000                0
PSSC        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          1            50        1           60           10            3000                0
PSSD        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          1            50        1           60           10            3000                0

LIQA        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                3		X17_LIQA_Signal_7.txt X17_LIQB_Signal_0.txt X17_LIQA_Signal_avg2.txt
LIQB        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                3		X17_LIQA_Signal_7.txt X17_LIQB_Signal_0.txt X17_LIQB_Signal_avg2.txt
LIQC        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                3		X17_LIQA_Signal_7.txt X17_LIQB_Signal_0.txt X17_LIQC_Signal_avg2.txt
LIQD        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                3		X17_LIQA_Signal_7.txt X17_LIQB_Signal_0.txt X17_LIQD_Signal_avg2.txt
