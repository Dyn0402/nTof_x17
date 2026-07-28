.                                                DETECTOR SPECIFIC PARAMETERS (Lines may be commented with '#' sign!)
===================================================================================================================================================================================================================================
DETECTOR   DETECTOR   DETECTOR  STEP   TIMING    MIXED     EXPAND   SMOOTHING  TIME     G-FLASH    G-FLASH     G-FLASH     G-FLASH   BASELINE   BASELINE   AMPLITUDE   AMPLITUDE   AREA/AMP.   AREA/AMP.   SIGNAL WIDTH   SIGNAL WIDTH    NUMBER OF     PULSE SHAPE
  NAME      NUMBER     CLASS    SIZE   FILTER   POLARITY   PULSES    FILTER    LIMIT    OPTION    THRESHOLD    MIN_WIDTH   WINDOW     OPTION     FILTER     OPTION     THRESHOLD   LOW THR.    HIGH THR.   LOW THR.       HIGH THR.      PULSE SHAPES     ADDRESS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# X17 EAR2 2026 -- variant v7_step  =  v4_walshapes  +
#   LIQ*  STEP SIZE -> 2/3
#   PSS*  STEP SIZE -> 2/3
#   WAL*  STEP SIZE -> 5/5
#
# Pileup resolution, which is the other lever on how much signal is
# recovered -- and at early times it is the dominant one. The DREAM regression
# measures an n_TOF rate of 13.76 hits/us in the 1-3 ms bin, i.e. a mean spacing
# of 73 ns, against a wall pulse of 74 ns FWHM. The walls are therefore
# self-piled-up exactly where the matcher is weakest.
#
# The PSA guide's first practical advice: "Reducing the STEP SIZE -- even at the
# price of worsening the signal-to-noise ratio in the derivative -- can often
# help in resolving pileups."
#
# WAL 8/7 -> 5/5, PSS 3/4 -> 2/3, LIQ 2/4 -> 2/3.
#
PKUP        0         PSA     300/6     0        0          3        50      100000      0          100          1         0          -1         300         0           300        0.0       2000          1            4000               0

SILI        1           PSA    150       0        0         1         0      1e5    0       5000.        0.          0        1/70      1e4         1          2500        200         2000         400            4000     	 	  0
SILI        2           PSA    150       0        0         1         0      1e5    0       5000.        0.          0        1/70      1e4         1          2500        200         2000         400            4000     	 	  0
SILI        3           PSA    150       0        0         1         0      1e5    0       5000.        0.          0        1/70      1e4         1          2500        200         2000         400            4000     	 	  0
SILI        4           PSA    150       0        0         1         0      1e5    0       5000.        0.          0        1/70      1e4         1          2500        200         2000         400            4000     	 	  0

WALA        0           PSA    5/5       0        0        1        50/5     40000        0        250/11400      0.          0        4/150      800          2           50        10          200          5/100         4000     	 	  3		X17_WALA_Signal_avg0.txt X17_WALA_Signal_avg1.txt X17_WALA_Signal_avg2.txt
WALB        0           PSA    5/5       0        0        1        50/5     40000        0        250/11400      0.          0        4/150      800          2           50        10          200          5/100         4000     	 	  3		X17_WALB_Signal_avg0.txt X17_WALB_Signal_avg1.txt X17_WALB_Signal_avg2.txt
WALC        0           PSA    5/5       0        0        1        50/5     40000        0        250/11400      0.          0        4/150      800          2           50        10          200          5/100         4000     	 	  3		X17_WALC_Signal_avg0.txt X17_WALC_Signal_avg1.txt X17_WALC_Signal_avg2.txt
WALD        0           PSA    5/5       0        0        1        50/5     40000        0        250/11400      0.          0        4/150      800          2           50        10          200          5/100         4000     	 	  3		X17_WALD_Signal_avg0.txt X17_WALD_Signal_avg1.txt X17_WALD_Signal_avg2.txt
PSSA        0           PSA    2/3       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          1            50        1           60           10            3000                0
PSSB        0           PSA    2/3       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          1            50        1           60           10            3000                0
PSSC        0           PSA    2/3       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          1            50        1           60           10            3000                0
PSSD        0           PSA    2/3       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          1            50        1           60           10            3000                0

LIQA        0           PSA    2/3        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQB        0           PSA    2/3        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQC        0           PSA    2/3        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQD        0           PSA    2/3        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
