.                                                DETECTOR SPECIFIC PARAMETERS (Lines may be commented with '#' sign!)
===================================================================================================================================================================================================================================
DETECTOR   DETECTOR   DETECTOR  STEP   TIMING    MIXED     EXPAND   SMOOTHING  TIME     G-FLASH    G-FLASH     G-FLASH     G-FLASH   BASELINE   BASELINE   AMPLITUDE   AMPLITUDE   AREA/AMP.   AREA/AMP.   SIGNAL WIDTH   SIGNAL WIDTH    NUMBER OF     PULSE SHAPE
  NAME      NUMBER     CLASS    SIZE   FILTER   POLARITY   PULSES    FILTER    LIMIT    OPTION    THRESHOLD    MIN_WIDTH   WINDOW     OPTION     FILTER     OPTION     THRESHOLD   LOW THR.    HIGH THR.   LOW THR.       HIGH THR.      PULSE SHAPES     ADDRESS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# X17 EAR2 2026 -- variant v8_pssfit  =  v4_walshapes  +
#   PSS*  AMPLITUDE OPTION -> 2
#   PSS*  NUMBER OF PULSE SHAPES -> 3
#   PSS*  pulse shapes -> 3 (X17_{tree}_Signal_avg0.txt, X17_{tree}_Signal_avg1.txt, X17_{tree}_Signal_avg2.txt)
#
# Turn on pulse-shape fitting for the plastics (AMPLITUDE OPTION
# 1 -> 2) with the measured 101 ns averaged templates, one per amplitude regime.
#
# The plastics currently use the parabolic-top option, i.e. no deconvolution at
# all. They are the leg the wall AND plastic trigger is limited by, they are the
# highest-rate tree in the file, and their pulse is 13 ns FWHM -- so pileup
# resolution should be exactly where their remaining inefficiency lives. This is
# the change the earlier rounds deferred as "riskier"; it gets its own variant so
# a regression can be attributed.
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
PSSA        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          2            50        1           60           10            3000                3		X17_PSSA_Signal_avg0.txt X17_PSSA_Signal_avg1.txt X17_PSSA_Signal_avg2.txt
PSSB        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          2            50        1           60           10            3000                3		X17_PSSB_Signal_avg0.txt X17_PSSB_Signal_avg1.txt X17_PSSB_Signal_avg2.txt
PSSC        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          2            50        1           60           10            3000                3		X17_PSSC_Signal_avg0.txt X17_PSSC_Signal_avg1.txt X17_PSSC_Signal_avg2.txt
PSSD        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          2            50        1           60           10            3000                3		X17_PSSD_Signal_avg0.txt X17_PSSD_Signal_avg1.txt X17_PSSD_Signal_avg2.txt

LIQA        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQB        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQC        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQD        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
