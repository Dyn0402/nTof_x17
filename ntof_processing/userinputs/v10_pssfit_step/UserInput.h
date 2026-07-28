.                                                DETECTOR SPECIFIC PARAMETERS (Lines may be commented with '#' sign!)
===================================================================================================================================================================================================================================
DETECTOR   DETECTOR   DETECTOR  STEP   TIMING    MIXED     EXPAND   SMOOTHING  TIME     G-FLASH    G-FLASH     G-FLASH     G-FLASH   BASELINE   BASELINE   AMPLITUDE   AMPLITUDE   AREA/AMP.   AREA/AMP.   SIGNAL WIDTH   SIGNAL WIDTH    NUMBER OF     PULSE SHAPE
  NAME      NUMBER     CLASS    SIZE   FILTER   POLARITY   PULSES    FILTER    LIMIT    OPTION    THRESHOLD    MIN_WIDTH   WINDOW     OPTION     FILTER     OPTION     THRESHOLD   LOW THR.    HIGH THR.   LOW THR.       HIGH THR.      PULSE SHAPES     ADDRESS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# X17 EAR2 2026 -- variant v10_pssfit_step  =  v8_pssfit  +
#   PSS*  STEP SIZE -> 2/3
#
# Push on the one thing that has actually worked. v8_pssfit won by
# +1.2 points overall and +2.1 in the 1-3 ms bin, and the mechanism is now
# measured: it produces FEWER plastic hits at every amplitude cut (0.72-0.99 of
# v4) yet MORE valid wall AND plastic candidates (103,816 vs 101,809). So the
# gain is plastic TIMING in pileup, not plastic yield -- shape fitting merges
# fragments back into one correctly-timed pulse.
#
# The leg diagnostic says that is exactly where the remaining loss is: wall-only
# efficiency is 98.9 % and flat in time, the AND is 96.4 %, so the plastic leg
# costs 2.5 % overall but 3.4-3.7 % at 1-10 ms -- a pileup signature.
#
# v7_step tested a finer STEP SIZE WITHOUT shape fitting and lost. With shape
# fitting the derivative search only has to find candidates for the fit to
# resolve, so a finer step should compound rather than fragment. PSS only; the
# walls were neutral-to-worse in v7 (T1 sigma +2.2 %).
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
PSSA        0           PSA    2/3       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          2            50        1           60           10            3000                3		X17_PSSA_Signal_avg0.txt X17_PSSA_Signal_avg1.txt X17_PSSA_Signal_avg2.txt
PSSB        0           PSA    2/3       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          2            50        1           60           10            3000                3		X17_PSSB_Signal_avg0.txt X17_PSSB_Signal_avg1.txt X17_PSSB_Signal_avg2.txt
PSSC        0           PSA    2/3       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          2            50        1           60           10            3000                3		X17_PSSC_Signal_avg0.txt X17_PSSC_Signal_avg1.txt X17_PSSC_Signal_avg2.txt
PSSD        0           PSA    2/3       0        0        -1        0        25000        0        2000/1e4       0.       1000         1        200          2            50        1           60           10            3000                3		X17_PSSD_Signal_avg0.txt X17_PSSD_Signal_avg1.txt X17_PSSD_Signal_avg2.txt

LIQA        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQB        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQC        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQD        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
