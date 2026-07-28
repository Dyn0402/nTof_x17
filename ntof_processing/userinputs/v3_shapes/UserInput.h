.                                                DETECTOR SPECIFIC PARAMETERS (Lines may be commented with '#' sign!)
===================================================================================================================================================================================================================================
DETECTOR   DETECTOR   DETECTOR  STEP   TIMING    MIXED     EXPAND   SMOOTHING  TIME     G-FLASH    G-FLASH     G-FLASH     G-FLASH   BASELINE   BASELINE   AMPLITUDE   AMPLITUDE   AREA/AMP.   AREA/AMP.   SIGNAL WIDTH   SIGNAL WIDTH    NUMBER OF     PULSE SHAPE
  NAME      NUMBER     CLASS    SIZE   FILTER   POLARITY   PULSES    FILTER    LIMIT    OPTION    THRESHOLD    MIN_WIDTH   WINDOW     OPTION     FILTER     OPTION     THRESHOLD   LOW THR.    HIGH THR.   LOW THR.       HIGH THR.      PULSE SHAPES     ADDRESS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# X17 EAR2 2026 -- variant v3_shapes  =  v2_elim  +  measured pulse-shape templates.
#
# Only the PULSE SHAPE ADDRESS column changes with respect to v2_elim.  The
# templates shipped with the current UserInput are each a SINGLE raw pulse and
# they are too short for the detector they describe:
#
#   file                     length   the pulse is still at ... after the peak
#   X17_WAL*_Signal_*.txt     314 ns   4.1 % at 200 ns, 0.5 % at 500 ns
#   X17_LIQA_Signal_7.txt      59 ns
#   X17_LIQB_Signal_0.txt      24 ns   (!)
#
# A template that ends inside the tail biases every fitted amplitude and area
# and cripples the pileup deconvolution -- which is where the liquids were said
# to be weakest.  The replacements are median averages of clean, isolated,
# late-time (1-15 ms) pulses from run 224572, aligned on the 50 % leading edge
# with sub-sample interpolation, one per tree per amplitude regime
# (~200-13000 pulses each), 551 ns long for PSS/LIQ and 720-861 ns for WAL.
# Built by ntof_processing/make_pulse_shapes.py; regenerate for other periods.
#
# The wall tail IS mildly amplitude-dependent (4.8-5.8 % at 200 ns for the
# lowest amplitude bin vs 4.0-4.1 % for the highest), so the three-shape
# machinery is kept and now carries three genuinely different shapes.
# LIQC has too few low-amplitude pulses for a bin-0 template, so it gets two.
#
# NOT changed here: PSS still uses AMPLITUDE OPTION=1 (parabolic top).  Turning
# on shape fitting for the plastics is a separate, riskier experiment -- keep it
# for a v4 so that a regression can be attributed.
#
# NOTE: the shape addresses below are BARE FILENAMES.  RunProcessing.sh needs
# FULL PATHS; ntof_processing/deploy_userinput.sh rewrites them for you.
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

LIQA        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                3		X17_LIQA_Signal_avg0.txt X17_LIQA_Signal_avg1.txt X17_LIQA_Signal_avg2.txt
LIQB        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                3		X17_LIQB_Signal_avg0.txt X17_LIQB_Signal_avg1.txt X17_LIQB_Signal_avg2.txt
LIQC        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                2		X17_LIQC_Signal_avg1.txt X17_LIQC_Signal_avg2.txt
LIQD        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1             5000                3		X17_LIQD_Signal_avg0.txt X17_LIQD_Signal_avg1.txt X17_LIQD_Signal_avg2.txt
