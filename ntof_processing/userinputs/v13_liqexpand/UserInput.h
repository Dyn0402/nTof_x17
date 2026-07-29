.                                                DETECTOR SPECIFIC PARAMETERS (Lines may be commented with '#' sign!)
===================================================================================================================================================================================================================================
DETECTOR   DETECTOR   DETECTOR  STEP   TIMING    MIXED     EXPAND   SMOOTHING  TIME     G-FLASH    G-FLASH     G-FLASH     G-FLASH   BASELINE   BASELINE   AMPLITUDE   AMPLITUDE   AREA/AMP.   AREA/AMP.   SIGNAL WIDTH   SIGNAL WIDTH    NUMBER OF     PULSE SHAPE
  NAME      NUMBER     CLASS    SIZE   FILTER   POLARITY   PULSES    FILTER    LIMIT    OPTION    THRESHOLD    MIN_WIDTH   WINDOW     OPTION     FILTER     OPTION     THRESHOLD   LOW THR.    HIGH THR.   LOW THR.       HIGH THR.      PULSE SHAPES     ADDRESS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# X17 EAR2 2026 -- variant v13_liqexpand  =  v12_liqpileup  +
#   LIQ*  EXPAND PULSES -> 1
#   LIQ*  SIGNAL WIDTH LOW -> 1/150
#
# v12 enabled the fast/slow split and it came back USELESS:
# `afast` is filled for 100 % of hits but `aslow` is ~0, so
# slow/(fast+slow) = 0.000 on all four liquids.
#
# The reason is in the guide's definition: aslow is integrated "starting from the
# boundary up to the END OF THE PULSE". The liquid row has EXPAND PULSES = 0, so
# the pulse boundary is wherever the derivative-based recognition closes it --
# about 20-40 ns for a 6 ns FWHM pulse. The slow component runs to ~150 ns in the
# raw waveforms, so it lies entirely OUTSIDE the reconstructed pulse and nothing
# is left for aslow to integrate.
#
# That is also why the liquids carry no pulse-shape-discrimination information no
# matter where the boundary is put, and it means the reported liquid `area` has
# been missing its slow component all along.
#
#   EXPAND PULSES 0 -> 1        push the pulse end forward until the signal
#                               returns to baseline, which is what puts the slow
#                               component inside the pulse. Expansion is blocked
#                               by the next pulse in line, so it cannot trample
#                               neighbours -- which matters here, because 76-92 %
#                               of liquid pulses are piled up.
#   SIGNAL WIDTH LOW 1 -> 1/150 add a SUGGESTED WIDTH of 150 ns, matched to the
#                               measured extent of the slow component, so pulses
#                               cut short by a neighbour are still widened before
#                               the area and baseline steps.
#
# If aslow is still empty after this, the slow component cannot be captured
# pulse-by-pulse in this framework and PSD needs the raw waveforms.
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

LIQA        0           PSA    1/3        0        0         1        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1/150             5000/30                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQB        0           PSA    1/3        0        0         1        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1/150             5000/30                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQC        0           PSA    1/3        0        0         1        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1/150             5000/30                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQD        0           PSA    1/3        0        0         1        0        25000        0       500/1e4       100.      1000         1        100          2           50        1           60           1/150             5000/30                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
