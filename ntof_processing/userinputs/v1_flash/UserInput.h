.                                                DETECTOR SPECIFIC PARAMETERS (Lines may be commented with '#' sign!)
===================================================================================================================================================================================================================================
DETECTOR   DETECTOR   DETECTOR  STEP   TIMING    MIXED     EXPAND   SMOOTHING  TIME     G-FLASH    G-FLASH     G-FLASH     G-FLASH   BASELINE   BASELINE   AMPLITUDE   AMPLITUDE   AREA/AMP.   AREA/AMP.   SIGNAL WIDTH   SIGNAL WIDTH    NUMBER OF     PULSE SHAPE
  NAME      NUMBER     CLASS    SIZE   FILTER   POLARITY   PULSES    FILTER    LIMIT    OPTION    THRESHOLD    MIN_WIDTH   WINDOW     OPTION     FILTER     OPTION     THRESHOLD   LOW THR.    HIGH THR.   LOW THR.       HIGH THR.      PULSE SHAPES     ADDRESS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# X17 EAR2 2026 -- variant v1_flash.
# Derived from UserInput_2026_EAR2_X17.h (Riccardo, 2026-07-17).
# ONLY the G-FLASH THRESHOLD column is changed, on WAL / PSS / LIQ.
# Rationale and the raw-waveform evidence: ntof_processing/FLASH_TIME_BASE.md
#
#   PSS  50.        -> 2000/1e4   the plastic flash saturates the ADC (>30000
#                                 channels); a 50-channel threshold with no
#                                 lower time limit latches onto pre-flash noise
#                                 in 37-85 % of bunches.
#   WAL  500.       -> 250/11400  the SiPM signal is diverted 11.24-12.25 us,
#                                 so the wall records (a) the gate-close
#                                 transient at 11.24 us and (b) the attenuated
#                                 real flash leaking through at 11.60 us.  The
#                                 time limit selects (b), which is the physical
#                                 flash and is what PSS/LIQ/PKUP also time.
#                                 The threshold must stay <=400: at 600 the
#                                 weakest channels miss the leak and fall
#                                 through to the gate-release transient.
#   LIQ  500.       -> 500/1e4    already 0 % failures; the time limit is pure
#                                 insurance against sample-activity pulses.
#   PKUP unchanged  -- 0 % failures, the absolute-time anchor.
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

PSSA        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.          0         1        200          1           100        2           20           10            3000                0
PSSB        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.          0         1        200          1           100        2           20           10            3000                0
PSSC        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.          0         1        200          1           100        2           20           10            3000                0
PSSD        0           PSA    3/4       0        0        -1        0        25000        0        2000/1e4       0.          0         1        200          1           100        2           20           10            3000                0

LIQA        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        2           10           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQB        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        2           10           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQC        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        2           10           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
LIQD        0           PSA    2/4        0        0         0        0        25000        0       500/1e4       100.      1000         1        100          2           50        2           10           1             5000                2		X17_LIQA_Signal_7.txt  X17_LIQB_Signal_0.txt
