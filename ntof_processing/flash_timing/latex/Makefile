# Rebuild both documents (figures come from ../figures, regenerate those first
# with ../scripts/make_figures.py and ../scripts/make_figures_v2.py)
all: flash_timing_calibration.pdf flash_timing_slides.pdf

%.pdf: %.tex
	pdflatex -interaction=nonstopmode $< >/dev/null
	pdflatex -interaction=nonstopmode $< >/dev/null

clean:
	rm -f *.aux *.log *.out *.nav *.snm *.toc
