# Rebuild the slides. Figures come from ../figures -- regenerate them first with
# ../scripts/make_figures.py.
all: dream_ntof_matching_slides.pdf

%.pdf: %.tex
	pdflatex -interaction=nonstopmode $< >/dev/null
	pdflatex -interaction=nonstopmode $< >/dev/null

clean:
	rm -f *.aux *.log *.out *.nav *.snm *.toc
