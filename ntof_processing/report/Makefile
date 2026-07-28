# figures are regenerated from the data; the PDF from the figures
all: comparison.pdf

figures/fig1_flash_id.pdf:
	python make_figures.py

comparison.pdf: comparison.tex figures/fig1_flash_id.pdf
	pdflatex -interaction=nonstopmode comparison.tex
	pdflatex -interaction=nonstopmode comparison.tex

clean:
	rm -f *.aux *.log *.out comparison.pdf
