# Build IEEE Access manuscript PDF

MAIN=main
LATEX=pdflatex
BIBTEX=bibtex

.PHONY: pdf clean

pdf:
	$(LATEX) $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex
	@echo "Output: $(MAIN).pdf"

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).pdf $(MAIN).synctex.gz

figures:
	cd .. && python scripts/generate_publication.py

all: figures pdf
