# Build IEEE Access manuscript PDF

MAIN=main
LATEX=pdflatex -interaction=nonstopmode
BIBTEX=bibtex

.PHONY: pdf docx clean

pdf:
	$(LATEX) $(MAIN).tex
	-$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex
	@echo "Output: $(MAIN).pdf"

docx:
	cd .. && python scripts/generate_manuscript_docx.py
	@echo "Output: manuscript.docx (journal-neutral Word)"

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).pdf $(MAIN).synctex.gz

figures:
	cd .. && python scripts/generate_publication.py

all: figures pdf docx
