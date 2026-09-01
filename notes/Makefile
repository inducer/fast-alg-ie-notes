PYTHON := python

FIGURE_SCRIPT := make-fast-alg-figures.py
FIGURES := $(filter-out %-crop.pdf,$(wildcard media/bhut-*.pdf media/fmm-*.pdf))
GENERATED_FIGURES := $(filter-out %-crop.pdf,$(wildcard media/bhut-*.pdf media/fmm-mtol-*.pdf))
CROPPED_FIGURES := $(FIGURES:.pdf=-crop.pdf)
ORGMK_FILES := $(shell find orgmk -type f)
PDFS := notes.pdf notes-folded.pdf

.PHONY: all all-versions figures crops watch

all: notes.pdf

all-versions: $(PDFS)

figures: $(GENERATED_FIGURES)

crops: $(CROPPED_FIGURES)

# One invocation produces every figure emitted by the Python generator.
$(GENERATED_FIGURES) &: $(FIGURE_SCRIPT)
	$(PYTHON) $<

$(CROPPED_FIGURES): media/%-crop.pdf: media/%.pdf
	pdfcrop "$<" "$@"

notes.tex: notes.org run-org-conversion.sh orgmk-extra-conf.el $(ORGMK_FILES) $(CROPPED_FIGURES)
	./run-org-conversion.sh

notes-folded.tex: notes.tex
	grep -v '\\showhiddentrue' "$<" > "$@"

ifneq ($(shell command -v latexrun),)
$(PDFS): %.pdf: %.tex
	latexrun --verbose-cmds -O out "$<"
else
$(PDFS): %.pdf: %.tex
	mkdir -p out
	pdflatex -output-directory=out "$<"
	pdflatex -output-directory=out "$<"
	pdflatex -output-directory=out "$<"
	cp "out/$*.pdf" "$@"
endif

watch:
	git ls-files | entr $(MAKE) all
