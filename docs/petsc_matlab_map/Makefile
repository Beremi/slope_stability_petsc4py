LATEXMK ?= latexmk
LATEXMKFLAGS ?= -pdf -interaction=nonstopmode -halt-on-error
REPORT_BASENAME = petsc_to_matlab_solution_map
MAIN = $(REPORT_BASENAME).tex

.PHONY: pdf clean

pdf:
	$(LATEXMK) $(LATEXMKFLAGS) $(MAIN)

clean:
	$(LATEXMK) -C $(MAIN)
