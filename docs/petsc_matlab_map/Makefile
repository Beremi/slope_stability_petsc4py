LATEXMK ?= latexmk
LATEXMKFLAGS ?= -pdf -interaction=nonstopmode -halt-on-error
MAIN = main.tex

.PHONY: pdf clean

pdf:
	$(LATEXMK) $(LATEXMKFLAGS) $(MAIN)

clean:
	$(LATEXMK) -C $(MAIN)
