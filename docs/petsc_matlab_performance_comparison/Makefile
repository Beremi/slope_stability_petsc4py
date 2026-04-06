LATEXMK ?= latexmk
LATEXMKFLAGS ?= -pdf -interaction=nonstopmode -halt-on-error
PYTHON ?= python
SCRIPTS = scripts
REPORT_BASENAME = petsc_vs_matlab_3d_ssr_performance_comparison
MAIN = $(REPORT_BASENAME).tex

.PHONY: data assets pdf clean

data:
	$(PYTHON) $(SCRIPTS)/collect_results.py

assets: data
	$(PYTHON) $(SCRIPTS)/build_report_assets.py

pdf: assets
	$(LATEXMK) $(LATEXMKFLAGS) $(MAIN)

clean:
	$(LATEXMK) -C $(MAIN)
