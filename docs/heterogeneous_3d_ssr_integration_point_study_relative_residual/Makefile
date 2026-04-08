LATEXMK ?= latexmk
LATEXMKFLAGS ?= -pdf -interaction=nonstopmode -halt-on-error
PYTHON ?= $(abspath ../../.venv/bin/python)
SCRIPTS = scripts
REPORT_BASENAME = heterogeneous_3d_ssr_integration_point_study_relative_residual
MAIN = $(REPORT_BASENAME).tex

.PHONY: assets pdf clean

assets:
	PYTHONPATH=../../src $(PYTHON) $(SCRIPTS)/build_report_assets.py

pdf: assets
	$(LATEXMK) $(LATEXMKFLAGS) $(MAIN)

clean:
	$(LATEXMK) -C $(MAIN)
