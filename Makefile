PETSC_DIR ?= .build/src/petsc-3.24.5
PETSC_ARCH ?= linux-c-opt
PYTHON ?= $(abspath .venv/bin/python)
ENGINE_PYTHONPATH := $(abspath src)$(if $(PYTHONPATH),:$(PYTHONPATH))

.DEFAULT_GOAL := all

.PHONY: all extension smoke clean

all: extension

extension:
	PETSC_DIR=$(PETSC_DIR) PETSC_ARCH=$(PETSC_ARCH) $(PYTHON) setup.py build_ext --inplace

smoke: extension
	OMP_NUM_THREADS=1 PYTHONPATH="$(ENGINE_PYTHONPATH)" mpiexec -n 1 $(PYTHON) -m petsc_ssr.runners.local_case \
	  --use-box-mesh --pc-variant none --petsc-opt=-ksp_type --petsc-opt=preonly \
	  --petsc-opt=-pc_type --petsc-opt=lu --omega-max 10 --continuation-step-max 3 \
	  --output-dir .local/tmp/smoke_box

clean:
	$(RM) -r build dist petsc_ssr.egg-info
	$(RM) src/petsc_ssr/native/_core*.so src/petsc_ssr/native/_core.c src/petsc_ssr/native/cython/_core.c
	find src/petsc_ssr/native -name '*.o' -delete
