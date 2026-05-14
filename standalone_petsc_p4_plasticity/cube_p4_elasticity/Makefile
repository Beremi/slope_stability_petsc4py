PETSC_DIR ?= ../../.build/src/petsc-3.24.5
PETSC_ARCH ?= linux-c-opt

.DEFAULT_GOAL := all

include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscvariables
include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscrules

CFLAGS += -std=c11 -O2 -g
CPPFLAGS += $(PETSC_CC_INCLUDES)

cube_p4_elasticity: cube_p4_elasticity.o
	$(PCC_LINKER) -o $@ cube_p4_elasticity.o $(PETSC_LIB)

all: cube_p4_elasticity

clean::
	$(RM) cube_p4_elasticity.o cube_p4_elasticity

.PHONY: all
