PETSC_DIR ?= ../.build/src/petsc-3.24.5
PETSC_ARCH ?= linux-c-opt

.DEFAULT_GOAL := all

include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscvariables
include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscrules

CFLAGS += -std=c11 -O2 -g
CPPFLAGS += $(PETSC_CC_INCLUDES)

OBJS = p4_plasticity.o assembly.o material_mc.o p4_basis.o

p4_plasticity: $(OBJS)
	$(PCC_LINKER) -o $@ $(OBJS) $(PETSC_LIB)

all: p4_plasticity

clean::
	$(RM) $(OBJS) p4_plasticity

.PHONY: all
