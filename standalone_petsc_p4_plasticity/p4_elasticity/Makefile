PETSC_DIR ?= ../../.build/src/petsc-3.24.5
PETSC_ARCH ?= linux-c-opt

.DEFAULT_GOAL := all

include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscvariables
include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscrules

CFLAGS += -std=c11 -O2 -g
CPPFLAGS += $(PETSC_CC_INCLUDES)

COMMON_OBJS = p4_elasticity_common.o

cube_elasticity: cube_elasticity.o $(COMMON_OBJS)
	$(PCC_LINKER) -o $@ cube_elasticity.o $(COMMON_OBJS) $(PETSC_LIB)

l1_elasticity: l1_elasticity.o $(COMMON_OBJS)
	$(PCC_LINKER) -o $@ l1_elasticity.o $(COMMON_OBJS) $(PETSC_LIB)

all: cube_elasticity l1_elasticity

clean::
	$(RM) *.o cube_elasticity l1_elasticity

.PHONY: all
