ROOT_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
INCL_DIR = -I$(ROOT_DIR)/SourceCodes -I$(ROOT_DIR)
# sm_70 was dropped in CUDA 13; native targets the GPU present at build time
CMPL_OPT = -O3 -std=c++17 --ptxas-options=-v --gpu-architecture=native -lineinfo -maxrregcount=128
SOURCE   = Bench.cu

# Problem, solver, trajectory count and precision are compile-time constants; run_ode_cpp.sh and .ps1 pass the same defines.
PROBLEM   ?= lorenz
SOLVER    ?= RKCK45
NT        ?= 8388608
PRECISION ?= float
# SD overrides the problem's state count (lorenz96).
SD        ?=
DEFINES  = -DPROBLEM_HEADER='"problems/$(PROBLEM).cuh"' -DSOLVER_CHOICE=$(SOLVER) -DNT_VALUE=$(NT) -DPRECISION_TYPE=$(PRECISION)
ifneq ($(SD),)
DEFINES += -DPROBLEM_SD=$(SD)
endif

all: Bench.exe

Bench.exe: $(SOURCE) protocol.h grid.cuh trial.cuh problems/$(PROBLEM).cuh problems/stubs.cuh
	nvcc -o	Bench.exe $(SOURCE) $(INCL_DIR) $(DEFINES) $(CMPL_OPT)

clean:
	rm -f Bench.exe
