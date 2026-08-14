ROOT_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
INCL_DIR = -I$(ROOT_DIR)/SourceCodes -I$(ROOT_DIR)
# sm_70 was dropped in CUDA 13; native targets the GPU present at build time
CMPL_OPT = -O3 -std=c++11 --ptxas-options=-v --gpu-architecture=native -lineinfo -maxrregcount=128
SOURCE   = Bench.cu

# Problem, solver and trajectory count are compile-time constants.
PROBLEM ?= lorenz
SOLVER  ?= RKCK45
NT      ?= 8388608
DEFINES  = -DPROBLEM_HEADER='"problems/$(PROBLEM).cuh"' -DSOLVER_CHOICE=$(SOLVER) -DNT_VALUE=$(NT)

all: Bench.exe

Bench.exe: $(SOURCE) problems/$(PROBLEM).cuh problems/stubs.cuh
	nvcc -o	Bench.exe $(SOURCE) $(INCL_DIR) $(DEFINES) $(CMPL_OPT)

clean:
	rm -f Bench.exe
