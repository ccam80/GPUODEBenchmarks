ROOT_DIR = $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
INCL_DIR = -I$(ROOT_DIR)/SourceCodes
# sm_70 was dropped in CUDA 13; native targets the GPU present at build time
CMPL_OPT = -O3 -std=c++11 --ptxas-options=-v --gpu-architecture=native -lineinfo -maxrregcount=128
SOURCE   = Lorenz.cu

all: Lorenz.exe

Lorenz.exe: $(SOURCE)
	nvcc -o	Lorenz.exe $(SOURCE) $(INCL_DIR) $(CMPL_OPT)

clean:
	rm -f Lorenz.exe
