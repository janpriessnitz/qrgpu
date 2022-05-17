# CC = nvcc
CC = /usr/local/cuda-11/bin/nvcc

# CUDAPATH = ${CUDA_PATH}
CUDAPATH = /home/jpriessnitz/hpc_sdk/Linux_x86_64/2021/cuda/11.0
MATHPATH = /home/jpriessnitz/hpc_sdk/Linux_x86_64/21.7/math_libs/11.0
KTTPATH = /home/jpriessnitz/bakina/KTT/Build/x86_64_Release
KTTINCLUDE = /home/jpriessnitz/bakina/KTT/Source

SOURCES = \
	BlockHouseholderKernel.cu \
	CPUValidation.cpp \
	Matrix.cpp \
	QRCPU.cpp \
	QRInvoke.cu \
	QRReference.cu

all: main tune

main: $(SOURCES) *.h JustKernels.cu.nocompile
	$(CC) -std=c++17 -o main -I$(CUDAPATH)/include -I$(MATHPATH)/include -L$(CUDAPATH)/lib64 -L$(MATHPATH)/lib64 -lcudart -lcublas -lcusolver $(SOURCES) main.cu

tune: $(SOURCES) *.h JustKernels.cu.nocompile mainTune.cpp QRTuning.cpp
	$(CC) -std=c++17 -o tune -I$(CUDAPATH)/include -I$(MATHPATH)/include -I$(KTTINCLUDE) -L$(CUDAPATH)/lib64 -L$(MATHPATH)/lib64 -L$(KTTPATH) -lcudart -lcublas -lcusolver -lktt -lnvrtc $(SOURCES) mainTune.cpp QRTuning.cpp