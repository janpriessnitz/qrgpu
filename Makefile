CC = nvc++

CUDAPATH = ${CUDA_PATH}
MATHPATH = /home/jpriessnitz/hpc_sdk/Linux_x86_64/21.7/math_libs

SOURCES = *.cpp *.cu
# {
# 	CPUValidation.cpp
# 	Matrix.cpp
# 	ProblemGenerator.cpp
# 	QRTuning.cpp
# }

main: $(SOURCES)
	$(CC) -std=c++11 -pedantic -Wall -Wextra -o main -I$(CUDAPATH)/include -I$(MATHPATH)/include -L$(CUDAPATH)/libs -L$(MATHPATH)/lib64 -lcudart -lcublas -lcusolver $(SOURCES)
