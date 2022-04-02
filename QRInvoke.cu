#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <utility>

#include "Matrix.h"
#include "QRKernel.h"
#include "BlockHouseholderKernel.h"

#include "QRInvoke.h"

void InvokeSolve(Matrix *A, int cols) {
  int rows = A->rows;
  int cols_expanded = A->cols;
  auto AT = A->getT();
  real *dA = NULL;

  int device = 0;
  if (cudaSetDevice(device) != cudaSuccess){
      fprintf(stderr, "Cannot set CUDA device!\n");
      exit(1);
  }
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  printf("Using device %d: \"%s\"\n", device, deviceProp.name);

  // allocate and set device memory
  if (cudaMalloc((void**)&dA, rows*cols_expanded*sizeof(dA[0])) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      goto cleanup;
  }
  cudaMemcpy(dA, AT.data, rows*cols_expanded*sizeof(dA[0]), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  QRBlockSolve(dA, rows, cols, cols_expanded - cols, rows, 64);
  // QRSolve(dA, rows, cols, cols_expanded - cols, rows);
  printf("Kernel launch error: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaDeviceSynchronize();

  printf("Copying back to resulting matrices\n");

  cudaMemcpy(AT.data, dA, rows*cols_expanded*sizeof(dA[0]), cudaMemcpyDeviceToHost);
  *A = AT.getT();
  cleanup:
    if (dA) cudaFree(dA);

}
