#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <utility>

#include "Matrix.h"
#include "QRKernel.h"
#include "BlockHouseholderKernel.h"

#include "QRInvoke.h"

Matrix InvokeSolve(Matrix *A, real *taus, int cols, uint64_t *usec_taken) {
  int rows = A->rows;
  int cols_expanded = A->cols;
  auto AT = A->getT();
  real *dA = NULL;
  real *dTaus = NULL;
  Matrix QR(A->cols, A->rows);

  int device = 0;
  if (cudaSetDevice(device) != cudaSuccess){
      fprintf(stderr, "Cannot set CUDA device!\n");
      exit(1);
  }
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

  // allocate and set device memory
  if (cudaMalloc((void**)&dA, rows*cols_expanded*sizeof(dA[0])) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      goto cleanup;
  }
  cudaMemcpy(dA, AT.data, rows*cols_expanded*sizeof(dA[0]), cudaMemcpyHostToDevice);

  if (cudaMalloc((void**)&dTaus, sizeof(real)*cols) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      goto cleanup;
  }

  cudaDeviceSynchronize();
  QRBlockSolve(dA, dTaus, rows, cols, rows, usec_taken);
  // QRSolve(dA, rows, cols, cols_expanded - cols, rows);
  cudaDeviceSynchronize();

  // cudaMemcpy(taus, dTaus, cols*sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(QR.data, dA, rows*cols_expanded*sizeof(dA[0]), cudaMemcpyDeviceToHost);
  cleanup:
    if (dA) cudaFree(dA);
    if (dTaus) cudaFree(dTaus);
    cudaDeviceReset();

  return QR.getT();
}
