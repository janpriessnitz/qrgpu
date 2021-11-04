#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <utility>

#include "Matrix.h"
#include "QRKernel.h"

#include "QRInvoke.h"

std::pair<Matrix, Matrix> Invoke(const Matrix &A) {
  int n = A.rows;
  int m = A.cols;
  Matrix Q(n, n);
  Matrix R(n, m);

  double *dA = NULL;
  double *dQ = NULL;
  double *dR = NULL;

  int device = 0;
  if (cudaSetDevice(device) != cudaSuccess){
      fprintf(stderr, "Cannot set CUDA device!\n");
      exit(1);
  }
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  printf("Using device %d: \"%s\"\n", device, deviceProp.name);

  // allocate and set device memory
  if (cudaMalloc((void**)&dA, n*m*sizeof(dA[0])) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      goto cleanup;
  }
  if (cudaMalloc((void**)&dQ, n*n*sizeof(dQ[0])) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      goto cleanup;
  }
  if (cudaMalloc((void**)&dR, n*m*sizeof(dR[0])) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      goto cleanup;
  }
  cudaMemcpy(dA, A.data, n*m*sizeof(dA[0]), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  QRHouseholder(dA, n, m, dQ, dR);
  printf("Kernel launch error: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaDeviceSynchronize();

  printf("Copying back to resulting matrices\n");

  cudaMemcpy(Q.data, dQ, n*n*sizeof(dQ[0]), cudaMemcpyDeviceToHost);
  cudaMemcpy(R.data, dR, n*m*sizeof(dR[0]), cudaMemcpyDeviceToHost);
  cleanup:
    if (dA) cudaFree(dA);
    if (dQ) cudaFree(dQ);
    if (dR) cudaFree(dR);

  return std::pair<Matrix, Matrix>(Q, R);
}

void InvokeSolve(Matrix *A, int cols) {
  int n = A->rows;
  int m = A->cols;

  double *dA = NULL;

  int device = 0;
  if (cudaSetDevice(device) != cudaSuccess){
      fprintf(stderr, "Cannot set CUDA device!\n");
      exit(1);
  }
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  printf("Using device %d: \"%s\"\n", device, deviceProp.name);

  // allocate and set device memory
  if (cudaMalloc((void**)&dA, n*m*sizeof(dA[0])) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      goto cleanup;
  }
  cudaMemcpy(dA, A->data, n*m*sizeof(dA[0]), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  QRSolve(dA, n, cols, m);
  printf("Kernel launch error: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaDeviceSynchronize();

  printf("Copying back to resulting matrices\n");

  cudaMemcpy(A->data, dA, n*m*sizeof(dA[0]), cudaMemcpyDeviceToHost);
  cleanup:
    if (dA) cudaFree(dA);

}
