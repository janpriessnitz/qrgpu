
#ifndef QRKERNEL_H
#define QRKERNEL_H


#include <cuda_runtime.h>

void QRBlockHouseholder(double *A, int n, int m, double *Q, double *R);
__global__ void house(double *A, int n, int m, double *Q, double *R, int c);
void QRHouseholder(double *A, int n, int m, double *Q, double *R);

void QRSolve(double *A, int n, int m, int m_expanded);

#endif
