
#ifndef BLOCKHOUSEHOLDERKERNEL_H
#define BLOCKHOUSEHOLDERKERNEL_H


#include <cuda_runtime.h>

// __global__ void house(double *A, int n, int m, int c);
// M - extended matrix
void QRBlockSolve(double *A, int m, int na, int nb, int ld, int R);

#endif
