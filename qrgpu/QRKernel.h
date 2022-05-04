
#ifndef QRKERNEL_H
#define QRKERNEL_H


#include <cuda_runtime.h>

void QRSolve(double *A, int m, int na, int nb, int ld);
#endif
