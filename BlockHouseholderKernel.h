
#ifndef BLOCKHOUSEHOLDERKERNEL_H
#define BLOCKHOUSEHOLDERKERNEL_H


#include "Matrix.h"
#include <cuda_runtime.h>

void QRBlockSolve(real *A, int m, int na, int nb, int ld, int R);

#endif
