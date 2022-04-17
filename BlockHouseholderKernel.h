
#ifndef BLOCKHOUSEHOLDERKERNEL_H
#define BLOCKHOUSEHOLDERKERNEL_H


#include "Matrix.h"
#include <cuda_runtime.h>

void QRBlockSolve(real *A, real *taus, int m, int na, int ld, uint64_t *usec_taken);

#endif
