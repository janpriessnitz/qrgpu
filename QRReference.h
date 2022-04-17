
#ifndef QRREFERENCE_H
#define QRREFERENCE_H


#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "Matrix.h"

Matrix QRReferenceCuSolver(const Matrix &A, uint64_t *us_taken);

#endif
