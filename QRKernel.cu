// CUDA rhypot function!!

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

#include "QRKernel.h"

#define POS(n, m, r, c) ((r)*(m) + c)
#define MAX_N 1024
#define BLOCKDIM_X 1024




__global__ void blockHouse(double *A, int n, int m, double *Q, double *R, int c) {

}

void QRBlockHouseholder(double *A, int n, int m, double *Q, double *R) {

}





// What dimensions??
// blockDim = m/colBlocks
// blocks = colBlocks

// blockDim = n - c
__global__ void house2(double *A, int n, int m, int c) {
    int x = threadIdx.x;
    int dimX = blockDim.x;
    int blockX = blockIdx.x;

    double __shared__ magx2 = 0;
    double magv2 = 0;
    double __shared__ x_sum[BLOCKDIM_X];

    x_sum[x] = 0;
    for (int i = c+x; i < n; i += dimX) {
        if (i < n) {
            x_sum[x] += A[POS(n, m, i, c)]*A[POS(n, m, i, c)];
        }
    }
    // TODO: make it faster
    if (x == 0) {
        for (int i = 0; i < dimX; ++i) {
            magx2 += x_sum[i];
        }
    }
    __syncthreads();
    double magx = sqrt(magx2);
    double oldVal = A[POS(n, m, c, c)];
    double newVal = oldVal - magx;
    magv2 = magx2 - oldVal*oldVal + newVal*newVal;
    double beta = -2/magv2;

    // CALCULATE R
    double AprimeX;

    // do a special case for v[c] which is x[c] - |x|
    AprimeX = A[POS(n, m, c, x+c)]*newVal;
    for (int i = c+1; i < n; ++i) {
        AprimeX += A[POS(n, m, i, x+c)]*A[POS(n, m, i, c)];
    }
    // precompute |x| for next iteration??

    // do a special case for v[c] which is x[c] - |x|
    A[POS(n, m, c, x+c)] = A[POS(n, m, c, x+c)] + beta*newVal*AprimeX;
    // R[x] = A[x];
    // R[x] = dimX;
    for (int i = c+1; i < n; ++i) {
        A[POS(n, m, i, x+c)] = A[POS(n, m, i, x+c)] + beta*A[POS(n, m, i, c)]*AprimeX;
    }

}

// blockdim = m - c
__global__ void house(double *A, int n, int m, double *Q, double *R, int c) {
    int x = threadIdx.x;
    int dimX = blockDim.x;
    int blockX = blockIdx.x;

    double __shared__ magx2 = 0;
    double magv2 = 0;
    double __shared__ x_sum[BLOCKDIM_X];

    x_sum[x] = 0;
    for (int i = c+x; i < n; i += dimX) {
        if (i < n) {
            x_sum[x] += A[POS(n, m, i, c)]*A[POS(n, m, i, c)];
        }
    }
    // TODO: make it faster
    if (x == 0) {
        for (int i = 0; i < dimX; ++i) {
            magx2 += x_sum[i];
        }
    }
    __syncthreads();
    double magx = sqrt(magx2);
    double oldVal = A[POS(n, m, c, c)];
    double newVal = oldVal - magx;
    magv2 = magx2 - oldVal*oldVal + newVal*newVal;
    double beta = -2/magv2;

    // CALCULATE R
    double AprimeX;

    // do a special case for v[c] which is x[c] - |x|
    AprimeX = A[POS(n, m, c, x+c)]*newVal;
    for (int i = c+1; i < n; ++i) {
        AprimeX += A[POS(n, m, i, x+c)]*A[POS(n, m, i, c)];
    }
    // precompute |x| for next iteration??

    // do a special case for v[c] which is x[c] - |x|
    R[POS(n, m, c, x+c)] = A[POS(n, m, c, x+c)] + beta*newVal*AprimeX;
    // R[x] = A[x];
    // R[x] = dimX;
    for (int i = c+1; i < n; ++i) {
        R[POS(n, m, i, x+c)] = A[POS(n, m, i, x+c)] + beta*A[POS(n, m, i, c)]*AprimeX;
    }

    // CALCULATE Q
    for (int j = x; j < n; j += dimX) {
        Q[POS(n, n, j, 0)] = beta*A[POS(n, m, j, c)]*newVal;
        for (int i = 1; i < n; ++i) {
            Q[POS(n, n, j, i)] = beta*A[POS(n, m, j, c)]*A[POS(n, m, i, c)];
        }
        Q[POS(n, n, j, j)] += 1;
    }
    // Q[x] = beta;
    // Clean for future iterations
    // A[POS(n, m, x+c, c)] = 0;
    // A[POS(n, m, x+c, c)] = POS(n, m, x+c, c);
    // A[((x+c)*m + c)] = 0;
    // A[POS(n, m, c, x+c)] = R[POS(n, m, c, x+c)];
    // A[POS(n, m, c, x+c)] = POS(n, m+100, 0, 0);


}

__global__ void copy(double *target, double *dest, int num) {
    int x = threadIdx.x;
    int dimX = blockDim.x;
    int blockX = blockIdx.x;
    int numBlocks = gridDim.x;

    for (int i = x + dimX*blockX; i < num; i += dimX*numBlocks) {
        if (i < num) {
            target[i] = dest[i];
        }
    }
}

void QRHouseholder(double *A, int n, int m, double *Q, double *R) {
    // // int iters = (n == m) ? (n - 1) : m;
    // int iters = 3;
    // for (int c = 0; c < iters; ++c) {
    //     if(c % 2 == 0) {
    //         house<<<dim3(1), dim3(m-c)>>>(A, n, m, Q, R, c);
    //     } else {
    //         // reuse R as A and vice versa
    //         house<<<dim3(1), dim3(m-c)>>>(R, n, m, Q, A, c);
    //     }
    // }
    // // // TODO: memcpy(R, A) in case of (iters % 2) == 0
    // if (iters % 2 == 0) {
    //     copy<<<dim3(10), dim3(10)>>>(R, A, n*m);
    // }
    int iters = (n == m) ? (n - 1) : m;
    // int iters = 3;
    for (int c = 0; c < iters; ++c) {
        house2<<<dim3(1), dim3(m-c)>>>(A, n, m, c);
    }
    copy<<<dim3(10), dim3(10)>>>(R, A, n*m);
}

void QRSolve(double *A, int n, int m, int m_expanded) {
    int iters = m;
    for (int c = 0; c < iters; ++c) {
        house2<<<dim3(1), dim3(m_expanded-c)>>>(A, n, m_expanded, c);
    }
}
