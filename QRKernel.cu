// CUDA rhypot function!!

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

#include "QRKernel.h"

#define BLOCKDIM_X_HOUSE 32
#define BLOCKDIM_X_SUMSQUARE 32
#define BLOCKDIM_X_VPRIME 32
#define BLOCKDIM_X_ADD 64
#define BLOCKDIM_X_ADD_SEQ 32


#define POS(r, c, ld) ((c)*(ld) + (r))


// TODO: recursive kernel calling for multiple-block reduction
__global__ void sumSquares1(double *A, int start, int end, double *result) {
    int x = threadIdx.x;
    int dimX = blockDim.x;

    __shared__ double s[];

    s[x] = 0;

    for (unsigned int i = start + x; i < end; i += dimX) {
        s[x] += A[i]*A[i];
    }
    __syncthreads();
    for (unsigned int bound = dimX/2; bound > 0; bound /= 2) {
        if (x < bound) {
            s[x] += s[x + bound];
        }
        __syncthreads();
    }
    if (x == 0) {
        *result = s[0];
    }
}

__global__ void house_internal1(double *A, int m, int ld, int col, double *V, double *magX2) {
    int x = threadIdx.x + blockDim.x*blockIdx.x + col;
    if (x == col) V[x] = A[POS(x, col, ld)] - sqrt(*magX2);
    else if (x > col and x < m)
        V[x] = A[POS(x, col, ld)];
}

__global__ void calc_beta1(double *magX2, double *oldVal, double *beta) {
    double newVal = *oldVal - sqrt(*magX2);
    double magV2 = *magX2 - (*oldVal)*(*oldVal) + newVal*newVal;
    *beta = -2/magV2;
}

void house1(double *A, int m, int ld, int col, double *V, double *magX2, double *beta) {
    int vlen = m - col;
    sumSquares1<<<1, min(vlen, BLOCKDIM_X_SUMSQUARE)>>>(A, POS(col, col, ld), POS(m, col, ld), magX2);
    calc_beta1<<<1, 1>>>(magX2, A + POS(col, col, ld), beta);
    int blockxdim =  min(vlen, BLOCKDIM_X_HOUSE);
    house_internal1<<<(vlen+blockxdim-1)/blockxdim, blockxdim>>>(A, m, ld, col, V, magX2);
}

__global__ void calc_Vprime1(double *A, int m, int ld, int n, double *V, double startc, double *Vprime) {
    int col = blockIdx.x + startc;
    int x = threadIdx.x;
    int dimX = blockDim.x;

    __shared__ double s[];

    s[x] = 0;
    for (unsigned int i = startc + x; i < m; i += dimX) {
        s[x] += V[i]*A[POS(i, col, ld)];
    }
    __syncthreads();
    for (unsigned int bound = dimX/2; bound > 0; bound /= 2) {
        if (x < bound) {
            s[x] += s[x + bound];
        }
        __syncthreads();
    }
    if (x == 0) {
        Vprime[col] = s[0];
    }
}

__global__ void add_VVprime1(double *A, int m, int ld, int n, double *V, double startc, double *Vprime, double *beta) {
    int r = threadIdx.x + blockDim.x*blockIdx.x + startc;
    int c = threadIdx.y + blockDim.y*blockIdx.y + startc;
    if (r < m && c < n)
        A[POS(r, c, ld)] += (*beta)*V[r]*Vprime[c];
}

__global__ void add_VVprime1_seq(double *A, int m, int ld, int n, double *V, double startc, double *Vprime, double *beta) {
    int r = threadIdx.x + blockDim.x*blockIdx.x + startc;
    int rStep = blockDim.x*gridDim.x;
    int c = threadIdx.y + blockDim.y*blockIdx.y + startc;
    if (c < n) {
        for (int ri = r; ri < m; ri += rStep) {
            A[POS(ri, c, ld)] += (*beta)*V[ri]*Vprime[c];
        }
    }
}

void rankOneUpdate1(double *A, int m, int ld, int n, double *V, double *beta, int startc, double *Vprime) {
    calc_Vprime1<<<n-startc, min(m - startc, BLOCKDIM_X_VPRIME)>>>(A, m, ld, n, V, startc, Vprime);
    int blockdimx = min((m - startc), BLOCKDIM_X_ADD);
    add_VVprime1<<<dim3((m-startc+blockdimx-1)/blockdimx, n - startc, 1), dim3(blockdimx, 1, 1)>>>(A, m, ld, n, V, startc, Vprime, beta);
    // int blockdimx = min((m - startc), BLOCKDIM_X_ADD_SEQ);
    // add_VVprime1_seq<<<dim3(1, n - startc, 1), dim3(blockdimx, 1, 1)>>>(A, m, ld, n, V, startc, Vprime, beta);
}

// __global__ void zeroLowerTriangular(double *A, int rows, int cols, int cols_extended) {
//     int x = threadIdx.x;
//     int dimX = blockDim.x;
//     int blockX = blockIdx.x;
//     int numBlocks = gridDim.x;
//     for (int r = 0; r < rows; ++r) {
//         for (int c = x; c < cols && c < r; c += dimX) {
//             A[POS(rows, cols_extended, r, c)] = 0;
//         }
//     }
// }

void QRSolve(double *A, int m, int na, int nb, int ld) {
    double *magX2;
    double *beta;
    double *V;
    double *Vprime;

    if (cudaMalloc((void**)&magX2, sizeof(magX2)) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }
    if (cudaMalloc((void**)&beta, sizeof(beta)) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }
    if (cudaMalloc((void**)&V, sizeof(V)*m) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }
    if (cudaMalloc((void**)&Vprime, sizeof(Vprime)*(na+nb)) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }

    for (int i = 0; i < na; ++i) {
        house1(A, m, ld, i, V, magX2, beta);
        rankOneUpdate1(A, m, ld, na+nb, V, beta, i, Vprime);
    }

    // MagX2<<<dim3(1), dim3(BLOCKDIM_X)>>>(A, rows, 0, cols_extended, magX2);
    // for (int i = 0; i < iters; ++i) {
    //     house<<<dim3(1), dim3(cols_extended-i)>>>(A, rows, cols_extended, i, magX2);
    // }
    // zeroLowerTriangular<<<dim3(1), dim3(BLOCKDIM_X)>>>(A, rows, cols, cols_extended);
    // cudaFree(magX2);
}



__global__ void copy1(double *target, double *dest, int num) {
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
