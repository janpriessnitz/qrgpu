#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <chrono>

#include "BlockHouseholderKernel.h"

#define BLOCKDIM_X_HOUSE 1024
#define BLOCKDIM_X_V 1024

#define BLOCKDIM_X_CALC_YPRIME 128
#define BLOCKDIM_X_CALCW 128
#define BLOCKDIM_X_COPYW 1024

#define HOUSEHOLDER_BLOCK_SIZE 96

#define POS(r, c, ld) ((c)*(ld) + (r))


cublasHandle_t cublasH = NULL;
cudaStream_t stream1, stream2, stream3, stream4;



__device__ void warpReduce(volatile real *s, int tid) {
    s[tid] += s[tid + 32];
    s[tid] += s[tid + 16];
    s[tid] += s[tid + 8];
    s[tid] += s[tid + 4];
    s[tid] += s[tid + 2];
    s[tid] += s[tid + 1];
}

__global__ void householder_calc_beta(real *A, int m, int ld, int col, real *V, int startc) {
    int x = threadIdx.x;
    int dimX = blockDim.x;
    __shared__ real s[BLOCKDIM_X_HOUSE];
    __shared__ real beta;
    __shared__ real tau;

    s[x] = 0;
    for (unsigned int i = col + x; i < m; i += dimX) {
        s[x] += A[POS(i, col, ld)]*A[POS(i, col, ld)];
    }
    __syncthreads();
    for (unsigned int bound = dimX/2; bound > 32; bound >>= 1) {
        if (x < bound) {
            s[x] += s[x + bound];
        }
        __syncthreads();
    }
    if (x < 32) {
         warpReduce(s, x);
    }

    // s[0] is magX2
    __shared__ real magX;
    __shared__ real sqrtbeta;
    if (x == 0) {
        magX = sqrt(s[0]);

        if (A[POS(col, col, ld)] > 0) {
            real oldVal = A[POS(col, col, ld)];
            real newVal = oldVal + magX;
            real magV2 = s[0] - oldVal*oldVal + newVal*newVal;
            beta = 2/magV2;
            sqrtbeta = sqrt(beta);
            tau = (A[POS(col, col, ld)] + magX)*sqrtbeta;
            A[POS(col, col, ld)] = -magX;

        } else {
            real oldVal = A[POS(col, col, ld)];
            real newVal = oldVal - magX;
            real magV2 = s[0] - oldVal*oldVal + newVal*newVal;
            beta = 2/magV2;
            sqrtbeta = sqrt(beta);
            tau = (A[POS(col, col, ld)] - magX)*sqrtbeta;
            A[POS(col, col, ld)] = magX;
        }
        V[col] = tau;
    }

    __syncthreads();

    for (unsigned int i = x + startc; i < m; i += dimX) {
        if (i < col) V[i] = 0;
        else if (i == col) {
        }
        else {
            V[i] = A[POS(i, col, ld)]*sqrtbeta;
            A[POS(i, col, ld)] *= sqrtbeta/tau;
        }
    }
}

__global__ void calc_and_add_V(real *A, int m, int ld, real *V, real *Vprime) {
    int col = blockIdx.x;
    int x = threadIdx.x;
    int dimX = blockDim.x;

    __shared__ real s[BLOCKDIM_X_V];

    if (x < m) {
        s[x] = V[x]*A[POS(x, col, ld)];
    } else {
        s[x] = 0;
    }
    for (unsigned int i = x + dimX; i < m; i += dimX) {
        s[x] += V[i]*A[POS(i, col, ld)];
    }
    __syncthreads();
    for (unsigned int bound = dimX/2; bound > 32; bound >>= 1) {
        if (x < bound) {
            s[x] += s[x + bound];
        }
        __syncthreads();
    }
    if (x < 32) {
        warpReduce(s, x);
    }
    __syncthreads();

    for (int i = x; i < m; i += dimX) {
        A[POS(i, col, ld)] -= V[i]*s[0];
    }
}

__global__ void calc_W(int m, int startc, real *W, real *Y, real *Yprime, int R) {
    int x = threadIdx.x + blockIdx.x*blockDim.x + startc;
    if (x < m) {
        for (int col = 0; col < R; ++col) {
            real z = 0;
            for (int i = 0; i < col; ++i) {
                z -= W[POS(x, i, m)]*Yprime[POS(i, col, R)];
            }
            z -= Y[POS(x, col, m)];
            W[POS(x, col, m)] = z;
        }
    }
}

__global__ void copy_W(int m, real *Y, real *W) {
    int x = threadIdx.x;
    int dimX = blockDim.x;
    for (int i = x; i < m; i += dimX) {
        W[i] = -Y[i];
    }
}

__global__ void calc_Yprime(real *Y, int m, int startc, int R, real *Yprime) {
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;
    int x = threadIdx.x;
    int dimX = blockDim.x;

    __shared__ real s[BLOCKDIM_X_CALC_YPRIME];

    int col = blockX, endcol = blockY;
    if (col > endcol) return;

    s[x] = 0;  // Yprime
    for (unsigned int i = startc + x; i < m; i += dimX) {
        s[x] += Y[POS(i, col, m)]*Y[POS(i, endcol, m)];
    }
    __syncthreads();
    for (unsigned int bound = dimX/2; bound > 32; bound >>= 1) {
        if (x < bound) {
            s[x] += s[x + bound];
        }
        __syncthreads();
    }
    if (x < 32) {
        warpReduce(s, x);
    }
    __syncthreads();

    if (x == 0) {
        Yprime[POS(col, endcol, R)] = s[0];
    }
}

void doOneBlock(real *A, int m, int ld, int n, int R, int col, real *Y, real *W, real *Wprime, real *taus) {
    int startc = col;
    int startn = col + R;

    real one = 1;
    real zero = 0;

    int blockdimW = BLOCKDIM_X_CALCW;
    while(blockdimW > m - col && blockdimW > 64) blockdimW >>= 1;

    int blockdimH = BLOCKDIM_X_HOUSE;
    while(blockdimH > m - col && blockdimH > 64) blockdimH >>= 1;

    for (int i = 0; i < R; ++i) {
        int curcol = col + i;
        if (curcol >= m - 1) goto decomp_finished;
        real *curV = Y + POS(0, i, m);
        householder_calc_beta<<<1, blockdimH, 0, stream1>>>(A, m, ld, curcol, curV, col);

        if (R - i - 1 > 0) {
            int blockdim = BLOCKDIM_X_V;
            while(blockdim > m - curcol && blockdim > 64) blockdim >>= 1;
            calc_and_add_V<<<R-i-1, blockdim, 0, stream1>>>(A + POS(curcol, curcol+1, ld), m - curcol, ld, curV + curcol, Wprime);
        }
    }
    calc_Yprime<<<dim3(R, R), BLOCKDIM_X_CALC_YPRIME, 0, stream1>>>(Y, m, col, R, Wprime);
    copy_W<<<1, BLOCKDIM_X_COPYW, 0, stream1>>>(m - col, Y + col, W + col);
    calc_W<<<(m - col + blockdimW-1)/blockdimW, blockdimW, 0, stream1>>>(m, col, W, Y, Wprime, R);

    // TODO: TSMTTSM https://journals.sagepub.com/doi/full/10.1177/1094342020965661
    // taking 12/72
    // cublas_gemm(cublasH,
    //             CUBLAS_OP_T, CUBLAS_OP_N,
    //             n - startn, R, m - startc,
    //             &one,
    //             A + POS(startc, startn, ld), ld,
    //             W + POS(startc, 0, m), m,
    //             &zero,
    //             Wprime + POS(startn, 0, n), n);

    cublas_gemm(cublasH,
                CUBLAS_OP_T, CUBLAS_OP_N,
                R, n - startn, m - startc,
                &one,
                W + POS(startc, 0, m), m,
                A + POS(startc, startn, ld), ld,
                &zero,
                Wprime + POS(0, startn, R), R);

    // matmul_TN<<<dim3(n - startn, R), dim3(1, 1)>>>(
    //     A + POS(startc, startn, ld), ld,
    //     W + POS(startc, 0, m), m,
    //     Wprime + POS(startn, 0, n), n,
    //     n - startn, R, m - startc);

    // TODO: TSMM https://journals.sagepub.com/doi/full/10.1177/1094342020965661
    // taking 22/72
    // cublas_gemm(cublasH,
    //             CUBLAS_OP_N, CUBLAS_OP_T,
    //             m - startc, n - startn, R,
    //             &one,
    //             Y + POS(startc, 0, m), m,
    //             Wprime + POS(startn, 0, n), n,
    //             &one,
    //             A + POS(startc, startn, ld), ld);

    cublas_gemm(cublasH,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m - startc, n - startn, R,
            &one,
            Y + POS(startc, 0, m), m,
            Wprime + POS(0, startn, R), R,
            &one,
            A + POS(startc, startn, ld), ld);
    // floatTSM2Kernel<8,8,8><<<>>>(Y + POS(startc, 0, m), Wprime + POS(startn, 0, n), A + POS(startc, startn, ld), R, m - startc, n - startn);
    // matmul_NT_add<<<(m-startc + MATMULNT_T1 - 1)/MATMULNT_T1, MATMULNT_T1>>>(
    //     Y + POS(startc, 0, m), m,
    //     Wprime + POS(startn, 0, n), n,
    //     A + POS(startc, startn, ld), ld,
    //     m - startc, n - startn, R);

    decomp_finished:;
}

void QRBlockSolve(real *A, real *taus, int m, int n, int ld, uint64_t *usec_taken) {
    real *Y;
    real *W;
    real *Wprime;

    int R = HOUSEHOLDER_BLOCK_SIZE;

    cublasStatus_t cublas_status = cublasCreate(&cublasH);

    cudaStreamCreate( &stream1);
    cudaStreamCreate( &stream2);
    cudaStreamCreate( &stream3);

    cublasSetStream(cublasH, stream1);

    if (cudaMalloc((void**)&Y, sizeof(Y)*R*m) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }
    if (cudaMalloc((void**)&W, sizeof(W)*R*m) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }
    if (cudaMalloc((void**)&Wprime, sizeof(Wprime)*R*n) != cudaSuccess) {  // TODO: might be faulty for (na+nb) < R
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }

    cudaDeviceSynchronize();
    auto cuStart = std::chrono::high_resolution_clock::now();

    for (int col = 0; col < n-1; col += R) {
        if (n-1-col <= 4096) {
            R = 64;
        }
        // if (n-1-col <= 2048) {
        //     R = 48;
        // }
        if (n-1-col <= 1024) {
            R = 32;
        }
        int curR = min(R, n - col);
        doOneBlock(A, m, ld, n, curR, col, Y, W, Wprime, taus);
    }

    cudaDeviceSynchronize();
    auto cuEnd = std::chrono::high_resolution_clock::now();
    auto cuDuration = std::chrono::duration_cast<std::chrono::microseconds>(cuEnd - cuStart).count();
    *usec_taken = cuDuration;

    if (cublasH) cublasDestroy(cublasH);

    cudaFree(Y);
    cudaFree(W);
    cudaFree(Wprime);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
}
