// CUDA rhypot function!!

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <chrono>

#include "BlockHouseholderKernel.h"

#define BLOCKDIM_X_HOUSE 1024
#define BLOCKDIM_X_V 512
#define BLOCKDIM_X_CALC_YPRIME 512
#define BLOCKDIM_X_CALCW 32
#define BLOCKDIM_X_COPYW 32

#define POS(r, c, ld) ((c)*(ld) + (r))


cublasHandle_t cublasH = NULL;


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
    for (unsigned int bound = dimX/2; bound > 0; bound /= 2) {
        if (x < bound) {
            s[x] += s[x + bound];
        }
        __syncthreads();
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

    for (unsigned int i = x; i < m; i += dimX) {
        if (i < col) V[i] = 0;
        else if (i == col) {
        }
        else {
            V[i] = A[POS(i, col, ld)]*sqrtbeta;
            A[POS(i, col, ld)] *= sqrtbeta/tau;
        }
    }
}

void house(real *A, int m, int ld, int col, real *V, real *taus, real *beta, int startc) {
    householder_calc_beta<<<1, BLOCKDIM_X_HOUSE>>>(A, m, ld, col, V, startc); 
}

__global__ void calc_and_add_V(real *A, int m, int ld, int n, real *V, real startc, real *Vprime) {
    int col = blockIdx.x + startc + 1;
    int x = threadIdx.x;
    int dimX = blockDim.x;

    __shared__ real s[BLOCKDIM_X_V];

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

    for (int i = x + startc; i < m; i += dimX) {
        A[POS(i, col, ld)] -= V[i]*s[0];
    }
}

void rankOneUpdate(real *A, int m, int ld, int n, real *V, int startc, real *Vprime) {
    if (n - startc - 1 > 0)
        calc_and_add_V<<<n-startc-1, BLOCKDIM_X_V>>>(A, m, ld, n, V, startc, Vprime);
}

void rankRUpdate(real *A, int m, int ld, int n, int startc, int startn, int R, real *Y, real *W, real *Wprime) {
    real one = 1;
    real zero = 0;
    cublas_gemm(cublasH,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n - startn, R, m - startc,
                &one,
                A + POS(startc, startn, ld), ld,
                W + POS(startc, 0, m), m,
                &zero,
                Wprime + POS(startn, 0, n), n);
                
    cublas_gemm(cublasH,
                CUBLAS_OP_N, CUBLAS_OP_T,
                m - startc, n - startn, R,
                &one,
                Y + POS(startc, 0, m), m,
                Wprime + POS(startn, 0, n), n,
                &one,
                A + POS(startc, startn, ld), ld);
}


__global__ void calc_Yprime(int m, int startc, real *Y, real *V, real *Yprime) {
    int j = blockIdx.x;
    int x = threadIdx.x;
    int dimX = blockDim.x;

    __shared__ real s[BLOCKDIM_X_CALC_YPRIME];

    s[x] = 0;
    for (unsigned int i = startc + x; i < m; i += dimX) {
        s[x] += V[i]*Y[POS(i, j, m)];
    }
    __syncthreads();
    for (unsigned int bound = dimX/2; bound > 0; bound /= 2) {
        if (x < bound) {
            s[x] += s[x + bound];
        }
        __syncthreads();
    }
    if (x == 0) {
        Yprime[j] = s[0];
    }
}

__global__ void calc_W(int m, int col, int startc, real *W, real *Y, real *Yprime) {
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    if (x < m) {
        real z = 0;
        for (int i = 0; i < col; ++i) {
            z += W[POS(x, i, m)]*Yprime[i];
        }
        z += Y[POS(x, col, m)];
        z *= -1;
        W[POS(x, col, m)] = z;
    }
}

__global__ void copy_W(int m, int startc, real *Y, real *W) {
    int x = threadIdx.x + blockIdx.x*blockDim.x + startc;
    if (x < m) {
        W[x] = -Y[x];
    }
}

void append_W(int m, int col, int startc, real *Y, real *W, real *Wprime) {
    if (col == 0) {
        int blockdim = min(BLOCKDIM_X_COPYW, m - startc);
        copy_W<<<(m-startc+blockdim-1)/blockdim, blockdim>>>(m ,startc, Y, W);
    } else {
        real *V = Y + POS(0, col, m);
        calc_Yprime<<<col, BLOCKDIM_X_CALC_YPRIME>>>(m, startc, Y, V, Wprime);
        int blockdim = min(BLOCKDIM_X_CALCW, m);
        calc_W<<<(m + blockdim-1)/blockdim, blockdim>>>(m, col, startc, W, Y, Wprime);
    }
}

void doOneBlock(real *A, int m, int ld, int n, int R, int col, real *Y, real *W, real *Wprime, real *beta, real *taus) {
    for (int i = 0; i < R; ++i) {
        int curcol = col + i;
        if (curcol >= m - 1) goto decomp_finished;
        real *curV = Y + POS(0, i, m);
        house(A, m, ld, curcol, curV, taus, beta, col);
        rankOneUpdate(A, m, ld, col + R, curV, curcol, Wprime);
        append_W(m, i, curcol, Y, W, Wprime);
    }
    rankRUpdate(A, m, ld, n, col, col+R, R, Y, W, Wprime);
    decomp_finished:;
}

void QRBlockSolve(real *A, real *taus, int m, int na, int nb, int ld, int R, uint64_t *usec_taken) {
    real *beta;
    real *Y;
    real *W;
    real *Wprime;


    cublasStatus_t cublas_status = cublasCreate(&cublasH);

    if (cudaMalloc((void**)&beta, sizeof(beta)) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }
    if (cudaMalloc((void**)&Y, sizeof(Y)*R*m) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }
    if (cudaMalloc((void**)&W, sizeof(W)*R*m) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }
    if (cudaMalloc((void**)&Wprime, sizeof(Wprime)*R*(na+nb)) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }

    cudaDeviceSynchronize();
    auto cuStart = std::chrono::high_resolution_clock::now();

    for (int col = 0; col < na-1; col += R) {
        int curR = min(R, na - col);
        doOneBlock(A, m, ld, (na+nb), curR, col, Y, W, Wprime, beta, taus);
    }

    cudaDeviceSynchronize();
    auto cuEnd = std::chrono::high_resolution_clock::now();
    auto cuDuration = std::chrono::duration_cast<std::chrono::microseconds>(cuEnd - cuStart).count();
    *usec_taken = cuDuration;

    if (cublasH) cublasDestroy(cublasH);

    cudaFree(beta);
    cudaFree(Y);
    cudaFree(W);
    cudaFree(Wprime);

}
