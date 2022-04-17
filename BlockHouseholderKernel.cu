// CUDA rhypot function!!

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <chrono>

#include "BlockHouseholderKernel.h"

#define BLOCKDIM_X_HOUSE 1024
#define BLOCKDIM_X_V 1024

#define BLOCKDIM_X_CALC_YPRIME 128
#define BLOCKDIM_X_CALCW 64
#define BLOCKDIM_X_COPYW 1024

#define BLOCKDIM_MATMUL_SKINNY 16

#define HOUSEHOLDER_BLOCK_SIZE 48

#define POS(r, c, ld) ((c)*(ld) + (r))


cublasHandle_t cublasH = NULL;
cudaStream_t stream1, stream2, stream3, stream4;


// NOTE- m is the common dimension between matrix A and B
template <int t1, int t2, int t3>
__global__ void floatTSM2Kernel(const float* A, const float* B, float* C,
                                const unsigned int m, const unsigned int n,
                                const unsigned int k)
{
    // Names mostly follow the published code
    __shared__ float currB[t1 * t2];

    float currA[t3];
    float nextA[t3];
    float nextB[t2];
    float currC[t2];

    const int tid = threadIdx.x;
    int threadBase = (blockIdx.x * blockDim.x);
    int thread;

    // This implementation can respond to arbitrary input

    // We cannot rule out a thread's participation based on
    // whether it corresponds to a row in Matrix A, so we
    // introduce threadBase.
    for (; threadBase < m; threadBase += blockDim.x * gridDim.x)
    {
        thread = threadBase + tid;
        for (int p = 0; p < n; p += t2)
        {
            // Load loops have extra conditionals to ensure
            // they do not make bad memory accesses

            // Loads first tile of output registers and A
            if (thread < m)
            {
                #pragma unroll
                for (int i = 0; i < t2; ++i)
                {
                    if (p + i < n)
                    {
                        currC[i] = C[thread + ((p + i) * m)];
                    }
                }
                // Loads currA
                #pragma unroll
                for (int i = 0; i < t3; ++i)
                {
                    if (i < k)
                    {
                        currA[i] = A[thread + (i * m)];
                    }
                }
            }
            // Loads tile of B
            if (tid < k)
            {
                #pragma unroll
                for (int i = 0; i < t2; ++i)
                {
                    if (p + i < n)
                    {
                        currB[tid + (i * t1)] = B[tid + ((p + i) * k)];
                    }
                }
            }

            // Outer product loop
            for (int j = 0; j < k; j += t1)
            {
                __syncthreads();
                // Loads next tile of B
                if (j + t1 + tid < k)
                {
                    #pragma unroll
                    for (int i = 0; i < t2; ++i)
                    {
                        if (p + i < n)
                        {
                            nextB[i] = B[(j + t1 + tid) + ((p + i) * k)];
                        }
                    }
                }

                const int t3mod = t1 % t3;

                // Loop over A's columns
                for (int l = j; l < j + (t1 - t3mod) && l < k; l += t3)
                {
                    // Loads next A
                    #pragma unroll
                    for (int i = 0; i < t3; ++i)
                    {
                        if (l + t3 + i < k && thread < m)
                        {
                            nextA[i] = A[thread + ((l + t3 + i) * m)];
                        }
                    }

                    // Floating Point Operations (lines 32-34)
                    // Each thread does t2 * t3 mults

                    // Either dispatch guarded or unguarded instructions based on
                    // position in matrix A
                    if (l + t3 <= k)
                    {
                        // It is assumed that B[(l - j) .. (l - j) + t3 - 1, _]  exist
                        #pragma unroll
                        for (int a = 0; a < t2; ++a)
                        {
                            #pragma unroll
                            for (int b = 0; b < t3; ++b)
                            {
                                currC[a] += currA[b] * currB[(l - j) + b + (a * t1)];
                            }
                        }
                    }
                    else
                    {
                        #pragma unroll
                        for (int a = 0; a < t2; ++a)
                        {
                            #pragma unroll
                            for (int b = 0; b < t3; ++b)
                            {
                                if (l + b < k)
                                {
                                    currC[a] += currA[b] * currB[(l - j) + b + (a * t1)];
                                }
                            }
                        }
                    }

                    // Stores next A in curr A
                    #pragma unroll
                    for (int i = 0; i < t3; ++i)
                    {
                        currA[i] = nextA[i];
                    }
                }
                // Accommodates t3 that do not divide t1.
                #pragma unroll
                for (int a = 0; a < t2; ++a)
                {
                    #pragma unroll
                    for (int b = 0; b < t3mod; ++b)
                    {
                        if (j + t1 - t3mod + b < k)
                        {
                            currC[a] += currA[b] * currB[(t1 - t3mod + b) + (a * t1)];
                        }
                    }
                }

                __syncthreads();

                // Loads currB from each thread's nextB
                #pragma unroll
                for (int i = 0; i < t2; ++i)
                {
                    currB[tid + (i * t1)] = nextB[i];
                }

                // Loads next currA
                if (t3mod != 0)
                {
                    #pragma unroll
                    for (int i = 0; i < t3; ++i)
                    {
                        if (j + t1 + i < k && thread < m)
                        {
                            currA[i] = A[thread + ((j + t1 + i) * m)];
                        }
                    }
                }
            }
            // Stores C
            if (thread < m)
            {
                #pragma unroll
                for (int i = 0; i < t2; ++i)
                {
                    if (p + i < n)
                    {
                        C[thread + ((p + i) * m)] = currC[i];
                    }
                }
            }
        }
    }
}


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

void house(real *A, int m, int ld, int col, real *V, real *taus, real *beta, int startc) {
    startc >>= 5;
    startc <<= 5;
    householder_calc_beta<<<1, BLOCKDIM_X_HOUSE, 0, stream1>>>(A, m, ld, col, V, startc);
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

void rankOneUpdate(real *A, int m, int ld, int n, real *V, int startc, real *Vprime) {
    if (n - startc - 1 == 0)
        return;
    int blockdim =  BLOCKDIM_X_V;
    while(blockdim > m - startc && blockdim > 64) blockdim >>= 1;
    calc_and_add_V<<<n-startc-1, blockdim, 0, stream1>>>(A + POS(startc, startc+1, ld), m - startc, ld, V + startc, Vprime);
}

void rankRUpdate(real *A, int m, int ld, int n, int startc, int startn, int R, real *Y, real *W, real *Wprime) {
    real one = 1;
    real zero = 0;
    // TODO: TSMTTSM https://journals.sagepub.com/doi/full/10.1177/1094342020965661
    // taking 12/72
    cublas_gemm(cublasH,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n - startn, R, m - startc,
                &one,
                A + POS(startc, startn, ld), ld,
                W + POS(startc, 0, m), m,
                &zero,
                Wprime + POS(startn, 0, n), n);

    // TODO: TSMM https://journals.sagepub.com/doi/full/10.1177/1094342020965661
    // taking 22/72
    cublas_gemm(cublasH,
                CUBLAS_OP_N, CUBLAS_OP_T,
                m - startc, n - startn, R,
                &one,
                Y + POS(startc, 0, m), m,
                Wprime + POS(startn, 0, n), n,
                &one,
                A + POS(startc, startn, ld), ld);
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

__global__ void copy_W2(int m, int startc, real *Y, real *W) {
    int x = threadIdx.x + startc;
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

void doOneBlock(real *A, int m, int ld, int n, int R, int col, real *Y, real *W, real *Wprime, real *beta, real *taus) {
    int blockdimW = BLOCKDIM_X_CALCW;
    while(blockdimW > m - col && blockdimW > 64) blockdimW >>= 1;

    for (int i = 0; i < R; ++i) {
        int curcol = col + i;
        if (curcol >= m - 1) goto decomp_finished;
        real *curV = Y + POS(0, i, m);
        house(A, m, ld, curcol, curV, taus, beta, col);
        rankOneUpdate(A, m, ld, col + R, curV, curcol, Wprime);
    }
    calc_Yprime<<<dim3(R, R), BLOCKDIM_X_CALC_YPRIME, 0, stream1>>>(Y, m, col, R, Wprime);
    copy_W2<<<1, BLOCKDIM_X_COPYW, 0, stream1>>>(m, col, Y, W);
    calc_W<<<(m - col + blockdimW-1)/blockdimW, blockdimW, 0, stream1>>>(m, col, W, Y, Wprime, R);

    rankRUpdate(A, m, ld, n, col, col+R, R, Y, W, Wprime);
    decomp_finished:;
}

void QRBlockSolve(real *A, real *taus, int m, int na, int ld, uint64_t *usec_taken) {
    real *beta;
    real *Y;
    real *W;
    real *Wprime;

    int R = HOUSEHOLDER_BLOCK_SIZE;

    cublasStatus_t cublas_status = cublasCreate(&cublasH);

    cudaStreamCreate( &stream1);
    cudaStreamCreate( &stream2);
    cudaStreamCreate( &stream3);

    cublasSetStream(cublasH, stream1);

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
    if (cudaMalloc((void**)&Wprime, sizeof(Wprime)*R*na) != cudaSuccess) {  // TODO: might be faulty for (na+nb) < R
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }

    cudaDeviceSynchronize();
    auto cuStart = std::chrono::high_resolution_clock::now();

    for (int col = 0; col < na-1; col += R) {
        if (na-1-col <= 1000) {
            R = 32;
        }
        int curR = min(R, na - col);
        doOneBlock(A, m, ld, na, curR, col, Y, W, Wprime, beta, taus);
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

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
}
