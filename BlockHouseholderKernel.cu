// CUDA rhypot function!!

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <chrono>

#include "BlockHouseholderKernel.h"

#define BLOCKDIM_X_HOUSE 512
#define BLOCKDIM_X_SUMSQUARE 1024
#define BLOCKDIM_X_VPRIME 512
#define BLOCKDIM_X_ADD 32
#define BLOCKDIM_Y_ADD 8
#define BLOCKDIM_X_ADD_SEQ 32
#define BLOCKDIM_X_CALC_YPRIME 256
#define BLOCKDIM_X_WPRIME 32
#define BLOCKDIM_X_RADD 16
#define BLOCKDIM_Y_RADD 16
#define BLOCKDIM_X_CALCW 32
#define BLOCKDIM_X_COPYW 32
#define TILE_SIZE 16
#define TILE_SIZE_W 16
#define TILE_SIZE_A 16
#define TILE_SIZE_VERT 16
#define TILE_SIZE_ADD 16

#define WPRIME_VERT 16
#define WPRIME_N 32

#define POS(r, c, ld) ((c)*(ld) + (r))


cublasHandle_t cublasH = NULL;

__global__ void sumSquares(real *A, int start, int end, real *result) {
    int x = threadIdx.x;
    int dimX = blockDim.x;

    __shared__ real s[BLOCKDIM_X_SUMSQUARE];

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

__global__ void house_internal(real *A, int m, int ld, int col, real *V, real *magX2) {
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    if (x < col) V[x] = 0;
    else if (x == col) V[x] = A[POS(x, col, ld)] - sqrt(*magX2);
    else if (x > col and x < m)
        V[x] = A[POS(x, col, ld)];
}

__global__ void calc_beta(real *magX2, real *oldVal, real *beta) {
    real newVal = *oldVal - sqrt(*magX2);
    real magV2 = *magX2 - (*oldVal)*(*oldVal) + newVal*newVal;
    *beta = 2/magV2;
}

__global__ void householder_calc_beta(real *A, int m, int ld, int col, real *V) {
    int x = threadIdx.x;
    int dimX = blockDim.x;
    __shared__ real s[BLOCKDIM_X_SUMSQUARE];
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
    if (x == 0) {
        if (A[POS(col, col, ld)] > 0) {
            real oldVal = A[POS(col, col, ld)];
            real newVal = oldVal + sqrt(s[0]);
            real magV2 = s[0] - oldVal*oldVal + newVal*newVal;
            beta = 2/magV2;
            tau = (A[POS(col, col, ld)] + sqrt(s[0]))*sqrt(beta);
            A[POS(col, col, ld)] = -sqrt(s[0]);

        } else {
            real oldVal = A[POS(col, col, ld)];
            real newVal = oldVal - sqrt(s[0]);
            real magV2 = s[0] - oldVal*oldVal + newVal*newVal;
            beta = 2/magV2;
            tau = (A[POS(col, col, ld)] - sqrt(s[0]))*sqrt(beta);
            A[POS(col, col, ld)] = sqrt(s[0]);
        }
        V[col] = tau;
    }

    __syncthreads();

    for (unsigned int i = x; i < m; i += dimX) {
        if (i < col) V[i] = 0;
        else if (i == col) {
            // A[POS(col, col, ld)] -=
            // V[i] = tau;
            // A[POS(col, col, ld)] = sqrt(s[0]);
            // A[POS(col, col, ld)] = V[i];
            // A[POS(col, col, ld)] = (sqrt(s[0])+2)*sqrt(*beta);
        }
        else {
            V[i] = A[POS(i, col, ld)]*sqrt(beta);
            // A[POS(i, col, ld)] = A[POS(i, col, ld)]/(sqrt(s[0])+2);
            A[POS(i, col, ld)] *= sqrt(beta)/tau;
        }
    }
}

void house(real *A, int m, int ld, int col, real *V, real *taus, real *beta) {
    // sumSquares<<<1, BLOCKDIM_X_SUMSQUARE>>>(A, POS(col, col, ld), POS(m, col, ld), magX2);
    // calc_beta<<<1, 1>>>(magX2, A + POS(col, col, ld), beta);
    // int blockxdim =  min(m, BLOCKDIM_X_HOUSE);
    // house_internal<<<(m+blockxdim-1)/blockxdim, blockxdim>>>(A, m, ld, col, V, magX2);
    householder_calc_beta<<<1, BLOCKDIM_X_SUMSQUARE>>>(A, m, ld, col, V); 
}

__global__ void calc_Vprime(real *A, int m, int ld, int n, real *V, real startc, real *Vprime) {
    int col = blockIdx.x + startc;
    int x = threadIdx.x;
    int dimX = blockDim.x;

    __shared__ real s[BLOCKDIM_X_VPRIME];

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

__global__ void calc_and_add_V(real *A, int m, int ld, int n, real *V, real startc, real *Vprime) {
    int col = blockIdx.x + startc + 1;
    int x = threadIdx.x;
    int dimX = blockDim.x;

    __shared__ real s[BLOCKDIM_X_VPRIME];

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
    // if (x == 0) {
    //     Vprime[col] = s[0];
    // }
    // __syncthreads();

    for (int i = x + startc; i < m; i += dimX) {
        // A[POS(i, col, ld)] -= (*beta)*V[i]*Vprime[col];
        // A[POS(i, col, ld)] -= (*beta)*V[i]*s[0];
        A[POS(i, col, ld)] -= V[i]*s[0];
    }
}


__global__ void add_VVprime(real *A, int m, int ld, int n, real *V, int startc, real *Vprime, real *beta) {
    int r = threadIdx.x + blockDim.x*blockIdx.x + startc;
    int c = threadIdx.y + blockDim.y*blockIdx.y + startc;
    if (r < m && c < n)
        A[POS(r, c, ld)] -= (*beta)*V[r]*Vprime[c];
}

__global__ void add_VVprime_seq(real *A, int m, int ld, int n, real *V, int startc, real *Vprime, real *beta) {
    int r = threadIdx.x + blockDim.x*blockIdx.x + startc;
    int rStep = blockDim.x*gridDim.x;
    int c = threadIdx.y + blockDim.y*blockIdx.y + startc;
    if (c < n) {
        for (int ri = r; ri < m; ri += rStep) {
            A[POS(ri, c, ld)] -= (*beta)*V[ri]*Vprime[c];
        }
    }
}

void rankOneUpdate(real *A, int m, int ld, int n, real *V, int startc, real *Vprime) {
    // real one = 1;
    // real zero = 0;
    // cublas_gemv(cublasH, CUBLAS_OP_T,
    //                        m - startc, n - startc,
    //                        &one,
    //                        A + POS(startc, startc, ld), ld,
    //                        V + startc, 1,
    //                        &zero,
    //                        Vprime + startc, 1);
    // calc_Vprime<<<n-startc, BLOCKDIM_X_VPRIME>>>(A, m, ld, n, V, startc, Vprime);
    // int blockdimx = min((m - startc), BLOCKDIM_X_ADD);
    // int blockdimy = min((n - startc), BLOCKDIM_Y_ADD);
    // add_VVprime<<<dim3((m-startc+blockdimx-1)/blockdimx, (n - startc+blockdimy-1)/blockdimy, 1), dim3(blockdimx, blockdimy, 1)>>>(A, m, ld, n, V, startc, Vprime, beta);
    if (n - startc - 1 > 0)
        calc_and_add_V<<<n-startc-1, BLOCKDIM_X_VPRIME>>>(A, m, ld, n, V, startc, Vprime);

    // int blockdimx = min((m - startc), BLOCKDIM_X_ADD_SEQ);
    // add_VVprime1_seq<<<dim3(1, n - startc, 1), dim3(blockdimx, 1, 1)>>>(A, m, ld, n, V, startc, Vprime, beta);
}

__global__ void calc_Wprime(real *A, int m, int ld, int n, int startc, int startn, int R, real *W, real *Wprime) {
    int col = blockIdx.x + startn;
    int x = threadIdx.x;
    int dimX = blockDim.x;

    int j = threadIdx.y + blockIdx.y*blockDim.y;

    __shared__ real s[BLOCKDIM_X_WPRIME];

    s[x] = 0;
    for (unsigned int i = x + startc; i < m; i += dimX) {
        s[x] += W[POS(i, j, m)]*A[POS(i, col, ld)];
    }
    __syncthreads();
    for (unsigned int bound = dimX/2; bound > 0; bound /= 2) {
        if (x < bound) {
            s[x] += s[x + bound];
        }
        __syncthreads();
    }
    if (x == 0) {
        Wprime[POS(col, j, n)] = s[0];
    }
}

__global__ void add_YWprime(real *A, int m, int ld, int n, int startc, int startn, int R, real *Y, real *Wprime) {
    int r = threadIdx.x + blockDim.x*blockIdx.x + startc;
    int c = threadIdx.y + blockDim.y*blockIdx.y + startn;

    if (r < m && c < n) {
        real addVal = 0;
        for (int i = 0; i < R; ++i) {
            addVal += Y[POS(r, i, m)]*Wprime[POS(c, i, n)];
        }
        A[POS(r, c, ld)] += addVal;
    }
}

void rankRUpdate(real *A, int m, int ld, int n, int startc, int startn, int R, real *Y, real *W, real *Wprime) {
    // calc_Wprime<<<dim3(n-startn, R, 1), dim3(BLOCKDIM_X_WPRIME, 1, 1)>>>(A, m, ld, n, startc, startn, R, W, Wprime);
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


    // int blockdimx = min((m - startc), BLOCKDIM_X_RADD);
    // int blockdimy = min((n - startn), BLOCKDIM_Y_RADD);
    // add_YWprime<<<dim3((m-startc+blockdimx-1)/blockdimx, (n - startn + blockdimy-1)/blockdimy, 1), dim3(blockdimx, blockdimy, 1)>>>(A, m, ld, n, startc, startn, R, Y, Wprime);
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
    // if (x < startc + col) {
        // W[POS(x, col, m)] = 0;
    // }
    if (x < m) {
        real z = 0;
        for (int i = 0; i < col; ++i) {
            z += W[POS(x, i, m)]*Yprime[i];
        }
        z += Y[POS(x, col, m)];
        // z *= -(*beta);
        z *= -1;
        W[POS(x, col, m)] = z;
    }
}

__global__ void copy_W(int m, int startc, real *Y, real *W) {
    int x = threadIdx.x + blockIdx.x*blockDim.x + startc;
    if (x < m) {
        // W[x] = -(*beta)*Y[x];
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
        house(A, m, ld, curcol, curV, taus, beta);
        rankOneUpdate(A, m, ld, col + R, curV, curcol, Wprime);
        append_W(m, i, curcol, Y, W, Wprime);
    }
    rankRUpdate(A, m, ld, n, col, col+R, R, Y, W, Wprime);
    decomp_finished:;
    // printY<<<dim3(m, R), dim3(1,1,1)>>>(A, m, ld, n, R, Y, W);
    // printbeta<<<1, 1>>>(A, beta);
}

// __global__ void zeroLowerTriangular(real *A, int rows, int cols, int cols_extended) {
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



__global__ void copy(real *target, real *dest, int num) {
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
