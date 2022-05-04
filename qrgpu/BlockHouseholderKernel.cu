#include <cstdio>
#include <cublas_v2.h>
#include <chrono>

#include "BlockHouseholderKernel.h"
#include "JustKernels.cu.nocompile"

cublasHandle_t cublasH = NULL;

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
        if ((n == m && curcol >= n - 1) || curcol >= n) goto decomp_finished;
        real *curV = Y + POS(0, i, m);
        householder_calc_beta<<<1, blockdimH>>>(A, m, ld, curcol, curV, col);

        if (R - i - 1 > 0) {
            int blockdim = BLOCKDIM_X_V;
            while(blockdim > m - curcol && blockdim > 64) blockdim >>= 1;
            calc_and_add_V<<<R-i-1, blockdim>>>(A + POS(curcol, curcol+1, ld), m - curcol, ld, curV + curcol, Wprime);
        }
    }
    calc_Yprime<<<dim3(R, R), BLOCKDIM_X_CALC_YPRIME>>>(Y, m, col, R, Wprime);
    copy_W<<<1, BLOCKDIM_X_COPYW>>>(m - col, Y + col, W + col);
    calc_W<<<(m - col + blockdimW-1)/blockdimW, blockdimW>>>(m, col, W, Y, Wprime, R);

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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int R = HOUSEHOLDER_BLOCK_SIZE;

    cublasStatus_t cublas_status = cublasCreate(&cublasH);

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

    // Trick: a phony call to cublas so it initializes now and will not hold up the computation later
    real one = 1;
    real zero = 0;
    cublas_gemm(cublasH,
            CUBLAS_OP_T, CUBLAS_OP_N,
            1, 1, 1,
            &one,
            W, 1,
            A, 1,
            &zero,
            Wprime, 1);


    cudaEventRecord(start);
    // cudaDeviceSynchronize();
    // auto cuStart = std::chrono::high_resolution_clock::now();

    for (int col = 0; (n == m && col < n-1) || col < n; col += R) {
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
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    *usec_taken = milliseconds*1000;

    // cudaDeviceSynchronize();
    // auto cuEnd = std::chrono::high_resolution_clock::now();
    // auto cuDuration = std::chrono::duration_cast<std::chrono::microseconds>(cuEnd - cuStart).count();
    // *usec_taken = cuDuration;

    if (cublasH) cublasDestroy(cublasH);

    cudaFree(Y);
    cudaFree(W);
    cudaFree(Wprime);
}
