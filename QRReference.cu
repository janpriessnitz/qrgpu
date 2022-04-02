
#include "QRReference.h"

#include <cstdio>
#include "Matrix.h"


void printMatrix(int m, int n, const real*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            real Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}


// Taken from https://docs.nvidia.com/cuda/cusolver/index.html#ormqr-example1
Matrix QRReferenceCuSolver(const Matrix &A, const Matrix &B) {
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;

    // cuda algorithms use matrix representation with row as leading dimension
    auto AT = A.getT();
    auto BT = B.getT();
    const int cols = A.cols;
    const int rows = A.rows;
    const int lda = rows;
    const int nrhs = B.cols; // number of right hand side vectors
    const int ldb = B.rows;
    /*      | 1 2 3 |
    *   A = | 4 5 6 |
    *       | 2 1 1 |
    *       | 1 1 10 |
    *
    *   x = (1 1 1)'
    *   b = (6 15 4 1)'
    */

    Matrix Xprime(nrhs, rows); // solution matrix from GPU with "incorrect" size
    Matrix XT(nrhs, cols); // correct size
/* device memory */
    real *d_A = NULL;
    real *d_tau = NULL;
    real *d_B  = NULL;
    int *devInfo = NULL;
    real *d_work = NULL;
    int  lwork_geqrf = 0;
    int  lwork_ormqr = 0;
    int  lwork = 0;

    int info_gpu = 0;

    const real one = 1;

    // printf("A = (matlab base-1)\n");
    // printMatrix(rows, cols, AT.data, lda, "A");
    // printf("=====\n");
    // printf("B = (matlab base-1)\n");
    // printMatrix(rows, nrhs, BT.data, ldb, "B");
    // printf("=====\n");

/* step 1: create cudense/cublas handle */
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

/* step 2: copy A and B to device */
    cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(real) * lda * cols);
    cudaStat2 = cudaMalloc ((void**)&d_tau, sizeof(real) * cols);
    cudaStat3 = cudaMalloc ((void**)&d_B  , sizeof(real) * ldb * nrhs);
    cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    cudaStat1 = cudaMemcpy(d_A, AT.data, sizeof(real) * lda * cols   , cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, BT.data, sizeof(real) * ldb * nrhs, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    /* step 3: query working space of geqrf and ormqr */
    cusolver_status = cusolverDnDgeqrf_bufferSize(
    // cusolver_status = cusolverDnSgeqrf_bufferSize(
        cusolverH,
        rows,
        cols,
        d_A,
        lda,
        &lwork_geqrf);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cusolver_status= cusolverDnDormqr_bufferSize(
    // cusolver_status= cusolverDnSormqr_bufferSize(
        cusolverH,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,
        rows,
        nrhs,
        cols,
        d_A,
        lda,
        d_tau,
        d_B,
        ldb,
        &lwork_ormqr);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    lwork = (lwork_geqrf > lwork_ormqr)? lwork_geqrf : lwork_ormqr;

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(real)*lwork);
    assert(cudaSuccess == cudaStat1);

/* step 4: compute QR factorization */
    cusolver_status = cusolverDnDgeqrf(
    // cusolver_status = cusolverDnSgeqrf(
        cusolverH,
        rows,
        cols,
        d_A,
        lda,
        d_tau,
        d_work,
        lwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    /* check if QR is good or not */
    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    // printf("after geqrf: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);
    /* step 5: compute Q^T*B */
    cusolver_status= cusolverDnDormqr(
    // cusolver_status= cusolverDnSormqr(
        cusolverH,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,
        rows,
        nrhs,
        cols,
        d_A,
        lda,
        d_tau,
        d_B,
        ldb,
        d_work,
        lwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);
    /* check if QR is good or not */
    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    // printf("after ormqr: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

/* step 6: compute x = R \ Q^T*B */
    cublas_status = cublasDtrsm(
    // cublas_status = cublasStrsm(
         cublasH,
         CUBLAS_SIDE_LEFT,
         CUBLAS_FILL_MODE_UPPER,
         CUBLAS_OP_N,
         CUBLAS_DIAG_NON_UNIT,
         cols,
         nrhs,
         &one,
         d_A,
         lda,
         d_B,
         ldb);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(Xprime.data, d_B, sizeof(real)*rows*nrhs, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    // printf("X = (matlab base-1)\n");
    // printMatrix(rows, nrhs, Xprime.data, ldb, "X");

    // "cut" Xprime to the right size
    for (int r = 0; r < nrhs; ++r) {
        for (int c = 0; c < cols; ++c) {
            XT(r, c) = Xprime(r, c);
        }
    }

/* free resources */
    if (d_A    ) cudaFree(d_A);
    if (d_tau  ) cudaFree(d_tau);
    if (d_B    ) cudaFree(d_B);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    if (cublasH ) cublasDestroy(cublasH);
    if (cusolverH) cusolverDnDestroy(cusolverH);
    cudaDeviceReset();
    return XT.getT();
}
