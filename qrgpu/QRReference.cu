
#include "QRReference.h"

#include <cstdio>
#include <chrono>

#include "Matrix.h"

// Taken from https://docs.nvidia.com/cuda/cusolver/index.html#ormqr-example1
Matrix QRReferenceCuSolver(const Matrix &A, uint64_t *us_taken) {
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
    const int cols = A.cols;
    const int rows = A.rows;
    const int lda = rows;
    /*      | 1 2 3 |
    *   A = | 4 5 6 |
    *       | 2 1 1 |
    *       | 1 1 10 |
    */

/* device memory */
    real *d_A = NULL;
    real *d_tau = NULL;
    int *devInfo = NULL;
    real *d_work = NULL;
    int  lwork_geqrf = 0;
    int  lwork = 0;

/* step 1: create cudense/cublas handle */
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

/* step 2: copy A and B to device */
    cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(real) * lda * cols);
    cudaStat2 = cudaMalloc ((void**)&d_tau, sizeof(real) * cols);
    cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    cudaStat1 = cudaMemcpy(d_A, AT.data, sizeof(real) * lda * cols, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    /* step 3: query working space of geqrf and ormqr */
    cusolver_status = cusolverDn_geqrf_bufferSize(
        cusolverH,
        rows,
        cols,
        d_A,
        lda,
        &lwork_geqrf);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    lwork = lwork_geqrf;

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(real)*lwork);
    assert(cudaSuccess == cudaStat1);

    cudaDeviceSynchronize();
    auto cuStart = std::chrono::high_resolution_clock::now();


/* step 4: compute QR factorization */
    cusolver_status = cusolverDn_geqrf(
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

    cudaDeviceSynchronize();
    auto cuEnd = std::chrono::high_resolution_clock::now();
    auto cuDuration = std::chrono::duration_cast<std::chrono::microseconds>(cuEnd - cuStart).count();
    *us_taken = cuDuration;

    cudaStat1 = cudaMemcpy(AT.data, d_A, sizeof(real) * lda * cols, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    // real taus[100];
    // cudaStat1 = cudaMemcpy(taus, d_tau, sizeof(real) * cols, cudaMemcpyDeviceToHost);
    // assert(cudaSuccess == cudaStat1);

    // printf("cuSolver:\n");
    // AT.getT().print();
    // printf("taus:\n");
    // for (int i = 0; i < cols; ++i) {
    //     printf("%.4f ", taus[i]);
    // }
    // printf("\n");

    if (d_A    ) cudaFree(d_A);
    if (d_tau  ) cudaFree(d_tau);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    if (cublasH ) cublasDestroy(cublasH);
    if (cusolverH) cusolverDnDestroy(cusolverH);
    cudaDeviceReset();

    return AT.getT();
}
