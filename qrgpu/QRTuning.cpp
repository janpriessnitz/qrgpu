#include <Ktt.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>


#include "Matrix.h"

#include "QRTuning.h"

void QRTuning(pos_t rows, pos_t cols, std::string paramsFile) {

    auto A = Matrix::GenerateRandom(cols, rows);
    auto AT = A.getT();
    real *dA = NULL;
    real *dTaus = NULL;
    real *dY;
    real *dW;
    real *dWprime;

    if (cudaSetDevice(0) != cudaSuccess){
        fprintf(stderr, "Cannot set CUDA device!\n");
        exit(1);
    }

    // allocate and set device memory
    if (cudaMalloc((void**)&dA, rows*cols*sizeof(dA[0])) != cudaSuccess) {
        fprintf(stderr, "Device memory allocation error!\n");
        exit(1);
    }

    if (cudaMalloc((void**)&dTaus, sizeof(real)*cols) != cudaSuccess) {
        fprintf(stderr, "Device memory allocation error!\n");
        exit(1);
    }


    int maxR = 256;

    if (cudaMalloc((void**)&dY, sizeof(dY[0])*maxR*rows) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }
    if (cudaMalloc((void**)&dW, sizeof(dW[0])*maxR*rows) != cudaSuccess) {
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }
    if (cudaMalloc((void**)&dWprime, sizeof(dWprime[0])*maxR*cols) != cudaSuccess) {  // TODO: might be faulty for (na+nb) < R
      fprintf(stderr, "Device memory allocation error!\n");
      return;
    }


    ktt::DeviceIndex deviceIndex = 0;
    std::string kernelFile = "JustKernels.cu.nocompile";

    ktt::Tuner tuner(0, deviceIndex, ktt::ComputeApi::CUDA);


    const ktt::DimensionVector householderGriddim(1);
    const ktt::DimensionVector householderBlockdim(1);
    const ktt::KernelDefinitionId householder = tuner.AddKernelDefinitionFromFile("householder_calc_beta", kernelFile, householderGriddim,
        householderBlockdim);
    const ktt::ArgumentId householder_A = tuner.AddArgumentVector<real>(reinterpret_cast<ktt::ComputeBuffer>(dA), rows*cols*sizeof(dA[0]), ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);
    const ktt::ArgumentId householder_m = tuner.AddArgumentScalar(rows);
    const ktt::ArgumentId householder_ld = tuner.AddArgumentScalar(rows);
    const ktt::ArgumentId householder_col = tuner.AddArgumentScalar(0);
    const ktt::ArgumentId householder_V = tuner.AddArgumentVector<real>(reinterpret_cast<ktt::ComputeBuffer>(dY), rows*maxR*sizeof(dY[0]), ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);
    const ktt::ArgumentId householder_startc = tuner.AddArgumentScalar(0);
    tuner.SetArguments(householder, {householder_A, householder_m, householder_ld, householder_col, householder_V, householder_startc});

    const ktt::DimensionVector calcVGriddim(1);
    const ktt::DimensionVector calcVBlockdim(1);
    const ktt::KernelDefinitionId calcV = tuner.AddKernelDefinitionFromFile("calc_and_add_V", kernelFile, calcVGriddim,
        calcVBlockdim);
    const ktt::ArgumentId calcV_A = tuner.AddArgumentVector<real>(reinterpret_cast<ktt::ComputeBuffer>(dA), rows*cols*sizeof(dA[0]), ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);
    const ktt::ArgumentId calcV_m = tuner.AddArgumentScalar(rows);
    const ktt::ArgumentId calcV_ld = tuner.AddArgumentScalar(rows);
    const ktt::ArgumentId calcV_V = tuner.AddArgumentVector<real>(reinterpret_cast<ktt::ComputeBuffer>(dY), rows*maxR*sizeof(dY[0]), ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);
    const ktt::ArgumentId calcV_Vprime = tuner.AddArgumentVector<real>(reinterpret_cast<ktt::ComputeBuffer>(dWprime), maxR*cols*sizeof(dWprime[0]), ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);
    tuner.SetArguments(calcV, {calcV_A, calcV_m, calcV_ld, calcV_V, calcV_Vprime});

    const ktt::DimensionVector calcWGriddim(1);
    const ktt::DimensionVector calcWBlockdim(1);
    const ktt::KernelDefinitionId calcW = tuner.AddKernelDefinitionFromFile("calc_W", kernelFile, calcWGriddim,
        calcWBlockdim);
    const ktt::ArgumentId calcW_m = tuner.AddArgumentScalar(rows);
    const ktt::ArgumentId calcW_startc = tuner.AddArgumentScalar(0);
    const ktt::ArgumentId calcW_W = tuner.AddArgumentVector<real>(reinterpret_cast<ktt::ComputeBuffer>(dW), rows*maxR*sizeof(dW[0]), ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);
    const ktt::ArgumentId calcW_Y = tuner.AddArgumentVector<real>(reinterpret_cast<ktt::ComputeBuffer>(dY), rows*maxR*sizeof(dY[0]), ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);
    const ktt::ArgumentId calcW_Yprime = tuner.AddArgumentVector<real>(reinterpret_cast<ktt::ComputeBuffer>(dWprime), maxR*cols*sizeof(dWprime[0]), ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);
    const ktt::ArgumentId calcW_R = tuner.AddArgumentScalar(64);
    tuner.SetArguments(calcW, {calcW_m, calcW_startc, calcW_W, calcW_Y, calcW_Yprime, calcW_R});

    const ktt::DimensionVector copyWGriddim(1);
    const ktt::DimensionVector copyWBlockdim(1);
    const ktt::KernelDefinitionId copyW = tuner.AddKernelDefinitionFromFile("copy_W", kernelFile, copyWGriddim,
        copyWBlockdim);
    const ktt::ArgumentId copyW_m = tuner.AddArgumentScalar(rows);
    const ktt::ArgumentId copyW_Y = tuner.AddArgumentVector<real>(reinterpret_cast<ktt::ComputeBuffer>(dY), rows*maxR*sizeof(dY[0]), ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);
    const ktt::ArgumentId copyW_W = tuner.AddArgumentVector<real>(reinterpret_cast<ktt::ComputeBuffer>(dW), rows*maxR*sizeof(dW[0]), ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);
    tuner.SetArguments(copyW, {copyW_m, copyW_Y, copyW_W});

    const ktt::DimensionVector calcYprimeGriddim(1);
    const ktt::DimensionVector calcYprimeBlockdim(1);
    const ktt::KernelDefinitionId calcYprime = tuner.AddKernelDefinitionFromFile("calc_Yprime", kernelFile, calcYprimeGriddim,
        calcYprimeBlockdim);
    const ktt::ArgumentId calcYprime_Y = tuner.AddArgumentVector<real>(reinterpret_cast<ktt::ComputeBuffer>(dY), rows*maxR*sizeof(dY[0]), ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);
    const ktt::ArgumentId calcYprime_m = tuner.AddArgumentScalar(rows);
    const ktt::ArgumentId calcYprime_startc = tuner.AddArgumentScalar(0);
    const ktt::ArgumentId calcYprime_R = tuner.AddArgumentScalar(64);
    const ktt::ArgumentId calcYprime_Yprime = tuner.AddArgumentVector<real>(reinterpret_cast<ktt::ComputeBuffer>(dWprime), cols*maxR*sizeof(dWprime[0]), ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);
    tuner.SetArguments(calcYprime, {calcYprime_Y, calcYprime_m, calcYprime_startc, calcYprime_R, calcYprime_Yprime});


    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = cublasCreate(&cublasH);


    auto kernel = tuner.CreateCompositeKernel("QRBlockHouseholder", {householder, calcV, calcW, copyW, calcYprime},
    [AT, dA, householder, cols, rows, householder_col, householder_startc, calcV, calcV_m,
     calcW, calcW_startc, calcW_R, copyW, copyW_m, calcYprime, calcYprime_startc, calcYprime_R,
     dY, dW, dWprime, cublasH](ktt::ComputeInterface& interface)
    {
        cudaMemcpy(dA, AT.data, AT.rows*AT.cols*sizeof(dA[0]), cudaMemcpyHostToDevice);
        int R = 0;
        int blockdimV = 0;
        int blockdimH = 0;
        int blockdimW = 0;
        int blockdimCopyW = 0;
        int blockdimCalcYprime = 0;
        auto conf = interface.GetCurrentConfiguration();
        for (auto confpair : conf.GetPairs()) {
            if (confpair.GetName() == "HOUSEHOLDER_BLOCK_SIZE") {
                R = confpair.GetValue();
            } else if (confpair.GetName() == "BLOCKDIM_X_V") {
                blockdimV = confpair.GetValue();
            } else if (confpair.GetName() == "BLOCKDIM_X_HOUSE") {
                blockdimH = confpair.GetValue();
            } else if (confpair.GetName() == "BLOCKDIM_X_CALCW") {
                blockdimW = confpair.GetValue();
            } else if (confpair.GetName() == "BLOCKDIM_X_CALC_YPRIME") {
                blockdimCalcYprime = confpair.GetValue();
            }
        }
        // for (int i = 0; i < cols; ++i) {
        //     interface.RunKernel(householder);
        // }

        for (int col = 0; col < cols - 1; col += R) {
            int startc = col;
            int startn = col + R;

            real one = 1;
            real zero = 0;

            // int blockdimW = BLOCKDIM_X_CALCW;
            // while(blockdimW > m - col && blockdimW > 64) blockdimW >>= 1;

            // int blockdimH = BLOCKDIM_X_HOUSE;
            // while(blockdimH > m - col && blockdimH > 64) blockdimH >>= 1;

            for (int i = 0; i < R; ++i) {
                int curcol = col + i;
                if (curcol >= cols - 1) goto decomp_finished;
                // real *curV = Y + POS(0, i, m);

                int curblockdimH = blockdimH;
                while(curblockdimH > rows - col && curblockdimH > 64) curblockdimH >>= 1;

                interface.UpdateScalarArgument(householder_col, &curcol);
                interface.UpdateScalarArgument(householder_startc, &col);
                interface.RunKernel(householder, ktt::DimensionVector(1), ktt::DimensionVector(curblockdimH));
                // householder_calc_beta<<<1, blockdimH>>>(A, m, ld, curcol, curV, col);

                if (R - i - 1 > 0) {
                    int blockdim = blockdimV;
                    while(blockdim > cols - curcol && blockdim > 64) blockdim >>= 1;

                    // calc_and_add_V<<<R-i-1, blockdim>>>(A + POS(curcol, curcol+1, ld), m - curcol, ld, curV + curcol, Wprime);
                    int m = rows - curcol;
                    interface.UpdateScalarArgument(calcV_m, &m);
                    interface.RunKernel(calcV, ktt::DimensionVector(R-i-1), ktt::DimensionVector(blockdim));
                }
            }
            // calc_Yprime<<<dim3(R, R), BLOCKDIM_X_CALC_YPRIME>>>(Y, m, col, R, Wprime);
            interface.UpdateScalarArgument(calcYprime_startc, &col);
            interface.UpdateScalarArgument(calcYprime_R, &R);
            interface.RunKernel(calcYprime, ktt::DimensionVector(R, R), ktt::DimensionVector(blockdimCalcYprime));

            // interface.RunKernel(calcv)
            // copy_W<<<1, BLOCKDIM_X_COPYW>>>(m - col, Y + col, W + col);
            int m = rows - col;
            interface.UpdateScalarArgument(copyW_m, &m);
            interface.RunKernel(copyW);

            // calc_W<<<(m - col + blockdimW-1)/blockdimW, blockdimW>>>(m, col, W, Y, Wprime, R);
            interface.UpdateScalarArgument(calcW_startc, &col);
            interface.UpdateScalarArgument(calcW_R, &R);
            interface.RunKernel(calcW, ktt::DimensionVector((rows - col + blockdimW - 1)/blockdimW), ktt::DimensionVector(blockdimW));

            cublas_gemm(cublasH,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        R, cols - startn, rows - startc,
                        &one,
                        dW, rows,
                        dA, rows,
                        &zero,
                        dWprime, R);

            cublas_gemm(cublasH,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    rows - startc, cols - startn, R,
                    &one,
                    dY, rows,
                    dWprime, R,
                    &one,
                    dA, rows);
        }

        decomp_finished:;
    });

    // tuner.AddParameter(kernel, "BLOCKDIM_X_HOUSE", std::vector<uint64_t>({512, 1024}));
    tuner.AddParameter(kernel, "BLOCKDIM_X_HOUSE", std::vector<uint64_t>({512, 1024}));
    // tuner.AddThreadModifier(kernel, {householder}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCKDIM_X_HOUSE", ktt::ModifierAction::Multiply);

    tuner.AddParameter(kernel, "BLOCKDIM_X_V", std::vector<uint64_t>({256, 512, 1024}));
    // tuner.AddThreadModifier(kernel, {calcV}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCKDIM_X_V", ktt::ModifierAction::Multiply);

    tuner.AddParameter(kernel, "HOUSEHOLDER_BLOCK_SIZE", std::vector<uint64_t>({32, 48, 64, 96, 128}));
    // tuner.AddParameter(kernel, "HOUSEHOLDER_BLOCK_SIZE", std::vector<uint64_t>({32, 48, 64, 96, 128}));
    // tuner.AddParameter(kernel, "HOUSEHOLDER_BLOCK_SIZE", std::vector<uint64_t>({32, 48, 64, 96, 128}));

    tuner.AddParameter(kernel, "BLOCKDIM_X_CALCW", std::vector<uint64_t>({32, 64, 128, 256}));

    tuner.AddParameter(kernel, "BLOCKDIM_X_COPYW", std::vector<uint64_t>({1024}));
    tuner.AddThreadModifier(kernel, {copyW}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCKDIM_X_COPYW", ktt::ModifierAction::Multiply);

    // tuner.AddParameter(kernel, "BLOCKDIM_X_CALC_YPRIME", std::vector<uint64_t>({64, 128, 256}));
    tuner.AddParameter(kernel, "BLOCKDIM_X_CALC_YPRIME", std::vector<uint64_t>({64, 128, 256}));


    auto results = tuner.Tune(kernel);

    ktt::KernelResult best = results[0];
    for (ktt::KernelResult result : results) {
        if (result.GetTotalDuration() < best.GetTotalDuration()) {
            best = result;
        }
    }
    auto bestConf = best.GetConfiguration();
    printf("Best conf with time %lu us:\n%s\n", best.GetTotalDuration()/1000, bestConf.GetString().c_str());

    FILE *paramsFileFp = fopen(paramsFile.c_str(), "w");
    for (auto paramPair : bestConf.GetPairs()) {
        fprintf(paramsFileFp, "#define %s %lu\n", paramPair.GetName().c_str(), paramPair.GetValue());
    }
    fclose(paramsFileFp);


    // char fname[100];
    // sprintf(fname, "tuneresults_%u_%u", rows, cols);
    // printf("Saving to %s.json", fname);
    // tuner.SaveResults(results, fname, ktt::OutputFormat::JSON);

    if (cublasH) cublasDestroy(cublasH);

    cudaFree(dA);
    cudaFree(dTaus);
    cudaFree(dY);
    cudaFree(dW);
    cudaFree(dWprime);

    // tuner.SetLauncher(kernel, [kernel, AT, dA](ktt::ComputeInterface& interface)
    // {
    //     cudaMemcpy(dA, AT.data, AT.rows*AT.cols*sizeof(dA[0]), cudaMemcpyHostToDevice);
    //     // interface.UploadBuffer(aId);
    //     // interface.UploadBuffer(bId);

    //     // interface.RunKernel(kernel);

    //     // interface.ClearBuffer(aId);
    //     // interface.ClearBuffer(bId);
    // });

  // const ktt::DimensionVector blockDimensions(256);
  // const ktt::DimensionVector gridDimensions(numberOfElements / blockDimensions.GetSizeX());


  // const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("QRDecomposition", kernelFile, gridDimensions,
  //       blockDimensions);
}
