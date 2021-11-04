// #include <Ktt.h>
// #include <Matrix.h>

// void QRTuning(pos_t rows, pos_t cols) {

//   ktt::DeviceIndex deviceIndex = 0;
//   std::string kernelFile = kernelPrefix + "QRKernel.cu";

//   ktt::Tuner tuner(0, deviceIndex, ktt::ComputeApi::CUDA);

//   const ktt::DimensionVector blockDimensions(256);
//   const ktt::DimensionVector gridDimensions(numberOfElements / blockDimensions.GetSizeX());


//   const ktt::KernelDefinitionId definition = tuner.AddKernelDefinitionFromFile("QRDecomposition", kernelFile, gridDimensions,
//         blockDimensions);
// }
