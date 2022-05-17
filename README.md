Bachelor thesis - CUDA GPU implementation of QR factorization

The entry function for the QR factorization code is `qrgpu/BlockHouseholderKernel.h:QRBlockSolve`.

The Makefile in `qrgpu` directory contains two targets: `main` target is used for experimental evaluation against cuSolver,
`tune` target is used for autotuning.
Users have to specify path to several dependencies in the Makefile: Kernel Tuning Toolkit, nvcc compiler and CUDA toolkit.

The `tune.sh` script is used to perform autotuning and then evaluation for different input matrix sizes.
The measurement results can be later processed with `process_results.py` script, which calculates the mean and error of speedup against cuSolver.
