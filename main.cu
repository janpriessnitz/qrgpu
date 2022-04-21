#include <cstdio>
#include <ctime>
#include <chrono>

#include "Matrix.h"
#include "CPUValidation.h"
#include "QRInvoke.h"
#include "QRReference.h"

#include "BlockHouseholderKernel.h"

int main(int argc, char** argv) {
    // srand(time(nullptr));
    srand(2);

    int reps = 1;

    int rows = 1024*20;
    int cols = 1024*20;

    auto A = Matrix::GenerateRandom(rows, cols);

    // printf("Input A:\n");
    // A.print();

    // warm up the GPU
    uint64_t cuDuration = 0;
    auto cuSolverX = QRReferenceCuSolver(A, &cuDuration);

    cuDuration = 0;
    for (int i = 0; i < reps; ++i) {
        uint64_t cuDurationCur = 0;
        // QRReferenceCuSolver(A, &cuDuration);
        QRReferenceCuSolver(A, &cuDurationCur);
        cuDuration += cuDurationCur;
    }


    uint64_t myDuration = 0;
    auto QR = InvokeSolve(&A, NULL, cols, &myDuration);
    myDuration = 0;

    for (int i = 0; i < reps; ++i) {
        uint64_t myDurationCur;
        InvokeSolve(&A, NULL, cols, &myDurationCur);
        myDuration += myDurationCur;
    }

    printf("Error %.20lf\n", Matrix::SquareDifference(QR, cuSolverX));
    printf("my time [us]: %ld, cuSolver time [us]: %ld, %fx speedup\n", myDuration, cuDuration, cuDuration/(double)myDuration);

    return 0;
}
