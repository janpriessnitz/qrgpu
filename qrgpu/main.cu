#include <cstdio>
#include <ctime>
#include <chrono>

#include "Matrix.h"
#include "CPUValidation.h"
#include "QRInvoke.h"
#include "QRReference.h"

#include "QRTuning.h"

#include "BlockHouseholderKernel.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Need 2 arguments - matrix rows, cols\n");
        return 1;
    }
    int rows = strtol(argv[1], nullptr, 10);
    int cols = strtol(argv[2], nullptr, 10);
    if (rows <= 0) {
        fprintf(stderr, "Invalid first argument\n");
        return 1;
    }
    if (cols <= 0) {
        fprintf(stderr, "Invalid second argument\n");
        return 1;
    }

    if (rows < cols) {
        fprintf(stderr, "Invalid matrix size - number of rows must be greater or equal than number of cols\n");
        return 1;
    }


    // srand(time(nullptr));
    srand(2);

    int reps = 10;

    auto A = Matrix::GenerateRandom(rows, cols);

    // printf("Input A:\n");
    // A.print();

    // warm up the GPU
    uint64_t cuDuration = 0;
    auto cuSolverX = QRReferenceCuSolver(A, &cuDuration);

    // printf("cuSolver:\n");
    // cuSolverX.print();

    cuDuration = 0;
    for (int i = 0; i < reps; ++i) {
        uint64_t cuDurationCur = 0;
        // QRReferenceCuSolver(A, &cuDuration);
        QRReferenceCuSolver(A, &cuDurationCur);
        cuDuration += cuDurationCur;
        printf("%lu ", cuDurationCur);
    }
    printf("\n");


    uint64_t myDuration = 0;
    auto QR = InvokeSolve(&A, NULL, cols, &myDuration);
    myDuration = 0;

    // printf("my:\n");
    // QR.print();

    for (int i = 0; i < reps; ++i) {
        uint64_t myDurationCur;
        InvokeSolve(&A, NULL, cols, &myDurationCur);
        myDuration += myDurationCur;
        printf("%lu ", myDurationCur);
    }
    printf("\n");

    printf("Error %.20lf\n", Matrix::SquareDifference(QR, cuSolverX));
    printf("my time [us]: %ld, cuSolver time [us]: %ld, %fx speedup\n", myDuration, cuDuration, cuDuration/(double)myDuration);

    return 0;
}
