#include <cstdio>
#include <ctime>
#include <chrono>

#include "Matrix.h"
#include "CPUValidation.h"
#include "QRInvoke.h"
#include "QRReference.h"


int main(int argc, char** argv) {
    // srand(time(nullptr));
    srand(2);

    int rows = 8000;
    int cols = 2000;

    auto A = Matrix::GenerateRandom(rows, cols);

    // printf("Input A:\n");
    // A.print();


    uint64_t cuDuration;
    QRReferenceCuSolver(A, &cuDuration);
    QRReferenceCuSolver(A, &cuDuration);

    auto cuSolverX = QRReferenceCuSolver(A, &cuDuration);

    uint64_t myDuration = 0;
    InvokeSolve(&A, NULL, cols, &myDuration);
    // printf("my:\n");
    // A.print();

    // auto solPair = Matrix::DivideHorizontal(Aexpanded, cols);
    // auto myX = QRCPUSolver::SolveTriangular(solPair.first, solPair.second);


    printf("Error %.20lf\n", Matrix::SquareDifference(A, cuSolverX));
    printf("my time [us]: %ld, cuSolver time [us]: %ld, %fx speedup\n", myDuration, cuDuration, cuDuration/(double)myDuration);

    return 0;
}
