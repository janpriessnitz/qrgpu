#include <cstdio>
#include <ctime>
#include <chrono>

#include "Matrix.h"
#include "CPUValidation.h"
#include "QRInvoke.h"
#include "QRReference.h"

void cuSolver(int rows, int cols, int nrhs) {
    auto A = Matrix::GenerateRandom(rows, cols);
    auto B = Matrix::GenerateRandom(rows, nrhs);
    auto cuSolverX = QRReferenceCuSolver(A, B);
}

int main(int argc, char** argv) {
    // srand(time(nullptr));
    srand(1);

    int rows = 2000;
    int cols = 2000;
    int nrhs = 1;

    auto A = Matrix::GenerateRandom(rows, cols);
    auto B = Matrix::GenerateRandom(rows, nrhs);

    auto Aexpanded = Matrix::ConcatHorizontal(A, B);
    // Aexpanded.print();

    auto myStart = std::chrono::high_resolution_clock::now();
    InvokeSolve(&Aexpanded, cols);
    auto myEnd = std::chrono::high_resolution_clock::now();
    auto myDuration = std::chrono::duration_cast<std::chrono::microseconds>(myEnd - myStart).count();

    // Aexpanded.print();

    auto solPair = Matrix::DivideHorizontal(Aexpanded, cols);

    auto myX = QRCPUSolver::SolveTriangular(solPair.first, solPair.second);

    auto cuStart = std::chrono::high_resolution_clock::now();
    auto cuSolverX = QRReferenceCuSolver(A, B);

    auto cuEnd = std::chrono::high_resolution_clock::now();
    auto cuDuration = std::chrono::duration_cast<std::chrono::microseconds>(cuEnd - cuStart).count();

    printf("Error %.20lf\n", Matrix::SquareDifference(myX, cuSolverX));
    printf("my time [us]: %ld, cuSolver time [us]: %ld, %fx speedup\n", myDuration, cuDuration, cuDuration/(double)myDuration);

    return 0;
}
