#include <cstdio>
#include <ctime>
#include <chrono>

#include "Matrix.h"
#include "CPUValidation.h"
#include "QRInvoke.h"
#include "QRReference.h"
#include "SVDReference.h"


int main(int argc, char** argv) {
    // srand(time(nullptr));
    srand(1);

    int rows = 4000;
    int cols = 4000;
    int nrhs = 1;

    auto A = Matrix::GenerateRandom(rows, cols);
    auto B = Matrix::GenerateRandom(rows, nrhs);

    auto Aexpanded = Matrix::ConcatHorizontal(A, B);
    // Aexpanded.print();

  
    // Aexpanded.print();

    
    uint64_t cuDuration;
    QRReferenceCuSolver(A, B, &cuDuration);
    auto cuSolverX = QRReferenceCuSolver(A, B, &cuDuration);
    // SVDReferenceCuSolver(A, &cuDuration);

      uint64_t myDuration;
    InvokeSolve(&Aexpanded, cols, &myDuration);
    auto solPair = Matrix::DivideHorizontal(Aexpanded, cols);
    auto myX = QRCPUSolver::SolveTriangular(solPair.first, solPair.second);


    printf("Error %.20lf\n", Matrix::SquareDifference(myX, cuSolverX));
    printf("my time [us]: %ld, cuSolver time [us]: %ld, %fx speedup\n", myDuration, cuDuration, cuDuration/(double)myDuration);

    return 0;
}
