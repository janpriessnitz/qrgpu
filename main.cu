#include <cstdio>
#include <ctime>

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
    srand(time(nullptr));

    int rows = 1000;
    int cols = 900;
    int nrhs = 10;

    auto A = Matrix::GenerateRandom(rows, cols);
    auto B = Matrix::GenerateRandom(rows, nrhs);

    auto Aexpanded = Matrix::GenerateRandom(4, 4);
    Aexpanded.print();

    InvokeSolve(&Aexpanded, 3);

    Aexpanded.print();

    // Matrix A(4, 3);
    // A(0, 0) = 12;
    // A(0, 1) = -51;
    // A(0, 2) = 4;
    // A(1, 0) = 6;
    // A(1, 1) = 167;
    // A(1, 2) = -68;
    // A(2, 0) = -4;
    // A(2, 1) = 24;
    // A(2, 2) = -41;
    // A(3, 0) = 1;
    // A(3, 1) = 1;
    // A(3, 2) = 1;

    // Matrix rhs(4, 1);
    // rhs(0, 0) = 1;
    // rhs(1, 0) = 2;
    // rhs(2, 0) = 3;
    // rhs(3, 0) = 4;

    // auto solPair = Invoke(A);
    // printf("A\n");
    // A.print();
    // printf("GPU:\n");
    // printf("R\n");
    // solPair.second.print();
    // printf("Q\n");
    // solPair.first.print();
    // auto sol = QRCPUSolver::SolveHouseholder(A, rhs);
    // sol.print();

    // printf("CPU validation:\n");
    // auto CPUSol = QRCPUSolver::DecompHouseholder(A);

    // printf("Error %.20lf\n", Matrix::SquareDifference(solPair.second, CPUSol.second));

    // auto myX = QRCPUSolver::SolveHouseholder(A, B);
    // myX.print();

    // auto cuSolverX = QRReferenceCuSolver(A, B);
    // cuSolverX.print();
    // printf("Error %.20lf\n", Matrix::SquareDifference(myX, cuSolverX));
    // cuSolver(40000, 20000, 10);
    return 0;
}
