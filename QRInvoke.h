#ifndef QRINVOKE_H
#define QRINVOKE_H

std::pair<Matrix, Matrix> Invoke(const Matrix &A);
void InvokeSolve(Matrix *A, int cols, uint64_t *usec_taken);
#endif
