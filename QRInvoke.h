#ifndef QRINVOKE_H
#define QRINVOKE_H

std::pair<Matrix, Matrix> Invoke(const Matrix &A);
Matrix InvokeSolve(Matrix *A, real *taus, int cols, uint64_t *usec_taken);
#endif
