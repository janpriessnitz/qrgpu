#include "CPUValidation.h"

#include <cmath>
#include <cstdio>
#include <algorithm>

LSSolution *QRCPUSolver::SolveGivens(const Matrix &in, const Matrix &rhs) {
  auto decomp_pair = DecompGivens(in);
  return nullptr;
}

Matrix QRCPUSolver::SolveHouseholder(const Matrix &in, const Matrix &rhs) {
  auto decomp_pair = DecompHouseholder(in);
  auto Q = decomp_pair.first;
  auto R = decomp_pair.second;

  auto d = Matrix::mul(Q.getT(), rhs);
  auto x = SolveTriangular(R, d);
  return x;
}

std::pair<Matrix, Matrix> QRCPUSolver::DecompGivens(const Matrix &in) {
  assert(in.rows >= in.cols);

  // Matrix *Q = new Matrix();
  // Matrix *R = new Matrix();

  return std::pair<Matrix, Matrix>(Matrix(0,0), Matrix(0,0));
}

// https://en.wikipedia.org/wiki/QR_decomposition#Using_Householder_reflections
std::pair<Matrix, Matrix> QRCPUSolver::DecompHouseholder(const Matrix &A) {
  assert(A.rows >= A.cols);

  Matrix Q(A.rows, A.rows);
  Q.setIdentity();
  Matrix R(A);

  for (pos_t cur_col = 0; cur_col < std::min(A.cols, A.rows-1); ++cur_col) {
    auto x = R.getCol(cur_col);
    for (pos_t i = 0; i < cur_col; ++i) {
      x(i, 0) = 0;
    }
    auto mag_x = Matrix::mul(x.getT(), x);
    real alpha = sqrt(mag_x(0, 0));

    x(cur_col, 0) -= alpha;

    mag_x = Matrix::mul(x.getT(), x);
    // printf("beta no. %u: %lf\n", cur_col, -2/mag_x(0, 0));

    auto Qcur = Matrix::mul(x, x.getT());
    Qcur.times(-2/mag_x(0, 0));
    for (pos_t i = 0; i < Qcur.cols; ++i) {
      Qcur(i, i) += 1;
    }

    Q = Matrix::mul(Qcur, Q);
    R = Matrix::mul(Qcur, R);
    // printf("iteration %d\n", cur_col);
    // printf("R:\n");
    // R.print();
    // printf("Q:\n");
    // Q.print();
  }

  return std::pair<Matrix, Matrix>(Q.getT(), R);
}

Matrix QRCPUSolver::SolveTriangular(const Matrix &A, const Matrix &rhs) {
  Matrix B(rhs);
  Matrix x(A.cols, B.cols);

  for (int b_r = 0; b_r < B.cols; ++b_r) {
    for (int c = A.cols - 1; c >= 0; --c) {
      x(c, b_r) = B(c, b_r)/A(c, c);
      for (int cx = c-1; cx >= 0; --cx) {
        B(cx, b_r) -= B(c, b_r)*A(cx, c)/A(c, c);
      }
    }
  }

  return x;
}
