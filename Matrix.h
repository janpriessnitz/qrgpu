#ifndef MATRIX_H
#define MATRIX_H

#include <cstdint>
#include <cassert>
#include <cstdio>

typedef double real;
typedef int pos_t;

struct Matrix {
  Matrix(pos_t rows, pos_t cols);
  // rule of five
  Matrix(const Matrix &other);
  Matrix(Matrix &&other) noexcept;
  ~Matrix();
  Matrix& operator=(const Matrix &other);
  Matrix& operator=(Matrix &&other) noexcept;

  void clear();
  void setIdentity();
  void times(real scalar);
  void print(FILE *out = stdout) const;

  Matrix getRow(pos_t row) const;
  Matrix getCol(pos_t col) const;
  Matrix getT() const;

  pos_t pos(pos_t row, pos_t col) const;
  real& operator()(pos_t row, pos_t col);
  real operator()(pos_t row, pos_t col) const;

  pos_t rows;
  pos_t cols;
  real *data;

  static Matrix mul(const Matrix &A, const Matrix &B);

  static Matrix GenerateRandom(pos_t rows, pos_t cols);

  static real SquareDifference(const Matrix &A, const Matrix &B);
};

struct LSSolution {
  Matrix *in;
  Matrix *in_rhs;

  Matrix *sol;
  real err;
};

#endif
