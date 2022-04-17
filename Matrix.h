#ifndef MATRIX_H
#define MATRIX_H

#include <cstdint>
#include <cassert>
#include <cstdio>

#include <utility>

#define DOUBLE_PRECISION 0

#if DOUBLE_PRECISION
#define cublas_gemm cublasDgemm
#define cusolverDn_geqrf_bufferSize cusolverDnDgeqrf_bufferSize
#define cusolverDn_ormqr_bufferSize cusolverDnDormqr_bufferSize
#define cusolverDn_geqrf cusolverDnDgeqrf
#define cusolverDn_ormqr cusolverDnDormqr
#define cublas_trsm cublasDtrsm
#define cublas_gemv cublasDgemv
#define cublas_syrk cublasDsyrk
typedef double real;
#else
#define cublas_gemm cublasSgemm
#define cusolverDn_geqrf_bufferSize cusolverDnSgeqrf_bufferSize
#define cusolverDn_ormqr_bufferSize cusolverDnSormqr_bufferSize
#define cusolverDn_geqrf cusolverDnSgeqrf
#define cusolverDn_ormqr cusolverDnSormqr
#define cublas_trsm cublasStrsm
#define cublas_gemv cublasSgemv
#define cublas_syrk cublasSsyrk
typedef float real;
#endif

typedef int pos_t;

struct Matrix {
  Matrix();
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

  static Matrix ConcatHorizontal(const Matrix &A, const Matrix &B);
  static std::pair<Matrix, Matrix> DivideHorizontal(const Matrix &A, pos_t col);
};

struct LSSolution {
  Matrix *in;
  Matrix *in_rhs;

  Matrix *sol;
  real err;
};

#endif
