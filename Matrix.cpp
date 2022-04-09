#include "Matrix.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <utility>

Matrix::Matrix()
  : data(nullptr)
{}

Matrix::Matrix(pos_t rows, pos_t cols)
  : rows(rows)
  , cols(cols)
{
  data = new real[rows*cols];
}

Matrix::Matrix(const Matrix &other) {
  rows = other.rows;
  cols = other.cols;
  data = new real[rows*cols];
  memcpy(data, other.data, sizeof(real)*rows*cols);
}

Matrix::Matrix(Matrix &&other) noexcept {
  rows = other.rows;
  cols = other.cols;
  data = nullptr;
  std::swap(data, other.data);
}

Matrix& Matrix::operator=(const Matrix &other) {
  return *this = Matrix(other);
}

Matrix& Matrix::operator=(Matrix &&other) noexcept {
  std::swap(data, other.data);
  return *this;
}

Matrix::~Matrix() {
  delete[] data;
}

void Matrix::clear() {
  for (pos_t i = 0; i < rows*cols; ++i) {
    data[i] = 0;
  }
}

void Matrix::setIdentity() {
  assert(rows == cols);
  clear();
  for (pos_t i = 0; i < rows; ++i) {
    (*this)(i, i) = 1;
  }
}

pos_t Matrix::pos(pos_t row, pos_t col) const {
  return row*cols + col;
}

Matrix Matrix::getRow(pos_t row) const {
  assert(row >= 0 && row < rows);
  Matrix out(1, cols);
  for (pos_t c = 0; c < cols; ++c) {
    out(0, c) = (*this)(row, c);
  }
  return out;
}

Matrix Matrix::getCol(pos_t col) const {
  assert(col >= 0 && col < cols);
  Matrix out(rows, 1);
  for (pos_t r = 0; r < rows; ++r) {
    out(r, 0) = (*this)(r, col);
  }
  return out;
}

Matrix Matrix::getT() const {
  Matrix out(cols, rows);
  for (pos_t r = 0; r < rows; ++r) {
    for (pos_t c = 0; c < cols; ++c) {
      out(c, r) = (*this)(r, c);
    }
  }
  return out;
}



real& Matrix::operator()(pos_t row, pos_t col) {
  assert(row >= 0 && row < rows);
  assert(col >= 0 && col < cols);

  return data[pos(row, col)];
}

real Matrix::operator()(pos_t row, pos_t col) const {
  assert(row >= 0 && row < rows);
  assert(col >= 0 && col < cols);

  return data[pos(row, col)];
}

Matrix Matrix::mul(const Matrix &A, const Matrix &B) {
  assert(A.cols == B.rows);
  Matrix C(A.rows, B.cols);
  C.clear();
  for (pos_t cur_row = 0; cur_row < C.rows; ++cur_row) {
    for (pos_t cur_col = 0; cur_col < C.cols; ++cur_col) {
      for (pos_t i = 0; i < A.cols; ++i) {
        C(cur_row, cur_col) += A(cur_row, i)*B(i, cur_col);
      }
    }
  }

  return C;
}

void Matrix::times(real scalar) {
  for (pos_t r = 0; r < rows; ++r) {
    for (pos_t c = 0; c < cols; ++c) {
      (*this)(r, c) *= scalar;
    }
  }
}

void Matrix::print(FILE *out) const {
  assert(out != nullptr);
  for (pos_t r = 0; r < rows; ++r) {
    for (pos_t c = 0; c < cols; ++c) {
      fprintf(out, "%.4lf\t", (*this)(r, c));
    }
    fprintf(out, "\n");
  }
}

Matrix Matrix::GenerateRandom(pos_t rows, pos_t cols) {
  int MAX_NUM = 10;
  int MIN_NUM = 1;
  Matrix out(rows, cols);
  for (pos_t r = 0; r < rows; ++r) {
    for (pos_t c = 0; c < cols; ++c) {
      out(r, c) = (rand() % (MAX_NUM-MIN_NUM)) + MIN_NUM;
    }
  }
  return out;
}

real Matrix::SquareDifference(const Matrix &A, const Matrix &B) {
  assert(A.rows == B.rows);
  assert(A.cols == B.cols);

  real diff = 0;

  for (pos_t r = 0; r < A.rows; ++r) {
    for (pos_t c = 0; c < A.cols; ++c) {
      diff += (A(r, c) - B(r, c))*(A(r, c) - B(r, c));
    }
  }

  return diff;
}

Matrix Matrix::ConcatHorizontal(const Matrix &A, const Matrix &B) {
  assert(A.rows == B.rows);

  Matrix out(A.rows, A.cols + B.cols);

  for (pos_t r = 0; r < A.rows; ++r) {
    for (pos_t c = 0; c < A.cols; ++c) {
      out(r, c) = A(r, c);
    }
    for (pos_t c = 0; c < B.cols; ++c) {
      out(r, c + A.cols) = B(r, c);
    }
  }
  return out;
}

std::pair<Matrix, Matrix> Matrix::DivideHorizontal(const Matrix &in, pos_t col) {
  assert(col < in.cols);
  assert (col >= 0);
  pos_t rows = in.rows;

  Matrix A(rows, col);
  Matrix B(rows, in.cols - col);

  for (pos_t r = 0; r < rows; ++r) {
    for (pos_t c = 0; c < col; ++c) {
      A(r, c) = in(r, c);
    }
    for (pos_t c = 0; c < in.cols - col; ++c) {
      B(r, c) = in(r, c + col);
    }
  }
  return std::pair<Matrix, Matrix>(A, B);
}

