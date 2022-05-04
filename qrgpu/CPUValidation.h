#ifndef CPUVALIDATION_H
#define CPUVALIDATION_H

#include <cassert>
#include <utility>

#include "Matrix.h"

class QRCPUSolver {
public:
  static LSSolution *SolveGivens(const Matrix &in, const Matrix &rhs);
  static Matrix SolveHouseholder(const Matrix &in, const Matrix &rhs);
  static std::pair<Matrix, Matrix> DecompGivens(const Matrix &in);
  static std::pair<Matrix, Matrix> DecompMGS(const Matrix &in);
  static std::pair<Matrix, Matrix> DecompHouseholder(const Matrix &A);

  static Matrix SolveTriangular(const Matrix &A, const Matrix &rhs);
};

#endif
