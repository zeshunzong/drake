#pragma once

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

// See https://en.wikipedia.org/wiki/Levi-Civita_symbol for details
// Return (i, j, k)th entry of the third order permutation tensor
template <typename T>
T LeviCivita(int i, int j, int k) {
  // Even permutation
  if ((i == 0 && j == 1 && k == 2) || (i == 1 && j == 2 && k == 0) ||
      (i == 2 && j == 0 && k == 1)) {
    return 1.0;
  }
  // Odd permutation
  if ((i == 2 && j == 1 && k == 0) || (i == 0 && j == 2 && k == 1) ||
      (i == 1 && j == 0 && k == 2)) {
    return -1.0;
  }
  return 0.0;
}

// See https://en.wikipedia.org/wiki/Levi-Civita_symbol for details
// Calculate A:ε
template <typename T>
Vector3<T> ContractionWithLeviCivita(const Matrix3<T>& A) {
  Vector3<T> A_dot_eps = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k) {
    for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 3; ++i) {
        A_dot_eps(k) += A(i, j) * LeviCivita<T>(i, j, k);
      }
    }
  }
  return A_dot_eps;
}

/**
   Prevent x from getting more than eps-close to zero
   See accompanied math_utils.md
 */
template <typename T>
T ClampToEpsilon(const T& x, const T& eps) {
  DRAKE_ASSERT(eps >= 0);
  if (x < -eps) {
    return x;
  } else if (x < 0) {
    return -eps;
  } else if (x < eps) {
    return eps;
  } else {
    return x;
  }
}

/**
   Robustly computing log(x+1)/x
   See accompanied math_utils.md
 */
template <typename T>
T CalcLogXPlus1OverX(const T& x, const T& eps) {
  DRAKE_ASSERT(eps > 0);
  using std::abs;  // ADL overload to AutoDiff type
  if (abs(x) < eps) {
    return 1 - x / (2) + x * x / (3) - x * x * x / (4);
  } else {
    // return std::log1p(x) / x;
    using std::log;  // ADL overload to AutoDiff type, std::log1p not supperted
    return log(1 + x) / x;
  }
}

/**
   Robustly computing (logx-logy)/(x-y)
   See accompanied math_utils.md
 */
template <typename T>
T CalcLogXMinusLogYOverXMinusY(const T& x, const T& y, const T& eps) {
  DRAKE_ASSERT(eps > 0);
  T p = x / y - 1;
  return CalcLogXPlus1OverX(p, eps) / y;
}

/**
   Robustly computing (x logy - y logx)/(x-y)
   See accompanied math_utils.md
 */
template <typename T>
T CalcXLogYMinusYLogXOverXMinusY(const T& x, const T& y, const T& eps) {
  DRAKE_ASSERT(eps > 0);
  using std::abs;  // ADL overload to AutoDiff type
  return log(y) - y * CalcLogXMinusLogYOverXMinusY(x, y, eps);
}

/**
   Robustly computing (expx-1)/x
   See accompanied math_utils.md
 */
template <typename T>
T CalcExpXMinus1OverX(const T& x, const T& eps) {
  DRAKE_ASSERT(eps > 0);
  using std::abs;
  if (abs(x) < eps) {
    return 1 + x / (2) + x * x / (6) + x * x * x / (24);
  } else {
    using std::exp;
    // return std::expm1(x) / x;
    return (exp(x) - 1) / x;
  }  // ADL overload to AutoDiff type, std::expm1 not supperted
}

/**
   Robustly computing (expx-expy)/(x-y)
   See accompanied math_utils.md
 */
template <typename T>
T CalcExpXMinusExpYOverXMinusY(const T& x, const T& y, const T& eps) {
  DRAKE_ASSERT(eps > 0);
  T p = x - y;
  using std::exp;
  return CalcExpXMinus1OverX(p, eps) * exp(y);
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
