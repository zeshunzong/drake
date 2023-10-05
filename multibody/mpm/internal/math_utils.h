#pragma once

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

constexpr double kEps = 1e-6;

// See https://en.wikipedia.org/wiki/Levi-Civita_symbol for details
// Return (i, j, k)th entry of the third order permutation tensor
// @pre i, j, k ∈ {0, 1, 2}
inline double LeviCivita(int i, int j, int k) {
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
// Computes A:ε
// TODO(@zeshunzong): Consider unroll the loop and maybe also remove
// multiplication
template <typename T>
Vector3<T> ContractionWithLeviCivita(const Matrix3<T>& A) {
  Vector3<T> A_dot_eps = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k) {
    for (int j = 0; j < 3; ++j) {
      for (int i = 0; i < 3; ++i) {
        A_dot_eps(k) += A(i, j) * LeviCivita(i, j, k);
      }
    }
  }
  return A_dot_eps;
}

/**
 * Prevents x from getting more than eps-close to zero.See accompanied
 * math_utils.md. Its derivative is 0 when x ∈ (−ε,  ε).
 * @pre eps > 0
 */
template <typename T>
T ClampToEpsilon(T x, T eps) {
  DRAKE_ASSERT(eps >= 0);
  using std::abs;  // ADL overload to AutoDiff type
  if (abs(x) >= eps) {
    return x;
  } else if (x >= 0) {
    if constexpr (std::is_same_v<T, AutoDiffXd>) {
      return AutoDiffXd(eps.value(), 1, 1);

    } else {
      return eps;
    }
  } else {
    if constexpr (std::is_same_v<T, AutoDiffXd>) {
      return AutoDiffXd(-eps.value(), 1, 1);
    } else {
      return -eps;
    }
  }
}

/**
   Robustly computes log(x+1)/x based on Taylor expansion.
   See accompanied math_utils.md.
   @pre x > -1
 */
template <typename T>
T CalcLogXPlus1OverX(const T& x) {
  DRAKE_ASSERT(x > -1);
  using std::abs;  // ADL overload to AutoDiff type
  if (abs(x) < kEps) {
    T x_squared = x * x;
    return 1 - x / (2) + x_squared / (3) - x_squared * x / (4);
  } else {
    // return std::log1p(x) / x;
    using std::log;  // ADL overload to AutoDiff type, std::log1p not supperted
    return log(1 + x) / x;
  }
}

/**
   Robustly computes (logx-logy)/(x-y).
   Approximation via Taylor expansion when |x/y-1|<kEps.
   See accompanied math_utils.md
   @pre x > 0
   @pre y > 0
   @pre x != y
 */
template <typename T>
T CalcLogXMinusLogYOverXMinusY(const T& x, const T& y) {
  DRAKE_ASSERT(x > 0);
  DRAKE_ASSERT(y > 0);
  DRAKE_ASSERT(x != y);
  T p = x / y - 1;
  return CalcLogXPlus1OverX(p) / y;
}

/**
   Robustly computes (x logy - y logx)/(x-y)
   See accompanied math_utils.md
   @pre x > 0
   @pre y > 0
   @pre x != y
 */
template <typename T>
T CalcXLogYMinusYLogXOverXMinusY(const T& x, const T& y) {
  DRAKE_ASSERT(x > 0);
  DRAKE_ASSERT(y > 0);
  DRAKE_ASSERT(x != y);
  using std::abs;  // ADL overload to AutoDiff type
  return log(y) - y * CalcLogXMinusLogYOverXMinusY(x, y);
}

/**
   Robustly computes (expx-1)/x based on Taylor expansion.
   See accompanied math_utils.md
 */
template <typename T>
T CalcExpXMinus1OverX(const T& x) {
  using std::abs;
  if (abs(x) < kEps) {
    T x_squared = x * x;
    return 1 + x / (2) + x_squared / (6) + x_squared * x / (24);
  } else {
    using std::exp;
    // return std::expm1(x) / x;
    return (exp(x) - 1) / x;
  }  // ADL overload to AutoDiff type, std::expm1 not supperted
}

/**
   Robustly computes (expx-expy)/(x-y).
   Approximation via Taylor expansion when |x-y|<kEps.
   See accompanied math_utils.md
 */
template <typename T>
T CalcExpXMinusExpYOverXMinusY(const T& x, const T& y) {
  T p = x - y;
  using std::exp;
  return CalcExpXMinus1OverX(p) * exp(y);
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
