#pragma once

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/fem/matrix_utilities.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

constexpr double kEps = 1e-6;

// See https://en.wikipedia.org/wiki/Levi-Civita_symbol for details.
// Returns (i, j, k)th entry of the third order permutation tensor.
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

// See https://en.wikipedia.org/wiki/Levi-Civita_symbol for details.
// Computes A:ε
// TODO(@zeshunzong): Consider unroll the loop and maybe also remove
// multiplication.
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

// Prevents x from getting more than eps-close to zero. See
// accompaniedmath_utils.md. Its derivative is 0 when x ∈ (−ε, ε).
// @pre eps > 0
template <typename T>
T ClampToEpsilon(const T& x, double eps) {
  DRAKE_ASSERT(eps >= 0);
  using std::abs;  // ADL overload to AutoDiff type
  if (abs(x) >= eps) {
    return x;
  } else if (x >= 0) {
    if constexpr (std::is_same_v<T, AutoDiffXd>) {
      return AutoDiffXd(eps, VectorX<double>::Zero(x.derivatives().size()));

    } else {
      return eps;
    }
  } else {
    if constexpr (std::is_same_v<T, AutoDiffXd>) {
      return AutoDiffXd(-eps, VectorX<double>::Zero(x.derivatives().size()));
    } else {
      return -eps;
    }
  }
}

// Robustly computes log(x+1)/x based on Taylor expansion. See accompanied
// math_utils.md.
// @pre x > -1
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

// Robustly computes (logx-logy)/(x-y). Approximation via Taylor expansion when
// |x/y-1|<kEps. See accompanied math_utils.md.
// @pre x > 0
// @pre y > 0
// @pre x != y
template <typename T>
T CalcLogXMinusLogYOverXMinusY(const T& x, const T& y) {
  DRAKE_ASSERT(x > 0);
  DRAKE_ASSERT(y > 0);
  DRAKE_ASSERT(x != y);
  T p = x / y - 1;
  return CalcLogXPlus1OverX(p) / y;
}

// Robustly computes (x logy - y logx)/(x-y). See accompanied math_utils.md.
// @pre x > 0
// @pre y > 0
// @pre x != y
template <typename T>
T CalcXLogYMinusYLogXOverXMinusY(const T& x, const T& y) {
  DRAKE_ASSERT(x > 0);
  DRAKE_ASSERT(y > 0);
  DRAKE_ASSERT(x != y);
  using std::abs;  // ADL overload to AutoDiff type
  return log(y) - y * CalcLogXMinusLogYOverXMinusY(x, y);
}

// Robustly computes (expx-1)/x based on Taylor expansion. See accompanied
// math_utils.md.
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

// Robustly computes (expx-expy)/(x-y). Approximation via Taylor expansion when
// |x-y|<kEps. See accompanied math_utils.md.
template <typename T>
T CalcExpXMinusExpYOverXMinusY(const T& x, const T& y) {
  T p = x - y;
  using std::exp;
  return CalcExpXMinus1OverX(p) * exp(y);
}

// Returns true if index_1 < index_2
inline bool CompareIndex3DLexicographically(const Vector3<int>& index_1,
                                            const Vector3<int>& index_2) {
  if (index_1(0) != index_2(0)) return index_1(0) < index_2(0);
  if (index_1(1) != index_2(1)) return index_1(1) < index_2(1);
  return index_1(2) < index_2(2);
}

// Computes the 3d position of a 3d index.
// The position for node indexed by (i, j, k) is (ih, jh, kh) by construction.
inline Vector3<double> ComputePositionFromIndex3D(const Vector3<int>& index,
                                                  double h) {
  return Vector3<double>{index(0) * h, index(1) * h, index(2) * h};
}

// Computes the base node (i.e., the closet h-integral node) of a given
// position.
template <typename T>
Vector3<int> ComputeBaseNodeFromPosition(const Vector3<T>& position, double h) {
  using std::round;
  return Vector3<int>{static_cast<int>(round(position(0) / h)),
                      static_cast<int>(round(position(1) / h)),
                      static_cast<int>(round(position(2) / h))};
}

// project the matrix so that eigenvalues >= 0
template <typename T>
void MakePD(Matrix3<T>* symmetric_matrix) {
  Eigen::SelfAdjointEigenSolver<Matrix3<T>> eigenSolver(*symmetric_matrix);
  if (eigenSolver.eigenvalues()[0] >= 0.0) {
    return;
  }
  Eigen::DiagonalMatrix<T, 3> D(eigenSolver.eigenvalues());
  for (int i = 0; i < 3; ++i) {
    if (D.diagonal()[i] < 0.0) {
      D.diagonal()[i] = 0.0;
    } else {
      break;
    }
  }
  (*symmetric_matrix) =
      eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
}

template <typename T>
void MakePD2D(Matrix2<T>* symmetric_matrix) {
  using std::abs;
  using std::sqrt;
  Matrix2<T>& symmetric_matrix_ref = *symmetric_matrix;

  const T b = (symmetric_matrix_ref(0, 1) + symmetric_matrix_ref(1, 0)) / 2.0;
  T b2 = b * b;
  if (b2 == 0) {
    for (int d = 0; d < 2; ++d) {
      if (symmetric_matrix_ref(d, d) < 0) {
        symmetric_matrix_ref(d, d) = 0;
      }
      // symmetric_matrix_ref(d, d) = std::max(0.0, symmetric_matrix_ref(d, d));
    }
    return;
  }

  const T a = symmetric_matrix_ref(0, 0);
  const T d = symmetric_matrix_ref(1, 1);
  const T D = a * d - b2;
  const T T_div_2 = (a + d) / 2.0;
  const T sqrtTT4D = sqrt(abs(T_div_2 * T_div_2 - D));
  const T L2 = T_div_2 - sqrtTT4D;  // smallest eigenvalue
  if (L2 < 0.0) {
    const T L1 = T_div_2 + sqrtTT4D;  // largest eigenvalue
    if (L1 <= 0.0) {
      symmetric_matrix_ref.setZero();
    } else {
      const T L1md = L1 - d;
      const T L1md2 = L1md * L1md;
      const T denom = L1md2 + b2;
      symmetric_matrix_ref(0, 0) = L1 * L1md2 / denom;
      symmetric_matrix_ref(1, 1) = L1 * b2 / denom;
      symmetric_matrix_ref(0, 1) = symmetric_matrix_ref(1, 0) =
          L1 * b * L1md / denom;
    }
  }
}

// F = Q*S, where Q is rotation and S is symmetric
// Solve for B = 0.5 * (R^T F + F^T R), keeping Q fixed
// @pre B is symmetric
template <typename T>
void SolveForNewF(const Matrix3<T>& B, const Matrix3<T>& R, Matrix3<T>* F) {
  Matrix3<T> Q, S;
  fem::internal::PolarDecompose<T>(*F, &Q, &S);
  Eigen::Matrix<T, 6, 6> C;
  Eigen::Vector<T, 6> rhs;
  rhs(0) = B(0, 0);
  C(0, 0) = Q(0, 0) * R(0, 0) + Q(1, 0) * R(1, 0) + Q(2, 0) * R(2, 0);  // S00
  C(0, 1) = Q(0, 1) * R(0, 0) + Q(1, 1) * R(1, 0) + Q(2, 1) * R(2, 0);  // S01
  C(0, 2) = Q(0, 2) * R(0, 0) + Q(1, 2) * R(1, 0) + Q(2, 2) * R(2, 0);  // S02
  C(0, 3) = 0.0;
  C(0, 4) = 0.0;
  C(0, 5) = 0.0;

  rhs(1) = B(0, 1);
  C(1, 0) =
      0.5 * (Q(0, 0) * R(0, 1) + Q(1, 0) * R(1, 1) + Q(2, 0) * R(2, 1));  // S00
  C(1, 1) =
      0.5 * (Q(0, 0) * R(0, 0) + Q(0, 1) * R(0, 1) + Q(2, 0) * R(2, 0) +
             Q(1, 1) * R(1, 1) + Q(1, 0) * R(1, 0) + Q(2, 1) * R(2, 1));  // S01
  C(1, 2) =
      0.5 * (Q(0, 2) * R(0, 1) + Q(1, 2) * R(1, 1) + Q(2, 2) * R(2, 1));  // S02
  C(1, 3) =
      0.5 * (Q(0, 1) * R(0, 0) + Q(1, 1) * R(1, 0) + Q(2, 1) * R(2, 0));  // S11
  C(1, 4) =
      0.5 * (Q(0, 2) * R(0, 0) + Q(1, 2) * R(1, 0) + Q(2, 2) * R(2, 0));  // S12
  C(1, 5) = 0.0;                                                          // S22

  rhs(2) = B(0, 2);
  C(2, 0) =
      0.5 * (Q(0, 0) * R(0, 2) + Q(1, 0) * R(1, 2) + Q(2, 0) * R(2, 2));  // S00
  C(2, 1) =
      0.5 * (Q(0, 1) * R(0, 2) + Q(1, 1) * R(1, 2) + Q(2, 1) * R(2, 2));  // S01

  C(2, 2) =
      0.5 * (Q(0, 0) * R(0, 0) + Q(0, 2) * R(0, 2) + Q(1, 0) * R(1, 0) +
             Q(1, 2) * R(1, 2) + Q(2, 0) * R(2, 0) + Q(2, 2) * R(2, 2));  // S02
  C(2, 3) = 0.0;                                                          // S11
  C(2, 4) =
      0.5 * (Q(0, 1) * R(0, 0) + Q(1, 1) * R(1, 0) + Q(2, 1) * R(2, 0));  // S12
  C(2, 5) =
      0.5 * (Q(0, 2) * R(0, 0) + Q(1, 2) * R(1, 0) + Q(2, 2) * R(2, 0));  // S22

  rhs(3) = B(1, 1);
  C(3, 0) = 0.0;                                                        // S00
  C(3, 1) = Q(0, 0) * R(0, 1) + Q(1, 0) * R(1, 1) + Q(2, 0) * R(2, 1);  // S01
  C(3, 2) = 0.0;                                                        // S02
  C(3, 3) = Q(0, 1) * R(0, 1) + Q(1, 1) * R(1, 1) + Q(2, 1) * R(2, 1);  // S11
  C(3, 4) = Q(0, 2) * R(0, 1) + Q(1, 2) * R(1, 1) + Q(2, 2) * R(2, 1);  // S12
  C(3, 5) = 0.0;                                                        // S22

  rhs(4) = B(1, 2);
  C(4, 0) = 0.0;  // S00
  C(4, 1) =
      0.5 * (Q(0, 0) * R(0, 2) + Q(1, 0) * R(1, 2) + Q(2, 0) * R(2, 2));  // S01
  C(4, 2) =
      0.5 * (Q(0, 0) * R(0, 1) + Q(1, 0) * R(1, 1) + Q(2, 0) * R(2, 1));  // S02
  C(4, 3) =
      0.5 * (Q(0, 1) * R(0, 2) + Q(1, 1) * R(1, 2) + Q(2, 1) * R(2, 2));  // S11
  C(4, 4) =
      0.5 * (Q(0, 1) * R(0, 1) + Q(0, 2) * R(0, 2) + Q(1, 1) * R(1, 1) +
             Q(1, 2) * R(1, 2) + Q(2, 1) * R(2, 1) + Q(2, 2) * R(2, 2));  // S12
  C(4, 5) =
      0.5 * (Q(0, 2) * R(0, 1) + Q(1, 2) * R(1, 1) + Q(2, 2) * R(2, 1));  // S22

  rhs(5) = B(2, 2);
  C(5, 0) = 0.0;                                                        // S00
  C(5, 1) = 0.0;                                                        // S01
  C(5, 2) = Q(0, 0) * R(0, 2) + Q(1, 0) * R(1, 2) + Q(2, 0) * R(2, 2);  // S02
  C(5, 3) = 0.0;                                                        // S11
  C(5, 4) = Q(0, 1) * R(0, 2) + Q(1, 1) * R(1, 2) + Q(2, 1) * R(2, 2);  // S12
  C(5, 5) = Q(0, 2) * R(0, 2) + Q(1, 2) * R(1, 2) + Q(2, 2) * R(2, 2);  // S22

  Eigen::ColPivHouseholderQR<Eigen::Matrix<T, 6, 6>> decomp(C);
  Eigen::Vector<T, 6> y = decomp.solve(rhs);
  DRAKE_ASSERT(&y != nullptr);
  S(0, 0) = y(0);
  S(0, 1) = y(1);
  S(0, 2) = y(2);
  S(1, 0) = S(0, 1);
  S(1, 1) = y(3);
  S(1, 2) = y(4);
  S(2, 0) = S(0, 2);
  S(2, 1) = S(1, 2);
  S(2, 2) = y(5);

  (*F) = (Q * S);
}

// F = Q*S, where Q is rotation and S is symmetric
// Solve for B = 0.5 * (F R^T + R F^T), keeping Q fixed
// @pre B is symmetric
template <typename T>
void SolveForNewF2(const Matrix3<T>& B, const Matrix3<T>& R, Matrix3<T>* F) {
  Matrix3<T> Q, S;
  fem::internal::PolarDecompose<T>(*F, &Q, &S);
  Eigen::Matrix<T, 6, 6> C;
  Eigen::Vector<T, 6> rhs;
  rhs(0) = B(0, 0);
  C(0, 0) = Q(0, 0) * R(0, 0);                      // S00
  C(0, 1) = Q(0, 0) * R(0, 1) + Q(0, 1) * R(0, 0);  // S01
  C(0, 2) = Q(0, 0) * R(0, 2) + Q(0, 2) * R(0, 0);  // S02
  C(0, 3) = Q(0, 1) * R(0, 1);                      // S11
  C(0, 4) = Q(0, 1) * R(0, 2) + Q(0, 2) * R(0, 1);  // S12
  C(0, 5) = Q(0, 2) * R(0, 2);                      // S12

  rhs(1) = B(0, 1);
  C(1, 0) = 0.5 * (Q(0, 0) * R(1, 0) + Q(1, 0) * R(0, 0));  // S00
  C(1, 1) = 0.5 * (Q(0, 0) * R(1, 1) + Q(0, 1) * R(1, 0) + Q(1, 0) * R(0, 1) +
                   Q(1, 1) * R(0, 0));  // S01
  C(1, 2) = 0.5 * (Q(0, 0) * R(1, 2) + Q(0, 2) * R(1, 0) + Q(1, 0) * R(0, 2) +
                   Q(1, 2) * R(0, 0));                      // S02
  C(1, 3) = 0.5 * (Q(0, 1) * R(1, 1) + Q(1, 1) * R(0, 1));  // S11
  C(1, 4) = 0.5 * (Q(0, 1) * R(1, 2) + Q(0, 2) * R(1, 1) + Q(1, 1) * R(0, 2) +
                   Q(1, 2) * R(0, 1));                      // S12
  C(1, 5) = 0.5 * (Q(0, 2) * R(1, 2) + Q(1, 2) * R(0, 2));  // S22

  rhs(2) = B(0, 2);
  C(2, 0) = 0.5 * (Q(0, 0) * R(2, 0) + Q(2, 0) * R(0, 0));  // S00
  C(2, 1) = 0.5 * (Q(0, 0) * R(2, 1) + Q(0, 1) * R(2, 0) + Q(2, 0) * R(0, 1) +
                   Q(2, 1) * R(0, 0));  // S01
  C(2, 2) = 0.5 * (Q(0, 0) * R(2, 2) + Q(0, 2) * R(2, 0) + Q(2, 0) * R(0, 2) +
                   Q(2, 2) * R(0, 0));                      // S02
  C(2, 3) = 0.5 * (Q(0, 1) * R(2, 1) + Q(2, 1) * R(0, 1));  // S11
  C(2, 4) = 0.5 * (Q(0, 1) * R(2, 2) + Q(0, 2) * R(2, 1) + Q(2, 1) * R(0, 2) +
                   Q(2, 2) * R(0, 1));                      // S12
  C(2, 5) = 0.5 * (Q(0, 2) * R(2, 2) + Q(2, 2) * R(0, 2));  // S22

  rhs(3) = B(1, 1);
  C(3, 0) = Q(1, 0) * R(1, 0);                      // S00
  C(3, 1) = Q(1, 0) * R(1, 1) + Q(1, 1) * R(1, 0);  // S01
  C(3, 2) = Q(1, 0) * R(1, 2) + Q(1, 2) * R(1, 0);  // S02
  C(3, 3) = Q(1, 1) * R(1, 1);                      // S11
  C(3, 4) = Q(1, 1) * R(1, 2) + Q(1, 2) * R(1, 1);  // S12
  C(3, 5) = Q(1, 2) * R(1, 2);                      // S22

  rhs(4) = B(1, 2);
  C(4, 0) = 0.5 * (Q(1, 0) * R(2, 0) + Q(2, 0) * R(1, 0));  // S00
  C(4, 1) = 0.5 * (Q(1, 0) * R(2, 1) + Q(1, 1) * R(2, 0) + Q(2, 0) * R(1, 1) +
                   Q(2, 1) * R(1, 0));  // S01
  C(4, 2) = 0.5 * (Q(1, 0) * R(2, 2) + Q(1, 2) * R(2, 0) + Q(2, 0) * R(1, 2) +
                   Q(2, 2) * R(1, 0));
  C(4, 3) = 0.5 * (Q(1, 1) * R(2, 1) + Q(2, 1) * R(1, 1));  // S11
  C(4, 4) = 0.5 * (Q(1, 1) * R(2, 2) + Q(1, 2) * R(2, 1) + Q(2, 1) * R(1, 2) +
                   Q(2, 2) * R(1, 1));                      // S12
  C(4, 5) = 0.5 * (Q(1, 2) * R(2, 2) + Q(2, 2) * R(1, 2));  // S22

  rhs(5) = B(2, 2);
  C(5, 0) = Q(2, 0) * R(2, 0);                      // S00
  C(5, 1) = Q(2, 0) * R(2, 1) + Q(2, 1) * R(2, 0);  // S01
  C(5, 2) = Q(2, 0) * R(2, 2) + Q(2, 2) * R(2, 0);  // S02
  C(5, 3) = Q(2, 1) * R(2, 1);                      // S11
  C(5, 4) = Q(2, 1) * R(2, 2) + Q(2, 2) * R(2, 1);  // S12
  C(5, 5) = Q(2, 2) * R(2, 2);                      // S22

  Eigen::ColPivHouseholderQR<Eigen::Matrix<T, 6, 6>> decomp(C);
  Eigen::Vector<T, 6> y = decomp.solve(rhs);
  DRAKE_ASSERT(&y != nullptr);
  S(0, 0) = y(0);
  S(0, 1) = y(1);
  S(0, 2) = y(2);
  S(1, 0) = S(0, 1);
  S(1, 1) = y(3);
  S(1, 2) = y(4);
  S(2, 0) = S(0, 2);
  S(2, 1) = S(1, 2);
  S(2, 2) = y(5);

  (*F) = (Q * S);
}

// the infinite cylinder is defined as axis along vector (1,1,1), and given
// radius
template <typename T>
void ProjectToSkewedCylinder(double radius, Vector3<T>* point) {
  // the disc parallel to the cylinder that passes the point will center at
  T c = ((*point)(0) + (*point)(1) + (*point)(2)) / 3.0;
  Vector3<T> center(c, c, c);

  Vector3<T> c2p = *point - center;
  T distance = c2p.norm();
  if (distance <= radius) {
    return;
  } else {
    (*point) = center + c2p * radius / distance;
  }
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
