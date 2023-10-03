#include "drake/multibody/mpm/internal/b_spline.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

template <typename T>
BSpline<T>::BSpline() : h_(1.0), one_over_h_(1.0), center_(0.0, 0.0, 0.0) {}

template <typename T>
BSpline<T>::BSpline(double h, const Vector3<double>& center) {
  DRAKE_DEMAND(h > 0.0);
  h_ = h;
  one_over_h_ = 1.0 / h;
  center_ = center;
}

template <typename T>
T BSpline<T>::ComputeValue(const Vector3<T>& x) const {
  return ComputeReferenceValue((x(0) - center_(0)) * one_over_h_) *
         ComputeReferenceValue((x(1) - center_(1)) * one_over_h_) *
         ComputeReferenceValue((x(2) - center_(2)) * one_over_h_);
}

template <typename T>
Vector3<T> BSpline<T>::ComputeGradient(const Vector3<T>& x) const {
  Vector3<T> coordinate = Vector3<T>(one_over_h_ * (x(0) - center_(0)),
                                     one_over_h_ * (x(1) - center_(1)),
                                     one_over_h_ * (x(2) - center_(2)));
  Vector3<T> basis_val = Vector3<T>(ComputeReferenceValue(coordinate(0)),
                                    ComputeReferenceValue(coordinate(1)),
                                    ComputeReferenceValue(coordinate(2)));
  return Vector3<T>(one_over_h_ * (ComputeReferenceDerivative(coordinate(0))) *
                        basis_val(1) * basis_val(2),
                    one_over_h_ * (ComputeReferenceDerivative(coordinate(1))) *
                        basis_val(0) * basis_val(2),
                    one_over_h_ * (ComputeReferenceDerivative(coordinate(2))) *
                        basis_val(0) * basis_val(1));
}

template <typename T>
std::pair<T, Vector3<T>> BSpline<T>::ComputeValueAndGradient(
    const Vector3<T>& x) const {
  return std::pair<T, Vector3<T>>(ComputeValue(x), ComputeGradient(x));
}

template <typename T>
T BSpline<T>::ComputeReferenceValue(const T& x) const {
  using std::abs;
  T x_abs = abs(x);
  if (x_abs >= 1.5) {
    return 0.0;
  } else if (x_abs < 1.5 && x_abs >= 0.5) {
    return .5 * (1.5 - x_abs) * (1.5 - x_abs);
  } else {
    return 0.75 - x_abs * x_abs;
  }
}

template <typename T>
T BSpline<T>::ComputeReferenceDerivative(const T& x) const {
  if (x <= 0.5 && x >= -0.5) {
    return -2.0 * x;
  } else if (x >= 0.5 && x < 1.5) {
    return -1.5 + x;
  } else if (x <= -0.5 && x > -1.5) {
    return 1.5 + x;
  } else {
    return 0.0;
  }
}

template class BSpline<double>;
template class BSpline<AutoDiffXd>;

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
