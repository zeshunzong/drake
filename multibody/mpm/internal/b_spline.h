#pragma once

#include <utility>

#include "drake/common/autodiff.h"
#include "drake/common/eigen_types.h"
namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

// An implementation of 3D quadratic B spline. This gives an (interpolation)
// function centered at center with support being [-1.5h,1.5h] in each dimension
// See https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf page
// 32-34.
template <typename T>
class BSpline {
 public:
  // Constructs a 3D quadratic B spline (function) centered at center with
  // scaling h. See class documentation for details.
  // @pre h > 0
  BSpline(double h, const Vector3<double>& center);

  double h() const { return h_; }
  const Vector3<double>& center() const { return center_; }

  // Computes BSpline(x), parameterized by h_ and center_.
  T ComputeValue(const Vector3<T>& x) const;

  // Computes ∇BSpline(x).
  Vector3<T> ComputeGradient(const Vector3<T>& x) const;

  // Computes {BSpline(x), ∇BSpline(x)}.
  // If both BSpline(x) and ∇BSpline(x) are needed, calling this function is
  // more efficient than calling ComputeValue(x) and ComputeGradient(x)
  // separately.
  std::pair<T, Vector3<T>> ComputeValueAndGradient(const Vector3<T>& x) const;

 private:
  // A (1D) reference bpline is a 1D function N(x) defined with h = 1 and (1d)
  // center = 0, as in eqn(123) in
  // https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf
  // Denote the reference function as N(x).
  // Computes N(x), N'(x)
  T ComputeReferenceValue(const T& x) const;
  T ComputeReferenceDerivative(const T& x) const;

  // The scaling of the reference domain. Since the class is for the usage of
  // MPM, we assume we are on the uniform grid, and the physical domain (x, y,
  // z) can be transformed to the reference domain (r, s, t) through a linear
  // transformation \phi^-1(x, y, z) = (r, s, t) = 1/h*(x-xc, y-yc, z-zc). The
  // basis on physical domain will have support [-1.5h, 1.5h].

  double h_{};
  double one_over_h_{};
  // The "center" of the Bspline, or (xc, yc, zc) in above comment
  Vector3<double> center_{};
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
