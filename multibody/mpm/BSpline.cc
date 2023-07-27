#include "drake/multibody/mpm/BSpline.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
BSpline<T>::BSpline(): h_(1.0), position_(0.0, 0.0, 0.0) {}

template <typename T>
BSpline<T>::BSpline(const T h, const Vector3<T>& pos) {
    DRAKE_DEMAND(h > 0.0);
    h_        = h;
    position_ = pos;
}

// TODO(yiminlin.tri): For below routines, After determining the implementations
// of particles and grid points, we can pass these objects as arguments for
// convinience.
template <typename T>
bool BSpline<T>::InSupport(const Vector3<T>& x) const {
    using std::abs;
    return (abs(x(0)-position_(0))/h_ < 1.5
         && abs(x(1)-position_(1))/h_ < 1.5
         && abs(x(2)-position_(2))/h_ < 1.5);
}

template <typename T>
T BSpline<T>::EvalBasis(const Vector3<T>& x) const {
    // If the basis is not in the support, we simply return 0.0
    if (!InSupport(x)) {
        return 0.0;
    } else {
        return Eval1DBasis((x(0)-position_(0))/h_)
              *Eval1DBasis((x(1)-position_(1))/h_)
              *Eval1DBasis((x(2)-position_(2))/h_);
    }
}

template <typename T>
Vector3<T> BSpline<T>::EvalGradientBasis(const Vector3<T>& x) const {
    if (!InSupport(x)) {
        return Vector3<T>(0.0, 0.0, 0.0);
    } else {
        T scale = 1.0/h_;
        Vector3<T> coordinate = Vector3<T>(scale*(x(0)-position_(0)),
                                                     scale*(x(1)-position_(1)),
                                                     scale*(x(2)-position_(2)));
        Vector3<T> basis_val = Vector3<T>(Eval1DBasis(coordinate(0)),
                                                    Eval1DBasis(coordinate(1)),
                                                    Eval1DBasis(coordinate(2)));
        return Vector3<T>(scale*(EvalGradient1DBasis(coordinate(0)))
                              *basis_val(1)*basis_val(2),
                               scale*(EvalGradient1DBasis(coordinate(1)))
                              *basis_val(0)*basis_val(2),
                               scale*(EvalGradient1DBasis(coordinate(2)))
                              *basis_val(0)*basis_val(1));
    }
}

template <typename T>
std::pair<T, Vector3<T>>
    BSpline<T>::EvalBasisAndGradient(const Vector3<T>& x) const {
    if (!InSupport(x)) {
        return std::pair<T, Vector3<T>>(0.0,
                                                Vector3<T>(0.0, 0.0, 0.0));
    } else {
        T scale = 1.0/h_;
        Vector3<T> coordinate = Vector3<T>(scale*(x(0)-position_(0)),
                                                     scale*(x(1)-position_(1)),
                                                     scale*(x(2)-position_(2)));
        Vector3<T> basis_val = Vector3<T>(Eval1DBasis(coordinate(0)),
                                                    Eval1DBasis(coordinate(1)),
                                                    Eval1DBasis(coordinate(2)));
        return std::pair<T, Vector3<T>>(
              basis_val(0)*basis_val(1)*basis_val(2),
              Vector3<T>(scale*(EvalGradient1DBasis(coordinate(0)))
                             *basis_val(1)*basis_val(2),
                              scale*(EvalGradient1DBasis(coordinate(1)))
                             *basis_val(0)*basis_val(2),
                              scale*(EvalGradient1DBasis(coordinate(2)))
                             *basis_val(0)*basis_val(1)));
    }
}

template <typename T>
T BSpline<T>::get_h() const {
    return h_;
}

template <typename T>
Vector3<T> BSpline<T>::get_position() const {
    return position_;
}

template <typename T>
T BSpline<T>::Eval1DBasis(T r) const {
    using std::abs;
    T r_abs = abs(r);
    if (r_abs >= 1.5) {
        return 0.0;
    } else if (r_abs < 1.5 && r_abs >= 0.5) {
        return .5*(1.5-r_abs)*(1.5-r_abs);
    } else {
        return 0.75-r_abs*r_abs;
    }
}

template <typename T>
T BSpline<T>::EvalGradient1DBasis(T r) const {
    if (r <= 0.5 && r >= -0.5) {
        return -2.0*r;
    } else if (r >= 0.5 && r < 1.5) {
        return -1.5 + r;
    } else if (r <= -0.5 && r > -1.5) {
        return 1.5 + r;
    } else {
        return 0.0;
    }
}


template class BSpline<double>;
template class BSpline<AutoDiffXd>;


}  // namespace mpm
}  // namespace multibody
}  // namespace drake
