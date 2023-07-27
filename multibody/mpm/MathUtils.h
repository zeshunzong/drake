#pragma once

#include "drake/common/eigen_types.h"
#include "drake/common/autodiff.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace mathutils {

// Return (i, j, k)th entry of the third order permutation tensor
template <typename T>
T LeviCivita(int i, int j, int k){
    // Even permutation
    if ((i == 0 && j == 1 && k == 2) || (i == 1 && j == 2 && k == 0)
        || (i == 2 && j == 0 && k == 1)) {
        return 1.0;
    }
    // Odd permutation
    if ((i == 2 && j == 1 && k == 0) || (i == 0 && j == 2 && k == 1)
        || (i == 1 && j == 0 && k == 2)) {
        return -1.0;
    }
    return 0.0;
}

// Calculate A:ε
template <typename T>
Vector3<T> ContractionWithLeviCivita(const Matrix3<T>& A){
    Vector3<T> A_dot_eps = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k) {
    for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
        A_dot_eps(k) += A(i, j)*LeviCivita<T>(i, j, k);
    }
    }
    }
    return A_dot_eps;
}

/**
   Prevent x from getting more than eps-close to zero
   if x ∈ (−ε, ε), set x to −ε (if x < 0) or −ε (if x > 0)
 */
template <typename T>
T clamp_small_magnitude(const T& x, const T& eps){
    DRAKE_DEMAND(eps >= 0);
    if (x < -eps){
        return x;}
    else if (x < 0){
        return -eps;}
    else if (x < eps){
        return eps;}
    else{
        return x;}
}

/**
   Robustly computing log(x+1)/x
//  */
template <typename T>
T log_1px_over_x(const T& x, const T& eps) {
    DRAKE_DEMAND(eps > 0);
    using std::abs; // ADL overload to AutoDiff type
    if (abs(x) < eps){
        return 1 - x / (2) + x * x / (3) - x * x * x / (4);}
    else{
        // return std::log1p(x) / x;
        using std::log; // ADL overload to AutoDiff type, std::log1p not supperted
        return log(1+x) / x;}
}

/**
   Robustly computing (logx-logy)/(x-y)
 */
template <typename T>
T diff_log_over_diff(const T& x, const T& y, const T& eps) {
    DRAKE_DEMAND(eps > 0);
    T p = x / y - 1;
    return log_1px_over_x(p, eps) / y;
}

/**
   Robustly computing (x logy - y logx)/(x-y)
 */
template <typename T>
T diff_interlock_log_over_diff(const T& x, const T& y, const T& logy, const T& eps) {
    DRAKE_DEMAND(eps > 0);
    return logy - y * diff_log_over_diff(x, y, eps);
}

/**
   Robustly computing (expx-expy)/(x-y)
 */
template <class T>
T exp_m1x_over_x(const T& x, const T& eps) {
    DRAKE_DEMAND(eps > 0);
    using std::abs;
    if (abs(x) < eps){
        return 1 + x / (2) + x * x / (6) + x * x * x / (24);}
    else {
        using std::exp;
        // return std::expm1(x) / x;
        return (exp(x)-1) / x;} // ADL overload to AutoDiff type, std::expm1 not supperted
}

/**
   Robustly computing (expx-1)/x
 */
template <class T>
T diff_exp_over_diff(const T& x, const T& y, const T& eps) {
    DRAKE_DEMAND(eps > 0);
    T p = x - y;
    using std::exp;
    return exp_m1x_over_x(p, eps) * exp(y);
}

}  // namespace mathutils
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
