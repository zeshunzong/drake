#include "drake/multibody/mpm/MathUtils.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace mathutils {

double LeviCivita(int i, int j, int k) {
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

Vector3<double> ContractionWithLeviCivita(const Matrix3<double>& A) {
    Vector3<double> A_dot_eps = {0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k) {
    for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
        A_dot_eps(k) += A(i, j)*LeviCivita(i, j, k);
    }
    }
    }
    return A_dot_eps;
}

}  // namespace mathutils
}  // namespace mpm
}  // namespace multibody
}  // namespace drake

