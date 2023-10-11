#include "drake/multibody/mpm/internal/poisson_disk.h"

#include <poisson_disk_sampling.h>

namespace thinks {

// Specialization of vector traits for Vector3
template <>
struct VecTraits<drake::Vector3<double>> {
  using ValueType = typename drake::Vector3<double>::Scalar;
  static constexpr auto kSize = 3;

  static auto Get(const drake::Vector3<double>& vec, const std::size_t i)
      -> ValueType {
    return vec(i);
  }

  static void Set(drake::Vector3<double>* vec, const std::size_t i,
                  const ValueType val) {
    (*vec)(i) = val;
  }
};

}  // namespace thinks

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

std::vector<Vector3<double>> PoissonDiskSampling(
    double r, const std::array<double, 3>& x_min,
    const std::array<double, 3>& x_max) {
  return thinks::PoissonDiskSampling<double, 3, Vector3<double>>(r, x_min,
                                                                 x_max);
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
