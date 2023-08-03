#include "drake/geometry/query_results/mpm_contact.h"

#include <utility>

namespace drake {
namespace geometry {
namespace internal {

using multibody::contact_solvers::internal::PartialPermutation;


template class MpmContact<double>;

}  // namespace internal
}  // namespace geometry
}  // namespace drake
