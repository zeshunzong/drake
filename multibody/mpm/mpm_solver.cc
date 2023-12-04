#include "drake/multibody/mpm/mpm_solver.h"

namespace drake {
namespace multibody {
namespace mpm {

template class MpmSolver<double>;
template class MpmSolver<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
