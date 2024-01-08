#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

#include "drake/multibody/mpm/mpm_model.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
class MatrixReplacement;

}
}  // namespace multibody
}  // namespace drake

namespace Eigen {
namespace internal {
// MatrixReplacement looks-like a SparseMatrix, so let's inherits its traits:
template <typename T>
struct traits<drake::multibody::mpm::MatrixReplacement<T>>
    : public Eigen::internal::traits<Eigen::SparseMatrix<T>> {};
}  // namespace internal
}  // namespace Eigen

namespace drake {
namespace multibody {
namespace mpm {

// Example of a matrix-free wrapper from a user type to Eigen's compatible type
// For the sake of simplicity, this example simply wrap a Eigen::SparseMatrix.
template <typename T>
class MatrixReplacement : public Eigen::EigenBase<MatrixReplacement<T>> {
 public:
  // Required typedefs, constants, and method:
  typedef T Scalar;
  typedef T RealScalar;
  typedef int StorageIndex;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = false
  };
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MatrixReplacement);

  Eigen::EigenBase<MatrixReplacement<T>>::Index rows() const {
    return state().grid_data().num_active_nodes() * 3;
  }
  Eigen::EigenBase<MatrixReplacement<T>>::Index cols() const { return rows(); }

  template <typename Rhs>
  Eigen::Product<MatrixReplacement<T>, Rhs, Eigen::AliasFreeProduct> operator*(
      const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<MatrixReplacement<T>, Rhs, Eigen::AliasFreeProduct>(
        *this, x.derived());
  }

  MatrixReplacement(const MpmModel<T>& model, const DeformationState<T>& state,
                    const MpmTransfer<T>& transfer, double dt)
      : model_(model), state_(state), transfer_(transfer) {
    dt_ = dt;
  }

  const MpmModel<T>& model() const { return model_; }
  const DeformationState<T>& state() const { return state_; }
  const MpmTransfer<T>& transfer() const { return transfer_; }
  double dt() const { return dt_; }

 private:
  const MpmModel<T>& model_;
  const DeformationState<T>& state_;
  const MpmTransfer<T>& transfer_;
  double dt_ = 0.0;
};

}  // namespace mpm
}  // namespace multibody
}  // namespace drake

// Implementation of MatrixReplacement * Eigen::DenseVector though a
// specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {

template <typename Rhs, typename T>
struct generic_product_impl<drake::multibody::mpm::MatrixReplacement<T>, Rhs,
                            SparseShape, DenseShape,
                            GemvProduct>  // GEMV stands for matrix-vector
    : generic_product_impl_base<
          drake::multibody::mpm::MatrixReplacement<T>, Rhs,
          generic_product_impl<drake::multibody::mpm::MatrixReplacement<T>,
                               Rhs>> {
  typedef
      typename Product<drake::multibody::mpm::MatrixReplacement<T>, Rhs>::Scalar
          Scalar;

  template <typename Dest>
  static void scaleAndAddTo(
      Dest& dst, const drake::multibody::mpm::MatrixReplacement<T>& lhs,
      const Rhs& rhs, const Scalar& alpha) {
    // This method should implement "dst += alpha * lhs * rhs" inplace,
    // however, for iterative solvers, alpha is always equal to 1, so let's not
    // bother about it.
    assert(alpha == Scalar(1) && "scaling is not implemented");
    EIGEN_ONLY_USED_FOR_DEBUG(alpha);

    // // Here we could simply call dst.noalias() += lhs.my_matrix() * rhs,
    // // but let's do something fancier (and less efficient):
    // for (Index i = 0; i < lhs.cols(); ++i)
    //   dst += rhs(i) * lhs.my_matrix().col(i);

    lhs.model().AddD2EnergyDV2TimesZ(rhs, lhs.transfer(), lhs.state(), lhs.dt(),
                                     &dst);
  }
};
}  // namespace internal
}  // namespace Eigen
