#include "drake/multibody/mpm/particles.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
Particles<T>::Particles() {}

template <typename T>
void Particles<T>::AddParticle(const Vector3<T>& position,
                               const Vector3<T>& velocity, const T& mass,
                               const T& reference_volume,
                               const Matrix3<T>& trial_deformation_gradient,
                               const Matrix3<T>& elastic_deformation_gradient,
                               const Matrix3<T>& B_matrix) {
  positions_.emplace_back(position);
  velocities_.emplace_back(velocity);
  masses_.emplace_back(mass);
  reference_volumes_.emplace_back(reference_volume);
  trial_deformation_gradients_.emplace_back(trial_deformation_gradient);
  elastic_deformation_gradients_.emplace_back(elastic_deformation_gradient);
  B_matrices_.emplace_back(B_matrix);

  temporary_scalar_field_.emplace_back();
  temporary_vector_field_.emplace_back();
  temporary_matrix_field_.emplace_back();
  temporary_base_nodes_.emplace_back();

  base_nodes_.emplace_back();
  weights_.emplace_back();

  permutation_.emplace_back();
  CheckAttributesSize();
}

template <typename T>
void Particles<T>::AddParticle(const Vector3<T>& position,
                               const Vector3<T>& velocity, const T& mass,
                               const T& reference_volume) {
  AddParticle(position, velocity, mass, reference_volume,
              Matrix3<T>::Identity(), Matrix3<T>::Identity(),
              Matrix3<T>::Zero());
}

template <typename T>
void Particles<T>::Prepare(double h) {
  DRAKE_DEMAND(num_particles() > 0);
  // reserve space for batch_starts_ and batch_sizes
  // they can be upper-bounded by num_particles() as there is no empty batch
  if (batch_starts_.size() == 0) {
    batch_starts_.reserve(num_particles());
    batch_sizes_.reserve(num_particles());
  }

  // 1) compute the base node for each particle
  for (size_t p = 0; p < num_particles(); ++p) {
    base_nodes_[p] = internal::ComputeBaseNodeFromPosition<T>(positions_[p], h);
  }
  // 2) sorts particle attributes
  // 2.1) get a sorted permutation
  std::iota(permutation_.begin(), permutation_.end(), 0);
  std::sort(permutation_.begin(), permutation_.end(),
            [this](size_t i1, size_t i2) {
              return internal::CompareIndex3DLexicographically(base_nodes_[i1],
                                                               base_nodes_[i2]);
            });
  // 2.2) shuffle particle data based on permutation
  Reorder(permutation_);  // including base_nodes_
  // 3) compute batch_starts_ and batch_sizes_
  batch_starts_.clear();
  batch_starts_.push_back(0);
  batch_sizes_.clear();
  Vector3<int> current_3d_index = base_nodes_[0];
  size_t count = 1;
  for (size_t p = 1; p < num_particles(); ++p) {
    if (base_nodes_[p] == current_3d_index) {
      ++count;
    } else {
      batch_starts_.push_back(p);
      batch_sizes_.push_back(count);
      // continue to next batch
      current_3d_index = base_nodes_[p];
      count = 1;
    }
  }
  batch_sizes_.push_back(count);

  DRAKE_DEMAND(batch_sizes_.size() == batch_starts_.size());

  // 4) compute d and dw
  for (size_t p = 1; p < num_particles(); ++p) {
    weights_[p].Reset(positions_[p], base_nodes_[p], h);
  }
  // 5) mark that the reordering has been done
  need_reordering_ = false;
}

template <typename T>
void Particles<T>::Reorder(const std::vector<size_t>& new_order) {
  DRAKE_DEMAND((new_order.size()) == num_particles());
  for (size_t i = 0; i < num_particles() - 1; ++i) {
    // don't need to sort the last element if the first (n-1) elements have
    // already been sorted
    size_t ind = new_order[i];
    // the i-th element should be placed at ind-th position
    if (ind < i) {
      // its correct position is before i. In this case, the element must have
      // already been swapped to a position after i. find out where it has been
      // swapped to.
      while (ind < i) {
        ind = new_order[ind];
      }
    }
    // after this operation, ind is either equal to i or larger than i
    if (ind == i) {
      // at its correct position, nothing needs to be done
      continue;
    } else if (ind > i) {
      // TODO(zeshunzong): update this as more attributes are added

      std::swap(masses_[i], masses_[ind]);
      std::swap(reference_volumes_[i], reference_volumes_[ind]);

      std::swap(positions_[i], positions_[ind]);
      std::swap(velocities_[i], velocities_[ind]);

      std::swap(elastic_deformation_gradients_[i],
                elastic_deformation_gradients_[ind]);
      std::swap(trial_deformation_gradients_[i],
                trial_deformation_gradients_[ind]);
      std::swap(B_matrices_[i], B_matrices_[ind]);

      std::swap(base_nodes_[i], base_nodes_[ind]);
    } else {
      DRAKE_UNREACHABLE();
    }
  }
}

template <typename T>
void Particles<T>::Reorder2(const std::vector<size_t>& new_order) {
  DRAKE_DEMAND((new_order.size()) == num_particles());

  for (size_t i = 0; i < num_particles(); ++i) {
    temporary_scalar_field_[i] = masses_[new_order[i]];
  }
  masses_.swap(temporary_scalar_field_);

  for (size_t i = 0; i < num_particles(); ++i) {
    temporary_scalar_field_[i] = reference_volumes_[new_order[i]];
  }
  reference_volumes_.swap(temporary_scalar_field_);

  for (size_t i = 0; i < num_particles(); ++i) {
    temporary_vector_field_[i] = positions_[new_order[i]];
  }
  positions_.swap(temporary_vector_field_);

  for (size_t i = 0; i < num_particles(); ++i) {
    temporary_vector_field_[i] = velocities_[new_order[i]];
  }
  velocities_.swap(temporary_vector_field_);

  for (size_t i = 0; i < num_particles(); ++i) {
    temporary_matrix_field_[i] = trial_deformation_gradients_[new_order[i]];
  }
  trial_deformation_gradients_.swap(temporary_matrix_field_);

  for (size_t i = 0; i < num_particles(); ++i) {
    temporary_matrix_field_[i] = elastic_deformation_gradients_[new_order[i]];
  }
  elastic_deformation_gradients_.swap(temporary_matrix_field_);

  for (size_t i = 0; i < num_particles(); ++i) {
    temporary_matrix_field_[i] = B_matrices_[new_order[i]];
  }
  B_matrices_.swap(temporary_matrix_field_);

  for (size_t i = 0; i < num_particles(); ++i) {
    temporary_base_nodes_[i] = base_nodes_[new_order[i]];
  }
  base_nodes_.swap(temporary_base_nodes_);
}

template <typename T>
void Particles<T>::CheckAttributesSize() const {
  // by construction num_particles() = positions_.size()
  DRAKE_DEMAND(num_particles() == velocities_.size());
  DRAKE_DEMAND(num_particles() == masses_.size());
  DRAKE_DEMAND(num_particles() == reference_volumes_.size());
  DRAKE_DEMAND(num_particles() == trial_deformation_gradients_.size());
  DRAKE_DEMAND(num_particles() == elastic_deformation_gradients_.size());
  DRAKE_DEMAND(num_particles() == B_matrices_.size());

  DRAKE_DEMAND(num_particles() == temporary_scalar_field_.size());
  DRAKE_DEMAND(num_particles() == temporary_vector_field_.size());
  DRAKE_DEMAND(num_particles() == temporary_matrix_field_.size());

  DRAKE_DEMAND(num_particles() == base_nodes_.size());
  DRAKE_DEMAND(num_particles() == weights_.size());

  DRAKE_DEMAND(num_particles() == permutation_.size());
}

template class Particles<double>;
template class Particles<AutoDiffXd>;
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
