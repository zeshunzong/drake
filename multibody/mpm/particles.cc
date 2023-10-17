#include "drake/multibody/mpm/particles.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
Particles<T>::Particles() : num_particles_(0) {}

template <typename T>
Particles<T>::Particles(size_t num_particles)
    : num_particles_(num_particles),
      positions_(num_particles),
      velocities_(num_particles),
      masses_(num_particles),
      reference_volumes_(num_particles),
      trial_deformation_gradients_(num_particles),
      elastic_deformation_gradients_(num_particles),
      B_matrices_(num_particles),
      temporary_scalar_field_(num_particles),
      temporary_vector_field_(num_particles),
      temporary_matrix_field_(num_particles) {}

template <typename T>
void Particles<T>::Reorder(const std::vector<size_t>& new_order) {
  DRAKE_DEMAND((new_order.size()) == num_particles_);
  for (size_t i = 0; i < num_particles_ - 1; ++i) {
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
    } else {
      DRAKE_UNREACHABLE();
    }
  }
}

template <typename T>
void Particles<T>::Reorder2(const std::vector<size_t>& new_order) {
  DRAKE_DEMAND((new_order.size()) == num_particles_);

  for (size_t i = 0; i < num_particles_; ++i) {
    temporary_scalar_field_[i] = masses_[new_order[i]];
  }
  masses_.swap(temporary_scalar_field_);

  for (size_t i = 0; i < num_particles_; ++i) {
    temporary_scalar_field_[i] = reference_volumes_[new_order[i]];
  }
  reference_volumes_.swap(temporary_scalar_field_);

  for (size_t i = 0; i < num_particles_; ++i) {
    temporary_vector_field_[i] = positions_[new_order[i]];
  }
  positions_.swap(temporary_vector_field_);

  for (size_t i = 0; i < num_particles_; ++i) {
    temporary_vector_field_[i] = velocities_[new_order[i]];
  }
  velocities_.swap(temporary_vector_field_);

  for (size_t i = 0; i < num_particles_; ++i) {
    temporary_matrix_field_[i] = trial_deformation_gradients_[new_order[i]];
  }
  trial_deformation_gradients_.swap(temporary_matrix_field_);

  for (size_t i = 0; i < num_particles_; ++i) {
    temporary_matrix_field_[i] = elastic_deformation_gradients_[new_order[i]];
  }
  elastic_deformation_gradients_.swap(temporary_matrix_field_);

  for (size_t i = 0; i < num_particles_; ++i) {
    temporary_matrix_field_[i] = B_matrices_[new_order[i]];
  }
  B_matrices_.swap(temporary_matrix_field_);
}

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
  ++num_particles_;

  temporary_scalar_field_.emplace_back();
  temporary_vector_field_.emplace_back();
  temporary_matrix_field_.emplace_back();
}

// template <typename T>
// void Particles<T>::Scratch::ReorderVectorWithScratch(
//     const std::vector<size_t>& new_order, std::vector<Vector3<T>>* data) {
//   for (size_t i = 0; i < new_order.size(); ++i) {
//     scratch_.vector_scratch[i] = *data[new_order[i]];
//   }
//   data->swap(scratch_.vector_scratch);
// }

// template <typename T>
// void Particles<T>::Scratch::ReorderMatrixWithScratch(
//     const std::vector<size_t>& new_order, std::vector<Matrix3<T>>* data) {
//   for (size_t i = 0; i < new_order.size(); ++i) {
//     scratch_.matrix_scratch[i] = *data[new_order[i]];
//   }
//   data->swap(scratch_.vector_scratch);
// }

// template <typename T>
// void Particles<T>::Scratch::ReorderScalarWithScratch(
//     const std::vector<size_t>& new_order, std::vector<T>* data) {
//   for (size_t i = 0; i < new_order.size(); ++i) {
//     scratch_.scalar_scratch[i] = *data[new_order[i]];
//   }
//   data->swap(scratch_.scalar_scratch);
// }

template class Particles<double>;
template class Particles<AutoDiffXd>;
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
