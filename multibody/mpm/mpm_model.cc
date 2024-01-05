#include "drake/multibody/mpm/mpm_model.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
T MpmModel<T>::ComputeKineticAndGravitationalEnergy(
    const std::vector<Vector3<T>>& v_prev,
    const DeformationState<T>& deformation_state, double dt) const {
  T sum = 0;
  for (size_t i = 0; i < deformation_state.grid_data().num_active_nodes();
       ++i) {
    const Vector3<T>& dv =
        deformation_state.grid_data().GetVelocityAt(i) - v_prev[i];
    sum +=
        0.5 * deformation_state.grid_data().GetMassAt(i) * (dv).squaredNorm();
    sum -= dt * deformation_state.grid_data().GetMassAt(i) * dv.dot(gravity_);
  }
  return sum;
}

template <typename T>
void MpmModel<T>::ComputeMinusDEnergyDV(
    const MpmTransfer<T>& transfer, const std::vector<Vector3<T>>& v_prev,
    const DeformationState<T>& deformation_state, double dt,
    std::vector<Vector3<T>>* dEnergydV, TransferScratch<T>* scratch) const {
  DRAKE_ASSERT(dEnergydV != nullptr);
  DRAKE_ASSERT(scratch != nullptr);
  ComputeMinusDElasticEnergyDV(transfer, deformation_state, dt, dEnergydV,
                               scratch);
  MinusDKineticEnergyDVAndDGravitationalEnergyDV(v_prev, deformation_state, dt,
                                                 dEnergydV);
}

template <typename T>
void MpmModel<T>::ComputeMinusDElasticEnergyDV(
    const MpmTransfer<T>& transfer,
    const DeformationState<T>& deformation_state, double dt,
    std::vector<Vector3<T>>* dEnergydV, TransferScratch<T>* scratch) const {
  // -dedx
  transfer.ComputeGridElasticForces(deformation_state.particles(),
                                    deformation_state.sparse_grid(),
                                    deformation_state.Ps(), dEnergydV, scratch);
  // transform to -dedv, differ by a scale of dt due to chain rule
  for (size_t i = 0; i < dEnergydV->size(); ++i) {
    (*dEnergydV)[i] *= dt;
  }
}

template <typename T>
void MpmModel<T>::MinusDKineticEnergyDVAndDGravitationalEnergyDV(
    const std::vector<Vector3<T>>& v_prev,
    const DeformationState<T>& deformation_state, double dt,
    std::vector<Vector3<T>>* dEnergydV) const {
  DRAKE_ASSERT(dEnergydV != nullptr);
  DRAKE_ASSERT(dEnergydV->size() ==
               deformation_state.grid_data().num_active_nodes());
  for (size_t i = 0; i < deformation_state.grid_data().num_active_nodes();
       ++i) {
    const Vector3<T>& dv =
        deformation_state.grid_data().GetVelocityAt(i) - v_prev[i];
    (*dEnergydV)[i] -=
        (deformation_state.grid_data().GetMassAt(i) * dv -
         dt * deformation_state.grid_data().GetMassAt(i) * gravity_);
  }
}

template <typename T>
void MpmModel<T>::ComputeD2EnergyDV2(
    const MpmTransfer<T>& transfer,
    const DeformationState<T>& deformation_state, double dt,
    MatrixX<T>* hessian) const {
  ComputeD2ElasticEnergyDV2(transfer, deformation_state, dt, hessian);
  for (size_t i = 0; i < deformation_state.grid_data().num_active_nodes();
       ++i) {
    (*hessian)(3 * i, 3 * i) += deformation_state.grid_data().GetMassAt(i);
    (*hessian)(3 * i + 1, 3 * i + 1) +=
        deformation_state.grid_data().GetMassAt(i);
    (*hessian)(3 * i + 2, 3 * i + 2) +=
        deformation_state.grid_data().GetMassAt(i);
  }
}

template <typename T>
void MpmModel<T>::ComputeD2EnergyDV2TimesZ(
    const std::vector<Vector3<T>>& z, const MpmTransfer<T>& transfer,
    const DeformationState<T>& deformation_state, double dt,
    std::vector<Vector3<T>>* hessian_times_z) const {
  ComputeElasticHessianTimesZ(z, transfer, deformation_state, hessian_times_z);
  for (size_t i = 0; i < deformation_state.grid_data().num_active_nodes();
       ++i) {
    (*hessian_times_z)[i] *= (dt * dt);
    (*hessian_times_z)[i] += deformation_state.grid_data().GetMassAt(i) * z[i];
  }
}

template <typename T>
void MpmModel<T>::ComputeD2ElasticEnergyDV2(
    const MpmTransfer<T>& transfer,
    const DeformationState<T>& deformation_state, double dt,
    MatrixX<T>* hessian) const {
  ComputeElasticHessian(transfer, deformation_state, hessian);
  (*hessian) *= (dt * dt);
}

template class MpmModel<double>;
template class MpmModel<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
