#include "drake/multibody/mpm/constitutive_model/elastoplastic_model.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace constitutive_model {

template <typename T>
ElastoPlasticModel<T>::ElastoPlasticModel(const T& youngs_modulus,
                                          const T& poissons_ratio)
    : lambda_(youngs_modulus * poissons_ratio / (1 + poissons_ratio) /
              (1 - 2 * poissons_ratio)),
      mu_(youngs_modulus / (2 * (1 + poissons_ratio))),
      youngs_modulus_(youngs_modulus),
      poissons_ratio_{poissons_ratio} {
  DRAKE_ASSERT(youngs_modulus >= 0);
  DRAKE_ASSERT(poissons_ratio > -1.0 && poissons_ratio < 0.5);
}

template class ElastoPlasticModel<double>;
template class ElastoPlasticModel<AutoDiffXd>;

}  // namespace constitutive_model
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
