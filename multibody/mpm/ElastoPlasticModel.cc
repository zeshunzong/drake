#include "drake/multibody/mpm/ElastoPlasticModel.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
ElastoPlasticModel<T>::ElastoPlasticModel(): ElastoPlasticModel(9e4, 0.49) {}

template <typename T>
ElastoPlasticModel<T>::ElastoPlasticModel(T E, T nu):
                                                lambda_(E*nu/(1+nu)/(1-2*nu)),
                                                mu_(E/(2*(1+nu))) {
    DRAKE_ASSERT(E >= 0);
    DRAKE_ASSERT(nu > -1.0 && nu < 0.5);
}


template class ElastoPlasticModel<double>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake

