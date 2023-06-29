#pragma once

#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

}
}
}
}

// A simple hash function for Vector3<int>, the coordinate in index space
namespace std {
template <>
struct hash<drake::Vector3<int>> {
    std::size_t operator()(const drake::Vector3<int>& vec) const {
        return vec(0) + (31*vec(1)) + (31*31*vec(2));
    }
};
}  // namespace std
