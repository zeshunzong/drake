#pragma once

#include <vector>

#include "drake/common/eigen_types.h"

// A simple hash function for Vector3<int>
namespace std {
template <>
struct hash<drake::Vector3<int>> {
  std::size_t operator()(const drake::Vector3<int>& vec) const {
    std::size_t res = 17;
    res = res * 31 + hash<int>()(vec(0));
    res = res * 31 + hash<int>()(vec(1));
    res = res * 31 + hash<int>()(vec(2));
    return res;
  }
};
}  // namespace std
