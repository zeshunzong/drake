#include "drake/multibody/mpm/internal/analytic_level_set.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

AnalyticLevelSet::AnalyticLevelSet() : volume_(0.0) {}

Vector3<double> AnalyticLevelSet::GetNormal(
    const Vector3<double>& position) const {
  if (IsInClosure(position)) {
    return DoGetNormal(position);
  } else {
    throw std::logic_error(
        "Normal vector is not avaible for position in exterior of level set.");
  }
}

HalfSpaceLevelSet::HalfSpaceLevelSet(const Vector3<double>& normal)
    : AnalyticLevelSet() {
  this->volume_ = std::numeric_limits<double>::infinity();
  this->bounding_box_ = {
      -std::numeric_limits<double>::infinity() * Vector3<double>::Ones(),
      std::numeric_limits<double>::infinity() * Vector3<double>::Ones()};
  DRAKE_DEMAND((normal.array() != 0.0).any());
  normal_ = normal.normalized();
}

bool HalfSpaceLevelSet::IsInClosure(const Vector3<double>& position) const {
  return (position.dot(normal_) <= 0.0);
}

Vector3<double> HalfSpaceLevelSet::DoGetNormal(
    const Vector3<double>& position) const {
  unused(position);
  return normal_;
}

SphereLevelSet::SphereLevelSet(double radius)
    : AnalyticLevelSet(), radius_(radius) {
  DRAKE_DEMAND(radius > 0);
  this->volume_ = 4.0 / 3.0 * M_PI * radius * radius * radius;
  this->bounding_box_ = {-radius * Vector3<double>::Ones(),
                         radius * Vector3<double>::Ones()};
}

bool SphereLevelSet::IsInClosure(const Vector3<double>& position) const {
  return position.norm() <= radius_;
}

Vector3<double> SphereLevelSet::DoGetNormal(
    const Vector3<double>& position) const {
  if (position.norm() == 0) {
    return Vector3<double>{1.0, 0.0, 0.0};
  }
  return position.normalized();
}

BoxLevelSet::BoxLevelSet(const Vector3<double>& xscale)
    : AnalyticLevelSet(), xscale_(xscale) {
  DRAKE_DEMAND(xscale_(0) > 0);
  DRAKE_DEMAND(xscale_(1) > 0);
  DRAKE_DEMAND(xscale_(2) > 0);
  this->volume_ = 8 * xscale(0) * xscale(1) * xscale(2);
  this->bounding_box_ = {-xscale, xscale};
}

bool BoxLevelSet::IsInClosure(const Vector3<double>& position) const {
  return ((std::abs(position(0)) <= xscale_(0)) &&
          (std::abs(position(1)) <= xscale_(1)) &&
          (std::abs(position(2)) <= xscale_(2)));
}

Vector3<double> BoxLevelSet::DoGetNormal(
    const Vector3<double>& position) const {
  double dist_left = (position(0) + xscale_(0));
  double dist_right = (xscale_(0) - position(0));
  double dist_front = (position(1) + xscale_(1));
  double dist_back = (xscale_(1) - position(1));
  double dist_bottom = (position(2) + xscale_(2));
  double dist_top = (xscale_(2) - position(2));
  double min_dist = std::min(
      {dist_left, dist_right, dist_front, dist_back, dist_bottom, dist_top});
  if (min_dist == dist_left) {
    return Vector3<double>{-1.0, 0.0, 0.0};
  }
  if (min_dist == dist_right) {
    return Vector3<double>{1.0, 0.0, 0.0};
  }
  if (min_dist == dist_front) {
    return Vector3<double>{0.0, -1.0, 0.0};
  }
  if (min_dist == dist_back) {
    return Vector3<double>{0.0, 1.0, 0.0};
  }
  if (min_dist == dist_bottom) {
    return Vector3<double>{0.0, 0.0, -1.0};
  }
  if (min_dist == dist_top) {
    return Vector3<double>{0.0, 0.0, 1.0};
  }
  DRAKE_UNREACHABLE();
}

CylinderLevelSet::CylinderLevelSet(double height, double radius)
    : AnalyticLevelSet(), height_(height), radius_(radius) {
  DRAKE_DEMAND(height > 0);
  DRAKE_DEMAND(radius > 0);
  this->volume_ = 2.0 * M_PI * radius * radius * height;
  this->bounding_box_ = {
      {{-radius, -radius, -height}, {radius, radius, height}}};
}

bool CylinderLevelSet::IsInClosure(const Vector3<double>& position) const {
  return (((position(0) * position(0) + position(1) * position(1)) <=
           radius_ * radius_) &&
          (std::abs(position(2)) <= height_));
}

Vector3<double> CylinderLevelSet::DoGetNormal(
    const Vector3<double>& position) const {
  Vector3<double> projection_xy(position(0), position(1), 0.0);
  // distance to top/bottom plate, whichever is closer
  double distance_z = height_ - std::abs(position(2));
  // distance to the cylindrical surface
  double distance_r = radius_ - projection_xy.norm();
  if (distance_r <= distance_z) {
    // closer to cylindrical surface
    return projection_xy.normalized();
  } else {
    if (position(2) >= 0) {
      // closer to the top disk
      return Vector3<double>{0.0, 0.0, 1.0};
    } else {
      // closer to the bottom disk
      return Vector3<double>{0.0, 0.0, -1.0};
    }
  }
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
