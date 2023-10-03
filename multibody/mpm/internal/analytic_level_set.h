#pragma once

#include <math.h>

#include <algorithm>
#include <array>
#include <limits>

#include "drake/common/eigen_types.h"
#include "drake/common/unused.h"
#include "drake/math/rotation_matrix.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

/**
 * A base class providing the interface of primitive geometries' level set in
 * reference configuration
 */
class AnalyticLevelSet {
 public:
  /**
   * Returns true if a point at position is in the closure of this level set.
   * Disambiguation:
   * Interior: inside the level set, excluding boundary
   * Exterior: outside the level set, excluding boundary
   * Closure: interior + boundary
   */
  virtual bool IsInClosure(const Vector3<double>& position) const = 0;

  /**
   * Returns the outward unit normal of an interior point at position in the
   * analytic level set.
   * @throws std::exception if the point is in exterior or on boundary
   */
  Vector3<double> GetNormal(const Vector3<double>& position) const;

  virtual Vector3<double> DoGetNormal(
      const Vector3<double>& position) const = 0;

  /**
   * Returns the volume enclosed by the level set
   */
  double volume() const { return volume_; }

  /**
   * Returns the bounding box of the geometry
   * We denote the first component of bounding_box_ as xmin_, second component
   * is xmax_. xmin_, xmax_ represents the bounding box of the geometry. i.e.
   * the geometry lies in [xmin_(0), xmax_(0)]X[xmin_(1), xmax_(1)]X[xmin_(2),
   * xmax_(2)]
   */
  const std::array<Vector3<double>, 2>& bounding_box() const {
    return bounding_box_;
  }

  virtual ~AnalyticLevelSet() = default;

 protected:
  /**
   * Creates an empty level set, with volume = 0 and bounding box = {[0,0,0],
   * [0,0,0]}
   */
  AnalyticLevelSet();
  double volume_;
  std::array<Vector3<double>, 2> bounding_box_{};
};

/**
 * An analytic level set class for half space with the given `normal` and center
 * (0, 0, 0). A point x is inside the (closure of) the half plane if dot(x,
 * normal) <= 0. e.g. in 2D, the half plane y>=0 should be defined by normal =
 * (0,-1).
 */
class HalfSpaceLevelSet : public AnalyticLevelSet {
 public:
  // @pre the normal is a nonzero vector
  explicit HalfSpaceLevelSet(const Vector3<double>& normal);
  bool IsInClosure(const Vector3<double>& position) const final;
  /**
   * By construction for any point in the half space, its outward normal is the
   * normal used to define the half space
   */
  Vector3<double> DoGetNormal(const Vector3<double>& position) const final;

 private:
  Vector3<double> normal_;
};

/**
 * An analytic level set class for sphere with radius radius_ and center (0,0,0)
 */
class SphereLevelSet : public AnalyticLevelSet {
 public:
  // @pre radius > 0
  explicit SphereLevelSet(double radius);
  bool IsInClosure(const Vector3<double>& position) const final;

  /**
   * For x != (0, 0, 0), its normal is x/||x||.
   * For the singularity x = (0, 0, 0), we defines its outward normal vector to
   * be (1, 0, 0)
   */
  Vector3<double> DoGetNormal(const Vector3<double>& position) const final;

 private:
  double radius_;
};

// An analytic level set class for box of size
// [-xscale(0), xscale(0)] X [-xscale(1), xscale(1)] X [xscale(2), xscale(2)]
// centered at (0, 0, 0)
// Imagine the box is aligned like:
//          .+------+
//        .' |    .'|
//       +---+--+'  |
// left  |   |  |   |     right
//       |  .+--+---+                     y
//       |.'    | .'                 z | /
//       +------+'                       - x
//        bottom

class BoxLevelSet : public AnalyticLevelSet {
 public:
  // @pre Each entry in scale is positive.
  explicit BoxLevelSet(const Vector3<double>& xscale);
  bool IsInClosure(const Vector3<double>& position) const final;
  /**
   * For any point inside the box, we define its interior normal as the outward
   * normal of its closest face's normal. If multiple faces are closest to the
   * point. We break even in the order of left, right, front, back, bottom, and
   * top faces.
   */
  Vector3<double> DoGetNormal(const Vector3<double>& position) const final;

 private:
  Vector3<double> xscale_{};
};

// An analytic level set class for right cylinder
//         _.-----._
//       .-         -.
//       |-_       _-|
//       |  ~-----~  |                     y
//       |           |                z | /
//       `._       _.'                    - x
//          "-----"
// where the circular area is parallel to the xy-plane, and the central axis is
// parallel to z-axis.  The cylinder is centered at (0,0,0). The level set can
// be parametetrized as [rcosθ, rsinθ, z], where r is the radius given, and z ∈
// [-height, height].

class CylinderLevelSet : public AnalyticLevelSet {
 public:
  // @pre the height and the radius of the cylinder are positive
  explicit CylinderLevelSet(double height, double radius);
  bool IsInClosure(const Vector3<double>& position) const final;

  /**
   * The outward normal for a point on the cylindrical surface (including its
   * boundary) is the normal of the cylindrical surface at that point. The
   * outward normal for a point on the top/bottom disk (excluding boundary) is
   * (0,0,1)/(0,0,-1) The outward normal for an interior point is the normal of
   * its closest surface. When the point is equidistant from the (either top or
   * bottom) disk and the cylindrical surface, the normal is taken to be that of
   * the cylindrical surface (N.B. there is a discontinuity here).
   */
  Vector3<double> DoGetNormal(const Vector3<double>& position) const final;

 private:
  double height_;
  double radius_;
};

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
