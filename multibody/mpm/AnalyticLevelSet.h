#pragma once

#include <math.h>

#include <algorithm>
#include <array>
#include <limits>

#include "drake/common/eigen_types.h"
#include "drake/math/rotation_matrix.h"

namespace drake {
namespace multibody {
namespace mpm {

// A base class providing the interface of primitive geometries' level set in
// reference configuration
class AnalyticLevelSet {
 public:
    AnalyticLevelSet(double volume,
                     const std::array<Vector3<double>, 2>& bounding_box);

    // Return true if the position is in the interiror of the level set.
    virtual bool InInterior(const Vector3<double>& position) const = 0;

    // Return the outward unit normal of the interior of the analytic level set.
    // @throws exception if the position is outside of the geometry
    virtual Vector3<double> Normal(const Vector3<double>& position)
                                                                     const = 0;

    // Return the volume enclosed by the level set
    double get_volume() const;

    // Return the bounding bound of the geometry
    // We denote the first component of bounding_box_ as xmin_, second component
    // is xmax_. xmin_, xmax_ represents the bounding box of the geometry
    // i.e. the geometry lies in
    // [xmin_(0), xmax_(0)]X[xmin_(1), xmax_(1)]X[xmin_(2), xmax_(2)]
    const std::array<Vector3<double>, 2>& get_bounding_box() const;

    virtual ~AnalyticLevelSet() = default;

 protected:
    double volume_;
    std::array<Vector3<double>, 2> bounding_box_{};
};  // class AnalyticLevelSet

// An analytic level set class for half space with the given `normal` and center
// (0, 0, 0)
class HalfSpaceLevelSet : public AnalyticLevelSet {
 public:
    // @pre the normal is nonzero
    explicit HalfSpaceLevelSet(const Vector3<double>& normal);
    bool InInterior(const Vector3<double>& position) const final;

    Vector3<double> Normal(const Vector3<double>& position) const final;

 private:
    Vector3<double> normal_;
};  // class HalfSpaceLevelSet

// An analytic level set class for sphere with radius radius_ and center
// (0, 0, 0)
class SphereLevelSet : public AnalyticLevelSet {
 public:
    // @pre the radius of the sphere is positive
    explicit SphereLevelSet(double radius);
    bool InInterior(const Vector3<double>& position) const final;

    // At the singularity (0, 0, 0), we define the interior normal vector
    // as (1, 0, 0)
    Vector3<double> Normal(const Vector3<double>& position) const final;

 private:
    double radius_;
};  // class SphereLevelSet

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
// For any point inside the box, we define its interior normal as the outward
// normal of its closest face's normal. If multiple faces are closest to the
// point. We break even in the order of left, right, front, back, bottom, and
// top faces
class BoxLevelSet : public AnalyticLevelSet {
 public:
    // @pre Each entry in scale is positive.
    explicit BoxLevelSet(const Vector3<double>& xscale);
    bool InInterior(const Vector3<double>& position) const final;
    Vector3<double> Normal(const Vector3<double>& position) const final;

 private:
    Vector3<double> xscale_{};
};  // class BoxLevelSet

// An analytic level set class for right cylinder
//         _.-----._
//       .-         -.
//       |-_       _-|
//       |  ~-----~  |                     y
//       |           |                z | /
//       `._       _.'                    - x
//          "-----"
// Where the circular area is parallel to the xy-plane, and the central axis is
// parallel to z-axis. The level set can be parametetrized as [rcosθ, rsinθ, z],
// where r is the radius given, and z ∈ [-height, height].
// For all interior points, the normal is defined as the vector from its
// position to the closest point on the cylindrical surface (ignoring the lid).
// For points on the z axis, their normals are (1.0, 0.0, 0.0).
class CylinderLevelSet : public AnalyticLevelSet {
 public:
    // @pre the height and the radius of the cylinder are positive
    explicit CylinderLevelSet(double height, double radius);
    bool InInterior(const Vector3<double>& position) const final;
    Vector3<double> Normal(const Vector3<double>& position) const final;

 private:
    double height_;
    double radius_;
};  // class CylinderLevelSet

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
