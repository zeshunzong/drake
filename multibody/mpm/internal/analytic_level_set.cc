#include "drake/multibody/mpm/internal/analytic_level_set.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {

AnalyticLevelSet::AnalyticLevelSet(double volume,
                            const std::array<Vector3<double>, 2>& bounding_box):
                                volume_(volume), bounding_box_(bounding_box) {}

double AnalyticLevelSet::get_volume() const {
    return volume_;
}

const std::array<Vector3<double>, 2>& AnalyticLevelSet::get_bounding_box()
                                                                        const {
    return bounding_box_;
}

HalfSpaceLevelSet::HalfSpaceLevelSet(const Vector3<double>& normal):
                AnalyticLevelSet(std::numeric_limits<double>::infinity(),
                                 {{-std::numeric_limits<double>::infinity()
                                    *Vector3<double>::Ones(),
                                    std::numeric_limits<double>::infinity()
                                    *Vector3<double>::Ones()}}) {
    DRAKE_DEMAND(!(normal(0) == 0 && normal(1) == 0 && normal(2) == 0));
    normal_ = normal.normalized();
}

bool HalfSpaceLevelSet::InInterior(const Vector3<double>& position) const {
    return (position.dot(normal_) <= 0.0);
}

Vector3<double> HalfSpaceLevelSet::Normal(const Vector3<double>& position)
                                                                        const {
    if (InInterior(position)) {
        return normal_;
    } else {
        throw
            std::logic_error("Normal outside of the level set is unavailable");
    }
}

SphereLevelSet::SphereLevelSet(double radius):
                    AnalyticLevelSet(4.0/3.0*M_PI*radius*radius*radius,
                                    {{-radius*Vector3<double>::Ones(),
                                       radius*Vector3<double>::Ones()}}),
                    radius_(radius) {
    DRAKE_DEMAND(radius > 0);
}

bool SphereLevelSet::InInterior(const Vector3<double>& position) const {
    return position.norm() <= radius_;
}

Vector3<double> SphereLevelSet::Normal(const Vector3<double>& position) const {
    if (InInterior(position)) {
        if (position.norm() == 0) {
            return Vector3<double>{1.0, 0.0, 0.0};
        }
        return position.normalized();
    } else {
        throw
            std::logic_error("Normal outside of the level set is unavailable");
    }
}

BoxLevelSet::BoxLevelSet(const Vector3<double>& xscale):
                    AnalyticLevelSet(8*xscale(0)*xscale(1)*xscale(2),
                                    {{-xscale, xscale}}),
                    xscale_(xscale) {
    DRAKE_ASSERT(xscale_(0) > 0);
    DRAKE_ASSERT(xscale_(1) > 0);
    DRAKE_ASSERT(xscale_(2) > 0);
}

bool BoxLevelSet::InInterior(const Vector3<double>& position) const {
    return ((std::abs(position(0)) <= xscale_(0))
         && (std::abs(position(1)) <= xscale_(1))
         && (std::abs(position(2)) <= xscale_(2)));
}

Vector3<double> BoxLevelSet::Normal(const Vector3<double>& position) const {
    if (InInterior(position)) {
        double dist_left   = (position(0) + xscale_(0));
        double dist_right  = (xscale_(0) - position(0));
        double dist_front  = (position(1) + xscale_(1));
        double dist_back   = (xscale_(1) - position(1));
        double dist_bottom = (position(2) + xscale_(2));
        double dist_top    = (xscale_(2) - position(2));
        double min_dist    = std::min({dist_left, dist_right,
                                       dist_front, dist_back,
                                       dist_bottom, dist_top});
        if (min_dist == dist_left)
                            { return Vector3<double>{-1.0, 0.0, 0.0}; }
        if (min_dist == dist_right)
                            { return Vector3<double>{1.0, 0.0,  0.0}; }
        if (min_dist == dist_front)
                            { return Vector3<double>{0.0, -1.0, 0.0}; }
        if (min_dist == dist_back)
                            { return Vector3<double>{0.0,  1.0, 0.0}; }
        if (min_dist == dist_bottom)
                            { return Vector3<double>{0.0, 0.0, -1.0}; }
        if (min_dist == dist_top)
                            { return Vector3<double>{0.0, 0.0, 1.0}; }
        DRAKE_UNREACHABLE();
    } else {
        throw
            std::logic_error("Normal outside of the level set is unavailable");
    }
}

CylinderLevelSet::CylinderLevelSet(double height, double radius):
                            AnalyticLevelSet(2.0*M_PI*radius*radius*height,
                                            {{{-radius, -radius, -height},
                                              { radius,  radius,  height}}}),
                                            height_(height), radius_(radius) {
    DRAKE_DEMAND(height > 0);
    DRAKE_DEMAND(radius > 0);
}

bool CylinderLevelSet::InInterior(const Vector3<double>& position) const {
    return (((position(0)*position(0) + position(1)*position(1))
                                                            <= radius_*radius_)
         && (std::abs(position(2)) <= height_));
}

Vector3<double> CylinderLevelSet::Normal(const Vector3<double>& position)
                                                                        const {
    if (InInterior(position)) {
        Vector3<double> projection_xy(position(0), position(1), 0.0);
        if (projection_xy.norm() == 0) {
            return Vector3<double>{1.0, 0.0, 0.0};
        }
        return projection_xy.normalized();
    } else {
        throw
            std::logic_error("Normal outside of the level set is unavailable");
    }
}

}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
