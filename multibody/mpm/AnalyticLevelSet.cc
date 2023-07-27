#include "drake/multibody/mpm/AnalyticLevelSet.h"

namespace drake {
namespace multibody {
namespace mpm {

template <typename T>
AnalyticLevelSet<T>::AnalyticLevelSet(T volume,
                            const std::array<Vector3<T>, 2>& bounding_box):
                                volume_(volume), bounding_box_(bounding_box) {}

template <typename T>
T AnalyticLevelSet<T>::get_volume() const {
    return volume_;
}

template <typename T>
const std::array<Vector3<T>, 2>& AnalyticLevelSet<T>::get_bounding_box()
                                                                        const {
    return bounding_box_;
}

template <typename T>
HalfSpaceLevelSet<T>::HalfSpaceLevelSet(const Vector3<T>& normal):
                AnalyticLevelSet<T>(std::numeric_limits<T>::infinity(),
                                 {{-std::numeric_limits<T>::infinity()
                                    *Vector3<T>::Ones(),
                                    std::numeric_limits<T>::infinity()
                                    *Vector3<T>::Ones()}}) {
    DRAKE_DEMAND(!(normal(0) == 0 && normal(1) == 0 && normal(2) == 0));
    normal_ = normal.normalized();
}

template <typename T>
bool HalfSpaceLevelSet<T>::InInterior(const Vector3<T>& position) const {
    return (position.dot(normal_) <= 0.0);
}

template <typename T>
Vector3<T> HalfSpaceLevelSet<T>::Normal(const Vector3<T>& position)
                                                                        const {
    if (InInterior(position)) {
        return normal_;
    } else {
        throw
            std::logic_error("Normal outside of the level set is unavailable");
    }
}

template <typename T>
SphereLevelSet<T>::SphereLevelSet(T radius):
                    AnalyticLevelSet<T>(4.0/3.0*M_PI*radius*radius*radius,
                                    {{-radius*Vector3<T>::Ones(),
                                       radius*Vector3<T>::Ones()}}),
                    radius_(radius) {
    DRAKE_DEMAND(radius > 0);
}

template <typename T>
bool SphereLevelSet<T>::InInterior(const Vector3<T>& position) const {
    return position.norm() <= radius_;
}

template <typename T>
Vector3<T> SphereLevelSet<T>::Normal(const Vector3<T>& position) const {
    if (InInterior(position)) {
        if (position.norm() == 0) {
            return Vector3<T>{1.0, 0.0, 0.0};
        }
        return position.normalized();
    } else {
        throw
            std::logic_error("Normal outside of the level set is unavailable");
    }
}

template <typename T>
BoxLevelSet<T>::BoxLevelSet(const Vector3<T>& xscale):
                    AnalyticLevelSet<T>(8*xscale(0)*xscale(1)*xscale(2),
                                    {{-xscale, xscale}}),
                    xscale_(xscale) {
    DRAKE_ASSERT(xscale_(0) > 0);
    DRAKE_ASSERT(xscale_(1) > 0);
    DRAKE_ASSERT(xscale_(2) > 0);
}

template <typename T>
bool BoxLevelSet<T>::InInterior(const Vector3<T>& position) const {
    using std::abs;
    return ((abs(position(0)) <= xscale_(0))
         && (abs(position(1)) <= xscale_(1))
         && (abs(position(2)) <= xscale_(2)));
}

template <typename T>
Vector3<T> BoxLevelSet<T>::Normal(const Vector3<T>& position) const {
    if (InInterior(position)) {
        using std::min;
        T dist_left   = (position(0) + xscale_(0));
        T dist_right  = (xscale_(0) - position(0));
        T dist_front  = (position(1) + xscale_(1));
        T dist_back   = (xscale_(1) - position(1));
        T dist_bottom = (position(2) + xscale_(2));
        T dist_top    = (xscale_(2) - position(2));
        T min_dist    = min({dist_left, dist_right,
                                       dist_front, dist_back,
                                       dist_bottom, dist_top});
        if (min_dist == dist_left)
                            { return Vector3<T>{-1.0, 0.0, 0.0}; }
        if (min_dist == dist_right)
                            { return Vector3<T>{1.0, 0.0,  0.0}; }
        if (min_dist == dist_front)
                            { return Vector3<T>{0.0, -1.0, 0.0}; }
        if (min_dist == dist_back)
                            { return Vector3<T>{0.0,  1.0, 0.0}; }
        if (min_dist == dist_bottom)
                            { return Vector3<T>{0.0, 0.0, -1.0}; }
        if (min_dist == dist_top)
                            { return Vector3<T>{0.0, 0.0, 1.0}; }
        DRAKE_UNREACHABLE();
    } else {
        throw
            std::logic_error("Normal outside of the level set is unavailable");
    }
}

template <typename T>
CylinderLevelSet<T>::CylinderLevelSet(T height, T radius):
                            AnalyticLevelSet<T>(2.0*M_PI*radius*radius*height,
                                            {{{-radius, -radius, -height},
                                              { radius,  radius,  height}}}),
                                            height_(height), radius_(radius) {
    DRAKE_DEMAND(height > 0);
    DRAKE_DEMAND(radius > 0);
}

template <typename T>
bool CylinderLevelSet<T>::InInterior(const Vector3<T>& position) const {
    using std::abs;
    return (((position(0)*position(0) + position(1)*position(1))
                                                            <= radius_*radius_)
         && (abs(position(2)) <= height_));
}

template <typename T>
Vector3<T> CylinderLevelSet<T>::Normal(const Vector3<T>& position)
                                                                        const {
    if (InInterior(position)) {
        Vector3<T> projection_xy(position(0), position(1), 0.0);
        if (projection_xy.norm() == 0) {
            return Vector3<T>{1.0, 0.0, 0.0};
        }
        return projection_xy.normalized();
    } else {
        throw
            std::logic_error("Normal outside of the level set is unavailable");
    }
}

template class AnalyticLevelSet<double>;
template class AnalyticLevelSet<AutoDiffXd>;

template class HalfSpaceLevelSet<double>;
template class HalfSpaceLevelSet<AutoDiffXd>;

template class SphereLevelSet<double>;
template class SphereLevelSet<AutoDiffXd>;

template class BoxLevelSet<double>;
template class BoxLevelSet<AutoDiffXd>;

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
