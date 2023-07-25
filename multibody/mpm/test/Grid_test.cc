#include "drake/multibody/fem/mpm-dev/Grid.h"

#include <cmath>
#include <memory>
#include <utility>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/geometry/proximity/posed_half_space.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/multibody/fem/mpm-dev/AnalyticLevelSet.h"
#include "drake/multibody/fem/mpm-dev/SpatialVelocityTimeDependent.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace internal {
namespace {

constexpr double kEps = 4.0 * std::numeric_limits<double>::epsilon();

GTEST_TEST(GridClassTest, TestSetGet) {
    Vector3<int> num_gridpt_1D = {6, 3, 4};
    double h = 1.0;
    Vector3<int> bottom_corner  = {0, 0, 0};
    Grid grid = Grid(num_gridpt_1D, h, bottom_corner);
    double tmpscaling = 1.0;
    // Test the geometry of the grid
    EXPECT_EQ(grid.get_num_gridpt(), 72);
    EXPECT_TRUE(CompareMatrices(grid.get_num_gridpt_1D(),
                                Vector3<int>(6, 3, 4)));
    EXPECT_EQ(grid.get_h(), 1.0);
    EXPECT_TRUE(CompareMatrices(grid.get_bottom_corner(), bottom_corner));

    // Check whether the grid point positions are populated correctly
    for (int k = 0; k < 4; ++k) {
    for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 6; ++i) {
        EXPECT_TRUE(CompareMatrices(grid.get_position(i, j, k),
                                    Vector3<double>(i, j, k)));
        // Randomly put some values in
        tmpscaling = 20.0*k + 10.0*j + i;
        grid.set_mass(i, j, k, tmpscaling);
        grid.set_velocity(i, j, k, Vector3<double>(tmpscaling,
                                                  -tmpscaling,
                                                   tmpscaling));
        grid.set_force(i, j, k, Vector3<double>(-tmpscaling,
                                                 tmpscaling,
                                                -tmpscaling));
    }
    }
    }

    EXPECT_TRUE(CompareMatrices(grid.get_velocity(1, 1, 1),
                                Vector3<double>(31.0, -31.0, 31.0)));
    EXPECT_TRUE(CompareMatrices(grid.get_velocity(grid.Reduce3DIndex(1, 1, 1)),
                                Vector3<double>(31.0, -31.0, 31.0)));
    EXPECT_TRUE(CompareMatrices(grid.get_force(1, 1, 1),
                                Vector3<double>(-31.0, 31.0, -31.0)));
    EXPECT_TRUE(CompareMatrices(grid.get_force(grid.Reduce3DIndex(1, 1, 1)),
                                Vector3<double>(-31.0, 31.0, -31.0)));
    EXPECT_EQ(grid.get_mass(1, 1, 1), 31.0);
    EXPECT_EQ(grid.get_mass(grid.Reduce3DIndex(1, 1, 1)), 31.0);
    EXPECT_TRUE(CompareMatrices(grid.get_velocity(4, 2, 2),
                                Vector3<double>(64.0, -64.0, 64.0)));
    EXPECT_TRUE(CompareMatrices(grid.get_velocity(grid.Reduce3DIndex(4, 2, 2)),
                                Vector3<double>(64.0, -64.0, 64.0)));
    EXPECT_TRUE(CompareMatrices(grid.get_force(4, 2, 2),
                                Vector3<double>(-64.0, 64.0, -64.0)));
    EXPECT_TRUE(CompareMatrices(grid.get_force(grid.Reduce3DIndex(4, 2, 2)),
                                Vector3<double>(-64.0, 64.0, -64.0)));
    EXPECT_EQ(grid.get_mass(4, 2, 2), 64.0);
    EXPECT_EQ(grid.get_mass(grid.Reduce3DIndex(4, 2, 2)), 64.0);

    // Test on a new grid
    num_gridpt_1D = {3, 3, 3};
    h = 0.5;
    bottom_corner  = {-2, 2, -2};
    grid = Grid(num_gridpt_1D, h, bottom_corner);

    // Test the geometry of the grid
    EXPECT_EQ(grid.get_num_gridpt(), 27);
    EXPECT_TRUE(CompareMatrices(grid.get_num_gridpt_1D(),
                                Vector3<int>(3, 3, 3)));
    EXPECT_EQ(grid.get_h(), 0.5);
    EXPECT_TRUE(CompareMatrices(grid.get_bottom_corner(), bottom_corner));

    // Check whether the grid point positions are populated correctly
    EXPECT_TRUE(CompareMatrices(grid.get_position(-2, 2, -2),
                                Vector3<double>(-1.0, 1.0, -1.0)));
    EXPECT_TRUE(CompareMatrices(grid.get_position(-1, 2, -2),
                                Vector3<double>(-0.5, 1.0, -1.0)));
    EXPECT_TRUE(CompareMatrices(grid.get_position(-2, 3, -2),
                                Vector3<double>(-1.0, 1.5, -1.0)));
    EXPECT_TRUE(CompareMatrices(grid.get_position(-1, 3, -2),
                                Vector3<double>(-0.5, 1.5, -1.0)));
    EXPECT_TRUE(CompareMatrices(grid.get_position(-2, 2, -1),
                                Vector3<double>(-1.0, 1.0, -0.5)));
    EXPECT_TRUE(CompareMatrices(grid.get_position(-1, 2, -1),
                                Vector3<double>(-0.5, 1.0, -0.5)));
    EXPECT_TRUE(CompareMatrices(grid.get_position(-2, 3, -1),
                                Vector3<double>(-1.0, 1.5, -0.5)));
    EXPECT_TRUE(CompareMatrices(grid.get_position(-1, 3, -1),
                                Vector3<double>(-0.5, 1.5, -0.5)));
    EXPECT_TRUE(CompareMatrices(grid.get_position(0, 4, 0),
                                Vector3<double>(0.0, 2.0, 0.0)));
}

GTEST_TEST(GridClassTest, TestExpand1DIndex) {
    int count;
    Vector3<int> num_gridpt_1D = {6, 3, 4};
    double h = 1.0;
    Vector3<int> bottom_corner  = {0, 0, 0};
    Grid grid = Grid(num_gridpt_1D, h, bottom_corner);

    // Check expand 1D index
    count = 0;
    for (int k = 0; k < 4; ++k) {
    for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 6; ++i) {
        EXPECT_TRUE(CompareMatrices(grid.Expand1DIndex(count++),
                                    Vector3<int>(i, j, k)));
    }
    }
    }

    // Test on a new grid
    num_gridpt_1D = {3, 3, 3};
    h = 0.5;
    bottom_corner  = {-2, 2, -2};
    grid = Grid(num_gridpt_1D, h, bottom_corner);

    // Check expand 1D index
    count = 0;
    for (int k = bottom_corner(2); k < bottom_corner(2)+num_gridpt_1D(2); ++k) {
    for (int j = bottom_corner(1); j < bottom_corner(1)+num_gridpt_1D(1); ++j) {
    for (int i = bottom_corner(0); i < bottom_corner(0)+num_gridpt_1D(0); ++i) {
        EXPECT_TRUE(CompareMatrices(grid.Expand1DIndex(count++),
                                    Vector3<int>(i, j, k)));
    }
    }
    }
}

GTEST_TEST(GridClassTest, TestGetIndices) {
    int count;
    Vector3<int> num_gridpt_1D = {6, 3, 4};
    double h = 1.0;
    Vector3<int> bottom_corner  = {0, 0, 0};
    Grid grid = Grid(num_gridpt_1D, h, bottom_corner);

    // Check expand 1D index
    count = 0;
    for (const auto& [index_flat, index_3d] : grid.get_indices()) {
        EXPECT_EQ(count++, index_flat);
        EXPECT_TRUE(CompareMatrices(index_3d,
                                    grid.Expand1DIndex(index_flat)));
    }
}

GTEST_TEST(GridClassTest, TestResetStatesAndAccumulationAndRescale) {
    Vector3<int> num_gridpt_1D = {6, 3, 4};
    double h = 1.0;
    Vector3<int> bottom_corner  = {0, 0, 0};
    Grid grid = Grid(num_gridpt_1D, h, bottom_corner);
    double tmpscaling = 1.0;
    // Test the geometry of the grid
    EXPECT_EQ(grid.get_num_gridpt(), 72);
    EXPECT_TRUE(CompareMatrices(grid.get_num_gridpt_1D(),
                                Vector3<int>(6, 3, 4)));
    EXPECT_EQ(grid.get_h(), 1.0);
    EXPECT_TRUE(CompareMatrices(grid.get_bottom_corner(), bottom_corner));

    for (int k = 0; k < 4; ++k) {
    for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 6; ++i) {
        // Randomly put some values in
        tmpscaling = 1.2*k + 0.3*j + i;
        grid.set_mass(i, j, k, tmpscaling);
        grid.set_velocity(i, j, k, Vector3<double>(tmpscaling,
                                                  -tmpscaling,
                                                   tmpscaling));
        grid.set_force(i, j, k, Vector3<double>(-tmpscaling,
                                                 tmpscaling,
                                                -tmpscaling));
    }
    }
    }

    for (int k = 1; k < 4; ++k) {
    for (int j = 1; j < 3; ++j) {
    for (int i = 1; i < 6; ++i) {
        EXPECT_TRUE(!CompareMatrices(grid.get_velocity(i, j, k),
                                     Vector3<double>::Zero()));
    }
    }
    }

    grid.ResetStates();

    // Test ResetStates sets all states to zero
    for (int k = 0; k < 4; ++k) {
    for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 6; ++i) {
        EXPECT_TRUE(CompareMatrices(grid.get_velocity(i, j, k),
                                    Vector3<double>::Zero()));
        EXPECT_TRUE(CompareMatrices(grid.get_force(i, j, k),
                                    Vector3<double>::Zero()));
        EXPECT_EQ(grid.get_mass(i, j, k), 0.0);
    }
    }
    }

    // Test accumulation
    Vector3<int> grid_index;
    for (int k = 0; k < 4; ++k) {
    for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 6; ++i) {
        grid_index(0) = i;
        grid_index(1) = j;
        grid_index(2) = k;
        // Randomly put some values in
        grid.AccumulateMass(i, j, k, 1.0);
        grid.AccumulateVelocity(i, j, k, Vector3<double>(1.0, -1.0, 1.0));
        grid.AccumulateForce(i, j, k, Vector3<double>(-1.0, -1.0, 1.0));
        EXPECT_TRUE(CompareMatrices(grid.get_velocity(i, j, k),
                                    Vector3<double>(1.0, -1.0, 1.0), kEps));
        EXPECT_TRUE(CompareMatrices(grid.get_force(i, j, k),
                                    Vector3<double>(-1.0, -1.0, 1.0), kEps));
        EXPECT_EQ(grid.get_mass(i, j, k), 1.0);

        grid.AccumulateMass(grid_index, 1.2);
        grid.AccumulateVelocity(grid_index, Vector3<double>(1.2, -1.2, 1.2));
        grid.AccumulateForce(grid_index, Vector3<double>(-1.2, -1.2, 1.2));
        EXPECT_TRUE(CompareMatrices(grid.get_velocity(i, j, k),
                                    Vector3<double>(2.2, -2.2, 2.2), kEps));
        EXPECT_TRUE(CompareMatrices(grid.get_force(i, j, k),
                                    Vector3<double>(-2.2, -2.2, 2.2), kEps));
        EXPECT_EQ(grid.get_mass(i, j, k), 2.2);
    }
    }
    }

    // Test rescale
    grid.RescaleVelocities();
    for (int k = 0; k < 4; ++k) {
    for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 6; ++i) {
        // Randomly put some values in
        EXPECT_TRUE(CompareMatrices(grid.get_velocity(i, j, k),
                                    Vector3<double>(1.0, -1.0, 1.0), kEps));
        EXPECT_TRUE(CompareMatrices(grid.get_force(i, j, k),
                                    Vector3<double>(-2.2, -2.2, 2.2), kEps));
        EXPECT_EQ(grid.get_mass(i, j, k), 2.2);
    }
    }
    }
}

GTEST_TEST(GridClassTest, TestUpdateVelocity) {
    Vector3<int> num_gridpt_1D = {6, 3, 4};
    double h = 1.0;
    Vector3<int> bottom_corner  = {0, 0, 0};
    Grid grid = Grid(num_gridpt_1D, h, bottom_corner);
    double tmpscaling = 1.0;
    double dt = 0.2;

    for (int k = 0; k < 4; ++k) {
    for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 6; ++i) {
        // Randomly put some values in
        tmpscaling = 1.2*k + 0.3*j + i;
        grid.set_mass(i, j, k, tmpscaling);
        grid.set_velocity(i, j, k, Vector3<double>(tmpscaling,
                                                  -tmpscaling,
                                                   tmpscaling));
        grid.set_force(i, j, k, Vector3<double>(-tmpscaling,
                                                 tmpscaling,
                                                -tmpscaling));
    }
    }
    }

    grid.UpdateVelocity(dt);

    // v^{n+1} = v^n + dt*f^n/m = [t, -t, t] + dt*[-t, t, -t]/t
    //                          = [t, -t, t] + [-dt, dt, -dt]
    //                          = [t-dt, -t+dt, t-dt]
    // t: tmpscaling
    for (int k = 1; k < 4; ++k) {
    for (int j = 1; j < 3; ++j) {
    for (int i = 1; i < 6; ++i) {
        tmpscaling = 1.2*k + 0.3*j + i;
        EXPECT_TRUE(CompareMatrices(grid.get_velocity(i, j, k),
            Vector3<double>(tmpscaling-dt, -tmpscaling+dt, tmpscaling-dt),
            kEps));
    }
    }
    }
}

GTEST_TEST(GridClassTest, TestGridSumState) {
    // In this test case, we construct a 2x2x2 grid
    Vector3<int> num_gridpt_1D = {2, 2, 2};
    double h = 1.0;
    Vector3<int> bottom_corner  = {0, 0, 0};
    Grid grid = Grid(num_gridpt_1D, h, bottom_corner);

    // Initialize the mass and velocity of the grid with dummy masses
    //    7 o -- o 8
    //  5  /| 6 /|
    //    o -- o |
    //    | o 3| o 4
    //    o -- o
    //    1    2
    double dummy_m;
    Vector3<double> dummy_v = {1.0, 1.0, 1.0};
    int pc = 0;
    for (int k = bottom_corner(2);
                k < bottom_corner(2)+num_gridpt_1D(2); ++k) {
    for (int j = bottom_corner(1);
                j < bottom_corner(1)+num_gridpt_1D(1); ++j) {
    for (int i = bottom_corner(0);
                i < bottom_corner(0)+num_gridpt_1D(0); ++i) {
        dummy_m = ++pc;
        grid.set_velocity(i, j, k, dummy_v);
        grid.set_mass(i, j, k, dummy_m);
    }
    }
    }

    TotalMassAndMomentum sum_state = grid.GetTotalMassAndMomentum();
    // Sum of mass shall be ∑ i, i = 1 ... 8 = 36, as a result
    // Sum of momentum shall be (36, 36, 36)
    // mi ∑ xi.cross(vi) = (∑ mi xi).cross(vi) = (20 22 26) cross (1, 1, 1)
    // because vi constant, and cross product is a bilinear operator.
    EXPECT_EQ(sum_state.sum_mass, 36.0);
    EXPECT_TRUE(CompareMatrices(sum_state.sum_momentum,
                                Vector3<double>(36.0, 36.0, 36.0), kEps));
    EXPECT_TRUE(CompareMatrices(sum_state.sum_angular_momentum,
                                Vector3<double>(-4.0, 6.0, -2.0), kEps));
}

GTEST_TEST(GridClassTest, TestWallBoundaryConditionWithHalfSpace) {
    // In this test case, we enforce slip wall boundary condition to the
    // boundary of a 10x20x30 grid.
    Vector3<int> num_gridpt_1D = {10, 20, 30};
    double h = 1.0;
    Vector3<int> bottom_corner  = {0, 0, 0};
    Grid grid = Grid(num_gridpt_1D, h, bottom_corner);
    // Velocity of all grid points in the domain
    Vector3<double> velocity_grid = {1.0, 1.0, 1.0};
    // We assume an ideal slip boundary condition
    double mu = 0.0;
    KinematicCollisionObjects objects = KinematicCollisionObjects();

    // Initialize the "wall" by halfspaces with suitable normals
    multibody::SpatialVelocity<double> zero_velocity;
    zero_velocity.SetZero();
    std::unique_ptr<SpatialVelocityTimeDependent> zero_velocity_ptr
            = std::make_unique<SpatialVelocityTimeDependent>(zero_velocity);
    Vector3<double> right_wall_n  = {-1.0,  0.0,  0.0};
    Vector3<double> left_wall_n   = { 1.0,  0.0,  0.0};
    Vector3<double> front_wall_n  = { 0.0,  1.0,  0.0};
    Vector3<double> back_wall_n   = { 0.0, -1.0,  0.0};
    Vector3<double> top_wall_n    = { 0.0,  0.0, -1.0};
    Vector3<double> bottom_wall_n = { 0.0,  0.0,  1.0};
    std::unique_ptr<AnalyticLevelSet> right_wall_level_set =
                            std::make_unique<HalfSpaceLevelSet>(right_wall_n);
    std::unique_ptr<AnalyticLevelSet> left_wall_level_set =
                            std::make_unique<HalfSpaceLevelSet>(left_wall_n);
    std::unique_ptr<AnalyticLevelSet> front_wall_level_set =
                            std::make_unique<HalfSpaceLevelSet>(front_wall_n);
    std::unique_ptr<AnalyticLevelSet> back_wall_level_set =
                            std::make_unique<HalfSpaceLevelSet>(back_wall_n);
    std::unique_ptr<AnalyticLevelSet> top_wall_level_set =
                            std::make_unique<HalfSpaceLevelSet>(top_wall_n);
    std::unique_ptr<AnalyticLevelSet> bottom_wall_level_set =
                            std::make_unique<HalfSpaceLevelSet>(bottom_wall_n);
    Vector3<double> right_wall_translation  = {9.0,  0.0,  0.0};
    Vector3<double> left_wall_translation   = {0.0,  0.0,  0.0};
    Vector3<double> front_wall_translation  = {0.0,  0.0,  0.0};
    Vector3<double> back_wall_translation   = {0.0, 19.0,  0.0};
    Vector3<double> top_wall_translation    = {0.0,  0.0, 29.0};
    Vector3<double> bottom_wall_translation = {0.0,  0.0,  0.0};
    math::RigidTransform<double> right_wall_pose =
                        math::RigidTransform<double>(right_wall_translation);
    math::RigidTransform<double> left_wall_pose =
                        math::RigidTransform<double>(left_wall_translation);
    math::RigidTransform<double> front_wall_pose =
                        math::RigidTransform<double>(front_wall_translation);
    math::RigidTransform<double> back_wall_pose =
                        math::RigidTransform<double>(back_wall_translation);
    math::RigidTransform<double> top_wall_pose =
                        math::RigidTransform<double>(top_wall_translation);
    math::RigidTransform<double> bottom_wall_pose =
                        math::RigidTransform<double>(bottom_wall_translation);

    objects.AddCollisionObject(std::move(right_wall_level_set),
                               std::move(right_wall_pose),
                               std::move(zero_velocity_ptr->Clone()), mu);
    objects.AddCollisionObject(std::move(left_wall_level_set),
                               std::move(left_wall_pose),
                               std::move(zero_velocity_ptr->Clone()), mu);
    objects.AddCollisionObject(std::move(front_wall_level_set),
                               std::move(front_wall_pose),
                               std::move(zero_velocity_ptr->Clone()), mu);
    objects.AddCollisionObject(std::move(back_wall_level_set),
                               std::move(back_wall_pose),
                               std::move(zero_velocity_ptr->Clone()), mu);
    objects.AddCollisionObject(std::move(top_wall_level_set),
                               std::move(top_wall_pose),
                               std::move(zero_velocity_ptr->Clone()), mu);
    objects.AddCollisionObject(std::move(bottom_wall_level_set),
                               std::move(bottom_wall_pose),
                               std::move(zero_velocity_ptr->Clone()), mu);

    // Populate the grid with nonzero velocities, nonzero mass
    double dummy_mass = 1.0;
    for (int k = bottom_corner(2); k < bottom_corner(2)+num_gridpt_1D(2); ++k) {
    for (int j = bottom_corner(1); j < bottom_corner(1)+num_gridpt_1D(1); ++j) {
    for (int i = bottom_corner(0); i < bottom_corner(0)+num_gridpt_1D(0); ++i) {
        grid.set_mass(i, j, k, dummy_mass);
        grid.set_velocity(i, j, k, velocity_grid);
        EXPECT_TRUE(!grid.get_velocity(i, j, k).isZero());
    }
    }
    }

    // Enforce slip BC
    grid.EnforceBoundaryCondition(objects);

    // Check velocity after enforcement, hardcode values for verification
    for (int k = bottom_corner(2); k < bottom_corner(2)+num_gridpt_1D(2); ++k) {
    for (int j = bottom_corner(1); j < bottom_corner(1)+num_gridpt_1D(1); ++j) {
    for (int i = bottom_corner(0); i < bottom_corner(0)+num_gridpt_1D(0); ++i) {
        const Vector3<double>& velocity_i = grid.get_velocity(i, j, k);
        bool on_right_wall = i == bottom_corner(0)+num_gridpt_1D(0)-1;
        bool on_back_wall  = j == bottom_corner(1)+num_gridpt_1D(1)-1;
        bool on_top_wall   = k == bottom_corner(2)+num_gridpt_1D(2)-1;
        // If on the right, top, or back wall, the velocity will hit the wall,
        // so boundary condition is enforced.
        // Right wall
        if (on_right_wall) {
            EXPECT_NEAR(velocity_i(0), 0, kEps);
        }
        // Back wall
        if (on_back_wall) {
            EXPECT_NEAR(velocity_i(1), 0, kEps);
        }
        // Top wall
        if (on_top_wall) {
            EXPECT_NEAR(velocity_i(2), 0, kEps);
        }

        // If on the left, bottom, or front wall, the velocity is in the same
        // direction as wall's outward normal, so no boundary condition shall
        // be enforced. (We ignore the edges and corners with other three walls,
        // since the behavior at singularities is not well-defined)
        if (!on_right_wall && !on_back_wall && !on_top_wall) {
            if (i == bottom_corner(0)) {
                EXPECT_TRUE(CompareMatrices(velocity_i, velocity_grid, kEps));
            }
            if (j == bottom_corner(1)) {
                EXPECT_TRUE(CompareMatrices(velocity_i, velocity_grid, kEps));
            }
            if (k == bottom_corner(2)) {
                EXPECT_TRUE(CompareMatrices(velocity_i, velocity_grid, kEps));
            }
        }
    }
    }
    }
}

GTEST_TEST(GridClassTest, TestWallBoundaryConditionWithBox) {
    // In this test case, we enforce slip wall boundary condition to the
    // boundary of a 10x20x30 grid.
    Vector3<int> num_gridpt_1D = {10, 20, 30};
    double h = 1.0;
    Vector3<int> bottom_corner  = {0, 0, 0};
    Grid grid = Grid(num_gridpt_1D, h, bottom_corner);
    // Velocity of all grid points in the domain
    Vector3<double> velocity_grid = Vector3<double>(-1.0, -1.0, -1.0);
    // We assume an ideal slip boundary condition
    double mu = 0.0;
    KinematicCollisionObjects objects = KinematicCollisionObjects();

    // Initialize the "wall" by boxes with suitable sizes
    multibody::SpatialVelocity<double> zero_velocity;
    zero_velocity.SetZero();
    std::unique_ptr<SpatialVelocityTimeDependent> zero_velocity_ptr
            = std::make_unique<SpatialVelocityTimeDependent>(zero_velocity);
    Vector3<double> right_wall_scale  = {1.0, 10.0, 15.0};
    Vector3<double> left_wall_scale   = {1.0, 10.0, 15.0};
    Vector3<double> front_wall_scale  = {5.0,  1.0, 15.0};
    Vector3<double> back_wall_scale   = {5.0,  1.0, 15.0};
    Vector3<double> top_wall_scale    = {5.0, 10.0, 1.0};
    Vector3<double> bottom_wall_scale = {5.0, 10.0, 1.0};
    std::unique_ptr<AnalyticLevelSet> right_wall_level_set =
                            std::make_unique<BoxLevelSet>(right_wall_scale);
    std::unique_ptr<AnalyticLevelSet> left_wall_level_set =
                            std::make_unique<BoxLevelSet>(left_wall_scale);
    std::unique_ptr<AnalyticLevelSet> front_wall_level_set =
                            std::make_unique<BoxLevelSet>(front_wall_scale);
    std::unique_ptr<AnalyticLevelSet> back_wall_level_set =
                            std::make_unique<BoxLevelSet>(back_wall_scale);
    std::unique_ptr<AnalyticLevelSet> top_wall_level_set =
                            std::make_unique<BoxLevelSet>(top_wall_scale);
    std::unique_ptr<AnalyticLevelSet> bottom_wall_level_set =
                            std::make_unique<BoxLevelSet>(bottom_wall_scale);
    Vector3<double> right_wall_translation  = {-1.0,  9.5, 14.5};
    Vector3<double> left_wall_translation   = {10.0,  9.5, 14.5};
    Vector3<double> front_wall_translation  = { 4.5, -1.0, 14.5};
    Vector3<double> back_wall_translation   = { 4.5, 20.0, 14.5};
    Vector3<double> top_wall_translation    = { 4.5,  9.5, -1.0};
    Vector3<double> bottom_wall_translation = { 4.5,  9.5, 30.0};
    math::RigidTransform<double> right_wall_pose =
                        math::RigidTransform<double>(right_wall_translation);
    math::RigidTransform<double> left_wall_pose =
                        math::RigidTransform<double>(left_wall_translation);
    math::RigidTransform<double> front_wall_pose =
                        math::RigidTransform<double>(front_wall_translation);
    math::RigidTransform<double> back_wall_pose =
                        math::RigidTransform<double>(back_wall_translation);
    math::RigidTransform<double> top_wall_pose =
                        math::RigidTransform<double>(top_wall_translation);
    math::RigidTransform<double> bottom_wall_pose =
                        math::RigidTransform<double>(bottom_wall_translation);

    objects.AddCollisionObject(std::move(right_wall_level_set),
                               std::move(right_wall_pose),
                               std::move(zero_velocity_ptr->Clone()), mu);
    objects.AddCollisionObject(std::move(left_wall_level_set),
                               std::move(left_wall_pose),
                               std::move(zero_velocity_ptr->Clone()), mu);
    objects.AddCollisionObject(std::move(front_wall_level_set),
                               std::move(front_wall_pose),
                               std::move(zero_velocity_ptr->Clone()), mu);
    objects.AddCollisionObject(std::move(back_wall_level_set),
                               std::move(back_wall_pose),
                               std::move(zero_velocity_ptr->Clone()), mu);
    objects.AddCollisionObject(std::move(top_wall_level_set),
                               std::move(top_wall_pose),
                               std::move(zero_velocity_ptr->Clone()), mu);
    objects.AddCollisionObject(std::move(bottom_wall_level_set),
                               std::move(bottom_wall_pose),
                               std::move(zero_velocity_ptr->Clone()), mu);

    // Populate the grid with nonzero velocities
    double dummy_mass = 1.0;
    for (int k = bottom_corner(2); k < bottom_corner(2)+num_gridpt_1D(2); ++k) {
    for (int j = bottom_corner(1); j < bottom_corner(1)+num_gridpt_1D(1); ++j) {
    for (int i = bottom_corner(0); i < bottom_corner(0)+num_gridpt_1D(0); ++i) {
        grid.set_mass(i, j, k, dummy_mass);
        grid.set_velocity(i, j, k, velocity_grid);
        EXPECT_TRUE(!grid.get_velocity(i, j, k).isZero());
    }
    }
    }

    // Enforce slip BC
    grid.EnforceBoundaryCondition(objects);

    // Check velocity after enforcement, hardcode values for verification
    for (int k = bottom_corner(2); k < bottom_corner(2)+num_gridpt_1D(2); ++k) {
    for (int j = bottom_corner(1); j < bottom_corner(1)+num_gridpt_1D(1); ++j) {
    for (int i = bottom_corner(0); i < bottom_corner(0)+num_gridpt_1D(0); ++i) {
        const Vector3<double>& velocity_i = grid.get_velocity(i, j, k);
        bool on_left_wall   = i == bottom_corner(0);
        bool on_front_wall  = j == bottom_corner(1);
        bool on_bottom_wall = k == bottom_corner(2);
        // If on the right, top, or back wall, the velocity will hit the wall,
        // so boundary condition is enforced.
        // Left wall
        if (on_left_wall) {
            EXPECT_NEAR(velocity_i(0), 0, kEps);
        }
        // Front wall
        if (on_front_wall) {
            EXPECT_NEAR(velocity_i(1), 0, kEps);
        }
        // Bottom wall
        if (on_bottom_wall) {
            EXPECT_NEAR(velocity_i(2), 0, kEps);
        }

        // If on the right, back, or top wall, the velocity is in the same
        // direction as wall's outward normal, so no boundary condition shall
        // be enforced. (We ignore the edges and corners with other three walls,
        // since the behavior at singularities is not well-defined)
        if (!on_left_wall && !on_front_wall && !on_bottom_wall) {
            if (i == bottom_corner(0)+num_gridpt_1D(0)-1) {
                EXPECT_TRUE(CompareMatrices(velocity_i, velocity_grid, kEps));
            }
            if (j == bottom_corner(1)+num_gridpt_1D(1)-1) {
                EXPECT_TRUE(CompareMatrices(velocity_i, velocity_grid, kEps));
            }
            if (k == bottom_corner(2)+num_gridpt_1D(2)-1) {
                EXPECT_TRUE(CompareMatrices(velocity_i, velocity_grid, kEps));
            }
        }
    }
    }
    }
}

GTEST_TEST(GridClassTest, TestRotatedPlaneBC) {
    // In this test case, we enforce wall boundary condition to a 21x21x21 grid
    // of the domain [0, 10]x[0, 10]x[0, 10]. The boundary plane has normal (1,
    // 1, 1), contains the point (5, 5, 5).
    Vector3<int> num_gridpt_1D = {21, 21, 21};
    double h = 0.5;
    Vector3<int> bottom_corner  = {0, 0, 0};
    Vector3<double> velocity_grid = Vector3<double>(-1.0, -2.0, -3.0);
    Grid grid = Grid(num_gridpt_1D, h, bottom_corner);
    // Friction coefficient
    double mu = 0.05;

    // Initialize the boundary spaces for testing
    geometry::internal::PosedHalfSpace<double> wall_half_space =
                                                    {{1, 1, 1}, {5, 5, 5}};

    // Initialize the collision object
    KinematicCollisionObjects objects = KinematicCollisionObjects();

    multibody::SpatialVelocity<double> zero_velocity;
    zero_velocity.SetZero();
    std::unique_ptr<SpatialVelocityTimeDependent> zero_velocity_ptr
            = std::make_unique<SpatialVelocityTimeDependent>(zero_velocity);
    Vector3<double> wall_normal = {1.0, 1.0, 1.0};
    std::unique_ptr<AnalyticLevelSet> wall_level_set =
                            std::make_unique<HalfSpaceLevelSet>(wall_normal);
    Vector3<double> wall_translation  = {5.0, 5.0, 5.0};
    math::RigidTransform<double> wall_pose =
                        math::RigidTransform<double>(wall_translation);
    objects.AddCollisionObject(std::move(wall_level_set), std::move(wall_pose),
                               std::move(zero_velocity_ptr), mu);

    // Populate the grid with nonzero velocities
    double dummy_mass = 1.0;
    for (int k = bottom_corner(2); k < bottom_corner(2)+num_gridpt_1D(2); ++k) {
    for (int j = bottom_corner(1); j < bottom_corner(1)+num_gridpt_1D(1); ++j) {
    for (int i = bottom_corner(0); i < bottom_corner(0)+num_gridpt_1D(0); ++i) {
        grid.set_mass(i, j, k, dummy_mass);
        grid.set_velocity(i, j, k, velocity_grid);
        EXPECT_TRUE(!grid.get_velocity(i, j, k).isZero());
    }
    }
    }

    // Enforce wall boundary condition
    grid.EnforceBoundaryCondition(objects);

    // Check velocity after enforcement, hardcode values for verification
    for (int k = bottom_corner(2); k < bottom_corner(2)+num_gridpt_1D(2); ++k) {
    for (int j = bottom_corner(1); j < bottom_corner(1)+num_gridpt_1D(1); ++j) {
    for (int i = bottom_corner(0); i < bottom_corner(0)+num_gridpt_1D(0); ++i) {
        const Vector3<double>& position_i = grid.get_position(i, j, k);
        const Vector3<double>& velocity_i = grid.get_velocity(i, j, k);
        double dist = wall_half_space.CalcSignedDistance(position_i);
        // If the grid point is not on/in the boundary, velocity shall be the
        // same
        if (dist > 1e-6) {
            EXPECT_TRUE(CompareMatrices(velocity_i,
                                    Vector3<double>{-1.0, -2.0, -3.0}, kEps));
        }
        // If the grid point is on/in the boundary, velocity shall be the same
        if (dist < -1e-6) {
            // Given v_i = (-1, -2, -3), n = 1/sqrt(3)*(-1, -1, -1)
            // vₙ = (v ⋅ n)n = (-2, -2, -2), vₜ = v - vₙ = (1, 0, -1)
            // v_new = vₜ - μ‖vₙ‖t = (1-0.05*sqrt(6))*(1, 0, -1)
            EXPECT_TRUE(CompareMatrices(velocity_i,
                            (1.0-0.05*sqrt(6))*Vector3<double>(1.0, 0.0, -1.0),
                            kEps));
        }
    }
    }
    }
}

GTEST_TEST(GridClassTest, TestMovingCylindricalBC) {
    // In this test case, we enforce frictional wall boundary condition to the
    // plane on a 11x11x11 grid of the domain [0, 10]x[0, 10]x[0, 10]. The
    // moving cylindrical boundary has central axis parallel to x-axis, centered
    // at (5.0, 5.0, 5.0), height 1.0, radius 0.5 and moving with velocity
    // (1.0, 1.0, 1.0)
    Vector3<int> num_gridpt_1D = {11, 11, 11};
    double h = 1.0;
    Vector3<int> bottom_corner  = {0, 0, 0};
    Vector3<double> velocity_grid = Vector3<double>(1.0, 2.0, 3.0);
    Grid grid = Grid(num_gridpt_1D, h, bottom_corner);

    // Initialize the collision object
    KinematicCollisionObjects objects = KinematicCollisionObjects();

    multibody::SpatialVelocity<double> cylinder_velocity;
    cylinder_velocity.translational() = Vector3<double>(1.0, 1.0, 1.0);
    cylinder_velocity.rotational() = Vector3<double>::Zero();
    std::unique_ptr<SpatialVelocityTimeDependent> cylinder_velocity_ptr
            = std::make_unique<SpatialVelocityTimeDependent>(cylinder_velocity);
    double cylinder_height = 1.0;
    double cylinder_radius = 0.5;
    double cylinder_mu     = 0.1;
    std::unique_ptr<AnalyticLevelSet> cylinder_level_set =
                            std::make_unique<CylinderLevelSet>(cylinder_height,
                                                               cylinder_radius);
    Vector3<double> cylinder_translation = {5.0, 5.0, 5.0};
    math::RollPitchYaw cylinder_rpw = {0.0, M_PI/2.0, 0.0};
    math::RigidTransform<double> cylinder_pose = {cylinder_rpw,
                                                  cylinder_translation};
    objects.AddCollisionObject(std::move(cylinder_level_set),
                               std::move(cylinder_pose),
                               std::move(cylinder_velocity_ptr), cylinder_mu);

    // Populate the grid with nonzero velocities
    double dummy_mass = 1.0;
    for (int k = bottom_corner(2); k < bottom_corner(2)+num_gridpt_1D(2); ++k) {
    for (int j = bottom_corner(1); j < bottom_corner(1)+num_gridpt_1D(1); ++j) {
    for (int i = bottom_corner(0); i < bottom_corner(0)+num_gridpt_1D(0); ++i) {
        grid.set_mass(i, j, k, dummy_mass);
        grid.set_velocity(i, j, k, velocity_grid);
        EXPECT_TRUE(!grid.get_velocity(i, j, k).isZero());
    }
    }
    }

    // Cylinder move t = 1.1, so the cylinder contains exactly two grid points
    // (6.0, 6.0, 6.0) and (7.0, 6.0, 6.0)
    double dt = 1.1;
    objects.AdvanceOneTimeStep(dt);
    // Enforce wall boundary condition
    grid.EnforceBoundaryCondition(objects);

    // Check velocity after enforcement, hardcode values for verification
    for (int k = bottom_corner(2); k < bottom_corner(2)+num_gridpt_1D(2); ++k) {
    for (int j = bottom_corner(1); j < bottom_corner(1)+num_gridpt_1D(1); ++j) {
    for (int i = bottom_corner(0); i < bottom_corner(0)+num_gridpt_1D(0); ++i) {
        const Vector3<double>& velocity_i = grid.get_velocity(i, j, k);
        if ((j == 6) && (k == 6) && ((i == 6) || (i == 7))) {
            // In this case, the relative velocity of the grid point is
            // (0, 1, 2), and the outward normal direction is
            // 1/sqrt(2)*(0, -1, -1). Then
            // vₙ = (v ⋅ n)n = (0, 3/2, 3/2), vₜ = v - vₙ = (0, -1/2, 1/2)
            // v_new = vₜ - μ‖vₙ‖t = (0, -1/2, 1/2) - 0.1*(0, -3, 3)/2
            // In physical frame, v_new = (1, 1, 1) + (0, -0.35, 0.35)
            EXPECT_TRUE(CompareMatrices(velocity_i,
                                    Vector3<double>{1.0, 0.65, 1.35}, kEps));
        } else {
            // If the grid point is not in the cylinder , velocity shall be the
            // same
            EXPECT_TRUE(CompareMatrices(velocity_i,
                                    Vector3<double>{1.0, 2.0, 3.0}, kEps));
        }
    }
    }
    }
}

}  // namespace
}  // namespace internal
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
