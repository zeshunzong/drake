#include <vector>

#include <gtest/gtest.h>

#include "drake/multibody/mpm/particles.h"
#include "drake/multibody/mpm/sparse_grid.h"

namespace drake {
namespace multibody {
namespace mpm {
namespace {

// we manully add five particles
// three of them should fall in the same batch, so a total of three distinct
// batches among the three batches, two of them have overlapping neighbor grid
// nodes
GTEST_TEST(SparseGridTest, TestParticlesAndGridSorting) {
  const double h = 1.0;
  SparseGrid<double> grid(h);

  Particles<double> particles = Particles<double>();

  Vector3<double> particle_A_position{2.1, 2.9, 3.9};
  // its batch index should be {2,3,4}

  Vector3<double> particle_B_position{2.9, 4.1, 5.1};
  // its batch index should be {3,4,5}

  Vector3<double> particle_C_position{-2.9, 0.7, -0.7};
  // its 3d batch index should be {-3, 1, -1}

  Vector3<double> particle_D_position{2.3, 3.2, 4.1};
  // its batch index should be {2,3,4}

  Vector3<double> particle_E_position{1.7, 3.1, 4.4};
  // its batch index should be {2,3,4}

  particles.AddParticle(particle_A_position, Vector3<double>::Zero(), 1.0, 1.0);
  particles.AddParticle(particle_B_position, Vector3<double>::Zero(), 2.0, 2.0);
  particles.AddParticle(particle_C_position, Vector3<double>::Zero(), 3.0, 3.0);
  particles.AddParticle(particle_D_position, Vector3<double>::Zero(), 4.0, 4.0);
  particles.AddParticle(particle_E_position, Vector3<double>::Zero(), 5.0, 5.0);

  grid.Update(particles.positions(), &particles.GetMutableBatchIndices());
  particles.SortParticles();

  // Up to this point both grid and particles have been ordered

  // check particle ordering
  // correct particles ordering should be [C, ADE, B]
  // since particles are sorted based on their batch indices, there is no
  // particular ordering between A, D and E. Should expect a first come first
  // serve ordering.

  EXPECT_EQ(particles.GetPositionAt(0), particle_C_position);
  EXPECT_EQ(particles.GetPositionAt(1), particle_A_position);
  EXPECT_EQ(particles.GetPositionAt(2), particle_D_position);
  EXPECT_EQ(particles.GetPositionAt(3), particle_E_position);
  EXPECT_EQ(particles.GetPositionAt(4), particle_B_position);

  std::vector<double> truth_masses_and_volumes{3.0, 1.0, 4.0, 5.0, 2.0};
  EXPECT_EQ(particles.masses(), truth_masses_and_volumes);
  EXPECT_EQ(particles.reference_volumes(), truth_masses_and_volumes);

  EXPECT_EQ(particles.GetBatchIndexAt(0), Vector3<int>(-3, 1, -1));
  EXPECT_EQ(particles.GetBatchIndexAt(1), Vector3<int>(2, 3, 4));
  EXPECT_EQ(particles.GetBatchIndexAt(2), Vector3<int>(2, 3, 4));
  EXPECT_EQ(particles.GetBatchIndexAt(3), Vector3<int>(2, 3, 4));
  EXPECT_EQ(particles.GetBatchIndexAt(4), Vector3<int>(3, 4, 5));

  // check grid nodes ordering

  // all active nodes are the following:

  // {-4, 0, -2} // 0
  // {-4, 0, -1} // 1
  // {-4, 0,  0} // 2
  // {-4, 1, -2} // 3
  // {-4, 1, -1} // 4
  // {-4, 1,  0} // 5
  // {-4, 2, -2} // 6
  // {-4, 2, -1} // 7
  // {-4, 2,  0} // 8

  // {-3, 0, -2} // 9
  // {-3, 0, -1} // 10
  // {-3, 0,  0} // 11
  // {-3, 1, -2} // 12
  // {-3, 1, -1} // 13
  // {-3, 1,  0} // 14
  // {-3, 2, -2} // 15
  // {-3, 2, -1} // 16
  // {-3, 2,  0} // 17

  // {-2, 0, -2} // 18
  // {-2, 0, -1} // 19
  // {-2, 0,  0} // 20
  // {-2, 1, -2} // 21
  // {-2, 1, -1} // 22
  // {-2, 1,  0} // 23
  // {-2, 2, -2} // 24
  // {-2, 2, -1} // 25
  // {-2, 2,  0} // 26

  // {1,2,3} // 27
  // {1,2,4} // 28
  // {1,2,5} // 29
  // {1,3,3} // 30
  // {1,3,4} // 31
  // {1,3,5} // 32
  // {1,4,3} // 33
  // {1,4,4} // 34
  // {1,4,5} // 35

  // {2,2,3} // 36
  // {2,2,4} // 37
  // {2,2,5} // 38
  // {2,3,3} // 39
  // {2,3,4} // 40
  // {2,3,5} // 41

  // {2,4,3} // 43
  // {2,4,4} // 44
  // {2,4,5} // 45

  // {3,2,3} // 50
  // {3,2,4} // 51
  // {3,2,5} // 52
  // {3,3,3} // 53
  // {3,3,4} // 54
  // {3,3,5} // 55

  // {3,4,3} // 57
  // {3,4,4} // 58
  // {3,4,5} // 59

  // {2,3,4} // repeated
  // {2,3,5} // repeated
  // {2,3,6} // 42
  // {2,4,4} // repeated
  // {2,4,5} // repeated
  // {2,4,6} // 46
  // {2,5,4} // 47
  // {2,5,5} // 48
  // {2,5,6} // 49

  // {3,3,4} // repeated
  // {3,3,5} // repeated
  // {3,3,6} // 56
  // {3,4,4} // repeated
  // {3,4,5} // repeated
  // {3,4,6} // 60
  // {3,5,4} // 61
  // {3,5,5} // 62
  // {3,5,6} // 63

  // {4,3,4} // 64
  // {4,3,5} // 65
  // {4,3,6} // 66
  // {4,4,4} // 67
  // {4,4,5} // 68
  // {4,4,6} // 69
  // {4,5,4} // 70
  // {4,5,5} // 71
  // {4,5,6} // 72

  EXPECT_EQ(grid.num_active_nodes(), 27 + 27 + 27 - 8);
  // eight of them overlap

  // // check a few
  EXPECT_EQ(grid.Expand1DIndex(0), Vector3<int>(-4, 0, -2));
  EXPECT_EQ(grid.Expand1DIndex(1), Vector3<int>(-4, 0, -1));
  EXPECT_EQ(grid.Expand1DIndex(2), Vector3<int>(-4, 0, 0));
  EXPECT_EQ(grid.Expand1DIndex(4), Vector3<int>(-4, 1, -1));
  EXPECT_EQ(grid.Expand1DIndex(6), Vector3<int>(-4, 2, -2));
  EXPECT_EQ(grid.Expand1DIndex(8), Vector3<int>(-4, 2, 0));
  EXPECT_EQ(grid.Expand1DIndex(10), Vector3<int>(-3, 0, -1));
  EXPECT_EQ(grid.Expand1DIndex(17), Vector3<int>(-3, 2, 0));
  EXPECT_EQ(grid.Expand1DIndex(24), Vector3<int>(-2, 2, -2));
  EXPECT_EQ(grid.Expand1DIndex(26), Vector3<int>(-2, 2, 0));

  EXPECT_EQ(grid.Expand1DIndex(27), Vector3<int>(1, 2, 3));
  EXPECT_EQ(grid.Expand1DIndex(28), Vector3<int>(1, 2, 4));
  EXPECT_EQ(grid.Expand1DIndex(29), Vector3<int>(1, 2, 5));
  EXPECT_EQ(grid.Expand1DIndex(30), Vector3<int>(1, 3, 3));
  EXPECT_EQ(grid.Expand1DIndex(33), Vector3<int>(1, 4, 3));
  EXPECT_EQ(grid.Expand1DIndex(36), Vector3<int>(2, 2, 3));
  EXPECT_EQ(grid.Expand1DIndex(39), Vector3<int>(2, 3, 3));
  EXPECT_EQ(grid.Expand1DIndex(41), Vector3<int>(2, 3, 5));
  EXPECT_EQ(grid.Expand1DIndex(42), Vector3<int>(2, 3, 6));
  EXPECT_EQ(grid.Expand1DIndex(44), Vector3<int>(2, 4, 4));
  EXPECT_EQ(grid.Expand1DIndex(45), Vector3<int>(2, 4, 5));
  EXPECT_EQ(grid.Expand1DIndex(46), Vector3<int>(2, 4, 6));
  EXPECT_EQ(grid.Expand1DIndex(47), Vector3<int>(2, 5, 4));
  EXPECT_EQ(grid.Expand1DIndex(48), Vector3<int>(2, 5, 5));
  EXPECT_EQ(grid.Expand1DIndex(49), Vector3<int>(2, 5, 6));
  EXPECT_EQ(grid.Expand1DIndex(50), Vector3<int>(3, 2, 3));
  EXPECT_EQ(grid.Expand1DIndex(51), Vector3<int>(3, 2, 4));
  EXPECT_EQ(grid.Expand1DIndex(55), Vector3<int>(3, 3, 5));
  EXPECT_EQ(grid.Expand1DIndex(56), Vector3<int>(3, 3, 6));
  EXPECT_EQ(grid.Expand1DIndex(57), Vector3<int>(3, 4, 3));
  EXPECT_EQ(grid.Expand1DIndex(58), Vector3<int>(3, 4, 4));
  EXPECT_EQ(grid.Expand1DIndex(59), Vector3<int>(3, 4, 5));
  EXPECT_EQ(grid.Expand1DIndex(60), Vector3<int>(3, 4, 6));
  EXPECT_EQ(grid.Expand1DIndex(61), Vector3<int>(3, 5, 4));
  EXPECT_EQ(grid.Expand1DIndex(64), Vector3<int>(4, 3, 4));
  EXPECT_EQ(grid.Expand1DIndex(72), Vector3<int>(4, 5, 6));

  const std::vector<size_t> batch_sizes = grid.batch_sizes();
  EXPECT_EQ(batch_sizes.size(), 27 + 27 + 27 - 8);
  for (size_t i = 0; i < 27 + 27 + 27 - 8; ++i) {
    if (i == 13) {
      EXPECT_EQ(batch_sizes[i], 1);
    } else if (i == 40) {
      EXPECT_EQ(batch_sizes[i], 3);
    } else if (i == 59) {
      EXPECT_EQ(batch_sizes[i], 1);
    } else {
      EXPECT_EQ(batch_sizes[i], 0);
    }
  }
}

}  // namespace
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
