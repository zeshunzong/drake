#include "drake/multibody/mpm/sparse_grid.h"

#include <vector>

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace mpm {
namespace {

constexpr double kTolerance = 4 * std::numeric_limits<double>::epsilon();

GTEST_TEST(SparseGridTest, TestSorting) {
  const double h = 3.14159;
  SparseGrid<double> grid(h);
  grid.Reserve(20);  // we don't need to test with more than 20 nodes

  // these nodes are lexicographically ordered
  Vector3<int> node_A{-1, -2, -3};
  Vector3<int> node_B{-1, -2, -1};
  Vector3<int> node_C{-1, -2, 4};
  Vector3<int> node_D{-1, 0, -10};
  Vector3<int> node_E{-1, 0, 7};
  Vector3<int> node_F{-1, 4, -3};
  Vector3<int> node_G{-1, 5, -3};
  Vector3<int> node_H{2, -9, -23};
  Vector3<int> node_I{2, -2, -33};
  Vector3<int> node_J{4, -7, 100};
  Vector3<int> node_K{4, -7, 101};
  Vector3<int> node_L{4, -6, 101};

  // append them to sparsegrid in a random order
  grid.AppendActiveNode(node_C);
  grid.AppendActiveNode(node_F);
  grid.AppendActiveNode(node_A);
  grid.AppendActiveNode(node_G);
  grid.AppendActiveNode(node_D);
  grid.AppendActiveNode(node_E);
  grid.AppendActiveNode(node_L);
  grid.AppendActiveNode(node_I);
  grid.AppendActiveNode(node_K);
  grid.AppendActiveNode(node_J);
  grid.AppendActiveNode(node_H);
  grid.AppendActiveNode(node_B);

  grid.SortActiveNodes();

  EXPECT_EQ(grid.Reduce3DIndex(node_A), 0);
  EXPECT_EQ(grid.Reduce3DIndex(node_B), 1);
  EXPECT_EQ(grid.Reduce3DIndex(node_C), 2);
  EXPECT_EQ(grid.Reduce3DIndex(node_D), 3);
  EXPECT_EQ(grid.Reduce3DIndex(node_E), 4);
  EXPECT_EQ(grid.Reduce3DIndex(node_F), 5);
  EXPECT_EQ(grid.Reduce3DIndex(node_G), 6);
  EXPECT_EQ(grid.Reduce3DIndex(node_H), 7);
  EXPECT_EQ(grid.Reduce3DIndex(node_I), 8);
  EXPECT_EQ(grid.Reduce3DIndex(node_J), 9);
  EXPECT_EQ(grid.Reduce3DIndex(node_K), 10);
  EXPECT_EQ(grid.Reduce3DIndex(node_L), 11);

  EXPECT_THROW(grid.Reduce3DIndex(Vector3<int>{0, 0, 0}), std::exception);
  EXPECT_FALSE(grid.IsActive(Vector3<int>{0, 0, 0}));

  EXPECT_EQ(grid.num_active_nodes(), 12);

  for (size_t i = 0; i < 12; ++i) {
    EXPECT_EQ(grid.Reduce3DIndex(grid.Expand1DIndex(i)), i);
  }

  EXPECT_NEAR(grid.GetPositionAt(node_A)(0), node_A(0) * h, kTolerance);
  EXPECT_NEAR(grid.GetPositionAt(node_A)(1), node_A(1) * h, kTolerance);
  EXPECT_NEAR(grid.GetPositionAt(node_A)(2), node_A(2) * h, kTolerance);

  // after clearing, should have no nodes
  grid.ClearActiveNodes();
  EXPECT_EQ(grid.num_active_nodes(), 0);
  EXPECT_THROW(grid.Reduce3DIndex(node_A), std::exception);
  EXPECT_FALSE(grid.IsActive(node_A));
}

GTEST_TEST(SparseGridTest, TestComputeBatchIndices) {
  const double h = 0.5;
  SparseGrid<double> grid(h);

  Vector3<double> particle_A_position{0.46, 0.99, -1.4};
  // it should be closet to {0.5, 1, -1.5}
  Vector3<int> particle_A_batch_index{1, 2, -3};

  Vector3<double> particle_B_position{7.77, 3.45, -1.6};
  // it should be cloaset to {8.0, 3.5, -1.5};
  Vector3<int> particle_B_batch_index{16, 7, -3};

  std::vector<Vector3<double>> positions;
  positions.emplace_back(std::move(particle_A_position));
  positions.emplace_back(std::move(particle_B_position));

  const std::vector<Vector3<int>> result = grid.ComputeBatchIndices(positions);

  EXPECT_EQ(result[0], particle_A_batch_index);
  EXPECT_EQ(result[1], particle_B_batch_index);
}

GTEST_TEST(SparseGridTest, TestUpdateActiveNodesOneParticle) {
  const double h = 1.0;
  SparseGrid<double> grid(h);

  Vector3<double> particle_A_position{0.46, 0.99, -1.4};
  // it should be closet to {0, 1, -1}
  // one single particle should activate 27 neighbor nodes
  // the 27 active nodes should be (in this order)
  // {-1,0,-2}, {-1,0,-1}, {-1,0,0}
  // {-1,1,-2}, {-1,1,-1}, {-1,1,0}
  // {-1,2,-2}, {-1,2,-1}, {-1,2,0}
  // {0,0,-2}, {0,0,-1}, {0,0,0}
  // {0,1,-2}, {0,1,-1}, {0,1,0}
  // {0,2,-2}, {0,2,-1}, {0,2,0}
  // {1,0,-2}, {1,0,-1}, {1,0,0}
  // {1,1,-2}, {1,1,-1}, {1,1,0}
  // {1,2,-2}, {1,2,-1}, {1,2,0}
  std::vector<Vector3<double>> positions;
  positions.emplace_back(std::move(particle_A_position));

  grid.UpdateActiveNodesFromParticlePositions(positions);

  EXPECT_EQ(grid.num_active_nodes(), 27);
  // check a few
  EXPECT_EQ(grid.Expand1DIndex(0), Vector3<int>(-1, 0, -2));
  EXPECT_EQ(grid.Expand1DIndex(1), Vector3<int>(-1, 0, -1));
  EXPECT_EQ(grid.Expand1DIndex(2), Vector3<int>(-1, 0, 0));
  EXPECT_EQ(grid.Expand1DIndex(3), Vector3<int>(-1, 1, -2));
  EXPECT_EQ(grid.Expand1DIndex(4), Vector3<int>(-1, 1, -1));
  EXPECT_EQ(grid.Expand1DIndex(5), Vector3<int>(-1, 1, 0));
  EXPECT_EQ(grid.Expand1DIndex(6), Vector3<int>(-1, 2, -2));
  EXPECT_EQ(grid.Expand1DIndex(9), Vector3<int>(0, 0, -2));
  EXPECT_EQ(grid.Expand1DIndex(12), Vector3<int>(0, 1, -2));
  EXPECT_EQ(grid.Expand1DIndex(15), Vector3<int>(0, 2, -2));
  EXPECT_EQ(grid.Expand1DIndex(18), Vector3<int>(1, 0, -2));
  EXPECT_EQ(grid.Expand1DIndex(21), Vector3<int>(1, 1, -2));
  EXPECT_EQ(grid.Expand1DIndex(24), Vector3<int>(1, 2, -2));
}

// two particles share the same batch index.
GTEST_TEST(SparseGridTest, TestUpdateActiveNodesTwoNearbyParticles) {
  const double h = 1.0;
  SparseGrid<double> grid(h);

  Vector3<double> particle_A_position{0.46, 0.99, -1.4};
  Vector3<double> particle_B_position{-0.3, 0.92, -0.8};
  // both should be closet to {0, 1, -1}
  // should activate 27 neighbor nodes
  // the 27 active nodes should be (in this order)
  // {-1,0,-2}, {-1,0,-1}, {-1,0,0}
  // {-1,1,-2}, {-1,1,-1}, {-1,1,0}
  // {-1,2,-2}, {-1,2,-1}, {-1,2,0}
  // {0,0,-2}, {0,0,-1}, {0,0,0}
  // {0,1,-2}, {0,1,-1}, {0,1,0}
  // {0,2,-2}, {0,2,-1}, {0,2,0}
  // {1,0,-2}, {1,0,-1}, {1,0,0}
  // {1,1,-2}, {1,1,-1}, {1,1,0}
  // {1,2,-2}, {1,2,-1}, {1,2,0}
  std::vector<Vector3<double>> positions;
  positions.emplace_back(std::move(particle_A_position));
  positions.emplace_back(std::move(particle_B_position));

  grid.UpdateActiveNodesFromParticlePositions(positions);

  EXPECT_EQ(grid.num_active_nodes(), 27);
  // check a few
  EXPECT_EQ(grid.Expand1DIndex(0), Vector3<int>(-1, 0, -2));
  EXPECT_EQ(grid.Expand1DIndex(1), Vector3<int>(-1, 0, -1));
  EXPECT_EQ(grid.Expand1DIndex(12), Vector3<int>(0, 1, -2));
  EXPECT_EQ(grid.Expand1DIndex(15), Vector3<int>(0, 2, -2));
  EXPECT_EQ(grid.Expand1DIndex(24), Vector3<int>(1, 2, -2));
}

// some of their neighbor nodes overlap
// manually computed lexicographical ordering is labelled
GTEST_TEST(SparseGridTest, TestUpdateActiveNodesTwoParticles) {
  const double h = 1.0;
  SparseGrid<double> grid(h);

  Vector3<double> particle_A_position{2.1, 2.9, 3.9};
  // its batch index should be {2,3,4}
  // its neighbor nodes are:
  // {1,2,3} // 0
  // {1,2,4} // 1
  // {1,2,5} // 2
  // {1,3,3} // 3
  // {1,3,4} // 4
  // {1,3,5} // 5
  // {1,4,3} // 6
  // {1,4,4} // 7
  // {1,4,5} // 8

  // {2,2,3} // 9
  // {2,2,4} // 10
  // {2,2,5} // 11
  // {2,3,3} // 12
  // {2,3,4} // 13
  // {2,3,5} // 14
  // {2,4,3} // 16
  // {2,4,4} // 17
  // {2,4,5} // 18

  // {3,2,3} // 23
  // {3,2,4} // 24
  // {3,2,5} // 25
  // {3,3,3} // 26
  // {3,3,4} // 27
  // {3,3,5} // 28
  // {3,4,3} // 30
  // {3,4,4} // 31
  // {3,4,5} // 32

  Vector3<double> particle_B_position{2.9, 4.1, 5.1};
  // its batch index should be {3,4,5}
  // its neighbor nodes are:
  // {2,3,4} // repeated
  // {2,3,5} // repeated
  // {2,3,6} // 15
  // {2,4,4} // repeated
  // {2,4,5} // repeated
  // {2,4,6} // 19
  // {2,5,4} // 20
  // {2,5,5} // 21
  // {2,5,6} // 22

  // {3,3,4} // repeated
  // {3,3,5} // repeated
  // {3,3,6} // 29
  // {3,4,4} // repeated
  // {3,4,5} // repeated
  // {3,4,6} // 33
  // {3,5,4} // 34
  // {3,5,5} // 35
  // {3,5,6} // 36

  // {4,3,4} // 37
  // {4,3,5} // 38
  // {4,3,6} // 39
  // {4,4,4} // 40
  // {4,4,5} // 41
  // {4,4,6} // 42
  // {4,5,4} // 43
  // {4,5,5} // 44
  // {4,5,6} // 45

  std::vector<Vector3<double>> positions;
  positions.emplace_back(std::move(particle_A_position));
  positions.emplace_back(std::move(particle_B_position));

  grid.UpdateActiveNodesFromParticlePositions(positions);


  EXPECT_EQ(grid.num_active_nodes(), 27+27-8);
  // eight of them overlap

  // check a few
  EXPECT_EQ(grid.Expand1DIndex(0), Vector3<int>(1,2,3));
  EXPECT_EQ(grid.Expand1DIndex(1), Vector3<int>(1,2,4));
  EXPECT_EQ(grid.Expand1DIndex(2), Vector3<int>(1,2,5));
  EXPECT_EQ(grid.Expand1DIndex(3), Vector3<int>(1,3,3));
  EXPECT_EQ(grid.Expand1DIndex(6), Vector3<int>(1,4,3));
  EXPECT_EQ(grid.Expand1DIndex(9), Vector3<int>(2,2,3));
  EXPECT_EQ(grid.Expand1DIndex(12), Vector3<int>(2,3,3));
  EXPECT_EQ(grid.Expand1DIndex(14), Vector3<int>(2,3,5));
  EXPECT_EQ(grid.Expand1DIndex(15), Vector3<int>(2,3,6));
  EXPECT_EQ(grid.Expand1DIndex(17), Vector3<int>(2,4,4));
  EXPECT_EQ(grid.Expand1DIndex(18), Vector3<int>(2,4,5));
  EXPECT_EQ(grid.Expand1DIndex(19), Vector3<int>(2,4,6));
  EXPECT_EQ(grid.Expand1DIndex(20), Vector3<int>(2,5,4));
  EXPECT_EQ(grid.Expand1DIndex(21), Vector3<int>(2,5,5));
  EXPECT_EQ(grid.Expand1DIndex(22), Vector3<int>(2,5,6));
  EXPECT_EQ(grid.Expand1DIndex(23), Vector3<int>(3,2,3));
  EXPECT_EQ(grid.Expand1DIndex(24), Vector3<int>(3,2,4));
  EXPECT_EQ(grid.Expand1DIndex(28), Vector3<int>(3,3,5));
  EXPECT_EQ(grid.Expand1DIndex(29), Vector3<int>(3,3,6));
  EXPECT_EQ(grid.Expand1DIndex(30), Vector3<int>(3,4,3));
  EXPECT_EQ(grid.Expand1DIndex(31), Vector3<int>(3,4,4));
  EXPECT_EQ(grid.Expand1DIndex(32), Vector3<int>(3,4,5));
  EXPECT_EQ(grid.Expand1DIndex(33), Vector3<int>(3,4,6));
  EXPECT_EQ(grid.Expand1DIndex(34), Vector3<int>(3,5,4));
  EXPECT_EQ(grid.Expand1DIndex(37), Vector3<int>(4,3,4));
  EXPECT_EQ(grid.Expand1DIndex(45), Vector3<int>(4,5,6));
}

// 27 + 27 neighbor nodes
GTEST_TEST(SparseGridTest, TestUpdateActiveNodesTwoFarawayParticles) {
  const double h = 0.1657;
  SparseGrid<double> grid(h);

  Vector3<double> particle_A_position{0.46, 0.99, -1.4};
  Vector3<double> particle_B_position{-10.3, 10.92, -10.8};

  std::vector<Vector3<double>> positions;
  positions.emplace_back(std::move(particle_A_position));
  positions.emplace_back(std::move(particle_B_position));

  grid.UpdateActiveNodesFromParticlePositions(positions);

  EXPECT_EQ(grid.num_active_nodes(), 27+27);
  
}


}  // namespace
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
