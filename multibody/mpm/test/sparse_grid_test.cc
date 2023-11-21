#include "drake/multibody/mpm/sparse_grid.h"

#include <vector>

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
namespace mpm {
namespace {

// We manually provide five base nodes (with repeated elements) and test if all
// active nodes are indeed marked as active.
GTEST_TEST(SparseGridTest, TestParticlesAndGridSorting) {
  const double h = 1.0;
  SparseGrid<double> grid(h);

  Vector3<int> base_node1{2, 3, 4};
  Vector3<int> base_node2{3, 4, 5};
  Vector3<int> base_node3{-3, 1, -1};
  Vector3<int> base_node4{2, 3, 4};
  Vector3<int> base_node5{2, 3, 4};

  std::vector<Vector3<int>> base_nodes;

  base_nodes.push_back(base_node1);
  base_nodes.push_back(base_node2);
  base_nodes.push_back(base_node3);
  base_nodes.push_back(base_node4);
  base_nodes.push_back(base_node5);

  grid.MarkActiveNodes(base_nodes);

  // all active nodes are the following:

  // {-4, 0, -2}
  // {-4, 0, -1}
  // {-4, 0,  0}
  // {-4, 1, -2}
  // {-4, 1, -1}
  // {-4, 1,  0}
  // {-4, 2, -2}
  // {-4, 2, -1}
  // {-4, 2,  0}

  // {-3, 0, -2}
  // {-3, 0, -1}
  // {-3, 0,  0}
  // {-3, 1, -2}
  // {-3, 1, -1}
  // {-3, 1,  0}
  // {-3, 2, -2}
  // {-3, 2, -1}
  // {-3, 2,  0}

  // {-2, 0, -2}
  // {-2, 0, -1}
  // {-2, 0,  0}
  // {-2, 1, -2}
  // {-2, 1, -1}
  // {-2, 1,  0}
  // {-2, 2, -2}
  // {-2, 2, -1}
  // {-2, 2,  0}

  // {1,2,3}
  // {1,2,4}
  // {1,2,5}
  // {1,3,3}
  // {1,3,4}
  // {1,3,5}
  // {1,4,3}
  // {1,4,4}
  // {1,4,5}

  // {2,2,3}
  // {2,2,4}
  // {2,2,5}
  // {2,3,3}
  // {2,3,4}
  // {2,3,5}

  // {2,4,3}
  // {2,4,4}
  // {2,4,5}

  // {3,2,3}
  // {3,2,4}
  // {3,2,5}
  // {3,3,3}
  // {3,3,4}
  // {3,3,5}

  // {3,4,3}
  // {3,4,4}
  // {3,4,5}

  // {2,3,4}
  // {2,3,5}
  // {2,3,6}
  // {2,4,4}
  // {2,4,5}
  // {2,4,6}
  // {2,5,4}
  // {2,5,5}
  // {2,5,6}

  // {3,3,4}
  // {3,3,5}
  // {3,3,6}
  // {3,4,4}
  // {3,4,5}
  // {3,4,6}
  // {3,5,4}
  // {3,5,5}
  // {3,5,6}

  // {4,3,4}
  // {4,3,5}
  // {4,3,6}
  // {4,4,4}
  // {4,4,5}
  // {4,4,6}
  // {4,5,4}
  // {4,5,5}
  // {4,5,6}

  // eight of them overlap
  EXPECT_EQ(grid.num_active_nodes(), 27 + 27 + 27 - 8);

  EXPECT_TRUE(grid.IsActive(Vector3<int>(-4, 0, -2)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(-4, 0, -1)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(-4, 0, 0)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(-4, 1, -1)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(-4, 2, -2)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(-4, 2, 0)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(-3, 0, -1)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(-3, 2, 0)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(-2, 2, -2)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(-2, 2, 0)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(1, 2, 3)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(1, 2, 4)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(1, 2, 5)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(1, 3, 3)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(1, 4, 3)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(2, 2, 3)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(2, 3, 3)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(2, 3, 5)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(2, 4, 4)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(2, 4, 5)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(2, 4, 6)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(2, 5, 4)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(2, 5, 5)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(2, 5, 6)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(3, 2, 3)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(3, 2, 4)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(3, 3, 5)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(3, 3, 6)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(3, 4, 3)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(3, 4, 4)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(3, 4, 5)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(3, 4, 6)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(3, 5, 4)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(4, 3, 4)));
  EXPECT_TRUE(grid.IsActive(Vector3<int>(4, 5, 6)));

  // check that the two maps (3d->1d and 1d->3d) are indeed bijective

  for (size_t i = 0; i < 27 + 27 + 27 - 8; ++i) {
    EXPECT_EQ(grid.To1DIndex(grid.To3DIndex(i)), i);
  }
}

}  // namespace
}  // namespace mpm
}  // namespace multibody
}  // namespace drake
