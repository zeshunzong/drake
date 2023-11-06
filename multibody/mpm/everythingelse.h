template <typename T>
class InterpolationWeights {
 public:
  void Reset(const Vector3<int>& base_node, const Vector3<T>& position,
             double dx) {
    // Sets weights_ and weight_graidents_;
  }

  void SplatParticleDataToPad(const Particles<T>& particles, int p,
                              Pad<T>* pad) {
    // Get p-th particle's data from `particles` and write it to pad with the
    // weights_ and wieght_gradients_. */
  }

 private:
  std::array<T, 27> weights_{};
  std::array<Vector3<T>, 27> weight_gradients_{};
};

template <typename T>
struct Pad {
  std::array<T, 27> mass;
  std::array<Vector3<T>, 27> velocity;
  Vector3<int> base_node;

  // @pre a, b, c = [-1, 0, +1]
  const T& GetMassAt(int a, int b, int c) {
    return mass[...]
  }
};

template <typename T>
class GridData {
    std::vector<T> masses_;
};


template <typename T>
class SparseGrid {
 public:
  void MarkActiveGridNodes(const std::vector<Vector3<int>>& base_nodes) {
    // Mark the one ring of `base_nodes` as active.
    // Build 1d_to_3d_ and 3d_to_1d_;
  }

  // @pre Pads contain data needed for all active grid nodes
  void GatherFomPads(const std::vector<Pad<T>>& pads,
                     GridData<T>* grid_data) const {
    grid_data->Resize(map_1d_to_3d_.size());
    for (const Pad<T>& pad : pads) {
      const Vector3<int>& base_node = pad.base_node;
      // Add pad data to grid data.
      for (int a = -1, 0, 1) {
        for (int b = -1, 0, 1) {
          for (int c = -1, 0, 1) {
            int index = map_3d_to_1d_.at(Vector3<int>(a, b, c) + base_node);
            (*grid_data).masses_[index] += pad.GetMassAt(a,b,c);
            // ...
          }
        }
      }
    }
  }

 private:
  std::vector<Vector3<int>> map_1d_to_3d_;
  std::unordered_map<Vector3<int>, int> map_3d_to_1d_;

  double h;
};


template <typename T>
struct ScratchForBatch {
    std::array<Vector3<T>, 27> velocity;
    // other stuff for F and B
};


template <typename T>
class MpmTransfer {
  static void Initialize(Particles<T>* particles, SparseGrid<T>* grid) {
    particles->SortIntoBatches(grid->h());
    grid->MarkActiveGridNodes(particles->get_base_nodes());
  }

  void P2G(const Particles<T>& particles, const SparseGrid<T>& grid,
           GridData<T>* data) {
    DRAKE_DEMAND(!particles->positions_have_changed_);
    particles.SplatParticleDataToPad(&pads_);
    grid.GatherFomPads(pads_, data);
  }

  void G2P(const SparseGrid<T>& grid, const GridData<T>& grid_data,
           Particles<T>* particles, double dt) {
    DRAKE_DEMAND(!particles->positions_have_changed_);
    ScratchForBatch<T> scratch;

    for (size_t batch_i = 0; batch_i < particles->num_batches(); ++batch_i) {
        // write grid_data of 27 nodes neighboring to base_node = particles->base_nodes_[batch_i]
        scratch = ... 
        // update scratch to all particles in batch i
        // e.g. particles->UpdateVelocitiesAndDeformationGradientsForBaseNodeI(scratch, batch_i)
    }        
    particles->AdvectParticles(dt);           
  }

 private:
  std::vector<Pad<T>> pads_;
};