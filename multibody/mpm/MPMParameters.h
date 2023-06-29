#pragma once

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "drake/common/eigen_types.h"

namespace drake {
namespace multibody {
namespace mpm {

// A class holding the global parameters used in MPMDriver
class MPMParameters {
 public:
    struct PhysicalParameters {
        // Gravitational acceleration
        Vector3<double> g;
        // Prescribed velocity field
        std::function<void(Vector3<double>,
                           double,
                           Vector3<double>*)> velocity_field;
    };

    // @pre endtime nonnegative, dt ∈ (0, endtime]
    // @pre min_num_particles_per_cell >= 1
    struct SolverParameters {
        // Run the simulation with timestep size dt till endtime
        double endtime;
        double dt;
        // Grid parameters, as documented in SparseGrid Class
        double h;
        // CFL number
        double CFL;
    };

    // IO parameters. Write the output as output_directory/case_name($i).bgeo
    // write_interval is the length of time between outputting, and i is id of
    // the output file, ordered ascendingly in time.
    struct IOParameters {
        std::string case_name;
        std::string output_directory;
        double write_interval;
    };

    MPMParameters(PhysicalParameters p_param,
                  SolverParameters s_param, IOParameters i_param):
                                physical_param(std::move(p_param)),
                                solver_param(std::move(s_param)),
                                io_param(std::move(i_param)) {}

    // Member variables
    PhysicalParameters physical_param;
    SolverParameters solver_param;
    IOParameters io_param;
};  // class MPMParameters

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
