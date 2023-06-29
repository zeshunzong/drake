#include "drake/multibody/mpm/MPMDriver.h"

namespace drake {
namespace multibody {
namespace mpm {

MPMDriver::MPMDriver(MPMParameters param):
                                param_(std::move(param)),
                                grid_(param.solver_param.h),
                                gravitational_force_(param.physical_param.g),
                                dilatational_wavespd_(0.0) {
    DRAKE_DEMAND(param.solver_param.endtime >= 0.0);
    DRAKE_DEMAND(param.solver_param.dt > 0.0);
}

void MPMDriver::InitializeKinematicCollisionObjects(KinematicCollisionObjects
                                                    objects) {
    collision_objects_ = std::move(objects);
}

void MPMDriver::DoTimeStepping() {
    int step = 0;
    int io_step = 0;
    double endtime = param_.solver_param.endtime;
    double dt = param_.solver_param.dt;
    bool is_last_step = false;
    using Clock = std::chrono::steady_clock;
    using Duration = std::chrono::duration<double>;
    std::chrono::time_point<Clock> start_timestepping_time = Clock::now();
    std::chrono::time_point<Clock> start_io_time;
    std::cout << "dt from param is " << dt << std::endl; getchar();

    // Advance time steps until endtime
    for (double t = 0; t < endtime; t += dt) {
        is_last_step = (endtime - t <= dt);
        // If at the final timestep, modify the timestep size to match endtime
        if (endtime - t < dt) {
            dt = endtime - t;
        }
        // Check the CFL condition
        if (t > 0) {
            UpdateTimeStep(&dt);
        }
        std::cout << "dt after checking CFL is " << dt << std::endl; getchar();

        AdvanceOneTimeStep(dt, t+dt);  // TODO(yiminlin.tri): we used updated
                                       //                     velocity field
        step++;
    
        // TotalMassAndMomentum total =
        // particles_.GetTotalMassAndMomentum(param_.physical_param.g[2]);
        // std::ofstream output_momentum;
        // output_momentum.open(param_.io_param.output_directory + "/statistics_momentum.dat",
        //                         std::fstream::app);
        // if (output_momentum.is_open()) {
        //     output_momentum << t                        << std::endl;
        //     output_momentum << total.sum_momentum.norm() << std::endl;
        // }

        std::ofstream output_statistics;
        output_statistics.open(param_.io_param.output_directory + "/statistics.dat",
                               std::fstream::app);
        if ((t >= io_step*param_.io_param.write_interval) || (is_last_step)) {
            std::cout << "==== MPM Step " << step << " iostep " << io_step <<
                         ", at t = " << t+dt << std::endl;
            start_io_time = Clock::now();
            // Write to output
            WriteParticlesToBgeo(io_step++);
            // Write to statistics
            DumpStatistics(t, step, output_statistics);
            const auto elapsed_time
                    = std::chrono::duration_cast<Duration>(Clock::now()
                                                    - start_io_time).count();
            run_time_statistics_.time_io += elapsed_time;
        }
    }

    const auto total_time = std::chrono::duration_cast<Duration>(Clock::now()
                                            - start_timestepping_time).count();
    run_time_statistics_.time_total += total_time;

    PrintRunTimeStatistics();
}

void MPMDriver::DumpStatistics(double t, double step, std::ofstream& output_statistics)
                                                                        const {
    TotalMassEnergyMomentum total =
        particles_.GetTotalMassEnergyMomentum(param_.physical_param.g[2]);
    double total_energy = total.sum_kinetic_energy
                        + total.sum_strain_energy
                        + total.sum_potential_energy;
    if (output_statistics.is_open()) {
        output_statistics << t                           << std::endl;
        output_statistics << step                        << std::endl;
        output_statistics << total.sum_mass              << std::endl;
        output_statistics << total.sum_kinetic_energy    << std::endl;
        output_statistics << total.sum_strain_energy     << std::endl;
        output_statistics << total.sum_potential_energy  << std::endl;
        output_statistics << total_energy                << std::endl;
        output_statistics << total.sum_momentum.norm()   << std::endl;
        output_statistics << total.sum_angular_momentum.norm()  << std::endl;
        output_statistics << sum_boundary_impulse_n_  << std::endl;
        output_statistics << sum_boundary_impulse_t_  << std::endl;
        output_statistics << sum_gravity_impulse_n_  << std::endl;
        output_statistics << sum_gravity_impulse_t_  << std::endl;
    }
}

// TODO(yiminlin.tri): Not tested.
void MPMDriver::InitializeParticles(const AnalyticLevelSet& level_set,
                                    const math::RigidTransform<double>& pose,
                                    MaterialParameters m_param) {
    DRAKE_DEMAND(m_param.density > 0.0);
    DRAKE_DEMAND(m_param.min_num_particles_per_cell >= 1);

    double h = param_.solver_param.h;
    const std::array<Vector3<double>, 2> bounding_box =
                                    level_set.get_bounding_box();

    // Distances between generated particles are at at least sample_r apart, and
    // there are at least min_num_particles_per_cell particles per cell. r =
    // argmax(⌊h/r⌋)^3 >= min_num_particles_per_cell, in other words, if we pick
    // particles located at the grid with grid size r, there are at least
    // min_num_particles_per_cell particles in a cell with size h.
    double sample_r =
            h/(std::cbrt(m_param.min_num_particles_per_cell)+1);
    multibody::SpatialVelocity<double> init_v = m_param.initial_velocity;
    std::array<double, 3> xmin = {bounding_box[0][0], bounding_box[0][1],
                                  bounding_box[0][2]};
    std::array<double, 3> xmax = {bounding_box[1][0], bounding_box[1][1],
                                  bounding_box[1][2]};
    // Generate sample particles in the reference frame (centered at the origin
    // with canonical basis e_i)
    std::vector<Vector3<double>> particles_sample_positions =
        thinks::PoissonDiskSampling<double, 3, Vector3<double>>(sample_r,
                                                                xmin, xmax);

    // Pick out sampled particles that are in the object
    int num_samples = particles_sample_positions.size();
    std::vector<Vector3<double>> particles_positions, particles_velocities;
    for (int p = 0; p < num_samples; ++p) {
        // Denote the particle by P, the object by B, the frame centered at the
        // origin of the object by Bo, and the frame of the particle by Bp.

        // The pose and spatial velocity of the object in the world frame
        const math::RigidTransform<double>& X_WB       = pose;
        const multibody::SpatialVelocity<double>& V_WB = init_v;

        // Rotation matrix from the object to the world
        const math::RotationMatrix<double>& Rot_WB     = X_WB.rotation();

        // Sample particles and get the position of the particle with respect to
        // the object in the object's frame
        const Vector3<double>& p_BoBp_B = particles_sample_positions[p];

        // Express the relative position of the particle with respect to the
        // object in the world frame
        const Vector3<double>& p_BoBp_W = Rot_WB*p_BoBp_B;

        // Compute the spatial velocity of the particle
        SpatialVelocity<double> V_WBp   = V_WB.Shift(p_BoBp_W);

        // If the particle is in the level set
        if (level_set.InInterior(p_BoBp_B)) {
            // TODO(yiminlin.tri): Initialize the affine matrix C_p using
            //                     V_WBp.rotational() ?
            particles_velocities.emplace_back(V_WBp.translational());
            // Place the particle's position in world frame
            particles_positions.emplace_back(X_WB*p_BoBp_B);
        }
    }

    int num_particles = particles_positions.size();
    // We assume every particle have the same volume and mass
    double reference_volume_p = level_set.get_volume()/num_particles;
    double init_m = m_param.density*reference_volume_p;

    // Add particles
    for (int p = 0; p < num_particles; ++p) {
        const Vector3<double>& xp = particles_positions[p];
        const Vector3<double>& vp = particles_velocities[p];
        Matrix3<double> elastic_deformation_grad_p
                                                = Matrix3<double>::Identity();
        Matrix3<double> kirchhoff_stress_p = Matrix3<double>::Identity();
        Matrix3<double> B_p                = Matrix3<double>::Zero();
        std::unique_ptr<ElastoPlasticModel> elastoplastic_model_p
                                        = m_param.elastoplastic_model->Clone();
        particles_.AddParticle(xp, vp, init_m, reference_volume_p,
                               elastic_deformation_grad_p,
                               kirchhoff_stress_p,
                               B_p, std::move(elastoplastic_model_p));
    }

    // Update dilatational wave speed
    double lambda = m_param.elastoplastic_model->get_lambda();
    double mu     = m_param.elastoplastic_model->get_mu();
    dilatational_wavespd_ = std::max(dilatational_wavespd_,
                                     std::sqrt((lambda+2*mu)/m_param.density));
    std::cout << "total number of particles generated: " << num_particles << std::endl;getchar();
}

void MPMDriver::WriteParticlesToBgeo(int io_step) {
    std::string output_filename = param_.io_param.output_directory + "/"
                                + param_.io_param.case_name
                                + std::to_string(io_step) + ".bgeo";
    internal::WriteParticlesToBgeo(output_filename, particles_.get_positions(),
                                                    particles_.get_velocities(),
                                                    particles_.get_masses());
}

void MPMDriver::PrintRunTimeStatistics() const {
    std::cout << "===================" << std::endl;
    std::cout << "==== Total run                   time: "
            << run_time_statistics_.time_total
            << " seconds" << std::endl;
    std::cout << "==== Total IO                    time: "
            << run_time_statistics_.time_io
            << " seconds" << std::endl;
    std::cout << "==== Total stress/plastic update time: "
            << run_time_statistics_.time_update_stress_and_plasticity
            << " seconds" << std::endl;
    std::cout << "==== Total transfer setup        time: "
            << run_time_statistics_.time_setup_transfer
            << " seconds" << std::endl;
    std::cout << "==== Total P2G                   time: "
            << run_time_statistics_.time_P2G
            << " seconds" << std::endl;
    std::cout << "==== Total update velocity       time: "
            << run_time_statistics_.time_update_grid_velocity
            << " seconds" << std::endl;
    std::cout << "==== Total apply forces          time: "
            << run_time_statistics_.time_apply_external_forces
            << " seconds" << std::endl;
    std::cout << "==== Total collision objs update time: "
            << run_time_statistics_.time_collision_objects_update
            << " seconds" << std::endl;
    std::cout << "==== Total enforce BC            time: "
            << run_time_statistics_.time_enforce_bc
            << " seconds" << std::endl;
    std::cout << "==== Total G2P                   time: "
            << run_time_statistics_.time_G2P
            << " seconds" << std::endl;
    std::cout << "==== Total particles advection   time: "
            << run_time_statistics_.time_advect_particles
            << " seconds" << std::endl;
}

void MPMDriver::UpdateTimeStep(double* dt) {
    double h = grid_.get_h();
    double dt_new = std::numeric_limits<double>::infinity();
    for (const auto& v : particles_.get_velocities()) {
        dt_new = std::min(dt_new,
                          h/std::max({std::abs(dilatational_wavespd_+v(0)),
                                      std::abs(dilatational_wavespd_+v(1)),
                                      std::abs(dilatational_wavespd_+v(2))}));
    }
    *dt = param_.solver_param.CFL*dt_new;
}

void MPMDriver::AdvanceOneTimeStep(double dt, double t) {
    using Clock = std::chrono::steady_clock;
    using Duration = std::chrono::duration<double>;
    std::chrono::time_point<Clock> start_time;
    Duration::rep elapsed_time;

    // Apply plasticity and update Kirchhoff stress on particles
    start_time = Clock::now();
    particles_.ApplyPlasticityAndUpdateKirchhoffStresses();
    elapsed_time = std::chrono::duration_cast<Duration>(Clock::now()
                                                      - start_time).count();
    run_time_statistics_.time_update_stress_and_plasticity += elapsed_time;

    // Set up the transfer routines (Preallocations, sort the particles)
    start_time = Clock::now();
    mpm_transfer_.SetUpTransfer(&grid_, &particles_);
    elapsed_time = std::chrono::duration_cast<Duration>(Clock::now()
                                                      - start_time).count();
    run_time_statistics_.time_setup_transfer += elapsed_time;

    // Main Algorithm:
    // P2G
    start_time = Clock::now();
    mpm_transfer_.TransferParticlesToGrid(particles_, &grid_);
    elapsed_time = std::chrono::duration_cast<Duration>(Clock::now()
                                                      - start_time).count();
    run_time_statistics_.time_P2G += elapsed_time;

    // Update grid velocity
    start_time = Clock::now();
    grid_.UpdateVelocity(dt);
    elapsed_time = std::chrono::duration_cast<Duration>(Clock::now()
                                                      - start_time).count();
    run_time_statistics_.time_update_grid_velocity += elapsed_time;

    // Apply gravitational force
    start_time = Clock::now();
    gravitational_force_.ApplyGravitationalForces(dt, &grid_);
    elapsed_time = std::chrono::duration_cast<Duration>(Clock::now()
                                                      - start_time).count();
    run_time_statistics_.time_apply_external_forces += elapsed_time;

    // Update Collision Objects
    start_time = Clock::now();
    collision_objects_.AdvanceOneTimeStep(dt);
    elapsed_time = std::chrono::duration_cast<Duration>(Clock::now()
                                                      - start_time).count();
    run_time_statistics_.time_collision_objects_update += elapsed_time;

    // Enforce boundary conditions
    start_time = Clock::now();
    // TODO(yiminlin.tri): hardcoded...
    std::tie(sum_boundary_impulse_n_, sum_boundary_impulse_t_,
             sum_gravity_impulse_n_, sum_gravity_impulse_t_)
                = grid_.EnforceBoundaryCondition(collision_objects_, dt, t);
    elapsed_time = std::chrono::duration_cast<Duration>(Clock::now()
                                                      - start_time).count();
    run_time_statistics_.time_enforce_bc += elapsed_time;

    // std::ofstream output_impulse;
    // output_impulse.open(param_.io_param.output_directory + "/statistics_impulse.dat",
    //                         std::fstream::app);
    // if (output_impulse.is_open()) {
    //     output_impulse << t                        << std::endl;
    //     output_impulse << sum_boundary_impulse_n_  << std::endl;
    //     output_impulse << sum_boundary_impulse_t_  << std::endl;
    //     output_impulse << sum_gravity_impulse_n_  << std::endl;
    //     output_impulse << sum_gravity_impulse_t_  << std::endl;
    // }

    // Overwrite velocity field
    start_time = Clock::now();
    grid_.OverwriteGridVelocity(param_.physical_param.velocity_field, t);
    elapsed_time = std::chrono::duration_cast<Duration>(Clock::now()
                                                      - start_time).count();
    run_time_statistics_.time_enforce_bc += elapsed_time;

    // G2P
    start_time = Clock::now();
    mpm_transfer_.TransferGridToParticles(grid_, dt, &particles_);
    elapsed_time = std::chrono::duration_cast<Duration>(Clock::now()
                                                      - start_time).count();
    run_time_statistics_.time_G2P += elapsed_time;

    // Advect particles
    start_time = Clock::now();
    particles_.AdvectParticles(dt);
    elapsed_time = std::chrono::duration_cast<Duration>(Clock::now()
                                                      - start_time).count();
    run_time_statistics_.time_advect_particles += elapsed_time;
}

}  // namespace mpm
}  // namespace multibody
}  // namespace drake
