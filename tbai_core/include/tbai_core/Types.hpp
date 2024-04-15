#ifndef TBAI_CORE_INCLUDE_TBAI_CORE_TYPES_HPP_
#define TBAI_CORE_INCLUDE_TBAI_CORE_TYPES_HPP_

#include <Eigen/Dense>

namespace tbai {

/** Scalar type */
using scalar_t = double;

/** Vector type */
using vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;

/** Matrix type */
using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;

/** Velocity command */
struct VelocityCommand {
    /** Velocity in the x direction, [m/s] */
    scalar_t velocity_x = 0.0;

    /** Velocity in the y direction, [m/s] */
    scalar_t velocity_y = 0.0;

    /** Angular velocity around the z axis, [rad/s] */
    scalar_t yaw_rate = 0.0;
};

}  // namespace tbai

#endif  // TBAI_CORE_INCLUDE_TBAI_CORE_TYPES_HPP_
