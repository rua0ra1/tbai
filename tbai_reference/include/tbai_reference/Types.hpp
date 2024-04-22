#ifndef TBAI_REFERENCE_INCLUDE_TBAI_REFERENCE_TYPES_HPP_
#define TBAI_REFERENCE_INCLUDE_TBAI_REFERENCE_TYPES_HPP_

#include <tbai_core/Types.hpp>

namespace tbai {
namespace reference {

/** Reference velocity */
struct ReferenceVelocity {
    /** Velocity in the x direction, [m/s] */
    scalar_t velocity_x = 0.0;

    /** Velocity in the y direction, [m/s] */
    scalar_t velocity_y = 0.0;

    /** Angular velocity around the z axis, [rad/s] */
    scalar_t yaw_rate = 0.0;
};

}  // namespace reference
}  // namespace tbai

#endif  // TBAI_REFERENCE_INCLUDE_TBAI_REFERENCE_TYPES_HPP_
