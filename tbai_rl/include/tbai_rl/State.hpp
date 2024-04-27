#ifndef TBAI_RL_INCLUDE_TBAI_RL_STATE_HPP_
#define TBAI_RL_INCLUDE_TBAI_RL_STATE_HPP_

#include <tbai_core/Types.hpp>

namespace tbai {
namespace rl {

using tbai::scalar_t;
using tbai::vector_t;

struct State {
    using Vector3 = Eigen::Matrix<scalar_t, 3, 1>;
    using Vector4 = Eigen::Matrix<scalar_t, 4, 1>;
    using Vector12 = Eigen::Matrix<scalar_t, 12, 1>;

    /** Base position in the world frame */
    Vector3 basePositionWorld;

    /** Base orientation in the world frame - xyzw quaternion */
    Vector4 baseOrientationWorld;

    /** Base linear velocity expressed int the base frame */
    Vector3 baseLinearVelocityBase;

    /** Base angular velocity expressed in the base frame */
    Vector3 baseAngularVelocityBase;

    /** Gravity vector expressed in the base frame */
    Vector3 normalizedGravityBase;

    /** Joint angles */
    Vector12 jointPositions;

    /** Joint velocities */
    Vector12 jointVelocities;

    /** LF foot position in the world frame */
    Vector3 lfFootPositionWorld;

    /** LH foot position in the world frame */
    Vector3 lhFootPositionWorld;

    /** RF foot position in the world frame */
    Vector3 rfFootPositionWorld;

    /** RH foot position in the world frame */
    Vector3 rhFootPositionWorld;
};

}  // namespace rl
}  // namespace tbai

#endif  // TBAI_RL_INCLUDE_TBAI_RL_STATE_HPP_