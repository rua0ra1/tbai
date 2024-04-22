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

/** 3 vector */
using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;

/** 3x3 matrix */
using matrix3_t = Eigen::Matrix<scalar_t, 3, 3>;

using angleaxis_t = Eigen::AngleAxis<scalar_t>;

using quaternion_t = Eigen::Quaternion<scalar_t>;

struct State {
    using Vector3 = Eigen::Matrix<scalar_t, 3, 1>;
    using Vector4 = Eigen::Matrix<scalar_t, 4, 1>;
    using Vector12 = Eigen::Matrix<scalar_t, 12, 1>;

    Vector3 basePositionWorld;
    Vector4 baseOrientationWorld;  // quaternion, xyzw
    Vector3 baseLinearVelocityBase;
    Vector3 baseAngularVelocityBase;
    Vector3 normalizedGravityBase;
    Vector12 jointPositions;
    Vector12 jointVelocities;
    Vector3 lfFootPositionWorld;
    Vector3 lhFootPositionWorld;
    Vector3 rfFootPositionWorld;
    Vector3 rhFootPositionWorld;
};

}  // namespace tbai

#endif  // TBAI_CORE_INCLUDE_TBAI_CORE_TYPES_HPP_
