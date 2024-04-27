#include <pinocchio/fwd.hpp>
#include <pinocchio/math/rpy.hpp>
#include <tbai_core/Types.hpp>

namespace tbai {
namespace core {

/**
 * @brief Convert roll-pitch-yaw euler angles to quaternion
 *
 * @param rpy : roll-pitch-yaw euler angles
 * @return quaternion_t : quaternion
 */
inline quaternion_t rpy2quat(const vector3_t &rpy) {
    return angleaxis_t(rpy(2), vector3_t::UnitZ()) * angleaxis_t(rpy(1), vector3_t::UnitY()) *
           angleaxis_t(rpy(0), vector3_t::UnitX());
}

/**
 * @brief Convert quaternion to a 3x3 rotation matrix
 *
 * @param q : quaternion
 * @return matrix3_t : rotation matrix
 */
inline matrix3_t quat2mat(const quaternion_t &q) {
    return q.toRotationMatrix();
}

/**
 * @brief Convert a 3x3 rotation matrix to roll-pitch-yaw euler angles
 *
 * @param R : rotation matrix
 * @return vector3_t : roll-pitch-yaw euler angles
 */
inline vector3_t mat2rpy(const matrix3_t &R) {
    return pinocchio::rpy::matrixToRpy(R);
}

/**
 * @brief Convert roll-pitch-yaw euler angles to a 3x3 rotation matrix
 *
 * @param rpy : roll-pitch-yaw euler angles
 * @return matrix3_t : rotation matrix
 */
inline matrix3_t rpy2mat(const vector3_t &rpy) {
    return pinocchio::rpy::rpyToMatrix(rpy);
}

/**
 * @brief Convert a 3x3 rotation matrix to axis-angle representation
 *
 * @param R : rotation matrix
 * @return vector3_t : axis-angle representation
 */
inline vector3_t mat2aa(const matrix3_t &R) {
    tbai::angleaxis_t aa(R);
    return aa.axis() * aa.angle();
}

}  // namespace core
}  // namespace tbai