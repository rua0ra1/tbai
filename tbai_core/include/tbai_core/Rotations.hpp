#include <pinocchio/fwd.hpp>
#include <pinocchio/math/rpy.hpp>
#include <tbai_core/Types.hpp>

namespace tbai {
namespace core {

inline quaternion_t rpy2quat(const vector3_t &rpy) {
    return angleaxis_t(rpy(2), vector3_t::UnitZ()) * angleaxis_t(rpy(1), vector3_t::UnitY()) *
           angleaxis_t(rpy(0), vector3_t::UnitX());
}

inline matrix3_t quat2mat(const quaternion_t &q) {
    return q.toRotationMatrix();
}

inline vector3_t mat2rpy(const matrix3_t &R) {
    return pinocchio::rpy::matrixToRpy(R);
}

inline matrix3_t rpy2mat(const vector3_t &rpy) {
    return pinocchio::rpy::rpyToMatrix(rpy);
}

inline vector3_t mat2aa(const matrix3_t &R) {
    tbai::angleaxis_t aa(R);
    return aa.axis() * aa.angle();
}

}  // namespace core
}  // namespace tbai