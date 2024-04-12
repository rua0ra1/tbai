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

}  // namespace tbai

#endif  // TBAI_CORE_INCLUDE_TBAI_CORE_TYPES_HPP_
