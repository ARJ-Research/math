#ifndef STAN_MATH_PRIM_MAT_META_AS_EIGEN_HPP
#define STAN_MATH_PRIM_MAT_META_AS_EIGEN_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>
#include <vector>

namespace stan {
namespace math {

/**
 * Converts a matrix type to an array.
 *
 * @tparam T Type of scalar element.
 * @tparam R Row type of input matrix.
 * @tparam C Column type of input matrix.
 * @param v Specified matrix.
 * @return Matrix converted to an array.
 */
template <typename T>
const auto& as_eigen(const T& v) {
  return v;
}

/**
 * Converts a std::vector type to an array.
 *
 * @tparam T Type of scalar element.
 * @param v Specified vector.
 * @return Matrix converted to an array.
 */
template <typename T>
const auto as_eigen(const std::vector<T>& v) {
  return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(v.data(),
                                                               v.size());
}

}  // namespace math
}  // namespace stan

#endif
