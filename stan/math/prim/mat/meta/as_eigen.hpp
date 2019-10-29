#ifndef STAN_MATH_PRIM_MAT_META_AS_EIGEN_HPP
#define STAN_MATH_PRIM_MAT_META_AS_EIGEN_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>
#include <vector>

namespace stan {
namespace math {


template <typename T, typename = require_eigen_t<T>>
const auto& as_eigen(const T& v) {
  return v;
}

template <typename T>
const auto as_eigen(const std::vector<T>& v) {
  return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(v.data(),
                                                               v.size());
}

}  // namespace math
}  // namespace stan

#endif
