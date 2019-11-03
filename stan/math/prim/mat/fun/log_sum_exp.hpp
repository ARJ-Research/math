#ifndef STAN_MATH_PRIM_MAT_FUN_LOG_SUM_EXP_HPP
#define STAN_MATH_PRIM_MAT_FUN_LOG_SUM_EXP_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>
#include <vector>
#include <cmath>

namespace stan {
namespace math {

/**
 * Return the log of the sum of the exponentiated values of the specified
 * matrix of values.  The matrix may be a full matrix, a vector,
 * or a row vector.
 *
 * The function is defined as follows to prevent overflow in exponential
 * calculations.
 *
 * \f$\log \sum_{n=1}^N \exp(x_n) = \max(x) + \log \sum_{n=1}^N \exp(x_n -
 * \max(x))\f$.
 *
 * @param[in] x Matrix of specified values
 * @return The log of the sum of the exponentiated vector values.
 */
template <typename T, typename = require_vector_like_st<std::is_arithmetic, T>>
double log_sum_exp(const T& v) {
  const auto& x = as_eigen(v);
  auto max = x.maxCoeff();
  if (!std::isfinite(max))
    return max;
  return max + std::log((x.array() - max).exp().sum());
}


template <typename T, typename = require_vector_like_st<std::is_arithmetic, T>>
std::vector<double> log_sum_exp(const std::vector<T>& v) {
  std::vector<double> result(v.size());
  for(int i = 0; i < v.size(); i++){
    result[i] = log_sum_exp(v[i]);
  }
  return result;
}

}  // namespace math
}  // namespace stan

#endif
