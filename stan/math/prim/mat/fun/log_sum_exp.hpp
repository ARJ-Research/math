#ifndef STAN_MATH_PRIM_MAT_FUN_LOG_SUM_EXP_HPP
#define STAN_MATH_PRIM_MAT_FUN_LOG_SUM_EXP_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>
#include <vector>
#include <cmath>

namespace stan {
namespace math {


template<typename T1, typename T2, typename = require_vector_like_t<typename T1::value_type>>
std::vector<T2> match_input(std::vector<T2>& v){return v;}

template<typename T1, typename T2, typename = require_not_vector_like_t<typename T1::value_type>>
T2 match_input(std::vector<T2>& v){return v[0];}

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
template <typename T,
          typename = require_vector_like_st<std::is_arithmetic, T>>
auto log_sum_exp(const T& v) {
  eigen_seq_view<T> x(v);
  std::vector<double> result(x.size());
  double max = 0;
  for(int i = 0; i < x.size(); i++){
    max = x[i].maxCoeff();
    if (!std::isfinite(max)){
      result[i] = max;
      continue;
    }
    result[i] = max + std::log((x[i].array() - max).exp().sum());
  }
  return match_input<T>(result);
}

}  // namespace math
}  // namespace stan

#endif
