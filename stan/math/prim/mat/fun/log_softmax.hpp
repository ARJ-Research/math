#ifndef STAN_MATH_PRIM_MAT_FUN_LOG_SOFTMAX_HPP
#define STAN_MATH_PRIM_MAT_FUN_LOG_SOFTMAX_HPP

#include <stan/math/prim/arr/err/check_nonzero_size.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/log_sum_exp.hpp>
#include <stan/math/prim/mat/meta/return_container_type.hpp>

namespace stan {
namespace math {

/**
 * Return the natural logarithm of the softmax of the specified
 * vector.
 *
 * \f$
 * \log \mbox{softmax}(y)
 * \ = \ y - \log \sum_{k=1}^K \exp(y_k)
 * \ = \ y - \mbox{log\_sum\_exp}(y).
 * \f$
 *
 * For the log softmax function, the entries in the Jacobian are
 * \f$
 * \frac{\partial}{\partial y_m} \mbox{softmax}(y)[k]
 * = \left\{
 * \begin{array}{ll}
 * 1 - \mbox{softmax}(y)[m]
 * & \mbox{ if } m = k, \mbox{ and}
 * \\[6pt]
 * \mbox{softmax}(y)[m]
 * & \mbox{ if } m \neq k.
 * \end{array}
 * \right.
 * \f$
 *
 * @tparam T Scalar type of values in vector.
 * @param[in] v Vector to transform.
 * @return Unit simplex result of the softmax transform of the vector.
 */
template <typename T, typename = require_vector_like_st<std::is_arithmetic, T>,
          typename PromotedScalarT = return_type_t<typename T::value_type>>
inline auto log_softmax(const T& v) {
  check_nonzero_size("log_softmax", "v", v);
  auto u = as_eigen(v).template cast<PromotedScalarT>();
  return_container_t<T> result(u.size());
  as_eigen(result) = u.array() - log_sum_exp(u);
  return result;
}

template <typename Vec, require_vector_st<std::is_arithmetic, Vec>...,
          require_vector_vt<is_vector, Vec>...>
inline auto log_softmax(Vec&& v) {
  check_nonzero_size("log_softmax", "v", v);
  std::vector<return_container_t<value_type_t<Vec>>> result(v.size());
  for(int i = 0; i < v.size(); i++){
    result[i] = log_softmax(std::forward<decltype(v[i])>(v[i]));
  }
  return result;
}


}  // namespace math
}  // namespace stan
#endif
