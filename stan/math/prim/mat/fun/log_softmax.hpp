#ifndef STAN_MATH_PRIM_MAT_FUN_LOG_SOFTMAX_HPP
#define STAN_MATH_PRIM_MAT_FUN_LOG_SOFTMAX_HPP

#include <stan/math/prim/arr/err/check_nonzero_size.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/log_sum_exp.hpp>

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
template <typename T, typename T_v = typename T::value_type,
          typename nest_vec_t = std::conditional_t<is_vector_like<T_v>::value,
                                                   T_v, T>,
          typename = require_vector_like_st<std::is_arithmetic, T>>
inline auto log_softmax(const T& v) {
  eigen_seq_view<T> u(v);
  check_nonzero_size("log_softmax", "v", u[0]);
  std::vector<nest_vec_t> result(u.size());
  for(int i = 0; i < u.size(); i++){
    result[i].resize(u[i].size());
    as_eigen(result[i]) = u[i].array() - log_sum_exp(u[i]);
  }
  return match_input_dim<T>(result);
}

}  // namespace math
}  // namespace stan
#endif
