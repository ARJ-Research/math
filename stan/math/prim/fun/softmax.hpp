#ifndef STAN_MATH_PRIM_FUN_SOFTMAX_HPP
#define STAN_MATH_PRIM_FUN_SOFTMAX_HPP

#include <stan/math/prim/err.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <cmath>

namespace stan {
namespace math {

/**
 * Return the softmax of the specified vector.
 *
 * <p>
 * \f$
 * \mbox{softmax}(y)
 * = \frac{\exp(y)}
 * {\sum_{k=1}^K \exp(y_k)},
 * \f$
 *
 * <p>The entries in the Jacobian of the softmax function are given by
 * \f$
 * \begin{array}{l}
 * \displaystyle
 * \frac{\partial}{\partial y_m} \mbox{softmax}(y)[k]
 * \\[8pt]
 * \displaystyle
 * \mbox{ } \ \ \ = \left\{
 * \begin{array}{ll}
 * \mbox{softmax}(y)[k] \times (1 - \mbox{softmax}(y)[m])
 * & \mbox{ if } m = k, \mbox{ and}
 * \\[6pt]
 * -\mbox{softmax}(y)[k] \times \mbox{softmax}(y)[m]
 * & \mbox{ if } m \neq k.
 * \end{array}
 * \right.
 * \end{array}
 * \f$
 *
 * @tparam T type of elements in the vector
 * @param[in] v Vector to transform.
 * @return Unit simplex result of the softmax transform of the vector.
 */
template <typename T, require_t<std::is_arithmetic<scalar_type_t<T>>>...>
inline auto softmax(const T& x) {
  return apply_vector_unary<T>::apply(x, [&](const auto& v) {
    check_nonzero_size("softmax", "v", v);
    Eigen::Matrix<scalar_type_t<T>, Eigen::Dynamic, 1> theta(v.size());
    theta = (v.array() - v.maxCoeff()).exp();
    return (theta.array() / theta.sum()).matrix().eval();
  });
}

}  // namespace math
}  // namespace stan

#endif
