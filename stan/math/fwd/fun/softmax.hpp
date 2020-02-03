#ifndef STAN_MATH_FWD_FUN_SOFTMAX_HPP
#define STAN_MATH_FWD_FUN_SOFTMAX_HPP

#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/softmax.hpp>

namespace stan {
namespace math {

/**
 * Return the softmax of the specified vector.
 *
 * @tparam T type of elements in the vector
 * @param[in] v Vector to transform.
 * @return Unit simplex result of the softmax transform of the vector.
 */
template <typename T, require_t<is_fvar<scalar_type_t<T>>>...>
inline auto softmax(const T& x) {
  return apply_vector_unary<T>::apply(x, [&](const auto& alpha) {
    using T_fvar = scalar_type_t<T>;
    using T_fvar_inner = typename T_fvar::Scalar;
    Eigen::Matrix<T_fvar, -1, 1> softmax_alpha(alpha.size());
    softmax_alpha.val() = softmax(alpha.val().eval());
    softmax_alpha.d().fill(0);

    for (int m = 0; m < alpha.size(); ++m) {
      T_fvar_inner negative_alpha_m_d_times_softmax_alpha_t_m
          = -alpha(m).d_ * softmax_alpha(m).val();
      for (int k = 0; k < alpha.size(); ++k) {
        if (m == k) {
          softmax_alpha(k).d_
              += softmax_alpha(k).val()
                 * (alpha(m).d_ + negative_alpha_m_d_times_softmax_alpha_t_m);
        } else {
          softmax_alpha(k).d_
              += negative_alpha_m_d_times_softmax_alpha_t_m
                 * softmax_alpha(k).val();
        }
      }
    }

    return softmax_alpha;
  });
}

}  // namespace math
}  // namespace stan
#endif
