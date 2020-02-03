#ifndef STAN_MATH_FWD_FUN_SUM_HPP
#define STAN_MATH_FWD_FUN_SUM_HPP

#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/fun/Eigen.hpp>

namespace stan {
namespace math {

/**
 * Return the sum of the entries of the specified matrix.
 *
 * @tparam T inner type of the fvar matrix
 * @tparam R number of rows, can be Eigen::Dynamic
 * @tparam C number of columns, can be Eigen::Dynamic
 *
 * @param m Matrix.
 * @return Sum of matrix entries.
 */
template <typename T, require_t<is_fvar<scalar_type_t<T>>>...>
inline auto sum(const T& x) {
  return apply_vector_unary<T>::reduce(x, [&](const auto& m) {
    if (m.size() == 0) {
      return scalar_type_t<T>(0.0);
    }
    using T_fvar = scalar_type_t<T>;
    return T_fvar(m.matrix().val().sum(),m.matrix().d().sum());
  });
}

}  // namespace math
}  // namespace stan
#endif
