#ifndef STAN_MATH_PRIM_FUN_SUM_HPP
#define STAN_MATH_PRIM_FUN_SUM_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <cstddef>
#include <numeric>
#include <vector>

namespace stan {
namespace math {

/**
 * Returns specified input value.
 *
 * @tparam T Type of element.
 * @param v Specified value.
 * @return Same value (the sum of one value).
 */
inline double sum(double v) { return v; }

/**
 * Returns specified input value.
 *
 * @tparam T Type of element.
 * @param v Specified value.
 * @return Same value (the sum of one value).
 */
inline int sum(int v) { return v; }

/**
 * Returns the sum of the coefficients of the specified
 * Eigen Matrix, Array or expression.
 *
 * @tparam Derived type of argument
 * @param v argument
 * @return Sum of coefficients of argument.
 */
template <typename T, require_t<std::is_arithmetic<scalar_type_t<T>>>...>
inline auto sum(const T& x) {
  return apply_vector_unary<T>::reduce(x, [&](const auto& m) {
    return m.sum();
  });
}

}  // namespace math
}  // namespace stan

#endif
