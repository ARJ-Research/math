#ifndef STAN_MATH_PRIM_FUN_MIN_HPP
#define STAN_MATH_PRIM_FUN_MIN_HPP

#include <stan/math/prim/err.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/constants.hpp>

namespace stan {
namespace math {

/**
 * Returns the minimum coefficient in the specified
 * column vector.
 *
 * @param x specified vector
 * @return minimum coefficient value in the vector
 * @throws <code>std::invalid_argument</code> if the vector is size zero
 */
template <typename T, require_t<std::is_integral<scalar_type_t<T>>>...>
inline auto min(const T& x) {
  return apply_vector_unary<T>::reduce(x, [&](const auto& m) {
    check_nonzero_size("min", "int vector", m);
    return m.minCoeff();
  });
}

/**
 * Returns the minimum coefficient in the specified
 * column vector.
 *
 * @tparam T type of elements in the vector
 * @param x specified vector
 * @return minimum coefficient value in the vector, or infinity if the vector is
 * size zero
 */
template <typename T, require_not_t<std::is_integral<scalar_type_t<T>>>...>
inline auto min(const T& x) {
  return apply_vector_unary<T>::reduce(x, [&](const auto& m) {
    if (m.size() == 0) {
      return scalar_type_t<T>(NEGATIVE_INFTY);
    }
    return m.minCoeff();
  });
}

}  // namespace math
}  // namespace stan

#endif
