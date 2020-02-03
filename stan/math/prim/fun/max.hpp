#ifndef STAN_MATH_PRIM_FUN_MAX_HPP
#define STAN_MATH_PRIM_FUN_MAX_HPP

#include <stan/math/prim/err.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/constants.hpp>

namespace stan {
namespace math {

/**
 * Returns the maximum coefficient in the specified
 * column vector.
 *
 * @param x specified vector
 * @return maximum coefficient value in the vector
 * @throws <code>std::invalid_argument</code> if the vector is size zero
 */
template <typename T, require_t<std::is_integral<scalar_type_t<T>>>...>
inline auto max(const T& x) {
  return apply_vector_unary<T>::reduce(x, [&](const auto& m) {
    check_nonzero_size("max", "int vector", m);
    return m.maxCoeff();
  });
}

/**
 * Returns the maximum coefficient in the specified
 * column vector.
 *
 * @tparam type of elements in the vector
 * @param x specified vector
 * @return maximum coefficient value in the vector, or -infinity if the vector
 * is size zero
 */
template <typename T, require_not_t<std::is_integral<scalar_type_t<T>>>...>
inline auto max(const T& x) {
  return apply_vector_unary<T>::reduce(x, [&](const auto& m) {
    if (m.size() == 0) {
      return scalar_type_t<T>(NEGATIVE_INFTY);
    }
    return m.maxCoeff();
  });
}

}  // namespace math
}  // namespace stan

#endif
