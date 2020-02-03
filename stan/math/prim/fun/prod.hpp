#ifndef STAN_MATH_PRIM_FUN_PROD_HPP
#define STAN_MATH_PRIM_FUN_PROD_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/Eigen.hpp>

namespace stan {
namespace math {

/**
 * Returns the product of the coefficients of the specified
 * vector.
 *
 * @tparam T type of vector
 * @param x Specified vector.
 * @return Product of coefficients of vector.
 */
template <typename T>
inline auto prod(const T& x) {
  return apply_vector_unary<T>::reduce(x, [&](const auto& v) {
    if (v.size() == 0) {
      return scalar_type_t<T>(1.0);
    }
    return v.prod();
  });
}

}  // namespace math
}  // namespace stan

#endif
