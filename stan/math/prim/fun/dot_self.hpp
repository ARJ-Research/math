#ifndef STAN_MATH_PRIM_FUN_DOT_SELF_HPP
#define STAN_MATH_PRIM_FUN_DOT_SELF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <cstddef>
#include <vector>

namespace stan {
namespace math {

/**
 * Returns the dot product of the specified vector, or container of vectors,
 * with itself.
 *
 * @tparam T Type of vector or container of vectors
 * @param x Vector.
 * @throw std::domain_error If x is not vector dimensioned.
 */
template <typename T, require_t<std::is_arithmetic<scalar_type_t<T>>>...>
inline auto dot_self(const T& x) {
  return apply_vector_unary<T>::reduce(x, [&](const auto& v) {
    check_vector("dot_self", "v", v);
    return v.squaredNorm();
  });
}

}  // namespace math
}  // namespace stan

#endif
