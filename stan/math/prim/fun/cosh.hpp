#ifndef STAN_MATH_PRIM_FUN_COSH_HPP
#define STAN_MATH_PRIM_FUN_COSH_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <cmath>

namespace stan {
namespace math {

/**
 * Structure to wrap `cosh()` so it can be vectorized.
 *
 * @tparam T type of argument
 * @param x angle in radians
 * @return Hyperbolic cosine of x.
 */
template <typename T, require_arithmetic_t<T>...>
auto cosh(const T& x) {
  using std::cosh;
  return cosh(x);
}

/**
 * Version of `cosh()` that accepts std::vectors, Eigen Matrix/Array objects
 *  or expressions, and containers of these.
 *
 * @tparam Container Type of x
 * @param x Container
 * @return Hyberbolic cosine of x.
 */
template <typename Container,
          require_container_st<is_container, std::is_arithmetic, Container>...>
inline auto cosh(const Container& x) {
  return apply_vector_unary<Container>::apply(
      x, [](const auto& v) { return v.array().cosh(); });
}
}  // namespace math
}  // namespace stan

#endif
