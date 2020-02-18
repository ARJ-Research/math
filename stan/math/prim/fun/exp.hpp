#ifndef STAN_MATH_PRIM_FUN_EXP_HPP
#define STAN_MATH_PRIM_FUN_EXP_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/match_wrapper.hpp>
#include <cmath>

namespace stan {
namespace math {

inline double exp(int x) { return std::exp(x); }

/**
 * Structure to wrap <code>exp()</code> so that it can be
 * vectorized.
 */
struct exp_fun {
  /**
   * Return the exponential of the specified scalar argument.
   *
   * @tparam T type of argument
   * @param[in] x argument
   * @return Exponential of argument.
   */
  template <typename T>
  static inline T fun(const T& x) {
    using std::exp;
    return exp(x);
  }
};

/**
 * Return the elementwise exponentiation of the specified argument,
 * which may be a scalar or any Stan container of numeric scalars.
 * The return type is the same as the argument type.
 *
 * @tparam T type of container
 * @param[in] x container
 * @return Elementwise application of exponentiation to the argument.
 */
template <typename T, typename = require_not_container_st<is_container,
														  std::is_arithmetic,
														  T>>
inline auto exp(const T& x) {
  return apply_scalar_unary<exp_fun, T>::apply(x);
}

/**
 * Version of exp() that accepts Eigen Matrix or matrix expressions.
 * @tparam Derived derived type of x
 * @param x Matrix or matrix expression
 * @return Elementwise application of exponentiation to the argument.
 */
template <typename T, require_container_st<is_container, std::is_arithmetic, T>* = nullptr>
inline auto exp(const T& x) {
  return apply_vector_unary<T>::apply(x, [&](const auto& v) {
    return match_wrapper<decltype(v)>(v.derived().array().exp()).eval();
  });
}

}  // namespace math
}  // namespace stan

#endif
