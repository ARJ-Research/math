#ifndef STAN_MATH_PRIM_FUN_VALUE_OF_HPP
#define STAN_MATH_PRIM_FUN_VALUE_OF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <cstddef>
#include <vector>

namespace stan {
namespace math {

/**
 * Return the value of the specified scalar argument
 * converted to a double value.
 *
 * <p>See the <code>primitive_value</code> function to
 * extract values without casting to <code>double</code>.
 *
 * <p>This function is meant to cover the primitive types. For
 * types requiring pass-by-reference, this template function
 * should be specialized.
 *
 * @tparam T type of scalar.
 * @param x scalar to convert to double
 * @return value of scalar cast to double
 */
template <typename T, require_not_container_t<T>...,
          require_not_double_or_int_t<T>...>
inline double value_of(const T x) {
  return static_cast<double>(x);
}

/**
 * Return the specified argument.
 *
 * <p>See <code>value_of(T)</code> for a polymorphic
 * implementation using static casts.
 *
 * <p>This inline pass-through no-op should be compiled away.
 *
 * @param x value
 * @return input value
 */
template <typename T, require_double_or_int_t<T>...>
inline auto value_of(T&& x) {
  return std::forward<T>(x);
}

/**
 * Return the specified argument.
 *
 * <p>See <code>value_of(T)</code> for a polymorphic
 * implementation using static casts.
 *
 * <p>This inline pass-through no-op should be compiled away.
 *
 * @param x Specified std::vector.
 * @return Specified std::vector.
 */
template <typename T, require_container_vt<std::is_arithmetic, T>...>
inline decltype(auto) value_of(T&& x) {
  return std::forward<T>(x);
}
/**
 * Convert a matrix of type T to a matrix of doubles.
 *
 * T must implement value_of. See
 * test/math/fwd/fun/value_of.cpp for fvar and var usage.
 *
 * @tparam T type of elements in the matrix
 * @tparam R number of rows in the matrix, can be Eigen::Dynamic
 * @tparam C number of columns in the matrix, can be Eigen::Dynamic
 *
 * @param[in] M Matrix to be converted
 * @return Matrix of values
 **/
template <typename T, require_container_st<is_autodiff, T>...>
inline decltype(auto) value_of(const T& M) {
  return apply_vector_unary<T>::apply(M, [&](const auto& v) {
    return v.unaryExpr([](const auto& x){ return value_of(x); });
  });
}

}  // namespace math
}  // namespace stan

#endif
