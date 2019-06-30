#ifndef STAN_MATH_PRIM_SCAL_FUN_IS_NONPOSITIVE_INTEGER_HPP
#define STAN_MATH_PRIM_SCAL_FUN_IS_NONPOSITIVE_INTEGER_HPP

#include <stan/math/prim/scal/fun/is_integer.hpp>

namespace stan {
namespace math {

/**
 * Returns true if the input is a nonpositive integer and false otherwise.
 *
 * @param x Value to test.
 * @return <code>true</code> if the value is an integer
 */
template <typename T>
inline bool is_nonpositive_integer(T x) {
  return x <= 0.0 && is_integer(x);
}

}  // namespace math
}  // namespace stan

#endif
