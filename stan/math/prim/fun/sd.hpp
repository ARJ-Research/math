#ifndef STAN_MATH_PRIM_FUN_SD_HPP
#define STAN_MATH_PRIM_FUN_SD_HPP

#include <stan/math/prim/err.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/variance.hpp>
#include <stan/math/prim/fun/sqrt.hpp>
#include <cmath>

namespace stan {
namespace math {

/**
 * Returns the unbiased sample standard deviation of the
 * coefficients in the specified vector, row vector, or matrix.
 *
 * @tparam T type of vector, row vector, matrix, or container of these.
 *
 * @param x Specified vector, row vector, matrix, or container of these.
 * @return Sample standard deviation.
 */
template <typename T, require_not_t<is_var<scalar_type_t<T>>>...>
inline auto sd(const T& x) {
  return apply_vector_unary<T>::reduce(x, [&](const auto& m) {
    using std::sqrt;
    check_nonzero_size("sd", "m", m);
    if (m.size() == 1) {
      return scalar_type_t<T>(0.0);
    }
    return sqrt(variance(m));
  });
}

}  // namespace math
}  // namespace stan

#endif
