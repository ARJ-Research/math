#ifndef STAN_MATH_PRIM_FUN_VARIANCE_HPP
#define STAN_MATH_PRIM_FUN_VARIANCE_HPP

#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/mean.hpp>
#include <vector>

namespace stan {
namespace math {

/**
 * Returns the sample variance (divide by length - 1) of the
 * coefficients in the specified matrix
 *
 * @tparam T type of vector
 *
 * @param x matrix
 * @return sample variance of coefficients
 * @throw <code>std::invalid_argument</code> if the matrix has size zero
 */
template <typename T, require_not_t<is_var<scalar_type_t<T>>>...>
inline auto variance(const T& x) {
  return apply_vector_unary<T>::reduce(x, [&](const auto& m) {
    check_nonzero_size("variance", "m", m);
    if (m.size() == 1) {
      return scalar_type_t<T>(0.0);
    }
    return (m.array() - m.mean()).matrix().squaredNorm() / (m.size() - 1);
  });
}

}  // namespace math
}  // namespace stan

#endif
