#ifndef STAN_MATH_PRIM_FUN_MEAN_HPP
#define STAN_MATH_PRIM_FUN_MEAN_HPP

#include <stan/math/prim/err.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/Eigen.hpp>

namespace stan {
namespace math {

/**
 * Returns the sample mean (i.e., average) of the coefficients
 * in the specified vector, row vector, matrix, or container of these.
 *
 * @tparam T type of elements in the matrix
 *
 * @param x Specified vector, row vector, or matrix.
 * @return Sample mean of vector coefficients.
 */
template <typename T>
inline auto mean(const T& x) {
  return apply_vector_unary<T>::reduce(x, [&](const auto& m) {
    check_nonzero_size("mean", "m", m);
    return m.mean();
  });
}

}  // namespace math
}  // namespace stan

#endif
