#ifndef STAN_MATH_FWD_MAT_FUN_SUM_HPP
#define STAN_MATH_FWD_MAT_FUN_SUM_HPP

#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
namespace math {

/**
 * Return the sum of the entries of the specified matrix.
 *
 * @tparam T Type of matrix entries.
 * @tparam R Row type of matrix.
 * @tparam C Column type of matrix.
 * @param m Matrix.
 * @return Sum of matrix entries.
 */
template <typename T, int R, int C>
inline fvar<T> sum(const Eigen::Matrix<fvar<T>, R, C>& m) {
  if (m.size() == 0)
    return 0.0;

  return fvar<T>(m.val_().sum(), m.d_().sum());
}

}  // namespace math
}  // namespace stan
#endif
