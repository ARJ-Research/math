#ifndef STAN_MATH_FWD_MAT_FUN_COLUMNS_DOT_SELF_HPP
#define STAN_MATH_FWD_MAT_FUN_COLUMNS_DOT_SELF_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/mat/fun/dot_self.hpp>
#include <vector>

namespace stan {
namespace math {

template <typename T, int R, int C>
inline Eigen::Matrix<fvar<T>, 1, C> columns_dot_self(
    const Eigen::Matrix<fvar<T>, R, C>& x) {
  Eigen::Matrix<fvar<T>, 1, C> ret(1, x.cols());
  Eigen::Matrix<T, R, C> xval = x.val_();

  ret.val_() = (xval.transpose() * xval).diagonal();
  ret.d_() = (xval.transpose() * x.d_()).diagonal().array() * 2.0;
  return ret;
}
}  // namespace math
}  // namespace stan
#endif
