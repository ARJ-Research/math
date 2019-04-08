#ifndef STAN_MATH_FWD_MAT_FUN_CROSSPROD_HPP
#define STAN_MATH_FWD_MAT_FUN_CROSSPROD_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/fwd/core.hpp>

namespace stan {
namespace math {

template <typename T, int R, int C>
inline Eigen::Matrix<fvar<T>, C, C> crossprod(
    const Eigen::Matrix<fvar<T>, R, C>& m) {
  if (m.rows() == 0)
    return Eigen::Matrix<fvar<T>, C, C>(0, 0);
  Eigen::Matrix<fvar<T>, C, C> ret(m.cols(),m.cols());
  Eigen::Matrix<T, R, C> mval = m.val_();

  ret.val_() = mval.transpose() * mval;
  ret.d_() = mval.transpose() * m.d_() + m.d_().transpose() * mval;
  return ret;
}

}  // namespace math
}  // namespace stan
#endif
