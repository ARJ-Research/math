#ifndef STAN_MATH_FWD_MAT_FUN_DIVIDE_HPP
#define STAN_MATH_FWD_MAT_FUN_DIVIDE_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/mat/fun/to_fvar.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <vector>

namespace stan {
namespace math {

template <typename T, int R, int C>
inline Eigen::Matrix<fvar<T>, R, C> divide(
    const Eigen::Matrix<fvar<T>, R, C>& v, const fvar<T>& c) {
  Eigen::Matrix<fvar<T>, R, C> res(v.rows(), v.cols());
  res.val_() = v.val_().array() / c.val_;
  res.d_() = (v.d_().array() * c.val_ - v.val_().array() * c.d_)
                / (c.val_ * c.val_);
  return res;
}

template <typename T, int R, int C>
inline Eigen::Matrix<fvar<T>, R, C> divide(
    const Eigen::Matrix<fvar<T>, R, C>& v, double c) {
  Eigen::Matrix<fvar<T>, R, C> res(v.rows(), v.cols());
  res.val_() = v.val_().array() / c;
  res.d_() = v.d_().array() / c;
  return res;
}

template <typename T, int R, int C>
inline Eigen::Matrix<fvar<T>, R, C> divide(const Eigen::Matrix<double, R, C>& v,
                                           const fvar<T>& c) {
  Eigen::Matrix<fvar<T>, R, C> res(v.rows(), v.cols());
  res.val_() = v.array() / c.val_;
  res.d_() = (-1.0 * v.array() * c.d_) / (c.val_ * c.val_);
  return res;
}

template <typename T, int R, int C>
inline Eigen::Matrix<fvar<T>, R, C> operator/(
    const Eigen::Matrix<fvar<T>, R, C>& v, const fvar<T>& c) {
  return divide(v, c);
}

template <typename T, int R, int C>
inline Eigen::Matrix<fvar<T>, R, C> operator/(
    const Eigen::Matrix<fvar<T>, R, C>& v, double c) {
  return divide(v, c);
}

template <typename T, int R, int C>
inline Eigen::Matrix<fvar<T>, R, C> operator/(
    const Eigen::Matrix<double, R, C>& v, const fvar<T>& c) {
  return divide(v, c);
}
}  // namespace math
}  // namespace stan
#endif
