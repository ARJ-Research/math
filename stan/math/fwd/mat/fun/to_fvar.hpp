#ifndef STAN_MATH_FWD_MAT_FUN_TO_FVAR_HPP
#define STAN_MATH_FWD_MAT_FUN_TO_FVAR_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/to_fvar.hpp>
#include <stan/math/prim/mat/err/check_matching_dims.hpp>

namespace stan {
namespace math {

template <int R, int C, typename T>
inline Eigen::Matrix<T, R, C> to_fvar(const Eigen::Matrix<T, R, C>& m) {
  return m;
}

template <int R, int C>
inline Eigen::Matrix<fvar<double>, R, C> to_fvar(
    const Eigen::Matrix<double, R, C>& m) {
  Eigen::Matrix<fvar<double>, R, C> m_fd(m.rows(), m.cols());
  m_fd.val_() = m;
  return m_fd;
}

template <typename T, int R, int C>
inline Eigen::Matrix<fvar<T>, R, C> to_fvar(
    const Eigen::Matrix<T, R, C>& val, const Eigen::Matrix<T, R, C>& deriv) {
  check_matching_dims("to_fvar", "value", val, "deriv", deriv);
  Eigen::Matrix<fvar<T>, R, C> ret(val.rows(), val.cols());
  ret.val_() = val;
  ret.d_() = deriv;
  return ret;
}

}  // namespace math
}  // namespace stan
#endif
