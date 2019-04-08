#ifndef STAN_MATH_FWD_MAT_FUN_INVERSE_HPP
#define STAN_MATH_FWD_MAT_FUN_INVERSE_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/mat/fun/to_fvar.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>

namespace stan {
namespace math {

template <typename T, int R, int C>
inline Eigen::Matrix<fvar<T>, R, C> inverse(
    const Eigen::Matrix<fvar<T>, R, C>& m) {
  check_square("inverse", "m", m);
  Eigen::Matrix<T, R, C> m_deriv(m.rows(), m.cols());
  Eigen::Matrix<T, R, C> m_inv(m.rows(), m.cols());

  m_inv = m.val_().inverse();

  m_deriv = (m_inv * m.d_() * m_inv).array() * -1.0;

  return to_fvar(m_inv, m_deriv);
}

}  // namespace math
}  // namespace stan
#endif
